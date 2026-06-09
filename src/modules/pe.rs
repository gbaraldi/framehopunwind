//! Windows PE static-module enumeration.
//!
//! `ExplicitModuleSectionInfo` cannot produce a framehop PE module (it has no
//! `.pdata`/`.xdata`/`.rdata` arm), so we implement [`framehop::ModuleSectionInfo`]
//! ourselves and return those sections. Modules are enumerated with
//! `EnumProcessModules` + `GetModuleInformation` (done off any thread-suspend window), and
//! each PE image is parsed in memory to locate the unwind sections.
//!
//! Note: Julia's Windows sampler suspends the target thread and unwinds on a *separate*
//! thread, so the (allocating) PE path here is fine — but enumeration must not run while a
//! target is suspended (loader-lock hazard).

use std::collections::HashMap;
use std::ops::Range;

use framehop::{Module, ModuleSectionInfo};
use windows_sys::Win32::Foundation::HMODULE;
use windows_sys::Win32::System::ProcessStatus::{
    EnumProcessModules, GetModuleInformation, MODULEINFO,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

use super::EnumResult;
use crate::state::Bytes;

// IMAGE_SECTION_HEADER is a stable 40-byte layout:
//   name[8], virtual_size(u32@8), virtual_address(u32@12), size_of_raw_data(u32@16), ...
// We read its fields by offset (the in-memory image is only byte-aligned).
const SECTION_HEADER_SIZE: usize = 40;

pub fn enumerate(known: &HashMap<u64, u64>) -> EnumResult {
    let mut current = HashMap::new();
    let mut new_modules = Vec::new();

    let process = unsafe { GetCurrentProcess() };

    // Query needed size, then enumerate module handles.
    let mut needed: u32 = 0;
    unsafe {
        EnumProcessModules(process, core::ptr::null_mut(), 0, &mut needed);
    }
    if needed == 0 {
        return EnumResult {
            current,
            new_modules,
        };
    }
    let count = needed as usize / core::mem::size_of::<HMODULE>();
    let mut handles: Vec<HMODULE> = vec![0 as HMODULE; count];
    let mut needed2: u32 = 0;
    let ok = unsafe { EnumProcessModules(process, handles.as_mut_ptr(), needed, &mut needed2) };
    if ok == 0 {
        return EnumResult {
            current,
            new_modules,
        };
    }
    let actual = (needed2 as usize / core::mem::size_of::<HMODULE>()).min(count);

    for &hmod in &handles[..actual] {
        let mut info: MODULEINFO = unsafe { core::mem::zeroed() };
        let ok = unsafe {
            GetModuleInformation(
                process,
                hmod,
                &mut info,
                core::mem::size_of::<MODULEINFO>() as u32,
            )
        };
        if ok == 0 || info.lpBaseOfDll.is_null() {
            continue;
        }
        let base = info.lpBaseOfDll as u64;
        let size = info.SizeOfImage as u64;
        let key = base;
        let fp = super::fingerprint_of(&[size], b"");
        current.insert(key, fp);
        if known.get(&key) == Some(&fp) {
            continue;
        }
        if let Some(module) = unsafe { build_pe_module(base, size) } {
            new_modules.push((key, module));
        }
    }

    EnumResult {
        current,
        new_modules,
    }
}

/// Parse the in-memory PE at `base` and build a framehop PE module if it has `.pdata`.
unsafe fn build_pe_module(base: u64, size: u64) -> Option<Module<Bytes>> {
    let base_ptr = base as *const u8;
    // IMAGE_DOS_HEADER: "MZ", e_lfanew at 0x3c.
    if read_u16(base_ptr, 0) != 0x5a4d {
        return None;
    }
    let e_lfanew = read_u32(base_ptr, 0x3c) as usize;
    // IMAGE_NT_HEADERS64: Signature(4) "PE\0\0", then FILE_HEADER(20), then OPTIONAL_HEADER.
    if read_u32(base_ptr, e_lfanew) != 0x0000_4550 {
        return None;
    }
    let file_header = e_lfanew + 4;
    let num_sections = read_u16(base_ptr, file_header + 2) as usize;
    let size_of_optional = read_u16(base_ptr, file_header + 16) as usize;
    let optional_header = file_header + 20;
    let section_table = optional_header + size_of_optional;

    // Read section-header fields through copy helpers (the in-memory image is only
    // guaranteed byte-aligned, so we never form a reference to the packed struct).
    let sec = |want: &[u8]| -> Option<(u32, Bytes)> {
        for i in 0..num_sections {
            let sh_off = section_table + i * SECTION_HEADER_SIZE;
            let mut name = [0u8; 8];
            core::ptr::copy_nonoverlapping(base_ptr.add(sh_off), name.as_mut_ptr(), 8);
            if section_name_eq(&name, want) {
                let virtual_size = read_u32(base_ptr, sh_off + 8);
                let rva = read_u32(base_ptr, sh_off + 12);
                // In the loaded image only the section's *virtual* extent is guaranteed
                // mapped; SizeOfRawData is a file-alignment quantity that may legally
                // exceed it, so copying max(virtual, raw) could read past the mapping.
                // Clamp to the image extent for the (also legal) rva+size > SizeOfImage.
                let len = (virtual_size as u64).min(size.saturating_sub(rva as u64)) as usize;
                if len == 0 {
                    return None;
                }
                let bytes = core::slice::from_raw_parts(base_ptr.add(rva as usize), len)
                    .to_vec()
                    .into_boxed_slice();
                return Some((rva, bytes));
            }
        }
        None
    };

    let pdata = sec(b".pdata")?; // no .pdata => not framehop-unwindable here
    let xdata = sec(b".xdata");
    let rdata = sec(b".rdata");
    let text = sec(b".text");

    let info = PeSectionInfo {
        base_svma: base,
        pdata: Some(pdata),
        xdata,
        rdata,
        text,
    };

    Some(Module::new(
        format!("<pe:{base:#x}>"),
        Range {
            start: base,
            end: base + size,
        },
        base,
        info,
    ))
}

/// A framehop `ModuleSectionInfo` that returns PE unwind sections. `*_svma` ranges are
/// absolute (`base + rva`); framehop turns them back into RVAs via `base_svma`.
struct PeSectionInfo {
    base_svma: u64,
    pdata: Option<(u32, Bytes)>,
    xdata: Option<(u32, Bytes)>,
    rdata: Option<(u32, Bytes)>,
    text: Option<(u32, Bytes)>,
}

impl PeSectionInfo {
    fn slot(&mut self, name: &[u8]) -> Option<&mut Option<(u32, Bytes)>> {
        match name {
            b".pdata" => Some(&mut self.pdata),
            b".xdata" => Some(&mut self.xdata),
            b".rdata" => Some(&mut self.rdata),
            b".text" => Some(&mut self.text),
            _ => None,
        }
    }
}

impl ModuleSectionInfo<Bytes> for PeSectionInfo {
    fn base_svma(&self) -> u64 {
        self.base_svma
    }

    fn section_svma_range(&mut self, name: &[u8]) -> Option<Range<u64>> {
        let base = self.base_svma;
        let s = self.slot(name)?.as_ref()?;
        let rva = s.0 as u64;
        let len = s.1.len() as u64;
        Some(Range {
            start: base + rva,
            end: base + rva + len,
        })
    }

    fn section_data(&mut self, name: &[u8]) -> Option<Bytes> {
        let slot = self.slot(name)?;
        slot.take().map(|(_rva, bytes)| bytes)
    }
}

unsafe fn read_u16(base: *const u8, off: usize) -> u16 {
    let mut b = [0u8; 2];
    core::ptr::copy_nonoverlapping(base.add(off), b.as_mut_ptr(), 2);
    u16::from_le_bytes(b)
}
unsafe fn read_u32(base: *const u8, off: usize) -> u32 {
    let mut b = [0u8; 4];
    core::ptr::copy_nonoverlapping(base.add(off), b.as_mut_ptr(), 4);
    u32::from_le_bytes(b)
}

fn section_name_eq(field: &[u8; 8], name: &[u8]) -> bool {
    if name.len() > 8 {
        return false;
    }
    for (i, &b) in name.iter().enumerate() {
        if field[i] != b {
            return false;
        }
    }
    name.len() == 8 || field[name.len()] == 0
}
