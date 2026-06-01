//! Linux / FreeBSD static-module enumeration via `dl_iterate_phdr`.
//!
//! For each loaded ELF object we locate the in-memory `.eh_frame_hdr`
//! (`PT_GNU_EH_FRAME`), decode it to find `.eh_frame`, copy both sections into owned
//! memory, and build a framehop `EhFrameHdrAndEhFrame` module. ELF SVMA == AVMA - bias,
//! so `base_svma = 0` and `base_avma = dlpi_addr`.

use std::collections::HashSet;
use std::ffi::CStr;
use std::ops::Range;

use framehop::{ExplicitModuleSectionInfo, Module};

use super::EnumResult;
use crate::state::Bytes;

const PT_LOAD: u32 = 1;
const PT_GNU_EH_FRAME: u32 = 0x6474_e550;
const PF_X: u32 = 1;

/// A module's copied unwind sections + geometry, captured under the dl lock.
struct RawSpec {
    key: u64,
    name: String,
    base_avma: u64,
    avma: Range<u64>,
    eh_frame_hdr: Bytes,
    eh_frame_hdr_svma: Range<u64>,
    eh_frame: Bytes,
    eh_frame_svma: Range<u64>,
    text_svma: Option<Range<u64>>,
}

struct Collector<'a> {
    known: &'a HashSet<u64>,
    current_keys: HashSet<u64>,
    specs: Vec<RawSpec>,
}

pub fn enumerate(known: &HashSet<u64>) -> EnumResult {
    let mut collector = Collector {
        known,
        current_keys: HashSet::new(),
        specs: Vec::new(),
    };
    // SAFETY: standard dl_iterate_phdr usage; callback does not unwind or call dlopen.
    unsafe {
        libc::dl_iterate_phdr(Some(callback), &mut collector as *mut _ as *mut libc::c_void);
    }

    let new_modules = collector
        .specs
        .into_iter()
        .map(|s| (s.key, build_module(s)))
        .collect();

    EnumResult {
        current_keys: collector.current_keys,
        new_modules,
    }
}

fn build_module(s: RawSpec) -> Module<Bytes> {
    let info: ExplicitModuleSectionInfo<Bytes> = ExplicitModuleSectionInfo {
        base_svma: 0,
        text_svma: s.text_svma,
        eh_frame_hdr: Some(s.eh_frame_hdr),
        eh_frame_hdr_svma: Some(s.eh_frame_hdr_svma),
        eh_frame: Some(s.eh_frame),
        eh_frame_svma: Some(s.eh_frame_svma),
        ..Default::default()
    };
    Module::new(s.name, s.avma, s.base_avma, info)
}

extern "C" fn callback(
    info: *mut libc::dl_phdr_info,
    _size: libc::size_t,
    data: *mut libc::c_void,
) -> libc::c_int {
    // SAFETY: libc guarantees `info` and `data` are valid for the duration of the call.
    let info = unsafe { &*info };
    let collector = unsafe { &mut *(data as *mut Collector) };

    let bias = info.dlpi_addr as u64;
    if info.dlpi_phdr.is_null() || info.dlpi_phnum == 0 {
        return 0;
    }
    let phdrs = unsafe {
        core::slice::from_raw_parts(info.dlpi_phdr, info.dlpi_phnum as usize)
    };

    // Geometry: union of PT_LOAD; remember the executable segment and the eh_frame_hdr.
    let mut avma_lo = u64::MAX;
    let mut avma_hi = 0u64;
    let mut text_svma: Option<Range<u64>> = None;
    let mut hdr_phdr: Option<&libc::Elf64_Phdr> = None;

    for ph in phdrs {
        match ph.p_type {
            PT_LOAD => {
                let lo = bias + ph.p_vaddr as u64;
                let hi = lo + ph.p_memsz as u64;
                if lo < avma_lo {
                    avma_lo = lo;
                }
                if hi > avma_hi {
                    avma_hi = hi;
                }
                if (ph.p_flags & PF_X) != 0 && text_svma.is_none() {
                    text_svma = Some(Range {
                        start: ph.p_vaddr as u64,
                        end: ph.p_vaddr as u64 + ph.p_memsz as u64,
                    });
                }
            }
            PT_GNU_EH_FRAME => hdr_phdr = Some(ph),
            _ => {}
        }
    }

    if avma_lo == u64::MAX || avma_hi <= avma_lo {
        return 0; // no loadable segments
    }
    let key = avma_lo;
    collector.current_keys.insert(key);

    // Only build (and copy bytes for) modules we don't already know about.
    if collector.known.contains(&key) {
        return 0;
    }

    let hdr_phdr = match hdr_phdr {
        Some(p) => p,
        None => return 0, // no unwind info; framehop will fp-fallback for this module
    };

    let hdr_avma = bias + hdr_phdr.p_vaddr as u64;
    let hdr_len = hdr_phdr.p_memsz as usize;
    if hdr_len < 4 {
        return 0;
    }

    // Decode the eh_frame start address from the header.
    let eh_frame_avma = match unsafe { decode_eh_frame_start(hdr_avma as *const u8, hdr_len) } {
        Some(a) => a,
        None => return 0,
    };

    // Bound the eh_frame slice by the end of its containing PT_LOAD segment (an overlong
    // slice is safe: the eh_frame_hdr index makes framehop touch only the target FDE).
    let eh_frame_end = match containing_segment_end(phdrs, bias, eh_frame_avma) {
        Some(e) if e > eh_frame_avma => e,
        _ => return 0,
    };
    let eh_frame_len = (eh_frame_end - eh_frame_avma) as usize;

    // Copy both sections (under the dl lock, so no concurrent dlclose).
    let eh_frame_hdr: Bytes = unsafe {
        core::slice::from_raw_parts(hdr_avma as *const u8, hdr_len)
    }
    .to_vec()
    .into_boxed_slice();
    let eh_frame: Bytes = unsafe {
        core::slice::from_raw_parts(eh_frame_avma as *const u8, eh_frame_len)
    }
    .to_vec()
    .into_boxed_slice();

    let name = if info.dlpi_name.is_null() || unsafe { *info.dlpi_name } == 0 {
        String::from("<main>")
    } else {
        unsafe { CStr::from_ptr(info.dlpi_name) }
            .to_string_lossy()
            .into_owned()
    };

    collector.specs.push(RawSpec {
        key,
        name,
        base_avma: bias,
        avma: Range {
            start: avma_lo,
            end: avma_hi,
        },
        eh_frame_hdr,
        // SVMA = AVMA - bias (base_svma == 0 for ELF).
        eh_frame_hdr_svma: Range {
            start: hdr_avma - bias,
            end: hdr_avma - bias + hdr_len as u64,
        },
        eh_frame,
        eh_frame_svma: Range {
            start: eh_frame_avma - bias,
            end: eh_frame_avma - bias + eh_frame_len as u64,
        },
        text_svma,
    });

    0
}

/// Find the end AVMA of the PT_LOAD segment containing `addr`.
fn containing_segment_end(phdrs: &[libc::Elf64_Phdr], bias: u64, addr: u64) -> Option<u64> {
    for ph in phdrs {
        if ph.p_type != PT_LOAD {
            continue;
        }
        let lo = bias + ph.p_vaddr as u64;
        let hi = lo + ph.p_memsz as u64;
        if addr >= lo && addr < hi {
            return Some(hi);
        }
    }
    None
}

/// Decode the `eh_frame_ptr` field of an in-memory `.eh_frame_hdr` to an absolute AVMA.
///
/// Layout: `version(u8), eh_frame_ptr_enc(u8), fde_count_enc(u8), table_enc(u8),
/// eh_frame_ptr(encoded)`.
///
/// # Safety
/// `hdr` must point at `hdr_len` readable bytes.
unsafe fn decode_eh_frame_start(hdr: *const u8, hdr_len: usize) -> Option<u64> {
    let b = core::slice::from_raw_parts(hdr, hdr_len);
    if b.len() < 4 || b[0] != 1 {
        return None; // unknown version
    }
    let enc = b[1];
    let app = enc & 0x70;
    let fmt = enc & 0x0f;
    let field_off = 4usize;
    let field_avma = hdr as u64 + field_off as u64;

    let (value, _sz) = decode_value(&b[field_off..], fmt)?;

    let base = match app {
        0x00 => 0u64,         // DW_EH_PE_absptr
        0x10 => field_avma,   // DW_EH_PE_pcrel
        _ => return None,     // datarel/textrel/etc. not expected for eh_frame_ptr
    };
    Some(base.wrapping_add(value))
}

/// Decode a fixed-size DWARF encoded value; returns `(sign-extended value, byte size)`.
fn decode_value(b: &[u8], fmt: u8) -> Option<(u64, usize)> {
    match fmt {
        0x02 => {
            // udata2
            let v = u16::from_le_bytes(b.get(0..2)?.try_into().ok()?);
            Some((v as u64, 2))
        }
        0x0a => {
            // sdata2
            let v = i16::from_le_bytes(b.get(0..2)?.try_into().ok()?);
            Some((v as i64 as u64, 2))
        }
        0x03 => {
            // udata4
            let v = u32::from_le_bytes(b.get(0..4)?.try_into().ok()?);
            Some((v as u64, 4))
        }
        0x0b => {
            // sdata4 (most common: pcrel|sdata4 == 0x1b)
            let v = i32::from_le_bytes(b.get(0..4)?.try_into().ok()?);
            Some((v as i64 as u64, 4))
        }
        0x04 => {
            // udata8
            let v = u64::from_le_bytes(b.get(0..8)?.try_into().ok()?);
            Some((v, 8))
        }
        0x0c => {
            // sdata8
            let v = i64::from_le_bytes(b.get(0..8)?.try_into().ok()?);
            Some((v as u64, 8))
        }
        0x00 => {
            // absptr (native word size; assume 8 on our 64-bit targets)
            let v = u64::from_le_bytes(b.get(0..8)?.try_into().ok()?);
            Some((v, 8))
        }
        _ => None, // leb128 etc. not expected for eh_frame_ptr
    }
}
