//! Windows PE static-module enumeration.
//!
//! `ExplicitModuleSectionInfo` cannot produce a framehop PE module (it has no
//! `.pdata`/`.xdata`/`.rdata` arm), so we implement [`framehop::ModuleSectionInfo`]
//! ourselves and return those sections. Init enumerates modules once with
//! `EnumProcessModules` + `GetModuleInformation`. After that, the DLL notifications
//! (`LdrRegisterDllNotification`) keep the set current. The loaded notification runs after
//! the image is mapped and snapped, before its `DllMain`. The unloaded notification runs
//! before the image is unmapped. Thus the set is exact, and sections are read in place (see
//! `Bytes`). The initial enumeration pins each module (`GetModuleHandleExW`, released
//! with `FreeLibrary`) while it parses the image, so a concurrent `FreeLibrary` cannot
//! unmap it. An image that it parsed can be unloaded after the pin is released, before the
//! result is published. The unloaded notification then makes the enumeration run again.
//!
//! If the notifications cannot be registered, nothing tells us of an unload, and every
//! enumeration copies the unwind sections, as on ELF. `.text` stays in place (see `Bytes`).
//!
//! The notification callback runs with the loader busy: every `LoadLibrary`, `FreeLibrary`
//! and thread creation in the process waits until it returns. `GetModuleHandleExW`,
//! `EnumProcessModules`, the heap and SRW locks do not wait. The callback allocates and
//! takes the registry and writer locks. This is safe because code that holds those locks
//! never waits on the loader (see the lock order in `modules/mod.rs`). `FreeLibrary` waits
//! on the loader, so the enumeration releases its pin before it takes the registry lock.
//!
//! Julia's Windows sampler suspends the target thread and unwinds it from a separate thread.
//! The PE unwind path does not allocate (see `vendor/README.md`), so the target can hold the
//! heap lock. Enumeration must not run while a target is suspended (loader-lock hazard).

use core::ffi::c_void;
use core::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Mutex;

use framehop::{Module, ModuleSectionInfo};
use windows_sys::Win32::Foundation::{FreeLibrary, HMODULE};
use windows_sys::Win32::System::LibraryLoader::{
    GetModuleHandleA, GetModuleHandleExW, GetProcAddress, GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
};
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

/// Enumerate loaded PE modules. Returns `None` if enumeration itself fails, so the
/// caller keeps its previous module set instead of treating it as "all unloaded".
pub fn enumerate(known: &HashMap<u64, u64>) -> Option<EnumResult> {
    let mut current = HashMap::new();
    let mut new_modules = Vec::new();
    // Without notifications, a module can outlive its image in the set.
    let copy = !notifications_active();

    let process = unsafe { GetCurrentProcess() };

    // Grow the buffer until the list fits: a truncated list would make refresh()
    // "unload" every module past the truncation point.
    let mut handles: Vec<HMODULE> = Vec::new();
    for attempt in 0.. {
        if attempt >= 8 {
            return None; // list keeps growing under us; give up, keep the old snapshot
        }
        let cb = (handles.len() * core::mem::size_of::<HMODULE>()) as u32;
        let mut needed: u32 = 0;
        let ok = unsafe { EnumProcessModules(process, handles.as_mut_ptr(), cb, &mut needed) };
        if ok == 0 {
            return None;
        }
        let count = needed as usize / core::mem::size_of::<HMODULE>();
        if count <= handles.len() {
            handles.truncate(count);
            break;
        }
        // Leave headroom so a concurrently-loading DLL doesn't force another round.
        handles.resize(count + 8, core::ptr::null_mut());
    }

    for &hmod in &handles {
        // Pin the module (loader refcount) before touching its image: the enumerated
        // handles are unpinned, so a concurrent FreeLibrary could unmap the image while we
        // parse it. Vanished-before-pin modules are skipped (== unloaded).
        let mut pinned: HMODULE = core::ptr::null_mut();
        let ok = unsafe {
            GetModuleHandleExW(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                hmod as *const u16, // FROM_ADDRESS: "name" is any address inside the module
                &mut pinned,
            )
        };
        if ok == 0 || pinned.is_null() {
            continue;
        }

        let mut info: MODULEINFO = unsafe { core::mem::zeroed() };
        let ok = unsafe {
            GetModuleInformation(
                process,
                pinned,
                &mut info,
                core::mem::size_of::<MODULEINFO>() as u32,
            )
        };
        if ok != 0 && !info.lpBaseOfDll.is_null() {
            let base = info.lpBaseOfDll as u64;
            let size = info.SizeOfImage as u64;
            let key = base;
            let fp = unsafe { fingerprint(base, size) };
            current.insert(key, fp);
            if known.get(&key) != Some(&fp) {
                if let Some(module) = unsafe { build_pe_module(base, size, copy) } {
                    new_modules.push((key, module));
                }
            }
        }
        unsafe { FreeLibrary(pinned) };
    }

    Some(EnumResult {
        current,
        new_modules,
    })
}

/// Fingerprint of the image at `base`. Size alone cannot tell two DLLs apart across
/// base-address reuse. The PE link timestamp almost always can.
///
/// # Safety
/// `base` must be the base of a mapped PE image that stays mapped for the call.
unsafe fn fingerprint(base: u64, size: u64) -> u64 {
    let timestamp = pe_timestamp(base) as u64;
    super::fingerprint_of(&[size, timestamp], b"")
}

// ---------------------------------------------------------------------------
// DLL notifications. ntdll exports these but no import library or windows-sys binding
// covers them, so they are resolved by name.
// ---------------------------------------------------------------------------

#[repr(C)]
struct UnicodeString {
    length: u16,
    maximum_length: u16,
    buffer: *const u16,
}

/// `LDR_DLL_LOADED_NOTIFICATION_DATA` and `LDR_DLL_UNLOADED_NOTIFICATION_DATA` have the
/// same layout.
#[repr(C)]
struct NotificationData {
    flags: u32,
    full_dll_name: *const UnicodeString,
    base_dll_name: *const UnicodeString,
    dll_base: *mut c_void,
    size_of_image: u32,
}

const LDR_DLL_NOTIFICATION_REASON_LOADED: u32 = 1;
const LDR_DLL_NOTIFICATION_REASON_UNLOADED: u32 = 2;

type NotificationFn =
    unsafe extern "system" fn(reason: u32, data: *const NotificationData, context: *const c_void);
type RegisterFn = unsafe extern "system" fn(
    flags: u32,
    function: NotificationFn,
    context: *const c_void,
    cookie: *mut *mut c_void,
) -> i32;
type UnregisterFn = unsafe extern "system" fn(cookie: *mut c_void) -> i32;

/// The registration cookie, null when not registered.
static COOKIE: AtomicPtr<c_void> = AtomicPtr::new(core::ptr::null_mut());
/// True while notifications keep the module set current, so `refresh` can return early.
static ACTIVE: AtomicBool = AtomicBool::new(false);
/// Serializes registration and unregistration.
static REGISTRATION: Mutex<()> = Mutex::new(());

unsafe fn ntdll_fn(name: &core::ffi::CStr) -> Option<*const c_void> {
    let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
    if ntdll.is_null() {
        return None;
    }
    GetProcAddress(ntdll, name.as_ptr().cast()).map(|f| f as *const c_void)
}

/// Register for DLL load and unload notifications. Idempotent. Returns whether
/// notifications are active afterwards. Off the signal path.
pub fn register_notifications() -> bool {
    let _guard = REGISTRATION.lock().unwrap_or_else(|e| e.into_inner());
    if !COOKIE.load(Ordering::Acquire).is_null() {
        return true;
    }
    // SAFETY: resolving and calling documented ntdll exports with the documented
    // signatures; the callback is a process-lifetime `extern "system"` function.
    unsafe {
        let Some(register) = ntdll_fn(c"LdrRegisterDllNotification") else {
            return false;
        };
        let register: RegisterFn = core::mem::transmute(register);
        let mut cookie: *mut c_void = core::ptr::null_mut();
        if register(0, on_notification, core::ptr::null(), &mut cookie) != 0 || cookie.is_null() {
            return false;
        }
        COOKIE.store(cookie, Ordering::Release);
    }
    ACTIVE.store(true, Ordering::Release);
    true
}

/// Unregister the notifications, so that no callback runs after this (see `super::fini`).
/// Idempotent. Off the signal path.
pub fn unregister_notifications() {
    let _guard = REGISTRATION.lock().unwrap_or_else(|e| e.into_inner());
    let cookie = COOKIE.swap(core::ptr::null_mut(), Ordering::AcqRel);
    if cookie.is_null() {
        return;
    }
    ACTIVE.store(false, Ordering::Release);
    // SAFETY: `cookie` came from a successful registration above.
    unsafe {
        if let Some(unregister) = ntdll_fn(c"LdrUnregisterDllNotification") {
            let unregister: UnregisterFn = core::mem::transmute(unregister);
            unregister(cookie);
        }
    }
}

/// True while the loader's notifications keep the module set current.
pub fn notifications_active() -> bool {
    ACTIVE.load(Ordering::Acquire)
}

/// The notification callback. Runs on the loading or unloading thread with the loader busy
/// (see the module docs). It must never call the loader.
unsafe extern "system" fn on_notification(
    reason: u32,
    data: *const NotificationData,
    _context: *const c_void,
) {
    if data.is_null() {
        return;
    }
    // SAFETY: the loader passes a valid structure for the duration of the call.
    let d = &*data;
    let base = d.dll_base as u64;
    let size = d.size_of_image as u64;
    if base == 0 || size == 0 {
        return;
    }
    match reason {
        LDR_DLL_NOTIFICATION_REASON_LOADED => {
            // The image is mapped and the loader holds it until the callback returns.
            let fp = fingerprint(base, size);
            let module = build_pe_module(base, size, false);
            super::image_added(base, fp, module);
        }
        LDR_DLL_NOTIFICATION_REASON_UNLOADED => super::image_removed(base),
        _ => {}
    }
}

/// Read IMAGE_FILE_HEADER.TimeDateStamp from the loaded image at `base` (0 if the
/// headers don't validate).
///
/// # Safety
/// `base` must be the base of a currently-pinned, mapped PE image.
unsafe fn pe_timestamp(base: u64) -> u32 {
    let p = base as *const u8;
    if read_u16(p, 0) != 0x5a4d {
        return 0;
    }
    let e_lfanew = read_u32(p, 0x3c) as usize;
    if read_u32(p, e_lfanew) != 0x0000_4550 {
        return 0;
    }
    // IMAGE_FILE_HEADER starts after the 4-byte PE signature; TimeDateStamp is at +4.
    read_u32(p, e_lfanew + 4 + 4)
}

/// Parse the in-memory PE at `base` and build a framehop PE module if it has `.pdata`.
/// `copy` copies the unwind sections instead of reading them in place.
unsafe fn build_pe_module(base: u64, size: u64, copy: bool) -> Option<Module<Bytes>> {
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
    let sec = |want: &[u8], copy: bool| -> Option<(u32, Bytes)> {
        for i in 0..num_sections {
            let sh_off = section_table + i * SECTION_HEADER_SIZE;
            let mut name = [0u8; 8];
            core::ptr::copy_nonoverlapping(base_ptr.add(sh_off), name.as_mut_ptr(), 8);
            if section_name_eq(&name, want) {
                let virtual_size = read_u32(base_ptr, sh_off + 8);
                let rva = read_u32(base_ptr, sh_off + 12);
                // In the loaded image only the section's *virtual* extent is guaranteed
                // mapped; SizeOfRawData is a file-alignment quantity that may legally
                // exceed it, so using max(virtual, raw) could read past the mapping.
                // Clamp to the image extent for the (also legal) rva+size > SizeOfImage.
                let len = (virtual_size as u64).min(size.saturating_sub(rva as u64)) as usize;
                if len == 0 {
                    return None;
                }
                let bytes = core::slice::from_raw_parts(base_ptr.add(rva as usize), len);
                let bytes = if copy {
                    Bytes::copy_of(bytes)
                } else {
                    Bytes::Mapped(bytes)
                };
                return Some((rva, bytes));
            }
        }
        None
    };

    let pdata = sec(b".pdata", copy)?; // no .pdata => not framehop-unwindable here
    let xdata = sec(b".xdata", copy);
    let rdata = sec(b".rdata", copy); // MSVC places unwind info in .rdata
    let text = sec(b".text", false);

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
