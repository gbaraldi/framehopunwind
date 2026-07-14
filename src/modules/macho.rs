//! macOS Mach-O static-module tracking via dyld's add/remove-image callbacks.
//!
//! [`init_callbacks`] installs `_dyld_register_func_for_add_image` /
//! `_dyld_register_func_for_remove_image` once. dyld invokes the add callback
//! synchronously for every already-loaded image at installation time and then for every
//! later load; the remove callback fires while the unloading image is still mapped. Both
//! run under dyld's lock, so — unlike the index-based `_dyld_image_count()` /
//! `_dyld_get_image_header(i)` APIs — the header pointer can never dangle under us.
//!
//! For each image we read its `__TEXT` segment + relevant sections (`__unwind_info`,
//! `__eh_frame`, `__text`, `__stubs`, `__stub_helper`, `__got`), copy them into owned
//! memory, and build a framehop module. Section bytes live at `section.addr + slide`; the
//! un-slid `addr` is the SVMA.

// libc deprecated its Mach-O types (mach_header_64, segment_command_64, ...) in favor of
// the `mach2` crate. The layouts are ABI-stable and these are our only Mach-O uses, so
// keep libc's definitions rather than adding a dependency for a handful of structs.
#![allow(deprecated)]

use std::ops::Range;
use std::sync::Once;

use framehop::{ExplicitModuleSectionInfo, Module};

use crate::state::Bytes;

const LC_SEGMENT_64: u32 = 0x19;

/// Cap on copying an image's `__TEXT` segment. framehop only consults the text bytes for
/// instruction analysis at prologue/epilogue boundaries (compact unwind); retaining the
/// whole `__TEXT` of every image would cost ~100 MB of permanent RSS in a Julia process
/// (libLLVM alone is ~60 MB). Larger images simply skip the copy and take the rule-only
/// behavior at frame edges.
const TEXT_COPY_MAX: u64 = 4 << 20;

// `section_64` is not exposed by the `libc` crate on Apple targets, so we define it here
// (stable Mach-O layout).
#[repr(C)]
struct Section64 {
    sectname: [libc::c_char; 16],
    segname: [libc::c_char; 16],
    addr: u64,
    size: u64,
    offset: u32,
    align: u32,
    reloff: u32,
    nreloc: u32,
    flags: u32,
    reserved1: u32,
    reserved2: u32,
    reserved3: u32,
}

extern "C" {
    // Callbacks receive (const struct mach_header*, intptr_t vmaddr_slide); on 64-bit
    // every loaded image's header is really a mach_header_64.
    fn _dyld_register_func_for_add_image(
        func: extern "C" fn(*const libc::mach_header, libc::intptr_t),
    );
    fn _dyld_register_func_for_remove_image(
        func: extern "C" fn(*const libc::mach_header, libc::intptr_t),
    );
}

static CALLBACKS: Once = Once::new();

/// Install the dyld image callbacks (idempotent). Registration replays the add callback
/// for every already-loaded image synchronously, so batch those into one snapshot.
pub fn init_callbacks() {
    CALLBACKS.call_once(|| {
        super::macos_begin_batch();
        // SAFETY: registering process-lifetime callbacks; this cdylib is never unloaded.
        unsafe {
            _dyld_register_func_for_add_image(on_add_image);
            _dyld_register_func_for_remove_image(on_remove_image);
        }
        super::macos_end_batch();
    });
}

extern "C" fn on_add_image(header: *const libc::mach_header, slide: libc::intptr_t) {
    if header.is_null() {
        return;
    }
    // SAFETY: dyld passes a valid, mapped image header (and holds its lock for the
    // duration of the callback); we only parse memory the image covers.
    if let Some((key, fp, Some(m))) =
        unsafe { build_image(header as *const libc::mach_header_64, slide as u64) }
    {
        super::macos_add_image(key, fp, m);
    }
}

extern "C" fn on_remove_image(header: *const libc::mach_header, slide: libc::intptr_t) {
    if header.is_null() {
        return;
    }
    // SAFETY: the unloading image is still mapped while the callback runs.
    if let Some(key) = unsafe { image_key(header as *const libc::mach_header_64, slide as u64) } {
        super::macos_remove_image(key);
    }
}

/// First-pass scan: the image's key (`__TEXT` AVMA). Cheap; no copies.
unsafe fn image_key(header: *const libc::mach_header_64, slide: u64) -> Option<u64> {
    let h = &*header;
    let mut p = (header as *const u8).add(core::mem::size_of::<libc::mach_header_64>());
    for _ in 0..h.ncmds {
        let lc = &*(p as *const libc::load_command);
        if lc.cmd == LC_SEGMENT_64 {
            let seg = &*(p as *const libc::segment_command_64);
            if seg_name_eq(&seg.segname, b"__TEXT") {
                return Some(seg.vmaddr.wrapping_add(slide));
            }
        }
        p = p.add(lc.cmdsize as usize);
    }
    None
}

/// Parse one loaded image. Returns `(key, fingerprint, Some(module))`, or
/// `(key, fingerprint, None)` if it has no unwind info, or `None` without usable geometry.
unsafe fn build_image(
    header: *const libc::mach_header_64,
    slide: u64,
) -> Option<(u64, u64, Option<Module<Bytes>>)> {
    let h = &*header;
    let ncmds = h.ncmds;

    let mut text_vmaddr: Option<u64> = None;
    let mut text_vmsize: u64 = 0;

    let mut unwind_info: Option<Bytes> = None;
    let mut eh_frame: Option<(Bytes, Range<u64>)> = None;
    let mut text_segment: Option<(Bytes, Range<u64>)> = None;
    let mut stubs_svma: Option<Range<u64>> = None;
    let mut stub_helper_svma: Option<Range<u64>> = None;
    let mut got_svma: Option<Range<u64>> = None;
    let mut text_svma: Option<Range<u64>> = None;

    // First pass: find __TEXT to learn the key.
    {
        let mut p = (header as *const u8).add(core::mem::size_of::<libc::mach_header_64>());
        for _ in 0..ncmds {
            let lc = &*(p as *const libc::load_command);
            if lc.cmd == LC_SEGMENT_64 {
                let seg = &*(p as *const libc::segment_command_64);
                if seg_name_eq(&seg.segname, b"__TEXT") {
                    text_vmaddr = Some(seg.vmaddr);
                    text_vmsize = seg.vmsize;
                }
            }
            p = p.add(lc.cmdsize as usize);
        }
    }

    let text_vmaddr = text_vmaddr?;
    let base_avma = text_vmaddr.wrapping_add(slide);
    let key = base_avma;
    // With exact dyld add/remove events the fingerprint is informational (event ordering
    // already handles base-address reuse); keep it cheap.
    let fp = text_vmaddr ^ text_vmsize.rotate_left(32);

    // Second pass: copy sections.
    let mut p = (header as *const u8).add(core::mem::size_of::<libc::mach_header_64>());
    for _ in 0..ncmds {
        let lc = &*(p as *const libc::load_command);
        if lc.cmd == LC_SEGMENT_64 {
            let seg = &*(p as *const libc::segment_command_64);
            let is_text = seg_name_eq(&seg.segname, b"__TEXT");
            if is_text && seg.vmsize <= TEXT_COPY_MAX {
                // Copy __TEXT for instruction analysis — but only for modest images (see
                // TEXT_COPY_MAX); large ones skip the copy rather than bloat RSS.
                let bytes = copy_mem(
                    (seg.vmaddr.wrapping_add(slide)) as *const u8,
                    seg.vmsize as usize,
                );
                text_segment = Some((
                    bytes,
                    Range {
                        start: seg.vmaddr,
                        end: seg.vmaddr + seg.vmsize,
                    },
                ));
            }
            // Iterate sections of this segment.
            let sects = p.add(core::mem::size_of::<libc::segment_command_64>()) as *const Section64;
            for s in 0..seg.nsects {
                let sec = &*sects.add(s as usize);
                let svma = Range {
                    start: sec.addr,
                    end: sec.addr + sec.size,
                };
                let data_ptr = (sec.addr.wrapping_add(slide)) as *const u8;
                match () {
                    _ if sect_name_eq(&sec.sectname, b"__text") => text_svma = Some(svma),
                    _ if sect_name_eq(&sec.sectname, b"__unwind_info") => {
                        unwind_info = Some(copy_mem(data_ptr, sec.size as usize));
                    }
                    _ if sect_name_eq(&sec.sectname, b"__eh_frame") => {
                        eh_frame = Some((copy_mem(data_ptr, sec.size as usize), svma));
                    }
                    _ if sect_name_eq(&sec.sectname, b"__stubs") => stubs_svma = Some(svma),
                    _ if sect_name_eq(&sec.sectname, b"__stub_helper") => {
                        stub_helper_svma = Some(svma)
                    }
                    _ if sect_name_eq(&sec.sectname, b"__got") => got_svma = Some(svma),
                    _ => {}
                }
            }
        }
        p = p.add(lc.cmdsize as usize);
    }

    // Need at least one unwind-info source to be useful.
    if unwind_info.is_none() && eh_frame.is_none() {
        return Some((key, fp, None));
    }

    // No name: resolving it would need dladdr / index-based dyld APIs, which are unsafe
    // to call from inside an image callback. The key is enough for diagnostics.
    let name = format!("<image:{key:#x}>");

    let (eh_frame_bytes, eh_frame_svma) = match eh_frame {
        Some((b, r)) => (Some(b), Some(r)),
        None => (None, None),
    };
    let (text_bytes, text_segment_svma) = match text_segment {
        Some((b, r)) => (Some(b), Some(r)),
        None => (None, None),
    };

    let info: ExplicitModuleSectionInfo<Bytes> = ExplicitModuleSectionInfo {
        base_svma: text_vmaddr,
        text_svma,
        unwind_info,
        eh_frame: eh_frame_bytes,
        eh_frame_svma,
        text_segment: text_bytes,
        text_segment_svma,
        stubs_svma,
        stub_helper_svma,
        got_svma,
        ..Default::default()
    };

    let module = Module::new(
        name,
        Range {
            start: base_avma,
            end: base_avma + text_vmsize,
        },
        base_avma,
        info,
    );
    Some((key, fp, Some(module)))
}

unsafe fn copy_mem(ptr: *const u8, len: usize) -> Bytes {
    core::slice::from_raw_parts(ptr, len)
        .to_vec()
        .into_boxed_slice()
}

fn seg_name_eq(field: &[libc::c_char; 16], name: &[u8]) -> bool {
    name_eq(field, name)
}
fn sect_name_eq(field: &[libc::c_char; 16], name: &[u8]) -> bool {
    name_eq(field, name)
}

/// Compare a fixed 16-byte (possibly NUL-padded, possibly full) Mach-O name field.
fn name_eq(field: &[libc::c_char; 16], name: &[u8]) -> bool {
    if name.len() > 16 {
        return false;
    }
    for (i, &b) in name.iter().enumerate() {
        if field[i] as u8 != b {
            return false;
        }
    }
    // The remainder must be NUL (unless the name fills all 16 bytes).
    name.len() == 16 || field[name.len()] == 0
}
