//! macOS Mach-O static-module enumeration via dyld.
//!
//! For each loaded image we read its `__TEXT` segment + relevant sections
//! (`__unwind_info`, `__eh_frame`, `__text`, `__stubs`, `__stub_helper`, `__got`),
//! copy them into owned memory, and build a framehop module. Section bytes live at
//! `section.addr + slide`; the un-slid `addr` is the SVMA.

use std::collections::HashSet;
use std::ffi::CStr;
use std::ops::Range;

use framehop::{ExplicitModuleSectionInfo, Module};

use super::EnumResult;
use crate::state::Bytes;

const LC_SEGMENT_64: u32 = 0x19;

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

pub fn enumerate(known: &HashSet<u64>) -> EnumResult {
    let mut current_keys = HashSet::new();
    let mut new_modules = Vec::new();

    let count = unsafe { libc::_dyld_image_count() };
    for i in 0..count {
        let header = unsafe { libc::_dyld_get_image_header(i) } as *const libc::mach_header_64;
        if header.is_null() {
            continue;
        }
        let slide = unsafe { libc::_dyld_get_image_vmaddr_slide(i) } as u64;
        let name_ptr = unsafe { libc::_dyld_get_image_name(i) };

        if let Some((key, module)) = unsafe { build_image(header, slide, name_ptr, known) } {
            current_keys.insert(key);
            if let Some(m) = module {
                new_modules.push((key, m));
            }
        }
    }

    EnumResult {
        current_keys,
        new_modules,
    }
}

/// Parse one loaded image. Returns `(key, Some(module))` for new images, `(key, None)` for
/// already-known images, or `None` if it has no usable geometry.
unsafe fn build_image(
    header: *const libc::mach_header_64,
    slide: u64,
    name_ptr: *const libc::c_char,
    known: &HashSet<u64>,
) -> Option<(u64, Option<Module<Bytes>>)> {
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

    // We do a first pass only to learn the key (so we can skip copying for known images).
    // Find __TEXT first.
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
    let base_avma = text_vmaddr + slide;
    let key = base_avma;
    if known.contains(&key) {
        return Some((key, None));
    }

    // Second pass: copy sections.
    let mut p = (header as *const u8).add(core::mem::size_of::<libc::mach_header_64>());
    for _ in 0..ncmds {
        let lc = &*(p as *const libc::load_command);
        if lc.cmd == LC_SEGMENT_64 {
            let seg = &*(p as *const libc::segment_command_64);
            let is_text = seg_name_eq(&seg.segname, b"__TEXT");
            if is_text {
                // Copy the whole __TEXT segment for instruction analysis.
                let bytes = copy_mem((seg.vmaddr + slide) as *const u8, seg.vmsize as usize);
                text_segment = Some((
                    bytes,
                    Range {
                        start: seg.vmaddr,
                        end: seg.vmaddr + seg.vmsize,
                    },
                ));
            }
            // Iterate sections of this segment.
            let sects = (p as *const u8).add(core::mem::size_of::<libc::segment_command_64>())
                as *const Section64;
            for s in 0..seg.nsects {
                let sec = &*sects.add(s as usize);
                let svma = Range {
                    start: sec.addr,
                    end: sec.addr + sec.size,
                };
                let data_ptr = (sec.addr + slide) as *const u8;
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
        return Some((key, None)); // tracked (so we don't rescan), but no module
    }

    let name = if name_ptr.is_null() {
        String::from("<image>")
    } else {
        CStr::from_ptr(name_ptr).to_string_lossy().into_owned()
    };

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
    Some((key, Some(module)))
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
