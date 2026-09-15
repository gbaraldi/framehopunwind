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

// libc deprecated its Mach-O types in favor of the `mach2` crate; the layouts are
// ABI-stable, so keep them rather than add a dependency for a few structs.
#![allow(deprecated)]

use std::ops::Range;
use std::sync::Once;

use framehop::{ExplicitModuleSectionInfo, Module};

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
            if is_text {
                // Compact unwind rules need instruction bytes even for large images:
                // prologues/epilogues and indirect stack sizes cannot be recovered
                // reliably from the compact opcode alone.
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

#[cfg(test)]
mod tests {
    use super::*;
    use framehop::x86_64::{CacheX86_64, UnwindRegsX86_64, UnwinderX86_64};
    use framehop::{FrameAddress, MustNotAllocateDuringUnwind, Unwinder};

    #[test]
    fn large_image_keeps_text_for_prologue_and_epilogue_unwinding() {
        // A mapped Mach-O image larger than the old 4 MiB cutoff. Only its data
        // is interpreted, so this x86-64 fixture also runs on an arm64 Mac.
        let mut image = vec![0u64; (5 * 1024 * 1024) / 8];
        let base = image.as_ptr() as u64;
        let len = image.len() * 8;
        let bytes =
            unsafe { core::slice::from_raw_parts_mut(image.as_mut_ptr().cast::<u8>(), len) };
        fn put32(b: &mut [u8], at: usize, value: u32) {
            b[at..at + 4].copy_from_slice(&value.to_le_bytes());
        }
        fn put64(b: &mut [u8], at: usize, value: u64) {
            b[at..at + 8].copy_from_slice(&value.to_le_bytes());
        }
        put32(bytes, 0, 0xfeedfacf);
        put32(bytes, 4, 0x01000007); // CPU_TYPE_X86_64
        put32(bytes, 16, 1); // ncmds
        put32(bytes, 20, 72 + 2 * 80);
        let segment = 32;
        put32(bytes, segment, LC_SEGMENT_64);
        put32(bytes, segment + 4, 72 + 2 * 80);
        bytes[segment + 8..segment + 14].copy_from_slice(b"__TEXT");
        put64(bytes, segment + 24, base);
        put64(bytes, segment + 32, len as u64);
        put64(bytes, segment + 48, len as u64);
        put32(bytes, segment + 64, 2);
        for (i, name, offset, size) in [
            (0, b"__text".as_slice(), 0x1000, 16),
            (1, b"__unwind_info".as_slice(), 0x2000, 68),
        ] {
            let section = segment + 72 + i * 80;
            bytes[section..section + name.len()].copy_from_slice(name);
            bytes[section + 16..section + 22].copy_from_slice(b"__TEXT");
            put64(bytes, section + 32, base + offset);
            put64(bytes, section + 40, size);
        }
        // push rbp; mov rbp,rsp; nop; pop rbp; ret
        bytes[0x1000..0x1007].copy_from_slice(&[0x55, 0x48, 0x89, 0xe5, 0x90, 0x5d, 0xc3]);
        // Compact unwind header, index with sentinel, regular second-level page.
        let words = [
            1,
            0,
            0,
            0,
            0,
            28,
            2,
            0x1000,
            52,
            0,
            0x1010,
            0,
            0,
            2,
            (1 << 16) | 8,
            0x1000,
            0x01000000,
        ];
        for (i, word) in words.into_iter().enumerate() {
            put32(bytes, 0x2000 + i * 4, word);
        }
        let (_, _, module) =
            unsafe { build_image(image.as_ptr().cast::<libc::mach_header_64>(), 0) }.unwrap();
        let mut unw = UnwinderX86_64::new();
        unw.add_module(module.unwrap());
        // The snapshot must retain its own text copy after the image goes away.
        drop(image);
        let mut stack = [0u64; 0x180 / 8];
        stack[0x130 / 8] = 0x160;
        stack[0x138 / 8] = 0x12345;
        for (offset, sp, bp) in [(1, 0x130, 0x160), (4, 0x130, 0x130), (6, 0x138, 0x160)] {
            let pc = base + 0x1000 + offset;
            let mut regs = UnwindRegsX86_64::new(pc, sp, bp);
            let mut cache = CacheX86_64::<MustNotAllocateDuringUnwind>::new_in();
            let result = unw.unwind_frame(
                FrameAddress::from_instruction_pointer(pc),
                &mut regs,
                &mut cache,
                &mut |addr| stack.get((addr / 8) as usize).copied().ok_or(()),
            );
            assert_eq!(result, Ok(Some(0x12345)), "instruction offset {offset}");
            assert_eq!(regs.sp(), 0x140);
            assert_eq!(regs.bp(), 0x160);
        }
    }
}
