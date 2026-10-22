//! Linux / FreeBSD static-module enumeration via `dl_iterate_phdr`.
//!
//! For each loaded ELF object we locate the in-memory `.eh_frame_hdr`
//! (`PT_GNU_EH_FRAME`), decode it to find `.eh_frame`, back both sections with bytes that
//! outlive an unload (see `Bytes` and `section_bytes`), and build a framehop
//! `EhFrameHdrAndEhFrame` module. ELF SVMA == AVMA - bias,
//! so `base_svma = 0` and `base_avma = dlpi_addr`.

use std::collections::HashMap;
use std::ffi::CStr;
use std::ops::Range;
use std::sync::Arc;

use framehop::{ExplicitModuleSectionInfo, Module};

use super::EnumResult;
use crate::state::{Bytes, FileMap};

const PT_LOAD: u32 = 1;
const PT_DYNAMIC: u32 = 2;
const PT_NOTE: u32 = 4;
const PT_GNU_EH_FRAME: u32 = 0x6474_e550;
const PF_X: u32 = 1;
const DT_NULL: u64 = 0;
const DT_TEXTREL: u64 = 22;
const DT_FLAGS: u64 = 30;
const DF_TEXTREL: u64 = 0x4;
const NT_GNU_BUILD_ID: u32 = 3;
/// Bytes compared at each end of `.eh_frame_hdr` for an object without a build ID.
const HDR_SAMPLE: usize = 256;

/// A module's unwind sections + geometry, captured under the dl lock.
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
    known: &'a HashMap<u64, u64>,
    current: HashMap<u64, u64>,
    specs: Vec<RawSpec>,
}

pub fn enumerate(known: &HashMap<u64, u64>) -> EnumResult {
    let collector = collect(known);
    let new_modules = collector
        .specs
        .into_iter()
        .map(|s| (s.key, build_module(s)))
        .collect();

    EnumResult {
        current: collector.current,
        new_modules,
    }
}

fn collect(known: &HashMap<u64, u64>) -> Collector<'_> {
    let mut collector = Collector {
        known,
        current: HashMap::new(),
        specs: Vec::new(),
    };
    // SAFETY: standard dl_iterate_phdr usage; callback does not unwind or call dlopen.
    unsafe {
        libc::dl_iterate_phdr(
            Some(callback),
            &mut collector as *mut _ as *mut libc::c_void,
        );
    }
    collector
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

    let bias = info.dlpi_addr;
    if info.dlpi_phdr.is_null() || info.dlpi_phnum == 0 {
        return 0;
    }
    let phdrs = unsafe { core::slice::from_raw_parts(info.dlpi_phdr, info.dlpi_phnum as usize) };

    // Geometry: union of PT_LOAD; remember the executable segment and the eh_frame_hdr.
    let mut avma_lo = u64::MAX;
    let mut avma_hi = 0u64;
    let mut text_svma: Option<Range<u64>> = None;
    let mut hdr_phdr: Option<&libc::Elf64_Phdr> = None;

    for ph in phdrs {
        match ph.p_type {
            PT_LOAD => {
                let lo = bias + ph.p_vaddr;
                let hi = lo + ph.p_memsz;
                if lo < avma_lo {
                    avma_lo = lo;
                }
                if hi > avma_hi {
                    avma_hi = hi;
                }
                if (ph.p_flags & PF_X) != 0 && text_svma.is_none() {
                    text_svma = Some(Range {
                        start: ph.p_vaddr,
                        end: ph.p_vaddr + ph.p_memsz,
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

    // Fingerprint the image so a dlclose/dlopen pair that reuses the same base address is
    // detected as a *different* module (and re-scanned) instead of silently keeping the
    // old image's unwind data. Name + eh_frame_hdr geometry + load span is cheap and
    // distinguishes any two real images.
    let name_bytes: &[u8] = if info.dlpi_name.is_null() {
        b""
    } else {
        unsafe { CStr::from_ptr(info.dlpi_name) }.to_bytes()
    };
    let (hdr_vaddr, hdr_memsz) = match hdr_phdr {
        Some(p) => (p.p_vaddr, p.p_memsz),
        None => (0, 0),
    };
    let fp = super::fingerprint_of(&[hdr_vaddr, hdr_memsz, avma_hi - avma_lo], name_bytes);
    collector.current.insert(key, fp);

    // Only build modules we don't already know about (same key AND
    // same fingerprint; a changed fingerprint means the base address was reused).
    if collector.known.get(&key) == Some(&fp) {
        return 0;
    }

    let hdr_phdr = match hdr_phdr {
        Some(p) => p,
        None => return 0, // no unwind info; framehop will fp-fallback for this module
    };

    let hdr_avma = bias + hdr_phdr.p_vaddr;
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

    // SAFETY: both ranges are mapped, and dl_iterate_phdr holds the loader lock, so the
    // object cannot be unloaded during the call.
    let (eh_frame_hdr, eh_frame) = unsafe {
        section_bytes(
            info,
            phdrs,
            avma_lo,
            hdr_avma..hdr_avma + hdr_len as u64,
            eh_frame_avma..eh_frame_avma + eh_frame_len as u64,
        )
    };

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

/// Bytes for the `.eh_frame_hdr` and `.eh_frame` AVMA ranges of one object, valid for as long
/// as the module that holds them.
///
/// The objects that the kernel maps are never unloaded, so their sections are read in place.
/// Other objects get a fresh mapping of their file (see `map_file`), or a copy where that
/// fails.
///
/// # Safety
/// Both ranges must be mapped, and the object must stay loaded during the call.
unsafe fn section_bytes(
    info: &libc::dl_phdr_info,
    phdrs: &[libc::Elf64_Phdr],
    avma_lo: u64,
    hdr: Range<u64>,
    eh_frame: Range<u64>,
) -> (Bytes, Bytes) {
    let live = |r: &Range<u64>| unsafe {
        core::slice::from_raw_parts(r.start as *const u8, (r.end - r.start) as usize)
    };
    if mapped_by_kernel(info, avma_lo) {
        // SAFETY: the object stays mapped until the process exits.
        let hdr: &'static [u8] = live(&hdr);
        let eh_frame: &'static [u8] = live(&eh_frame);
        return (Bytes::Mapped(hdr), Bytes::Mapped(eh_frame));
    }
    let lo = hdr.start.min(eh_frame.start);
    let hi = hdr.end.max(eh_frame.end);
    if let Some((map, map_avma)) = map_file(info, phdrs, lo..hi, &hdr) {
        let slice = |r: &Range<u64>| {
            Bytes::File(
                map.clone(),
                (r.start - map_avma) as usize..(r.end - map_avma) as usize,
            )
        };
        return (slice(&hdr), slice(&eh_frame));
    }
    (Bytes::copy_of(live(&hdr)), Bytes::copy_of(live(&eh_frame)))
}

/// True for the objects that the kernel maps before the loader runs: the executable (or the
/// interpreter, when it is run as a command), the interpreter, and the vDSO. These are
/// never unloaded.
fn mapped_by_kernel(info: &libc::dl_phdr_info, avma_lo: u64) -> bool {
    let phdr = info.dlpi_phdr as u64;
    #[cfg(target_os = "linux")]
    let bases = [auxv(libc::AT_BASE), auxv(libc::AT_SYSINFO_EHDR)];
    #[cfg(target_os = "freebsd")]
    let bases = [auxv(libc::AT_BASE)];
    auxv(libc::AT_PHDR) == phdr || bases.iter().any(|&b| b != 0 && b == avma_lo)
}

/// The auxiliary vector entry `kind`, or 0 if it is absent.
#[cfg(target_os = "linux")]
fn auxv(kind: libc::c_ulong) -> u64 {
    // SAFETY: getauxval has no preconditions.
    unsafe { libc::getauxval(kind) as u64 }
}

/// The auxiliary vector entry `kind`, or 0 if it is absent.
#[cfg(target_os = "freebsd")]
fn auxv(kind: libc::c_int) -> u64 {
    let mut value: libc::c_ulong = 0;
    // SAFETY: `value` is a writable buffer of the size passed.
    let ok = unsafe {
        libc::elf_aux_info(
            kind,
            &mut value as *mut _ as *mut libc::c_void,
            core::mem::size_of::<libc::c_ulong>() as libc::c_int,
        )
    } == 0;
    if ok {
        value as u64
    } else {
        0
    }
}

/// A private, read-only mapping of the part of the object's file that holds the AVMA range
/// `want`, and the AVMA that the start of the mapping corresponds to.
///
/// Returns `None` if the bytes in the file could differ from the bytes in memory. The
/// loader writes to the read-only segment only for text relocations, so an object with them
/// is refused. The file at the object's path must be the one that was loaded: it must have
/// the same program headers, and the same build ID or, without one, the same first and last
/// `HDR_SAMPLE` bytes of `.eh_frame_hdr` (the `hdr` range). The search table there holds
/// the address of the first and last FDE, so a rebuild with different code is very likely
/// to differ. Every check costs the same for any section size, because a cost that grows
/// with `.eh_frame` is what the mapping avoids. A file that passes but differs gives wrong
/// frames, not bad reads: framehop reads only within the slices, which lie in the mapping.
///
/// The check compares contents, not inode numbers, because overlayfs reports different ones
/// to `stat` and `/proc/self/maps`.
///
/// # Safety
/// `want` and `hdr` must be mapped, and the object must stay loaded during the call.
unsafe fn map_file(
    info: &libc::dl_phdr_info,
    phdrs: &[libc::Elf64_Phdr],
    want: Range<u64>,
    hdr: &Range<u64>,
) -> Option<(Arc<FileMap>, u64)> {
    if info.dlpi_name.is_null() {
        return None;
    }
    let path = CStr::from_ptr(info.dlpi_name);
    // Names without a slash (the vDSO, objects the loader names but did not open by path)
    // are not openable paths.
    if !path.to_bytes().contains(&b'/') || has_text_relocations(phdrs, info.dlpi_addr) {
        return None;
    }

    // The bytes must come from one PT_LOAD and lie within its file-backed part.
    let bias = info.dlpi_addr;
    let (lo, hi) = (want.start - bias, want.end - bias);
    let seg = phdrs
        .iter()
        .find(|ph| ph.p_type == PT_LOAD && ph.p_vaddr <= lo && hi <= ph.p_vaddr + ph.p_filesz)?;
    let page = libc::sysconf(libc::_SC_PAGESIZE);
    if page <= 0 {
        return None;
    }
    let page = page as u64;
    let off = seg.p_offset + (lo - seg.p_vaddr);
    let map_off = off & !(page - 1);
    let end_off = seg.p_offset + (hi - seg.p_vaddr);
    let map_avma = want.start - (off - map_off);
    // The loader maps segments page by page, so file offsets and addresses are congruent.
    if !map_avma.is_multiple_of(page) {
        return None;
    }

    let fd = Fd(libc::open(path.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC));
    if fd.0 < 0 {
        return None;
    }
    let mut st: libc::stat = core::mem::zeroed();
    if libc::fstat(fd.0, &mut st) != 0
        || (st.st_mode & libc::S_IFMT) != libc::S_IFREG
        || (st.st_size as u64) < end_off
    {
        return None;
    }
    if !same_phdrs(fd.0, phdrs) {
        return None;
    }
    let build_id = same_build_id(fd.0, phdrs, bias)?;

    let len = (end_off - map_off) as usize;
    let ptr = libc::mmap(
        core::ptr::null_mut(),
        len,
        libc::PROT_READ,
        libc::MAP_PRIVATE,
        fd.0,
        map_off as libc::off_t,
    );
    if ptr == libc::MAP_FAILED {
        return None;
    }
    let map = Arc::new(FileMap::from_raw(ptr as *const u8, len));
    if !build_id {
        let len = (hdr.end - hdr.start) as usize;
        let n = len.min(HDR_SAMPLE);
        let live = core::slice::from_raw_parts(hdr.start as *const u8, len);
        let file = &map.bytes()[(hdr.start - map_avma) as usize..][..len];
        if file[..n] != live[..n] || file[len - n..] != live[len - n..] {
            return None;
        }
    }
    Some((map, map_avma))
}

/// A file descriptor, closed on drop.
struct Fd(libc::c_int);

impl Drop for Fd {
    fn drop(&mut self) {
        if self.0 >= 0 {
            // SAFETY: we own the descriptor.
            unsafe { libc::close(self.0) };
        }
    }
}

/// Read exactly `buf.len()` bytes at `off`.
fn pread_exact(fd: libc::c_int, buf: &mut [u8], off: u64) -> bool {
    let mut done = 0;
    while done < buf.len() {
        // SAFETY: the destination is the unwritten tail of `buf`.
        let n = unsafe {
            libc::pread(
                fd,
                buf[done..].as_mut_ptr() as *mut libc::c_void,
                buf.len() - done,
                (off + done as u64) as libc::off_t,
            )
        };
        if n <= 0 {
            return false;
        }
        done += n as usize;
    }
    true
}

/// True if the file's program headers equal the loaded object's, so that file offsets
/// computed from the loaded headers are valid in the file.
fn same_phdrs(fd: libc::c_int, phdrs: &[libc::Elf64_Phdr]) -> bool {
    let mut ehdr = [0u8; 64];
    if !pread_exact(fd, &mut ehdr, 0) || ehdr[..4] != *b"\x7fELF" || ehdr[4] != 2 {
        return false; // not a 64-bit ELF file
    }
    let phoff = u64::from_ne_bytes(ehdr[32..40].try_into().unwrap());
    let phentsize = u16::from_ne_bytes(ehdr[54..56].try_into().unwrap()) as usize;
    let phnum = u16::from_ne_bytes(ehdr[56..58].try_into().unwrap()) as usize;
    let size = core::mem::size_of::<libc::Elf64_Phdr>();
    if phentsize != size || phnum != phdrs.len() {
        return false;
    }
    let mut file = vec![0u8; size * phnum];
    // SAFETY: Elf64_Phdr is plain data without padding.
    let live = unsafe { core::slice::from_raw_parts(phdrs.as_ptr() as *const u8, file.len()) };
    pread_exact(fd, &mut file, phoff) && file == live
}

/// Compare the loaded object's GNU build ID with the file's: `Some(true)` if they are equal,
/// `Some(false)` if the object has none, and `None` if they differ.
///
/// # Safety
/// The object's PT_NOTE segments must be mapped.
unsafe fn same_build_id(fd: libc::c_int, phdrs: &[libc::Elf64_Phdr], bias: u64) -> Option<bool> {
    for ph in phdrs.iter().filter(|ph| ph.p_type == PT_NOTE) {
        let notes =
            core::slice::from_raw_parts((bias + ph.p_vaddr) as *const u8, ph.p_filesz as usize);
        let align = if ph.p_align == 8 { 8 } else { 4 };
        let pad = |n: usize| (n + align - 1) & !(align - 1);
        let mut at = 0usize;
        while at + 12 <= notes.len() {
            let word = |i: usize| u32::from_ne_bytes(notes[at + i..at + i + 4].try_into().unwrap());
            let (namesz, descsz, kind) = (word(0) as usize, word(4) as usize, word(8));
            let desc = at + 12 + pad(namesz);
            let next = desc + pad(descsz);
            if desc + descsz > notes.len() {
                break;
            }
            if kind == NT_GNU_BUILD_ID && notes[at + 12..at + 12 + namesz] == *b"GNU\0" {
                let live = &notes[desc..desc + descsz];
                let mut file = vec![0u8; descsz];
                let off = ph.p_offset + desc as u64;
                return (pread_exact(fd, &mut file, off) && file == live).then_some(true);
            }
            at = next;
        }
    }
    Some(false)
}

/// True if the object has text relocations, which make the loader write to its read-only
/// segment.
///
/// # Safety
/// The object's PT_DYNAMIC segment must be mapped.
unsafe fn has_text_relocations(phdrs: &[libc::Elf64_Phdr], bias: u64) -> bool {
    let Some(dynamic) = phdrs.iter().find(|ph| ph.p_type == PT_DYNAMIC) else {
        return false;
    };
    let entries = core::slice::from_raw_parts(
        (bias + dynamic.p_vaddr) as *const [u64; 2],
        dynamic.p_memsz as usize / 16,
    );
    for &[tag, value] in entries {
        match tag {
            DT_NULL => break,
            DT_TEXTREL => return true,
            DT_FLAGS if value & DF_TEXTREL != 0 => return true,
            _ => {}
        }
    }
    false
}

/// The loader's counts of objects ever loaded and unloaded, or `None` if the C library does
/// not report them. Equal counts mean the set of loaded objects has not changed.
pub fn load_counts() -> Option<(u64, u64)> {
    extern "C" fn first(
        info: *mut libc::dl_phdr_info,
        size: libc::size_t,
        data: *mut libc::c_void,
    ) -> libc::c_int {
        let end = core::mem::offset_of!(libc::dl_phdr_info, dlpi_subs)
            + core::mem::size_of::<libc::c_ulonglong>();
        // Older C libraries pass a shorter struct without the counts.
        if size >= end {
            // SAFETY: libc guarantees `info` is valid for the call and `size` bytes long;
            // `data` is the `Option` passed below.
            let info = unsafe { &*info };
            let counts = unsafe { &mut *(data as *mut Option<(u64, u64)>) };
            *counts = Some((info.dlpi_adds, info.dlpi_subs));
        }
        1 // every entry carries the same counts
    }
    let mut counts: Option<(u64, u64)> = None;
    // SAFETY: standard dl_iterate_phdr usage; the callback only reads its arguments.
    unsafe {
        libc::dl_iterate_phdr(Some(first), &mut counts as *mut _ as *mut libc::c_void);
    }
    counts
}

/// Find the end AVMA of the PT_LOAD segment containing `addr`.
fn containing_segment_end(phdrs: &[libc::Elf64_Phdr], bias: u64, addr: u64) -> Option<u64> {
    for ph in phdrs {
        if ph.p_type != PT_LOAD {
            continue;
        }
        let lo = bias + ph.p_vaddr;
        let hi = lo + ph.p_memsz;
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
        0x00 => 0u64,       // DW_EH_PE_absptr
        0x10 => field_avma, // DW_EH_PE_pcrel
        _ => return None,   // datarel/textrel/etc. not expected for eh_frame_ptr
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Every loaded object's tables equal its live bytes, the executable's are read in
    /// place, and libc's come from a mapping of its file.
    #[test]
    fn section_bytes_match_live_memory() {
        let known = HashMap::new();
        let specs = collect(&known).specs;
        let live = |base: u64, svma: &Range<u64>| unsafe {
            core::slice::from_raw_parts(
                (base + svma.start) as *const u8,
                (svma.end - svma.start) as usize,
            )
        };
        for s in &specs {
            assert_eq!(
                *s.eh_frame_hdr,
                *live(s.base_avma, &s.eh_frame_hdr_svma),
                "{}",
                s.name
            );
            assert_eq!(
                *s.eh_frame,
                *live(s.base_avma, &s.eh_frame_svma),
                "{}",
                s.name
            );
        }
        let main = specs
            .iter()
            .find(|s| s.name == "<main>")
            .expect("no main object");
        assert!(matches!(main.eh_frame, Bytes::Mapped(_)));
        let libc = specs
            .iter()
            .find(|s| s.name.contains("/libc.so"))
            .expect("no libc");
        assert!(matches!(libc.eh_frame, Bytes::File(..)), "{}", libc.name);
    }
}
