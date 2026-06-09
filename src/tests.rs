//! End-to-end unwinding tests on the host target.
//!
//! These validate the two paths that matter most:
//!  * **eager / static** ELF unwinding through the test binary's own `.eh_frame`, using
//!    the full public read path (capture → init → step), cross-checked against an
//!    independent frame-pointer walk;
//!  * the **JIT** module field mapping (`base_svma == base_avma == text_lo`,
//!    `eh_frame_svma == runtime eh_frame address`) on real code.
//!
//! The frame-pointer oracle and the framehop walk are both taken in the *same* frame, so
//! the leaf frames align and a frame-by-frame equality check is meaningful. Frame pointers
//! are forced via `.cargo/config.toml`.

use core::hint::black_box;

use crate::arch::FhContext;
use crate::cursor::{self, FhCursor};
use crate::{capture, modules, state};

/// Walk the frame-pointer chain from `start_rbp`, returning the return addresses
/// `[*(rbp+8), *(*rbp+8), ...]`. x86_64 only.
#[cfg(target_arch = "x86_64")]
fn fp_chain(mut rbp: u64, max: usize) -> Vec<u64> {
    let mut out = Vec::new();
    for _ in 0..max {
        if rbp == 0 || (rbp & 0x7) != 0 {
            break;
        }
        // SAFETY: reading our own live stack frames; 8-aligned, on-stack.
        let saved_rbp = unsafe { core::ptr::read_volatile(rbp as *const u64) };
        let ret = unsafe { core::ptr::read_volatile((rbp + 8) as *const u64) };
        if ret == 0 || saved_rbp <= rbp {
            break;
        }
        out.push(ret);
        rbp = saved_rbp;
    }
    out
}

/// Unwind `ctx` via the public read path (eager/global unwinder), returning reported ips.
fn unwind_via_cursor(ctx: &FhContext, max: usize) -> Vec<u64> {
    let mut cur: FhCursor = unsafe { core::mem::zeroed() };
    let rc = cursor::cursor_init(&mut cur, ctx);
    assert!(rc == 0, "cursor_init failed: {rc}");

    let mut ips = Vec::new();
    let (mut ip, mut sp, mut last_sp) = (0u64, 0u64, 0u64);
    for _ in 0..max {
        let more = cursor::step(&mut cur, &mut ip, &mut sp);
        if ip != 0 {
            if last_sp != 0 {
                assert!(sp > last_sp, "sp not increasing: {sp:#x} <= {last_sp:#x}");
            }
            last_sp = sp;
            ips.push(ip);
        }
        if more <= 0 {
            break;
        }
    }
    cursor::cursor_fini(&mut cur);
    ips
}

/// Capture a context AND the frame-pointer chain in the *same* frame, then unwind via the
/// cursor API. Returns (framehop ips, fp return addresses). `fh[1..]` should equal `fp[..]`.
#[inline(never)]
fn capture_and_walk(max: usize) -> (Vec<u64>, Vec<u64>) {
    let mut ctx = FhContext::zeroed();
    capture::fh_capture_context(&mut ctx);

    #[cfg(target_arch = "x86_64")]
    let fp = {
        let mut rbp: u64;
        unsafe { core::arch::asm!("mov {}, rbp", out(reg) rbp) };
        fp_chain(rbp, max)
    };
    #[cfg(not(target_arch = "x86_64"))]
    let fp = Vec::new();

    let fh = unwind_via_cursor(&ctx, max);
    black_box((fh, fp))
}

// A chain of non-inlinable functions so there is a real call stack to unwind.
#[inline(never)]
fn depth_c() -> (Vec<u64>, Vec<u64>) {
    black_box(capture_and_walk(32))
}
#[inline(never)]
fn depth_b() -> (Vec<u64>, Vec<u64>) {
    black_box(depth_c())
}
#[inline(never)]
fn depth_a() -> (Vec<u64>, Vec<u64>) {
    black_box(depth_b())
}

fn ensure_init() {
    state::init(0);
    #[cfg(unix)]
    crate::stackbounds::register_current_thread();
    modules::init();
}

/// Build a minimal, *valid* `.eh_frame` (one CIE + one FDE covering
/// `[text_lo, text_lo + text_len)`, absptr-encoded so it works at any buffer address,
/// then the 4-byte terminator). Registration validates that sections parse end-to-end,
/// so churn tests need a real section, not just a terminator.
fn synthetic_eh_frame(text_lo: u64, text_len: u64) -> Vec<u8> {
    #[cfg(target_arch = "x86_64")]
    const RA_REG: u8 = 16;
    #[cfg(target_arch = "aarch64")]
    const RA_REG: u8 = 30;

    let mut v = Vec::with_capacity(52);
    // CIE: length 16, id 0, version 1, aug "zR\0", code_align 1, data_align -8 (sleb),
    // ra reg, aug len 1, R = DW_EH_PE_absptr, 3 nops of padding.
    v.extend_from_slice(&16u32.to_le_bytes());
    v.extend_from_slice(&0u32.to_le_bytes());
    v.extend_from_slice(&[1, b'z', b'R', 0, 0x01, 0x78, RA_REG, 0x01, 0x00, 0, 0, 0]);
    debug_assert_eq!(v.len(), 20);
    // FDE: length 24, cie_pointer 24 (distance from this field back to the CIE),
    // pc_begin/pc_range as absptr u64, aug len 0, 3 nops of padding.
    v.extend_from_slice(&24u32.to_le_bytes());
    v.extend_from_slice(&24u32.to_le_bytes());
    v.extend_from_slice(&text_lo.to_le_bytes());
    v.extend_from_slice(&text_len.to_le_bytes());
    v.extend_from_slice(&[0x00, 0, 0, 0]);
    debug_assert_eq!(v.len(), 48);
    // Terminator.
    v.extend_from_slice(&0u32.to_le_bytes());
    v
}

/// Compare framehop ips (offset by one for the capture-site leaf frame) against the fp
/// oracle, for the first `n` comparable frames.
#[cfg(target_arch = "x86_64")]
fn assert_frames_match(label: &str, fh: &[u64], fp: &[u64], min_frames: usize) {
    assert!(
        fh.len() >= min_frames + 1,
        "{label}: too few frames ({}): {fh:#x?}",
        fh.len()
    );
    assert!(!fp.is_empty(), "{label}: fp oracle empty");
    let n = fp.len().min(fh.len() - 1).min(6);
    assert!(n >= min_frames, "{label}: only {n} comparable frames");
    for i in 0..n {
        assert_eq!(
            fh[i + 1],
            fp[i],
            "{label}: frame {i} mismatch: framehop={:#x} fp={:#x}\n fh={fh:#x?}\n fp={fp:#x?}",
            fh[i + 1],
            fp[i]
        );
    }
}

/// Stress the hazard-pointer snapshot: writer threads churn JIT register/deregister while
/// reader threads continuously unwind. This must not crash, deadlock, or corrupt (run also
/// under ASan/TSan in CI for the strongest guarantee).
#[test]
fn concurrent_register_and_unwind_is_stable() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    ensure_init();

    // A small, valid-enough eh_frame-free region is fine: register/deregister just churns
    // the snapshot. We use distinct fake JIT keys with a tiny throwaway eh_frame buffer.
    let stop = Arc::new(AtomicBool::new(false));

    let writers: Vec<_> = (0..3)
        .map(|w| {
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut i = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    let text_lo = 0x10_0000_0000u64 + (w as u64) * 0x100_0000 + (i % 64) * 0x1000;
                    // A minimal valid one-FDE eh_frame so registration passes validation
                    // and actually churns the snapshot.
                    let eh = synthetic_eh_frame(text_lo, 0x800);
                    let rc = crate::modules::register_jit_eh_frame(
                        eh.as_ptr(),
                        eh.len(),
                        text_lo,
                        text_lo + 0x800,
                    );
                    assert_eq!(rc, 0, "synthetic eh_frame must register");
                    crate::modules::deregister_jit(text_lo);
                    i += 1;
                }
            })
        })
        .collect();

    let readers: Vec<_> = (0..4)
        .map(|_| {
            let stop = stop.clone();
            std::thread::spawn(move || {
                #[cfg(unix)]
                crate::stackbounds::register_current_thread();
                let mut count = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    let (fh, _fp) = capture_and_walk(32);
                    assert!(!fh.is_empty());
                    count += 1;
                    if count > 2000 {
                        break;
                    }
                }
            })
        })
        .collect();

    std::thread::sleep(std::time::Duration::from_millis(300));
    stop.store(true, Ordering::Relaxed);
    for r in readers {
        r.join().unwrap();
    }
    for w in writers {
        w.join().unwrap();
    }
}

#[test]
fn eager_self_backtrace_matches_frame_pointers() {
    ensure_init();
    let (fh_ips, fp_rets) = depth_a();
    assert!(
        fh_ips.len() >= 4,
        "expected several frames, got {}: {:#x?}",
        fh_ips.len(),
        fh_ips
    );
    #[cfg(target_arch = "x86_64")]
    assert_frames_match("eager", &fh_ips, &fp_rets, 3);
}

/// Explicit bounds are honored verbatim: a tiny window truncates the walk cleanly, and
/// the thread's real stack range walks as far as the default heuristic.
#[test]
fn cursor_init_bounds_clamps_reads() {
    ensure_init();

    let mut ctx = FhContext::zeroed();
    capture::fh_capture_context(&mut ctx);
    let sp = crate::arch::context_sp(&ctx);

    // Window covering almost nothing above sp: the walk must stop quickly and cleanly.
    let mut cur: FhCursor = unsafe { core::mem::zeroed() };
    let rc = cursor::cursor_init_bounds(&mut cur, &ctx, sp & !0x7, (sp & !0x7) + 16);
    assert_eq!(rc, 0, "cursor_init_bounds failed: {rc}");
    let (mut ip, mut spo) = (0u64, 0u64);
    let mut frames = 0;
    for _ in 0..64 {
        let more = cursor::step(&mut cur, &mut ip, &mut spo);
        if ip != 0 {
            frames += 1;
        }
        if more <= 0 {
            break;
        }
    }
    cursor::cursor_fini(&mut cur);
    assert!(
        frames <= 3,
        "tiny window should truncate the walk, got {frames} frames"
    );

    // The thread's real (registered) stack range, passed explicitly, unwinds normally.
    let (lo, hi) = crate::stackbounds::current();
    if lo != 0 && hi != 0 {
        let mut cur: FhCursor = unsafe { core::mem::zeroed() };
        let rc = cursor::cursor_init_bounds(&mut cur, &ctx, lo, hi);
        assert_eq!(rc, 0);
        let mut frames = 0;
        for _ in 0..64 {
            let more = cursor::step(&mut cur, &mut ip, &mut spo);
            if ip != 0 {
                frames += 1;
            }
            if more <= 0 {
                break;
            }
        }
        cursor::cursor_fini(&mut cur);
        assert!(
            frames >= 3,
            "explicit real bounds should unwind, got {frames} frames"
        );
    }
}

// ---------------------------------------------------------------------------
// Context-extraction layout tests: the embedder-facing entry points
// (fh_context_from_ucontext / fh_context_from_thread_state) are raw-offset reads of OS
// structs. Validate them against libc's struct definitions (the ground truth the kernel
// and libunwind share) with sentinel values, so silent layout drift fails the tests.
// ---------------------------------------------------------------------------

const SENT_IP: u64 = 0x1111_2222_3333_4444;
const SENT_SP: u64 = 0x5555_6666_7777_8888;
const SENT_FP: u64 = 0x9999_aaaa_bbbb_cccc;
#[cfg(any(target_os = "macos", all(target_os = "linux", target_arch = "aarch64")))]
const SENT_LR: u64 = 0xdddd_eeee_ffff_0123;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn darwin_thread_state_layout_matches_libc() {
    let mut ss: libc::__darwin_arm_thread_state64 = unsafe { core::mem::zeroed() };
    ss.__pc = SENT_IP;
    ss.__sp = SENT_SP;
    ss.__fp = SENT_FP;
    ss.__lr = SENT_LR;
    let mut ctx = FhContext::zeroed();
    capture::context_from_thread_state(&mut ctx, &ss as *const _ as *const core::ffi::c_void);
    assert_eq!(ctx.r[0], SENT_IP);
    assert_eq!(ctx.r[1], SENT_SP);
    assert_eq!(ctx.r[2], SENT_FP);
    assert_eq!(ctx.r[3], SENT_LR);
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
#[test]
fn darwin_thread_state_layout_matches_libc() {
    let mut ss: libc::__darwin_x86_thread_state64 = unsafe { core::mem::zeroed() };
    ss.__rip = SENT_IP;
    ss.__rsp = SENT_SP;
    ss.__rbp = SENT_FP;
    let mut ctx = FhContext::zeroed();
    capture::context_from_thread_state(&mut ctx, &ss as *const _ as *const core::ffi::c_void);
    assert_eq!(ctx.r[0], SENT_IP);
    assert_eq!(ctx.r[1], SENT_SP);
    assert_eq!(ctx.r[2], SENT_FP);
}

#[cfg(target_os = "macos")]
#[test]
fn darwin_ucontext_layout_matches_libc() {
    let mut mc: libc::__darwin_mcontext64 = unsafe { core::mem::zeroed() };
    #[cfg(target_arch = "aarch64")]
    {
        mc.__ss.__pc = SENT_IP;
        mc.__ss.__sp = SENT_SP;
        mc.__ss.__fp = SENT_FP;
        mc.__ss.__lr = SENT_LR;
    }
    #[cfg(target_arch = "x86_64")]
    {
        mc.__ss.__rip = SENT_IP;
        mc.__ss.__rsp = SENT_SP;
        mc.__ss.__rbp = SENT_FP;
    }
    let mut uc: libc::ucontext_t = unsafe { core::mem::zeroed() };
    uc.uc_mcontext = &mut mc;
    let mut ctx = FhContext::zeroed();
    capture::context_from_os(&mut ctx, &uc as *const _ as *const core::ffi::c_void);
    assert_eq!(ctx.r[0], SENT_IP);
    assert_eq!(ctx.r[1], SENT_SP);
    assert_eq!(ctx.r[2], SENT_FP);
    #[cfg(target_arch = "aarch64")]
    assert_eq!(ctx.r[3], SENT_LR);
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
fn linux_ucontext_layout_matches_libc() {
    let mut uc: libc::ucontext_t = unsafe { core::mem::zeroed() };
    uc.uc_mcontext.gregs[libc::REG_RIP as usize] = SENT_IP as i64;
    uc.uc_mcontext.gregs[libc::REG_RSP as usize] = SENT_SP as i64;
    uc.uc_mcontext.gregs[libc::REG_RBP as usize] = SENT_FP as i64;
    let mut ctx = FhContext::zeroed();
    capture::context_from_os(&mut ctx, &uc as *const _ as *const core::ffi::c_void);
    assert_eq!(ctx.r[0], SENT_IP);
    assert_eq!(ctx.r[1], SENT_SP);
    assert_eq!(ctx.r[2], SENT_FP);
}

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
#[test]
fn linux_ucontext_layout_matches_libc() {
    let mut uc: libc::ucontext_t = unsafe { core::mem::zeroed() };
    uc.uc_mcontext.pc = SENT_IP;
    uc.uc_mcontext.sp = SENT_SP;
    uc.uc_mcontext.regs[29] = SENT_FP;
    uc.uc_mcontext.regs[30] = SENT_LR;
    let mut ctx = FhContext::zeroed();
    capture::context_from_os(&mut ctx, &uc as *const _ as *const core::ffi::c_void);
    assert_eq!(ctx.r[0], SENT_IP);
    assert_eq!(ctx.r[1], SENT_SP);
    assert_eq!(ctx.r[2], SENT_FP);
    assert_eq!(ctx.r[3], SENT_LR);
}

// ---------------------------------------------------------------------------
// JIT-path validation: build a JIT-style framehop module from the test binary's own
// `.eh_frame` (base_svma == base_avma == bias, eh_frame_svma == runtime address) and
// confirm it unwinds the same return addresses as the eager path.
// ---------------------------------------------------------------------------

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod jit_path {
    use super::*;
    use framehop::{
        ExplicitModuleSectionInfo, FrameAddress, Module, MustNotAllocateDuringUnwind, Unwinder,
        UnwinderNative,
    };
    use std::ops::Range;

    type JitUnw = UnwinderNative<state::Bytes, MustNotAllocateDuringUnwind>;

    fn main_load_bias() -> Option<u64> {
        extern "C" fn cb(
            info: *mut libc::dl_phdr_info,
            _sz: libc::size_t,
            data: *mut libc::c_void,
        ) -> libc::c_int {
            unsafe { *(data as *mut u64) = (*info).dlpi_addr as u64 };
            1 // stop after the main program (first object)
        }
        let mut bias: u64 = u64::MAX;
        unsafe { libc::dl_iterate_phdr(Some(cb), &mut bias as *mut _ as *mut libc::c_void) };
        (bias != u64::MAX).then_some(bias)
    }

    /// Parse /proc/self/exe section headers to find (.eh_frame sh_addr, exact sh_size).
    fn eh_frame_section_from_exe() -> Option<(u64, usize)> {
        let data = std::fs::read("/proc/self/exe").ok()?;
        if data.len() < 64 || &data[0..4] != b"\x7fELF" || data[4] != 2 {
            return None;
        }
        let rd_u16 = |o: usize| -> Option<u16> {
            Some(u16::from_le_bytes(data.get(o..o + 2)?.try_into().ok()?))
        };
        let rd_u32 = |o: usize| -> Option<u32> {
            Some(u32::from_le_bytes(data.get(o..o + 4)?.try_into().ok()?))
        };
        let rd_u64 = |o: usize| -> Option<u64> {
            Some(u64::from_le_bytes(data.get(o..o + 8)?.try_into().ok()?))
        };

        let e_shoff = rd_u64(0x28)? as usize;
        let e_shentsize = rd_u16(0x3a)? as usize;
        let e_shnum = rd_u16(0x3c)? as usize;
        let e_shstrndx = rd_u16(0x3e)? as usize;
        if e_shoff == 0 || e_shentsize < 64 || e_shnum == 0 {
            return None;
        }
        let strtab_hdr = e_shoff + e_shstrndx * e_shentsize;
        let strtab_off = rd_u64(strtab_hdr + 0x18)? as usize;

        for i in 0..e_shnum {
            let sh = e_shoff + i * e_shentsize;
            let name_off = rd_u32(sh)? as usize;
            if read_cstr(&data, strtab_off + name_off) == ".eh_frame" {
                return Some((rd_u64(sh + 0x10)?, rd_u64(sh + 0x20)? as usize));
            }
        }
        None
    }

    fn read_cstr(data: &[u8], off: usize) -> &str {
        let mut end = off;
        while end < data.len() && data[end] != 0 {
            end += 1;
        }
        std::str::from_utf8(&data[off..end]).unwrap_or("")
    }

    fn build_jit_module(
        bias: u64,
        eh_frame_addr: u64,
        eh_frame_len: usize,
    ) -> Module<state::Bytes> {
        let eh_frame_copy: state::Bytes =
            unsafe { core::slice::from_raw_parts(eh_frame_addr as *const u8, eh_frame_len) }
                .to_vec()
                .into_boxed_slice();
        let info: ExplicitModuleSectionInfo<state::Bytes> = ExplicitModuleSectionInfo {
            base_svma: bias, // == base_avma == text_lo
            eh_frame: Some(eh_frame_copy),
            eh_frame_svma: Some(Range {
                start: eh_frame_addr,
                end: eh_frame_addr + eh_frame_len as u64,
            }),
            ..Default::default()
        };
        Module::new(
            "<jit-test>".to_string(),
            Range {
                start: bias,
                end: bias + (u32::MAX as u64),
            },
            bias,
            info,
        )
    }

    /// Capture + fp-walk in this frame, then unwind via the provided JIT unwinder.
    #[inline(never)]
    fn capture_and_walk_jit(unw: &JitUnw, max: usize) -> (Vec<u64>, Vec<u64>) {
        let mut ctx = FhContext::zeroed();
        capture::fh_capture_context(&mut ctx);

        let mut rbp: u64;
        unsafe { core::arch::asm!("mov {}, rbp", out(reg) rbp) };
        let fp = fp_chain(rbp, max);

        let mut cache = <state::Cache>::new_in();
        let mut regs = crate::arch::make_regs(&ctx, u64::MAX);
        let mut addr = FrameAddress::from_instruction_pointer(crate::arch::context_ip(&ctx));
        let sp0 = crate::arch::context_sp(&ctx);
        let lo = sp0 & !0xfff;
        let hi = lo + 64 * 1024 * 1024;
        let mut read = |a: u64| -> Result<u64, ()> {
            if a < lo || a + 8 > hi || (a & 0x7) != 0 {
                return Err(());
            }
            Ok(unsafe { core::ptr::read_volatile(a as *const u64) })
        };

        let mut ips = vec![crate::arch::context_ip(&ctx)];
        for _ in 0..max {
            match unw.unwind_frame(addr, &mut regs, &mut cache, &mut read) {
                Ok(Some(ra)) if ra != 0 => {
                    ips.push(ra);
                    addr = FrameAddress::ReturnAddress(core::num::NonZeroU64::new(ra).unwrap());
                }
                _ => break,
            }
        }
        black_box((ips, fp))
    }

    #[inline(never)]
    fn jit_b(unw: &JitUnw) -> (Vec<u64>, Vec<u64>) {
        black_box(capture_and_walk_jit(unw, 32))
    }
    #[inline(never)]
    fn jit_a(unw: &JitUnw) -> (Vec<u64>, Vec<u64>) {
        black_box(jit_b(unw))
    }

    #[test]
    fn jit_style_module_unwinds_real_code() {
        super::ensure_init();
        let bias = match main_load_bias() {
            Some(b) => b,
            None => return,
        };
        let (sh_addr, sh_size) = match eh_frame_section_from_exe() {
            Some(x) => x,
            None => {
                eprintln!("skipping: no .eh_frame section in /proc/self/exe");
                return;
            }
        };
        assert!(sh_size > 0);

        let module = build_jit_module(bias, bias + sh_addr, sh_size);
        let mut unw: JitUnw = UnwinderNative::new();
        unw.add_module(module);

        let (jit_ips, fp) = jit_a(&unw);
        assert!(
            jit_ips.len() >= 4,
            "JIT-path unwind too short ({}): {jit_ips:#x?}",
            jit_ips.len()
        );
        assert_frames_match("jit", &jit_ips, &fp, 3);
    }

    #[test]
    fn jit_auto_derives_code_range() {
        super::ensure_init();
        let bias = match main_load_bias() {
            Some(b) => b,
            None => return,
        };
        let (sh_addr, sh_size) = match eh_frame_section_from_exe() {
            Some(x) => x,
            None => return,
        };
        // fh_register_jit_auto must parse the FDEs and derive a valid range => returns 0.
        let eh_ptr = (bias + sh_addr) as *const u8;
        let rc = crate::modules::register_jit_eh_frame_auto(eh_ptr, sh_size);
        assert_eq!(rc, 0, "auto registration failed: {rc}");
        // Clean up (keyed by eh_frame address).
        crate::modules::deregister_jit_eh_frame(eh_ptr);
    }
}
