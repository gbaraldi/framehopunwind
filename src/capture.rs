//! Capturing register state: either from the *current* thread (a naked-asm routine that
//! perturbs no registers, replacing `unw_getcontext`) or from an OS-provided
//! `ucontext_t` / `CONTEXT` delivered to a signal handler / sampler.

use crate::arch::FhContext;

// ---------------------------------------------------------------------------
// (A) Current-thread capture. Naked so the prologue does not disturb sp/bp.
//
// We capture the *caller's* frame: ip = the return address on entry, sp = the
// caller's sp (entry sp + return-slot), bp/fp = the caller's frame pointer. The
// caller then typically skips its own frame anyway.
// ---------------------------------------------------------------------------

cfg_if::cfg_if! {
    if #[cfg(all(target_arch = "x86_64", not(target_os = "windows")))] {
        /// Capture the current (caller's) context into `*ctx`. SysV: arg in `rdi`.
        ///
        /// Writes ALL five context words (zeroing the unused ones): C callers pass a
        /// stack-allocated `fh_context`, and Rust later forms references to it, so no
        /// word may be left uninitialized.
        #[no_mangle]
        #[unsafe(naked)]
        pub extern "C" fn fh_capture_context(ctx: *mut FhContext) {
            core::arch::naked_asm!(
                "mov rax, [rsp]",     // return address -> caller ip
                "mov [rdi], rax",
                "lea rax, [rsp + 8]", // caller's sp (after the call returns)
                "mov [rdi + 8], rax",
                "mov [rdi + 16], rbp",// caller's frame pointer
                "mov qword ptr [rdi + 24], 0", // r[3] unused on x86_64
                "mov qword ptr [rdi + 32], 0", // r[4] reserved
                "ret",
            )
        }
    } else if #[cfg(all(target_arch = "x86_64", target_os = "windows"))] {
        /// Win64: first arg in `rcx`. Writes all five words (see the SysV variant).
        #[no_mangle]
        #[unsafe(naked)]
        pub extern "C" fn fh_capture_context(ctx: *mut FhContext) {
            core::arch::naked_asm!(
                "mov rax, [rsp]",
                "mov [rcx], rax",
                "lea rax, [rsp + 8]",
                "mov [rcx + 8], rax",
                "mov [rcx + 16], rbp",
                "mov qword ptr [rcx + 24], 0",
                "mov qword ptr [rcx + 32], 0",
                "ret",
            )
        }
    } else if #[cfg(target_arch = "aarch64")] {
        /// AAPCS64: arg in `x0`. r[0]=pc, r[1]=sp, r[2]=fp(x29), r[3]=lr(x30).
        /// Writes all five words (see the SysV variant).
        #[no_mangle]
        #[unsafe(naked)]
        pub extern "C" fn fh_capture_context(ctx: *mut FhContext) {
            core::arch::naked_asm!(
                "str x30, [x0]",       // pc = return address
                "mov x1, sp",
                "str x1, [x0, #8]",    // sp
                "str x29, [x0, #16]",  // fp
                "str x30, [x0, #24]",  // lr (== return address; unused for first frame)
                "str xzr, [x0, #32]",  // r[4] reserved
                "ret",
            )
        }
    } else {
        /// Unsupported arch: zero the context.
        #[no_mangle]
        pub extern "C" fn fh_capture_context(ctx: *mut FhContext) {
            if !ctx.is_null() {
                unsafe { *ctx = FhContext::zeroed() };
            }
        }
    }
}

// ---------------------------------------------------------------------------
// (B) Extraction from an OS context structure (signal handler / sampler).
// ---------------------------------------------------------------------------

/// Fill `*ctx` from an OS context pointer:
///   * Unix: a `ucontext_t *` (what a signal handler receives as its 3rd argument).
///   * Windows: a `CONTEXT *`.
///
/// Safe to call with a null `os_ctx` (yields a zeroed context).
pub fn context_from_os(ctx: &mut FhContext, os_ctx: *const core::ffi::c_void) {
    *ctx = FhContext::zeroed();
    if os_ctx.is_null() {
        return;
    }
    // SAFETY: caller guarantees `os_ctx` points at the platform context type.
    unsafe { fill_from_os(ctx, os_ctx) }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
unsafe fn fill_from_os(ctx: &mut FhContext, os_ctx: *const core::ffi::c_void) {
    let uc = &*(os_ctx as *const libc::ucontext_t);
    let g = &uc.uc_mcontext.gregs;
    ctx.r[0] = g[libc::REG_RIP as usize] as u64;
    ctx.r[1] = g[libc::REG_RSP as usize] as u64;
    ctx.r[2] = g[libc::REG_RBP as usize] as u64;
}

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
unsafe fn fill_from_os(ctx: &mut FhContext, os_ctx: *const core::ffi::c_void) {
    let uc = &*(os_ctx as *const libc::ucontext_t);
    let mc = &uc.uc_mcontext;
    ctx.r[0] = mc.pc;
    ctx.r[1] = mc.sp;
    ctx.r[2] = mc.regs[29]; // fp
    ctx.r[3] = mc.regs[30]; // lr
}

#[cfg(all(target_os = "freebsd", target_arch = "x86_64"))]
unsafe fn fill_from_os(ctx: &mut FhContext, os_ctx: *const core::ffi::c_void) {
    let uc = &*(os_ctx as *const libc::ucontext_t);
    let mc = &uc.uc_mcontext;
    ctx.r[0] = mc.mc_rip as u64;
    ctx.r[1] = mc.mc_rsp as u64;
    ctx.r[2] = mc.mc_rbp as u64;
}

#[cfg(all(target_os = "freebsd", target_arch = "aarch64"))]
unsafe fn fill_from_os(ctx: &mut FhContext, os_ctx: *const core::ffi::c_void) {
    let uc = &*(os_ctx as *const libc::ucontext_t);
    let gp = &uc.uc_mcontext.mc_gpregs;
    ctx.r[0] = gp.gp_elr as u64; // pc
    ctx.r[1] = gp.gp_sp as u64;
    ctx.r[2] = gp.gp_x[29] as u64; // fp
    ctx.r[3] = gp.gp_lr as u64; // lr
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
unsafe fn fill_from_os(ctx: &mut FhContext, os_ctx: *const core::ffi::c_void) {
    let uc = &*(os_ctx as *const libc::ucontext_t);
    let mc = &*uc.uc_mcontext; // uc_mcontext is a pointer on Darwin
    ctx.r[0] = mc.__ss.__rip;
    ctx.r[1] = mc.__ss.__rsp;
    ctx.r[2] = mc.__ss.__rbp;
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn fill_from_os(ctx: &mut FhContext, os_ctx: *const core::ffi::c_void) {
    let uc = &*(os_ctx as *const libc::ucontext_t);
    let mc = &*uc.uc_mcontext;
    ctx.r[0] = mc.__ss.__pc;
    ctx.r[1] = mc.__ss.__sp;
    ctx.r[2] = mc.__ss.__fp;
    ctx.r[3] = mc.__ss.__lr;
}

#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
unsafe fn fill_from_os(ctx: &mut FhContext, os_ctx: *const core::ffi::c_void) {
    let c = &*(os_ctx as *const windows_sys::Win32::System::Diagnostics::Debug::CONTEXT);
    ctx.r[0] = c.Rip;
    ctx.r[1] = c.Rsp;
    ctx.r[2] = c.Rbp;
}

/// Fill `*ctx` from a Darwin **mach thread state** — `__darwin_x86_thread_state64` or
/// `__darwin_arm_thread_state64`. This is what Julia's `bt_context_t` actually holds on
/// macOS (it casts the mach thread state to `unw_context_t`; both the profiler's
/// `thread_get_state` and `unw_getcontext` land in this layout). Reading a `ucontext_t`
/// there would be wrong, hence this separate entry point. Async-signal-safe.
pub fn context_from_thread_state(ctx: &mut FhContext, ts: *const core::ffi::c_void) {
    *ctx = FhContext::zeroed();
    if ts.is_null() {
        return;
    }
    // SAFETY: caller guarantees `ts` points at the platform thread-state struct.
    unsafe { fill_from_thread_state(ctx, ts) }
}

#[cfg(target_arch = "x86_64")]
unsafe fn fill_from_thread_state(ctx: &mut FhContext, ts: *const core::ffi::c_void) {
    // __darwin_x86_thread_state64: rax,rbx,rcx,rdx,rdi,rsi,rbp,rsp,r8..r15,rip,... (u64 each)
    let p = ts as *const u64;
    ctx.r[0] = core::ptr::read_unaligned(p.add(16)); // rip
    ctx.r[1] = core::ptr::read_unaligned(p.add(7)); // rsp
    ctx.r[2] = core::ptr::read_unaligned(p.add(6)); // rbp
}

#[cfg(target_arch = "aarch64")]
unsafe fn fill_from_thread_state(ctx: &mut FhContext, ts: *const core::ffi::c_void) {
    // __darwin_arm_thread_state64: x[0..28] (29), fp(x29), lr(x30), sp, pc, cpsr (u64 each)
    let p = ts as *const u64;
    ctx.r[0] = core::ptr::read_unaligned(p.add(32)); // pc
    ctx.r[1] = core::ptr::read_unaligned(p.add(31)); // sp
    ctx.r[2] = core::ptr::read_unaligned(p.add(29)); // fp (x29)
    ctx.r[3] = core::ptr::read_unaligned(p.add(30)); // lr (x30)
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
unsafe fn fill_from_thread_state(_ctx: &mut FhContext, _ts: *const core::ffi::c_void) {}

// Any (os, arch) we don't special-case: leave the context zeroed.
#[cfg(not(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "linux", target_arch = "aarch64"),
    all(target_os = "freebsd", target_arch = "x86_64"),
    all(target_os = "freebsd", target_arch = "aarch64"),
    all(target_os = "macos", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64"),
    all(target_os = "windows", target_arch = "x86_64"),
)))]
unsafe fn fill_from_os(_ctx: &mut FhContext, _os_ctx: *const core::ffi::c_void) {}
