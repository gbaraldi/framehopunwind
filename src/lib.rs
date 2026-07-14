//! framehopunwind — a clean C unwinding API backed by the `framehop` stack unwinder,
//! intended to replace the platform unwinder (libunwind on Linux/FreeBSD/macOS, dbghelp
//! on Windows) used by Julia for backtraces and profiling.
//!
//! It exposes:
//!
//!   * a cursor-based unwinding API (capture context, init cursor, step), mirroring the
//!     subset of libunwind that Julia's `stackwalk.c` uses, and matching `jl_unw_step`'s
//!     output-then-advance contract;
//!   * a JIT registration API (register/deregister `.eh_frame` unwind info for
//!     runtime-emitted code), replacing the `_U_dyn_register` / `RtlAddFunctionTable`
//!     path in `debuginfo.cpp`;
//!   * eager enumeration of loaded modules so the actual unwind step does no allocation
//!     and is async-signal-safe.
//!
//! ## Async-signal-safety
//!
//! The **read path** — [`fh_capture_context`], [`fh_context_from_ucontext`],
//! [`fh_cursor_init`], [`fh_step`], [`fh_get_reg`], [`fh_cursor_fini`] — is
//! async-signal-safe: no allocation, no locks, no `free`. The **mutating path**
//! ([`fh_init`], [`fh_register_jit`], [`fh_modules_refresh`], …) is not, and is meant to
//! run off the signal path (JIT compile time, dlopen, startup), exactly where Julia
//! already serializes with `jl_profile_atomic`.
//!
//! ## Supported targets
//!
//! x86_64 on Linux/FreeBSD/macOS/Windows, and aarch64 on Linux/FreeBSD/macOS. Other
//! targets (32-bit, Windows-aarch64, ppc64le, …) are not handled by framehop; callers
//! gate on [`fh_supported`] / `FRAMEHOP_SUPPORTED` and keep the existing unwinder.

use core::ffi::{c_int, c_void};

// Core machinery only exists for the arches framehop supports.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod arch;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod capture;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod cursor;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod modules;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod stackbounds;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod state;

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub use cursor::FhCursor;

#[cfg(all(test, any(target_arch = "x86_64", target_arch = "aarch64")))]
mod tests;

/// Whether framehop can unwind on this exact (os, arch) combination. Note Windows-aarch64
/// is excluded: framehop's PE backend is x86_64-only.
pub const FRAMEHOP_SUPPORTED: bool = cfg!(any(
    all(
        target_arch = "x86_64",
        any(
            target_os = "linux",
            target_os = "freebsd",
            target_os = "macos",
            target_os = "windows"
        )
    ),
    all(
        target_arch = "aarch64",
        any(
            target_os = "linux",
            target_os = "freebsd",
            target_os = "macos"
        )
    )
));

/// 1 if this build can unwind natively, 0 otherwise.
#[no_mangle]
pub extern "C" fn fh_supported() -> c_int {
    FRAMEHOP_SUPPORTED as c_int
}

// ===========================================================================
// Supported-arch API. Compiles for x86_64/aarch64 on any OS; on an OS without a module
// enumerator the module list is simply empty (callers should gate on `fh_supported()`).
// ===========================================================================
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod api {
    use super::*;
    use crate::arch::FhContext;
    use crate::cursor::FhCursor;

    /// Initialize the library: allocate the unwinding slot pool and enumerate the
    /// currently-loaded modules (on macOS, by installing dyld's add/remove-image
    /// callbacks, which keep the set current from then on). Idempotent. `num_slots == 0`
    /// selects a default (256). Returns 0 on success. **Not** async-signal-safe; call
    /// once at startup.
    #[no_mangle]
    pub extern "C" fn fh_init(num_slots: usize) -> c_int {
        install_panic_hook();
        crate::state::init(num_slots);
        crate::modules::init();
        0
    }

    /// In the shipped artifact (`panic = "abort"`) std would still run the default panic
    /// hook — which formats, locks stderr, and may allocate — before aborting. A panic
    /// reachable from the read path would therefore hang in malloc inside a signal
    /// handler instead of dying. The read path is written to be panic-free, but if a bug
    /// ever introduces one, degrade to a raw `write(2)` + abort instead.
    #[cfg(all(panic = "abort", unix))]
    fn install_panic_hook() {
        std::panic::set_hook(Box::new(|_| {
            let msg = b"framehopunwind: internal panic; aborting\n";
            // SAFETY: plain write(2) to stderr; async-signal-safe.
            unsafe { libc::write(2, msg.as_ptr().cast(), msg.len()) };
        }));
    }
    #[cfg(not(all(panic = "abort", unix)))]
    fn install_panic_hook() {}

    /// Capture per-thread state needed off the signal path: preallocated stack bounds.
    /// Call once per thread that will be unwound (profiled), before any sampling.
    #[no_mangle]
    pub extern "C" fn fh_thread_register() {
        crate::stackbounds::register_current_thread();
    }

    /// Re-scan loaded modules (call after `dlopen`/`dlclose`). Not async-signal-safe.
    #[no_mangle]
    pub extern "C" fn fh_modules_refresh() {
        crate::modules::refresh();
    }

    /// Diagnostic: number of JIT modules currently registered with framehop.
    #[no_mangle]
    pub extern "C" fn fh_jit_module_count() -> usize {
        crate::modules::jit_module_count()
    }

    /// Diagnostic: cumulative number of JIT registrations that failed (e.g. unparsable
    /// `.eh_frame`). A nonzero value means some JIT code fell back to frame pointers.
    #[no_mangle]
    pub extern "C" fn fh_jit_register_failures() -> usize {
        crate::modules::jit_register_failures()
    }

    /// Fill `*ctx` from an OS context: a `ucontext_t*` (Unix signal handler 3rd arg) or a
    /// `CONTEXT*` (Windows). Async-signal-safe.
    #[no_mangle]
    pub extern "C" fn fh_context_from_ucontext(ctx: *mut FhContext, os_ctx: *const c_void) {
        if ctx.is_null() {
            return;
        }
        // SAFETY: ctx is valid caller-allocated storage; fully initialize it through a
        // raw write BEFORE forming a Rust reference (the caller's struct may be uninit).
        unsafe { ctx.write(FhContext::zeroed()) };
        let ctx = unsafe { &mut *ctx };
        crate::capture::context_from_os(ctx, os_ctx);
    }

    /// Fill `*ctx` from a Darwin mach thread state (`__darwin_{x86,arm}_thread_state64`),
    /// which is what Julia's `bt_context_t` holds on macOS. Async-signal-safe.
    #[no_mangle]
    pub extern "C" fn fh_context_from_thread_state(ctx: *mut FhContext, ts: *const c_void) {
        if ctx.is_null() {
            return;
        }
        // SAFETY: see fh_context_from_ucontext — raw write before forming a reference.
        unsafe { ctx.write(FhContext::zeroed()) };
        let ctx = unsafe { &mut *ctx };
        crate::capture::context_from_thread_state(ctx, ts);
    }

    /// Initialize `cur` to unwind from `ctx`. Returns 0 on success, `<0` on failure (no
    /// modules published, or the slot pool is exhausted). Async-signal-safe.
    /// (`cur` may be uninitialized storage; the cursor layer only writes it.)
    #[no_mangle]
    pub extern "C" fn fh_cursor_init(cur: *mut FhCursor, ctx: *const FhContext) -> c_int {
        if cur.is_null() || ctx.is_null() {
            return -100;
        }
        crate::cursor::cursor_init(cur, ctx)
    }

    /// Like `fh_cursor_init`, with an explicit stack-read window `[stack_lo, stack_hi)`.
    /// Pass the *target* thread's (or task's) exact stack range when unwinding a context
    /// captured from another thread (a suspended-target sampler) or from an
    /// embedder-managed stack — the per-thread registered bounds cannot cover those — to
    /// make the stack reads fault-free. `0, 0` falls back to the registered-bounds /
    /// sp-window heuristic. Async-signal-safe.
    #[no_mangle]
    pub extern "C" fn fh_cursor_init_bounds(
        cur: *mut FhCursor,
        ctx: *const FhContext,
        stack_lo: u64,
        stack_hi: u64,
    ) -> c_int {
        if cur.is_null() || ctx.is_null() {
            return -100;
        }
        crate::cursor::cursor_init_bounds(cur, ctx, stack_lo, stack_hi)
    }

    /// Output the current frame's ip/sp into `*ip`/`*sp`, then advance one frame.
    /// Returns `>0` if more frames may follow, `0` at a clean end, `<0` on error.
    /// Async-signal-safe. (Outputs are written through raw pointers, so `ip == sp`
    /// aliasing is tolerated.)
    #[no_mangle]
    pub extern "C" fn fh_step(cur: *mut FhCursor, ip: *mut u64, sp: *mut u64) -> c_int {
        if cur.is_null() || ip.is_null() || sp.is_null() {
            return -100;
        }
        crate::cursor::step(cur, ip, sp)
    }

    /// Read the current ip/sp without advancing. Async-signal-safe.
    #[no_mangle]
    pub extern "C" fn fh_get_reg(cur: *const FhCursor, ip: *mut u64, sp: *mut u64) {
        if cur.is_null() || ip.is_null() || sp.is_null() {
            return;
        }
        crate::cursor::get_reg(cur, ip, sp);
    }

    /// Release the cursor's slot. Idempotent; async-signal-safe. Julia must call this when
    /// it stops stepping (the cursor owns a pooled slot until then).
    #[no_mangle]
    pub extern "C" fn fh_cursor_fini(cur: *mut FhCursor) {
        if cur.is_null() {
            return;
        }
        crate::cursor::cursor_fini(cur);
    }

    /// Register a JIT module from a live `.eh_frame` buffer covering code in
    /// `[text_lo, text_hi)`. The bytes are copied, so the caller may free its buffer after
    /// deregistration. Returns 0 on success, `<0` on bad arguments. Not async-signal-safe.
    #[no_mangle]
    pub extern "C" fn fh_register_jit(
        eh_frame: *const u8,
        eh_frame_len: usize,
        text_lo: u64,
        text_hi: u64,
    ) -> c_int {
        crate::modules::register_jit_eh_frame(eh_frame, eh_frame_len, text_lo, text_hi)
    }

    /// Like `fh_register_jit`, but derive the covered code range from the `.eh_frame`'s
    /// FDEs (the caller need not compute text_lo/text_hi). Returns 0 on success, <0 on
    /// error. Not async-signal-safe.
    #[no_mangle]
    pub extern "C" fn fh_register_jit_auto(eh_frame: *const u8, eh_frame_len: usize) -> c_int {
        crate::modules::register_jit_eh_frame_auto(eh_frame, eh_frame_len)
    }

    /// Deregister a JIT module previously registered with `text_lo`. Not async-signal-safe.
    #[no_mangle]
    pub extern "C" fn fh_deregister_jit(text_lo: u64) {
        crate::modules::deregister_jit(text_lo);
    }

    /// Deregister a JIT module by the `.eh_frame` pointer used at registration. Convenient
    /// for `deregister_eh_frames(Addr, Size)`, which only retains `Addr`. Not signal-safe.
    #[no_mangle]
    pub extern "C" fn fh_deregister_jit_eh_frame(eh_frame: *const u8) {
        crate::modules::deregister_jit_eh_frame(eh_frame);
    }
}

// ===========================================================================
// Fallback stubs for unsupported arches (no framehop backend), so Julia can always link
// the library and gate at the call site on `fh_supported()`. These return "unsupported".
// ===========================================================================
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
mod api_stub {
    use super::*;

    #[no_mangle]
    pub extern "C" fn fh_capture_context(_ctx: *mut c_void) {}
    #[no_mangle]
    pub extern "C" fn fh_init(_num_slots: usize) -> c_int {
        -1
    }
    #[no_mangle]
    pub extern "C" fn fh_thread_register() {}
    #[no_mangle]
    pub extern "C" fn fh_modules_refresh() {}
    #[no_mangle]
    pub extern "C" fn fh_jit_module_count() -> usize {
        0
    }
    #[no_mangle]
    pub extern "C" fn fh_jit_register_failures() -> usize {
        0
    }
    #[no_mangle]
    pub extern "C" fn fh_context_from_ucontext(_ctx: *mut c_void, _os_ctx: *const c_void) {}
    #[no_mangle]
    pub extern "C" fn fh_context_from_thread_state(_ctx: *mut c_void, _ts: *const c_void) {}
    #[no_mangle]
    pub extern "C" fn fh_cursor_init(_cur: *mut c_void, _ctx: *const c_void) -> c_int {
        -1
    }
    #[no_mangle]
    pub extern "C" fn fh_cursor_init_bounds(
        _cur: *mut c_void,
        _ctx: *const c_void,
        _stack_lo: u64,
        _stack_hi: u64,
    ) -> c_int {
        -1
    }
    #[no_mangle]
    pub extern "C" fn fh_step(_cur: *mut c_void, _ip: *mut u64, _sp: *mut u64) -> c_int {
        -1
    }
    #[no_mangle]
    pub extern "C" fn fh_get_reg(_cur: *const c_void, _ip: *mut u64, _sp: *mut u64) {}
    #[no_mangle]
    pub extern "C" fn fh_cursor_fini(_cur: *mut c_void) {}
    #[no_mangle]
    pub extern "C" fn fh_register_jit(
        _eh_frame: *const u8,
        _eh_frame_len: usize,
        _text_lo: u64,
        _text_hi: u64,
    ) -> c_int {
        -1
    }
    #[no_mangle]
    pub extern "C" fn fh_register_jit_auto(_eh_frame: *const u8, _eh_frame_len: usize) -> c_int {
        -1
    }
    #[no_mangle]
    pub extern "C" fn fh_deregister_jit(_text_lo: u64) {}
    #[no_mangle]
    pub extern "C" fn fh_deregister_jit_eh_frame(_eh_frame: *const u8) {}
}
