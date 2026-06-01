//! Per-thread stack bounds, used to keep the stack-memory reader from dereferencing
//! wild pointers during unwinding.
//!
//! Captured **off** the signal path (in `fh_thread_register`) via the platform's pthread
//! stack-introspection APIs, and stored in a thread-local that the signal-path reader
//! consults. If a thread was never registered, the reader falls back to a window derived
//! from the starting stack pointer (see `cursor`).

use core::cell::Cell;

thread_local! {
    /// `(lo, hi)` half-open range of this thread's usable stack, or `(0, 0)` if unknown.
    static BOUNDS: Cell<(u64, u64)> = const { Cell::new((0, 0)) };
}

/// Return the registered `(lo, hi)` bounds for the current thread, or `(0, 0)`.
/// Reading an already-initialized thread-local is async-signal-safe.
#[inline]
pub fn current() -> (u64, u64) {
    BOUNDS.with(|b| b.get())
}

/// Capture and cache the current thread's stack bounds. Off the signal path.
pub fn register_current_thread() {
    let (lo, hi) = unsafe { query_bounds() };
    BOUNDS.with(|b| b.set((lo, hi)));
}

/// Query the OS for the current thread's usable stack range. Returns `(0, 0)` if it
/// cannot be determined.
///
/// # Safety
/// Calls into pthread; must run on a normal (non-signal) context.
#[cfg(all(unix, not(target_os = "macos")))]
unsafe fn query_bounds() -> (u64, u64) {
    // Linux / FreeBSD: pthread_getattr_np (Linux) or pthread_attr_get_np (FreeBSD),
    // then pthread_attr_getstack -> (lowest address, size).
    let mut attr: libc::pthread_attr_t = core::mem::zeroed();
    let self_thread = libc::pthread_self();

    #[cfg(target_os = "linux")]
    let got = libc::pthread_getattr_np(self_thread, &mut attr) == 0;
    #[cfg(target_os = "freebsd")]
    let got = {
        // pthread_attr_get_np needs an initialized attr.
        libc::pthread_attr_init(&mut attr);
        libc::pthread_attr_get_np(self_thread, &mut attr) == 0
    };
    #[cfg(not(any(target_os = "linux", target_os = "freebsd")))]
    let got = false;

    if !got {
        return (0, 0);
    }

    let mut stackaddr: *mut libc::c_void = core::ptr::null_mut();
    let mut stacksize: libc::size_t = 0;
    let rc = libc::pthread_attr_getstack(&attr, &mut stackaddr, &mut stacksize);
    libc::pthread_attr_destroy(&mut attr);
    if rc != 0 || stackaddr.is_null() || stacksize == 0 {
        return (0, 0);
    }
    let lo = stackaddr as u64;
    let hi = lo.saturating_add(stacksize as u64);
    (lo, hi)
}

#[cfg(target_os = "macos")]
unsafe fn query_bounds() -> (u64, u64) {
    // Darwin: pthread_get_stackaddr_np returns the *base* (high end) of the stack,
    // pthread_get_stacksize_np its size; the usable range grows downward from the base.
    let self_thread = libc::pthread_self();
    let base = libc::pthread_get_stackaddr_np(self_thread) as u64;
    let size = libc::pthread_get_stacksize_np(self_thread) as u64;
    if base == 0 || size == 0 {
        return (0, 0);
    }
    let lo = base.saturating_sub(size);
    (lo, base)
}

#[cfg(not(unix))]
unsafe fn query_bounds() -> (u64, u64) {
    // Windows: stack bounds for the *current* thread can be read from the TIB, but the
    // sampler unwinds a *different* (suspended) thread, so per-thread bounds captured here
    // would be wrong. The Windows reader uses a different guard (see `cursor`). Return
    // unknown here.
    (0, 0)
}
