/* framehopunwind.h — C API for the framehop-backed unwinder.
 *
 * Replaces the subset of libunwind / dbghelp that Julia uses for backtraces and
 * profiling, plus JIT unwind-info registration.
 *
 * Two classes of entry point:
 *
 *   * Read path (async-signal-safe): fh_capture_context, fh_context_from_ucontext,
 *     fh_cursor_init, fh_step, fh_get_reg, fh_cursor_fini. Safe to call from a signal
 *     handler / profiler sampler. No allocation, no locks, no free. NOTE: fh_step reads
 *     target stack memory and, unless exact thread stack bounds were registered via
 *     fh_thread_register, the read is fault-*bounded* not fault-free; the embedder MUST
 *     install a SIGSEGV handler (Julia uses jl_set_safe_restore) to survive a bad read.
 *
 *   * Mutating path (NOT async-signal-safe): fh_init, fh_thread_register,
 *     fh_modules_refresh, fh_register_jit, fh_deregister_jit. Call off the signal path
 *     (startup, dlopen, JIT compile), where Julia already serializes with
 *     jl_profile_atomic.
 *
 * Mapping to libunwind:
 *   unw_getcontext / RtlCaptureContext      -> fh_capture_context
 *   (signal ucontext_t / CONTEXT)           -> fh_context_from_ucontext
 *   unw_init_local / unw_init_local2        -> fh_cursor_init
 *   unw_get_reg(IP/SP) + unw_step           -> fh_step  (output-then-advance)
 *   _U_dyn_register / RtlAddFunctionTable   -> fh_register_jit
 *   _U_dyn_cancel   / RtlDeleteFunctionTable-> fh_deregister_jit
 */
#ifndef FRAMEHOPUNWIND_H
#define FRAMEHOPUNWIND_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Caller-allocated register snapshot. Layout is private; only the size matters.
 *   x86_64 : r[0]=rip, r[1]=rsp, r[2]=rbp
 *   aarch64: r[0]=pc,  r[1]=sp,  r[2]=fp(x29), r[3]=lr(x30)
 */
typedef struct fh_context {
    uint64_t r[5];
} fh_context;

/* Caller-allocated, opaque unwind cursor. Treat as a 64-byte blob. */
typedef struct fh_cursor {
    uint64_t _opaque[8];
} fh_cursor;

/* 1 if this build can unwind natively on the current (os, arch); 0 otherwise.
 * When 0, all other functions are no-op stubs returning failure — the caller should
 * keep using the existing unwinder (libunwind / dbghelp). */
int fh_supported(void);

/* ---- lifecycle (NOT async-signal-safe) ---- */

/* Allocate the unwinding slot pool (num_slots; 0 => default 256) and enumerate the
 * currently-loaded modules. Idempotent. Returns 0 on success. Call once at startup. */
int fh_init(size_t num_slots);

/* Capture per-thread state needed off the signal path (stack bounds). Call once per
 * thread that will be unwound/profiled, before any sampling. */
void fh_thread_register(void);

/* Re-scan loaded modules (call after dlopen/dlclose; JIT modules are preserved). */
void fh_modules_refresh(void);

/* ---- context capture (async-signal-safe) ---- */

/* Capture the CURRENT thread's context into *ctx (replaces unw_getcontext). */
void fh_capture_context(fh_context *ctx);

/* Fill *ctx from an OS context: a ucontext_t* (Unix signal handler 3rd arg) or a
 * CONTEXT* (Windows). Replaces the jl_to_bt_context / signal-frame init path.
 * Precondition: os_ctx must be NULL (yields a zeroed context) or a valid pointer to the
 * platform context type; any other non-NULL value is undefined behavior. */
void fh_context_from_ucontext(fh_context *ctx, const void *os_ctx);

/* ---- cursor (async-signal-safe read path) ---- */

/* Initialize *cur to unwind from *ctx. Returns 0 on success, <0 on failure
 * (no modules published yet, or the slot pool is exhausted). */
int fh_cursor_init(fh_cursor *cur, const fh_context *ctx);

/* Output the CURRENT frame's ip/sp into *ip/*sp, then advance one frame.
 * Returns:  >0  a frame was produced and more may follow;
 *            0  reached the end of the stack (clean);
 *           <0  error. Mirrors Julia's jl_unw_step contract exactly. */
int fh_step(fh_cursor *cur, uint64_t *ip, uint64_t *sp);

/* Read the current ip/sp without advancing (mirrors jl_unw_get on a live cursor). */
void fh_get_reg(const fh_cursor *cur, uint64_t *ip, uint64_t *sp);

/* Release the cursor's pooled slot. Idempotent. MUST be called when stepping stops
 * (the cursor holds a pooled cache+slot until then). */
void fh_cursor_fini(fh_cursor *cur);

/* ---- JIT registration (NOT async-signal-safe) ---- */

/* Register a JIT module from a live .eh_frame buffer (DWARF platforms: Linux/FreeBSD/
 * macOS) covering code in [text_lo, text_hi). The bytes are copied, so the caller may
 * free its buffer after deregistration. Returns 0 on success, <0 on bad arguments. */
int fh_register_jit(const uint8_t *eh_frame, size_t eh_frame_len,
                    uint64_t text_lo, uint64_t text_hi);

/* Like fh_register_jit, but derive [text_lo, text_hi) from the .eh_frame's FDEs.
 * Convenient for register_eh_frames(Addr, Size), which has only the eh_frame buffer. */
int fh_register_jit_auto(const uint8_t *eh_frame, size_t eh_frame_len);

/* Deregister a JIT module previously registered with the same text_lo. */
void fh_deregister_jit(uint64_t text_lo);

/* Deregister a JIT module by the .eh_frame pointer used at registration. Convenient for
 * deregister_eh_frames(Addr, Size), which only retains Addr. */
void fh_deregister_jit_eh_frame(const uint8_t *eh_frame);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* FRAMEHOPUNWIND_H */
