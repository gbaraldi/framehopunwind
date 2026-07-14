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
 * ---------------------------------------------------------------------------
 * Fault & cursor-slot lifetime contract (IMPORTANT — read before consuming)
 * ---------------------------------------------------------------------------
 * fh_step reads the target's live stack. The read is bounds-checked, but it is
 * fault-*bounded*, not fault-free: with only the sp-derived fallback window the window
 * can include unmapped pages, so a corrupt stack / bad unwind info can still make the
 * read hit a SIGSEGV. To make reads fault-free, give the walk exact mapped bounds:
 *   - same-thread walks on the thread's own pthread stack: call fh_thread_register on
 *     that thread beforehand (records its pthread stack bounds in TLS);
 *   - cross-thread walks (a sampler unwinding a suspended target) or walks over an
 *     embedder-managed stack (e.g. a Julia task stack): the registered bounds CANNOT
 *     apply — pass the target's stack range to fh_cursor_init_bounds instead.
 * With exact bounds, out-of-stack addresses become a clean end-of-stack instead of a
 * fault. This crate does NOT install its own SIGSEGV handler (it would conflict with the
 * embedder's, e.g. Julia's).
 *
 * If the embedder instead survives a faulting read by longjmp/siglongjmp-ing out of
 * fh_step from its SIGSEGV handler (Julia's jl_set_safe_restore), be aware the jump
 * crosses this library's (Rust) frames. That is defined behavior only while those frames
 * hold no pending destructors; the read path is written allocation- and destructor-free
 * to keep that true, but exact bounds — which avoid the fault entirely — are the
 * *supported* configuration, and the longjmp path is best-effort.
 *
 * fh_thread_register also pre-faults this library's thread-local storage. Any thread
 * that will RUN the unwinder (including sampler/listener threads that only ever unwind
 * *other* threads) must call it once off the signal/suspend path: the first TLS access
 * from a thread can otherwise allocate (dyld lazy TLV on macOS, __tls_get_addr on ELF)
 * at exactly the wrong moment.
 *
 * A cursor owns a pooled slot (a preallocated unwind cache) from fh_cursor_init until
 * fh_cursor_fini. The pool is finite (fh_init's num_slots); if it is exhausted
 * fh_cursor_init returns <0 (skip that sample) — it never blocks or allocates.
 *
 * A cursor is single-owner: do not copy the struct, share it across threads, or step it
 * concurrently. (Defense-in-depth: each cursor carries a per-claim nonce, so fini on a
 * stale copy — or a double fini racing the slot's next owner — degrades to a no-op
 * instead of corrupting another walk; do not rely on this.) fh_cursor_init on a cursor
 * that is still live does NOT release the old slot (it never reads the caller's possibly
 * uninitialized memory) — fini first, or the slot leaks.
 *
 * Therefore the caller MUST guarantee fh_cursor_fini runs for every successful
 * fh_cursor_init, INCLUDING on any fault-recovery path. If you rely on a SIGSEGV handler
 * that longjmp/siglongjmps out of fh_step to survive a bad read, the recovery target must
 * be at or below the stepping scope so that fh_cursor_fini still runs afterwards;
 * otherwise the slot is never released and, over repeated faults, the pool drains and all
 * later fh_cursor_init calls fail (backtraces silently stop). Julia satisfies this: its
 * jl_set_safe_restore setjmp is local to jl_unw_stepn, so a caught fault returns from
 * jl_unw_stepn and the caller still runs fh_cursor_fini (jl_unw_fini).
 *
 * Mapping to libunwind:
 *   unw_getcontext / RtlCaptureContext      -> fh_capture_context
 *   (signal ucontext_t / CONTEXT)           -> fh_context_from_ucontext
 *   unw_init_local / unw_init_local2        -> fh_cursor_init
 *   (no libunwind equivalent; sampler path) -> fh_cursor_init_bounds
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

/* ABI lock: these sizes are hard-coded on both sides of the boundary (the Rust library
 * carries matching const assertions), because callers stack-allocate both types. */
#define FH_CONTEXT_SIZE 40
#define FH_CURSOR_SIZE 64
#if defined(__cplusplus) && __cplusplus >= 201103L
static_assert(sizeof(fh_context) == FH_CONTEXT_SIZE, "fh_context ABI size drift");
static_assert(sizeof(fh_cursor) == FH_CURSOR_SIZE, "fh_cursor ABI size drift");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(fh_context) == FH_CONTEXT_SIZE, "fh_context ABI size drift");
_Static_assert(sizeof(fh_cursor) == FH_CURSOR_SIZE, "fh_cursor ABI size drift");
#endif

/* 1 if this build can unwind natively on the current (os, arch); 0 otherwise.
 * When 0 on a non-x86_64/non-aarch64 build, all other functions are no-op stubs
 * returning failure. When 0 on a supported arch but unhandled OS, the lifecycle calls
 * (fh_init, fh_register_jit, ...) still succeed against an empty module set and only the
 * cursor path fails (fh_cursor_init < 0). Either way the caller should gate here and
 * keep using the existing unwinder (libunwind / dbghelp). */
int fh_supported(void);

/* ---- lifecycle (NOT async-signal-safe) ---- */

/* Allocate the unwinding slot pool (num_slots; 0 => default 256) and enumerate the
 * currently-loaded modules. Idempotent. Returns 0 on success. Call once at startup. */
int fh_init(size_t num_slots);

/* Capture per-thread state needed off the signal path (stack bounds). Call once per
 * thread that will be unwound/profiled, before any sampling. */
void fh_thread_register(void);

/* Re-scan loaded modules (call after dlopen AND dlclose; JIT modules are preserved).
 * On macOS this is a no-op once fh_init has run: dyld's add/remove-image callbacks keep
 * the module set current automatically. */
void fh_modules_refresh(void);

/* Diagnostics: number of JIT modules currently registered, and the cumulative count of
 * failed JIT registrations (bad arguments or unparsable .eh_frame — such code falls back
 * to frame-pointer stepping). Useful for asserting in embedder tests that registration
 * is actually happening. Not async-signal-safe (takes the registry lock). */
size_t fh_jit_module_count(void);
size_t fh_jit_register_failures(void);

/* ---- context capture (async-signal-safe) ---- */

/* Capture the CURRENT thread's context into *ctx (replaces unw_getcontext). */
void fh_capture_context(fh_context *ctx);

/* Fill *ctx from an OS context: a ucontext_t* (Unix signal handler 3rd arg) or a
 * CONTEXT* (Windows). Replaces the jl_to_bt_context / signal-frame init path.
 * Precondition: os_ctx must be NULL (yields a zeroed context) or a valid pointer to the
 * platform context type; any other non-NULL value is undefined behavior. */
void fh_context_from_ucontext(fh_context *ctx, const void *os_ctx);

/* Fill *ctx from a Darwin mach thread state (__darwin_x86_thread_state64 /
 * __darwin_arm_thread_state64) — what Julia's bt_context_t holds on macOS. */
void fh_context_from_thread_state(fh_context *ctx, const void *thread_state);

/* ---- cursor (async-signal-safe read path) ---- */

/* Initialize *cur to unwind from *ctx. Returns 0 on success, <0 on failure
 * (no modules published yet, or the slot pool is exhausted). */
int fh_cursor_init(fh_cursor *cur, const fh_context *ctx);

/* Like fh_cursor_init, with an explicit stack-read window [stack_lo, stack_hi) for the
 * walk. Pass the *target* thread's (or task's) exact stack range when unwinding a context
 * captured from another thread (suspended-target sampler) or running on an
 * embedder-managed stack — the fh_thread_register bounds only ever cover the *current*
 * thread's own pthread stack — which makes the stack reads fault-free. Passing 0, 0 (or
 * an empty range) falls back to the registered-bounds / sp-window heuristic. */
int fh_cursor_init_bounds(fh_cursor *cur, const fh_context *ctx,
                          uint64_t stack_lo, uint64_t stack_hi);

/* Output the CURRENT frame's ip/sp into *ip and *sp, then advance one frame.
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
