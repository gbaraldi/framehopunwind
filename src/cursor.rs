//! The unwind cursor: the async-signal-safe read path.
//!
//! A `fh_cursor` is an opaque, caller-allocated handle that just references a claimed
//! [`Slot`](crate::state::Slot) (which holds the real per-walk state + preallocated
//! cache). `fh_step` mirrors Julia's `jl_unw_step` contract exactly: it **outputs the
//! current frame's ip/sp, then advances** the cursor, returning `>0` if a further frame
//! exists, `0` at a clean end, `<0` on error.
//!
//! All entry points take **raw pointers** and copy the cursor in/out with `ptr::read` /
//! `ptr::write` rather than forming `&`/`&mut` references over caller memory: the C
//! caller's `fh_cursor` starts out uninitialized, and nothing stops a C caller from
//! passing aliased output pointers (`ip == sp`), so references would assert validity and
//! exclusivity guarantees the FFI boundary cannot provide.
//!
//! A cursor carries a per-claim **nonce** checked against the slot's claim sequence on
//! every access, so a stale cursor (a struct copy fini'd twice, or use after the slot was
//! recycled) degrades to a no-op / error instead of corrupting the slot's next owner.
//!
//! # Fault & slot-lifetime contract
//!
//! `fh_step`'s stack reads are bounds-checked but only *fault-bounded*: with the
//! sp-derived fallback window (a thread that did not call `fh_thread_register`) a bad
//! address can still hit `SIGSEGV`. We deliberately do **not** install a `SIGSEGV` handler
//! (it would conflict with the embedder's). Three consequences the caller must respect:
//!
//!  1. To make reads fault-free, give every walk exact mapped bounds (register the thread,
//!     or pass the target's stack range to [`cursor_init_bounds`]); out-of-stack reads
//!     then become a clean end-of-stack instead of a fault.
//!  2. If the embedder instead recovers from a faulting read by `longjmp`ing out of
//!     `fh_step` (Julia's `jl_set_safe_restore`), note that the jump crosses this
//!     library's Rust frames. That is only defined behavior while those frames have no
//!     pending destructors ("plain old frames", RFC 2945); the read path is written
//!     allocation- and destructor-free to keep that true, but exact bounds — which avoid
//!     the fault entirely — are the *supported* configuration, and the longjmp path is
//!     best-effort.
//!  3. A cursor holds a pooled slot from [`cursor_init`] until [`cursor_fini`]. The caller
//!     MUST run `cursor_fini` for every successful init, **including on a fault-recovery
//!     path** — the `longjmp` target must sit at or below the stepping scope so
//!     `cursor_fini` still runs afterwards; otherwise the slot leaks and, over repeated
//!     faults, the fixed pool drains and later `cursor_init`s fail (backtraces silently
//!     stop). Julia satisfies this: `jl_set_safe_restore`'s `setjmp` is local to
//!     `jl_unw_stepn`, so a caught fault returns from it and the caller still runs
//!     `fh_cursor_fini`. Note `cursor_init` on a still-live cursor does NOT release the
//!     old slot (it cannot read the old contents safely); fini first.

use core::ffi::c_int;

use framehop::{FrameAddress, Unwinder};

use crate::arch::{self, FhContext};
use crate::state::{self, Snapshot};

/// Magic written into a live cursor, to catch use-of-uninitialized / double-fini.
const CURSOR_MAGIC: u64 = 0x46_48_43_55_52_53_00_01; // "FHCURS\0\x01"

/// Caller-allocated opaque cursor. Must be at least this size; the C header sizes it
/// generously. We only store an index + magic + claim nonce; the heavy state lives in
/// the slot.
#[repr(C)]
pub struct FhCursor {
    magic: u64,
    slot: u64,
    /// The claim nonce returned by [`state::claim_slot`]; must still match the slot's
    /// claim sequence for this cursor to count as the slot's live owner.
    nonce: u64,
    /// Reserved padding to fix the opaque-cursor ABI at 64 bytes (must match
    /// `fh_cursor` in the C header). Not currently used; all per-walk state lives in the
    /// pooled `SlotInner`.
    _reserved: [u64; 5],
}

// Compile-time ABI lock: the C header hard-codes these layouts (fh_cursor as
// uint64_t[8], fh_context as uint64_t[5]) and Julia stack-allocates both, so silent
// drift would corrupt the caller's stack frame. The header carries the matching
// static_asserts via FH_CURSOR_SIZE / FH_CONTEXT_SIZE.
const _: () = {
    assert!(core::mem::size_of::<FhCursor>() == 64);
    assert!(core::mem::align_of::<FhCursor>() == 8);
    assert!(core::mem::size_of::<crate::arch::FhContext>() == 40);
    assert!(core::mem::align_of::<crate::arch::FhContext>() == 8);
};

impl FhCursor {
    /// The canonical "not a live cursor" value, written on init failure and fini.
    #[inline]
    fn dead() -> Self {
        FhCursor {
            magic: 0,
            slot: u64::MAX,
            nonce: 0,
            _reserved: [0; 5],
        }
    }
    /// Live check: magic + slot index + claim nonce must all still hold.
    #[inline]
    fn is_live(&self) -> bool {
        self.magic == CURSOR_MAGIC
            && self.slot != u64::MAX
            && state::nonce_matches(self.slot as usize, self.nonce)
    }
}

/// Compute the effective stack-read window for a walk starting at `sp`, when the caller
/// did not pass explicit bounds.
///
/// Prefer the thread's registered bounds — but note these are the *current* (unwinding)
/// thread's pthread-stack bounds, so they only apply to same-thread walks whose sp is on
/// that stack; a cross-thread walk (sampler unwinding a suspended target) or a walk over
/// an embedder-managed stack (e.g. a Julia task stack) never matches and takes the
/// fallback. Callers that know the target's exact stack range should pass it to
/// [`cursor_init_bounds`] instead, which both covers those cases and makes the reads
/// fault-free. The fallback is a *bounded* window above the starting sp (the stack grows
/// downward, so callers' frames are at higher addresses): it caps how far a stray pointer
/// can stray but does NOT guarantee fault-freedom (the window may include unmapped pages
/// past the real stack top), so a faulting read is still possible and the embedder must
/// catch SIGSEGV (Julia uses `jl_set_safe_restore`). The fallback span is kept near a
/// typical `RLIMIT_STACK` rather than wide.
fn stack_window(sp: u64) -> (u64, u64) {
    const FALLBACK_SPAN: u64 = 16 * 1024 * 1024; // 16 MiB above sp (~typical RLIMIT_STACK)
    let (lo, hi) = crate::stackbounds::current();
    if lo != 0 && hi != 0 && sp >= lo && sp < hi {
        (lo, hi)
    } else {
        let lo = sp & !0xfff; // page-align down
        (lo, lo.saturating_add(FALLBACK_SPAN))
    }
}

/// Initialize `*cur` to unwind starting from `*ctx`. Async-signal-safe.
///
/// Returns 0 on success, `<0` on failure (no published modules, or no free slot).
///
/// # Safety (crate-internal contract)
/// `cur` and `ctx` are non-null (checked by the FFI layer) and point at caller-allocated
/// storage of the right size/alignment. `*cur` may be uninitialized; it is only written.
pub fn cursor_init(cur: *mut FhCursor, ctx: *const FhContext) -> c_int {
    cursor_init_bounds(cur, ctx, 0, 0)
}

/// Like [`cursor_init`], but with an explicit stack-read window `[stack_lo, stack_hi)`
/// for the walk. Pass the *target* thread's (or task's) exact stack range when unwinding
/// a context that is not the current thread's — the per-thread registered bounds cannot
/// cover that case — which makes the stack reads fault-free. Passing `0, 0` (or an empty
/// range) falls back to the registered-bounds / sp-window heuristic. Async-signal-safe
/// (explicit bounds skip even the thread-local lookup).
pub fn cursor_init_bounds(
    cur: *mut FhCursor,
    ctx: *const FhContext,
    stack_lo: u64,
    stack_hi: u64,
) -> c_int {
    // Write-only teardown of whatever was in *cur (possibly uninitialized C stack
    // memory — never read it). A still-live cursor's slot is NOT released here; that
    // would require reading potentially-uninitialized memory. Fini before re-init.
    // SAFETY: cur is valid for writes (FFI layer null-checked; caller sized it).
    unsafe { cur.write(FhCursor::dead()) };
    // SAFETY: ctx is a valid, fully initialized FhContext (all five words are written by
    // every capture/extraction path).
    let ctx: FhContext = unsafe { ctx.read() };

    let (slot_idx, nonce) = match state::claim_slot() {
        Some(c) => c,
        None => return -1, // pool exhausted; skip this sample
    };
    let snap = state::acquire_snapshot(slot_idx);
    if snap.is_null() {
        state::release_slot(slot_idx, nonce);
        return -2; // nothing registered yet
    }

    // Derive the pointer-auth mask (aarch64/macOS) from the snapshot's max code address.
    // SAFETY: `snap` is hazard-protected for this slot until release.
    let max_code = unsafe { (*snap).max_code_addr };
    let mask = ptr_auth_mask(max_code);

    let sp0 = arch::context_sp(&ctx);
    let (lo, hi) = if stack_lo < stack_hi {
        (stack_lo, stack_hi) // explicit, trusted verbatim; no TLS touched
    } else {
        stack_window(sp0)
    };

    // Populate the slot's per-walk state.
    let slot = match state::slot(slot_idx) {
        Some(s) => s,
        None => {
            state::release_slot(slot_idx, nonce);
            return -3;
        }
    };
    // SAFETY: we exclusively own the slot (won the claim CAS).
    let inner = unsafe { slot.inner_mut() };
    inner.regs = arch::make_regs(&ctx, mask);
    inner.cur_addr = arch::initial_frame_address(&ctx);
    inner.cur_ip = arch::context_ip(&ctx);
    inner.done = false;
    inner.snapshot = snap;
    inner.stack_lo = lo;
    inner.stack_hi = hi;

    // SAFETY: cur is valid for writes (see above).
    unsafe {
        cur.write(FhCursor {
            magic: CURSOR_MAGIC,
            slot: slot_idx as u64,
            nonce,
            _reserved: [0; 5],
        });
    }
    0
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn ptr_auth_mask(max_code: u64) -> u64 {
    // macOS arm64e signs return addresses; derive a mask from the highest known code
    // address (leading zero bits are reserved for the PAC hash). On Linux/FreeBSD this is
    // effectively a no-op because code lives in the low canonical range.
    if max_code == 0 {
        u64::MAX
    } else {
        u64::MAX >> max_code.leading_zeros()
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn ptr_auth_mask(_max_code: u64) -> u64 {
    u64::MAX
}

/// Advance one frame. Outputs the current frame's ip/sp into `*out_ip`/`*out_sp`, then
/// steps to the caller. Returns `>0` if a further frame exists, `0` at end, `<0` on
/// error. Async-signal-safe. Output pointers are written through raw writes, so aliased
/// outputs (`out_ip == out_sp`) are last-write-wins rather than undefined behavior.
pub fn step(cur: *mut FhCursor, out_ip: *mut u64, out_sp: *mut u64) -> c_int {
    // SAFETY: non-null (FFI layer) and valid for writes per the C contract.
    unsafe {
        out_ip.write(0);
        out_sp.write(0);
    }
    // SAFETY: cur is non-null; copy it out so later output writes cannot alias our view.
    let c: FhCursor = unsafe { cur.read() };
    if !c.is_live() {
        return -1;
    }
    let slot_idx = c.slot as usize;
    let slot = match state::slot(slot_idx) {
        Some(s) => s,
        None => return -1,
    };
    // SAFETY: this cursor is the slot's live owner (nonce checked) until fini. We hold a
    // single `&mut` and split it into disjoint field borrows below.
    let inner = unsafe { slot.inner_mut() };
    if inner.done {
        return 0;
    }
    if inner.snapshot.is_null() {
        return -1;
    }

    // Output the CURRENT frame first (matches jl_unw_step semantics).
    // SAFETY: valid for writes (see above).
    unsafe {
        out_ip.write(inner.cur_ip);
        out_sp.write(inner.regs.sp());
    }

    // SAFETY: `snapshot` is hazard-protected for this slot until fini.
    let snap: &Snapshot = unsafe { &*inner.snapshot };

    let stack_lo = inner.stack_lo;
    let stack_hi = inner.stack_hi;
    let mut read_stack = move |addr: u64| -> Result<u64, ()> {
        // Bounded + aligned read of the target stack. This caps how far a stray pointer
        // can reach but is fault-*bounded*, not fault-free: with only the SP-derived
        // fallback window the range may include unmapped pages, so the read below can still
        // fault and the embedder must catch SIGSEGV (Julia: jl_set_safe_restore).
        if addr < stack_lo || addr.saturating_add(8) > stack_hi || (addr & 0x7) != 0 {
            return Err(());
        }
        // SAFETY: addr is within the thread's stack window and 8-aligned. With exact
        // registered bounds this never faults; with the fallback window a residual fault is
        // possible and is the embedder's SIGSEGV handler's responsibility.
        Ok(unsafe { core::ptr::read_volatile(addr as *const u64) })
    };

    // Disjoint field borrows of `*inner`: cur_addr (Copy), &mut regs, &mut cache.
    let cur_addr = inner.cur_addr;
    let result =
        snap.unw
            .unwind_frame(cur_addr, &mut inner.regs, &mut inner.cache, &mut read_stack);

    match result {
        // Total match (no unwrap): a null return address ends the stack, matching
        // framehop's iterator. framehop can legitimately surface Ok(Some(0)) via its
        // Uncacheable path, so this must be handled, not asserted — the read path must
        // never panic across the FFI boundary.
        Ok(Some(return_address)) => match core::num::NonZeroU64::new(return_address) {
            Some(ra) => {
                inner.cur_ip = ra.get();
                inner.cur_addr = FrameAddress::ReturnAddress(ra);
                1 // more frames may follow
            }
            None => {
                inner.done = true;
                0
            }
        },
        Ok(None) => {
            inner.done = true;
            0 // reached root; current frame already emitted
        }
        // Any unwind error — including a bad stack read (Error::CouldNotReadStack) — is
        // reported as a normal truncation: the frames emitted so far are valid, and the
        // caller cannot do anything more useful with a distinction here.
        Err(_) => {
            inner.done = true;
            0
        }
    }
}

/// Read the current ip/sp without advancing (mirrors `jl_unw_get` on an initialized
/// cursor). Safe to call after init and between steps. Read-only: takes only a shared
/// view of the slot state, so it never manufactures a `&mut` from a `const fh_cursor *`.
pub fn get_reg(cur: *const FhCursor, out_ip: *mut u64, out_sp: *mut u64) {
    // SAFETY: non-null (FFI layer) and valid for writes per the C contract.
    unsafe {
        out_ip.write(0);
        out_sp.write(0);
    }
    // SAFETY: cur is non-null; copy out (see `step`).
    let c: FhCursor = unsafe { cur.read() };
    if !c.is_live() {
        return;
    }
    if let Some(slot) = state::slot(c.slot as usize) {
        // SAFETY: owned by this cursor; no `&mut` is live (single-owner contract, and
        // `step`'s exclusive borrow ends before it returns).
        let inner = unsafe { slot.inner_ref() };
        // SAFETY: valid for writes (see above).
        unsafe {
            out_ip.write(inner.cur_ip);
            out_sp.write(inner.regs.sp());
        }
    }
}

/// Release the cursor's slot. Idempotent (the release CAS makes a stale or repeated fini
/// a no-op); async-signal-safe.
pub fn cursor_fini(cur: *mut FhCursor) {
    // SAFETY: cur is non-null (FFI layer).
    let c: FhCursor = unsafe { cur.read() };
    if c.magic == CURSOR_MAGIC && c.slot != u64::MAX {
        // release_slot verifies the nonce by CAS, so a stale copy cannot free the slot's
        // next owner.
        state::release_slot(c.slot as usize, c.nonce);
    }
    // SAFETY: valid for writes.
    unsafe { cur.write(FhCursor::dead()) };
}
