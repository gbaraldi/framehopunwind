//! The unwind cursor: the async-signal-safe read path.
//!
//! A `fh_cursor` is an opaque, caller-allocated handle that just references a claimed
//! [`Slot`](crate::state::Slot) (which holds the real per-walk state + preallocated
//! cache). `fh_step` mirrors Julia's `jl_unw_step` contract exactly: it **outputs the
//! current frame's ip/sp, then advances** the cursor, returning `>0` if a further frame
//! exists, `0` at a clean end, `<0` on error.

use core::ffi::c_int;

use framehop::{Error, FrameAddress, Unwinder};

use crate::arch::{self, FhContext};
use crate::state::{self, Snapshot};

/// Magic written into a live cursor, to catch use-of-uninitialized / double-fini.
const CURSOR_MAGIC: u64 = 0x46_48_43_55_52_53_00_01; // "FHCURS\0\x01"

/// Caller-allocated opaque cursor. Must be at least this size; the C header sizes it
/// generously. We only store an index + magic; the heavy state lives in the slot.
#[repr(C)]
pub struct FhCursor {
    magic: u64,
    slot: u64,
    /// Reserved padding to fix the opaque-cursor ABI at 64 bytes (must match
    /// `fh_cursor` in the C header). Not currently used; all per-walk state lives in the
    /// pooled `SlotInner`.
    _reserved: [u64; 6],
}

impl FhCursor {
    #[inline]
    fn invalidate(&mut self) {
        self.magic = 0;
        self.slot = u64::MAX;
    }
    #[inline]
    fn is_live(&self) -> bool {
        self.magic == CURSOR_MAGIC && self.slot != u64::MAX
    }
}

/// Compute the effective stack-read window for a walk starting at `sp`.
///
/// Prefer the thread's exact registered bounds (these are fault-safe — they cover only
/// mapped stack). Otherwise fall back to a *bounded* window above the starting sp (the
/// stack grows downward, so callers' frames are at higher addresses). The fallback only
/// caps how far a stray pointer can stray; it does NOT guarantee fault-freedom (the window
/// may include unmapped pages past the real stack top), so a faulting read is still
/// possible and the embedder must catch SIGSEGV (Julia uses `jl_set_safe_restore`). The
/// fallback span is kept near a typical `RLIMIT_STACK` rather than wide.
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

/// Initialize `cur` to unwind starting from `ctx`. Async-signal-safe.
///
/// Returns 0 on success, `<0` on failure (no published modules, or no free slot).
pub fn cursor_init(cur: &mut FhCursor, ctx: &FhContext) -> c_int {
    cur.invalidate();

    let slot_idx = match state::claim_slot() {
        Some(i) => i,
        None => return -1, // pool exhausted; skip this sample
    };
    let snap = state::acquire_snapshot(slot_idx);
    if snap.is_null() {
        state::release_slot(slot_idx);
        return -2; // nothing registered yet
    }

    // Derive the pointer-auth mask (aarch64/macOS) from the snapshot's max code address.
    // SAFETY: `snap` is hazard-protected for this slot until release.
    let max_code = unsafe { (*snap).max_code_addr };
    let mask = ptr_auth_mask(max_code);

    let sp0 = arch::context_sp(ctx);
    let (lo, hi) = stack_window(sp0);

    // Populate the slot's per-walk state.
    let slot = match state::slot(slot_idx) {
        Some(s) => s,
        None => {
            state::release_slot(slot_idx);
            return -3;
        }
    };
    // SAFETY: we exclusively own the slot (won the claim CAS).
    let inner = unsafe { slot.inner_mut() };
    inner.regs = arch::make_regs(ctx, mask);
    inner.cur_addr = arch::initial_frame_address(ctx);
    inner.cur_ip = arch::context_ip(ctx);
    inner.done = false;
    inner.snapshot = snap;
    inner.stack_lo = lo;
    inner.stack_hi = hi;

    cur.magic = CURSOR_MAGIC;
    cur.slot = slot_idx as u64;
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

/// Advance one frame. Outputs the current frame's ip/sp into `*ip`/`*sp`, then steps to
/// the caller. Returns `>0` if a further frame exists, `0` at end, `<0` on error.
/// Async-signal-safe.
pub fn step(cur: &mut FhCursor, out_ip: &mut u64, out_sp: &mut u64) -> c_int {
    *out_ip = 0;
    *out_sp = 0;
    if !cur.is_live() {
        return -1;
    }
    let slot_idx = cur.slot as usize;
    let slot = match state::slot(slot_idx) {
        Some(s) => s,
        None => return -1,
    };
    // SAFETY: this cursor exclusively owns the slot until fini. We hold a single `&mut`
    // and split it into disjoint field borrows below (never two refs to the same field).
    let inner = unsafe { slot.inner_mut() };
    if inner.done {
        return 0;
    }
    if inner.snapshot.is_null() {
        return -1;
    }

    // Output the CURRENT frame first (matches jl_unw_step semantics).
    *out_ip = inner.cur_ip;
    *out_sp = inner.regs.sp();

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
        Err(e) => {
            inner.done = true;
            match e {
                // A bad stack read is a normal truncation, not a hard error.
                Error::CouldNotReadStack(_) | Error::ReturnAddressIsNull => 0,
                _ => 0,
            }
        }
    }
}

/// Read the current ip/sp without advancing (mirrors `jl_unw_get` on an initialized
/// cursor). Safe to call after init and between steps.
pub fn get_reg(cur: &FhCursor, out_ip: &mut u64, out_sp: &mut u64) {
    *out_ip = 0;
    *out_sp = 0;
    if !cur.is_live() {
        return;
    }
    if let Some(slot) = state::slot(cur.slot as usize) {
        // SAFETY: owned by this cursor.
        let inner = unsafe { slot.inner_mut() };
        *out_ip = inner.cur_ip;
        *out_sp = inner.regs.sp();
    }
}

/// Release the cursor's slot. Idempotent; async-signal-safe.
pub fn cursor_fini(cur: &mut FhCursor) {
    if cur.is_live() {
        state::release_slot(cur.slot as usize);
    }
    cur.invalidate();
}
