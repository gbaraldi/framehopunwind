//! Global unwinder state, published as an immutable snapshot and read from
//! (potentially) a signal handler.
//!
//! # Concurrency model
//!
//! The set of modules (the framehop [`Unwinder`]) changes over time as code is JIT
//! compiled and shared objects are loaded/unloaded. Writers (JIT registration, dlopen
//! refresh) run **off** the signal path, may allocate and take a mutex among
//! themselves, and publish a brand-new immutable [`Snapshot`] via an atomic pointer.
//!
//! Readers (the unwind path, possibly inside a `SIGPROF`/`SIGUSR` handler) must be
//! async-signal-safe: no allocation, no lock that a writer could be holding. We use a
//! hand-rolled **hazard pointer** scheme (the same idea `arc-swap` uses internally, but
//! with guarantees we can audit). A reader publishes the snapshot pointer it is about
//! to use into a global hazard slot, then re-checks that the pointer is still current.
//! A writer only frees a retired snapshot once no hazard slot references it — and that
//! free happens on the writer thread, never in a signal handler.
//!
//! Each in-flight unwind also owns a preallocated framehop [`Cache`] (carried in the
//! same slot), so the signal path never allocates a cache. Slots are claimed with a
//! lock-free CAS scan, which is signal-safe.

use core::cell::UnsafeCell;
use core::ptr;
use core::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use framehop::{
    CacheNative, FrameAddress, Module, MustNotAllocateDuringUnwind, UnwindRegsNative, Unwinder,
    UnwinderNative,
};

/// Owned unwind-section bytes. We always **copy** the unwind data (`.eh_frame`,
/// `__unwind_info`, `.pdata`, …) into owned memory at registration time (off the signal
/// path), so a snapshot is self-contained: a later `dlclose`/`munmap` or a Julia free of
/// a JIT buffer cannot turn a snapshot into a dangling read.
pub type Bytes = Box<[u8]>;

/// The native-arch framehop unwinder, using the allocation-free unwind policy.
pub type Unw = UnwinderNative<Bytes, MustNotAllocateDuringUnwind>;
/// The native-arch framehop cache, preallocated per slot.
pub type Cache = CacheNative<MustNotAllocateDuringUnwind>;

/// An immutable published view of the module set. Never mutated after publication.
pub struct Snapshot {
    pub unw: Unw,
    /// `max_known_code_address()` cached for cheap aarch64 pointer-auth mask derivation.
    pub max_code_addr: u64,
}

// SAFETY: A published `Snapshot` is immutable. `Unw::unwind_frame` takes `&self` and only
// reads the module list (the mutable scratch state lives in the per-call `Cache` and the
// `read_stack` closure, both owned exclusively by one cursor). `Bytes` is immutable owned
// data. Therefore concurrent shared access from multiple unwinding threads is sound.
unsafe impl Send for Snapshot {}
unsafe impl Sync for Snapshot {}

/// Per-cursor mutable state. Touched only by the single cursor that has claimed the
/// enclosing [`Slot`] (guaranteed by `in_use`), so interior mutability is sound.
pub struct SlotInner {
    pub cache: Cache,
    pub regs: UnwindRegsNative,
    pub cur_addr: FrameAddress,
    pub cur_ip: u64,
    pub done: bool,
    /// The hazard-protected snapshot this walk is reading from.
    pub snapshot: *const Snapshot,
    /// Effective `[lo, hi)` stack-read window for this walk.
    pub stack_lo: u64,
    pub stack_hi: u64,
}

/// A reusable unwinding slot: a hazard cell + a preallocated cache + per-walk state.
pub struct Slot {
    /// Claimed exclusively by one cursor for the duration of a stack walk.
    pub in_use: AtomicBool,
    /// Claim sequence number, bumped on every claim AND every release. A cursor records
    /// the value observed at claim time as its nonce; any later use of a stale cursor
    /// (a copied struct, a double-fini after the slot was re-claimed) sees a mismatched
    /// seq and becomes a no-op instead of corrupting the slot's new owner.
    pub seq: AtomicU64,
    /// The snapshot pointer this slot is currently protecting from reclamation.
    pub hazard: AtomicPtr<Snapshot>,
    inner: UnsafeCell<SlotInner>,
}

// SAFETY: `inner` is only ever accessed through `&mut` by the unique cursor that won the
// `in_use` CAS; all cross-thread access (`in_use`, `hazard`) goes through atomics.
unsafe impl Sync for Slot {}
unsafe impl Send for Slot {}

impl Slot {
    fn new() -> Self {
        Slot {
            in_use: AtomicBool::new(false),
            seq: AtomicU64::new(0),
            hazard: AtomicPtr::new(ptr::null_mut()),
            inner: UnsafeCell::new(placeholder_inner()),
        }
    }

    /// Get exclusive access to the slot's inner state.
    ///
    /// # Safety
    /// The caller must hold this slot (won its `in_use` CAS) and must not alias.
    // clippy::mut_from_ref is deny-by-default because &self -> &mut is usually a bug;
    // here the exclusivity clippy cannot see is provided by the `in_use` CAS (exactly the
    // UnsafeCell interior-mutability pattern the lint carves out for cell types).
    #[allow(clippy::mut_from_ref)]
    #[inline]
    pub unsafe fn inner_mut(&self) -> &mut SlotInner {
        &mut *self.inner.get()
    }

    /// Get shared access to the slot's inner state (for read-only peeks like `fh_get_reg`).
    ///
    /// # Safety
    /// The caller must hold this slot, and no `&mut` from [`inner_mut`](Self::inner_mut)
    /// may be live for the duration of the borrow.
    #[inline]
    pub unsafe fn inner_ref(&self) -> &SlotInner {
        &*self.inner.get()
    }
}

fn placeholder_inner() -> SlotInner {
    SlotInner {
        cache: CacheNative::new_in(),
        regs: super::arch::make_regs(&super::arch::FhContext::zeroed(), u64::MAX),
        cur_addr: FrameAddress::from_instruction_pointer(0),
        cur_ip: 0,
        done: true,
        snapshot: ptr::null(),
        stack_lo: 0,
        stack_hi: 0,
    }
}

/// Writer-only bookkeeping, guarded by `WRITER`.
struct WriterState {
    /// Snapshots that have been replaced but may still be referenced by an in-flight
    /// reader. Freed lazily once no hazard slot references them.
    retired: Vec<*mut Snapshot>,
}

// SAFETY: only ever accessed under the `WRITER` mutex, on non-signal threads.
unsafe impl Send for WriterState {}

/// The currently published snapshot. Null until the first publish.
static CURRENT: AtomicPtr<Snapshot> = AtomicPtr::new(ptr::null_mut());

/// The slot pool, allocated once by [`init`].
static SLOTS: OnceLock<Box<[Slot]>> = OnceLock::new();

/// Serializes writers and guards the retired list. Never taken on the signal path.
static WRITER: Mutex<WriterState> = Mutex::new(WriterState {
    retired: Vec::new(),
});

const DEFAULT_SLOTS: usize = 256;

/// Count of module *mutations* (add/remove) applied through [`with_writer`], driving the
/// periodic rule-cache sweep below.
static MUTATIONS: AtomicU64 = AtomicU64::new(0);

/// framehop keys its per-cache unwind-rule entries by `(address, modules_generation)`
/// where the generation is a global **u16** bumped on every module add/remove — after
/// 65535 mutations it wraps, and a never-cleared entry in a rarely-claimed slot could
/// then resurrect a stale rule for recycled code. Sweeping all claimable slot caches
/// every `CACHE_SWEEP_MUTATIONS` *mutations* (counted exactly via [`WriterUnw`]; a single
/// publish can carry many, e.g. a module refresh) keeps every cache's contents well
/// inside one generation cycle, with an 8x margin. Writer-thread only; allocation is
/// fine here.
const CACHE_SWEEP_MUTATIONS: u64 = 8192;

fn sweep_slot_caches() {
    if let Some(slots) = slots() {
        for s in slots {
            if s.in_use
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                // SAFETY: we own the slot via the CAS; replacing the cache cannot race a
                // reader. A slot held by an in-flight walk is skipped — its cache entries
                // are at most one walk old and the next sweep will catch it.
                unsafe {
                    s.inner_mut().cache = CacheNative::new_in();
                }
                s.in_use.store(false, Ordering::Release);
            }
        }
    }
}

/// Initialize the slot pool. Idempotent; `num_slots == 0` selects a default.
/// Must be called off the signal path (it allocates the caches).
pub fn init(num_slots: usize) {
    SLOTS.get_or_init(|| {
        let n = if num_slots == 0 {
            DEFAULT_SLOTS
        } else {
            num_slots
        };
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(Slot::new());
        }
        v.into_boxed_slice()
    });
}

#[inline]
fn slots() -> Option<&'static [Slot]> {
    SLOTS.get().map(|b| &b[..])
}

/// Claim a free slot with a lock-free CAS scan. Signal-safe. Returns the slot index and
/// the claim nonce the owning cursor must present on every later access (see
/// [`Slot::seq`]).
pub fn claim_slot() -> Option<(usize, u64)> {
    let slots = slots()?;
    for (i, s) in slots.iter().enumerate() {
        if s.in_use
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
        {
            let nonce = s.seq.fetch_add(1, Ordering::Relaxed) + 1;
            return Some((i, nonce));
        }
    }
    None
}

/// True iff `nonce` is the current claim nonce of slot `idx` (i.e. the presenting cursor
/// is the slot's live owner, not a stale copy). Signal-safe.
#[inline]
pub fn nonce_matches(idx: usize, nonce: u64) -> bool {
    match slot(idx) {
        Some(s) => s.seq.load(Ordering::Relaxed) == nonce,
        None => false,
    }
}

#[inline]
pub fn slot(idx: usize) -> Option<&'static Slot> {
    slots()?.get(idx)
}

/// Release a slot claimed with `nonce`: retire the claim, clear the hazard, and mark the
/// slot free. Signal-safe. The release right is taken by CAS-ing `seq` from `nonce`, so
/// exactly one releaser wins — a stale cursor copy (or a double-fini racing the slot's
/// next owner) loses the CAS and becomes a no-op instead of freeing someone else's slot.
pub fn release_slot(idx: usize, nonce: u64) {
    if let Some(s) = slot(idx) {
        if s.seq
            .compare_exchange(
                nonce,
                nonce.wrapping_add(1),
                Ordering::AcqRel,
                Ordering::Relaxed,
            )
            .is_err()
        {
            return; // not the live owner of this claim
        }
        s.hazard.store(ptr::null_mut(), Ordering::SeqCst);
        // Drop the snapshot reference held in inner (just clears the raw pointer; the
        // snapshot itself is reclaimed by the writer, never here).
        // SAFETY: we won the release CAS, so we are the unique owner until in_use clears.
        unsafe {
            s.inner_mut().snapshot = ptr::null();
        }
        s.in_use.store(false, Ordering::Release);
    }
}

/// Acquire the current snapshot for `slot_idx` under hazard protection. Signal-safe.
/// Returns a raw pointer to a snapshot that is guaranteed not to be freed until this
/// slot's hazard is cleared (via [`release_slot`]). Returns null if no snapshot is
/// published yet.
pub fn acquire_snapshot(slot_idx: usize) -> *const Snapshot {
    let s = match slot(slot_idx) {
        Some(s) => s,
        None => return ptr::null(),
    };
    loop {
        let p = CURRENT.load(Ordering::Acquire);
        if p.is_null() {
            s.hazard.store(ptr::null_mut(), Ordering::SeqCst);
            return ptr::null();
        }
        s.hazard.store(p, Ordering::SeqCst);
        // Re-validate: if CURRENT is still p after we published the hazard, then p was
        // not retired before our hazard became visible, so it cannot be freed under us.
        //
        // This is a store-buffering (Dekker) pattern across two atomics (`hazard` and
        // `CURRENT`): the reader stores `hazard` then loads `CURRENT`, while the writer
        // swaps `CURRENT` then scans `hazard`. Correctness requires all four accesses to
        // share one total order, so the recheck load MUST be SeqCst (Acquire is too weak
        // on weakly-ordered hardware such as aarch64 — it would not order against the
        // writer's swap, admitting a use-after-free).
        if CURRENT.load(Ordering::SeqCst) == p {
            return p as *const Snapshot;
        }
        // Otherwise a writer swapped; retry against the new current.
    }
}

/// True iff `p` is referenced by any hazard slot.
fn is_hazarded(p: *mut Snapshot) -> bool {
    if let Some(slots) = slots() {
        for s in slots {
            if s.hazard.load(Ordering::SeqCst) == p {
                return true;
            }
        }
    }
    false
}

/// Free any retired snapshots that no reader still references. Writer-thread only.
fn reclaim(st: &mut WriterState) {
    st.retired.retain(|&p| {
        if is_hazarded(p) {
            true // keep; a reader still holds it
        } else {
            // SAFETY: not hazarded and already removed from CURRENT, so no new reader can
            // acquire it; this is the unique owner. Drop on the writer thread.
            unsafe { drop(Box::from_raw(p)) };
            false
        }
    });
}

/// Mutation-counting facade over [`Unw`], handed to [`with_writer`] closures. Counting
/// every add/remove exactly is what makes the cache-sweep bound real: framehop bumps its
/// (u16, wrapping) global modules-generation once per *mutation*, and one publish can
/// carry many (a refresh after loading many libraries), so counting publishes would
/// under-count.
pub struct WriterUnw<'a> {
    unw: &'a mut Unw,
    mutations: u64,
}

impl WriterUnw<'_> {
    pub fn add_module(&mut self, module: Module<Bytes>) {
        self.mutations += 1;
        self.unw.add_module(module);
    }
    pub fn remove_module(&mut self, module_address_range_start: u64) {
        self.mutations += 1;
        self.unw.remove_module(module_address_range_start);
    }
}

/// Run a mutation against a fresh clone of the current module set and publish it.
///
/// `f` receives a [`WriterUnw`] (cloned from the current snapshot, or empty) and adds /
/// removes modules. Off the signal path; allocates.
pub fn with_writer<R>(f: impl FnOnce(&mut WriterUnw) -> R) -> R {
    let mut guard = WRITER.lock().unwrap_or_else(|e| e.into_inner());

    let cur = CURRENT.load(Ordering::Acquire);
    let mut new_unw: Unw = if cur.is_null() {
        UnwinderNative::new()
    } else {
        // SAFETY: cur is the current published snapshot; immutable and alive (we hold the
        // writer lock so no concurrent reclaim of CURRENT; CURRENT is only swapped here).
        unsafe { (*cur).unw.clone() }
    };

    let mut writer = WriterUnw {
        unw: &mut new_unw,
        mutations: 0,
    };
    let ret = f(&mut writer);
    let mutations = writer.mutations;

    let max_code_addr = new_unw.max_known_code_address();
    let boxed = Box::into_raw(Box::new(Snapshot {
        unw: new_unw,
        max_code_addr,
    }));
    // SeqCst (not AcqRel): pairs with the reader's SeqCst recheck load in acquire_snapshot
    // so the publish and the subsequent hazard scan share one total order with the reader's
    // hazard store and recheck (see acquire_snapshot).
    let old = CURRENT.swap(boxed, Ordering::SeqCst);
    if !old.is_null() {
        guard.retired.push(old);
    }
    reclaim(&mut guard);
    let before = MUTATIONS.fetch_add(mutations, Ordering::Relaxed);
    if before / CACHE_SWEEP_MUTATIONS != (before + mutations) / CACHE_SWEEP_MUTATIONS {
        sweep_slot_caches();
    }
    ret
}
