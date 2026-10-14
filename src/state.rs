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
use core::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

use framehop::{
    CacheNative, FrameAddress, MustNotAllocateDuringUnwind, UnwindRegsNative, Unwinder,
    UnwinderNative,
};

/// Unwind-section and code bytes.
///
/// `Mapped` bytes are read in place and are valid only while their object stays mapped.
/// They are safe where a synchronous unload callback removes the module from the published
/// set before the loader unmaps it (dyld remove-image on macOS, DLL notifications on
/// Windows). A walk that holds an older snapshot never looks the module up, because no live
/// frame can be inside an object that is being unloaded.
///
/// `Owned` bytes are copies. Linux and FreeBSD have no unload callback, so a module can stay
/// in the set after its object is unloaded. If the loader reuses that address range, a walk
/// through live code there looks the stale module up. Thus ELF unwind tables are copies: a
/// copy gives wrong frames until the next refresh, but mapped bytes give reads of freed
/// memory. JIT `.eh_frame` is copied so that the embedder can free its buffer after it
/// deregisters the module.
///
/// Code bytes are `Mapped` everywhere. framehop analyzes Mach-O prologues and epilogues and
/// PE epilogues only for the first frame of a walk, and a first frame is code that the
/// thread will return to. On x86-64 Mach-O, framehop also reads the `sub rsp` immediate for
/// any frame whose compact-unwind opcode is frameless-indirect. This read is safe because
/// dyld drops the module before the image is unmapped, so a correct walk never reaches it.
pub enum Bytes {
    Owned(Box<[u8]>),
    Mapped(&'static [u8]),
}

impl Bytes {
    /// An owned copy of `bytes`.
    pub fn copy_of(bytes: &[u8]) -> Self {
        Bytes::Owned(Box::from(bytes))
    }
}

impl core::ops::Deref for Bytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Bytes::Owned(bytes) => bytes,
            Bytes::Mapped(bytes) => bytes,
        }
    }
}

impl Default for Bytes {
    fn default() -> Self {
        Bytes::Mapped(&[])
    }
}

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
// `read_stack` closure, both owned exclusively by one cursor). `Bytes` is immutable.
// Therefore concurrent shared access from multiple unwinding threads is sound.
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
    /// A second window, or `(0, 0)`: the stack a signal interrupted, when the walk starts
    /// on a signal stack.
    pub stack_lo2: u64,
    pub stack_hi2: u64,
}

/// A reusable unwinding slot: a hazard cell + a preallocated cache + per-walk state.
pub struct Slot {
    /// Claimed exclusively by one cursor for the duration of a stack walk.
    pub in_use: AtomicBool,
    /// Claim sequence, bumped on every claim and release. A cursor keeps the claim-time
    /// value as its nonce; a stale cursor (copy, double-fini) mismatches and no-ops.
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
    // mut_from_ref: exclusivity clippy can't see is provided by the `in_use` CAS.
    #[allow(clippy::mut_from_ref)]
    #[inline]
    pub unsafe fn inner_mut(&self) -> &mut SlotInner {
        &mut *self.inner.get()
    }

    /// Shared access to the slot's inner state (read-only peeks like `fh_get_reg`).
    ///
    /// # Safety
    /// The caller must hold this slot; no [`inner_mut`](Self::inner_mut) borrow may be live.
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
        stack_lo2: 0,
        stack_hi2: 0,
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

/// Slots per chunk of the slot pool.
const SLOT_CHUNK: usize = 16;
/// The most chunks the pool grows to.
const MAX_SLOT_CHUNKS: usize = 64;

/// The slot pool, in chunks that are published once and never freed, so the signal path
/// can scan them without locks. Grown by [`ensure_slots`], off the signal path. SeqCst
/// publication keeps a reclaiming writer from missing a chunk a reader has claimed from.
static SLOT_CHUNKS: [AtomicPtr<Slot>; MAX_SLOT_CHUNKS] =
    [const { AtomicPtr::new(ptr::null_mut()) }; MAX_SLOT_CHUNKS];
/// Serializes pool growth; holds the number of slots allocated.
static SLOTS_ALLOCATED: Mutex<usize> = Mutex::new(0);
/// Threads registered with [`thread_registered`].
static REGISTERED_THREADS: AtomicUsize = AtomicUsize::new(0);

/// Serializes writers and guards the retired list. Never taken on the signal path.
static WRITER: Mutex<WriterState> = Mutex::new(WriterState {
    retired: Vec::new(),
});

/// Slots allocated by `init(0)`, before any thread registers.
const DEFAULT_SLOTS: usize = 16;
/// Slots added per registered thread: one for its own walks, one for a walk from a signal
/// handler that interrupts it.
const SLOTS_PER_THREAD: usize = 2;

/// Allocate the initial slot pool: `num_slots`, or a default if 0. Off the signal path.
pub fn init(num_slots: usize) {
    ensure_slots(if num_slots == 0 {
        DEFAULT_SLOTS
    } else {
        num_slots
    });
}

/// Grow the pool for one more thread that runs the unwinder. Off the signal path.
pub fn thread_registered() {
    let threads = REGISTERED_THREADS.fetch_add(1, Ordering::Relaxed) + 1;
    ensure_slots(DEFAULT_SLOTS + SLOTS_PER_THREAD * threads);
}

/// Grow the pool to at least `n` slots (rounded up to whole chunks, up to the maximum).
/// Off the signal path: it allocates the slots' caches.
fn ensure_slots(n: usize) {
    let mut allocated = SLOTS_ALLOCATED.lock().unwrap_or_else(|e| e.into_inner());
    while *allocated < n && *allocated < SLOT_CHUNK * MAX_SLOT_CHUNKS {
        let chunk: Box<[Slot]> = (0..SLOT_CHUNK).map(|_| Slot::new()).collect();
        let first = Box::leak(chunk).as_mut_ptr();
        SLOT_CHUNKS[*allocated / SLOT_CHUNK].store(first, Ordering::SeqCst);
        *allocated += SLOT_CHUNK;
    }
}

#[inline]
fn slot_chunk(i: usize) -> Option<&'static [Slot]> {
    let first = SLOT_CHUNKS.get(i)?.load(Ordering::SeqCst);
    if first.is_null() {
        return None;
    }
    // SAFETY: a published chunk holds SLOT_CHUNK initialized slots and is never freed.
    Some(unsafe { core::slice::from_raw_parts(first, SLOT_CHUNK) })
}

/// Every allocated slot, in index order. Signal-safe.
#[inline]
fn slots() -> impl Iterator<Item = &'static Slot> {
    (0..MAX_SLOT_CHUNKS).map_while(slot_chunk).flatten()
}

/// Claim a free slot with a lock-free CAS scan. Signal-safe. Returns the slot index and
/// the claim nonce the cursor must present on later access (see [`Slot::seq`]).
pub fn claim_slot() -> Option<(usize, u64)> {
    for (i, s) in slots().enumerate() {
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

/// True iff `nonce` is slot `idx`'s current claim nonce (presenting cursor is the live
/// owner, not a stale copy). Signal-safe.
#[inline]
pub fn nonce_matches(idx: usize, nonce: u64) -> bool {
    match slot(idx) {
        Some(s) => s.seq.load(Ordering::Relaxed) == nonce,
        None => false,
    }
}

#[inline]
pub fn slot(idx: usize) -> Option<&'static Slot> {
    slot_chunk(idx / SLOT_CHUNK)?.get(idx % SLOT_CHUNK)
}

/// Release a slot claimed with `nonce`: retire the claim, clear the hazard, mark the
/// slot free. Signal-safe. The release right is taken by CAS-ing `seq` from `nonce`, so
/// exactly one releaser wins; a stale copy / racing double-fini no-ops.
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
        // Clears only the raw pointer; the snapshot is reclaimed by the writer, never here.
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
    slots().any(|s| s.hazard.load(Ordering::SeqCst) == p)
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

/// Run a mutation against a fresh clone of the current module set and publish it.
///
/// `f` receives an [`Unw`] (cloned from the current snapshot, or empty) and adds /
/// removes modules. Off the signal path; allocates.
pub fn with_writer<R>(f: impl FnOnce(&mut Unw) -> R) -> R {
    let mut guard = WRITER.lock().unwrap_or_else(|e| e.into_inner());

    let cur = CURRENT.load(Ordering::Acquire);
    let mut new_unw: Unw = if cur.is_null() {
        UnwinderNative::new()
    } else {
        // SAFETY: cur is the current published snapshot; immutable and alive (we hold the
        // writer lock so no concurrent reclaim of CURRENT; CURRENT is only swapped here).
        unsafe { (*cur).unw.clone() }
    };

    let ret = f(&mut new_unw);

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
    ret
}
