//! Module bookkeeping: enumerating loaded objects, registering JIT code, and publishing
//! new immutable snapshots. All of this runs **off** the signal path.
//!
//! Static modules are tracked as `key -> fingerprint`: `key` is the module's lowest load
//! address (what framehop indexes by) and the fingerprint folds in enough geometry (name,
//! unwind-section address/size, span) to tell two different images apart even when a
//! `dlclose`/`dlopen` pair reuses the same base address — `refresh()` then replaces the
//! module instead of silently serving the old image's unwind data.
//!
//! On macOS the set is maintained by dyld's add/remove-image callbacks (installed once by
//! [`init`]), which run synchronously under dyld's lock on every load and unload —
//! including a replay for already-loaded images at installation time. This avoids the
//! index-based `_dyld_image_count()` / `_dyld_get_image_header(i)` APIs, which can race
//! with a concurrent unload (the header pointer dangles). On Windows, [`init`] enumerates
//! the set once, and the DLL notifications then maintain it. ELF has no unload callback, so
//! [`refresh`] re-derives the set, and unwind tables are copies (see `Bytes`). So does
//! Windows if the notifications cannot be registered.
//!
//! Lock order is always loader -> REGISTRY -> WRITER, on every platform. The loader
//! callbacks (dyld on macOS, DLL notifications on Windows) take REGISTRY under the loader
//! lock. Thus code that holds REGISTRY or WRITER must not wait on the loader: no `dlopen`,
//! `dlclose`, `dl_iterate_phdr`, `LoadLibrary`, `FreeLibrary` or thread creation under
//! either lock. For this reason, `refresh` enumerates before it takes REGISTRY, and applies
//! the result under it. The signal path takes neither lock.

use std::collections::{HashMap, HashSet};
#[cfg(windows)]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, Mutex};

use framehop::{Module, Unwinder};

use crate::state::{self, Bytes};

#[cfg(any(target_os = "linux", target_os = "freebsd"))]
mod elf;
#[cfg(target_os = "macos")]
mod macho;
#[cfg(windows)]
mod pe;

mod jit;

pub use jit::{
    deregister_jit, deregister_jit_eh_frame, register_jit_eh_frame, register_jit_eh_frame_auto,
};

/// Result of enumerating the currently-loaded static modules (ELF/Windows).
#[cfg(not(target_os = "macos"))]
pub struct EnumResult {
    /// `key -> fingerprint` of *all* currently-loaded static modules.
    pub current: HashMap<u64, u64>,
    /// Modules to (re)publish: key not in `known`, or fingerprint differs.
    pub new_modules: Vec<(u64, Module<Bytes>)>,
}

/// Writer-side registry of which module keys are currently published, split by origin.
struct Registry {
    /// `key -> fingerprint` of published static modules.
    static_mods: HashMap<u64, u64>,
    jit_keys: HashSet<u64>,
    /// Maps a JIT module's `.eh_frame` runtime address to its `text_lo` key, so callers
    /// that only retain the eh_frame pointer (e.g. Julia's `deregister_eh_frames`) can
    /// deregister without recomputing the code range.
    jit_by_eh_frame: HashMap<u64, u64>,
    /// The reverse of `jit_by_eh_frame`.
    jit_eh_frame_by_key: HashMap<u64, u64>,
    /// ELF: the loader's load/unload counts at the last scan; see `elf::load_counts`.
    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    load_counts: Option<(u64, u64)>,
    /// Bumped on every change to `static_mods` and on every unload notification, so that
    /// `scan` can find a stale enumeration.
    static_seq: u64,
    /// macOS: add-image events recorded during the initial dyld callback replay, published
    /// in one batch by `macos_end_batch` (one snapshot instead of one per preloaded image).
    #[cfg(target_os = "macos")]
    batching: bool,
    #[cfg(target_os = "macos")]
    pending: Vec<(u64, u64, Module<Bytes>)>,
}

static REGISTRY: LazyLock<Mutex<Registry>> = LazyLock::new(|| {
    Mutex::new(Registry {
        static_mods: HashMap::new(),
        jit_keys: HashSet::new(),
        jit_by_eh_frame: HashMap::new(),
        jit_eh_frame_by_key: HashMap::new(),
        #[cfg(any(target_os = "linux", target_os = "freebsd"))]
        load_counts: None,
        static_seq: 0,
        #[cfg(target_os = "macos")]
        batching: false,
        #[cfg(target_os = "macos")]
        pending: Vec::new(),
    })
});

fn registry() -> std::sync::MutexGuard<'static, Registry> {
    REGISTRY.lock().unwrap_or_else(|e| e.into_inner())
}

/// Windows: a scan has published the set while the DLL notifications were registered.
/// Until then, `refresh` scans: a failed initial scan would otherwise leave the objects
/// loaded before `init` out of the set for good.
#[cfg(windows)]
static SCANNED: AtomicBool = AtomicBool::new(false);

/// Bring up static-module tracking. Idempotent. On macOS, the dyld callbacks replay all
/// already-loaded images when they are installed.
pub fn init() {
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    crate::sigreturn::init();
    #[cfg(target_os = "macos")]
    macho::init_callbacks();
    // Register before the scan, so that no load or unload is missed. The fingerprint
    // reconciles an image that both report. A notification during the scan makes it re-run.
    // Initialize the registry first, so that the callback never runs its first-use setup.
    // If registration fails, enumerations copy unwind tables (see `pe::enumerate`).
    #[cfg(windows)]
    {
        drop(registry());
        pe::register_notifications();
    }
    #[cfg(not(target_os = "macos"))]
    refresh_scan();
}

/// Unregister the Windows DLL notifications, so that no callback runs during process
/// teardown. A thread that `ExitProcess` killed can still own the registry lock there.
/// Off the signal path.
pub fn fini() {
    #[cfg(windows)]
    {
        pe::unregister_notifications();
        SCANNED.store(false, Ordering::Release);
    }
}

/// Re-scan loaded modules (`fh_modules_refresh`). Off the signal path.
#[cfg(not(target_os = "macos"))]
pub fn refresh() {
    #[cfg(windows)]
    if pe::notifications_active() && SCANNED.load(Ordering::Acquire) {
        return;
    }
    refresh_scan();
}

/// Run `scan`, and on Windows record whether the notifications now keep the set current.
#[cfg(not(target_os = "macos"))]
fn refresh_scan() {
    #[cfg(windows)]
    {
        let active = pe::notifications_active();
        if scan() && active {
            SCANNED.store(true, Ordering::Release);
        }
    }
    #[cfg(not(windows))]
    scan();
}

/// Enumerate the loaded modules and publish a snapshot with additions, removals, and
/// changed images (same base, different fingerprint). JIT modules are preserved. A failed
/// enumeration keeps the previous snapshot: an empty result must not read as "everything
/// unloaded". If the static set changed during the enumeration, the result can be stale,
/// so the enumeration runs again. Returns whether the published set is now current.
#[cfg(not(target_os = "macos"))]
fn scan() -> bool {
    // Equal counts mean no object was loaded or unloaded since the last scan. Read them
    // before scanning, so a load during the scan is picked up by the next refresh.
    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    let counts = {
        let counts = elf::load_counts();
        if counts.is_some() && counts == registry().load_counts {
            return true;
        }
        counts
    };
    for _attempt in 0..8 {
        let (known, seq) = {
            let reg = registry();
            (reg.static_mods.clone(), reg.static_seq)
        };
        let res = match enumerate_static(&known) {
            Some(r) => r,
            None => return false,
        };
        let mut reg = registry();
        if reg.static_seq != seq {
            continue; // a callback or another scan changed the set
        }
        apply_scan(&mut reg, res);
        #[cfg(any(target_os = "linux", target_os = "freebsd"))]
        {
            reg.load_counts = counts;
        }
        return true;
    }
    // The set kept changing. Whoever made the last change published it, but objects that
    // only this scan saw are not in the set.
    false
}

/// Apply one enumeration under REGISTRY.
#[cfg(not(target_os = "macos"))]
fn apply_scan(reg: &mut Registry, res: EnumResult) {
    let EnumResult {
        current,
        new_modules,
    } = res;
    let removed: Vec<u64> = reg
        .static_mods
        .keys()
        .filter(|k| !current.contains_key(k))
        .copied()
        .collect();

    // Keys present before and now, but with a different fingerprint: the base address was
    // reused by a different image. Drop the stale module even if the new image yields no
    // replacement (e.g. it has no unwind info).
    let changed: Vec<u64> = reg
        .static_mods
        .iter()
        .filter(|(k, fp)| current.get(k).is_some_and(|nfp| nfp != *fp))
        .map(|(k, _)| *k)
        .collect();

    if new_modules.is_empty() && removed.is_empty() && changed.is_empty() {
        return; // nothing changed
    }

    state::with_writer(|unw| {
        for k in &changed {
            unw.remove_module(*k);
        }
        for (_k, m) in new_modules {
            unw.add_module(m);
        }
        for k in &removed {
            // Don't remove a key that also belongs to a JIT module (keys are distinct in
            // practice, but be defensive).
            if !reg.jit_keys.contains(k) {
                unw.remove_module(*k);
            }
        }
    });

    reg.static_mods = current;
    reg.static_seq += 1;
}

/// macOS: the dyld add/remove-image callbacks installed by [`init`] keep the module set
/// current, so an explicit refresh only needs to make sure they are installed.
#[cfg(target_os = "macos")]
pub fn refresh() {
    macho::init_callbacks();
}

/// Enumerate the loaded static modules for this platform, building framehop modules for
/// any whose `(key, fingerprint)` is not already in `known`. Returns `None` if the
/// enumeration itself failed (as opposed to succeeding with an empty module set).
#[cfg(not(target_os = "macos"))]
fn enumerate_static(known: &HashMap<u64, u64>) -> Option<EnumResult> {
    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    {
        Some(elf::enumerate(known))
    }
    #[cfg(windows)]
    {
        pe::enumerate(known)
    }
    #[cfg(not(any(target_os = "linux", target_os = "freebsd", windows)))]
    {
        let _ = known;
        Some(EnumResult {
            current: HashMap::new(),
            new_modules: Vec::new(),
        })
    }
}

// ---------------------------------------------------------------------------
// Loader-callback plumbing: dyld's add/remove-image callbacks (macho.rs, under dyld's
// lock) and Windows' DLL notifications (pe.rs, with the loader busy). Both publish
// synchronously and never wait for a reader.
// ---------------------------------------------------------------------------

/// Begin batching add-image events (used around the initial dyld callback replay).
#[cfg(target_os = "macos")]
pub(crate) fn macos_begin_batch() {
    registry().batching = true;
}

/// Publish all batched add-image events in a single snapshot.
#[cfg(target_os = "macos")]
pub(crate) fn macos_end_batch() {
    let mut reg = registry();
    reg.batching = false;
    let pending = std::mem::take(&mut reg.pending);
    if pending.is_empty() {
        return;
    }
    for (k, fp, _) in &pending {
        reg.static_mods.insert(*k, *fp);
    }
    reg.static_seq += 1;
    state::with_writer(|unw| {
        for (_k, _fp, m) in pending {
            unw.add_module(m);
        }
    });
}

/// Record one loaded image (dyld add-image callback or its installation replay; Windows
/// loaded notification). `module` is `None` for an image without unwind info, which is
/// still tracked so its unload and any base-address reuse are seen.
#[cfg(any(target_os = "macos", windows))]
pub(crate) fn image_added(key: u64, fp: u64, module: Option<Module<Bytes>>) {
    let mut reg = registry();
    #[cfg(target_os = "macos")]
    if reg.batching {
        if let Some(module) = module {
            reg.pending.push((key, fp, module));
        }
        return;
    }
    let replace = reg.static_mods.contains_key(&key);
    if replace || module.is_some() {
        state::with_writer(|unw| {
            if replace {
                unw.remove_module(key);
            }
            if let Some(module) = module {
                unw.add_module(module);
            }
        });
    }
    reg.static_mods.insert(key, fp);
    reg.static_seq += 1;
}

/// Drop one unloading image (dyld remove-image callback, Windows unloaded notification).
/// The image is still mapped. See `Bytes` for why a walk on an older snapshot stays safe.
#[cfg(any(target_os = "macos", windows))]
pub(crate) fn image_removed(key: u64) {
    let mut reg = registry();
    #[cfg(target_os = "macos")]
    if reg.batching {
        reg.pending.retain(|(k, _, _)| *k != key);
    }
    // Bump even for an image that is not in the set: a running scan can have parsed it
    // before this unload, and must not publish it after the image is unmapped.
    reg.static_seq += 1;
    if reg.static_mods.remove(&key).is_none() {
        return;
    }
    if reg.jit_keys.contains(&key) {
        return; // defensive: never drop a JIT module from the static path
    }
    state::with_writer(|unw| {
        unw.remove_module(key);
    });
}

// ---------------------------------------------------------------------------
// JIT modules.
// ---------------------------------------------------------------------------

/// Add a single already-built JIT module under `key` (== `text_lo` == `avma_range.start`),
/// recording `eh_frame_addr` so it can later be deregistered by eh_frame pointer.
pub(crate) fn add_jit_module(key: u64, eh_frame_addr: u64, module: Module<Bytes>) {
    let mut reg = registry();
    state::with_writer(|unw| {
        // If re-registering the same key, remove the stale one first.
        if reg.jit_keys.contains(&key) {
            unw.remove_module(key);
        }
        unw.add_module(module);
    });
    reg.jit_keys.insert(key);
    // Purge any previous eh_frame mapping for this key so a late deregister with a stale
    // pointer cannot delete the module we are registering now.
    forget_jit_eh_frame(&mut reg, key);
    if let Some(prev_key) = reg.jit_by_eh_frame.insert(eh_frame_addr, key) {
        // That module can no longer be deregistered by this eh_frame address.
        reg.jit_eh_frame_by_key.remove(&prev_key);
    }
    reg.jit_eh_frame_by_key.insert(key, eh_frame_addr);
}

/// Drop `key`'s eh_frame mapping, if it still has one.
fn forget_jit_eh_frame(reg: &mut Registry, key: u64) {
    if let Some(eh_frame_addr) = reg.jit_eh_frame_by_key.remove(&key) {
        if reg.jit_by_eh_frame.get(&eh_frame_addr) == Some(&key) {
            reg.jit_by_eh_frame.remove(&eh_frame_addr);
        }
    }
}

/// Remove a JIT module keyed by `key`, with the registry lock already held. Keeping
/// lookup and removal under one guard closes the lookup->remove TOCTOU window.
fn remove_jit_locked(reg: &mut Registry, key: u64) {
    if !reg.jit_keys.contains(&key) {
        return;
    }
    state::with_writer(|unw| {
        unw.remove_module(key);
    });
    reg.jit_keys.remove(&key);
    forget_jit_eh_frame(reg, key);
}

/// Remove a JIT module keyed by `key` (== the `text_lo` used at registration).
pub(crate) fn remove_jit_module(key: u64) {
    let mut reg = registry();
    remove_jit_locked(&mut reg, key);
}

/// Remove a JIT module by the `.eh_frame` runtime address used at registration.
pub(crate) fn remove_jit_module_by_eh_frame(eh_frame_addr: u64) {
    let mut reg = registry();
    let key = match reg.jit_by_eh_frame.get(&eh_frame_addr) {
        Some(k) => *k,
        None => return,
    };
    remove_jit_locked(&mut reg, key);
}

/// Number of JIT modules currently registered (diagnostic).
pub(crate) fn jit_module_count() -> usize {
    registry().jit_keys.len()
}

/// Cumulative count of failed JIT registrations (diagnostic).
pub(crate) fn jit_register_failures() -> usize {
    jit::JIT_REGISTER_FAILURES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Hash helper for static-module fingerprints.
#[cfg(not(target_os = "macos"))]
pub(crate) fn fingerprint_of(parts: &[u64], name: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    parts.hash(&mut h);
    name.hash(&mut h);
    h.finish()
}
