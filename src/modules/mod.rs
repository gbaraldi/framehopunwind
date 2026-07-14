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
//! with a concurrent unload (the header pointer dangles). On ELF/Windows the set is
//! re-derived by [`refresh`].
//!
//! Lock order is always REGISTRY -> WRITER (never the reverse); neither is ever taken on
//! the signal path.

use std::collections::{HashMap, HashSet};
use std::sync::{LazyLock, Mutex};

use framehop::Module;

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
        #[cfg(target_os = "macos")]
        batching: false,
        #[cfg(target_os = "macos")]
        pending: Vec::new(),
    })
});

fn registry() -> std::sync::MutexGuard<'static, Registry> {
    REGISTRY.lock().unwrap_or_else(|e| e.into_inner())
}

/// Bring up static-module tracking: install the dyld callbacks on macOS (which replays
/// all already-loaded images), or do an initial scan elsewhere. Idempotent.
pub fn init() {
    #[cfg(target_os = "macos")]
    macho::init_callbacks();
    #[cfg(not(target_os = "macos"))]
    refresh();
}

/// Re-scan loaded modules and publish a snapshot reflecting additions, removals, and
/// changed images (same base, different fingerprint). The JIT modules are preserved.
/// If enumeration fails outright (Windows `EnumProcessModules` can), the previous
/// snapshot is kept untouched — an empty result must never be mistaken for "everything
/// was unloaded". Off the signal path.
#[cfg(not(target_os = "macos"))]
pub fn refresh() {
    let mut reg = registry();
    let res = match enumerate_static(&reg.static_mods) {
        Some(r) => r,
        None => return, // enumeration failed; keep the current snapshot
    };
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
// macOS dyld-callback plumbing (called from macho.rs under dyld's lock).
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
    state::with_writer(|unw| {
        for (_k, _fp, m) in pending {
            unw.add_module(m);
        }
    });
}

/// Record one loaded image (dyld add-image callback, or the installation replay).
#[cfg(target_os = "macos")]
pub(crate) fn macos_add_image(key: u64, fp: u64, module: Module<Bytes>) {
    let mut reg = registry();
    if reg.batching {
        reg.pending.push((key, fp, module));
        return;
    }
    let replace = reg.static_mods.contains_key(&key);
    state::with_writer(|unw| {
        if replace {
            unw.remove_module(key);
        }
        unw.add_module(module);
    });
    reg.static_mods.insert(key, fp);
}

/// Drop one unloading image (dyld remove-image callback). The callback runs while the
/// image is still mapped, so a concurrent reader of the *current* snapshot is unaffected
/// (and our snapshots own copies of the unwind bytes anyway).
#[cfg(target_os = "macos")]
pub(crate) fn macos_remove_image(key: u64) {
    let mut reg = registry();
    if reg.batching {
        reg.pending.retain(|(k, _, _)| *k != key);
    }
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
    reg.jit_by_eh_frame.retain(|_, v| *v != key);
    reg.jit_by_eh_frame.insert(eh_frame_addr, key);
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
    reg.jit_by_eh_frame.retain(|_, v| *v != key);
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
