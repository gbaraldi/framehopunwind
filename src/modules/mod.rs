//! Module bookkeeping: enumerating loaded objects, registering JIT code, and publishing
//! new immutable snapshots. All of this runs **off** the signal path.

use std::collections::{HashMap, HashSet};
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

/// Result of enumerating the currently-loaded static modules.
pub struct EnumResult {
    /// Keys (`avma_range.start`) of *all* currently-loaded static modules.
    pub current_keys: HashSet<u64>,
    /// Newly-seen modules (those whose key was not in the `known` set), ready to add.
    pub new_modules: Vec<(u64, Module<Bytes>)>,
}

/// Writer-side registry of which module keys are currently published, split by origin.
struct Registry {
    static_keys: HashSet<u64>,
    jit_keys: HashSet<u64>,
    /// Maps a JIT module's `.eh_frame` runtime address to its `text_lo` key, so callers
    /// that only retain the eh_frame pointer (e.g. Julia's `deregister_eh_frames`) can
    /// deregister without recomputing the code range.
    jit_by_eh_frame: HashMap<u64, u64>,
}

static REGISTRY: LazyLock<Mutex<Registry>> = LazyLock::new(|| {
    Mutex::new(Registry {
        static_keys: HashSet::new(),
        jit_keys: HashSet::new(),
        jit_by_eh_frame: HashMap::new(),
    })
});

fn registry() -> std::sync::MutexGuard<'static, Registry> {
    REGISTRY.lock().unwrap_or_else(|e| e.into_inner())
}

/// Enumerate the loaded static modules for this platform, building framehop modules for
/// any whose key is not already in `known`.
fn enumerate_static(known: &HashSet<u64>) -> EnumResult {
    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
    {
        elf::enumerate(known)
    }
    #[cfg(target_os = "macos")]
    {
        macho::enumerate(known)
    }
    #[cfg(windows)]
    {
        pe::enumerate(known)
    }
    #[cfg(not(any(
        target_os = "linux",
        target_os = "freebsd",
        target_os = "macos",
        windows
    )))]
    {
        let _ = known;
        EnumResult {
            current_keys: HashSet::new(),
            new_modules: Vec::new(),
        }
    }
}

/// Re-scan loaded modules and publish a snapshot reflecting additions/removals. The JIT
/// modules are preserved (only static modules are diffed here). Off the signal path.
pub fn refresh() {
    let mut reg = registry();
    let res = enumerate_static(&reg.static_keys);
    let removed: Vec<u64> = reg
        .static_keys
        .difference(&res.current_keys)
        .copied()
        .collect();

    if res.new_modules.is_empty() && removed.is_empty() {
        return; // nothing changed
    }

    state::with_writer(|unw| {
        for (_k, m) in res.new_modules {
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

    reg.static_keys = res.current_keys;
}

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
    reg.jit_by_eh_frame.insert(eh_frame_addr, key);
}

/// Remove a JIT module keyed by `key` (== the `text_lo` used at registration).
pub(crate) fn remove_jit_module(key: u64) {
    let mut reg = registry();
    if !reg.jit_keys.contains(&key) {
        return;
    }
    state::with_writer(|unw| {
        unw.remove_module(key);
    });
    reg.jit_keys.remove(&key);
    reg.jit_by_eh_frame.retain(|_, v| *v != key);
}

/// Remove a JIT module by the `.eh_frame` runtime address used at registration.
pub(crate) fn remove_jit_module_by_eh_frame(eh_frame_addr: u64) {
    let key = {
        let reg = registry();
        match reg.jit_by_eh_frame.get(&eh_frame_addr) {
            Some(k) => *k,
            None => return,
        }
    };
    remove_jit_module(key);
}
