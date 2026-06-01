//! JIT unwind-info registration.
//!
//! Replaces Julia's `register_eh_frames` / `_U_dyn_register` (DWARF platforms) and the
//! `RtlAddFunctionTable` path (Windows) for runtime-emitted code.
//!
//! ## The DWARF (`.eh_frame`) mapping — verified against framehop 0.16 + gimli 0.33
//!
//! A JIT object emits a `.eh_frame` (no `.eh_frame_hdr`) into memory at runtime address
//! `eh_frame_ptr`, describing code in `[text_lo, text_hi)`. framehop lands this in its
//! `DwarfCfiIndexAndEhFrame` variant (builds an FDE index at `add_module`). For lookups
//! to line up:
//!
//!   * `base_svma == base_avma == text_lo` — the index stores `fde.initial_address() -
//!     base_svma`, and lookups use `avma - base_avma`; equal bases make these consistent.
//!   * `eh_frame_svma.start == eh_frame_ptr` (the **runtime** address of the buffer) —
//!     gimli resolves `DW_EH_PE_pcrel` pc_begin as `eh_frame.section + offset + disp`,
//!     so the section base must be where the bytes actually live at runtime.
//!
//! We **copy** the `.eh_frame` bytes into owned memory (the copy can live anywhere; only
//! its contents and the absolute `eh_frame_svma.start` matter), so Julia may free its
//! buffer immediately after deregistration.

use core::ops::Range;

use framehop::{ExplicitModuleSectionInfo, Module};

use crate::state::Bytes;

/// Build and register a JIT module from a live `.eh_frame` buffer.
///
/// Returns 0 on success, `<0` on bad arguments.
pub fn register_jit_eh_frame(
    eh_frame_ptr: *const u8,
    eh_frame_len: usize,
    text_lo: u64,
    text_hi: u64,
) -> i32 {
    if eh_frame_ptr.is_null() || eh_frame_len == 0 || text_hi <= text_lo {
        return -1;
    }
    // framehop addresses a module relative to base_avma as a u32, so a single region
    // must be < 4 GiB. (Julia JIT regions are tiny; split if this ever trips.)
    if text_hi - text_lo > u32::MAX as u64 {
        return -2;
    }

    let eh_frame_runtime = eh_frame_ptr as u64;
    // Copy the bytes into owned memory (off the signal path).
    // SAFETY: the caller guarantees [eh_frame_ptr, +len) is readable at call time.
    let eh_frame_copy: Bytes =
        unsafe { core::slice::from_raw_parts(eh_frame_ptr, eh_frame_len) }
            .to_vec()
            .into_boxed_slice();

    let info: ExplicitModuleSectionInfo<Bytes> = ExplicitModuleSectionInfo {
        // base_svma == base_avma == text_lo (see module docs).
        base_svma: text_lo,
        eh_frame: Some(eh_frame_copy),
        // The *runtime* address range of the original buffer drives pcrel resolution.
        eh_frame_svma: Some(Range {
            start: eh_frame_runtime,
            end: eh_frame_runtime + eh_frame_len as u64,
        }),
        ..Default::default()
    };

    let module = Module::new(
        format!("<jit:{text_lo:#x}>"),
        Range {
            start: text_lo,
            end: text_hi,
        },
        text_lo, // base_avma
        info,
    );

    super::add_jit_module(text_lo, eh_frame_runtime, module);
    0
}

/// Register a JIT module from a live `.eh_frame` buffer, deriving the covered code range
/// `[text_lo, text_hi)` from the FDEs themselves (so the caller need not compute it).
/// Returns 0 on success, `<0` on bad arguments or if no FDEs were found.
pub fn register_jit_eh_frame_auto(eh_frame_ptr: *const u8, eh_frame_len: usize) -> i32 {
    if eh_frame_ptr.is_null() || eh_frame_len == 0 {
        return -1;
    }
    let eh_frame_runtime = eh_frame_ptr as u64;
    // SAFETY: caller guarantees [eh_frame_ptr, +len) is readable at call time.
    let bytes = unsafe { core::slice::from_raw_parts(eh_frame_ptr, eh_frame_len) };
    let (text_lo, text_hi) = match fde_pc_range(bytes, eh_frame_runtime) {
        Some(r) => r,
        None => return -3, // no usable FDEs
    };
    register_jit_eh_frame(eh_frame_ptr, eh_frame_len, text_lo, text_hi)
}

/// Compute the min `initial_address` and max `initial_address + address_range` over all
/// FDEs in an `.eh_frame`, resolving pcrel pc_begin against the section's runtime address.
/// Returns `(lo, hi)` AVMAs, or `None` if there are no FDEs.
fn fde_pc_range(eh_frame_bytes: &[u8], eh_frame_runtime: u64) -> Option<(u64, u64)> {
    use gimli::{BaseAddresses, CieOrFde, EhFrame, LittleEndian, UnwindSection};

    let mut eh_frame = EhFrame::new(eh_frame_bytes, LittleEndian);
    // Pin the address size to 8 like framehop does (gimli otherwise defaults to the native
    // word size); this governs DW_EH_PE_absptr widths and the pcrel base mask.
    eh_frame.set_address_size(8);
    // pcrel pc_begin is resolved relative to the section's runtime address.
    let bases = BaseAddresses::default().set_eh_frame(eh_frame_runtime);

    let mut entries = eh_frame.entries(&bases);
    let mut lo = u64::MAX;
    let mut hi = 0u64;
    let mut found = false;
    while let Ok(Some(entry)) = entries.next() {
        if let CieOrFde::Fde(partial) = entry {
            if let Ok(fde) = partial.parse(|_, bases, o| eh_frame.cie_from_offset(bases, o)) {
                let start = fde.initial_address();
                let end = start.wrapping_add(fde.len());
                if start < lo {
                    lo = start;
                }
                if end > hi {
                    hi = end;
                }
                found = true;
            }
        }
    }
    if found && hi > lo {
        Some((lo, hi))
    } else {
        None
    }
}

/// Deregister a JIT module previously registered with `text_lo`.
pub fn deregister_jit(text_lo: u64) {
    super::remove_jit_module(text_lo);
}

/// Deregister a JIT module by the `.eh_frame` runtime address used at registration. This
/// is what Julia's `deregister_eh_frames(Addr, Size)` can call, since it retains `Addr`.
pub fn deregister_jit_eh_frame(eh_frame_ptr: *const u8) {
    super::remove_jit_module_by_eh_frame(eh_frame_ptr as u64);
}
