//! Per-architecture glue between the C-visible [`FhContext`] register snapshot and
//! framehop's native register / frame types.
//!
//! Only x86_64 and aarch64 are real arches in framehop; on any other arch this module
//! is absent and the crate degrades to "unsupported" at the C API layer.

use framehop::{FrameAddress, UnwindRegsNative};

/// A minimal, `#[repr(C)]`, fixed-size register snapshot that both the
/// current-thread capture path and the `ucontext`/`CONTEXT` extraction path fill.
///
/// Field meaning is architecture dependent (kept as a flat array so the C side does
/// not need arch-specific structs and the size is stable across arches):
///
/// * **x86_64**: `r[0]=rip`, `r[1]=rsp`, `r[2]=rbp`.
/// * **aarch64**: `r[0]=pc`, `r[1]=sp`, `r[2]=fp (x29)`, `r[3]=lr (x30)`.
///
/// `r[4]` is reserved / padding.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FhContext {
    pub r: [u64; 5],
}

impl FhContext {
    #[inline]
    pub fn zeroed() -> Self {
        Self { r: [0; 5] }
    }
}

/// The instruction pointer of the captured frame (used to seed unwinding and as the
/// first reported frame).
#[inline]
pub fn context_ip(ctx: &FhContext) -> u64 {
    ctx.r[0]
}

/// The stack pointer of the captured frame.
#[inline]
pub fn context_sp(ctx: &FhContext) -> u64 {
    ctx.r[1]
}

/// Build framehop's native unwind registers from a captured context.
///
/// `code_ptr_auth_mask` is only meaningful on aarch64 (used to strip pointer
/// authentication bits from return addresses); pass `u64::MAX` for "no stripping".
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn make_regs(ctx: &FhContext, _code_ptr_auth_mask: u64) -> UnwindRegsNative {
    // UnwindRegsX86_64::new(ip, sp, bp)
    framehop::x86_64::UnwindRegsX86_64::new(ctx.r[0], ctx.r[1], ctx.r[2])
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn make_regs(ctx: &FhContext, code_ptr_auth_mask: u64) -> UnwindRegsNative {
    use framehop::aarch64::{PtrAuthMask, UnwindRegsAarch64};
    // r[0]=pc (not stored in regs), r[1]=sp, r[2]=fp, r[3]=lr
    UnwindRegsAarch64::new_with_ptr_auth_mask(
        PtrAuthMask(code_ptr_auth_mask),
        ctx.r[3], // lr
        ctx.r[1], // sp
        ctx.r[2], // fp
    )
}

/// The initial [`FrameAddress`] to start unwinding from (always an instruction pointer).
#[inline]
pub fn initial_frame_address(ctx: &FhContext) -> FrameAddress {
    FrameAddress::from_instruction_pointer(context_ip(ctx))
}
