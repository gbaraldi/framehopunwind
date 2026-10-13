//! Linux arm64 signal frames.
//!
//! A signal handler returns into the kernel's trampoline in the vDSO,
//! `__kernel_rt_sigreturn`. Its unwind info, when the vDSO has any, only describes the
//! frame record the kernel links to the interrupted code, which skips the interrupted
//! function and leaves the stack pointer on the signal stack. So, as libgcc does, recognize
//! the trampoline by its instructions and restore the interrupted registers from the
//! `rt_sigframe` the kernel saved at the stack pointer.

use std::sync::OnceLock;

use framehop::aarch64::UnwindRegsAarch64;

/// `mov x8, #139` (`__NR_rt_sigreturn`), then `svc #0`.
const SIGRETURN: [u32; 2] = [0xd280_1168, 0xd400_0001];

const PT_LOAD: u32 = 1;
const PF_X: u32 = 1;

/// The `rt_sigframe` is a 128-byte `siginfo`, then a `ucontext` whose `uc_mcontext` (the
/// `sigcontext`) is at offset 176 and holds `fault_address`, `regs[31]`, `sp` and `pc`.
const SIGCONTEXT: u64 = 128 + 176;
const SIGCONTEXT_REGS: u64 = 8;
const SIGCONTEXT_SP: u64 = 256;
const SIGCONTEXT_PC: u64 = 264;

const _: () = {
    use core::mem::{offset_of, size_of};
    assert!(size_of::<libc::siginfo_t>() == 128);
    assert!(offset_of!(libc::ucontext_t, uc_mcontext) == 176);
    assert!(offset_of!(libc::mcontext_t, regs) == SIGCONTEXT_REGS as usize);
    assert!(offset_of!(libc::mcontext_t, sp) == SIGCONTEXT_SP as usize);
    assert!(offset_of!(libc::mcontext_t, pc) == SIGCONTEXT_PC as usize);
};

/// The trampoline's address, or 0 if it was not found.
static TRAMPOLINE: OnceLock<u64> = OnceLock::new();

/// Find the trampoline in the vDSO. Off the signal path; idempotent.
pub fn init() {
    TRAMPOLINE.get_or_init(|| unsafe { find_trampoline() });
}

/// Scan the vDSO's executable segment for the trampoline's two instructions.
///
/// # Safety
/// Reads the vDSO, which stays mapped for the life of the process.
unsafe fn find_trampoline() -> u64 {
    let ehdr = libc::getauxval(libc::AT_SYSINFO_EHDR) as *const libc::Elf64_Ehdr;
    if ehdr.is_null() {
        return 0;
    }
    let base = ehdr as *const u8;
    let phdrs = core::slice::from_raw_parts(
        base.add((*ehdr).e_phoff as usize) as *const libc::Elf64_Phdr,
        (*ehdr).e_phnum as usize,
    );
    for ph in phdrs {
        if ph.p_type != PT_LOAD || ph.p_flags & PF_X == 0 {
            continue;
        }
        // The kernel maps the vDSO image as it is laid out in the file.
        let code = core::slice::from_raw_parts(
            base.add(ph.p_offset as usize) as *const u32,
            ph.p_filesz as usize / 4,
        );
        if let Some(i) = code.windows(2).position(|w| w == SIGRETURN) {
            return base.add(ph.p_offset as usize + 4 * i) as u64;
        }
    }
    0
}

/// True if `ip` is the trampoline. Async-signal-safe.
#[inline]
pub fn is_trampoline(ip: u64) -> bool {
    TRAMPOLINE.get().is_some_and(|&t| t != 0 && t == ip)
}

/// Restore the interrupted sp, fp and lr from the `rt_sigframe` at `regs.sp()`, and return
/// the interrupted pc. Async-signal-safe.
pub fn restore<F>(regs: &mut UnwindRegsAarch64, read_stack: &mut F) -> Option<u64>
where
    F: FnMut(u64) -> Result<u64, ()>,
{
    let sigcontext = regs.sp().checked_add(SIGCONTEXT)?;
    let mut field = |offset: u64| read_stack(sigcontext + offset).ok();
    let fp = field(SIGCONTEXT_REGS + 29 * 8)?;
    let lr = field(SIGCONTEXT_REGS + 30 * 8)?;
    let sp = field(SIGCONTEXT_SP)?;
    let pc = field(SIGCONTEXT_PC)?;
    regs.set_fp(fp);
    regs.set_lr(lr);
    regs.set_sp(sp);
    Some(pc)
}
