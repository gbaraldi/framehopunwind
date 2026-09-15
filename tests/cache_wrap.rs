//! Exercise the public C ABI in a separate process, so other tests cannot change
//! framehop's global generation or claim the single cursor slot.
#![cfg(target_arch = "x86_64")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

struct CheckedAllocator;
thread_local! {
    static CHECK_ALLOCATOR: Cell<bool> = const { Cell::new(false) };
}
static ALLOCATOR_CALLS: AtomicUsize = AtomicUsize::new(0);

fn note_allocator_call() {
    if CHECK_ALLOCATOR.try_with(Cell::get).unwrap_or(false) {
        ALLOCATOR_CALLS.fetch_add(1, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for CheckedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note_allocator_call();
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        note_allocator_call();
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        note_allocator_call();
        System.realloc(ptr, layout, size)
    }
}

#[global_allocator]
static ALLOCATOR: CheckedAllocator = CheckedAllocator;

extern "C" {
    fn fh_init(slots: usize) -> i32;
    fn fh_register_jit(eh: *const u8, len: usize, lo: u64, hi: u64) -> i32;
    fn fh_deregister_jit(lo: u64);
    fn fh_cursor_init_bounds(cur: *mut [u64; 8], ctx: *const [u64; 5], lo: u64, hi: u64) -> i32;
    fn fh_step(cur: *mut [u64; 8], ip: *mut u64, sp: *mut u64) -> i32;
    fn fh_get_reg(cur: *const [u64; 8], ip: *mut u64, sp: *mut u64);
    fn fh_cursor_fini(cur: *mut [u64; 8]);
}

const PC: u64 = 0x10_0000_0000;

fn eh_frame(cfa_offset: u8) -> Vec<u8> {
    let mut bytes = Vec::new();
    // CIE: zR, absolute PC encoding, CFA=rsp+offset, RA at CFA-8.
    bytes.extend_from_slice(&20u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&[
        1, b'z', b'R', 0, 1, 0x78, 16, 1, 0, 0x0c, 7, cfa_offset, 0x90, 1, 0, 0,
    ]);
    bytes.extend_from_slice(&24u32.to_le_bytes());
    bytes.extend_from_slice(&28u32.to_le_bytes());
    bytes.extend_from_slice(&PC.to_le_bytes());
    bytes.extend_from_slice(&0x100u64.to_le_bytes());
    bytes.extend_from_slice(&[0; 8]); // empty FDE augmentation/padding + terminator
    bytes
}

fn without_allocator<T>(f: impl FnOnce() -> T) -> T {
    CHECK_ALLOCATOR.set(true);
    let result = f();
    CHECK_ALLOCATOR.set(false);
    assert_eq!(ALLOCATOR_CALLS.load(Ordering::Relaxed), 0);
    result
}

#[test]
fn active_cursor_survives_generation_wrap_without_stale_rules_or_allocation() {
    assert_eq!(framehopunwind::fh_supported(), 1); // link the Rust library's C exports
    let old = eh_frame(8);
    let new = eh_frame(16);
    let stack = [0x111111u64, 0x222222, 0, 0];
    let sp = stack.as_ptr() as u64;
    let ctx = [PC + 8, sp, 0, 0, 0];
    let mut cur = [0; 8];
    let (mut ip, mut out_sp) = (0, 0);
    unsafe {
        assert_eq!(fh_init(1), 0);
        assert_eq!(fh_register_jit(old.as_ptr(), old.len(), PC, PC + 0x100), 0);
        assert_eq!(fh_cursor_init_bounds(&mut cur, &ctx, sp, sp + 32), 0);
        assert_eq!(fh_step(&mut cur, &mut ip, &mut out_sp), 1);
        fh_get_reg(&cur, &mut ip, &mut out_sp);
        assert_eq!(ip, stack[0]);

        // Keep this slot occupied through every writer sweep in the old design.
        // Each replacement removes and adds a module: 65536 mutations total.
        for _ in 0..32768 {
            assert_eq!(fh_register_jit(new.as_ptr(), new.len(), PC, PC + 0x100), 0);
        }
        fh_get_reg(&cur, &mut ip, &mut out_sp);
        assert_eq!(ip, stack[0]); // the live cursor still owns the old snapshot
        without_allocator(|| fh_cursor_fini(&mut cur));

        for _ in 0..2 {
            let (init, step) = without_allocator(|| {
                let init = fh_cursor_init_bounds(&mut cur, &ctx, sp, sp + 32);
                let step = fh_step(&mut cur, &mut ip, &mut out_sp);
                fh_get_reg(&cur, &mut ip, &mut out_sp);
                fh_cursor_fini(&mut cur);
                (init, step)
            });
            assert_eq!(init, 0);
            assert_eq!(step, 1);
            assert_eq!(ip, stack[1], "reused slot must use the replacement FDE");
            assert_eq!(out_sp, sp + 16);
        }
        fh_deregister_jit(PC);
    }
}
