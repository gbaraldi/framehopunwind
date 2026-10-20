//! Walk this thread's own stack on Windows through the public C ABI, with a cold rule
//! cache, so that every frame goes through the PE unwinder. The walk must not allocate
//! (the contract's read path), and must reach the OS thread entry in kernel32.
#![cfg(all(windows, target_arch = "x86_64"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};

use windows_sys::Win32::System::LibraryLoader::GetModuleHandleA;
use windows_sys::Win32::System::ProcessStatus::{GetModuleInformation, MODULEINFO};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetCurrentThreadStackLimits};

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
    fn fh_thread_register();
    fn fh_capture_context(ctx: *mut [u64; 5]);
    fn fh_cursor_init_bounds(cur: *mut [u64; 8], ctx: *const [u64; 5], lo: u64, hi: u64) -> i32;
    fn fh_step(cur: *mut [u64; 8], ip: *mut u64, sp: *mut u64) -> i32;
    fn fh_cursor_fini(cur: *mut [u64; 8]);
}

fn without_allocator<T>(f: impl FnOnce() -> T) -> T {
    CHECK_ALLOCATOR.set(true);
    let result = f();
    CHECK_ALLOCATOR.set(false);
    assert_eq!(
        ALLOCATOR_CALLS.load(Ordering::Relaxed),
        0,
        "the walk allocated"
    );
    result
}

fn module_range(name: &core::ffi::CStr) -> Range<u64> {
    unsafe {
        let module = GetModuleHandleA(name.as_ptr().cast());
        assert!(!module.is_null(), "{name:?} is not loaded");
        let mut info: MODULEINFO = core::mem::zeroed();
        let size = core::mem::size_of::<MODULEINFO>() as u32;
        assert_ne!(
            GetModuleInformation(GetCurrentProcess(), module, &mut info, size),
            0
        );
        let base = info.lpBaseOfDll as u64;
        base..base + info.SizeOfImage as u64
    }
}

#[test]
fn cold_walk_reaches_the_thread_entry_without_allocating() {
    assert_eq!(framehopunwind::fh_supported(), 1); // link the Rust library's C exports
    let (mut lo, mut hi) = (0usize, 0usize);
    unsafe {
        assert_eq!(fh_init(1), 0);
        fh_thread_register();
        GetCurrentThreadStackLimits(&mut lo, &mut hi);
    }
    let kernel32 = module_range(c"kernel32.dll");

    let mut ips = [0u64; 64];
    let (mut ctx, mut cur) = ([0u64; 5], [0u64; 8]);
    let (init, n) = without_allocator(|| unsafe {
        fh_capture_context(&mut ctx);
        let init = fh_cursor_init_bounds(&mut cur, &ctx, lo as u64, hi as u64);
        let mut n = 0;
        if init == 0 {
            let (mut ip, mut sp) = (0, 0);
            while n < ips.len() {
                let more = fh_step(&mut cur, &mut ip, &mut sp);
                if ip != 0 {
                    ips[n] = ip;
                    n += 1;
                }
                if more <= 0 {
                    break;
                }
            }
            fh_cursor_fini(&mut cur);
        }
        (init, n)
    });
    assert_eq!(init, 0);
    let ips = &ips[..n];
    assert!(
        ips.iter().any(|ip| kernel32.contains(ip)),
        "the walk stopped before kernel32: {ips:#x?}"
    );
}
