# framehopunwind

A small **C API for stack unwinding backed by [`framehop`](https://github.com/mstange/framehop)**, with **JIT unwind-info registration**, intended to replace the backtrace path of libunwind (and eventually dbghelp) inside [Julia](https://github.com/JuliaLang/julia).

The system libunwind's *local* unwinding is **not async-signal-safe** — `unw_step` goes through `dl_iterate_phdr` (loader lock) and can `malloc`, so a profiler signal that lands while the interrupted thread holds the loader/malloc lock can **deadlock**. This crate sidesteps that: it keeps an eagerly-enumerated, immutable module snapshot and unwinds with framehop's allocation-free engine, so the unwind path takes **no locks, does no allocation, and frees nothing**.

## What it provides

A cursor-based unwinding API mirroring the subset of libunwind Julia uses, plus JIT registration:

| libunwind / dbghelp | framehopunwind |
| --- | --- |
| `unw_getcontext` / `RtlCaptureContext` | `fh_capture_context` |
| signal `ucontext_t` / `CONTEXT` | `fh_context_from_ucontext` |
| `unw_init_local` / `unw_init_local2` | `fh_cursor_init` |
| `unw_get_reg(IP/SP)` + `unw_step` | `fh_step` (output-then-advance) |
| `_U_dyn_register` / `RtlAddFunctionTable` | `fh_register_jit` / `fh_register_jit_auto` |
| `_U_dyn_cancel` / `RtlDeleteFunctionTable` | `fh_deregister_jit` / `fh_deregister_jit_eh_frame` |

See [`include/framehopunwind.h`](include/framehopunwind.h) for the full, documented surface.

## Async-signal-safety

The split is deliberate:

* **Read path — async-signal-safe** (`fh_capture_context`, `fh_context_from_ucontext`, `fh_cursor_init`, `fh_step`, `fh_get_reg`, `fh_cursor_fini`): no heap allocation (framehop `MustNotAllocateDuringUnwind` + a preallocated per-cursor `Cache`), no locks (module set read via a hand-rolled hazard-pointer snapshot, never the writer mutex), no `free` (retired snapshots are reclaimed only on the writer thread). `panic = "abort"` in release prevents unwinding across the FFI boundary.
* **Mutating path — NOT signal-safe** (`fh_init`, `fh_thread_register`, `fh_modules_refresh`, `fh_register_jit*`, `fh_deregister_jit*`): runs off the signal path (startup, dlopen, JIT compile), exactly where Julia already serializes with `jl_profile_atomic`.

The one bounded resource is the cursor **slot pool** (`fh_init(num_slots)`, default 256). If every slot is busy, `fh_cursor_init` returns an error and that sample is skipped — it never blocks or allocates.

## Platform / architecture support

framehop is x86_64 + aarch64 only, and its PE backend is x86_64-only.

| OS | x86_64 | aarch64 | 32-bit |
| --- | --- | --- | --- |
| Linux / FreeBSD (ELF `.eh_frame`) | ✅ | ✅ | ❌ keep libunwind |
| macOS (Mach-O compact + `__eh_frame`) | ✅ | ✅ | n/a |
| Windows (PE `.pdata`/`.xdata`) | ✅ | ❌ keep dbghelp | ❌ |

`fh_supported()` / `FRAMEHOP_SUPPORTED` reports whether the current build can unwind natively; on unsupported targets every entry point is a no-op stub returning failure, so a caller can always link the library and gate at the call site.

**Verified in this repo:** Linux x86_64 is built and unit-tested (eager self-backtrace cross-checked against a frame-pointer walk; JIT `.eh_frame` unwinding of real code; a concurrent register/unwind stress test). aarch64-linux, x86_64-freebsd, macOS (both arches), and x86_64-windows-gnu are cross-compile-checked.

## Design notes

* **Hazard-pointer snapshot** (`src/state.rs`): writers publish a fresh immutable `Snapshot` (cloned framehop `Unwinder` + the mutation) via an atomic pointer; a reader publishes the pointer it is about to use into a global hazard slot and re-checks it is still current, guaranteeing the writer won't free a snapshot under a live reader. Reclamation happens on the writer thread.
* **Owned bytes**: all unwind sections (`.eh_frame`, `__unwind_info`, `.pdata`, …) are *copied* into owned memory at registration, so a later `dlclose`/`munmap` or a Julia free of a JIT buffer cannot dangle a snapshot mid-read.
* **JIT `.eh_frame` mapping** (the correctness crux, verified against framehop 0.16 + gimli 0.33): `base_svma == base_avma == text_lo`, and `eh_frame_svma.start` is set to the **runtime** address of the eh_frame buffer so gimli resolves `DW_EH_PE_pcrel` `pc_begin` correctly.

## Build

```sh
cargo build --release      # produces target/release/libframehopunwind.{a,so}
cargo test                 # host unit/integration tests (Linux x86_64)
```

## Julia integration

The patch in [`julia-framehop-integration.patch`](julia-framehop-integration.patch) is **additive and gated**: with `JL_USE_FRAMEHOP` undefined the Julia build is byte-for-byte unchanged. It is enabled by defining `JL_ENABLE_FRAMEHOP` at build time; `JL_USE_FRAMEHOP` is then derived only for Linux/FreeBSD on x86_64/aarch64 (extend the predicate in `julia_internal.h` as more platforms are validated).

What it changes (Linux/FreeBSD scope):

* `julia_internal.h` — under `JL_USE_FRAMEHOP`, `bt_cursor_t` becomes `fh_cursor`; `bt_context_t` stays `unw_context_t` (== `ucontext_t`), so `jl_to_bt_context`, longjmp simulation, and thread-suspend all keep working, and **libunwind stays linked** for `task.c` context switching and proc-info lookup.
* `stackwalk.c` — `jl_unw_init` converts the `ucontext_t` to framehop registers; `jl_unw_step` calls `fh_step`; a `jl_unw_fini` macro releases the cursor slot at each call site.
* `debuginfo.cpp` — `register_eh_frames` additionally calls `fh_register_jit` (alongside the existing `__register_frame` + `_U_dyn_register`); `deregister_eh_frames` calls `fh_deregister_jit_eh_frame`.
* `init.c` / `threading.c` — `fh_init(0)` at startup and `fh_thread_register()` per thread.
* `dlload.c` — `fh_modules_refresh()` after a successful `dlopen`.

Build glue (implemented in the patch via a `src/Makefile` block): build Julia with

```sh
make -j JULIA_FRAMEHOP=/abs/path/to/framehopunwind
```

which adds `-DJL_ENABLE_FRAMEHOP` + the crate `include/` to `JCPPFLAGS` and links the
crate's `libframehopunwind.so` (with rpath) into `libjulia-internal`. The crate header is
included under `#pragma GCC visibility push(default)` (like `libunwind.h`) so the `fh_*`
references bind to the shared library under Julia's `-fvisibility=hidden`. For a real
release this would become a `LibFramehopUnwind_jll` linked **alongside** `LibUnwind_jll`.

> Note: on x86_64/aarch64 Linux/FreeBSD, libunwind is *not* strictly required once
> backtraces use framehop — `task.c` context-switching uses hand-written asm
> (`JL_TASK_SWITCH_ASM`), and `debuginfo.cpp`'s `unw_get_proc_info_by_ip` is only a
> symbol-lookup fast path with dladdr/LLVM fallbacks. This patch keeps libunwind linked as
> the conservative first step; it could be dropped on these targets in a follow-up.

## License

MIT OR Apache-2.0, matching framehop.
