`framehop/` contains the source of the published framehop 0.16.0 crate
(MIT OR Apache-2.0), upstream commit
`fceeebab2e9cd77f47d47c63e5c2a6de3d4a88b3`. The upstream integration tests, unused integration-test dependencies,
and release profile are omitted; the published library sources and licenses
are retained.

Local patch: module generations and their rule-cache keys are widened from
16 to 64 bits. The generation allocator uses checked atomic increment so it
never reuses a generation, even on exhaustion. Cache-entry size assertions
are adjusted for the wider key.

This fixes stale-rule collisions when a cursor is retained across more than
65,535 module mutations. Generation assignment stays on the module-mutation
path. No periodic cache sweep or reader-side reset is needed, and no allocation
or freeing is introduced on the unwind path. On 64-bit targets the wider keys
add 8 bytes per x86-64 cache entry (509 entries per cursor); the aarch64
entry size remains 24 bytes due to its existing layout.

Local patch: x86-64 DWARF unwinding steps through signal trampolines. Expression
rules may read memory (through the bounded stack reader), which glibc's
`__restore_rt` needs to locate the interrupted registers. For an FDE whose CIE
marks a signal frame, the generic path is always taken, the stack pointer may
move to another stack, and the result is reported as `Interrupted`, so callers
unwind the interrupted frame from its exact address
(`Unwinder::unwind_frame_signal_aware`). `unwind_frame` is unchanged.

Local patch: the module list is kept in shared chunks of up to 128 modules,
so cloning an unwinder copies chunk pointers and a change copies only the chunk
it touches. framehopunwind publishes a clone on every module change, which made
each JIT registration proportional to the number of modules.

Remove this vendored copy once an upstream release provides equivalent
non-reusing module generations, signal-frame unwinding, and cheap module-list
updates.
