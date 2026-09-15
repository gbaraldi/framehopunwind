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

Remove this vendored copy once an upstream release provides equivalent
non-reusing module generations.
