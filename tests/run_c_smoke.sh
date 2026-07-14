#!/bin/sh
# Build the release cdylib, compile tests/c_smoke.c against the public header, and run
# it — this is the only test that exercises the real C ABI surface (including the
# header's static_asserts under a C compiler). Unix-only; CI runs it on Linux.
set -eu
cd "$(dirname "$0")/.."

cargo build --release

CC=${CC:-cc}
out=target/c_smoke
$CC -O2 -g -fno-omit-frame-pointer -Iinclude tests/c_smoke.c -o "$out" \
    -Ltarget/release -lframehopunwind -Wl,-rpath,"$PWD/target/release"
exec "$out"
