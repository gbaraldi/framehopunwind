#!/bin/sh
# Build the release cdylib, compile tests/c_smoke.c against the public header, run it —
# the only test exercising the real C ABI (incl. the header's static_asserts). Unix-only.
set -eu
cd "$(dirname "$0")/.."

cargo build --release

CC=${CC:-cc}
out=target/c_smoke
$CC -O2 -g -fno-omit-frame-pointer -Iinclude tests/c_smoke.c -o "$out" \
    -Ltarget/release -lframehopunwind -Wl,-rpath,"$PWD/target/release"
exec "$out"
