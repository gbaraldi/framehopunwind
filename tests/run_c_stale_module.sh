#!/bin/sh
# Build the release cdylib and run tests/c_stale_module.c: a walk through an object loaded
# into the range of an object unloaded without fh_modules_refresh must not fault. Linux
# only: the test relies on dlclose unmapping the object at once (glibc).
set -eu
cd "$(dirname "$0")/.."

case "$(uname -s)" in
    Linux) ;;
    *) echo "skipping: Linux only"; exit 0 ;;
esac

cargo build --release

CC=${CC:-cc}
out=target/c_stale_module
$CC -O1 -fPIC -shared -fno-omit-frame-pointer -DPLUGIN_PAD_MB=64 tests/stale_plugin.c -o "$out-a.so"
$CC -O1 -fPIC -shared -fno-omit-frame-pointer -DPLUGIN_PAD_MB=32 tests/stale_plugin.c -o "$out-b.so"
$CC -O2 -g -fno-omit-frame-pointer -Iinclude tests/c_stale_module.c -o "$out" \
    -Ltarget/release -lframehopunwind -Wl,-rpath,"$PWD/target/release" -ldl -lpthread
exec "$out" "$PWD/$out-a.so" "$PWD/$out-b.so"
