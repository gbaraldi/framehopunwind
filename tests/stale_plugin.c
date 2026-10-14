/* A plugin for tests/c_stale_module.c: its exported function calls back into the host
 * through a few frames, so a walk started in the callback has live frames in this object.
 *
 * Built twice, as A with PLUGIN_PAD_MB=64 and as B with PLUGIN_PAD_MB=32. The padding
 * makes A's address range much larger than its unwind tables, and B large enough that,
 * once A is unloaded, the only free range it fits in is A's. On ELF the padding is BSS
 * and the loader places B at the top of A's former range; on PE it is read-only data
 * that pushes A's unwind sections high, and both DLLs share a fixed image base. Either
 * way A's unwind tables are unmapped while B's code is live inside A's former range. */
#ifndef PLUGIN_PAD_MB
#define PLUGIN_PAD_MB 64
#endif
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
static const char pad[PLUGIN_PAD_MB << 20] = {1};
#else
#define EXPORT __attribute__((visibility("default")))
static char pad[PLUGIN_PAD_MB << 20];
#endif
EXPORT const char *plugin_pad = pad;

typedef void (*cb_t)(void *);
__attribute__((noinline)) static int inner(cb_t cb, void *arg, int depth)
{
    volatile int local[16];
    local[depth & 15] = depth;
    if (depth > 0)
        return inner(cb, arg, depth - 1) + local[depth & 15];
    cb(arg);
    return local[0];
}
EXPORT int plugin_call(cb_t cb, void *arg) { return inner(cb, arg, 3); }
