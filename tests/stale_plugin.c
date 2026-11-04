/* A plugin for tests/c_stale_module.c: its exported function calls back into the host
 * through a few frames, so a walk started in the callback has live frames in this object.
 *
 * Built twice, as A with PLUGIN_PAD_MB=64 and as B with PLUGIN_PAD_MB=32. The padding is
 * BSS: it makes A's address range much larger than its unwind tables, and B large enough
 * that, once A is unloaded, the only free range it fits in is A's. The loader places B at
 * the top of A's former range, so A's unwind tables at the bottom stay unmapped while B's
 * code is live inside A's former range. */
#ifndef PLUGIN_PAD_MB
#define PLUGIN_PAD_MB 64
#endif
#define EXPORT __attribute__((visibility("default")))
static char pad[PLUGIN_PAD_MB << 20];
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
