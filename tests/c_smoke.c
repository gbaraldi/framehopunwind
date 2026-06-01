/* Standalone C smoke test: exercises the framehopunwind C API the way Julia will
 * (capture -> init -> step), through the public header + shared library. */
#include "framehopunwind.h"
#include <stdio.h>
#include <stdint.h>

static int g_n;
static uint64_t g_ips[64];

__attribute__((noinline)) static void do_unwind(void)
{
    fh_context ctx;
    fh_capture_context(&ctx);

    fh_cursor cur;
    int rc = fh_cursor_init(&cur, &ctx);
    if (rc != 0) { fprintf(stderr, "cursor_init failed: %d\n", rc); return; }

    g_n = 0;
    uint64_t ip = 0, sp = 0, last_sp = 0;
    for (int i = 0; i < 64; i++) {
        int more = fh_step(&cur, &ip, &sp);
        if (ip) {
            if (last_sp && !(sp > last_sp)) { fprintf(stderr, "sp not increasing!\n"); }
            last_sp = sp;
            g_ips[g_n++] = ip;
        }
        if (more <= 0) break;
    }
    fh_cursor_fini(&cur);
}

__attribute__((noinline)) static void level_c(void) { do_unwind(); }
__attribute__((noinline)) static void level_b(void) { level_c(); asm volatile("" ::: "memory"); }
__attribute__((noinline)) static void level_a(void) { level_b(); asm volatile("" ::: "memory"); }

int main(void)
{
    printf("fh_supported() = %d\n", fh_supported());
    if (!fh_supported()) { printf("framehop not supported on this target; skipping\n"); return 0; }

    if (fh_init(0) != 0) { fprintf(stderr, "fh_init failed\n"); return 1; }
    fh_thread_register();

    level_a();

    printf("captured %d frames:\n", g_n);
    for (int i = 0; i < g_n; i++) printf("  #%2d  0x%016lx\n", i, (unsigned long)g_ips[i]);

    if (g_n < 4) { fprintf(stderr, "FAIL: too few frames (%d)\n", g_n); return 1; }
    printf("OK: framehopunwind C API produced a %d-frame native backtrace\n", g_n);
    return 0;
}
