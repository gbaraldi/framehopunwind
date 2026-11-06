/* A walk through live code in the address range of an unloaded object, after the
 * fh_modules_refresh that the contract requires after dlclose (Linux, FreeBSD).
 *
 * Without an unload callback the unloaded object's module stays in the published set until
 * the refresh. If it stayed, a walk through code that the loader maps into its range would
 * look up the stale module and read its unmapped tables. Plugin A is unloaded, the set
 * refreshed, and plugin B loaded into A's former range without a refresh, so no module
 * covers B: the walk through B is unwound by frame pointer and must not touch A's tables.
 * See stale_plugin.c for how the sizes make that placement happen. */
#define _GNU_SOURCE
#include "framehopunwind.h"
#include <dlfcn.h>
#include <errno.h>
#include <link.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

typedef void (*cb_t)(void *);
typedef int (*plugin_fn)(cb_t, void *);

/* The plugin's recursion leaves at least this many of its frames on the stack. */
#define PLUGIN_FRAMES 4

static uintptr_t range_lo, range_hi; /* the plugin walked through */
static uint64_t stack_lo, stack_hi;
static int walk_rc, walk_in_range, walk_after_range;

/* Walk from `ctx`. Counts the frames inside [range_lo, range_hi), and the frames after them
 * (the host's callers). `walk_rc` is the first error from init or a step, else 0. */
static void walk_from(const fh_context *ctx)
{
    fh_cursor cur;
    walk_in_range = 0;
    walk_after_range = 0;
    walk_rc = fh_cursor_init_bounds(&cur, ctx, stack_lo, stack_hi);
    if (walk_rc != 0)
        return;
    uint64_t ip = 0, sp = 0;
    for (int i = 0; i < 64; i++) {
        int more = fh_step(&cur, &ip, &sp);
        if (more < 0) {
            walk_rc = more;
            break;
        }
        if (ip >= range_lo && ip < range_hi)
            walk_in_range++;
        else if (walk_in_range > 0)
            walk_after_range++;
        if (more == 0)
            break;
    }
    fh_cursor_fini(&cur);
}

static void walk_here(void *arg)
{
    (void)arg;
    fh_context ctx;
    fh_capture_context(&ctx);
    walk_from(&ctx);
}

/* True if the last walk passed through all of the plugin's frames and back into the host. */
static int walked_through(const char *what)
{
    printf("walk through %s: rc=%d frames_in=%d frames_after=%d\n", what, walk_rc, walk_in_range,
           walk_after_range);
    if (walk_rc != 0 || walk_in_range < PLUGIN_FRAMES || walk_after_range < 1) {
        fprintf(stderr, "FAIL: the walk did not pass through %s back into the host\n", what);
        return 0;
    }
    return 1;
}

static void on_fault(int sig, siginfo_t *si, void *uc)
{
    (void)sig;
    (void)uc;
    char buf[96];
    int n = snprintf(buf, sizeof buf, "FAIL: SIGSEGV during the walk at %p\n", si->si_addr);
    if (write(2, buf, n) < 0) { }
    _exit(3);
}

static void install_fault_handler(void)
{
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = on_fault;
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);
}

static void get_stack_bounds(void)
{
    pthread_attr_t attr;
    void *addr;
    size_t size;
    pthread_getattr_np(pthread_self(), &attr);
    pthread_attr_getstack(&attr, &addr, &size);
    pthread_attr_destroy(&attr);
    stack_lo = (uint64_t)(uintptr_t)addr;
    stack_hi = stack_lo + size;
}

/* Load `path` and return its handle, address range and `.eh_frame_hdr` address. */
static void *load_plugin(const char *path, uintptr_t *lo, uintptr_t *hi, uintptr_t *eh_hdr)
{
    void *h = dlopen(path, RTLD_NOW);
    if (!h) {
        fprintf(stderr, "dlopen(%s) failed: %s\n", path, dlerror());
        return NULL;
    }
    struct link_map *lm;
    dlinfo(h, RTLD_DI_LINKMAP, &lm);
    const ElfW(Ehdr) *eh = (const ElfW(Ehdr) *)lm->l_addr;
    const ElfW(Phdr) *ph = (const ElfW(Phdr) *)(lm->l_addr + eh->e_phoff);
    *lo = UINTPTR_MAX;
    *hi = 0;
    *eh_hdr = 0;
    for (int i = 0; i < eh->e_phnum; i++) {
        uintptr_t a = lm->l_addr + ph[i].p_vaddr;
        if (ph[i].p_type == PT_GNU_EH_FRAME)
            *eh_hdr = a;
        if (ph[i].p_type != PT_LOAD)
            continue;
        uintptr_t b = a + ph[i].p_memsz;
        if (a < *lo) *lo = a;
        if (b > *hi) *hi = b;
    }
    return h;
}

/* True if no mapping covers the page that holds `addr`. */
static int unmapped(uintptr_t addr)
{
    uintptr_t page = (uintptr_t)sysconf(_SC_PAGESIZE);
    return msync((void *)(addr & ~(page - 1)), page, MS_ASYNC) != 0 && errno == ENOMEM;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s <plugin A> <plugin B>\n", argv[0]);
        return 2;
    }
    install_fault_handler();
    get_stack_bounds();
    if (fh_init(0) != 0) {
        fprintf(stderr, "fh_init failed\n");
        return 2;
    }
    fh_thread_register();

    uintptr_t lo, hi, a_hdr;
    void *a = load_plugin(argv[1], &lo, &hi, &a_hdr);
    if (!a)
        return 2;
    fh_modules_refresh();
    range_lo = lo;
    range_hi = hi;
    ((plugin_fn)dlsym(a, "plugin_call"))(walk_here, NULL);
    if (!walked_through("A"))
        return 1;

    /* Unload A and refresh, as the contract requires, then load B, which the loader places
     * inside A's former range. B is not refreshed in, so no module covers it. */
    dlclose(a);
    fh_modules_refresh();
    uintptr_t blo, bhi, b_hdr;
    void *b = load_plugin(argv[2], &blo, &bhi, &b_hdr);
    if (!b)
        return 2;
    printf("A unloaded and refreshed, range %p..%p; B loaded at %p..%p\n", (void *)lo, (void *)hi,
           (void *)blo, (void *)bhi);
    if (blo < lo || bhi > hi || !unmapped(a_hdr)) {
        printf("SKIP: B is not inside A's former range, or A's unwind tables are mapped\n");
        return 0;
    }
    range_lo = blo;
    range_hi = bhi;
    ((plugin_fn)dlsym(b, "plugin_call"))(walk_here, NULL);
    if (!walked_through("B (no module covers it)"))
        return 1;
    printf("OK: walked through an object loaded into a refreshed-out object's range\n");
    return 0;
}
