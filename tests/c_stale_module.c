/* Regression test: a walk through live code in the address range of an object that was
 * unloaded without fh_modules_refresh must not fault.
 *
 * Without an unload callback (Linux, FreeBSD) the unloaded object's module stays in the
 * published set. When the loader maps another object into that range, a walk through its
 * code looks the stale module up. With unwind tables read in place that dereferenced the
 * old object's unmapped pages; with copied tables the lookup is served from the copy and
 * the walk continues. Plugin A is unloaded and plugin B loaded into its range; see
 * stale_plugin.c for how the sizes make that placement happen. */
#ifndef _WIN32
#define _GNU_SOURCE
#endif
#include "framehopunwind.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <dlfcn.h>
#include <link.h>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#endif

typedef void (*cb_t)(void *);
typedef int (*plugin_fn)(cb_t, void *);

static uintptr_t range_lo, range_hi; /* the unloaded object's former range */
static uint64_t stack_lo, stack_hi;
static int walk_rc, walk_frames, walk_in_range;

/* Walk from `ctx`, counting frames and frames inside [range_lo, range_hi). */
static void walk_from(const fh_context *ctx)
{
    fh_cursor cur;
    walk_rc = fh_cursor_init_bounds(&cur, ctx, stack_lo, stack_hi);
    if (walk_rc != 0)
        return;
    walk_frames = 0;
    walk_in_range = 0;
    uint64_t ip = 0, sp = 0;
    for (int i = 0; i < 64; i++) {
        int more = fh_step(&cur, &ip, &sp);
        if (more < 0)
            break;
        if (ip >= range_lo && ip < range_hi)
            walk_in_range++;
        walk_frames++;
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

#ifdef _WIN32
static LONG CALLBACK on_fault(EXCEPTION_POINTERS *ep)
{
    if (ep->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION)
        return EXCEPTION_CONTINUE_SEARCH;
    fprintf(stderr, "FAIL: access violation during the walk at %p\n",
            (void *)ep->ExceptionRecord->ExceptionInformation[1]);
    fflush(stderr);
    ExitProcess(3);
}
static void install_fault_handler(void) { AddVectoredExceptionHandler(1, on_fault); }
static void get_stack_bounds(void)
{
    ULONG_PTR lo, hi;
    GetCurrentThreadStackLimits(&lo, &hi);
    stack_lo = lo;
    stack_hi = hi;
}
static void *load_plugin(const char *path, uintptr_t *lo, uintptr_t *hi)
{
    HMODULE h = LoadLibraryA(path);
    if (!h) {
        fprintf(stderr, "LoadLibrary(%s) failed: %lu\n", path, GetLastError());
        return NULL;
    }
    MODULEINFO mi;
    GetModuleInformation(GetCurrentProcess(), h, &mi, sizeof mi);
    *lo = (uintptr_t)mi.lpBaseOfDll;
    *hi = *lo + mi.SizeOfImage;
    return h;
}
static plugin_fn plugin_entry(void *h) { return (plugin_fn)GetProcAddress((HMODULE)h, "plugin_call"); }
static void unload_plugin(void *h) { FreeLibrary((HMODULE)h); }
#else
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
static void *load_plugin(const char *path, uintptr_t *lo, uintptr_t *hi)
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
    for (int i = 0; i < eh->e_phnum; i++) {
        if (ph[i].p_type != PT_LOAD)
            continue;
        uintptr_t a = lm->l_addr + ph[i].p_vaddr, b = a + ph[i].p_memsz;
        if (a < *lo) *lo = a;
        if (b > *hi) *hi = b;
    }
    return h;
}
static plugin_fn plugin_entry(void *h) { return (plugin_fn)dlsym(h, "plugin_call"); }
static void unload_plugin(void *h) { dlclose(h); }
#endif

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

    uintptr_t lo, hi;
    void *plugin = load_plugin(argv[1], &lo, &hi);
    if (!plugin)
        return 2;
    plugin_fn entry = plugin_entry(plugin);
    fh_modules_refresh();
    range_lo = lo;
    range_hi = hi;
    entry(walk_here, NULL);
    printf("walk through the loaded plugin: rc=%d frames=%d in_plugin=%d\n", walk_rc, walk_frames, walk_in_range);
    if (walk_rc != 0 || walk_in_range < 1) {
        fprintf(stderr, "FAIL: the walk did not pass through the plugin\n");
        return 1;
    }

    /* Unload A without a refresh, as code outside the embedder would, then load B, which
     * the loader places inside A's former range. */
    unload_plugin(plugin);
    uintptr_t blo, bhi;
    void *b = load_plugin(argv[2], &blo, &bhi);
    if (!b)
        return 2;
    printf("A unloaded (no refresh), range %p..%p; B loaded at %p..%p\n", (void *)lo, (void *)hi,
           (void *)blo, (void *)bhi);
    if (blo < lo || bhi > hi) {
        fprintf(stderr, "SKIP: the loader did not place B inside A's former range\n");
        return 0;
    }
    range_lo = blo;
    range_hi = bhi;
    plugin_entry(b)(walk_here, NULL);
    printf("walk through B (stale module A covers it): rc=%d frames=%d in_B=%d\n", walk_rc, walk_frames,
           walk_in_range);
    if (walk_rc != 0 || walk_in_range < 1) {
        fprintf(stderr, "FAIL: the walk did not pass through B\n");
        return 1;
    }
    printf("OK: no fault walking through an object loaded into an unloaded object's range\n");
    return 0;
}
