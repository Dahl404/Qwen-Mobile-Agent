#include "nn.c"
#include <stdio.h>
int main(int argc, char **argv) {
    const char *model = argc > 1 ? argv[1]
        : "/data/data/com.termux/files/home/projects/models/qwen3.6:35b:a3b-q4km.gguf.4k";
    setenv("QMA_NO_CL", "1", 1);
    static qma_t m; char err[256];
    if (qma_load(&m, model, err, sizeof(err)) != 0) { fprintf(stderr, "load: %s\n", err); return 1; }
    qma_ecache_arm(&m, (size_t)256 << 20, 4);

    int il = 1, e1 = 5, e2 = 200;
    /* first get: miss */
    const uint8_t *r1 = qma_ecache_get(&m.ecache, il, e1, expert_fetch, &m);
    printf("get1: %s\n", r1 ? "ok" : "NULL");
    /* immediate repeat: must HIT */
    const uint8_t *r2 = qma_ecache_get(&m.ecache, il, e1, expert_fetch, &m);
    printf("get2 (same rec): %s %s\n", r2 ? "ok" : "NULL", r2 == r1 ? "SAME-PTR(hit)" : "DIFF-PTR(miss?)");
    /* different record */
    const uint8_t *r3 = qma_ecache_get(&m.ecache, il, e2, expert_fetch, &m);
    (void)r3;
    /* back to first: should still be resident -> HIT */
    const uint8_t *r4 = qma_ecache_get(&m.ecache, il, e1, expert_fetch, &m);
    printf("get4 (back to e1): %s\n", r4 == r1 ? "SAME-PTR(hit)" : "DIFF-PTR(evicted!)");
    fprintf(stderr, "[ecprof]"); qma_ecache_prof_delta("");
    qma_ecache_teardown(&m);
    return 0;
}
