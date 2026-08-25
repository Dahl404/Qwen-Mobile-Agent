/* Real-pipeline decode profile: prefill N tokens, decode M tokens with
   QMA_TIMING buckets, dump the accumulated stage breakdown. */
#include "nn.c"
void qma_moe_timers_reset(void);

extern double g_moe_fetched;
static double m_fetched(void) { return g_moe_fetched; }
#include <stdio.h>

int main(int argc, char **argv) {
    const char *model = argc > 1 ? argv[1]
        : "/data/data/com.termux/files/home/projects/models/qwen3.6:35b:a3b-q4km.gguf.4k";
    /* buckets: 0 matmul 1 quant 2 gdn 3 conv 4 attn 5 norm 6 misc 7 tot 8 moe */
    setenv("QMA_NO_CL", "1", 1);
    setenv("QMA_TIMING", "1", 1);
    const int NP = argc > 2 ? atoi(argv[2]) : 256;
    const int ND = argc > 3 ? atoi(argv[3]) : 24;
    const int TH = getenv("THREADS") ? atoi(getenv("THREADS")) : 8;

    static qma_t m;
    char err[256];
    if (qma_load(&m, model, err, sizeof(err)) != 0) { fprintf(stderr, "load: %s\n", err); return 1; }
    runstate_t rs;
    if (runstate_init(&rs, 16384) != 0) return 1;
    qma_prefetch_init(&m);
    const char *ec = getenv("ECACHE_MB");
    if (ec && atoi(ec) > 0)
        qma_ecache_arm(&m, (size_t)atoi(ec) << 20, 8);
    else if (ec) /* ECACHE_MB=0 -> mmap zero-copy mode */
        m.mmap_exps = 1;

    int ids[4096];
    int n = qma_tokenize(&m,
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump! ", ids, 4000);
    while (n < NP) { ids[n] = ids[n % n]; n++; }   /* pad by repetition */

    double t0 = now_s();
    float logits[N_VOCAB];
    if (qma_eval(&m, &rs, ids, n, logits, TH, 4, 0) != 0) return 1;
    double prefill_s = now_s() - t0;
    fprintf(stderr, "[profile] prefill %d tok in %.2f s (%.0f tok/s)\n",
            n, prefill_s, n / prefill_s);

    int next = 0;
    for (int v = 0; v < N_VOCAB; v++) if (logits[v] > logits[next]) next = v;

    qma_moe_timers_reset();
    t0 = now_s();
    for (int d = 0; d < ND; d++) {
        if (qma_eval(&m, &rs, &next, 1, logits, TH, 4, 0) != 0) return 1;
        next = 0;
        for (int v = 0; v < N_VOCAB; v++) if (logits[v] > logits[next]) next = v;
    }
    fprintf(stderr, "[moe] fetch=%.0f ms (%.0f recs) compute=%.0f ms\n",
            g_moe_fetch_us / 1000.0, g_moe_fetched, g_moe_comp_us / 1000.0);
    if (m.ecache.n_slots > 0)
        fprintf(stderr, "[ec] slots=%d hits=%llu misses=%llu ev=%llu spec=%llu\n",
                m.ecache.n_slots,
                (unsigned long long)m.ecache.hits,
                (unsigned long long)m.ecache.misses,
                (unsigned long long)m.ecache.evictions,
                (unsigned long long)m.ecache.spec_issued);
    double dec_s = now_s() - t0;
    fprintf(stderr, "[profile] decode %d tok in %.2f s (%.1f ms/tok, %.1f tok/s)\n",
            ND, dec_s, dec_s / ND * 1e3, ND / dec_s);
    return 0;
}
