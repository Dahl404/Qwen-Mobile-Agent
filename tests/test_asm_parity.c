/* C-vs-asm kernel parity: quant, q4 dot, q6 dot, gateup.
 * Random data first (fast fail), then REAL model weights (the class of
 * bug that random-data parity missed before). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "qma.h"

/* asm entry points */
extern void asm_qma_q8k_quant(const float *, int, int8_t *, float *, int16_t *);
extern float asm_qma_q8k_dot(const void *, int, const int8_t *,
                             const float *, const int16_t *, int);
extern void asm_qma_q8k_gateup(const void *, const void *, const int8_t *,
                               const float *, const int16_t *, int,
                               float *, float *);

static uint64_t rs = 0x9e3779b97f4a7c15ULL;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 32); }

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9; }

static int fails = 0;

static void cmp_dot(int wtype, const uint8_t *row, const int8_t *xq,
                    const float *xd, const int16_t *xs, int n, const char *tag) {
    double c = ref_q8k_dot(row, wtype, xq, xd, xs, n);
    double a = asm_qma_q8k_dot(row, wtype, xq, xd, xs, n);
    double rel = fabs(a - c) / (fabs(c) + 1e-3);
    if (rel > 1e-3) {
        printf("FAIL %s: c=%.6f asm=%.6f rel=%.2e\n", tag, c, a, rel);
        fails++;
    }
}

int qma_load(qma_t *m, const char *path, char *err, size_t errlen);

int main(int argc, char **argv) {
    const char *model = argc > 1 ? argv[1]
        : "/data/data/com.termux/files/home/projects/models/qwen3.6:35b:a3b-q4km.gguf.4k";

    /* ---------- 1. quantize parity ---------- */
    {
        const int n = 2048;   /* N_EMBD real size */
        float x[n]; static int8_t cq[n], aq[n];
        static float cd[n / QK_K], ad[n / QK_K];
        static int16_t cs[n / 16], as_[n / 16];
        for (int i = 0; i < n; i++) x[i] = ((double)rnd() / 4294967295.0 - 0.5) * 10.0;
        ref_q8k_quant(x, n, cq, cd, cs);
        asm_qma_q8k_quant(x, n, aq, ad, as_);
        int bad = 0;
        for (int i = 0; i < n; i++) if (cq[i] != aq[i]) bad++;
        for (int i = 0; i < n / QK_K; i++) if (cd[i] != ad[i]) { printf("FAIL quant xd[%d]: %.9g vs %.9g\n", i, cd[i], ad[i]); fails++; }
        for (int i = 0; i < n / 16; i++) if (cs[i] != as_[i]) { printf("FAIL quant xsum[%d]: %d vs %d\n", i, cs[i], as_[i]); fails++; }
        if (bad) { printf("FAIL quant: %d/%d lanes differ\n", bad, n); fails++; }
        else printf("ok   quant (%d lanes)\n", n);
    }

    /* ---------- 2. random-block dot/gateup parity ---------- */
    {
        const int nb = 20;                       /* 5120 lanes */
        const size_t rowb = (size_t)nb * sizeof(block_q4_K);
        uint8_t *w4 = malloc(rowb);              /* random bytes */
        for (size_t b = 0; b < rowb; b++) w4[b] = (uint8_t)rnd();
        for (int i = 0; i < nb; i++) {
            block_q4_K *blk = (block_q4_K *)(w4 + i * sizeof(block_q4_K));
            blk->d = float_to_half(0.001f + (float)(rnd() % 1000) * 1e-4f);
            blk->dmin = float_to_half(0.0005f + (float)(rnd() % 500) * 1e-4f);
        }
        const int cols = nb * QK_K;
        float x[cols]; static int8_t xq[cols]; static float xd[nb]; static int16_t xs[nb / 16 * 16];
        for (int i = 0; i < cols; i++) x[i] = ((double)rnd() / 4294967295.0 - 0.5) * 8.0;
        ref_q8k_quant(x, cols, xq, xd, xs);
        memset(xs, 0, nb / 16 * 16 * sizeof(int16_t));
        for (int i = 0; i < cols; i++)
            for (int k = 0; k < 16 && (i / 16 * 16 + 15) < cols; k++) ;
        /* recompute xsum properly */
        for (int blk = 0; blk < nb; blk++)
            for (int g = 0; g < 16; g++) {
                int s = 0;
                for (int l = 0; l < 16; l++) s += xq[(size_t)blk * QK_K + g * 16 + l];
                xs[(size_t)blk * 16 + g] = (int16_t)s;
            }
        cmp_dot(GGML_TYPE_Q4_K, w4, xq, xd, xs, cols, "rand q4");

        /* gateup on adjacent rows: use row0 and row1 */
        float cg, cu, ag, au;
        ref_q8k_gateup(w4, w4 + rowb / 2 > w4 + rowb ? w4 : w4 + sizeof(block_q4_K) * nb / 2,
                       xq, xd, xs, cols, &cg, &cu);
        /* simpler: gate(g)=row0, up(u)=row at half */
        const uint8_t *u = w4 + (size_t)(nb / 2) * sizeof(block_q4_K);
        ref_q8k_gateup(w4, u, xq, xd, xs, cols, &cg, &cu);
        asm_qma_q8k_gateup(w4, u, xq, xd, xs, cols, &ag, &au);
        if (fabs(ag - cg) / (fabs(cg) + 1e-3) > 1e-3 || fabs(au - cu) / (fabs(cu) + 1e-3) > 1e-3) {
            printf("FAIL rand gateup: c=(%.4f,%.4f) asm=(%.4f,%.4f)\n", cg, cu, ag, au);
            fails++;
        } else printf("ok   rand gateup\n");
        free(w4);
    }

    /* ---------- 3. REAL model weights ---------- */
    if (argc > 0) {
        static qma_t m;
        char err[256] = "";
        if (qma_load(&m, model, err, sizeof(err)) == 0) {
            /* find attn layer with Q4_K wq and Q6_K down */
            int L4 = -1, L6 = -1;
            for (int il = 0; il < N_LAYER; il++) {
                if (m.layers[il].t_down_exps == GGML_TYPE_Q6_K && L6 < 0 && m.layers[il].ffn_down_shexp)
                    L6 = il;
                if (m.layers[il].wq && m.layers[il].t_wq == GGML_TYPE_Q4_K && L4 < 0) L4 = il;
            }
            printf("real weights: L4=%d L6=%d\n", L4, L6);
            const int n_in = N_EMBD;
            float x[N_EMBD > 4096 ? N_EMBD : 4096];
            static int8_t xq[N_EMBD]; static float xd[N_EMBD / QK_K]; static int16_t xs[N_EMBD / 16];
            for (int i = 0; i < n_in; i++) x[i] = sinf((float)i * 0.037f) * 3.0f;
            qma_q8k_quant(x, n_in, xq, xd, xs);

            if (L4 >= 0) {
                const uint8_t *W = (const uint8_t *)m.layers[L4].wq;
                size_t wrow = (size_t)(n_in / QK_K) * sizeof(block_q4_K);
                for (int r = 0; r < 8; r++)
                    cmp_dot(GGML_TYPE_Q4_K, W + (size_t)r * wrow, xq, xd, xs, n_in, "real q4");
            }
            if (L6 >= 0) {
                /* shared-expert down is [N_FF_SHEXP -> N_EMBD]... use its layout:
                   down rows are N_EMBD outputs over N_FF_SHEXP inputs; type may be
                   Q4_K or Q6_K. Test whatever type it is with correct row stride. */
                int wtype = (int)m.layers[L6].t_down_shexp;
                int nput = 512;   /* N_FF_SHEXP */
                size_t bs = (wtype == GGML_TYPE_Q4_K) ? sizeof(block_q4_K) : sizeof(block_q6_K);
                const uint8_t *W = (const uint8_t *)m.layers[L6].ffn_down_shexp;
                size_t wrow = (size_t)(nput / QK_K) * bs;
                /* activation over nput lanes: extend x pattern */
                static int8_t xq2[512]; static float xd2[512 / QK_K]; static int16_t xs2[512 / 16];
                float x2[512];
                for (int i = 0; i < nput; i++) x2[i] = cosf((float)i * 0.041f) * 2.5f;
                ref_q8k_quant(x2, nput, xq2, xd2, xs2);
                for (int r = 0; r < 8; r++)
                    cmp_dot(wtype, W + (size_t)r * wrow, xq2, xd2, xs2, nput, "real shexp-down");
            }
        } else {
            printf("WARN model load failed: %s (skipping real-weight section)\n", err);
        }
    }

    /* ---------- 4. perf spot-check ---------- */
    {
        const int nb = 20; const int cols = nb * QK_K;
        size_t rowb = (size_t)nb * sizeof(block_q4_K);
        uint8_t *w4 = malloc(rowb);
        for (size_t b = 0; b < rowb; b++) w4[b] = (uint8_t)rnd();
        for (int i = 0; i < nb; i++) {
            block_q4_K *blk = (block_q4_K *)(w4 + i * sizeof(block_q4_K));
            blk->d = float_to_half(0.01f); blk->dmin = float_to_half(0.005f);
        }
        static int8_t xq[cols]; static float xd[nb]; static int16_t xs[nb / 16 * 16];
        float x[cols];
        for (int i = 0; i < cols; i++) x[i] = sinf((float)i * 0.01f);
        ref_q8k_quant(x, cols, xq, xd, xs);
        for (int blk = 0; blk < nb; blk++)
            for (int g = 0; g < 16; g++) {
                int sum = 0;
                for (int l = 0; l < 16; l++) sum += xq[blk * QK_K + g * 16 + l];
                xs[blk * 16 + g] = (int16_t)sum;
            }
        volatile float sink = 0;
        double t0 = now();
        for (int r = 0; r < 20000; r++) sink += qma_q8k_dot(w4, GGML_TYPE_Q4_K, xq, xd, xs, cols);
        double tc = now() - t0;
        t0 = now();
        for (int r = 0; r < 20000; r++) sink += asm_qma_q8k_dot(w4, GGML_TYPE_Q4_K, xq, xd, xs, cols);
        double ta = now() - t0;
        printf("perf q4 dot: C %.1f ms vs asm %.1f ms (%.2fx)\n", tc * 1e3, ta * 1e3, tc / ta);
        free(w4);
    }

    printf("%s: asm parity (%d failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails != 0;
}
