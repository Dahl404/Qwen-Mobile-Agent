/* parity + perf check: i8mm vmmla Q4_K GEMM vs SDOT qma_q8k_dot vs f32 ref */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "qma.h"

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint64_t rs = 0x9e3779b97f4a7c15ULL;
static uint32_t rnd(void) {
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return (uint32_t)(rs >> 32);
}

int main(void) {
    if (!qma_q8k_available()) { printf("SKIP: no dotprod\n"); return 0; }
    if (!qma_q8k_gemm_available()) { printf("SKIP: no i8mm\n"); return 0; }

    const int n_in = 4096, n_out = 512;   /* multiples of QK_K / realistic */
    const int nb = n_in / QK_K;
    const size_t wrow = (size_t)nb * sizeof(block_q4_K);

    block_q4_K *W = malloc((size_t)n_out * wrow);
    float *x = malloc(sizeof(float) * 16 * n_in);
    float *ref = malloc(sizeof(float) * 16 * n_out);   /* f32 dequant ref */
    float *sdot = malloc(sizeof(float) * 16 * n_out);  /* SDOT path */
    float *gemm = malloc(sizeof(float) * 16 * n_out);  /* i8mm path */

    /* random valid Q4_K blocks: nibbles + scales/mins */
    for (int r = 0; r < n_out; r++) {
        uint8_t *row = (uint8_t *)W + (size_t)r * wrow;
        for (size_t b = 0; b < wrow; b++) row[b] = (uint8_t)rnd();
        for (int i = 0; i < nb; i++) {
            block_q4_K *blk = (block_q4_K *)(row + (size_t)i * sizeof(block_q4_K));
            /* sane positive fp16 scales: random RAW fp16 bits would decode
               to ±inf/65504 and swamp the comparison with fp noise */
            blk->d = float_to_half(0.001f + (float)(rnd() % 1000) * 1e-4f);
            blk->dmin = float_to_half(0.0005f + (float)(rnd() % 500) * 1e-4f);
        }
    }
    const int T = 7;   /* odd on purpose: exercises the dup-token tail */
    for (int i = 0; i < T * n_in; i++)
        x[i] = ((double)rnd() / 4294967295.0 - 0.5) * 8.0;

    /* quantize activations once (shared by both int paths) */
    static int8_t xq[16][4096];
    static float xd[16][4096 / QK_K];
    static int16_t xs[16][4096 / 16];
    for (int t = 0; t < T; t++)
        qma_q8k_quant(x + (size_t)t * n_in, n_in, xq[t], xd[t], xs[t]);

    /* f32 reference */
    for (int t = 0; t < T; t++)
        for (int r = 0; r < n_out; r++)
            ref[(size_t)t * n_out + r] =
                dot_q4_K_f32((const block_q4_K *)((const uint8_t *)W + (size_t)r * wrow),
                             x + (size_t)t * n_in, n_in);

    /* SDOT per-row path */
    for (int t = 0; t < T; t++)
        for (int r = 0; r < n_out; r++)
            sdot[(size_t)t * n_out + r] =
                qma_q8k_dot((const uint8_t *)W + (size_t)r * wrow, GGML_TYPE_Q4_K,
                            xq[t], xd[t], xs[t], n_in);

    /* i8mm GEMM: full range in one call, plus a second run split into two
     * calls to exercise chunked dispatch (even ranges per the contract;
     * odd tail rows are the caller's job, as in mm_worker) */
    memset(gemm, 0, sizeof(float) * T * n_out);
    qma_q8k_gemm_q4k((const uint8_t *)W, wrow, 0, n_out,
                     (const int8_t *)xq, (const float *)xd, (const int16_t *)xs,
                     n_in, n_out, T, gemm);
    memset(gemm, 0, sizeof(float) * T * n_out);
    qma_q8k_gemm_q4k((const uint8_t *)W, wrow, 0, 256,
                     (const int8_t *)xq, (const float *)xd, (const int16_t *)xs,
                     n_in, n_out, T, gemm);
    qma_q8k_gemm_q4k((const uint8_t *)W, wrow, 256, 256,
                     (const int8_t *)xq, (const float *)xd, (const int16_t *)xs,
                     n_in, n_out, T, gemm);

    double max_vs_sdot = 0, max_vs_ref = 0;
    for (int i = 0; i < T * n_out; i++) {
        double e1 = fabs(gemm[i] - sdot[i]) / (fabs(sdot[i]) + 1e-3);
        double e2 = fabs(gemm[i] - ref[i]) / (fabs(ref[i]) + 1e-3);
        if (e1 > max_vs_sdot) max_vs_sdot = e1;
        if (e2 > max_vs_ref) max_vs_ref = e2;
    }
    printf("parity: gemm-vs-sdot max_rel=%.2e  gemm-vs-f32 max_rel=%.2e\n",
           max_vs_sdot, max_vs_ref);

    /* perf: GEMV-shaped decode (T=1) vs paired GEMM (T=8), same total MACs */
    const int reps = 20;
    double t0 = now_s();
    volatile float sink = 0;
    for (int k = 0; k < reps; k++)
        for (int r = 0; r < n_out; r++)
            sink += qma_q8k_dot((const uint8_t *)W + (size_t)r * wrow,
                                GGML_TYPE_Q4_K, xq[0], xd[0], xs[0], n_in);
    double t_sdot = now_s() - t0;

    t0 = now_s();
    for (int k = 0; k < reps / 8; k++)   /* 8 tokens per call -> same MACs */
        qma_q8k_gemm_q4k((const uint8_t *)W, wrow, 0, n_out,
                         (const int8_t *)xq, (const float *)xd,
                         (const int16_t *)xs, n_in, n_out, 8, gemm);
    double t_gemm = now_s() - t0;

    printf("perf: %.1f ms SDOT(T=1)x%d vs %.1f ms i8mm(T=8)x%d  -> %.2fx\n",
           t_sdot * 1e3, reps, t_gemm * 1e3, reps / 8,
           t_sdot / t_gemm);

    /* fp32 rescale accumulation differs in order from the SDOT path
       (-ffast-math reassociation); anything <= ~0.5% is reorder noise */
    int fail = (max_vs_sdot > 5e-3);
    printf("%s: q8k gemm parity\n", fail ? "FAIL" : "PASS");
    free(W); free(x); free(ref); free(sdot); free(gemm);
    return fail;
}
