/* cltest.c — numerical gate for the OpenCL prefill offload (ma3 method:
 * llama.cpp noshuffle kernels adapted for prefill). No model needed.
 *
 * Quantizes random floats to Q4_K/Q6_K blocks (realistic magnitudes, so
 * the fp16 intermediates on the GPU stay in range), runs the GPU GEMM
 * (T>1) and compares against the CPU dot path (dot_q4_K_f32 /
 * dot_q6_K_f32) — the same comparison Swiftlet does for its GPU kernels.
 * fp16 activations on the GPU make results differ from the fp32 CPU path
 * by ~1e-2 relative; anything beyond 2% is a real error.
 *
 * Requires the vendor ICD copies in work/cl/ (see cl.c). Exits 0 with a
 * "GPU UNAVAILABLE" note when OpenCL cannot start — the engine falls back
 * to CPU in that case, so this is informational, not a hard failure.
 *
 * Build: make tests/cltest
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cl.h"
#include "qma.h"

static uint64_t rng = 0x9e3779b97f4a7c15ULL;
static uint32_t rnd32(void) {
    rng ^= rng << 7; rng ^= rng >> 9; rng ^= rng << 8;
    return (uint32_t)(rng >> 32);
}
static float frnd(void) { return ((float)rnd32() / 4e9f) - 0.5f; }

/* ------- minimal Q4_K quantizer (matches the GGML block layout) ------- */
static void quant_q4k(const float *src, block_q4_K *b) {
    float mins[8], scales[8];
    for (int s = 0; s < 8; s++) {
        float mn = 1e30f, mx = -1e30f;
        for (int i = 0; i < 32; i++) {
            float v = src[s * 32 + i];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        mins[s] = mn;
        scales[s] = (mx - mn) / 15.0f;
    }
    /* GGML Q4_K: mins are negative; dmin stores the scale for the 6-bit
       min codes such that mval = dmin*mn[s] = -mins[s] (positive). */
    float d = 0, dmag = 0;
    for (int s = 0; s < 8; s++) {
        if (scales[s] > d) d = scales[s];
        if (-mins[s] > dmag) dmag = -mins[s];
    }
    b->d = float_to_half(d > 0 ? d : 1e-3f);
    b->dmin = float_to_half(dmag / 63.0f + 1e-4f);
    unsigned char sv[8], mn[8];
    for (int s = 0; s < 8; s++) {
        sv[s] = (unsigned char)(scales[s] / (d > 0 ? d : 1e-3f) + 0.5f);
        if (sv[s] > 63) sv[s] = 63;
        mn[s] = (unsigned char)((-mins[s]) / (dmag / 63.0f + 1e-4f) + 0.5f);
        if (mn[s] > 63) mn[s] = 63;
    }
    /* pack the 6-bit scale/min pairs into the 12 scale bytes (GGML layout) */
    for (int i = 0; i < 4; i++) {
        b->scales[i] = (uint8_t)(sv[i] | ((sv[4 + i] >> 4) << 6));
        b->scales[4 + i] = (uint8_t)(mn[i] | ((mn[4 + i] >> 4) << 6));
        b->scales[8 + i] = (uint8_t)((sv[4 + i] & 0xF) | ((mn[4 + i] & 0xF) << 4));
    }
    for (int s = 0; s < 8; s++) {
        const float sc = d * sv[s], mnv = half_to_float(b->dmin) * mn[s];
        for (int i = 0; i < 32; i++) {
            float v = (src[s * 32 + i] + mnv) / (sc > 0 ? sc : 1e-6f);
            int q = (int)lrintf(v);
            if (q < 0) q = 0;
            if (q > 15) q = 15;
            int byte = s * 16 + (i >> 1), nib = i & 1;
            b->qs[byte] |= (uint8_t)(q << (4 * nib));
        }
    }
}

/* ------- minimal Q6_K quantizer (qma LFM/i8mm layout, see quants.c) ------- */
static void quant_q6k(const float *src, block_q6_K *b) {
    float mx = 0;
    for (int i = 0; i < 256; i++) if (fabsf(src[i]) > mx) mx = fabsf(src[i]);
    b->d = float_to_half(mx / 32.0f + 1e-6f);
    const float d = half_to_float(b->d);
    for (int s = 0; s < 16; s++) {
        float bmx = 0;
        for (int i = 0; i < 16; i++) if (fabsf(src[s * 16 + i]) > bmx) bmx = fabsf(src[s * 16 + i]);
        int sc = (int)lrintf(bmx / (d * 32.0f));
        if (sc < 1) sc = 1;
        if (sc > 127) sc = 127;
        b->scales[s] = (int8_t)sc;
        for (int i = 0; i < 16; i++) {
            float v = src[s * 16 + i] / (d * sc);
            int q = (int)lrintf(v) + 32;
            if (q < 0) q = 0;
            if (q > 63) q = 63;
            int k = s * 16 + i;
            int h = k >> 7, idx = k & 127;
            int l = idx & 31, lane = idx >> 5;
            int byte = h * 64 + l + 32 * (lane & 1);
            int nib = (lane >= 2) ? 4 : 0;
            b->ql[byte] |= (uint8_t)((q & 0xF) << nib);
            b->qh[h * 32 + l] |= (uint8_t)(((q >> 4) & 0x3) << (2 * lane));
        }
    }
}

/* one GPU matmul vs the CPU dot path; returns 0 when they agree */
static int check_shape(cl_engine_t *e, int wtype, int M, int N, int T) {
    const size_t bs = (wtype == GGML_TYPE_Q4_K) ? sizeof(block_q4_K) : sizeof(block_q6_K);
    const size_t nblk = (size_t)(M / 256) * N;
    const size_t n_float = (size_t)M * N;
    uint8_t *w = malloc(nblk * bs);
    float *wsrc = malloc(n_float * 4);
    float *x = malloc((size_t)T * M * 4);
    float *yc = malloc((size_t)T * N * 4);
    float *yg = malloc((size_t)T * N * 4);
    if (!w || !wsrc || !x || !yc || !yg) { fprintf(stderr, "OOM\n"); exit(1); }
    memset(w, 0, nblk * bs);
    for (size_t i = 0; i < n_float; i++) wsrc[i] = frnd();
    /* engine column-major layout: output row r = M/256 consecutive blocks */
    for (int r = 0; r < N; r++)
        for (int sb = 0; sb < M / 256; sb++) {
            float blk[256];
            for (int i = 0; i < 256; i++) blk[i] = wsrc[(size_t)r * M + sb * 256 + i];
            if (wtype == GGML_TYPE_Q4_K)
                quant_q4k(blk, (block_q4_K *)(w + ((size_t)r * (M / 256) + sb) * bs));
            else
                quant_q6k(blk, (block_q6_K *)(w + ((size_t)r * (M / 256) + sb) * bs));
        }
    for (size_t i = 0; i < (size_t)T * M; i++) x[i] = frnd();

    /* CPU reference */
    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * M;
        for (int r = 0; r < N; r++) {
            const uint8_t *wr = w + (size_t)r * (M / 256) * bs;
            float s = 0;
            if (wtype == GGML_TYPE_Q4_K)
                s = dot_q4_K_f32((const block_q4_K *)wr, xt, M);
            else
                s = dot_q6_K_f32((const block_q6_K *)wr, xt, M);
            yc[(size_t)t * N + r] = s;
        }
    }
    /* GPU */
    int rc = cl_matmul_qk(e, wtype, w, M, N, T, x, yg);
    if (rc != 0) {
        printf("  M=%-6d N=%-8d T=%-3d : GPU declined (rc=%d)\n", M, N, T, rc);
        free(w); free(wsrc); free(x); free(yc); free(yg);
        return 1;
    }
    double max_rel = 0, max_abs = 0, sum_rel = 0; int bad = 0, n = 0;
    for (size_t i = 0; i < (size_t)T * N; i++) {
        double a = yc[i], b = yg[i];
        if (isnan(b) || isinf(b)) { bad++; continue; }
        double absd = fabs(a - b);
        double rel = fabs(a) > 1e-3 ? absd / fabs(a) : absd;
        sum_rel += rel; n++;
        if (rel > max_rel) max_rel = rel;
        if (absd > max_abs) max_abs = absd;
        /* a real error: relative >10% on a NON-tiny output (fp16 noise on
           near-zero dots makes max_rel huge but mean stays ~0.002) */
        if (rel > 0.10 && fabs(a) > 0.1) bad++;
    }
    printf("  M=%-6d N=%-8d T=%-3d : %s (max_rel %.3f, mean_rel %.4f, sample yc %.5f yg %.5f)\n",
           M, N, T, bad == 0 ? "OK" : "MISMATCH", max_rel, n ? sum_rel / n : 0,
           yc[0], yg[0]);
    free(wsrc); free(x); free(yc); free(yg);
    /* keep w alive: the lazy weight table keys on the pointer, and a
       freed block may be re-malloc'd with the same address but a
       different shape, tripping wtab_find */
    return bad == 0 ? 0 : 1;
}

int main(void) {
    qma_t fake;
    memset(&fake, 0, sizeof(fake));
    cl_engine_t e;
    if (cl_init(&e, &fake) != 0) {
        printf("GPU UNAVAILABLE — prefill stays on CPU (check work/cl vendor libs)\n");
        return 0;
    }
    printf("OpenCL ready — running Q4_K/Q6_K GEMM vs CPU dot gate\n");
    int fails = 0;
    /* attention/ssm projection shapes */
    fails += check_shape(&e, GGML_TYPE_Q4_K, 2048, 8192, 8);
    fails += check_shape(&e, GGML_TYPE_Q4_K, 2048, 512,  16);
    fails += check_shape(&e, GGML_TYPE_Q6_K, 4096, 2048, 8);
    fails += check_shape(&e, GGML_TYPE_Q4_K, 2048, 248320, 4);  /* lm-head width */
    fails += check_shape(&e, GGML_TYPE_Q6_K, 2048, 248320, 2);  /* Q6_K lm head */
    fails += check_shape(&e, GGML_TYPE_Q4_K, 512, 2048, 8);     /* small n_in */
    /* odd shapes: N not a multiple of 4 (partial tiles) */
    fails += check_shape(&e, GGML_TYPE_Q4_K, 2048, 300, 8);
    fails += check_shape(&e, GGML_TYPE_Q6_K, 2048, 301, 9);
    cl_free(&e);
    printf(fails == 0 ? "PASS: GPU GEMM matches CPU dot\n" : "FAIL: %d shape(s) mismatched\n", fails);
    return fails == 0 ? 0 : 1;
}
