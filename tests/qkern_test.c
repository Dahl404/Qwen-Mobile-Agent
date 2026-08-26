/* qkern_test.c — parity tests for the Q5_K/Q3_K/IQ2_XS/IQ2_S kernels.
 *
 * Ground truth: llama.cpp's reference dequantizers (verbatim in
 * ref_dequant.inc). For random valid blocks and random activations:
 *   1. our dequantizers == ggml reference        (exact, < 1e-6 rel)
 *   2. our f32 dots       == ref dot             (exact)
 *   3. our q8k SDOT dots  == ref dot on q8-reconstructed activations
 *                                                (< 1e-3 rel, int math)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../src/qma.h"
#include "../src/iq2tables.h"

/* --- ggml shims so ref_dequant.inc compiles verbatim --- */
typedef uint16_t ggml_half;
/* qma.h already defines half_to_float; route ggml's macro to it */
#define GGML_FP16_TO_FP32(x) half_to_float(x)
#define GGML_RESTRICT restrict

/* ggml reference scale unpack used by ref_dequant.inc */
static inline void get_scale_min_k4(int j, const uint8_t *q,
                                    uint8_t *d, uint8_t *m)
{
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else       { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
                 *m = (q[j + 4] >> 4) | ((q[j + 0] >> 6) << 4); }
}

#include <assert.h>
#include "ref_dequant.inc"

/* --- ours (compiled from qkerns.c with rename macros, linked in) --- */
extern void our_dq_q5_K(const block_q5_K *x, float *y, int64_t k);
extern void our_dq_q3_K(const block_q3_K *x, float *y, int64_t k);
extern void our_dq_iq2_xs(const block_iq2_xs *x, float *y, int64_t k);
extern void our_dq_iq2_s(const block_iq2_s *x, float *y, int64_t k);

static uint64_t rs = 0x9e3779b97f4a7c15ull;
static uint32_t rnd(void) {
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return (uint32_t)(rs >> 32);
}
static float rndf(float lo, float hi) {
    return lo + (hi - lo) * ((rnd() >> 8) / 16777216.0f);
}

static int nfail = 0;
static void check(int tno, const char *what, double got, double want,
                  double tol)
{
    double err = fabs(got - want) / (fabs(want) > 1 ? fabs(want) : 1);
    if (!(err <= tol)) {
        printf("FAIL t%d %s: got %.8g want %.8g (rel %.3g)\n",
               tno, what, got, want, err);
        nfail++;
    }
}

/* fill one super-block of the given type with random-but-valid bytes */
static void rand_block(int type, void *b)
{
    memset(b, 0, 256); /* structs <= 210 B */
    uint16_t d16 = float_to_half(rndf(0.002f, 0.08f));
    switch (type) {
    case GGML_TYPE_Q5_K: {
        block_q5_K *p = b;
        p->d = d16; p->dmin = float_to_half(rndf(0.0001f, 0.01f));
        for (int i = 0; i < K_SCALE_SIZE; i++) p->scales[i] = rnd() & 63;
        for (int i = 0; i < QK_K/8; i++) p->qh[i] = (uint8_t)rnd();
        for (int i = 0; i < QK_K/2; i++) p->qs[i] = (uint8_t)rnd();
        break;
    }
    case GGML_TYPE_Q3_K: {
        block_q3_K *p = b;
        p->d = d16;
        for (int i = 0; i < 12; i++) p->scales[i] = rnd() & 63;
        for (int i = 0; i < QK_K/8; i++) p->hmask[i] = (uint8_t)rnd();
        for (int i = 0; i < QK_K/4; i++) p->qs[i] = (uint8_t)rnd();
        break;
    }
    case GGML_TYPE_IQ2_XS: {
        block_iq2_xs *p = b;
        p->d = d16;
        for (int i = 0; i < QK_K/32; i++) p->scales[i] = (uint8_t)rnd();
        for (int i = 0; i < QK_K/8; i++) p->qs[i] = rnd() & 0x7fff;
        break;
    }
    case GGML_TYPE_IQ2_S: {
        block_iq2_s *p = b;
        p->d = d16;
        for (int i = 0; i < QK_K/32; i++) p->scales[i] = (uint8_t)rnd();
        for (int i = 0; i < QK_K/4; i++) p->qs[i] = (uint8_t)rnd();
        break;
    }
    }
}


int main(void)
{
    enum { NB = 7 };  /* super-blocks per row (cols = 7*256 = 1792) */
    const int cols = NB * QK_K;

    struct { int type; size_t bs; const char *name; void (*ourdq)(); } ts[] = {
        { GGML_TYPE_Q5_K,   sizeof(block_q5_K),   "q5_K",   (void(*)())our_dq_q5_K },
        { GGML_TYPE_Q3_K,   sizeof(block_q3_K),   "q3_K",   (void(*)())our_dq_q3_K },
        { GGML_TYPE_IQ2_XS, sizeof(block_iq2_xs), "iq2_xs", (void(*)())our_dq_iq2_xs },
        { GGML_TYPE_IQ2_S,  sizeof(block_iq2_s),  "iq2_s",  (void(*)())our_dq_iq2_s },
    };

    uint8_t wbuf[NB * 256] __attribute__((aligned(64)));
    float x[4096], yref[4096], yours[4096];

    for (unsigned t = 0; t < sizeof(ts)/sizeof(ts[0]); t++) {
        const size_t bs = ts[t].bs;
        for (int i = 0; i < NB; i++)
            rand_block(ts[t].type, wbuf + i * bs);
        for (int i = 0; i < cols; i++) x[i] = rndf(-3.f, 3.f);

        /* 1: dequant parity vs ggml reference */
        memset(yref, 0, sizeof(yref));
        switch (ts[t].type) {
        case GGML_TYPE_Q5_K:   dequantize_row_q5_K((void*)wbuf, yref, cols); break;
        case GGML_TYPE_Q3_K:   dequantize_row_q3_K((void*)wbuf, yref, cols); break;
        case GGML_TYPE_IQ2_XS: dequantize_row_iq2_xs((void*)wbuf, yref, cols); break;
        case GGML_TYPE_IQ2_S:  dequantize_row_iq2_s((void*)wbuf, yref, cols); break;
        }
        ts[t].ourdq(wbuf, yours, cols);
        double mxe = 0;
        for (int i = 0; i < cols; i++) {
            double e = fabs(yref[i]-yours[i]) /
                       (fabs(yref[i]) > 1e-30 ? fabs(yref[i]) : 1);
            if (e > mxe) mxe = e;
        }
        if (mxe > 1e-6) { printf("FAIL %s dequant: max rel %.4g\n", ts[t].name, mxe); nfail++; }
        else printf("ok   %s dequant (max rel %.2g)\n", ts[t].name, mxe);

        /* 2: f32 dot parity */
        double sref = 0;
        for (int i = 0; i < cols; i++) sref += (double)yref[i] * x[i];
        float s32 = 0;
        switch (ts[t].type) {
        case GGML_TYPE_Q5_K:   s32 = dot_q5_K_f32((void*)wbuf, x, cols); break;
        case GGML_TYPE_Q3_K:   s32 = dot_q3_K_f32((void*)wbuf, x, cols); break;
        case GGML_TYPE_IQ2_XS: s32 = dot_iq2_xs_f32((void*)wbuf, x, cols); break;
        case GGML_TYPE_IQ2_S:  s32 = dot_iq2_s_f32((void*)wbuf, x, cols); break;
        }
        check(t, "dot_f32", s32, sref, 2e-5);
        printf("     %s dot_f32 %.8g vs %.8g\n", ts[t].name, (double)s32, sref);

        /* 3: q8k SDOT kernel parity on reconstructed activations */
        static int8_t xq[4096]; static float xd[64]; static int16_t xsum[1024];
        qma_q8k_quant(x, cols, xq, xd, xsum);
        /* reconstruct the exact activation the kernel sees */
        float x8[4096];
        for (int i = 0; i < cols; i++)
            x8[i] = (float)xq[i] * xd[i / QK_K];
        double sref8 = 0;
        for (int i = 0; i < cols; i++) sref8 += (double)yref[i] * x8[i];
        float sq8 = qma_q8k_dot(wbuf, ts[t].type, xq, xd, xsum, cols);
        check(t, "dot_q8k", sq8, sref8, 1e-3);
        printf("     %s dot_q8k %.8g vs %.8g\n", ts[t].name, (double)sq8, sref8);
    }

    if (nfail) { printf("%d FAILURES\n", nfail); return 1; }
    printf("ALL PASS\n");
    return 0;
}
