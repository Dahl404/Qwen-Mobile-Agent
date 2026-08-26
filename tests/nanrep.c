/* nanrep.c — replay the exact layer-34 expert computation that went
 * non-finite in-engine, using the dumped record + activation.
 * bad_expert.bin = [gate IQ2_XS][up IQ2_XS][down IQ2_S] slabs
 * bad_act.bin    = xq[2048] i8 | xd[8] f32 | xs[128] i16
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef uint16_t ggml_half;
#include "../src/qma.h"
#define GGML_FP16_TO_FP32(x) half_to_float(x)
#define GGML_RESTRICT restrict
static inline void get_scale_min_k4(int j, const uint8_t *q,
                                    uint8_t *d, uint8_t *m)
{
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else       { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
                 *m = (q[j + 4] >> 4) | ((q[j + 0] >> 6) << 4); }
}
#include <assert.h>
#include "../src/iq2tables.h"
#include "ref_dequant.inc"

int main(void)
{
    static uint8_t rec[942080];
    static int8_t xq[2048]; static float xd[8]; static int16_t xs[128];

    FILE *f = fopen("bad_expert.bin", "rb");
    if (!f || fread(rec, 1, sizeof(rec), f) != sizeof(rec)) { perror("rec"); return 1; }
    fclose(f);
    f = fopen("bad_act.bin", "rb");
    if (!f || fread(xq, 1, 2048, f) != 2048) { perror("act"); return 1; }
    fread(xd, 4, 8, f);
    fread(xs, 2, 128, f);
    fclose(f);

    extern float qma_q8k_dot(const void *, int, const int8_t *,
                             const float *, const int16_t *, int);
    const size_t ge = 303104, ue = 303104;

    /* gate / up dots */
    float g = qma_q8k_dot(rec, GGML_TYPE_IQ2_XS, xq, xd, xs, N_EMBD);
    float u = qma_q8k_dot(rec + ge, GGML_TYPE_IQ2_XS, xq, xd, xs, N_EMBD);
    printf("gate=%g up=%g\n", g, u);

    /* reference check: dequant row 0 of gate slab (first 256 w) and compare
       against our kernel's per-block contribution by zeroing all but block0 */
    {
        static float y[256];
        memset(y, 0, sizeof(y));
        /* reference dequant of the FIRST superblock */
        dequantize_row_iq2_xs((const block_iq2_xs *)rec, y, QK_K);
        float mx = 0; int nnan = 0;
        for (int i = 0; i < QK_K; i++) {
            if (!isfinite(y[i])) nnan++;
            else if (fabsf(y[i]) > mx) mx = fabsf(y[i]);
        }
        printf("gate blk0 ref-dequant: max=%g nan=%d\n", mx, nnan);
    }

    /* gated activation -> down */
    if (!isfinite(g) || !isfinite(u)) { printf("NaN in gate/up stage\n"); return 0; }
    float gated[N_FF_EXP];
    for (int i = 0; i < N_FF_EXP; i++)
        gated[i] = ((i < 1 ? g : g)) / (1.f + expf(-g)) * u; /* scalar g,u */

    /* proper: quantize the per-row gated vector like the engine does.
       The engine computes gate[]/up[] PER OUTPUT ROW; here we only have the
       scalar sums, so instead test the DOWN slab directly with a synthetic
       well-scaled activation (checks kernel numerics, not engine wiring). */
    static int8_t gq[N_FF_EXP]; static float gd[N_FF_EXP / QK_K];
    static int16_t gs2[N_FF_EXP / 16];
    extern void qma_q8k_quant(const float *, int, int8_t *, float *, int16_t *);
    for (int i = 0; i < N_FF_EXP; i++) gated[i] = ((float)((int)(gated[i] * 1000) % 7) - 3) * 0.01f;
    qma_q8k_quant(gated, N_FF_EXP, gq, gd, gs2);

    const uint8_t *de = rec + ge + ue;
    const size_t dn_row = (size_t)(N_FF_EXP / QK_K) * sizeof(block_iq2_s);
    int nbad_rows = 0, first_bad = -1;
    static float yref[N_FF_EXP];
    for (int r = 0; r < 8 && r < N_EMBD; r++) {   /* sample first 8 rows */
        float s = qma_q8k_dot(de + (size_t)r * dn_row, GGML_TYPE_IQ2_S,
                              gq, gd, gs2, N_FF_EXP);
        if (!isfinite(s)) { nbad_rows++; if (first_bad < 0) first_bad = r; }
    }
    printf("down q8k: %d/8 sampled rows non-finite (first=%d)\n", nbad_rows, first_bad);

    /* reference dequant of down row 0 */
    dequantize_row_iq2_s((const block_iq2_s *)de, yref, N_FF_EXP);
    {
        float mx = 0; int nnan = 0;
        for (int i = 0; i < N_FF_EXP; i++) {
            if (!isfinite(yref[i])) nnan++;
            else if (fabsf(yref[i]) > mx) mx = fabsf(yref[i]);
        }
        printf("down row0 ref-dequant: max=%g nan=%d\n", mx, nnan);
    }
    return 0;
}
