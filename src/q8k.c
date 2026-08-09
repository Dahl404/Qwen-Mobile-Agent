/* SPDX-License-Identifier: Apache-2.0
 * q8k.c — int8 quantized-activation dot kernels (port of waste gq_neon.c).
 *
 * The scalar dequant-then-fp32 path spends most of its cycles widening
 * weights to fp32. These kernels instead quantize the activation x to int8
 * once per layer (xq + per-256-block scale xd + per-16-lane sums xsum) and
 * compute <dequant(row), x> with NEON SDOT (vdotq_s32) — the llama.cpp
 * production path for Q4_K/Q6_K. Numerics differ from the f32 kernels only
 * by the activation quantization (~0.5-1% relative), which is why the
 * f32 path stays available as the bisection baseline.
 *
 * Requires __ARM_FEATURE_DOTPROD (asimddp). This phone has it.
 */

#include "qma.h"

#if defined(__ARM_FEATURE_DOTPROD)

#include <arm_neon.h>

/* 6-bit scale/min unpack for Q4_K (llama.cpp get_scale_min_k4 indexing). */
static inline void q8k_scale_min_k4(int j, const uint8_t *q,
                                    uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

/* x -> q8 blocks of 256 lanes: xd = amax/127 (1 when x is all zero), xq =
 * clamp(round(x/xd)), xsum = Σ xq per 16 lanes. cols is a multiple of 256. */
static void quantize_row_q8k(const float *x, int cols,
                             int8_t *xq, float *xd, int16_t *xsum)
{
    const int nb = cols / QK_K;
    for (int i = 0; i < nb; i++) {
        const float *xs = x + (size_t)i * QK_K;
        int8_t *q = xq + (size_t)i * QK_K;
        int16_t *sp = xsum + (size_t)i * (QK_K / 16);
        float32x4_t am = vdupq_n_f32(0.0f);
        for (int j = 0; j < QK_K; j += 4)
            am = vmaxq_f32(am, vabsq_f32(vld1q_f32(xs + j)));
        const float amax = vmaxvq_f32(am);
        const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
        xd[i] = s;
        const float inv = 1.0f / s;
        const int32x4_t lo = vdupq_n_s32(-127), hi = vdupq_n_s32(127);
        for (int j = 0; j < QK_K; j += 16) {
            int16x4_t a0, a1;
            {
                const int32x4_t v0 = vmaxq_s32(
                    vminq_s32(vcvtaq_s32_f32(
                                  vmulq_n_f32(vld1q_f32(xs + j + 0), inv)),
                              hi), lo);
                const int32x4_t v1 = vmaxq_s32(
                    vminq_s32(vcvtaq_s32_f32(
                                  vmulq_n_f32(vld1q_f32(xs + j + 4), inv)),
                              hi), lo);
                a0 = vqmovn_s32(v0);
                a1 = vqmovn_s32(v1);
                const int8x8_t b = vqmovn_s16(vcombine_s16(a0, a1));
                vst1_s8(q + j, b);
            }
            {
                const int32x4_t v0 = vmaxq_s32(
                    vminq_s32(vcvtaq_s32_f32(
                                  vmulq_n_f32(vld1q_f32(xs + j + 8), inv)),
                              hi), lo);
                const int32x4_t v1 = vmaxq_s32(
                    vminq_s32(vcvtaq_s32_f32(
                                  vmulq_n_f32(vld1q_f32(xs + j + 12), inv)),
                              hi), lo);
                const int16x4_t b0 = vqmovn_s32(v0);
                const int16x4_t b1 = vqmovn_s32(v1);
                const int8x8_t b = vqmovn_s16(vcombine_s16(b0, b1));
                vst1_s8(q + j + 8, b);
                const int16x8_t full = vcombine_s16(a0, a1);
                const int16x8_t rest = vcombine_s16(b0, b1);
                sp[j / 16] = (int16_t)vaddvq_s16(vaddq_s16(full, rest));
            }
        }
    }
}

/* Q4_K × q8: (d1, m1)/(d2, m2) per 64-lane group; nibble products in f32
 * with d·sc·xd, min bias subtracts dmin·m·xd·Σxq per 32-lane half-group. */
static float dot_q4_K_q8k(const block_q4_K *b, const int8_t *xq,
                          const float *xd, const int16_t *xsum, int cols)
{
    const int nb = cols / QK_K;
    float total = 0.0f;
    for (int i = 0; i < nb; i++) {
        const float d = half_to_float(b[i].d);
        const float mn = half_to_float(b[i].dmin);
        const float xd_blk = xd[i];
        const uint8_t *q = b[i].qs;
        const int8_t *xp = xq + (size_t)i * QK_K;
        const int16_t *sp = xsum + (size_t)i * (QK_K / 16);
        const uint8x16_t loM = vdupq_n_u8(0x0F);
        float32x4_t acc = vdupq_n_f32(0.0f);
        float bias = 0.0f;
        int is = 0;
        for (int g = 0; g < 4; g++) {
            uint8_t s1, m1, s2, m2;
            q8k_scale_min_k4(is + 0, b[i].scales, &s1, &m1);
            q8k_scale_min_k4(is + 1, b[i].scales, &s2, &m2);
            const float d1 = d * s1, mm1 = mn * m1;
            const float d2 = d * s2, mm2 = mn * m2;
            const uint8x16_t q0 = vld1q_u8(q);
            const uint8x16_t q1 = vld1q_u8(q + 16);
            const int8x16_t nib0 = vreinterpretq_s8_u8(vandq_u8(q0, loM));
            const int8x16_t nib1 = vreinterpretq_s8_u8(vandq_u8(q1, loM));
            const int8x16_t hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q0, 4));
            const int8x16_t hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q1, 4));
            int32x4_t pn = vdotq_s32(vdupq_n_s32(0), nib0, vld1q_s8(xp + 0));
            pn = vdotq_s32(pn, nib1, vld1q_s8(xp + 16));
            int32x4_t ph = vdotq_s32(vdupq_n_s32(0), hi0, vld1q_s8(xp + 32));
            ph = vdotq_s32(ph, hi1, vld1q_s8(xp + 48));
            const float sx0 = (float)(sp[0] + sp[1]);
            const float sx1 = (float)(sp[2] + sp[3]);
            acc = vfmaq_n_f32(acc, vcvtq_f32_s32(pn), d1 * xd_blk);
            acc = vfmaq_n_f32(acc, vcvtq_f32_s32(ph), d2 * xd_blk);
            bias += (mm1 * sx0 + mm2 * sx1) * xd_blk;
            q += 32; xp += 64; sp += 4; is += 2;
        }
        total += vaddvq_f32(acc) - bias;
    }
    return total;
}

/* Fused gate+up Q4_K × q8: compute <gate_row, x> and <up_row, x> in ONE
   pass over the shared quantized activation. gate and up rows are the same
   shape (N_EMBD Q4_K blocks) and adjacent in the record (up at +gu_row).
   Fusing halves the xq/xd/xsum traffic and the loop overhead vs two
   dot_q4_K_q8k calls — the expert hot path. silu is applied to the gate
   score before multiplying by up (the caller stores the result back into
   gate[] for the down projection). */
static void dot_q4_K_gateup_q8k(const block_q4_K *g, const block_q4_K *u,
                                const int8_t *xq, const float *xd,
                                const int16_t *xsum, int cols,
                                float *gate_out, float *up_out)
{
    const int nb = cols / QK_K;
    float gtot = 0.0f, utot = 0.0f;
    const uint8x16_t loM = vdupq_n_u8(0x0F);
    for (int i = 0; i < nb; i++) {
        const float gd = half_to_float(g[i].d);
        const float gmn = half_to_float(g[i].dmin);
        const float ud = half_to_float(u[i].d);
        const float umn = half_to_float(u[i].dmin);
        const float xd_blk = xd[i];
        const int16_t *sp = xsum + (size_t)i * (QK_K / 16);
        float32x4_t gacc = vdupq_n_f32(0.0f), uacc = vdupq_n_f32(0.0f);
        float gbias = 0.0f, ubias = 0.0f;
        int is = 0;
        for (int h = 0; h < 4; h++) {  /* 4 x 64-lane groups */
            uint8_t s1, m1, s2, m2;
            q8k_scale_min_k4(is + 0, g[i].scales, &s1, &m1);
            q8k_scale_min_k4(is + 1, g[i].scales, &s2, &m2);
            const float gd1 = gd * s1, gmm1 = gmn * m1;
            const float gd2 = gd * s2, gmm2 = gmn * m2;
            q8k_scale_min_k4(is + 0, u[i].scales, &s1, &m1);
            q8k_scale_min_k4(is + 1, u[i].scales, &s2, &m2);
            const float ud1 = ud * s1, umm1 = umn * m1;
            const float ud2 = ud * s2, umm2 = umn * m2;
            const uint8x16_t g0 = vld1q_u8(g[i].qs + (size_t)h * 32);
            const uint8x16_t g1 = vld1q_u8(g[i].qs + (size_t)h * 32 + 16);
            const uint8x16_t u0 = vld1q_u8(u[i].qs + (size_t)h * 32);
            const uint8x16_t u1 = vld1q_u8(u[i].qs + (size_t)h * 32 + 16);
            const int8x16_t gn0 = vreinterpretq_s8_u8(vandq_u8(g0, loM));
            const int8x16_t gn1 = vreinterpretq_s8_u8(vandq_u8(g1, loM));
            const int8x16_t gh0 = vreinterpretq_s8_u8(vshrq_n_u8(g0, 4));
            const int8x16_t gh1 = vreinterpretq_s8_u8(vshrq_n_u8(g1, 4));
            const int8x16_t un0 = vreinterpretq_s8_u8(vandq_u8(u0, loM));
            const int8x16_t un1 = vreinterpretq_s8_u8(vandq_u8(u1, loM));
            const int8x16_t uh0 = vreinterpretq_s8_u8(vshrq_n_u8(u0, 4));
            const int8x16_t uh1 = vreinterpretq_s8_u8(vshrq_n_u8(u1, 4));
            const int8x16_t xa = vld1q_s8(xq + (size_t)i * QK_K + (size_t)h * 64 + 0);
            const int8x16_t xb = vld1q_s8(xq + (size_t)i * QK_K + (size_t)h * 64 + 16);
            const int8x16_t xc = vld1q_s8(xq + (size_t)i * QK_K + (size_t)h * 64 + 32);
            const int8x16_t xd4 = vld1q_s8(xq + (size_t)i * QK_K + (size_t)h * 64 + 48);
            int32x4_t gpn = vdotq_s32(vdupq_n_s32(0), gn0, xa);
            gpn = vdotq_s32(gpn, gn1, xb);
            int32x4_t gph = vdotq_s32(vdupq_n_s32(0), gh0, xc);
            gph = vdotq_s32(gph, gh1, xd4);
            int32x4_t upn = vdotq_s32(vdupq_n_s32(0), un0, xa);
            upn = vdotq_s32(upn, un1, xb);
            int32x4_t uph = vdotq_s32(vdupq_n_s32(0), uh0, xc);
            uph = vdotq_s32(uph, uh1, xd4);
            const float sx0 = (float)(sp[0] + sp[1]);
            const float sx1 = (float)(sp[2] + sp[3]);
            gacc = vfmaq_n_f32(gacc, vcvtq_f32_s32(gpn), gd1 * xd_blk);
            gacc = vfmaq_n_f32(gacc, vcvtq_f32_s32(gph), gd2 * xd_blk);
            uacc = vfmaq_n_f32(uacc, vcvtq_f32_s32(upn), ud1 * xd_blk);
            uacc = vfmaq_n_f32(uacc, vcvtq_f32_s32(uph), ud2 * xd_blk);
            gbias += (gmm1 * sx0 + gmm2 * sx1) * xd_blk;
            ubias += (umm1 * sx0 + umm2 * sx1) * xd_blk;
            sp += 4; is += 2;
        }
        gtot += vaddvq_f32(gacc) - gbias;
        utot += vaddvq_f32(uacc) - ubias;
    }
    *gate_out = gtot;
    *up_out = utot;
}

/* Q6_K × q8: folded −32 bias, one int32 accumulator per block. */
static float dot_q6_K_q8k(const block_q6_K *b, const int8_t *xq,
                          const float *xd, int cols)
{
    const int nb = cols / QK_K;
    float total = 0.0f;
    for (int i = 0; i < nb; i++) {
        const float d = half_to_float(b[i].d);
        const float xd_blk = xd[i];
        const uint8_t *ql = b[i].ql;
        const uint8_t *qh = b[i].qh;
        const int8_t  *sc = b[i].scales;
        const int8_t *xp = xq + (size_t)i * QK_K;
        const uint8x16_t loM = vdupq_n_u8(0x0F);
        const int8x16_t m32 = vdupq_n_s8(32);
        int32x4_t acc = vdupq_n_s32(0);
        for (int j = 0; j < 2; j++) {
            const uint8x16_t ql0 = vld1q_u8(ql + 0);
            const uint8x16_t ql1 = vld1q_u8(ql + 16);
            const uint8x16_t ql2 = vld1q_u8(ql + 32);
            const uint8x16_t ql3 = vld1q_u8(ql + 48);
            const uint8x16_t qh0 = vld1q_u8(qh + 0);
            const uint8x16_t qh1 = vld1q_u8(qh + 16);
            const uint8x16_t raw[8] = {
                vorrq_u8(vandq_u8(ql0, loM),
                         vshlq_n_u8(vandq_u8(qh0, vdupq_n_u8(3)), 4)),
                vorrq_u8(vandq_u8(ql1, loM),
                         vshlq_n_u8(vandq_u8(qh1, vdupq_n_u8(3)), 4)),
                vorrq_u8(vandq_u8(ql2, loM),
                         vshlq_n_u8(vshrq_n_u8(vandq_u8(qh0, vdupq_n_u8(0xC)), 2), 4)),
                vorrq_u8(vandq_u8(ql3, loM),
                         vshlq_n_u8(vshrq_n_u8(vandq_u8(qh1, vdupq_n_u8(0xC)), 2), 4)),
                vorrq_u8(vshrq_n_u8(ql0, 4),
                         vshlq_n_u8(vshrq_n_u8(vandq_u8(qh0, vdupq_n_u8(0x30)), 4), 4)),
                vorrq_u8(vshrq_n_u8(ql1, 4),
                         vshlq_n_u8(vshrq_n_u8(vandq_u8(qh1, vdupq_n_u8(0x30)), 4), 4)),
                vorrq_u8(vshrq_n_u8(ql2, 4),
                         vshlq_n_u8(vshrq_n_u8(vandq_u8(qh0, vdupq_n_u8(0xC0)), 6), 4)),
                vorrq_u8(vshrq_n_u8(ql3, 4),
                         vshlq_n_u8(vshrq_n_u8(vandq_u8(qh1, vdupq_n_u8(0xC0)), 6), 4)),
            };
            for (int g = 0; g < 8; g++) {
                const int8x16_t w =
                    vsubq_s8(vreinterpretq_s8_u8(raw[g]), m32);
                const int32x4_t p =
                    vdotq_s32(vdupq_n_s32(0), w, vld1q_s8(xp + (size_t)g * 16));
                acc = vmlaq_n_s32(acc, p, (int32_t)sc[g]);
            }
            ql += 64; qh += 32; sc += 8; xp += 128;
        }
        total += d * xd_blk * (float)vaddvq_s32(acc);
    }
    return total;
}

#endif /* __ARM_FEATURE_DOTPROD */

/* ---- public dispatch --------------------------------------------------- */

/* Quantize x (n floats, n multiple of QK_K) for the q8k path. Returns 0
 * and fills xq/xd/xsum on success; returns -1 when the CPU lacks dotprod
 * (caller falls back to the f32 dots). */
int qma_q8k_available(void)
{
#if defined(__ARM_FEATURE_DOTPROD)
    return 1;
#else
    return 0;
#endif
}

void qma_q8k_quant(const float *x, int n, int8_t *xq, float *xd,
                      int16_t *xsum)
{
#if defined(__ARM_FEATURE_DOTPROD)
    quantize_row_q8k(x, n, xq, xd, xsum);
#else
    (void)x; (void)n; (void)xq; (void)xd; (void)xsum;
#endif
}

/* Fused gate+up dispatch (see the static kernel above). */
void qma_q8k_gateup(const void *g, const void *u, const int8_t *xq,
                       const float *xd, const int16_t *xsum, int n,
                       float *gate, float *up)
{
#if defined(__ARM_FEATURE_DOTPROD)
    dot_q4_K_gateup_q8k((const block_q4_K *)g, (const block_q4_K *)u,
                        xq, xd, xsum, n, gate, up);
#else
    (void)g; (void)u; (void)xq; (void)xd; (void)xsum; (void)n;
    *gate = *up = 0.0f;
#endif
}

/* <dequant(row), x> where x is the q8k-quantized activation. */
float qma_q8k_dot(const void *row, int wtype, const int8_t *xq,
                     const float *xd, const int16_t *xsum, int n)
{
#if defined(__ARM_FEATURE_DOTPROD)
    if (wtype == GGML_TYPE_Q4_K)
        return dot_q4_K_q8k((const block_q4_K *)row, xq, xd, xsum, n);
    if (wtype == GGML_TYPE_Q6_K)
        return dot_q6_K_q8k((const block_q6_K *)row, xq, xd, n);
#endif
    (void)row; (void)wtype; (void)xq; (void)xd; (void)xsum; (void)n;
    return 0.0f;
}
