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

#ifdef __ARM_FEATURE_MATMUL_INT8

/* i8mm GEMM core: one 64-lane group of a Q4_K super-block, one row PAIR,
 * one token PAIR. vmmlaq_s32 computes a 2x2 tile (2 weight rows x 2 token
 * columns) per instruction, so unlike the decode-side SDOT path every MAC
 * slot does useful work. Requires T >= 2 for the second column to be real.
 *
 * Q4_K value = d*sc*q - dmin*m per 32-lane half-group; the int32 dot is
 * accumulated raw and rescaled per group, the min bias folds through the
 * per-16-lane activation sums (xsum) exactly like dot_q4_K_q8k. */
static inline void q4k_group_2x2(
    int g,
    const uint8_t *qa, const uint8_t *qb,   /* 32 qs bytes of this group */
    const uint8_t *sa_, const uint8_t *sb_, /* 12 scale bytes of this block */
    float d_a, float mn_a, float d_b, float mn_b,
    float xd_a, float xd_b,                 /* activation block scales */
    const int16_t *spa, const int16_t *spb, /* per-16-lane sums, this block */
    const int8_t *xa, const int8_t *xb,     /* 64 activation lanes, both toks */
    float acc[2][2])                        /* [row][tok] fp32 running sum */
{
    const uint8x16_t loM = vdupq_n_u8(0x0F);
    const uint8x16_t qa0 = vld1q_u8(qa);
    const uint8x16_t qa1 = vld1q_u8(qa + 16);
    const uint8x16_t qb0 = vld1q_u8(qb);
    const uint8x16_t qb1 = vld1q_u8(qb + 16);
    const int8x16_t ga0 = vreinterpretq_s8_u8(vandq_u8(qa0, loM));
    const int8x16_t ga1 = vreinterpretq_s8_u8(vandq_u8(qa1, loM));
    const int8x16_t gb0 = vreinterpretq_s8_u8(vandq_u8(qb0, loM));
    const int8x16_t gb1 = vreinterpretq_s8_u8(vandq_u8(qb1, loM));
    const int8x16_t ha0 = vreinterpretq_s8_u8(vshrq_n_u8(qa0, 4));
    const int8x16_t ha1 = vreinterpretq_s8_u8(vshrq_n_u8(qa1, 4));
    const int8x16_t hb0 = vreinterpretq_s8_u8(vshrq_n_u8(qb0, 4));
    const int8x16_t hb1 = vreinterpretq_s8_u8(vshrq_n_u8(qb1, 4));

    /* x operand pairs: [8][2] tiles are COLUMN-contiguous — B[k][j] sits
       at byte j*8+k (probed empirically): low half = token A slice,
       high half = token B slice. NOT an interleave/zip. */
    const int8x16_t xa0 = vld1q_s8(xa), xa1 = vld1q_s8(xa + 16);
    const int8x16_t xa2 = vld1q_s8(xa + 32), xa3 = vld1q_s8(xa + 48);
    const int8x16_t xb0 = vld1q_s8(xb), xb1 = vld1q_s8(xb + 16);
    const int8x16_t xb2 = vld1q_s8(xb + 32), xb3 = vld1q_s8(xb + 48);
    const int8x16_t c00 = vcombine_s8(vget_low_s8(xa0), vget_low_s8(xb0));
    const int8x16_t c01 = vcombine_s8(vget_high_s8(xa0), vget_high_s8(xb0));
    const int8x16_t c10 = vcombine_s8(vget_low_s8(xa1), vget_low_s8(xb1));
    const int8x16_t c11 = vcombine_s8(vget_high_s8(xa1), vget_high_s8(xb1));
    const int8x16_t c20 = vcombine_s8(vget_low_s8(xa2), vget_low_s8(xb2));
    const int8x16_t c21 = vcombine_s8(vget_high_s8(xa2), vget_high_s8(xb2));
    const int8x16_t c30 = vcombine_s8(vget_low_s8(xa3), vget_low_s8(xb3));
    const int8x16_t c31 = vcombine_s8(vget_high_s8(xa3), vget_high_s8(xb3));

    /* weight row pairs: [2][8] tiles = row a slice | row b slice */
    const int8x16_t a00 = vcombine_s8(vget_low_s8(ga0), vget_low_s8(gb0));
    const int8x16_t a01 = vcombine_s8(vget_high_s8(ga0), vget_high_s8(gb0));
    const int8x16_t a10 = vcombine_s8(vget_low_s8(ga1), vget_low_s8(gb1));
    const int8x16_t a11 = vcombine_s8(vget_high_s8(ga1), vget_high_s8(gb1));
    const int8x16_t a20 = vcombine_s8(vget_low_s8(ha0), vget_low_s8(hb0));
    const int8x16_t a21 = vcombine_s8(vget_high_s8(ha0), vget_high_s8(hb0));
    const int8x16_t a30 = vcombine_s8(vget_low_s8(ha1), vget_low_s8(hb1));
    const int8x16_t a31 = vcombine_s8(vget_high_s8(ha1), vget_high_s8(hb1));

    /* lanes 0..31 -> scale s1, lanes 32..63 -> scale s2 (per row) */
    uint8_t s1a, m1a, s2a, m2a, s1b, m1b, s2b, m2b;
    q8k_scale_min_k4(2 * g + 0, sa_, &s1a, &m1a);
    q8k_scale_min_k4(2 * g + 1, sa_, &s2a, &m2a);
    q8k_scale_min_k4(2 * g + 0, sb_, &s1b, &m1b);
    q8k_scale_min_k4(2 * g + 1, sb_, &s2b, &m2b);

    int32x4_t p1 = vdupq_n_s32(0), p2 = vdupq_n_s32(0);
    p1 = vmmlaq_s32(p1, a00, c00);
    p1 = vmmlaq_s32(p1, a01, c01);
    p1 = vmmlaq_s32(p1, a10, c10);
    p1 = vmmlaq_s32(p1, a11, c11);
    p2 = vmmlaq_s32(p2, a20, c20);
    p2 = vmmlaq_s32(p2, a21, c21);
    p2 = vmmlaq_s32(p2, a30, c30);
    p2 = vmmlaq_s32(p2, a31, c31);

    const float d1a = d_a * s1a, mm1a = mn_a * m1a;
    const float d2a = d_a * s2a, mm2a = mn_a * m2a;
    const float d1b = d_b * s1b, mm1b = mn_b * m1b;
    const float d2b = d_b * s2b, mm2b = mn_b * m2b;
    const float ax1 = (float)(spa[0] + spa[1]), ax2 = (float)(spa[2] + spa[3]);
    const float bx1 = (float)(spb[0] + spb[1]), bx2 = (float)(spb[2] + spb[3]);

    /* coefficient matrix: rows a/b scales x tokens a/b activation scales */
    acc[0][0] += d1a * xd_a * (float)vgetq_lane_s32(p1, 0)
               + d2a * xd_a * (float)vgetq_lane_s32(p2, 0)
               - (mm1a * ax1 + mm2a * ax2) * xd_a;
    acc[1][0] += d1b * xd_a * (float)vgetq_lane_s32(p1, 2)
               + d2b * xd_a * (float)vgetq_lane_s32(p2, 2)
               - (mm1b * ax1 + mm2b * ax2) * xd_a;
    acc[0][1] += d1a * xd_b * (float)vgetq_lane_s32(p1, 1)
               + d2a * xd_b * (float)vgetq_lane_s32(p2, 1)
               - (mm1a * bx1 + mm2a * bx2) * xd_b;
    acc[1][1] += d1b * xd_b * (float)vgetq_lane_s32(p1, 3)
               + d2b * xd_b * (float)vgetq_lane_s32(p2, 3)
               - (mm1b * bx1 + mm2b * bx2) * xd_b;
}
/* i8mm GEMM for Q4_K x q8 activations, T >= 2 tokens at once:
 * out[t*n_out + r] = <dequant(W row r), x_t> for rows [r0, r0+nrows),
 * nrows even (the caller handles an odd tail row via the dot path).
 * Token loop is outermost so an odd final token just reuses column B.
 * ~2x instruction efficiency vs the per-row SDOT GEMV at T > 1; at
 * T == 1 half of every vmmla tile would be wasted, so decode stays on
 * qma_q8k_dot. */
void qma_q8k_gemm_q4k(const uint8_t *W, size_t wrow, int r0, int nrows,
                      const int8_t *xq, const float *xd, const int16_t *xsum,
                      int n_in, int n_out, int T, float *out)
{
    const int nb = n_in / QK_K;
    const int rend = r0 + nrows;

    for (int t0 = 0; t0 < T; t0 += 2) {
        const int t1 = (t0 + 1 < T) ? t0 + 1 : t0;
        const int8_t *xab = xq + (size_t)t0 * n_in;
        const int8_t *xbb = xq + (size_t)t1 * n_in;
        const float *xda = xd + (size_t)t0 * nb;
        const float *xdb = xd + (size_t)t1 * nb;
        const int16_t *xsa = xsum + (size_t)t0 * (n_in / 16);
        const int16_t *xsb = xsum + (size_t)t1 * (n_in / 16);

        for (int r = r0; r + 1 < rend; r += 2) {
            const uint8_t *wa = W + (size_t)r * wrow;
            const uint8_t *wb = wa + wrow;
            float s00 = 0.0f, s01 = 0.0f, s10 = 0.0f, s11 = 0.0f;
            for (int i = 0; i < nb; i++) {
                const block_q4_K *ba = (const block_q4_K *)wa + i;
                const block_q4_K *bb = (const block_q4_K *)wb + i;
                float acc[2][2] = {{0, 0}, {0, 0}};
                const int8_t *xa = xab + (size_t)i * QK_K;
                const int8_t *xb = xbb + (size_t)i * QK_K;
                const int16_t *spa = xsa + (size_t)i * (QK_K / 16);
                const int16_t *spb = xsb + (size_t)i * (QK_K / 16);
                const float da = half_to_float(ba->d);
                const float mna = half_to_float(ba->dmin);
                const float db = half_to_float(bb->d);
                const float mnb = half_to_float(bb->dmin);
                for (int g = 0; g < 4; g++)
                    q4k_group_2x2(g, ba->qs + g * 32, bb->qs + g * 32,
                                  ba->scales, bb->scales,
                                  da, mna, db, mnb,
                                  xda[i], xdb[i], spa + g * 4, spb + g * 4,
                                  xa + (size_t)g * 64, xb + (size_t)g * 64,
                                  acc);
                s00 += acc[0][0]; s01 += acc[0][1];
                s10 += acc[1][0]; s11 += acc[1][1];
            }
            out[(size_t)t0 * n_out + r]     = s00;
            out[(size_t)t0 * n_out + r + 1] = s10;
            if (t1 != t0) {
                out[(size_t)t1 * n_out + r]     = s01;
                out[(size_t)t1 * n_out + r + 1] = s11;
            }
        }
    }
}
#endif /* __ARM_FEATURE_MATMUL_INT8 */

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

/* i8mm (FEAT_I8MM / SMMLA) support: enables the token-paired Q4_K GEMM
 * used for T > 1 CPU matmuls. At decode (T == 1) SDOT remains optimal. */
int qma_q8k_gemm_available(void)
{
#if defined(__ARM_FEATURE_MATMUL_INT8)
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
    if (wtype == GGML_TYPE_Q5_K)
        return qma_dot_q5_K_q8k((const block_q5_K *)row, xq, xd, xsum, n);
    if (wtype == GGML_TYPE_Q3_K)
        return qma_dot_q3_K_q8k((const block_q3_K *)row, xq, xd, n);
    if (wtype == GGML_TYPE_IQ2_XS)
        return qma_dot_iq2_xs_q8k((const block_iq2_xs *)row, xq, xd, n);
    if (wtype == GGML_TYPE_IQ2_S)
        return qma_dot_iq2_s_q8k((const block_iq2_s *)row, xq, xd, n);
    if (wtype == GGML_TYPE_Q8_0)
        return qma_dot_q8_0_q8k(row, xq, xd, n);
#endif
    (void)row; (void)wtype; (void)xq; (void)xd; (void)xsum; (void)n;
    return 0.0f;
}
