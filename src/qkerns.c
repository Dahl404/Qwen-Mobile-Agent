/* qkerns.c — Q5_K / Q3_K / IQ2_XS / IQ2_S weight kernels.
 *
 * Adds the quant types needed for unsloth dynamic (UD) mixes:
 *   - Q5_K/Q6_K-style dense layers (Q5_K)
 *   - Q3_K family (incl. Q3_KXS — same super-block format)
 *   - IQ2_XS / IQ2_S codebook quants (the UD expert slabs)
 *
 * Two paths per type, mirroring q8k.c/quants.c conventions:
 *   - portable f32 dot + dequantize (reference / non-DOTPROD builds)
 *   - NEON SDOT kernel against our q8k activation layout
 *     (xq: int8, xd: per-superblock fp scale, xsum: per-16-lane sums —
 *      only types with an additive bias term consume xsum)
 *
 * IQ2 grid bytes max out at 43 so signed-int8 SDOT is exact — see
 * iq2tables.h generation note. */
#include <string.h>
#include <stdint.h>

#include "qma.h"
#include "iq2tables.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

/* 6-bit scale/min unpack for Q4_K/Q5_K super-blocks (same indexing as
 * quants.c get_scale_min_k4 / q8k.c q8k_scale_min_k4). */
static inline void qma_scale_min_k4(int j, const uint8_t *q,
                                    uint8_t *d, uint8_t *m)
{
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else       { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
                 *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4); }
}

/* ===================== dequantizers (portable) ===================== */

void dequantize_row_q5_K(const block_q5_K *x, float *y, int64_t k) {
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t *ql = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const float d   = half_to_float(x[i].d);
        const float min = half_to_float(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            qma_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc, m1 = min * m;
            qma_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; ++l)
                *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l)
                *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
        }
    }
}

void dequantize_row_q3_K(const block_q3_K *x, float *y, int64_t k) {
    const int64_t nb = k / QK_K;
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    const int8_t *sc = (const int8_t *)aux;
    for (int64_t i = 0; i < nb; i++) {
        const float d_all = half_to_float(x[i].d);
        const uint8_t *q  = x[i].qs;
        const uint8_t *hm = x[i].hmask;
        memcpy(aux, x[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        int is = 0;
        uint8_t mb = 1;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * (sc[is++] - 32);
                for (int l = 0; l < 16; ++l)
                    *y++ = dl * ((int8_t)((q[l] >> shift) & 3) -
                                 ((hm[l] & mb) ? 0 : 4));
                dl = d_all * (sc[is++] - 32);
                for (int l = 0; l < 16; ++l)
                    *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) -
                                 ((hm[l + 16] & mb) ? 0 : 4));
                shift += 2; mb <<= 1;
            }
            q += 32;
        }
    }
}

void dequantize_row_iq2_xs(const block_iq2_xs *x, float *y, int64_t k) {
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const float d = half_to_float(x[i].d);
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            const float db0 = d * (0.5f + (x[i].scales[ib32] & 0xf)) * 0.25f;
            const float db1 = d * (0.5f + (x[i].scales[ib32] >> 4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint64_t gr =
                    iq2xs_grid[x[i].qs[4 * ib32 + l] & 511];
                const uint8_t sg = ksigns_iq2xs[x[i].qs[4 * ib32 + l] >> 9];
                const float db = (l / 2) ? db1 : db0;
                const uint8_t *g = (const uint8_t *)&gr;
                for (int j = 0; j < 8; ++j)
                    *y++ = db * g[j] * ((sg >> j) & 1 ? -1.f : 1.f);
            }
        }
    }
}

void dequantize_row_iq2_s(const block_iq2_s *x, float *y, int64_t k) {
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const float d = half_to_float(x[i].d);
        const uint8_t *qs = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const uint8_t *signs = qs + QK_K / 8;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            const float db0 = d * (0.5f + (x[i].scales[ib32] & 0xf)) * 0.25f;
            const float db1 = d * (0.5f + (x[i].scales[ib32] >> 4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const float db = (l / 2) ? db1 : db0;
                const uint64_t gr =
                    iq2s_grid[qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)];
                const uint8_t *g = (const uint8_t *)&gr;
                for (int j = 0; j < 8; ++j)
                    *y++ = db * g[j] *
                           ((signs[l] >> j) & 1 ? -1.f : 1.f);
            }
            qs += 4; signs += 4;
        }
    }
}

/* ===================== f32 dots (portable) ===================== */
/* Straightforward W·x accumulating in f32 through the dequant formulas. */

float dot_q5_K_f32(const block_q5_K *W, const float *x, int n) {
    const int nb = n / QK_K;
    float total = 0.f;
    for (int i = 0; i < nb; i++) {
        const float d   = half_to_float(W[i].d);
        const float mn  = half_to_float(W[i].dmin);
        const uint8_t *ql = W[i].qs, *qh = W[i].qh;
        const float *xp = x + (size_t)i * QK_K;
        int is = 0; uint8_t sc, m; uint8_t u1 = 1, u2 = 2;
        float sum = 0.f;
        for (int j = 0; j < QK_K; j += 64) {
            qma_scale_min_k4(is + 0, W[i].scales, &sc, &m);
            const float d1 = d * sc, m1 = mn * m;
            qma_scale_min_k4(is + 1, W[i].scales, &sc, &m);
            const float d2 = d * sc, m2 = mn * m;
            for (int l = 0; l < 32; ++l)
                sum += (d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1)
                       * xp[j + l];
            for (int l = 0; l < 32; ++l)
                sum += (d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2)
                       * xp[j + 32 + l];
            ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
        }
        total += sum;
    }
    return total;
}

float dot_q3_K_f32(const block_q3_K *W, const float *x, int n) {
    const int nb = n / QK_K;
    float total = 0.f;
    /* reuse dequantizer per block for clarity — this is the slow
     * fallback path anyway */
    static _Thread_local float buf[QK_K];
    for (int i = 0; i < nb; i++) {
        dequantize_row_q3_K(&W[i], buf, QK_K);
        const float *xp = x + (size_t)i * QK_K;
        float sum = 0.f;
        for (int j = 0; j < QK_K; j++) sum += buf[j] * xp[j];
        total += sum;
    }
    return total;
}

float dot_iq2_xs_f32(const block_iq2_xs *W, const float *x, int n) {
    const int nb = n / QK_K;
    float total = 0.f;
    static _Thread_local float buf[QK_K];
    for (int i = 0; i < nb; i++) {
        dequantize_row_iq2_xs(&W[i], buf, QK_K);
        const float *xp = x + (size_t)i * QK_K;
        float sum = 0.f;
        for (int j = 0; j < QK_K; j++) sum += buf[j] * xp[j];
        total += sum;
    }
    return total;
}

float dot_iq2_s_f32(const block_iq2_s *W, const float *x, int n) {
    const int nb = n / QK_K;
    float total = 0.f;
    static _Thread_local float buf[QK_K];
    for (int i = 0; i < nb; i++) {
        dequantize_row_iq2_s(&W[i], buf, QK_K);
        const float *xp = x + (size_t)i * QK_K;
        float sum = 0.f;
        for (int j = 0; j < QK_K; j++) sum += buf[j] * xp[j];
        total += sum;
    }
    return total;
}

/* ===================== q8k SDOT kernels ===================== */

#if defined(__ARM_FEATURE_DOTPROD)

/* Q5_K × q8: exactly dot_q4_K_q8k plus the qh high-bit fold. Weight
 * values are 0..31 (low nibble + optional 16), so we can build the true
 * integer value in-lane and keep the additive min bias on xsum, same
 * split as Q4_K. */
float qma_dot_q5_K_q8k(const block_q5_K *b, const int8_t *xq,
                       const float *xd, const int16_t *xsum, int cols)
{
    const int nb = cols / QK_K;
    float total = 0.f;
    const uint8x16_t loM  = vdupq_n_u8(0x0F);
    const uint8x16_t b16M = vdupq_n_u8(0x10);
    for (int i = 0; i < nb; i++) {
        const float d  = half_to_float(b[i].d);
        const float mn = half_to_float(b[i].dmin);
        const float xdb = xd[i];
        const uint8_t *q  = b[i].qs;
        const uint8_t *qh = b[i].qh;
        const int8_t *xp = xq + (size_t)i * QK_K;
        const int16_t *sp = xsum + (size_t)i * (QK_K / 16);
        float32x4_t acc = vdupq_n_f32(0.f);
        float bias = 0.f;
        int is = 0;
        for (int g = 0; g < 4; g++) {
            uint8_t s1, m1, s2, m2;
            qma_scale_min_k4(is + 0, b[i].scales, &s1, &m1);
            qma_scale_min_k4(is + 1, b[i].scales, &s2, &m2);
            const float d1 = d * s1, mm1 = mn * m1;
            const float d2 = d * s2, mm2 = mn * m2;
            const uint8x16_t q0 = vld1q_u8(q);
            const uint8x16_t q1 = vld1q_u8(q + 16);
            /* qh byte l holds the high bit of weight l AND of weight
             * l+32 (ref: both nibble loops index qh[l]); masks differ:
             * bit 2g for lanes 0-63, bit 2g+1 for lanes 32-63... i.e.
             * low nibbles use mask u1=1<<2g, high nibbles u2=1<<2g+1,
             * each against all 32 qh bytes. */
            const uint8x16_t qa = vld1q_u8(qh);
            const uint8x16_t qb = vld1q_u8(qh + 16);
            const uint8x16_t mU1 = vdupq_n_u8(1u << (2 * g));
            const uint8x16_t mU2 = vdupq_n_u8(1u << (2 * g + 1));
            /* zero -> 0x10 (bit absent), nonzero -> 0 */
            const uint8x16_t e10 = vandq_u8(vmvnq_u8(vceqzq_u8(vandq_u8(qa, mU1))), b16M);
            const uint8x16_t e11 = vandq_u8(vmvnq_u8(vceqzq_u8(vandq_u8(qb, mU1))), b16M);
            const uint8x16_t e20 = vandq_u8(vmvnq_u8(vceqzq_u8(vandq_u8(qa, mU2))), b16M);
            const uint8x16_t e21 = vandq_u8(vmvnq_u8(vceqzq_u8(vandq_u8(qb, mU2))), b16M);
            const int8x16_t lo0 = vreinterpretq_s8_u8(
                vorrq_u8(vandq_u8(q0, loM), e10));
            const int8x16_t hi0 = vreinterpretq_s8_u8(
                vorrq_u8(vshrq_n_u8(q0, 4), e20));
            const int8x16_t lo1 = vreinterpretq_s8_u8(
                vorrq_u8(vandq_u8(q1, loM), e11));
            const int8x16_t hi1 = vreinterpretq_s8_u8(
                vorrq_u8(vshrq_n_u8(q1, 4), e21));
            int32x4_t pl = vdotq_s32(vdupq_n_s32(0), lo0, vld1q_s8(xp));
            pl = vdotq_s32(pl, lo1, vld1q_s8(xp + 16));
            int32x4_t ph = vdotq_s32(vdupq_n_s32(0), hi0, vld1q_s8(xp + 32));
            ph = vdotq_s32(ph, hi1, vld1q_s8(xp + 48));
            const float sx0 = (float)(sp[0] + sp[1]);
            const float sx1 = (float)(sp[2] + sp[3]);
            acc = vfmaq_n_f32(acc, vcvtq_f32_s32(pl), d1 * xdb);
            acc = vfmaq_n_f32(acc, vcvtq_f32_s32(ph), d2 * xdb);
            bias += (mm1 * sx0 + mm2 * sx1) * xdb;
            q += 32; xp += 64; sp += 4; is += 2;
        }
        total += vaddvq_f32(acc) - bias;
    }
    return total;
}

/* Q3_K × q8: follows dequantize_row_q3_K order exactly — 2 chunks of
 * 128, each chunk re-reads its 32 qs bytes with shift 0,2,4,6; every
 * 16-lane group has its own scale sc[G]-32 and hmask bit. Weight ints
 * are -4..3, no min bias (xsum unused). */
float qma_dot_q3_K_q8k(const block_q3_K *b, const int8_t *xq,
                       const float *xd, int cols)
{
    const int nb = cols / QK_K;
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    const uint8x16_t and3 = vdupq_n_u8(3);
    const int8x16_t m4   = vdupq_n_s8(-4);
    float total = 0.f;
    for (int i = 0; i < nb; i++) {
        const float dall = half_to_float(b[i].d);
        const float xdb = xd[i];
        const uint8_t *q  = b[i].qs;
        const uint8_t *hm = b[i].hmask;
        const int8_t *xp = xq + (size_t)i * QK_K;
        uint32_t aux[4];
        memcpy(aux, b[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t *sc = (const int8_t *)aux;
        uint8_t mb = 1;
        float32x4_t acc = vdupq_n_f32(0.f);
        for (int nch = 0; nch < 2; nch++) {
            for (int j = 0; j < 4; ++j) {
                /* high bit clear -> subtract 4 */
                const int8x16_t sa = vreinterpretq_s8_u8(
                    vceqzq_u8(vandq_u8(vld1q_u8(hm), vdupq_n_u8(mb))));
                const int8x16_t sb = vreinterpretq_s8_u8(
                    vceqzq_u8(vandq_u8(vld1q_u8(hm + 16), vdupq_n_u8(mb))));
                const uint8x16_t qa = vld1q_u8(q);
                const uint8x16_t qb = vld1q_u8(q + 16);
                const int8x16_t shj = vdupq_n_s8(-(int8_t)(2 * j));
                const uint8x16_t ta = vandq_u8(vshlq_u8(qa, shj), and3);
                const uint8x16_t tb = vandq_u8(vshlq_u8(qb, shj), and3);
                const int8x16_t wa = vaddq_s8(vreinterpretq_s8_u8(ta),
                                              vandq_s8(sa, m4));
                const int8x16_t wb = vaddq_s8(vreinterpretq_s8_u8(tb),
                                              vandq_s8(sb, m4));
                const int32x4_t pa = vdotq_s32(vdupq_n_s32(0), wa,
                                               vld1q_s8(xp));
                const int32x4_t pb = vdotq_s32(vdupq_n_s32(0), wb,
                                               vld1q_s8(xp + 16));
                acc = vfmaq_n_f32(acc, vcvtq_f32_s32(pa),
                                  dall * (sc[8 * nch + 2 * j] - 32) * xdb);
                acc = vfmaq_n_f32(acc, vcvtq_f32_s32(pb),
                                  dall * (sc[8 * nch + 2 * j + 1] - 32) * xdb);
                xp += 32; mb <<= 1;
            }
            q += 32;
        }
        total += vaddvq_f32(acc);
    }
    return total;
}

/* ---- shared IQ2 quad machinery ----
 * One quad = 8 weights from a grid row (bytes <= 43) with per-byte sign
 * bits. Build the signed i8 vector; caller SDOTs pairs of quads. */
static inline int8x8_t iq2_signed_grid(uint64_t grid, uint8_t signs)
{
    /* grid rows hold unsigned magnitudes <= 43; apply sign bits */
    const uint8x8_t gv = vreinterpret_u8_u64(vcreate_u64(grid));
    static const int8_t ls[8] = {0, -1, -2, -3, -4, -5, -6, -7};
    const int8x8_t sh = vld1_s8(ls);
    /* move bit j to bit 7 of lane j, isolate, map 1 -> -g, 0 -> g */
    const uint8x8_t spread = vshl_u8(vdup_n_u8(signs), vreinterpret_s8_u8(
                                        vreinterpret_u8_s8(sh)));
    /* spread lane j = signs >> j; its bit 0 IS sign bit j */
    const uint8x8_t bit = vand_u8(spread, vdup_n_u8(1));
    /* w = g ^ (0xFF if bit) + bit  ==  g*(-1)^bit  for small unsigned g */
    const uint8x8_t flip = vsub_u8(vdup_n_u8(0), bit); /* 0x00 / 0xFF */
    const uint8x8_t wg   = vadd_u8(veor_u8(gv, flip), bit);
    return vreinterpret_s8_u8(wg);
}

float qma_dot_iq2_xs_q8k(const block_iq2_xs *b, const int8_t *xq,
                         const float *xd, int cols)
{
    const int nb = cols / QK_K;
    float total = 0.f;
    for (int i = 0; i < nb; i++) {
        const float d = half_to_float(b[i].d);
        const float xdb = xd[i];
        const int8_t *xp = xq + (size_t)i * QK_K;
        float32x4_t acc = vdupq_n_f32(0.f);
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            const uint8_t scb = b[i].scales[ib32];
            const float db[2] = {
                d * (0.5f + (scb & 0xf)) * 0.25f,
                d * (0.5f + (scb >> 4)) * 0.25f
            };
            const uint16_t *idx = &b[i].qs[4 * ib32];
            const int8x16_t xa = vld1q_s8(xp);
            const int8x8_t w0 = iq2_signed_grid(iq2xs_grid[idx[0] & 511],
                                                ksigns_iq2xs[idx[0] >> 9]);
            const int8x8_t w1 = iq2_signed_grid(iq2xs_grid[idx[1] & 511],
                                                ksigns_iq2xs[idx[1] >> 9]);
            const int8x8_t w2 = iq2_signed_grid(iq2xs_grid[idx[2] & 511],
                                                ksigns_iq2xs[idx[2] >> 9]);
            const int8x8_t w3 = iq2_signed_grid(iq2xs_grid[idx[3] & 511],
                                                ksigns_iq2xs[idx[3] >> 9]);
            int32x4_t p01 = vdotq_s32(vdupq_n_s32(0),
                                      vcombine_s8(w0, w1), xa);
            int32x4_t p23 = vdotq_s32(vdupq_n_s32(0),
                                      vcombine_s8(w2, w3),
                                      vld1q_s8(xp + 16));
            acc = vfmaq_n_f32(acc, vcvtq_f32_s32(p01), db[0] * xdb);
            acc = vfmaq_n_f32(acc, vcvtq_f32_s32(p23), db[1] * xdb);
            xp += 32;
        }
        total += vaddvq_f32(acc);
    }
    return total;
}

float qma_dot_iq2_s_q8k(const block_iq2_s *b, const int8_t *xq,
                        const float *xd, int cols)
{
    const int nb = cols / QK_K;
    float total = 0.f;
    for (int i = 0; i < nb; i++) {
        const float d = half_to_float(b[i].d);
        const float xdb = xd[i];
        const int8_t *xp = xq + (size_t)i * QK_K;
        const uint8_t *qs = b[i].qs;
        const uint8_t *signs = qs + QK_K / 8;
        float32x4_t acc = vdupq_n_f32(0.f);
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            const uint8_t scb = b[i].scales[ib32];
            const float db[2] = {
                d * (0.5f + (scb & 0xf)) * 0.25f,
                d * (0.5f + (scb >> 4)) * 0.25f
            };
            const uint8_t qhb = b[i].qh[ib32];
            const int8x16_t xa = vld1q_s8(xp);
            const int8x8_t w0 = iq2_signed_grid(
                iq2s_grid[qs[0] | ((qhb << 8) & 0x300)], signs[0]);
            const int8x8_t w1 = iq2_signed_grid(
                iq2s_grid[qs[1] | ((qhb << 6) & 0x300)], signs[1]);
            const int8x8_t w2 = iq2_signed_grid(
                iq2s_grid[qs[2] | ((qhb << 4) & 0x300)], signs[2]);
            const int8x8_t w3 = iq2_signed_grid(
                iq2s_grid[qs[3] | ((qhb << 2) & 0x300)], signs[3]);
            int32x4_t p01 = vdotq_s32(vdupq_n_s32(0),
                                      vcombine_s8(w0, w1), xa);
            int32x4_t p23 = vdotq_s32(vdupq_n_s32(0),
                                      vcombine_s8(w2, w3),
                                      vld1q_s8(xp + 16));
            acc = vfmaq_n_f32(acc, vcvtq_f32_s32(p01), db[0] * xdb);
            acc = vfmaq_n_f32(acc, vcvtq_f32_s32(p23), db[1] * xdb);
            qs += 4; signs += 4; xp += 32;
        }
        total += vaddvq_f32(acc);
    }
    return total;
}


/* ===================== Q8_0 ===================== */

void dequantize_row_q8_0(const void *xv, float *y, int64_t k) {
    const block_q8_0 *x = xv;
    const int64_t nb = k / QK8_0;
    for (int64_t i = 0; i < nb; i++) {
        const float d = half_to_float(x[i].d);
        for (int j = 0; j < QK8_0; j++) y[i * QK8_0 + j] = d * x[i].qs[j];
    }
}

float dot_q8_0_f32(const void *wv, const float *x, int n) {
    const block_q8_0 *w = wv;
    const int nb = n / QK8_0;
    float total = 0.f;
    for (int i = 0; i < nb; i++) {
        const float d = half_to_float(w[i].d);
        const float *xp = x + (size_t)i * QK8_0;
        float sum = 0.f;
        for (int j = 0; j < QK8_0; j++) sum += w[i].qs[j] * xp[j];
        total += d * sum;
    }
    return total;
}

#if defined(__ARM_FEATURE_DOTPROD)
/* Q8_0 weights are already int8 with a per-32 fp16 scale; activation
   quant contributes xd per super-block (QK_K) and xsum is unused. */
float qma_dot_q8_0_q8k(const void *wv, const int8_t *xq,
                       const float *xd, int cols) {
    const block_q8_0 *w = wv;
    const int nblk = cols / QK8_0;
    float total = 0.f;
    for (int b = 0; b < nblk; b += QK_K / QK8_0) {   /* one super-block */
        const float xdb = xd[b / (QK_K / QK8_0)];
        float32x4_t acc = vdupq_n_f32(0.f);
        for (int g = 0; g < QK_K / QK8_0 && b + g < nblk; g++) {
            const block_q8_0 *wb = &w[b + g];
            const float d = half_to_float(wb->d);
            const int32x4_t p = vdotq_s32(vdupq_n_s32(0),
                                          vld1q_s8(wb->qs),
                                          vld1q_s8(xq));
            acc = vfmaq_n_f32(acc, vcvtq_f32_s32(p), d * xdb);
            xq += QK8_0;
        }
        total += vaddvq_f32(acc);
    }
    return total;
}
#endif /* __ARM_FEATURE_DOTPROD */

#endif /* __ARM_FEATURE_DOTPROD */
