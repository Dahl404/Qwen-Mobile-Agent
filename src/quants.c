/*
 * quants.c - Q4_K / Q6_K dequant + fp32 dot for qwen35moe
 *
 * Weight formats (from prism-fork ggml-common.h, ground truth):
 *   Q4_K: 256 weights/super-block. d,dmin fp16 + 12 B 6-bit scales +
 *         128 B nibbles. weight = d*sc*(nibble) - dmin*m, per 32-sub-block.
 *   Q6_K: 256 weights/super-block. 6-bit quants (ql/qh) + 8-bit scales.
 *         weight = d*sc*(q-32).
 *
 * The dot is fp32 (dequantize-then-dot) for correctness first; the int8
 * SIMD dot is a later optimization. Activations are q8_0 (fp16 scale +
 * 32 int8).
 */
#include "qma.h"
#include <arm_neon.h>

/* ---------- 6-bit scale/min unpack for Q4_K (ref: get_scale_min_k4) ------- */
static inline void get_scale_min_k4(int j, const uint8_t *q,
                                    uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

/* dequantize a row of Q4_K weights to fp32 (ref: dequantize_row_q4_K) */
void dequantize_row_q4_K(const block_q4_K *x, float *y, int64_t k) {
    const int nb = (int)(k / QK_K);
    for (int i = 0; i < nb; i++) {
        const uint8_t *q = x[i].qs;
        const float d   = half_to_float(x[i].d);
        const float min = half_to_float(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
            q += 32; is += 2;
        }
    }
}

/* dequantize a row of Q6_K weights to fp32 (ref: dequantize_row_q6_K) */
void dequantize_row_q6_K(const block_q6_K *x, float *y, int64_t k) {
    const int64_t nb = k / QK_K;
    for (int i = 0; i < nb; i++) {
        const float d = half_to_float(x[i].d);
        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t  *sc = x[i].scales;
        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

/* dot: out = sum_i W[i] * x[i], W dequantized on the fly from Q4_K.
 * n must be a multiple of QK_K. Activation x is fp32 (pre-quantized to
 * q8_0 elsewhere; here we take the fp32 row directly for correctness). */
float dot_q4_K_f32(const block_q4_K *W, const float *x, int n) {
    const int nb = n / QK_K;
    float acc = 0.0f;
    for (int i = 0; i < nb; i++) {
        const uint8_t *q = W[i].qs;
        const float d   = half_to_float(W[i].d);
        const float min = half_to_float(W[i].dmin);
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, W[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, W[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            const float *xp = x + (size_t)i * QK_K + j;
            float s1 = 0, s2 = 0;
            for (int l = 0; l < 32; ++l) {
                s1 += (d1 * (q[l] & 0xF) - m1) * xp[l];
                s2 += (d2 * (q[l]  >> 4) - m2) * xp[l + 32];
            }
            acc += s1 + s2;
            q += 32; is += 2;
        }
    }
    return acc;
}

/* dot: out = sum_i W[i] * x[i], W from Q6_K. */
float dot_q6_K_f32(const block_q6_K *W, const float *x, int n) {
    const int nb = (int)(n / QK_K);
    float acc = 0.0f;
    for (int i = 0; i < nb; i++) {
        const float d = half_to_float(W[i].d);
        const uint8_t *ql = W[i].ql;
        const uint8_t *qh = W[i].qh;
        const int8_t  *sc = W[i].scales;
        const float *xp = x + (size_t)i * QK_K;
        for (int n2 = 0; n2 < QK_K; n2 += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                acc += d * sc[is + 0] * q1 * xp[l +  0];
                acc += d * sc[is + 2] * q2 * xp[l + 32];
                acc += d * sc[is + 4] * q3 * xp[l + 64];
                acc += d * sc[is + 6] * q4 * xp[l + 96];
            }
            xp += 128; ql += 64; qh += 32; sc += 8;
        }
    }
    return acc;
}
