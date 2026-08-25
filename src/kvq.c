/* kvq.c - quantized KV cache kernels for HCM tiered attention.
 *
 * Formats (per kv-head, per token, N_EMBD_HEAD = 256 dims):
 *   KVQ4: 8 sub-blocks x 32 lanes. Each: fp16 scale d, fp16 min m,
 *         32 x 4-bit quants (16 B). 20 B x 8 = 160 B.
 *   KVQ2: 4 sub-blocks x 64 lanes. Each: fp16 d, fp16 m,
 *         64 x 2-bit quants (16 B). 20 B x 4 = 80 B.
 *   value = q * d + m.
 *
 * Storage: every position owns a fixed KVQ_SLOT (160 B) so q4->q2 demotion
 * re-encodes in place. The dot/vacc kernels are NEON: expand quant lanes to
 * fp32 (bandwidth is the win: 160 B read vs 1024 B fp32), then vmlaq.
 */
#include "kvq.h"
#include "qma.h"   /* half_to_float / float_to_half */
#include <arm_neon.h>
#include <string.h>

/* ---------------- scalar helpers (also used as NEON fallback) ------------ */

static inline uint8_t clamp_u8(int v, int hi) {
    return (uint8_t)(v < 0 ? 0 : (v > hi ? hi : v));
}

/* q4: 8 sub-blocks x 32 lanes; writes the slot's q4 region (offset 0) */
void kvq4_quant(const float *x, int n, uint8_t *slot) {
    const int sb_n = 32;
    for (int sb = 0; sb < n / sb_n; sb++) {
        const float *xp = x + sb * sb_n;
        uint8_t *q = slot + sb * 20;
        float mn = xp[0], mx = xp[0];
        for (int i = 1; i < sb_n; i++) {
            if (xp[i] < mn) mn = xp[i];
            if (xp[i] > mx) mx = xp[i];
        }
        float d = (mx - mn) / 15.0f;
        if (!(d > 0)) d = 1.0f;
        uint16_t dh = float_to_half(d), mh = float_to_half(mn);
        memcpy(q, &dh, 2); memcpy(q + 2, &mh, 2);
        for (int i = 0; i < sb_n; i++) {
            int qi = (int)lrintf((xp[i] - mn) / d);
            qi = clamp_u8(qi, 15);
            if (i & 1) q[4 + i / 2] = (uint8_t)((q[4 + i / 2] & 0x0F) | (qi << 4));
            else       q[4 + i / 2] = (uint8_t)((q[4 + i / 2] & 0xF0) | qi);
        }
    }
}

/* q2: 4 sub-blocks x 64 lanes, 2 bits per quant, 4 quants per byte;
 * writes the slot's q2 region (offset KVQ2_OFF) */
void kvq2_quant(const float *x, int n, uint8_t *slot) {
    const int sb_n = 64;
    for (int sb = 0; sb < n / sb_n; sb++) {
        const float *xp = x + sb * sb_n;
        uint8_t *q = slot + KVQ2_OFF + sb * 20;
        float mn = xp[0], mx = xp[0];
        for (int i = 1; i < sb_n; i++) {
            if (xp[i] < mn) mn = xp[i];
            if (xp[i] > mx) mx = xp[i];
        }
        float d = (mx - mn) / 3.0f;
        if (!(d > 0)) d = 1.0f;
        uint16_t dh = float_to_half(d), mh = float_to_half(mn);
        memcpy(q, &dh, 2); memcpy(q + 2, &mh, 2);
        for (int i = 0; i < sb_n; i++) {
            int qi = (int)lrintf((xp[i] - mn) / d);
            qi = clamp_u8(qi, 3);
            q[4 + i / 4] |= (uint8_t)(qi << (2 * (i & 3)));
        }
    }
}

static inline void sub_meta(const uint8_t *rec, int sb, float *d, float *m) {
    uint16_t dh, mh;
    memcpy(&dh, rec + sb * 20, 2);
    memcpy(&mh, rec + sb * 20 + 2, 2);
    *d = half_to_float(dh);
    *m = half_to_float(mh);
}

/* Q-Hitter quantization-friendliness: how well a K/V vector survives q4.
 * Returns relative RMS error of q4 reconstruction (0 = perfect, 1 = 100%).
 * Q-Hitter's insight: when picking tokens to keep at low precision, prefer
 * ones whose KV quantizes well — the same budget buys more fidelity. */
float kvq4_qf(const uint8_t *slot, const float *x, int n) {
    const int sb_n = 32;
    double se = 0, sx = 0;
    for (int sb = 0; sb < n / sb_n; sb++) {
        float d, m; sub_meta(slot, sb, &d, &m);
        const uint8_t *q = slot + sb * 20 + 4;
        for (int i = 0; i < sb_n; i++) {
            int qi = (i & 1) ? (q[i / 2] >> 4) : (q[i / 2] & 0x0F);
            float v = qi * d + m;
            float e = v - x[sb * sb_n + i];
            se += (double)e * e;
            sx += (double)x[sb * sb_n + i] * x[sb * sb_n + i];
        }
    }
    if (sx <= 0) return 0.0f;
    return (float)sqrt(se / sx);
}

void kvq4_dequant(const uint8_t *rec, float *y, int n) {
    const int sb_n = 32;
    for (int sb = 0; sb < n / sb_n; sb++) {
        float d, m; sub_meta(rec, sb, &d, &m);
        const uint8_t *q = rec + sb * 20 + 4;
        for (int i = 0; i < sb_n; i++) {
            int qi = (i & 1) ? (q[i / 2] >> 4) : (q[i / 2] & 0x0F);
            y[sb * sb_n + i] = qi * d + m;
        }
    }
}

void kvq2_dequant(const uint8_t *rec, float *y, int n) {
    const int sb_n = 64;
    for (int sb = 0; sb < n / sb_n; sb++) {
        float d, m; sub_meta(rec, sb, &d, &m);
        const uint8_t *q = rec + sb * 20 + 4;
        for (int i = 0; i < sb_n; i++) {
            int qi = (q[i / 4] >> (2 * (i & 3))) & 3;
            y[sb * sb_n + i] = qi * d + m;
        }
    }
}

/* ---------------- NEON kernels ------------------------------------------ */

/* expand 32 q4 nibbles (16 B) into 8 float32x4, apply *d + m.
 * Byte k holds lanes 2k (low nibble) and 2k+1 (high nibble); lo/hi
 * are the even/odd lanes, so zip them back into natural order. */
static inline void expand_q4(const uint8_t *q, float d, float m,
                             float32x4_t out[8]) {
    uint8x16_t v;
    memcpy(&v, q, 16);                    // unaligned safe
    uint8x16_t lo = vandq_u8(v, vdupq_n_u8(0x0F));
    uint8x16_t hi = vshrq_n_u8(v, 4);
    uint8x16_t g[2];
    g[0] = vzip1q_u8(lo, hi);  /* bytes 0-7  -> lanes 0..15 */
    g[1] = vzip2q_u8(lo, hi);  /* bytes 8-15 -> lanes 16..31 */
    for (int k = 0; k < 2; k++) {
        uint16x8_t l16 = vmovl_u8(vget_low_u8(g[k]));
        uint16x8_t h16 = vmovl_u8(vget_high_u8(g[k]));
        out[k * 4 + 0] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(l16)));
        out[k * 4 + 1] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(l16)));
        out[k * 4 + 2] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(h16)));
        out[k * 4 + 3] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(h16)));
    }
    float32x4_t dv = vdupq_n_f32(d), mv = vdupq_n_f32(m);
    for (int k = 0; k < 8; k++) out[k] = vmlaq_f32(mv, out[k], dv);
}

/* expand 64 q2 quants (16 B) into 16 float32x4, apply *d + m.
 * Lane i sits in byte i/4, bits (i%4)*2.. — the zip chain below reorders
 * the 4 shift-planes into natural lane order. */
static inline void expand_q2(const uint8_t *q, float d, float m,
                             float32x4_t out[16]) {
    uint8x16_t v;
    memcpy(&v, q, 16);                    // unaligned safe
    uint8x16_t three = vdupq_n_u8(3);
    uint8x16_t e0 = vandq_u8(v, three);
    uint8x16_t e1 = vandq_u8(vshrq_n_u8(v, 2), three);
    uint8x16_t e2 = vandq_u8(vshrq_n_u8(v, 4), three);
    uint8x16_t e3 = vandq_u8(vshrq_n_u8(v, 6), three);
    uint8x16_t z02 = vzip1q_u8(e0, e2);
    uint8x16_t z02h = vzip2q_u8(e0, e2);
    uint8x16_t z13 = vzip1q_u8(e1, e3);
    uint8x16_t z13h = vzip2q_u8(e1, e3);
    uint8x16_t g[4];
    g[0] = vzip1q_u8(z02, z13);    /* bytes 0-3  */
    g[1] = vzip2q_u8(z02, z13);    /* bytes 4-7  */
    g[2] = vzip1q_u8(z02h, z13h);  /* bytes 8-11 */
    g[3] = vzip2q_u8(z02h, z13h);  /* bytes 12-15 */
    for (int k = 0; k < 4; k++) {
        uint16x8_t l16 = vmovl_u8(vget_low_u8(g[k]));
        uint16x8_t h16 = vmovl_u8(vget_high_u8(g[k]));
        out[k * 4 + 0] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(l16)));
        out[k * 4 + 1] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(l16)));
        out[k * 4 + 2] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(h16)));
        out[k * 4 + 3] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(h16)));
    }
    float32x4_t dv = vdupq_n_f32(d), mv = vdupq_n_f32(m);
    for (int k = 0; k < 16; k++) out[k] = vmlaq_f32(mv, out[k], dv);
}

float kvq4_dot(const uint8_t *rec, const float *x, int n) {
    const int sb_n = 32, nsub = n / sb_n;
    float acc = 0.0f;
    for (int sb = 0; sb < nsub; sb++) {
        float d, m; sub_meta(rec, sb, &d, &m);
        float32x4_t e[8];
        expand_q4(rec + sb * 20 + 4, d, m, e);
        float32x4_t a = vdupq_n_f32(0.0f);
        const float *xp = x + sb * sb_n;
        for (int k = 0; k < 8; k++)
            a = vmlaq_f32(a, e[k], vld1q_f32(xp + k * 4));
        acc += vaddvq_f32(a);
    }
    return acc;
}

float kvq2_dot(const uint8_t *rec, const float *x, int n) {
    const int sb_n = 64, nsub = n / sb_n;
    float acc = 0.0f;
    for (int sb = 0; sb < nsub; sb++) {
        float d, m; sub_meta(rec, sb, &d, &m);
        float32x4_t e[16];
        expand_q2(rec + sb * 20 + 4, d, m, e);
        float32x4_t a = vdupq_n_f32(0.0f);
        const float *xp = x + sb * sb_n;
        for (int k = 0; k < 16; k++)
            a = vmlaq_f32(a, e[k], vld1q_f32(xp + k * 4));
        acc += vaddvq_f32(a);
    }
    return acc;
}

void kvq4_vacc(const uint8_t *rec, float w, float *oacc, int n) {
    const int sb_n = 32, nsub = n / sb_n;
    float32x4_t wv = vdupq_n_f32(w);
    for (int sb = 0; sb < nsub; sb++) {
        float d, m; sub_meta(rec, sb, &d, &m);
        float32x4_t e[8];
        expand_q4(rec + sb * 20 + 4, d, m, e);
        float *o = oacc + sb * sb_n;
        for (int k = 0; k < 8; k++)
            vst1q_f32(o + k * 4,
                      vmlaq_f32(vld1q_f32(o + k * 4), wv, e[k]));
    }
}

void kvq2_vacc(const uint8_t *rec, float w, float *oacc, int n) {
    const int sb_n = 64, nsub = n / sb_n;
    float32x4_t wv = vdupq_n_f32(w);
    for (int sb = 0; sb < nsub; sb++) {
        float d, m; sub_meta(rec, sb, &d, &m);
        float32x4_t e[16];
        expand_q2(rec + sb * 20 + 4, d, m, e);
        float *o = oacc + sb * sb_n;
        for (int k = 0; k < 16; k++)
            vst1q_f32(o + k * 4,
                      vmlaq_f32(vld1q_f32(o + k * 4), wv, e[k]));
    }
}
