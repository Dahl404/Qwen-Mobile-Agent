/* nn.c — LFM2.5-2.6B (lfm2) forward pass: compute kernels + eval.
 *
 * Pure C99, NEON-accelerated where it pays. The model is fully resident in
 * RAM (mmap) — no streaming, no prefetch. Q4_0 weight lines with Q8_0
 * activations; the token embedding and the tied lm head are Q6_K.
 *
 * Block (30 layers = 8 GQA attn + 22 short-conv, per cfg schedule):
 *   x += block(rms_norm(x))          # attn or shortconv
 *   x += ffn(rms_norm(x))            # SwiGLU parallel
 * Final: logits = rms_norm(x) · token_embd^T  (tied)
 *
 * Reference: llama.cpp src/models/lfm2.cpp.
 */
#include "lfm.h"
/* cl stripped */
/* injection stripped */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#define MAX_CHUNK 512

/* fwd */
static void lm_head_row(lfm_t *m, const float *x, float *out, int n_threads);

/* ---- coalescence capture hook (probe-only; NULL = zero cost) ---- */
/* emotion / conditioning / LoRA systems stripped for the worker */

/* ---------------- thread pool (port from ma2) ---------------- */
typedef struct {
    pthread_t th[16];
    int n;
    int n_active;
    void (*fn)(void *, int, int);
    void *arg;
    int n_items;
    int chunk;
    volatile int cursor;
    volatile int gen;
    volatile int jobs_done;
    volatile int stop;
    int my_gen[16];
    pthread_mutex_t mu;
    pthread_cond_t cv;
} pool_t;

typedef struct { pool_t *p; int me; } pool_arg_t;

static pool_t g_pool;
static int g_pool_ready = 0;
static int g_pool_active = 0;

static void *pool_worker(void *argp) {
    pool_arg_t *pa = argp;
    pool_t *p = pa->p;
    const int me = pa->me;
    free(pa);
    for (;;) {
        pthread_mutex_lock(&p->mu);
        while (p->gen == p->my_gen[me] && !p->stop)
            pthread_cond_wait(&p->cv, &p->mu);
        if (p->stop) { pthread_mutex_unlock(&p->mu); break; }
        p->my_gen[me] = p->gen;
        if (me >= p->n_active) {
            pthread_mutex_unlock(&p->mu);
            pthread_mutex_lock(&p->mu);
            p->jobs_done++;
            if (p->jobs_done == p->n) pthread_cond_broadcast(&p->cv);
            pthread_mutex_unlock(&p->mu);
            continue;
        }
        pthread_mutex_unlock(&p->mu);
        for (;;) {
            int i = __sync_fetch_and_add(&p->cursor, p->chunk);
            if (i >= p->n_items) break;
            int i1 = i + p->chunk < p->n_items ? i + p->chunk : p->n_items;
            p->fn(p->arg, i, i1);
        }
        pthread_mutex_lock(&p->mu);
        p->jobs_done++;
        if (p->jobs_done == p->n) pthread_cond_broadcast(&p->cv);
        pthread_mutex_unlock(&p->mu);
    }
    return NULL;
}

static void pool_init(int n) {
    if (g_pool_ready) return;
    if (n < 1) n = 1;
    if (n > 16) n = 16;
    g_pool.n = n;
    g_pool.stop = 0;
    g_pool.gen = 0;
    g_pool.jobs_done = 0;
    memset(g_pool.my_gen, 0, sizeof(g_pool.my_gen));
    pthread_mutex_init(&g_pool.mu, NULL);
    pthread_cond_init(&g_pool.cv, NULL);
    for (int i = 0; i < n; i++) {
        pool_arg_t *pa = malloc(sizeof(pool_arg_t));
        pa->p = &g_pool; pa->me = i;
        pthread_create(&g_pool.th[i], NULL, pool_worker, pa);
    }
    g_pool_ready = 1;
}

static void pool_run(void (*fn)(void *, int, int), void *arg, int n_items) {
    pool_t *p = &g_pool;
    if (n_items <= 0) return;
    p->n_active = (g_pool_active > 0 && g_pool_active < p->n) ? g_pool_active : p->n;
    if (p->n_active == 1 || n_items < p->n_active * 2) { fn(arg, 0, n_items); return; }
    p->fn = fn;
    p->arg = arg;
    p->n_items = n_items;
    p->chunk = n_items / (p->n_active * 8) + 1;
    __sync_lock_release(&p->cursor);
    p->cursor = 0;
    pthread_mutex_lock(&p->mu);
    p->jobs_done = 0;
    p->gen++;
    pthread_cond_broadcast(&p->cv);
    while (p->jobs_done != p->n) pthread_cond_wait(&p->cv, &p->mu);
    pthread_mutex_unlock(&p->mu);
}

/* ---------------- quantize q8_0 (activation) ---------------- */
static void quantize_row_q8_0_impl(const float *x, block_q8_0 *y, int64_t k) {
    for (int64_t i = 0; i < k; i += QK8_0) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            float a = fabsf(x[i + j]);
            if (a > amax) amax = a;
        }
        const float d = amax / 127.0f;
        y[i / QK8_0].d = d > 0 ? float_to_half(d) : 0;
        for (int j = 0; j < QK8_0; j++)
            y[i / QK8_0].qs[j] = d > 0 ? (int8_t)roundf(x[i + j] / d) : 0;
    }
}
void lfm_quantize_row_q8_0(const float *x, block_q8_0 *y, int64_t k) {
    quantize_row_q8_0_impl(x, y, k);
}

/* ---------------- Q4_0 x Q8_0 dot (block scales) ----------------
 * NEON (aarch64): dequant the 4-bit nibbles to int8 (half-split: low
 * nibble of qs[j] = weight j, high = weight j+16), then signed dot with
 * the q8_0 lanes. ~4-8x over the scalar loop on ARMv8.2+ (dotprod). */
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
void dot_q4_0_q8_0(int n, const block_q4_0 *x, const block_q8_0 *y, float *s) {
    const int nb = n / QK4_0;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const uint8x16_t m4b = vdupq_n_u8(0x0F);
    const int8x16_t  s8b = vdupq_n_s8(0x8);
    float32x4_t sumv = vdupq_n_f32(0.0f);
    for (int i = 0; i < nb; i++) {
        const float scale = half_to_float(x[i].d) * half_to_float(y[i].d);
        const uint8x16_t v0 = vld1q_u8(x[i].qs);
        const int8x16_t x_l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0, m4b)), s8b);
        const int8x16_t x_h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0, 4)), s8b);
        const int8x16_t y_l = vld1q_s8(y[i].qs);
        const int8x16_t y_h = vld1q_s8(y[i].qs + QK4_0 / 2);
        int32x4_t p = vdotq_s32(vdotq_s32(vdupq_n_s32(0), x_l, y_l), x_h, y_h);
        sumv = vmlaq_n_f32(sumv, vcvtq_f32_s32(p), scale);
    }
    *s = vaddvq_f32(sumv);
#else
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const float d4 = half_to_float(x[i].d);
        const float d8 = half_to_float(y[i].d);
        int sumi0 = 0, sumi1 = 0;
        for (int j = 0; j < QK4_0 / 2; j++) {
            const int v0 = (x[i].qs[j] & 0x0F) - 8;
            const int v1 = (x[i].qs[j] >> 4) - 8;
            sumi0 += v0 * y[i].qs[j];
            sumi1 += v1 * y[i].qs[j + QK4_0 / 2];
        }
        sumf += (sumi0 + sumi1) * d4 * d8;
    }
    *s = sumf;
#endif
}

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
/* i8mm (vmmlaq) 2x2 dot: computes [x0.y0, x0.y1, x1.y0, x1.y1] in one
 * pass per block-pair (ggml's nrc==2 packing). Callers pair columns
 * and/or rows; pass the same operand twice to duplicate it. */
void dot4_q4_0_q8_0_8mm(const block_q4_0 *x0, const block_q4_0 *x1,
                               const block_q8_0 *y0, const block_q8_0 *y1,
                               int nb, float s[4]) {
    const uint8x16_t m4b = vdupq_n_u8(0x0F);
    const int8x16_t  s8b = vdupq_n_s8(0x8);
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    for (int i = 0; i < nb; i++) {
        const float d0 = half_to_float(x0[i].d);
        const float d1 = half_to_float(x1[i].d);
        const float dy0 = half_to_float(y0[i].d);
        const float dy1 = half_to_float(y1[i].d);
        float32_t _scale[4] = { d0*dy0, d0*dy1, d1*dy0, d1*dy1 };
        const float32x4_t scale = vld1q_f32(_scale);
        const uint8x16_t v00 = vld1q_u8(x0[i].qs);
        const uint8x16_t v01 = vld1q_u8(x1[i].qs);
        const int8x16_t x0_l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v00, m4b)), s8b);
        const int8x16_t x0_h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v00, 4)), s8b);
        const int8x16_t x1_l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v01, m4b)), s8b);
        const int8x16_t x1_h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v01, 4)), s8b);
        const int8x16_t y0_l = vld1q_s8(y0[i].qs);
        const int8x16_t y0_h = vld1q_s8(y0[i].qs + 16);
        const int8x16_t y1_l = vld1q_s8(y1[i].qs);
        const int8x16_t y1_h = vld1q_s8(y1[i].qs + 16);
        const int8x16_t l0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));
        const int8x16_t l1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));
        const int8x16_t l2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));
        const int8x16_t l3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));
        const int8x16_t r0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));
        const int8x16_t r1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));
        const int8x16_t r2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));
        const int8x16_t r3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));
        sumv0 = vmlaq_f32(sumv0, vcvtq_f32_s32(
            vmmlaq_s32(vmmlaq_s32(vmmlaq_s32(vmmlaq_s32(vdupq_n_s32(0), l0, r0), l1, r1), l2, r2), l3, r3)),
            scale);
    }
    /* ggml final rearrange: s[0]=a, s[1]=c, s[2]=b, s[3]=d where
       [a,b,c,d] = [x0.y0, x0.y1, x1.y0, x1.y1] */
    float32x4_t sumv1 = vextq_f32(sumv0, sumv0, 2);
    float32x4_t sumv2 = vzip1q_f32(sumv0, sumv1);
    s[0] = vgetq_lane_f32(sumv2, 0);
    s[1] = vgetq_lane_f32(sumv2, 1);
    s[2] = vgetq_lane_f32(sumv2, 2);
    s[3] = vgetq_lane_f32(sumv2, 3);
}
#endif

/* ---------------- Q6_K x Q8_0 dot (16-weight sub-scales vs 32-lane q8_0) - */
float dot_q6_k_q8_0_256(const block_q6_k *x, const block_q8_0 *y) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    /* NEON: dequant the 256 weights to int8 (aux8), then signed dot with
       the q8_0 lanes, applying the 16 per-16 sub-scales. */
    int8_t aux8[QK6_K];
    const uint8_t *ql = x->ql;
    const uint8_t *qh = x->qh;
    const uint8x16_t m4b = vdupq_n_u8(0x0F);
    const uint8x16_t m3b = vdupq_n_u8(0x03);
    const int8x16_t s32b = vdupq_n_s8(32);
    int8_t *a = aux8;
    for (int half = 0; half < 2; half++) {
        /* ggml q6K layout is QUARTERS per 128 weights:
             q1: w 0-31    ql[0..31] lo nibble, qh[0..31] bits 0-1
             q2: w 32-63   ql[32..63] lo nibble, qh[0..31] bits 2-3
             q3: w 64-95   ql[0..31] hi nibble,  qh[0..31] bits 4-5
             q4: w 96-127  ql[32..63] hi nibble, qh[0..31] bits 6-7  */
        const uint8x16_t ql0 = vld1q_u8(ql);
        const uint8x16_t ql1 = vld1q_u8(ql + 16);
        const uint8x16_t ql2 = vld1q_u8(ql + 32);
        const uint8x16_t ql3 = vld1q_u8(ql + 48);
        const uint8x16_t qh0 = vld1q_u8(qh);
        const uint8x16_t qh1 = vld1q_u8(qh + 16);
        uint8x16_t t;
        t = vshlq_n_u8(vandq_u8(qh0, m3b), 4);
        vst1q_s8(a + 0,   vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql0, m4b), t)), s32b));
        t = vshlq_n_u8(vandq_u8(qh1, m3b), 4);
        vst1q_s8(a + 16,  vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql1, m4b), t)), s32b));
        t = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh0, 2), m3b), 4);
        vst1q_s8(a + 32,  vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql2, m4b), t)), s32b));
        t = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh1, 2), m3b), 4);
        vst1q_s8(a + 48,  vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql3, m4b), t)), s32b));
        t = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh0, 4), m3b), 4);
        vst1q_s8(a + 64,  vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql0, 4), t)), s32b));
        t = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh1, 4), m3b), 4);
        vst1q_s8(a + 80,  vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql1, 4), t)), s32b));
        t = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh0, 6), m3b), 4);
        vst1q_s8(a + 96,  vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql2, 4), t)), s32b));
        t = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh1, 6), m3b), 4);
        vst1q_s8(a + 112, vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql3, 4), t)), s32b));
        ql += 64; qh += 32; a += 128;
    }
    const float d = half_to_float(x->d);
    float sumf = 0.0f;
    for (int b = 0; b < QK6_K / QK8_0; b++) {
        const float d8 = half_to_float(y[b].d);
        const int8x16_t y_l = vld1q_s8(y[b].qs);
        const int8x16_t y_h = vld1q_s8(y[b].qs + 16);
        const int8x16_t x_l = vld1q_s8(aux8 + 32 * b);
        const int8x16_t x_h = vld1q_s8(aux8 + 32 * b + 16);
        int32_t s0 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), x_l, y_l));
        int32_t s1 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), x_h, y_h));
        sumf += d * d8 * ((float)s0 * (float)x->scales[2 * b] +
                          (float)s1 * (float)x->scales[2 * b + 1]);
    }
    return sumf;
#else
    const float d = half_to_float(x->d);
    const uint8_t *ql = x->ql;
    const uint8_t *qh = x->qh;
    float sumf = 0.0f;
    for (int g = 0; g < 16; g++) {
        const float dl = d * (float)x->scales[g];
        const int b8 = (16 * g) / QK8_0;
        const int l8 = (16 * g) % QK8_0;
        const float d8 = half_to_float(y[b8].d);
        int sumi = 0;
        for (int l = 0; l < 16; l++) {
            const int wi = 16 * g + l;
            const int half = wi >> 7;
            const int wi7 = wi & 0x7F;
            const int ql_idx = (wi7 & 0x3F) + 64 * half;
            const int qh_idx = (wi7 & 0x1F) + 32 * half;
            const int shift = ((wi7 >> 5) & 3) * 2;
            int8_t q;
            if (wi7 < 64)
                q = (int8_t)((ql[ql_idx] & 0xF) | (((qh[qh_idx] >> shift) & 3) << 4));
            else
                q = (int8_t)((ql[ql_idx] >> 4) | (((qh[qh_idx] >> shift) & 3) << 4));
            q -= 32;
            sumi += q * y[b8].qs[l8 + l];
        }
        sumf += dl * d8 * sumi;
    }
    return sumf;
#endif
}

/* dequant (debug / embedding rows). Q4_0 nibble layout is half-split:
 * low nibble of qs[j] = weight j, high nibble = weight j + QK4_0/2. */
void dequant_row_q4_0(const block_q4_0 *x, float *y, int64_t k) {
    for (int64_t i = 0; i < k / QK4_0; i++) {
        const float d = half_to_float(x[i].d);
        for (int j = 0; j < QK4_0 / 2; j++) {
            y[i * QK4_0 + j] = d * ((int8_t)(x[i].qs[j] & 0x0F) - 8);
            y[i * QK4_0 + j + QK4_0 / 2] = d * ((int8_t)(x[i].qs[j] >> 4) - 8);
        }
    }
}

void dequant_row_q6_k(const block_q6_k *x, float *y, int64_t k) {
    for (int64_t i = 0; i < k / QK6_K; i++) {
        const float d = half_to_float(x[i].d);
        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t *sc = x[i].scales;
        for (int n = 0; n < QK6_K; n += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[i * QK6_K + n + l + 0] = d * sc[is + 0] * q1;
                y[i * QK6_K + n + l + 32] = d * sc[is + 2] * q2;
                y[i * QK6_K + n + l + 64] = d * sc[is + 4] * q3;
                y[i * QK6_K + n + l + 96] = d * sc[is + 6] * q4;
            }
            ql += 64; qh += 32; sc += 8;
        }
    }
}

/* ---------------- Q4_K x Q8_0 dot (standard 148 B block) -------------
 * Standard llama.cpp block_q4_K layout: d + dmin fp16, scales[16] holding
 * 16 6-bit scale/min pairs (get_scale_min_k4), qs[128] nibbles. value =
 * d1*nib - m1 per 32-lane half-group, dotted against the engine's Q8_0
 * activation blocks (8 per 256 lanes). */
static inline void lfm_scale_min_k4(int j, const uint8_t *q,
                                    uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

static float dot_q4_K_q8_0_256(const block_q4_K *x, const block_q8_0 *y) {
    const float d = half_to_float(x->d);
    const float mn = half_to_float(x->dmin);
    const uint8_t *q = x->qs;
    const uint8_t *sc = x->scales;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const uint8x16_t loM = vdupq_n_u8(0x0F);
    float total = 0.0f;
    for (int g = 0; g < 4; g++) {
        uint8_t s0, m0, s1, m1;
        lfm_scale_min_k4(2*g + 0, sc, &s0, &m0);
        lfm_scale_min_k4(2*g + 1, sc, &s1, &m1);
        const float d1 = d * s0, mm1 = mn * m0;
        const float d2 = d * s1, mm2 = mn * m1;
        const block_q8_0 *y0 = y + 2*g;
        const float d8a = half_to_float(y0[0].d);
        const float d8b = half_to_float(y0[1].d);
        const int8x16_t a0 = vld1q_s8(y0[0].qs);
        const int8x16_t a1 = vld1q_s8(y0[0].qs + 16);
        const int8x16_t a2 = vld1q_s8(y0[1].qs);
        const int8x16_t a3 = vld1q_s8(y0[1].qs + 16);
        const uint8x16_t q0 = vld1q_u8(q + (size_t)g * 32);
        const uint8x16_t q1 = vld1q_u8(q + (size_t)g * 32 + 16);
        const int8x16_t n0 = vreinterpretq_s8_u8(vandq_u8(q0, loM));
        const int8x16_t n1 = vreinterpretq_s8_u8(vandq_u8(q1, loM));
        const int8x16_t h0 = vreinterpretq_s8_u8(vshrq_n_u8(q0, 4));
        const int8x16_t h1 = vreinterpretq_s8_u8(vshrq_n_u8(q1, 4));
        int32x4_t pl = vdotq_s32(vdupq_n_s32(0), n0, a0);
        pl = vdotq_s32(pl, n1, a1);
        int32x4_t ph = vdotq_s32(vdupq_n_s32(0), h0, a2);
        ph = vdotq_s32(ph, h1, a3);
        const int32_t sl = vaddlvq_s8(a0) + vaddlvq_s8(a1);
        const int32_t sh = vaddlvq_s8(a2) + vaddlvq_s8(a3);
        total += d8a * (d1 * (float)vaddvq_s32(pl) - mm1 * (float)sl)
               + d8b * (d2 * (float)vaddvq_s32(ph) - mm2 * (float)sh);
    }
    return total;
#else
    /* portable scalar: same semantics, 8 q8_0 blocks, 4 x 64-lane groups */
    float total = 0.0f;
    for (int g = 0; g < 4; g++) {
        uint8_t s0, m0, s1, m1;
        lfm_scale_min_k4(2*g + 0, sc, &s0, &m0);
        lfm_scale_min_k4(2*g + 1, sc, &s1, &m1);
        const float d1 = d * s0, mm1 = mn * m0;
        const float d2 = d * s1, mm2 = mn * m1;
        const block_q8_0 *y0 = y + 2*g;
        const float d8a = half_to_float(y0[0].d);
        const float d8b = half_to_float(y0[1].d);
        for (int l = 0; l < 32; l++) {
            const int nib_low = q[(size_t)g * 32 + l] & 0xF;
            const int nib_hi  = q[(size_t)g * 32 + l] >> 4;
            total += d8a * (d1 * nib_low - mm1) * y0[0].qs[l]
                   + d8b * (d2 * nib_hi  - mm2) * y0[1].qs[l];
        }
    }
    return total;
#endif
}

/* dot a whole M-lane Q4_K column against an M-lane Q8_0 activation row */
static float dot_q4_K_q8_0(const block_q4_K *col, const block_q8_0 *xq, int M) {
    float s = 0.0f;
    for (int b = 0; b < M / QK_K; b++)
        s += dot_q4_K_q8_0_256(col + b, xq + (size_t)b * 8);
    return s;
}

/* dot a whole M-lane Q6_K column against a Q8_0 activation row (the
   engine's lm-head kernel reused for the matmul path) */
static float dot_q6_K_q8_0(const block_q6_k *col, const block_q8_0 *xq, int M) {
    float s = 0.0f;
    for (int b = 0; b < M / QK6_K; b++)
        s += dot_q6_k_q8_0_256(col + b, xq + (size_t)b * 8);
    return s;
}

/* ---------------- matmul: out[T][N] = x[T][M] · W(M x N, Q4_0) -------------
 * GGUF weights are column-major (ne[0]=M fastest): output column n is
 * CONTIGUOUS nb = M/32 blocks starting at block n*nb. Verified against
 * ggml mul_mat: vec_dot(ne00, ..., src0_row + ir0*nb01). */
typedef struct {
    const float *x;        /* [T][M] */
    const uint8_t *w;      /* Q4_0/Q4_K/Q6_K [M][N] */
    block_q8_0 *xq;        /* [T][M/32] scratch */
    float *out;            /* [T][N] */
    int T, M, N;
    int wtype;             /* GGML_TYPE_* (Q4_0 / Q4_K / Q6_K) */
} mm_ctx;

/* one output column dot, dispatched on the weight type. The activation
   row is quantized to Q8_0 blocks (32 lanes each) for every wtype. */
static float mm_col_dot(const mm_ctx *c, const uint8_t *col,
                        const block_q8_0 *xq) {
    if (c->wtype == GGML_TYPE_Q4_K)
        return dot_q4_K_q8_0((const block_q4_K *)col, xq, c->M);
    if (c->wtype == GGML_TYPE_Q6_K)
        return dot_q6_K_q8_0((const block_q6_k *)col, xq, c->M);
    float s;
    dot_q4_0_q8_0(c->M, (const block_q4_0 *)col, xq, &s);
    return s;
}

/* byte stride of one output column of weight type wtype (M input lanes) */
static size_t mm_col_bytes(const mm_ctx *c) {
    if (c->wtype == GGML_TYPE_Q4_K) return (size_t)(c->M / QK_K) * sizeof(block_q4_K);
    if (c->wtype == GGML_TYPE_Q6_K) return (size_t)(c->M / QK6_K) * sizeof(block_q6_k);
    return (size_t)(c->M / QK4_0) * sizeof(block_q4_0);
}


static void mm_worker_col(void *arg, int i0, int i1) {
    mm_ctx *c = arg;
    /* T == 1 decode: parallel over output columns */
    const int nb = c->M / QK4_0;
    const block_q8_0 *xq = c->xq;
    float *orow = c->out;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    int n = i0;
    if (getenv("MA3_NO_8MM") || c->wtype != GGML_TYPE_Q4_0) {
        const size_t cstride = mm_col_bytes(c);
        for (; n < i1; n++) {
            orow[n] = mm_col_dot(c, c->w + (size_t)n * cstride, xq);
        }
        return;
    }
    for (; n + 1 < i1; n += 2) {       /* i8mm: pair columns, same activation */
        const block_q4_0 *col0 = (const block_q4_0 *)c->w + (size_t)n * nb;
        float s[4];
        dot4_q4_0_q8_0_8mm(col0, col0 + nb, xq, xq, nb, s);
        orow[n] = s[0];
        orow[n + 1] = s[1];
    }
    for (; n < i1; n++) {              /* odd leftover column */
        const block_q4_0 *col = (const block_q4_0 *)c->w + (size_t)n * nb;
        dot_q4_0_q8_0(c->M, col, xq, &orow[n]);
    }
#else
    {
        const size_t cstride = mm_col_bytes(c);
        for (int n = i0; n < i1; n++)
            orow[n] = mm_col_dot(c, c->w + (size_t)n * cstride, xq);
    }
#endif
}


/* T > 1 prefill: parallel over output COLUMNS so the weights are read
   ONCE per matmul (row-parallel re-reads them once per thread — 8x
   bandwidth waste on long prefills). */
static void mm_worker_colT(void *arg, int i0, int i1) {
    mm_ctx *c = arg;
    const int nb = c->M / QK4_0;
    const int qk = c->M / 32;   /* q8_0 blocks per row (all wtypes) */
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    int n = i0;
    if (getenv("MA3_NO_8MM") || c->wtype != GGML_TYPE_Q4_0) {
        const size_t cstride = mm_col_bytes(c);
        for (; n < i1; n++) {
            const uint8_t *col = c->w + (size_t)n * cstride;
            for (int t = 0; t < c->T; t++)
                c->out[(size_t)t * c->N + n] = mm_col_dot(c, col, c->xq + (size_t)t * qk);
        }
        return;
    }
    for (; n + 1 < i1; n += 2) {       /* i8mm: pair columns AND rows — all 4 lanes */
        const block_q4_0 *col0 = (const block_q4_0 *)c->w + (size_t)n * nb;
        const block_q4_0 *col1 = col0 + nb;
        int t = 0;
        for (; t + 1 < c->T; t += 2) {
            const block_q8_0 *xq0 = c->xq + (size_t)t * nb;
            const block_q8_0 *xq1 = xq0 + nb;
            float s[4];
            dot4_q4_0_q8_0_8mm(col0, col1, xq0, xq1, nb, s);
            c->out[(size_t)t * c->N + n] = s[0];
            c->out[(size_t)t * c->N + n + 1] = s[1];
            c->out[(size_t)(t + 1) * c->N + n] = s[2];
            c->out[(size_t)(t + 1) * c->N + n + 1] = s[3];
        }
        for (; t < c->T; t++) {        /* odd row leftover */
            const block_q8_0 *xq0 = c->xq + (size_t)t * nb;
            float s[4];
            dot4_q4_0_q8_0_8mm(col0, col1, xq0, xq0, nb, s);
            c->out[(size_t)t * c->N + n] = s[0];
            c->out[(size_t)t * c->N + n + 1] = s[1];
        }
    }
    for (; n < i1; n++) {              /* odd column leftover */
        const block_q4_0 *col = (const block_q4_0 *)c->w + (size_t)n * nb;
        for (int t = 0; t < c->T; t++) {
            const block_q8_0 *xq = c->xq + (size_t)t * nb;
            dot_q4_0_q8_0(c->M, col, xq, &c->out[(size_t)t * c->N + n]);
        }
    }
#else
    {
        const size_t cstride = mm_col_bytes(c);
        for (int n = i0; n < i1; n++) {
            const uint8_t *col = c->w + (size_t)n * cstride;
            for (int t = 0; t < c->T; t++)
                c->out[(size_t)t * c->N + n] = mm_col_dot(c, col, c->xq + (size_t)t * qk);
        }
    }
#endif
}

static void matmul(float *out, const uint8_t *w, int M, int N, int T,
                   block_q8_0 *xq, const float *x, int wtype) {
    /* quantize input rows (threaded over T); the Q8_0 activation layout
       feeds every weight type (Q4_0: 1 block/32, Q4_K: 8/256, Q6_K: 8/256) */
    for (int t = 0; t < T; t++)
        lfm_quantize_row_q8_0(x + (size_t)t * M, xq + (size_t)t * (M / QK4_0), M);
    mm_ctx c = { x, w, xq, out, T, M, N, wtype };
    /* decode (T=1): columns. prefill (T>1): columns too — the weights are
       shared across the T rows, so a column-parallel split reads them once. */
    if (T == 1)
        pool_run(mm_worker_col, &c, N);
    else
        pool_run(mm_worker_colT, &c, N);
}

/* ---------------- rms norm ---------------- */
typedef struct { const float *x; float *y; const float *w; int n, T; float eps; } rms_ctx;
static void rms_worker(void *arg, int i0, int i1) {
    rms_ctx *c = arg;
    for (int t = i0; t < i1; t++) {
        const float *x = c->x + (size_t)t * c->n;
        float *y = c->y + (size_t)t * c->n;
#if defined(__ARM_NEON)
        float32x4_t acc = vdupq_n_f32(0.0f);
        int i = 0;
        for (; i + 4 <= c->n; i += 4) {
            float32x4_t xv = vld1q_f32(x + i);
            acc = vmlaq_f32(acc, xv, xv);
        }
        float ss = vaddvq_f32(acc);
        for (; i < c->n; i++) ss += x[i] * x[i];
        float r = 1.0f / sqrtf(ss / (float)c->n + c->eps);
        float32x4_t rv = vdupq_n_f32(r);
        for (i = 0; i + 4 <= c->n; i += 4)
            vst1q_f32(y + i, vmulq_f32(vmulq_f32(vld1q_f32(x + i), rv), vld1q_f32(c->w + i)));
        for (; i < c->n; i++) y[i] = x[i] * r * c->w[i];
#else
        double ss = 0;
        for (int i = 0; i < c->n; i++) ss += (double)x[i] * x[i];
        float r = 1.0f / sqrtf((float)(ss / c->n) + c->eps);
        for (int i = 0; i < c->n; i++) y[i] = x[i] * r * c->w[i];
#endif
    }
}
static void rms_norm(float *y, const float *x, const float *w, int n, int T, float eps) {
    rms_ctx c = { x, y, w, n, T, eps };
    pool_run(rms_worker, &c, T);
}

/* per-head rms norm (QK-norm): q [T][n_head*64], weights [64] */
static void per_head_rms(float *v, const float *w, int T, int n_head, int hd) {
    for (int t = 0; t < T; t++)
        for (int h = 0; h < n_head; h++) {
            float *v0 = v + (size_t)t * n_head * hd + (size_t)h * hd;
            double ss = 0;
            for (int i = 0; i < hd; i++) ss += (double)v0[i] * v0[i];
            float r = 1.0f / sqrtf((float)(ss / hd) + 1e-5f);
            for (int i = 0; i < hd; i++) v0[i] *= r * w[i];
        }
}

/* ---------------- RoPE (NEOX: pairs offset by n_rot/2) ---------------- */
static void rope_neox(float *x, int T, int n_head, int hd, int n_rot,
                      const int *pos, float freq_base) {
    const int half = n_rot / 2;
    for (int t = 0; t < T; t++) {
        const float theta = (float)pos[t];
        for (int h = 0; h < n_head; h++) {
            float *v = x + (size_t)t * n_head * hd + (size_t)h * hd;
            for (int i = 0; i < half; i++) {
                const float freq = 1.0f / powf(freq_base, (float)(2 * i) / (float)n_rot);
                const float ang = theta * freq;
                const float cs = cosf(ang), sn = sinf(ang);
                const float a = v[i], b = v[i + half];
                v[i] = a * cs - b * sn;
                v[i + half] = a * sn + b * cs;
            }
        }
    }
}

/* ---------------- embedding (Q6_K): column tok = nb contiguous blocks ---- */

void embed_row(lfm_t *m, int tok, float *out) {
    const int n_embd = (int)m->cfg.n_embd;
    const int nb = n_embd / QK6_K;
    const block_q6_k *col = (const block_q6_k *)m->token_embd + (size_t)tok * nb;
    dequant_row_q6_k(col, out, n_embd);
}

/* ---------------- GQA attention with f16 KV cache ----------------
 * KV layout per attn layer: uint16 [2 * n_kv * hd * n_ctx]; K heads first,
 * then V heads: kv[h * n_ctx * hd + pos * hd + d]. */
typedef struct {
    const float *q;          /* [T][32*64] */
    const float *k, *v;      /* [T][8*64] (already normed+roped) */
    uint16_t *kv;            /* cache slice for this layer */
    float *scores;           /* [max_ctx] scratch */
    float *out;              /* [T][2048] attention output pre-wo */
    int T, n_head, n_kv, hd, n_ctx, pos0, il;
} attn_ctx;

#if defined(__ARM_NEON) && defined(__ARM_FP16_ARGS)
/* q (f32) dot K (f16): 4 lanes per iteration */
static inline float f16_dot_neon(const float *q, const uint16_t *k, int hd) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int d = 0; d < hd; d += 4)
        acc = vfmaq_f32(acc, vld1q_f32(q + d), vcvt_f32_f16(vld1_f16((const __fp16 *)(k + d))));
    return vaddvq_f32(acc);
}
#endif

static void attn_decode_one(attn_ctx *c, int t) {
    const int pos = c->pos0 + t;
    const int n_head = c->n_head, n_kv = c->n_kv, hd = c->hd;
    const float *qrow = c->q + (size_t)t * n_head * hd;
    float *orow = c->out + (size_t)t * n_head * hd;
    const uint16_t *K = c->kv;                 /* K heads */
    const uint16_t *V = c->kv + (size_t)n_kv * c->n_ctx * hd;
    const float scale = 1.0f / sqrtf((float)hd);
    for (int h = 0; h < n_head; h++) {
        const int kh = h * n_kv / n_head;      /* GQA group */
        const float *q = qrow + (size_t)h * hd;
        float *o = orow + (size_t)h * hd;
        const uint16_t *khdK = K + (size_t)kh * c->n_ctx * hd;
        const uint16_t *khdV = V + (size_t)kh * c->n_ctx * hd;
        /* scores over all positions */
        float maxs = -1e30f;
        for (int p = 0; p <= pos; p++) {
            const uint16_t *kp = khdK + (size_t)p * hd;
            float s;
#if defined(__ARM_NEON) && defined(__ARM_FP16_ARGS)
            s = f16_dot_neon(q, kp, hd);
#else
            {
                double ss = 0;
                for (int d = 0; d < hd; d++) ss += (double)q[d] * half_to_float(kp[d]);
                s = (float)ss;
            }
#endif
            float f = s * scale;
            c->scores[p] = f;
            if (f > maxs) maxs = f;
        }
        float esum = 0.0f;
        for (int p = 0; p <= pos; p++) {
            float e = expf(c->scores[p] - maxs);
            c->scores[p] = e;
            esum += e;
        }
        /* coalescence capture: mass = incoming attention (connection
         * weights), summed over heads — the model's own field. Rows are
         * GLOBAL positions (pos0 + t) so the field accumulates across
         * prefill chunks and decode steps. */
        /* all-layer capture (layer-as-time curve): per-layer field */
        /* weighted V: accumulate the 64-dim output vector per position */
#if defined(__ARM_NEON) && defined(__ARM_FP16_ARGS)
        float32x4_t oacc[16];
        for (int d = 0; d < hd; d += 4) oacc[d / 4] = vdupq_n_f32(0.0f);
        for (int p = 0; p <= pos; p++) {
            const uint16_t *vp = khdV + (size_t)p * hd;
            const float w = c->scores[p] / esum;
            const float32x4_t wv = vdupq_n_f32(w);
            for (int d = 0; d < hd; d += 4)
                oacc[d / 4] = vfmaq_f32(oacc[d / 4], wv, vcvt_f32_f16(vld1_f16((const __fp16 *)(vp + d))));
        }
        for (int d = 0; d < hd; d += 4) vst1q_f32(o + d, oacc[d / 4]);
#else
        for (int d = 0; d < hd; d++) {
            double acc = 0;
            for (int p = 0; p <= pos; p++) {
                const uint16_t *vp = khdV + (size_t)p * hd;
                acc += c->scores[p] * half_to_float(vp[d]);
            }
            o[d] = (float)(acc / esum);
        }
#endif
    }
}

/* ---------------- short-conv block ---------------- */
typedef struct {
    const float *in;         /* [T][6144] in_proj output: b[0], c[2048], x[4096] */
    const float *conv_w;     /* F32 [3][2048] */
    float *state;            /* [2][2048] rolling gated values */
    float *y;                /* [T][2048] gated conv output (pre out_proj) */
    int T;
} conv_ctx;

static void conv_worker(void *arg, int i0, int i1) {
    conv_ctx *c = arg;
    const int n_embd = 2048;
    for (int t = i0; t < i1; t++) {
        const float *b = c->in + (size_t)t * 6144 + 0;
        const float *g2 = c->in + (size_t)t * 6144 + 2048;
        const float *xx = c->in + (size_t)t * 6144 + 4096;
        float *y = c->y + (size_t)t * 2048;
        float *st = c->state;
        for (int ch = 0; ch < n_embd; ch++) {
            const float cur = b[ch] * xx[ch];
            /* conv kernel [3][2048] is ne[0]=3 fastest: k(i0,ch) at conv_w[ch*3+i0] */
            const float k0 = c->conv_w[ch * 3 + 0];
            const float k1 = c->conv_w[ch * 3 + 1];
            const float k2 = c->conv_w[ch * 3 + 2];
            const float conv_out = k0 * st[ch] + k1 * st[2048 + ch] + k2 * cur;
            y[ch] = g2[ch] * conv_out;
            st[ch] = st[2048 + ch];
            st[2048 + ch] = cur;
        }
    }
}

/* ---------------- eval ---------------- */

int lfm_eval(lfm_t *m, lfm_rs_t *rs, const int *tokens, int n_tokens,
             float *logits, int n_threads, int prefetch, int logits_all) {
    (void)prefetch;
    pool_init(n_threads);
    const model_config_t *cfg = &m->cfg;
    const int n_embd = (int)cfg->n_embd;
    const int n_ff = (int)cfg->n_ff;
    const int n_head = (int)cfg->n_head;
    const int hd = (int)cfg->n_embd_head;
    const int n_rot = (int)cfg->n_rot;
    const int n_ctx = rs->n_ctx;

    float *x = rs->sx, *xn = rs->sxn, *ra = rs->sra;
    float *sq = rs->sq, *sk = rs->sk, *sv = rs->sv;
    float *scin = rs->sconv_in, *scx = rs->sconv_x;
    float *satt = rs->satt_out;
    float *su = rs->sffn_u, *sg = rs->sffn_g, *sffn_out = rs->sffn_out;
    block_q8_0 *sq8 = rs->sq8;

    int done = 0;
    while (done < n_tokens) {
        const int T = (n_tokens - done < MAX_CHUNK) ? n_tokens - done : MAX_CHUNK;
        const int *tok = tokens + done;

        for (int t = 0; t < T; t++) embed_row(m, tok[t], x + (size_t)t * n_embd);

        for (int il = 0; il < (int)cfg->n_layer; il++) {
            const int n_kv = (int)cfg->head_count_kv[il];
            const int is_attn = n_kv > 0;

            
            rms_norm(xn, x, (const float *)m->layers[il].attn_norm, n_embd, T, cfg->rms_eps);

            if (is_attn) {
                const int wkv_dim = n_kv * hd;             /* 512 */
                matmul(sq, m->layers[il].wq, n_embd, n_embd, T, sq8, xn, m->layers[il].wtype[LFM_W_Q]);
                matmul(sk, m->layers[il].wk, n_embd, wkv_dim, T, sq8, xn, m->layers[il].wtype[LFM_W_K]);
                matmul(sv, m->layers[il].wv, n_embd, wkv_dim, T, sq8, xn, m->layers[il].wtype[LFM_W_V]);

                per_head_rms(sq, (const float *)m->layers[il].q_norm, T, n_head, hd);
                per_head_rms(sk, (const float *)m->layers[il].k_norm, T, n_kv, hd);

                int *pos = malloc(sizeof(int) * (T + 1));
                for (int t = 0; t < T; t++) pos[t] = rs->n_pos + done + t;
                rope_neox(sq, T, n_head, hd, n_rot, pos, cfg->freq_base);
                rope_neox(sk, T, n_kv, hd, n_rot, pos, cfg->freq_base);
                free(pos);

                /* append K/V to the f16 cache */
                uint16_t *kv = rs->kv_cache[il];
                const int pos0 = rs->n_pos + done;
                for (int t = 0; t < T; t++) {
                    const int p = pos0 + t;
                    const float *krow = sk + (size_t)t * wkv_dim;
                    const float *vrow = sv + (size_t)t * wkv_dim;
                    for (int h = 0; h < n_kv; h++) {
                        uint16_t *Kp = kv + ((size_t)h * n_ctx + p) * hd;
                        uint16_t *Vp = kv + ((size_t)n_kv * n_ctx + (size_t)h * n_ctx + p) * hd;
                        const float *kh = krow + (size_t)h * hd;
                        const float *vh = vrow + (size_t)h * hd;
                        for (int d = 0; d < hd; d++) { Kp[d] = float_to_half(kh[d]); Vp[d] = float_to_half(vh[d]); }
                    }
                }

                /* attention per token (memory-bounded) */
                attn_ctx ac = { sq, sk, sv, kv, rs->sscores, satt, T, n_head, n_kv, hd, n_ctx, pos0, il };
                for (int t = 0; t < T; t++) attn_decode_one(&ac, t);

                matmul(ra, m->layers[il].wo, n_embd, n_embd, T, sq8, satt, m->layers[il].wtype[LFM_W_O]);
            } else {
                /* short-conv block: in_proj -> split -> b*x -> conv -> c* -> out_proj */
                matmul(scin, m->layers[il].conv_in_proj, n_embd, 3 * n_embd, T, sq8, xn, m->layers[il].wtype[LFM_W_CIN]);
                {
                    conv_ctx cc = { scin, (const float *)m->layers[il].conv,
                                    rs->conv_state[il], scx, T };
                    for (int t = 0; t < T; t++)
                        conv_worker(&cc, t, t + 1);
                }
                matmul(ra, m->layers[il].conv_out_proj, n_embd, n_embd, T, sq8, scx, m->layers[il].wtype[LFM_W_COUT]);
            }

            /* residual + ffn */
            for (size_t i = 0; i < (size_t)n_embd * T; i++) ra[i] += x[i];
            rms_norm(xn, ra, (const float *)m->layers[il].ffn_norm, n_embd, T, cfg->rms_eps);
            matmul(su, m->layers[il].ffn_up, n_embd, n_ff, T, sq8, xn, m->layers[il].wtype[LFM_W_UP]);
            matmul(sg, m->layers[il].ffn_gate, n_embd, n_ff, T, sq8, xn, m->layers[il].wtype[LFM_W_GATE]);
            /* swiglu: silu(gate) * up, then down (llama.cpp: swiglu_split(gate, up)) */
            for (size_t i = 0; i < (size_t)n_ff * T; i++) sg[i] = (sg[i] / (1.0f + expf(-sg[i]))) * su[i];
                        matmul(sffn_out, m->layers[il].ffn_down, n_ff, n_embd, T, sq8, sg, m->layers[il].wtype[LFM_W_DOWN]);
                        for (size_t i = 0; i < (size_t)n_embd * T; i++) x[i] = ra[i] + sffn_out[i];

        }

        /* final norm + tied lm head (Q6_K) */
        rms_norm(xn, x, (const float *)m->token_embd_norm, n_embd, T, cfg->rms_eps);
        if (logits_all) {
            for (int t = 0; t < T; t++) {
                float *lg = logits + (size_t)(done + t) * m->n_vocab;
                lm_head_row(m, xn + (size_t)t * n_embd, lg, n_threads);
            }
        } else if (logits && done + T == n_tokens) {
            lm_head_row(m, xn + (size_t)(T - 1) * n_embd, logits, n_threads);
        }
        done += T;
    }
    rs->n_pos += n_tokens;
    return 0;
}

/* ---------------- lm head: logits[v] = x · emb[v]^T (Q6_K) ---------------- */
typedef struct {
    const float *x;          /* [n_embd] */
    const block_q6_k *emb;   /* token_embd */
    float *out;              /* [n_vocab] */
    block_q8_0 *xq;          /* [n_embd/32] */
    int n_embd, n_vocab;
} lm_ctx;

static void lm_worker(void *arg, int i0, int i1) {
    lm_ctx *c = arg;
    const int nb = c->n_embd / QK6_K;
    for (int v = i0; v < i1; v++) {
        const block_q6_k *col = c->emb + (size_t)v * nb;
        float s = 0;
        for (int b = 0; b < nb; b++)
            s += dot_q6_k_q8_0_256(col + b, c->xq + (size_t)b * (QK6_K / QK8_0));
        c->out[v] = s;
    }
}

static void lm_head_row(lfm_t *m, const float *x, float *out, int n_threads) {
    const int n_embd = (int)m->cfg.n_embd;
    block_q8_0 xq[2048 / QK8_0];
    lfm_quantize_row_q8_0(x, xq, n_embd);
    lm_ctx c = { x, (const block_q6_k *)m->token_embd, out, xq, n_embd, m->n_vocab };
    pool_run(lm_worker, &c, m->n_vocab);
}

/* ---------------- runstate ---------------- */
int lfm_rs_init(lfm_rs_t *rs, int n_ctx) {
    return lfm_rs_init_kv(rs, n_ctx, NULL, 0);
}

int lfm_rs_init_kv(lfm_rs_t *rs, int n_ctx, const char *path, int persist) {
    memset(rs, 0, sizeof(*rs));
    if (n_ctx < 16) n_ctx = 16;
    rs->n_ctx = n_ctx;
    /* cfg not available here (static): sizes computed from max dims.
       Real sizing happens in nn.c via lfm_eval using rs fields — the arena
       is over-allocated for the maximum supported config. */
    const int n_embd = 2048, n_ff = 10752, hd = 64;
    const int n_attn = 8, n_kv = 8;
    /* KV arena: n_attn layers x 2 x n_kv x hd x n_ctx f16 */
    size_t kv_bytes = (size_t)n_attn * 2 * n_kv * hd * n_ctx * 2;
    /* conv state: N_LAYER x 2 x n_embd f32 (only conv layers use it) */
    size_t conv_bytes = (size_t)N_LAYER * 2 * n_embd * 4;
    /* scratch: worst-case chunk buffers (must match the slice layout below) */
    size_t sc = (size_t)MAX_CHUNK;
    const int nb_max = n_ff / QK8_0;   /* biggest q8 input: ffn 10752 (336 blocks) */
    size_t scratch_bytes =
        sc * n_embd * 4 * 6 +              /* sx, sxn, sra, sq, sconv_x, satt_out */
        sc * 512 * 4 * 2 +                 /* sk, sv */
        sc * 3 * n_embd * 4 +              /* sconv_in */
        sc * n_ff * 4 * 2 +                /* sffn_u, sffn_g */
        sc * n_embd * 4 +                  /* sffn_out */
        (size_t)n_ctx * 4 +                /* sscores */
        sc * nb_max * sizeof(block_q8_0);  /* sq8 (reused by every matmul) */
    size_t total = kv_bytes + conv_bytes + scratch_bytes;

    /* resident by default: the arena is anonymous RAM (fixed size, no
       flash re-reads). The path is only used for explicit save/load at
       session boundaries. */
    rs->kv_map = mmap(NULL, total, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (rs->kv_map == MAP_FAILED) return -1;
    rs->kv_fd = -1;
    rs->kv_path = path;
    rs->kv_persist = persist;
    rs->kv_persist_bytes = kv_bytes + conv_bytes;
    rs->kv_map_len = total;

    uint8_t *p = rs->kv_map;
    for (int il = 0; il < N_LAYER; il++) {
        rs->kv_cache[il] = NULL;
        rs->conv_state[il] = NULL;
    }
    /* KV slices: layers 2,5,9,13,17,21,24,27 (lfm2 schedule, fixed) */
    static const int attn_layers[8] = {2, 5, 9, 13, 17, 21, 24, 27};
    size_t per_layer = (size_t)2 * n_kv * hd * n_ctx * 2;
    for (int i = 0; i < 8; i++) {
        rs->kv_cache[attn_layers[i]] = (uint16_t *)(p + (size_t)i * per_layer);
    }
    p += kv_bytes;
    /* conv state slices (all layers; only conv ones used) */
    size_t per_conv = (size_t)2 * n_embd * 4;
    for (int il = 0; il < N_LAYER; il++) {
        rs->conv_state[il] = (float *)(p + (size_t)il * per_conv);
    }
    p += conv_bytes;
    rs->scratch = p;
    uint8_t *q = p;
    rs->sx = (float *)q; q += sc * n_embd * 4;
    rs->sxn = (float *)q; q += sc * n_embd * 4;
    rs->sra = (float *)q; q += sc * n_embd * 4;
    rs->sq = (float *)q; q += sc * n_embd * 4;
    rs->sk = (float *)q; q += sc * 512 * 4;
    rs->sv = (float *)q; q += sc * 512 * 4;
    rs->sconv_in = (float *)q; q += sc * 3 * n_embd * 4;
    rs->sconv_b = rs->sconv_in;                       /* alias (unused) */
    rs->sconv_c = rs->sconv_in;                       /* alias (unused) */
    rs->sconv_x = (float *)q; q += sc * n_embd * 4;
    rs->sconv_y = rs->sconv_x;
    rs->satt_out = (float *)q; q += sc * n_embd * 4;
    rs->sffn_u = (float *)q; q += sc * n_ff * 4;
    rs->sffn_g = (float *)q; q += sc * n_ff * 4;
    rs->sffn_out = (float *)q; q += sc * n_embd * 4;
    rs->sscores = (float *)q; q += (size_t)n_ctx * 4;
    rs->sq8 = q;
    return 0;
}

void lfm_rs_free(lfm_rs_t *rs) {
    if (rs->kv_map && rs->kv_map != MAP_FAILED) {
        munmap(rs->kv_map, rs->kv_map_len);
    }
    if (rs->kv_fd >= 0) {
        close(rs->kv_fd);
        if (!rs->kv_persist && rs->kv_path) unlink(rs->kv_path);
    }
    memset(rs, 0, sizeof(*rs));
}

/* state.bin = n_pos + the persistent arena head (KV cache + conv state).
   The arena is anonymous RAM (resident); save/load snapshot it across
   sessions — one file write per exit, one read per resume. */
int lfm_rs_save(lfm_rs_t *rs, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int32_t np = rs->n_pos;
    if (fwrite(&np, sizeof(np), 1, f) != 1) { fclose(f); return -1; }
    if (rs->kv_map && rs->kv_persist_bytes > 0) {
        if (fwrite(rs->kv_map, 1, rs->kv_persist_bytes, f) != rs->kv_persist_bytes) {
            fclose(f); return -1;
        }
    }
    fclose(f);
    return 0;
}

int lfm_rs_load(lfm_rs_t *rs, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    int32_t np = 0;
    if (fread(&np, sizeof(np), 1, f) != 1) { fclose(f); return -1; }
    if (rs->kv_map && rs->kv_persist_bytes > 0) {
        if (fread(rs->kv_map, 1, rs->kv_persist_bytes, f) != rs->kv_persist_bytes) {
            fclose(f); return -1;
        }
    }
    fclose(f);
    rs->n_pos = np > 0 ? np : 0;
    return 0;
}

/* HCM no-ops */
int lfm_hcm_save(const char *path) { (void)path; return 0; }
int lfm_hcm_load(const char *path, int n_ctx) { (void)path; (void)n_ctx; return 0; }

/* prefetch no-ops (model resident) */
void lfm_prefetch_init(lfm_t *m) { (void)m; }
void lfm_prefetch_layer(int il, int dist, lfm_t *m) { (void)il; (void)dist; (void)m; }
void lfm_prefetch_whole(lfm_t *m) { (void)m; }
