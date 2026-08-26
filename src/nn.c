#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
/*
 * nn.c - compute kernels + forward pass (qwen35 MoE)
 *
 * All kernels are direct ports of PrismML-Eng/llama.cpp @ 9ca265a (branch
 * prism), the ground-truth implementation that runs this model well:
 *   - Q1_0_g128 dot product:  ggml/src/ggml-cpu/arch/arm/quants.c
 *                              ggml_vec_dot_q1_0_q8_0
 *   - q8_0 activation quant:  ggml/src/ggml-quants.c quantize_row_q8_0_ref
 *   - gated delta net:        ggml/src/ggml-cpu/ops.cpp
 *                              ggml_compute_forward_gated_delta_net_one_chunk
 *   - ssm conv:               ggml/src/ggml-cpu/ops.cpp
 *                              ggml_compute_forward_ssm_conv_f32
 *   - IMROPE:                 ggml/src/ggml-cpu/ops.cpp ggml_mrope_cache_init
 *                              + rotate_pairs
 *   - attention:              flash attention, GQA kv_head = q_head / 6
 *   - graph:                  src/models/qwen35.cpp
 *
 * Activation layout: token-major [T][n] (element (t,i) at t*n + i), i.e.
 * ggml's column-major with ne[0] = n contiguous.
 */
#include "qma.h"
#include "kvq.h"
#include "cl.h"
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <arm_neon.h>
#include <sched.h>
#include <sys/syscall.h>
#ifndef readahead
#define readahead(fd, off, len) syscall(SYS_readahead, (fd), (off), (len))
#endif

/* weight-streaming helpers (defined below, near the prefetch section) */
static int g_pf_dist = 0;  /* prefetch lookahead, set per eval call */
/* YALIS default vectors (waste gqdef): per-(layer, expert) running averages
   of the down-projection output, accumulated during the MoE. Used by
   predict_next to estimate the layer's MoE contribution (d_l), so the
   next-input guess is not "one residual short". NULL = disabled. */
static float   *g_def_sum = NULL;   /* [N_LAYER][N_EXPERT][N_EMBD] */
static uint32_t *g_def_cnt = NULL;  /* [N_LAYER][N_EXPERT] */
static int g_pf_trace = -1;  /* QMA_PFTRACE: log prediction hit rate */
/* getenv() is a linear environ scan (~60ns per miss); it must never run
 * inside per-token/per-channel loops. Cache the flags once (pattern used
 * by gdn_worker's `trace`). */
static int g_qma_trace = -1; /* QMA_TRACE: verbose per-layer debug dumps  */
static inline int qma_trace(void) {
    if (g_qma_trace < 0) g_qma_trace = getenv("QMA_TRACE") != NULL;
    return g_qma_trace;
}
static int g_pred_ids[N_LAYER][N_EXPERT_USED];
static int g_pred_gen[N_LAYER];
static int g_act_ids[N_LAYER][N_EXPERT_USED];
static int g_act_gen[N_LAYER];
/* history: previous eval's routing per layer — at decode the router is
   nearly deterministic between consecutive tokens, so this predicts the
   exact demand set of layer il during layer il's own compute (full layer
   of lead time, vs ~0 for the in-moe demand hint) */
static int g_hist_ids[N_LAYER][N_EXPERT_USED];
static int g_hist_valid[N_LAYER];
/* per-token MoE state (full def below the MoE section) */
typedef struct {
    int   sel[N_EXPERT_USED];   /* selected expert ids */
    float w[N_EXPERT_USED];     /* renormalized weights */
} moe_sel_t;
static void qma_prefetch_experts(qma_t *m, int il, const int *ids, int n);
static void qma_predict_prefetch(qma_t *m, int il, const float *x,
                                    const float *r_l, const moe_sel_t *sel);
static float dot_w_f32(int wtype, const uint8_t *wr, const float *xp, int n);
static void layer_slab_bytes(qma_t *m, int il, size_t *ge, size_t *ue, size_t *dn);
static int  expert_fetch(void *user, int layer, int expert, uint8_t *dst);
static size_t layer_rec_bytes(qma_t *m, int il);
static int  pread_all(int fd, void *dst, size_t n, size_t off);

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ---------------- NEON transcendentals (softmax / silu / sigmoid) ---------
 * Scalar expf costs ~20 cycles per lane and these loops run per token per
 * layer. Degree-5 minimax around ln2 + exponent-bit assembly: ~2e-7 max
 * relative error (well below activation quantization noise). Inputs are
 * clamped to roughly [-87.3, 88.7] so results stay finite. */
static inline float32x4_t v_expf32x4(float32x4_t x)
{
    const float32x4_t kLog2e = vdupq_n_f32(1.44269504088896341f);
    const float32x4_t kLn2   = vdupq_n_f32(0.69314718055994531f);
    /* clamp argument */
    /* clamp argument; upper bound keeps n <= 127 so the assembled
       exponent can never hit the Inf/NaN bit pattern (255) */
    x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.3f)), vdupq_n_f32(88.0f));
    /* n = rint(x * log2e); r = x - n*ln2  in [-ln2/2, ln2/2] */
    int32x4_t n = vcvtaq_s32_f32(vmulq_f32(x, kLog2e));
    float32x4_t r = vfmsq_f32(x, vcvtq_f32_s32(n), kLn2);
    /* exp(r) minimax degree 5, Horner:
       (((((1/120)r + 1/24) r + 1/6) r + 1/2) r + 1) r + 1 */
    float32x4_t p = vdupq_n_f32(1.0f / 120.0f);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.5f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    /* scale by 2^n via exponent bits */
    int32x4_t e = vshlq_n_s32(vaddq_s32(n, vdupq_n_s32(127)), 23);
    float32x4_t s = vreinterpretq_f32_s32(e);
    return vmulq_f32(p, s);
}

/* 1 / (1 + e^-x), vector */
static inline float32x4_t v_sigmoid32x4(float32x4_t x)
{
    float32x4_t t = vaddq_f32(vdupq_n_f32(1.0f), v_expf32x4(vnegq_f32(x)));
    float32x4_t inv = vrecpeq_f32(t);
    inv = vmulq_f32(inv, vrecpsq_f32(t, inv));   /* ~1 ulp */
    inv = vmulq_f32(inv, vrecpsq_f32(t, inv));
    return inv;
}

/* ---------------- thread pool ---------------- */
typedef struct {
    pthread_t th[16];
    int n;
    int n_active;      /* runtime throttle: how many workers may work (<= n) */
    void (*fn)(void *, int, int);
    void *arg;
    int n_items;
    int chunk;
    volatile int cursor;
    volatile int gen;
    volatile int jobs_done;
    volatile int stop;
    int my_gen[16];       /* per-worker last completed generation */
    pthread_mutex_t mu;
    pthread_cond_t cv;
} pool_t;

static pool_t g_pool;
static int g_pool_ready = 0;
static int g_pool_active = 0;   /* runtime throttle: 0 = all workers */

/* runtime worker cap for thermal throttling: n <= 0 restores all workers */
void qma_pool_set_max(int n) { g_pool_active = n < 0 ? 0 : n; }

typedef struct { pool_t *p; int me; } pool_arg_t;

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
        p->my_gen[me] = p->gen;   /* claim this job */
        if (me >= p->n_active) {
            /* throttled worker: skip the work but still report done so the
               caller's jobs_done wait completes */
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
    if (p->n_active == 1 || n_items < p->n_active) { fn(arg, 0, n_items); return; }
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

/* ---------------- quant types ---------------- */
/* quantize k floats (multiple of 32) to q8_0 blocks (ref: quantize_row_q8_0_ref)
 * NEON: vabsq+vmaxq for amax, vcvt+vqmovn for the int8 pack -- ~4x the
 * scalar version (which did fabsf/roundf per element). */
void quantize_row_q8_0(const float *x, block_q8_0 *y, int64_t k) {
    const int nb = (int)(k / QK8_0);
#if defined(__ARM_NEON)
    for (int i = 0; i < nb; i++) {
        const float *xp = x + (size_t)i * QK8_0;
        float32x4_t am = vdupq_n_f32(0.0f);
        for (int j = 0; j < QK8_0; j += 4)
            am = vmaxq_f32(am, vabsq_f32(vld1q_f32(xp + j)));
        float amax = vmaxvq_f32(am);
        float d = amax / 127.0f;
        float id = d ? 1.0f / d : 0.0f;
        y[i].d = float_to_half(d);
        int8_t qs[QK8_0];
        for (int j = 0; j < QK8_0; j += 8) {
            float32x4_t v0 = vmulq_n_f32(vld1q_f32(xp + j), id);
            float32x4_t v1 = vmulq_n_f32(vld1q_f32(xp + j + 4), id);
            int32x4_t s0 = vcvtq_s32_f32(v0);
            int32x4_t s1 = vcvtq_s32_f32(v1);
            int16x8_t s16 = vcombine_s16(vqmovn_s32(s0), vqmovn_s32(s1));
            vst1_s8(qs + j, vqmovn_s16(s16));
        }
        memcpy(y[i].qs, qs, QK8_0);
    }
#else
    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            float v = fabsf(x[i * QK8_0 + j]);
            if (v > amax) amax = v;
        }
        float d = amax / 127.0f;
        float id = d ? 1.0f / d : 0.0f;
        y[i].d = float_to_half(d);
        for (int j = 0; j < QK8_0; j++) y[i].qs[j] = (int8_t)roundf(x[i * QK8_0 + j] * id);
    }
#endif

}
/* ---------------- lightweight timing (env QMA_TIMING=1) ----------
 * Per-step profiler: exclusive per-op categories + per-matmul-size
 * breakdown + running totals. Set QMA_TIMING=1 to enable; a summary is
 * printed after every qma_eval call and a cumulative table at exit. */
static int g_timing = 0;
qma_ecache *g_prof_cache = NULL;
static double t_acc[9];   /* 0 matmul 1 quant 2 gdn 3 conv 4 attn 5 norm 6 misc 7 tot 8 moe */
static long t_cnt[9];
static const char *t_name[9] = {"matmul", "quant", "gdn", "conv", "attn", "norm", "misc", "tot", "moe"};

/* per-matmul-size profile: key = n_in*100000 + n_out */
#define MM_BUCKETS 24
static struct { int n_in, n_out; double ms; long cnt; } mm_prof[MM_BUCKETS];
static int mm_nb = 0;

static void timing_init(void) { g_timing = getenv("QMA_TIMING") != NULL; }
void qma_timing_reset(void) {
    for (int i = 0; i < 9; i++) { t_acc[i] = 0; t_cnt[i] = 0; }
}
static void timing_add(int i, double dt) { if (g_timing) { t_acc[i] += dt; t_cnt[i]++; } }
extern void qma_ecache_prof_delta(const char *tag);
static void timing_report(const char *tag) {
    if (!g_timing) return;
    fprintf(stderr, "[timing:%s]", tag);
    for (int i = 0; i < 9; i++) fprintf(stderr, " %s=%.1fms(%ld)", t_name[i], t_acc[i] * 1000, t_cnt[i]);
    fprintf(stderr, "\n");
    if (g_timing) qma_ecache_prof_delta(tag);
}
static void mm_prof_add(int n_in, int n_out, double ms) {
    if (!g_timing) return;
    for (int i = 0; i < mm_nb; i++) {
        if (mm_prof[i].n_in == n_in && mm_prof[i].n_out == n_out) {
            mm_prof[i].ms += ms; mm_prof[i].cnt++; return;
        }
    }
    if (mm_nb < MM_BUCKETS) { mm_prof[mm_nb].n_in = n_in; mm_prof[mm_nb].n_out = n_out; mm_prof[mm_nb].ms = ms; mm_prof[mm_nb].cnt = 1; mm_nb++; }
}
static void timing_summary(void) {
    if (!g_timing) return;
    fprintf(stderr, "[profile:matmul-by-size]");
    for (int i = 0; i < mm_nb; i++)
        fprintf(stderr, " %dx%d=%.1fms(%ld)", mm_prof[i].n_in, mm_prof[i].n_out, mm_prof[i].ms, mm_prof[i].cnt);
    fprintf(stderr, "\n");
}

/* ---------------- matmul: out[t][r] = sum_i W[r,i] * x[t,i] ---------------- */
typedef struct {
    const uint8_t *W;
    const float *xf;      /* fp32 activation row (for dequant dot) */
    float *out;
    int n_in, n_out, T;
    size_t wrow;          /* bytes per output row of W */
    int wtype;            /* GGML_TYPE_Q4_K / Q6_K */
    int qblocks;
    int q8k;              /* use int8 quantized-activation SDOT path */
    const int8_t *qbuf;   /* [T][n_in] quantized x */
    const float *qds;     /* [T][n_in/QK_K] block scales */
    const int16_t *qsum;  /* [T][n_in/16] per-16-lane sums */
} mm_ctx;

static void mm_worker(void *arg, int i0, int i1) {
    mm_ctx *c = arg;
    if (c->q8k) {
        if (c->T > 1 && c->wtype == GGML_TYPE_Q4_K && qma_q8k_gemm_available()) {
            /* i8mm token-paired GEMM over row pairs; odd tail row via dot */
            const int rend = (i1 - i0 >= 2) ? i0 + (i1 - i0) - ((i1 - i0) & 1) : i0;
            qma_q8k_gemm_q4k(c->W, c->wrow, i0, rend - i0,
                             c->qbuf, c->qds, c->qsum, c->n_in, c->n_out, c->T,
                             c->out);
            if (rend < i1) {
                const uint8_t *wr = c->W + (size_t)rend * c->wrow;
                for (int t = 0; t < c->T; t++)
                    c->out[(size_t)t * c->n_out + rend] =
                        qma_q8k_dot(wr, c->wtype,
                                    c->qbuf + (size_t)t * c->n_in,
                                    c->qds + (size_t)t * (c->n_in / QK_K),
                                    c->qsum + (size_t)t * (c->n_in / 16),
                                    c->n_in);
            }
            return;
        }
        for (int r = i0; r < i1; r++) {
            const uint8_t *wr = c->W + (size_t)r * c->wrow;
            for (int t = 0; t < c->T; t++) {
                const int8_t *xq = c->qbuf + (size_t)t * c->n_in;
                const float *xd = c->qds + (size_t)t * (c->n_in / QK_K);
                const int16_t *xsum = c->qsum + (size_t)t * (c->n_in / 16);
                c->out[(size_t)t * c->n_out + r] =
                    qma_q8k_dot(wr, c->wtype, xq, xd, xsum, c->n_in);
            }
        }
        return;
    }
    for (int r = i0; r < i1; r++) {
        const uint8_t *wr = c->W + (size_t)r * c->wrow;
        for (int t = 0; t < c->T; t++) {
            const float *xp = c->xf + (size_t)t * c->n_in;
            c->out[(size_t)t * c->n_out + r] =
                dot_w_f32(c->wtype, wr, xp, c->n_in);
        }
    }
}

/* ---------------- GPU offload (OpenCL, lazy) ---------------- */
static cl_engine_t g_cl;
static int g_cl_state = 0;      /* 0=uninit, 1=ok, -1=failed/disabled */

static void cl_maybe_init(qma_t *m) {
    if (g_cl_state) return;
    g_cl_state = -1;
    if (getenv("QMA_NO_CL")) return;
    /* memory guard: the lazy weight table grows toward ~1 GB during a
       long prefill (plus the 20 GB streamed model + system). Refuse to
       start the GPU when free memory is tight (this device class has
       crashed under memory pressure). */
    long avail_mb = -1;
    FILE *mf = fopen("/proc/meminfo", "r");
    if (mf) {
        char line[128];
        while (fgets(line, sizeof(line), mf)) {
            if (strncmp(line, "MemAvailable:", 13) == 0) {
                avail_mb = atol(line + 13) / 1024;
                break;
            }
        }
        fclose(mf);
    }
    if (avail_mb > 0 && avail_mb < 3200) {
        fprintf(stderr, "qma: OpenCL disabled — only %ld MB free (need >3200)\n", avail_mb);
        return;
    }
    if (cl_init(&g_cl, m) == 0) g_cl_state = 1;
}

/* route a T>1 Q4_K/Q6_K matmul to the GPU; returns 0 on success (else -1:
 * caller keeps the CPU path). */
static int cl_try_matmul(const uint8_t *w, int n_in, int n_out, int T,
                         int wtype, const float *x, float *out) {
    if (g_cl_state != 1 || T <= 1) return -1;
    if (wtype != GGML_TYPE_Q4_K && wtype != GGML_TYPE_Q6_K) return -1;
    if (n_in % 32 != 0) return -1;
    return cl_matmul_qk(&g_cl, wtype, w, n_in, n_out, T, x, out);
}

/* x: [T][n_in]; W: Q4_K/Q6_K [n_in, n_out]; out: [T][n_out]
 * int8 quantized-activation SDOT dot when available (q8k), else fp32. */
static void matmul(const float *x, const uint8_t *W, int n_in, int n_out, int T,
                   int wtype, float *out) {
    double t0 = 0, t1 = 0;
    if (g_timing) t0 = now_s();
    (void)t1;
    /* GPU first: prefill matmuls (T>1) on the dense projections; anything
       the GPU declines falls through to the CPU path unchanged. */
    if (cl_try_matmul(W, n_in, n_out, T, wtype, x, out) == 0) {
        if (g_timing) {
            double dt = now_s() - t0;
            timing_add(0, dt);
            mm_prof_add(n_in, n_out, dt * 1000.0);
        }
        return;
    }
    static int q8k = -1;
    if (q8k < 0) q8k = qma_q8k_available();
    int8_t *qbuf = NULL; float *qds = NULL; int16_t *qsum = NULL;
    if (q8k && n_in % QK_K == 0) {
        /* quantize x once for this call; all n_out rows share it */
        static int8_t *sb = NULL; static float *sd = NULL; static int16_t *ss = NULL;
        static size_t scap = 0;
        size_t need = (size_t)T * n_in + (size_t)T * (n_in / QK_K) * 2;
        if (need > scap) {
            free(sb); free(sd); free(ss);
            sb = malloc((size_t)T * n_in);
            sd = malloc((size_t)T * (n_in / QK_K) * sizeof(float));
            ss = malloc((size_t)T * (n_in / 16) * sizeof(int16_t));
            scap = need;
        }
        for (int t = 0; t < T; t++)
            qma_q8k_quant(x + (size_t)t * n_in, n_in,
                             sb + (size_t)t * n_in,
                             sd + (size_t)t * (n_in / QK_K),
                             ss + (size_t)t * (n_in / 16));
        qbuf = sb; qds = sd; qsum = ss;
    }
    size_t bs = qma_blk_size(wtype);
    mm_ctx c = { .W = W, .xf = x, .out = out, .n_in = n_in, .n_out = n_out,
                 .T = T, .wrow = (size_t)(n_in / QK_K) * bs, .wtype = wtype,
                 .qblocks = 0, .q8k = q8k && n_in % QK_K == 0,
                 .qbuf = qbuf, .qds = qds, .qsum = qsum };
    pool_run(mm_worker, &c, n_out);
    if (g_timing) {
        double dt = now_s() - t0;
        timing_add(0, dt);
        mm_prof_add(n_in, n_out, dt * 1000.0);
    }
}

/* ---------------- elementwise ---------------- */
static void rms_norm_m(float *out, const float *x, const float *w, int n, int T, float eps) {
    for (int t = 0; t < T; t++) {
        const float *xp = x + (size_t)t * n;
        float *op = out + (size_t)t * n;
        /* NEON sum-of-squares: the old double-precision scalar reduction
           defeated auto-vectorization on this hot per-layer path */
        float32x4_t sacc = vdupq_n_f32(0.0f);
        int i = 0;
        for (; i + 4 <= n; i += 4) {
            const float32x4_t v = vld1q_f32(xp + i);
            sacc = vfmaq_f32(sacc, v, v);
        }
        float sum = vaddvq_f32(sacc);
        for (; i < n; i++) sum += xp[i] * xp[i];
        const float r = 1.0f / sqrtf(sum / n + eps);
        const float32x4_t rv = vdupq_n_f32(r);
        for (i = 0; i + 4 <= n; i += 4)
            vst1q_f32(op + i,
                      vmulq_f32(vmulq_f32(vld1q_f32(xp + i), rv),
                                vld1q_f32(w + i)));
        for (; i < n; i++) op[i] = xp[i] * r * w[i];
    }
}

/* rms norm over n=128 per head (for the gated delta norm): x is [T][H*128] */
static void rms_norm_head(float *x, const float *w, int H, int T) {
    const int n = S_D_STATE;
    for (int t = 0; t < T; t++) {
        float *base = x + (size_t)t * H * n;
        for (int h = 0; h < H; h++) {
            float *v = base + (size_t)h * n;
            float32x4_t sacc = vdupq_n_f32(0.0f);
            int i = 0;
            for (; i + 4 <= n; i += 4) {
                const float32x4_t vv = vld1q_f32(v + i);
                sacc = vfmaq_f32(sacc, vv, vv);
            }
            float sum = vaddvq_f32(sacc);
            for (; i < n; i++) sum += v[i] * v[i];
            const float r = 1.0f / sqrtf(sum / n + RMS_EPS);
            const float32x4_t rv = vdupq_n_f32(r);
            for (i = 0; i + 4 <= n; i += 4)
                vst1q_f32(v + i,
                          vmulq_f32(vld1q_f32(v + i),
                                    vmulq_f32(rv, vld1q_f32(w + i))));
            for (; i < n; i++) v[i] *= r * w[i];
        }
    }
}

static void silu_m(float *x, int n, int T) {
    size_t total = (size_t)n * T;
    size_t i = 0;
    for (; i + 4 <= total; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        /* x / (1 + e^-x) = x * sigmoid(x) */
        vst1q_f32(x + i, vmulq_f32(v, v_sigmoid32x4(v)));
    }
    for (; i < total; i++) x[i] = x[i] / (1.0f + expf(-x[i]));
}

/* a = silu(a) * b  (in place on a) */
static void add_m(float *out, const float *a, const float *b, int n, int T) {
    for (size_t i = 0; i < (size_t)n * T; i++) out[i] = a[i] + b[i];
}

/* generic weight-type dispatch ------------------------------------------ */
/* f32-activation dot for any supported GGUF weight type */
static float dot_w_f32(int wtype, const uint8_t *wr, const float *xp, int n) {
    switch (wtype) {
    case GGML_TYPE_Q4_K:   return dot_q4_K_f32((const block_q4_K *)wr, xp, n);
    case GGML_TYPE_Q5_K:   return dot_q5_K_f32((const block_q5_K *)wr, xp, n);
    case GGML_TYPE_Q6_K:   return dot_q6_K_f32((const block_q6_K *)wr, xp, n);
    case GGML_TYPE_Q3_K:   return dot_q3_K_f32((const block_q3_K *)wr, xp, n);
    case GGML_TYPE_IQ2_XS: return dot_iq2_xs_f32((const block_iq2_xs *)wr, xp, n);
    case GGML_TYPE_IQ2_S:  return dot_iq2_s_f32((const block_iq2_s *)wr, xp, n);
    case GGML_TYPE_Q8_0:   return dot_q8_0_f32(wr, xp, n);
    default:               return 0.0f;
    }
}

/* fused gate+up against one quantized activation when both rows share the
   Q4_K fast path; otherwise falls back to two plain dots. */
static void gateup_w_q8k(const uint8_t *g, const uint8_t *u,
                         int gt, int ut,
                         const int8_t *xq, const float *xd,
                         const int16_t *xsum, int n,
                         float *go, float *uo) {
    if (gt == GGML_TYPE_Q4_K && ut == GGML_TYPE_Q4_K) {
        qma_q8k_gateup(g, u, xq, xd, xsum, n, go, uo);
        return;
    }
    *go = qma_q8k_dot(g, gt, xq, xd, xsum, n);
    *uo = qma_q8k_dot(u, ut, xq, xd, xsum, n);
}

/* ---------------- MoE FFN (ref: llama_graph build_moe_ffn + build_layer_ffn) --
 * Router: logits = gate_inp.x  [256, T] -> softmax -> top-8 -> renormalize
 *          (norm_w=true, w_scale=0: HF Qwen3_5MoeTopKRouter divides by the
 *          sum of the selected probs; there is no routed_scaling_factor).
 * Experts: per selected expert e, out += w_e * down(silu(gate_e.x) * up_e.x)
 *          expert slabs are contiguous in the GGUF (slowest dim = expert).
 * Shared:  out += sigmoid(gate_shexp.x) * down(silu(gate_sh.x) * up_sh.x)
 * The expert matmuls dequantize Q4_K/Q6_K to fp32 (correctness first). */

/* one selected expert, one token: acc[N_EMBD] += w * down(silu(gate.x)*up.x) */
static void moe_expert_one(const uint8_t *ge, const uint8_t *ue, const uint8_t *de,
                           uint32_t ge_type, uint32_t ue_type, uint32_t de_type,
                           const float *xp, float we, float *acc) {
    const size_t ge_row = (size_t)(N_EMBD / QK_K) * qma_blk_size(ge_type);
    const size_t ue_row = (size_t)(N_EMBD / QK_K) * qma_blk_size(ue_type);
    const size_t de_row = (size_t)(N_FF_EXP / QK_K) * qma_blk_size(de_type);
    float gate[N_FF_EXP], up[N_FF_EXP];
    for (int i = 0; i < N_FF_EXP; i++) {
        gate[i] = dot_w_f32(ge_type, ge + (size_t)i * ge_row, xp, N_EMBD);
        up[i]   = dot_w_f32(ue_type, ue + (size_t)i * ue_row, xp, N_EMBD);
        gate[i] = (gate[i] / (1.0f + expf(-gate[i]))) * up[i];  /* silu(gate)*up */
    }
    for (int i = 0; i < N_EMBD; i++) {
        acc[i] += we * dot_w_f32(de_type, de + (size_t)i * de_row, gate, N_FF_EXP);
    }
}

/* q8k variant: xp was quantized once for the whole layer (xq/xd/xsum); the
   gated activation is quantized once per expert and shared by the down rows.
   gate/up are always Q4_K in this GGUF; down is Q4_K or Q6_K. */
static void moe_expert_one_q8k(const uint8_t *ge, const uint8_t *ue,
                               const uint8_t *de, uint32_t ge_type,
                               uint32_t ue_type, uint32_t de_type,
                               const int8_t *xq, const float *xd,
                               const int16_t *xsum, float we,
                               int8_t *gq, float *gd, int16_t *gsum,
                               float *acc) {
    /* row stride: N_EMBD elements per gate/up row = N_EMBD/QK_K blocks */
    const size_t ge_row = (size_t)(N_EMBD / QK_K) * qma_blk_size(ge_type);
    const size_t ue_row = (size_t)(N_EMBD / QK_K) * qma_blk_size(ue_type);
    const size_t dn_row = (size_t)(N_FF_EXP / QK_K) * qma_blk_size(de_type);
    float gate[N_FF_EXP];
    for (int i = 0; i < N_FF_EXP; i++) {
        /* fused when both slabs are Q4_K; else two plain dots */
        float g, u;
        gateup_w_q8k(ge + (size_t)i * ge_row, ue + (size_t)i * ue_row,
                     ge_type, ue_type, xq, xd, xsum, N_EMBD, &g, &u);
        gate[i] = (g / (1.0f + expf(-g))) * u;
    }
    /* quantize the gated activation once; every down row shares it */
    qma_q8k_quant(gate, N_FF_EXP, gq, gd, gsum);
    for (int i = 0; i < N_EMBD; i++) {
        acc[i] += we * qma_q8k_dot((const uint8_t *)de + (size_t)i * dn_row,
                                      de_type, gq, gd, gsum, N_FF_EXP);
    }
}

/* shared expert: acc += sigmoid(gate_shexp.x) * down(silu(gate_sh.x)*up_sh.x) */

/* q8k variant of the shared expert: the FFN input was already quantized
   (xq/xd/xsum); the gated activation is quantized once and shared by the
   down rows. Mirrors moe_expert_one_q8k. */
static void moe_shared_q8k(qma_t *m, int il, uint32_t de_type,
                           const int8_t *xq, const float *xd,
                           const int16_t *xsum, float *acc) {
    const uint32_t ge_type = m->layers[il].t_gate_shexp;
    const uint32_t ue_type = m->layers[il].t_up_shexp;
    const float *gate_inp_sh = (const float *)m->layers[il].ffn_gate_inp_shexp;
    const uint8_t *gs4 = m->layers[il].ffn_gate_shexp;
    const uint8_t *us4 = m->layers[il].ffn_up_shexp;
    float gs = 0;
    for (int i = 0; i < N_EMBD; i++) gs += gate_inp_sh[i] * xq[i] * xd[i / QK_K];
    gs = 1.0f / (1.0f + expf(-gs));
    float gate[N_FF_SHEXP];
    int8_t gq[N_FF_SHEXP]; float gd[N_FF_SHEXP / QK_K]; int16_t gs2[N_FF_SHEXP / 16];
    const size_t ge_row = (size_t)(N_EMBD / QK_K) * qma_blk_size(ge_type);
    const size_t ue_row = (size_t)(N_EMBD / QK_K) * qma_blk_size(ue_type);
    for (int i = 0; i < N_FF_SHEXP; i++) {
        float g, u;
        gateup_w_q8k(gs4 + (size_t)i * ge_row, us4 + (size_t)i * ue_row,
                     ge_type, ue_type, xq, xd, xsum, N_EMBD, &g, &u);
        gate[i] = (g / (1.0f + expf(-g))) * u;
    }
    qma_q8k_quant(gate, N_FF_SHEXP, gq, gd, gs2);
    const size_t dn_row = (size_t)(N_FF_SHEXP / QK_K) * qma_blk_size(de_type);
    for (int i = 0; i < N_EMBD; i++) {
        acc[i] += gs * qma_q8k_dot((const uint8_t *)m->layers[il].ffn_down_shexp +
                                      (size_t)i * dn_row,
                                      de_type, gq, gd, gs2, N_FF_SHEXP);
    }
}

static void moe_shared(qma_t *m, int il, uint32_t de_type, const float *xp, float *acc) {
    const uint32_t ge_type = m->layers[il].t_gate_shexp;
    const uint32_t ue_type = m->layers[il].t_up_shexp;
    const size_t ge_row = (size_t)(N_EMBD / QK_K) * qma_blk_size(ge_type);
    const size_t ue_row = (size_t)(N_EMBD / QK_K) * qma_blk_size(ue_type);
    const size_t de_row = (size_t)(N_FF_SHEXP / QK_K) * qma_blk_size(de_type);
    const float *gate_inp_sh = (const float *)m->layers[il].ffn_gate_inp_shexp; /* [N_EMBD] */
    float gs = 0;
    for (int i = 0; i < N_EMBD; i++) gs += gate_inp_sh[i] * xp[i];
    gs = 1.0f / (1.0f + expf(-gs));   /* sigmoid */
    float gate[N_FF_SHEXP], up[N_FF_SHEXP];
    for (int i = 0; i < N_FF_SHEXP; i++) {
        gate[i] = dot_w_f32(ge_type, m->layers[il].ffn_gate_shexp + (size_t)i * ge_row, xp, N_EMBD);
        up[i]   = dot_w_f32(ue_type, m->layers[il].ffn_up_shexp + (size_t)i * ue_row, xp, N_EMBD);
        gate[i] = (gate[i] / (1.0f + expf(-gate[i]))) * up[i];
    }
    for (int i = 0; i < N_EMBD; i++) {
        acc[i] += gs * dot_w_f32(de_type, m->layers[il].ffn_down_shexp + (size_t)i * de_row, gate, N_FF_SHEXP);
    }
}

/* expert-parallel context (waste xpar): one task per (token, expert) pair */
typedef struct {
    qma_t *m; int il;
    const moe_sel_t *sel; const float *x;
    const int8_t *xq; const float *xd; const int16_t *xsum;
    int q8k;
    size_t ge_bytes;
    float *pacc;   /* [T][N_EXPERT_USED][N_EMBD] per-pair partials */
} moe_xpar_t;
static void moe_xpar_worker(void *arg, int b, int e);
static int8_t  *xq_buf = NULL;
static float   *xd_buf = NULL;
static int16_t *xs_buf = NULL;
static int      xq_T = 0;      /* capacity of the buffers above */

static void moe_ffn(qma_t *m, int il, const float *x, const float *r_l, float *out, int T) {
    const float *gate_inp = (const float *)m->layers[il].ffn_gate_inp; /* [N_EMBD, 256] */
    moe_sel_t sel[T];

    /* --- router: serial per token --- */
    for (int t = 0; t < T; t++) {
        const float *xp = x + (size_t)t * N_EMBD;
        float logits[N_EXPERT];
        float mx = -1e30f;
        for (int e = 0; e < N_EXPERT; e++) {
            const float *gi = gate_inp + (size_t)e * N_EMBD;
            /* NEON dot over 5120 dims x 256 experts per token per layer */
            float32x4_t sacc = vdupq_n_f32(0.0f);
            int i = 0;
            for (; i + 4 <= N_EMBD; i += 4)
                sacc = vfmaq_f32(sacc, vld1q_f32(gi + i), vld1q_f32(xp + i));
            float s = vaddvq_f32(sacc);
            for (; i < N_EMBD; i++) s += gi[i] * xp[i];
            logits[e] = s;
            if (s > mx) mx = s;
        }
        float sum = 0;
        {
            const float32x4_t vmx = vdupq_n_f32(mx);
            float32x4_t sacc = vdupq_n_f32(0.0f);
            int e = 0;
            for (; e + 4 <= N_EXPERT; e += 4) {
                float32x4_t lv = v_expf32x4(vsubq_f32(vld1q_f32(logits + e), vmx));
                vst1q_f32(logits + e, lv);
                sacc = vaddq_f32(sacc, lv);
            }
            sum = vaddvq_f32(sacc);
            for (; e < N_EXPERT; e++) { logits[e] = expf(logits[e] - mx); sum += logits[e]; }
        }
        for (int e = 0; e < N_EXPERT; e++) logits[e] /= sum;
        /* top-8 selection */
        for (int k = 0; k < N_EXPERT_USED; k++) {
            int best = 0; float bv = -1e30f;
            for (int e = 0; e < N_EXPERT; e++) if (logits[e] > bv) { bv = logits[e]; best = e; }
            sel[t].sel[k] = best; sel[t].w[k] = bv; logits[best] = -1e30f;
        }
        float ws = 0;
        for (int k = 0; k < N_EXPERT_USED; k++) ws += sel[t].w[k];
        for (int k = 0; k < N_EXPERT_USED; k++) sel[t].w[k] /= ws;
        if (qma_trace() && il == 0 && t == 0) {
            fprintf(stderr, "[router0] top8 experts: %d %d %d %d %d %d %d %d  w: %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
                    sel[t].sel[0], sel[t].sel[1], sel[t].sel[2], sel[t].sel[3],
                    sel[t].sel[4], sel[t].sel[5], sel[t].sel[6], sel[t].sel[7],
                    sel[t].w[0], sel[t].w[1], sel[t].w[2], sel[t].w[3],
                    sel[t].w[4], sel[t].w[5], sel[t].w[6], sel[t].w[7]);
        }
        if (g_pf_trace && t == T - 1) {
            for (int k = 0; k < N_EXPERT_USED; k++) g_act_ids[il][k] = sel[t].sel[k];
            g_act_gen[il]++;
        }
    }
    /* history for next eval (decode: next token routes nearly identically) */
    for (int k = 0; k < N_EXPERT_USED; k++) g_hist_ids[il][k] = sel[T - 1].sel[k];
    g_hist_valid[il] = 1;

    /* --- expert streaming: hint this layer's selection now, so the pages
       load while the expert matmuls below run (waste ecache hint + dedup).
       One madvise per distinct expert across the whole chunk: the kernel
       page cache then serves the repeated accesses from every token. */
    if (g_pf_dist > 0) {
        uint8_t seen[N_EXPERT / 8 + 1] = {0};
        int ids[N_EXPERT];
        int n = 0;
        for (int t = 0; t < T; t++) {
            for (int k = 0; k < N_EXPERT_USED; k++) {
                const int e = sel[t].sel[k];
                if (!(seen[e >> 3] & (1u << (e & 7)))) {
                    seen[e >> 3] |= (uint8_t)(1u << (e & 7));
                    ids[n++] = e;
                }
            }
        }
        qma_prefetch_experts(m, il, ids, n);
        /* predict the NEXT layer's experts (waste predict_next_moe, with
           the YALIS d_l correction: q = LN_{L+1}(r_l + Σ w_j·defaults)) */
        qma_predict_prefetch(m, il, x, r_l, sel);
    }

    /* --- experts: parallel over (token, expert) pairs --- */
    /* zero the accumulator */
    memset(out, 0, (size_t)N_EMBD * T * sizeof(float));
    /* expert-parallel: one task per (token, expert) pair on the matmul pool
       (waste xpar). Each task holds its expert's record via the cache and
       writes its weighted partial into a per-task accumulator; the serial
       sum after the fork is order-independent. */
    size_t ge_bytes, ue_bytes, dn_bytes;
    layer_slab_bytes(m, il, &ge_bytes, &ue_bytes, &dn_bytes);
    static int q8k_on = -1;
    if (q8k_on < 0) q8k_on = qma_q8k_available() && getenv("QMA_NOQ8K") == NULL;
    if (q8k_on) {
        /* grow-on-demand: the buffers are static and sized for the first
           call's T; a later call with more tokens (e.g. the second chat
           message re-prefilling history) would overflow them */
        if (!xq_buf || T > xq_T) {
            free(xq_buf); free(xd_buf); free(xs_buf);
            xq_buf = malloc((size_t)T * N_EMBD);
            xd_buf = malloc((size_t)T * (N_EMBD / QK_K) * sizeof(float));
            xs_buf = malloc((size_t)T * (N_EMBD / 16) * sizeof(int16_t));
            xq_T = T;
            if (!xq_buf || !xd_buf || !xs_buf) { xq_T = 0; return; }
        }
        for (int t = 0; t < T; t++)
            qma_q8k_quant(x + (size_t)t * N_EMBD, N_EMBD,
                             xq_buf + (size_t)t * N_EMBD,
                             xd_buf + (size_t)t * (N_EMBD / QK_K),
                             xs_buf + (size_t)t * (N_EMBD / 16));
    }
    /* persistent grow-on-demand buffer (same pattern as xq_buf above):
       this was malloc/free per layer per eval — heap churn and page faults
       right next to the weight-streaming hot path, 48x per token. */
    static float *pacc = NULL;
    static size_t pacc_cap = 0;
    const size_t pacc_need = (size_t)T * N_EXPERT_USED * N_EMBD;
    if (pacc_cap < pacc_need) {
        free(pacc);
        pacc = malloc(pacc_need * sizeof(float));
        if (!pacc) { pacc_cap = 0; return; }
        pacc_cap = pacc_need;
    }
    if (!pacc) return;
    /* the expert workers ACCUMULATE into pacc (acc[i] += ...); the old
       fresh malloc came back zero-filled (mmap threshold), the persistent
       buffer does not -- stale partials from the previous layer would
       poison every MoE output. */
    memset(pacc, 0, pacc_need * sizeof(float));
    moe_xpar_t xa;
    xa.m = m; xa.il = il; xa.sel = sel; xa.x = x;
    xa.xq = xq_buf; xa.xd = xd_buf; xa.xsum = xs_buf;
    xa.q8k = q8k_on;
    xa.ge_bytes = ge_bytes;
    xa.pacc = pacc;
    pool_run(moe_xpar_worker, &xa, T * N_EXPERT_USED);

    /* serial weighted sum (order-independent) */
    for (int t = 0; t < T; t++) {
        float *acc = out + (size_t)t * N_EMBD;
        for (int k = 0; k < N_EXPERT_USED; k++) {
            const float *part = pacc + ((size_t)t * N_EXPERT_USED + k) * N_EMBD;
            for (int i = 0; i < N_EMBD; i++) acc[i] += part[i];
        }
        /* shared expert: q8k path when enabled (input already quantized) */
        if (xa.q8k)
            moe_shared_q8k(m, il, m->layers[il].t_down_shexp,
                           xa.xq + (size_t)t * N_EMBD,
                           xa.xd + (size_t)t * (N_EMBD / QK_K),
                           xa.xsum + (size_t)t * (N_EMBD / 16), acc);
        else
            moe_shared(m, il, m->layers[il].t_down_shexp, x + (size_t)t * N_EMBD, acc);
    }
    /* YALIS capture (waste WASTE_DUMP_DEFAULTS): accumulate each expert's
       down-projection output per (layer, expert), unweighted — the routing
       weight is applied at prediction time (d_l = Σ w·d). The predict_next
       refinement uses these averages to estimate the MoE contribution, so
       the next-input guess is not "one residual short". */
    if (g_def_sum && g_def_cnt) {
        float *bs = g_def_sum + (size_t)il * N_EXPERT * N_EMBD;
        uint32_t *bc = g_def_cnt + (size_t)il * N_EXPERT;
        for (int t = 0; t < T; t++) {
            for (int k = 0; k < N_EXPERT_USED; k++) {
                const int ex = sel[t].sel[k];
                const float *part = pacc + ((size_t)t * N_EXPERT_USED + k) * N_EMBD;
                float *dst = bs + (size_t)ex * N_EMBD;
                bc[ex]++;
                for (int i = 0; i < N_EMBD; i++) dst[i] += part[i];
            }
        }
    }
}

/* MoE phase timers (aggregated us; exposed for harnesses/profiles) */
double g_moe_fetch_us = 0, g_moe_comp_us = 0, g_moe_fetched = 0;
void qma_moe_timers_reset(void) { g_moe_fetch_us = g_moe_comp_us = g_moe_fetched = 0; }

/* one (token, expert) task: fetch the record, compute the weighted partial */
static void moe_xpar_worker(void *arg, int b, int e)
{
    moe_xpar_t *xa = (moe_xpar_t *)arg;
    qma_t *m = xa->m;
    const int il = xa->il;
    for (int id = b; id < e; id++) {
        const int t = id / N_EXPERT_USED;
        const int k = id % N_EXPERT_USED;
        const int ex = xa->sel[t].sel[k];
        const float we = xa->sel[t].w[k];
        const float *xp = xa->x + (size_t)t * N_EMBD;
        float *acc = xa->pacc + ((size_t)t * N_EXPERT_USED + k) * N_EMBD;
        const size_t ge_bytes = xa->ge_bytes;
        const size_t ue_bytes = (size_t)N_EMBD * N_FF_EXP *
            qma_blk_size(m->layers[il].t_up_exps) / QK_K;
        const size_t dn = (size_t)N_EMBD * N_FF_EXP *
            qma_blk_size(m->layers[il].t_down_exps) / QK_K;
        const uint8_t *ge = NULL, *ue = NULL, *de = NULL;
        double tf = now_s();
        if (m->mmap_exps) {
            /* zero-copy: point the matmul directly at the mapped file pages.
               The kernel page cache is the cache; madvise (--preload) keeps
               the hot experts resident. No heap copy, no eviction churn. */
            ge = m->map + m->layers[il].off_gate_exps + (size_t)ex * ge_bytes;
            ue = m->map + m->layers[il].off_up_exps   + (size_t)ex * ue_bytes;
            de = m->map + m->layers[il].off_down_exps + (size_t)ex * dn;
        } else {
            const uint8_t *rec = NULL;
            if (m->ecache_on) {
                rec = qma_ecache_get(&m->ecache, il, ex, expert_fetch, m);
            } else {
                static __thread uint8_t *miss_buf = NULL;
                static __thread size_t miss_cap = 0;
                size_t need = ge_bytes + ue_bytes + dn;
                if (!miss_buf || miss_cap < need) {
                    free(miss_buf);
                    /* 4K-aligned: with .4k-aligned GGUFs every slab offset and
                       size is a 4096 multiple, so O_DIRECT can serve these
                       reads straight into the buffer (no page-cache fill,
                       no post-fetch drop needed). */
                    void *ap = NULL;
                    if (posix_memalign(&ap, 4096, (need + 4095) & ~4095UL) != 0)
                        ap = malloc(need);   /* fallback: unaligned, buffered */
                    miss_buf = ap;
                    miss_cap = need;
                }
                if (miss_buf && expert_fetch(m, il, ex, miss_buf) == 0)
                    rec = miss_buf;
            }
            if (!rec) {
                /* fetch failed: this pair contributes nothing. The pacc
                   region MUST still be defined -- it is persistent across
                   layers/calls now, and stale data here would poison the
                   serial weighted sum downstream. */
                memset(acc, 0, N_EMBD * sizeof(float));
                continue;
            }
            g_moe_fetch_us += (now_s() - tf) * 1e6;
            g_moe_fetched += 1;
            ge = rec;
            ue = rec + ge_bytes;
            de = rec + ge_bytes + ue_bytes;
        }
        double tc = now_s();
        if (xa->q8k) {
            int8_t gq[N_FF_EXP]; float gd[N_FF_EXP / QK_K]; int16_t gs[N_FF_EXP / 16];
            moe_expert_one_q8k(ge, ue, de,
                               m->layers[il].t_gate_exps, m->layers[il].t_up_exps,
                               m->layers[il].t_down_exps,
                               xa->xq + (size_t)t * N_EMBD,
                               xa->xd + (size_t)t * (N_EMBD / QK_K),
                               xa->xsum + (size_t)t * (N_EMBD / 16),
                               we, gq, gd, gs, acc);
        } else {
            moe_expert_one(ge, ue, de,
                           m->layers[il].t_gate_exps, m->layers[il].t_up_exps,
                           m->layers[il].t_down_exps, xp, we, acc);
        }
        g_moe_comp_us += (now_s() - tc) * 1e6;
    }
}

/* ---------------- ssm conv1d (ref: ggml_compute_forward_ssm_conv_f32) ------- */
/*
 * state: [10240][3] (state[ch][j], j = chronological previous samples)
 * x:     [T][10240] (qkv, token-major)
 * w:     [10240][4] (w[ch*4 + k])
 * out:   [10240][T] channel-major: out[ch*T + t]
 * state updated to the last 3 x samples.
 */
static void conv1d_layer(float *state, const float *x, const float *w, float *out, int T) {
    const int nc = S_D_CONV;
    for (int ch = 0; ch < CONV_DIM; ch++) {
        float *st = state + (size_t)ch * 3;
        const float *wc = w + (size_t)ch * nc;
        float *oc = out + (size_t)ch * T;
        for (int t = 0; t < T; t++) {
            float sum = 0.0f;
            for (int k = 0; k < nc; k++) {
                float v = (t + k < 3) ? st[t + k] : x[(size_t)(t + k - 3) * CONV_DIM + ch];
                sum += v * wc[k];
            }
            oc[t] = sum;
        }
        /* update state */
        for (int j = 0; j < 3; j++) {
            float v = (T + j < 3) ? st[T + j] : x[(size_t)(T + j - 3) * CONV_DIM + ch];
            st[j] = v;
        }
    }
}

/* l2 norm over the S_v dim, per head (ref: ggml_compute_forward_l2_norm_f32) */
static void l2norm_skv(float *x, int H, int T) {
    const int S_v = S_D_STATE;
    /* x points at the start of a q48/k48/v48 block inside the interleaved gkv
       buffer. Each token owns a full [q48|k48|v48] row of stride TS_PER_TOKEN
       (18432), so the head h of token t lives at t*TS + h*S_v -- NOT at
       t*S_v*H. The old t*S_v*H stride was only correct for T==1. */
    const size_t TS = (size_t)S_v * S_DT_RANK * 3;
    for (int t = 0; t < T; t++) {
        float *base = x + (size_t)t * TS;
        for (int h = 0; h < H; h++) {
            float *v = base + (size_t)h * S_v;
            float32x4_t sacc = vdupq_n_f32(0.0f);
            int i = 0;
            for (; i + 4 <= S_v; i += 4) {
                const float32x4_t vv = vld1q_f32(v + i);
                sacc = vfmaq_f32(sacc, vv, vv);
            }
            float sum = vaddvq_f32(sacc);
            for (; i < S_v; i++) sum += v[i] * v[i];
            const float r = 1.0f / fmaxf(sqrtf(sum), 1e-6f);
            const float32x4_t rv = vdupq_n_f32(r);
            for (i = 0; i + 4 <= S_v; i += 4)
                vst1q_f32(v + i, vmulq_f32(vld1q_f32(v + i), rv));
            for (; i < S_v; i++) v[i] *= r;
        }
    }
}

/* ---------------- gated delta net ---------------- */
/*
 * q/k/v live in one buffer gkv: [T][18432]
 *   q48:  [0..6144T)   head h at t*18432 + (h%16)*128
 *   k48:  [6144T..12288T)  head h at t*18432 + 6144 + (h%16)*128
 *   v48:  [12288T..18432T) head h at t*18432 + 12288 + h*128
 * state: [48][128][128] M[h][j*128+i] = S[i][j] (transposed storage)
 * gate,beta: [T][48]  (element (t,h) at t*48 + h)
 * out:   [T][6144] token-major
 *
 * Port of ggml_compute_forward_gated_delta_net_one_chunk (K=1), decay fused:
 *   M' = exp(g)*M + k*delta ;  attn_j = (exp(g)*dot(M_j,q) + delta_j*dot(k,q)) * scale
 */
typedef struct {
    float *state;
    const float *q, *k, *v;   /* base pointers into gkv */
    const float *gate, *beta;
    float *out;
    int T, tstride;           /* token stride of gkv = 18432 */
} gdn_ctx;

static void gdn_worker(void *arg, int i0, int i1) {
    gdn_ctx *c = arg;
    const int S_v = S_D_STATE, H = S_DT_RANK, K = S_N_GROUP;
    const float scale = 1.0f / sqrtf((float)S_v);
    /* debug: state norm per head (env WASTE-like trace) */
    static int trace = -1;
    if (trace < 0) trace = getenv("QMA_TRACE") != NULL;
    if (trace && i0 == 0) {
        double sn = 0, gn = 0;
        for (int h = 0; h < H; h++) {
            const float *M = c->state + (size_t)h * S_v * S_v;
            for (int i = 0; i < S_v * S_v; i++) sn += (double)M[i] * M[i];
            gn += fabsf(c->gate[h]);
        }
        /* per-token input norms */
        fprintf(stderr, "[statenorm] ||S||=%.3e mean|gate|=%.3f q0=%.3e k0=%.3e v0=%.3e\n",
                sqrt(sn), gn / H,
                c->q[0], c->k[0], c->v[0]);
    }
    for (int h = i0; h < i1; h++) {
        float *M = c->state + (size_t)h * S_v * S_v;
        const int qh = h % K;               /* q/k head (repeated) */
        for (int t = 0; t < c->T; t++) {
            const float *qp = c->q + (size_t)t * c->tstride + (size_t)qh * S_v;
            const float *kp = c->k + (size_t)t * c->tstride + (size_t)qh * S_v;
            const float *vp = c->v + (size_t)t * c->tstride + (size_t)h * S_v;
            float *op = c->out + (size_t)t * H * S_v + (size_t)h * S_v;
            const float g = c->gate[(size_t)t * H + h];
            const float b = c->beta[(size_t)t * H + h];
            const float expg = expf(g);

            float dq[S_D_STATE], dk[S_D_STATE];
            for (int j = 0; j < S_v; j++) {
                const float *Mj = M + (size_t)j * S_v;
                float s1 = 0.0f, s2 = 0.0f;
                for (int i = 0; i < S_v; i++) { s1 += Mj[i] * qp[i]; s2 += Mj[i] * kp[i]; }
                dq[j] = s1; dk[j] = s2;
            }
            float kq = 0.0f;
            for (int i = 0; i < S_v; i++) kq += kp[i] * qp[i];

            for (int j = 0; j < S_v; j++) {
                const float delta = (vp[j] - expg * dk[j]) * b;
                float *Mj = M + (size_t)j * S_v;
                for (int i = 0; i < S_v; i++) Mj[i] = expg * Mj[i] + kp[i] * delta;
                op[j] = (expg * dq[j] + delta * kq) * scale;
            }
        }
    }
}

/* ---------------- rope (IMROPE, text-only: p_t = p_h = p_w = p_e) ---------- */
/*
 * x: [T][n_head*head_dim] token-major, head h at t*n_head*head_dim + h*head_dim
 * rotates dims 0..63 in pairs (j, j+32): theta_j = pos * freq_base^(-2j/64)
 * (ref: ggml_mrope_cache_init + rotate_pairs with is_imrope)
 */
static void rope_imrope(float *x, int n_head, int head_dim, int hstride, int T, const int *pos) {
    const float theta_scale = powf(FREQ_BASE, -2.0f / N_ROT);
    for (int t = 0; t < T; t++) {
        const int p = pos[t];
        float cosv[N_ROT / 2], sinv[N_ROT / 2];
        float theta = (float)p;
        for (int j = 0; j < N_ROT / 2; j++) {
            cosv[j] = cosf(theta);
            sinv[j] = sinf(theta);
            theta *= theta_scale;
        }
        for (int h = 0; h < n_head; h++) {
            float *v = x + (size_t)t * n_head * hstride + (size_t)h * hstride;
            for (int j = 0; j < N_ROT / 2; j++) {
                float x0 = v[j], x1 = v[j + N_ROT / 2];
                v[j]             = x0 * cosv[j] - x1 * sinv[j];
                v[j + N_ROT / 2] = x0 * sinv[j] + x1 * cosv[j];
            }
        }
    }
}

/* ---------------- attention (causal, GQA kv_head = q_head/6) --------------- */
/*
 * Q: [T][12288] token-major (head h at t*12288 + h*512: 256 Q + 256 gate)
 * K,V: [T][1024] (head h at t*1024 + h*256)
 * gate: [T][6144]
 * cache: [2][4][n_ctx][256] floats (K then V)
 * out: [T][6144]
 */
typedef struct {
    const float *Q, *K, *V, *gate;
    float *out, *cache;
    int T, n_ctx, n_pos;
    int lay;      /* attention layer index (salience array) */
    int n_live;   /* live positions to attend (hcm_live[0..n_live)) */
    int n4;       /* end of all q4: hcm_live[0..n4) q4, [n4..n_live) q2 */
    int n_ring_q4;/* hcm_live[0..n_ring_q4) ring slots, [n_ring_q4..n4) arena */
    int rebuilt_at; /* n_pos when hcm_live was built; positions in
                       [rebuilt_at, n_pos) were appended since and must be
                       attended directly (they are not in the live index) */
    const uint8_t *pin_kv;  /* this layer's H2O arena (NULL if unused) */
    const uint8_t *arc_kv;  /* this layer's agent archive arena (NULL if unused) */
} attn_ctx;

/* ---- hierarchical context memory (H2O-style eviction + salience EMA) ----
 * Default ON. The KV cache stays append-only in the file; a "live index"
 * selects which positions attention actually reads each maintenance pass:
 *   sink (first 4, StreamingLLM) + hot window (last W_HOT) + pinned
 *   heavy-hitters (top HCM_H_PIN by accumulated attention mass, H2O).
 * Every position's salience is the EMA of softmax mass it received, updated
 * in attn_worker where scores are already computed — no extra model needed.
 * Tokens that fall out of every tier are simply not attended (not needed).
 * Toggle off with QMA_NOEVICT=1. */
#define HCM_W_HOT     4096   /* q4 recent window (KIVI: crucial, now q4)     */
#define HCM_H_PIN     2048   /* heavy-hitter budget (H2O: ~20% but cap small) */
#define HCM_W_GIST    32768  /* q2 gist window beyond hot (larger ctx pre-eviction) */
#define HCM_SINK      4      /* attention sinks (StreamingLLM: 4 tokens)      */
#define HCM_ALPHA     0.05f  /* salience EMA rate (~20-token half-life)        */
#define HCM_QF_BIAS   2.0f   /* Q-Hitter: weight of quantization-friendliness in
                                 pin selection (qf is relative error ~0.05)      */
#define HCM_REBUILD   128    /* re-rank every N tokens (amortized O(n/K))      */
/* Static ring: n_ctx is the FIXED working size (sink reserved, everything
   after wraps modulo n_ctx-HCM_SINK). The ring turning is the eviction:
   n_pos grows forever, old slots get recycled by new tokens, and the KV
   allocation stays constant — a Shepard-tone, perceived-infinite context.
   The live set (sink+hot+pinned+gist ~39K) always fits inside the ring
   (default n_ctx=65536), so no live position is ever recycled while live. */
static inline int hcm_slot(int p, int n_ctx) {
    if (p < HCM_SINK) return p;                    /* sinks: never recycled */
    return HCM_SINK + (p - HCM_SINK) % (n_ctx - HCM_SINK);
}
static int hcm_on = -1;      /* -1 = detect env, then 0/1 */
static int kvq_on = -1;      /* -1 = detect env (QMA_KVQ=0 disables), then 0/1 */
static int kvqi8_on = -1;    /* QMA_KVQI8=0 -> fp32-expansion score kernel */
static float *hcm_sal[N_ATTN_LAYER];   /* [lay][n_ctx] accumulated mass (ring slots) */
static float *hcm_qf;                   /* [n_ctx] Q-Hitter quantization friendliness
                                           (lower = quantizes better; 0 = unknown) */
static int    *hcm_live;               /* [n_ctx] live index; >= n_ctx encodes arena */
static uint8_t *hcm_tier;              /* [n_ctx] 0=evicted 1=q2 2=q4    */
static int     hcm_live_n = 0;         /* entries in hcm_live             */
static int     hcm_n4 = 0;             /* end of all q4 (ring+arena)       */
static int     hcm_n_ring_q4 = 0;      /* end of ring-q4 segment (start of arena) */
static int     hcm_rebuilt_at = 0;     /* n_pos when hcm_live was last built */
static int     hcm_ctx_cap = 0;        /* capacity (== n_ctx)             */
static int     hcm_next_rebuild = HCM_REBUILD;

/* ---- H2O heavy-hitter arena (protected, outside the ring) ----
 * Pinned heavy hitters are COPIED out of the ring into this arena, exactly
 * as H2O's paper does ("first K entries as heavy hitters, last K as most
 * recent"). Because the KV lives outside the ring, ring recycling can never
 * overwrite a pinned fact — it survives arbitrarily long, as long as it
 * keeps ranking in the top-HCM_H_PIN by accumulated attention. LRU-by-salience
 * replacement within the arena. */
static uint8_t *hcm_pin_kv[N_ATTN_LAYER];   /* [lay][2*N_HEAD_KV][HCM_H_PIN][KVQ_SLOT] */
static int32_t  hcm_pin_pos[HCM_H_PIN];         /* absolute pos in arena slot, -1 empty */
static float    hcm_pin_sal[N_ATTN_LAYER][HCM_H_PIN]; /* salience per (lay, slot) */
static int      hcm_pin_n = 0;              /* occupied arena slots */
#define HCM_ARENA_BASE (1 << 24)   /* hcm_live values >= this encode arena idx */

/* ---- agent archive (the manually-edited memory ring) ----
 * On top of the salience arena, the AGENT can pin text itself via the
 * memory_* tools: real K/V computed by a short eval, stored in a separate
 * protected arena, attended every token, never evicted by the salience
 * competition, surviving ring wrap and restarts (rebuilt from memory.json
 * at boot). This is what makes long-term consolidation reliable: the model
 * decides what is permanent instead of trusting a salience heuristic. */
#define HCM_ARC_MAX     1024    /* agent archive positions */
#define HCM_ARC_ENTRIES 64      /* max distinct memory entries */
#define HCM_ARC_BASE (1 << 25)  /* hcm_live encoding for agent-archive entries */
typedef struct { int32_t pos, n, abs0; uint64_t ts; char key[64]; } hcm_arc_ent_t;
static uint8_t      *hcm_arc_kv[N_ATTN_LAYER];   /* [lay][2*N_HEAD_KV][HCM_ARC_MAX][KVQ_SLOT] */
static hcm_arc_ent_t hcm_arc_ent[HCM_ARC_ENTRIES];
static int           hcm_arc_nent = 0;
static int           hcm_arc_n = 0;              /* arena append pointer */

int qma_kvq_on(void) {
    if (kvq_on < 0) {
        const char *e = getenv("QMA_KVQ");
        kvq_on = (e && e[0] == '0') ? 0 : 1;
    }
    return kvq_on;
}

static void hcm_init(int n_ctx) {
    if (hcm_on < 0) hcm_on = getenv("QMA_NOEVICT") == NULL;
    if (!hcm_on) return;
    if (n_ctx > hcm_ctx_cap) {
        for (int l = 0; l < N_ATTN_LAYER; l++) {
            free(hcm_sal[l]);
            hcm_sal[l] = calloc((size_t)n_ctx, sizeof(float));
        }
        free(hcm_live);
        hcm_live = malloc((size_t)n_ctx * sizeof(int));
        free(hcm_tier);
        hcm_tier = calloc((size_t)n_ctx, 1);   /* default q4 (2) for new pos */
        free(hcm_qf);
        hcm_qf = calloc((size_t)n_ctx, sizeof(float));
        /* H2O heavy-hitter arena: protected KV outside the ring */
        for (int l = 0; l < N_ATTN_LAYER; l++) {
            free(hcm_pin_kv[l]);
            hcm_pin_kv[l] = calloc((size_t)2 * N_HEAD_KV * HCM_H_PIN, KVQ_SLOT);
            /* agent archive arena (separate, agent-owned) */
            free(hcm_arc_kv[l]);
            hcm_arc_kv[l] = calloc((size_t)2 * N_HEAD_KV * HCM_ARC_MAX, KVQ_SLOT);
        }
        memset(hcm_pin_pos, -1, sizeof(hcm_pin_pos));
        memset(hcm_pin_sal, 0, sizeof(hcm_pin_sal));
        memset(hcm_arc_ent, 0, sizeof(hcm_arc_ent));
        hcm_arc_nent = 0;
        hcm_arc_n = 0;
        hcm_pin_n = 0;
        hcm_ctx_cap = n_ctx;
    }
}

/* rebuild the live index: sink + hot window + pinned heavy-hitters (q4)
   + q2 gist window. Called every HCM_REBUILD tokens (amortized). n_pos =
   total stored. Tiers: 2 = q4 (sink/hot/pinned), 1 = q2 (gist), 0 = evicted.
   hcm_live[0..hcm_n4) are q4 SLOTS; hcm_live[hcm_n4..hcm_live_n) are q2
   slots, so attn_worker runs two contiguous tier segments with zero modulo
   in the hot loop (slots precomputed here). RING: positions are absolute;
   the KV is a static ring, so every position we keep must be >= n_pos -
   ring_cap (older slots were recycled). */
static void hcm_rebuild(runstate_t *rs, int n_pos) {
    const int ring = hcm_ctx_cap - HCM_SINK;   /* recyclable slots */
    const int oldest = n_pos - ring;           /* first still-alive position */
    const int kvq = (kvq_on == 1);
    /* tier/salience are slot-indexed (n_ctx-sized): translate every
       absolute position through hcm_slot */
    if (!hcm_on || n_pos <= HCM_SINK + HCM_W_HOT) {
        /* nothing to evict yet: everything live at q4 (all < n_ctx anyway) */
        for (int p = 0; p < n_pos; p++) {
            hcm_live[p] = hcm_slot(p, hcm_ctx_cap);
            hcm_tier[hcm_slot(p, hcm_ctx_cap)] = 2;
        }
        hcm_live_n = n_pos;
        hcm_n4 = n_pos;
        hcm_n_ring_q4 = n_pos;
        /* agent archive is still attended during the all-live phase */
        if (kvq) {
            for (int e = 0; e < hcm_arc_nent; e++) {
                if (hcm_arc_ent[e].n <= 0) continue;
                for (int k = 0; k < hcm_arc_ent[e].n; k++)
                    hcm_live[hcm_live_n++] = HCM_ARC_BASE + hcm_arc_ent[e].pos + k;
            }
            hcm_n4 = hcm_live_n;
        }
        if (getenv("HCM_DBG")) fprintf(stderr, "[hcm] n_pos=%d all-live q4 (%d)\n", n_pos, hcm_live_n);
        return;
    }
    /* 1) sink + hot window are always live, q4 (ring slots) */
    int n = 0;
    int hot_lo = n_pos - HCM_W_HOT;
    if (hot_lo < oldest) hot_lo = oldest;      /* small ring: clamp to alive */
    for (int p = 0; p < HCM_SINK && p < hot_lo; p++) {
        hcm_live[n++] = hcm_slot(p, hcm_ctx_cap);
        hcm_tier[hcm_slot(p, hcm_ctx_cap)] = 2;
    }
    for (int p = hot_lo; p < n_pos; p++) {
        hcm_live[n++] = hcm_slot(p, hcm_ctx_cap);
        hcm_tier[hcm_slot(p, hcm_ctx_cap)] = 2;
    }
    int n_ring_q4 = n;

    /* 2) [moved to after arena/archive so hcm_live[] layout matches
       attn_worker: ring-q4, arena-q4, archive-q4, q2-gist] */

    /* 3) H2O heavy-hitter arena: still maintained for long-term pin
       survival across ring wrap, but ring tokens do NOT compete for it.
       The arena is populated from the hot window only (already q4 above),
       not from the noisy middle. This keeps the arena useful without
       dropping random ring tokens. */
    if (kvq) {
        /* promote hot-window tokens that have high salience into the arena
           for ring-wrap protection. Only hot-window candidates, not the
           noisy middle, so the selection is stable. */
        typedef struct { float sc; int pos; } hcm_ent;
        static hcm_ent *heap = NULL;
        static int heap_cap = 0;
        if (heap_cap < HCM_H_PIN) {
            free(heap);
            heap = malloc((size_t)HCM_H_PIN * sizeof(hcm_ent));
            heap_cap = HCM_H_PIN;
        }
        int hn = 0;
        
#define HCM_PUSH(SC, POS) do { \
        if (hn < HCM_H_PIN) { \
            int i = hn++; \
            heap[i].sc = (SC); heap[i].pos = (POS); \
            while (i > 0) { int par = (i-1)/2; if (heap[par].sc <= heap[i].sc) break; \
                hcm_ent t = heap[par]; heap[par] = heap[i]; heap[i] = t; i = par; } \
        } else if ((SC) > heap[0].sc) { \
            heap[0].sc = (SC); heap[0].pos = (POS); \
            int i = 0; \
            for (;;) { int lc = 2*i+1, rc = 2*i+2, sm = i; \
                if (lc < hn && heap[lc].sc < heap[sm].sc) sm = lc; \
                if (rc < hn && heap[rc].sc < heap[sm].sc) sm = rc; \
                if (sm == i) break; hcm_ent t = heap[sm]; heap[sm] = heap[i]; heap[i] = t; i = sm; } \
        } \
    } while (0)
        for (int p = hot_lo; p < n_pos; p++) {
            const int s = hcm_slot(p, hcm_ctx_cap);
            float sc = 0.0f;
            for (int l = 0; l < N_ATTN_LAYER; l++) sc += hcm_sal[l][s];
            HCM_PUSH(sc, p);
        }
        /* existing arena occupants compete to stay */
        for (int a = 0; a < HCM_H_PIN; a++) {
            if (hcm_pin_pos[a] < 0) continue;
            float sc = 0.0f;
            for (int l = 0; l < N_ATTN_LAYER; l++) sc += hcm_pin_sal[l][a];
            HCM_PUSH(sc, -1 - a);  /* negative pos encodes arena idx */
        }
#undef HCM_PUSH
        int new_arena[HCM_H_PIN];
        for (int i = 0; i < hn; i++) new_arena[i] = -1;
        for (int i = 0; i < hn; i++) {
            if (heap[i].pos < 0) new_arena[i] = -1 - heap[i].pos;
        }
        for (int i = 0; i < hn; i++) {
            if (heap[i].pos < 0) continue;  /* already an arena occupant */
            const int src = hcm_slot(heap[i].pos, hcm_ctx_cap);
            int slot = -1;
            for (int a = 0; a < HCM_H_PIN; a++) if (hcm_pin_pos[a] < 0) { slot = a; break; }
            if (slot < 0) {
                float worst = 1e30f; int wi = -1;
                for (int a = 0; a < HCM_H_PIN; a++) {
                    if (hcm_pin_pos[a] < 0) continue;
                    int kept = 0;
                    for (int k = 0; k < hn; k++) if (new_arena[k] == a) { kept = 1; break; }
                    if (kept) continue;
                    float sc = 0.0f;
                    for (int l = 0; l < N_ATTN_LAYER; l++) sc += hcm_pin_sal[l][a];
                    if (sc < worst) { worst = sc; wi = a; }
                }
                slot = wi;
            }
            if (slot < 0) continue;
            new_arena[i] = slot;
            for (int l = 0; l < N_ATTN_LAYER; l++) {
                uint8_t *dst = hcm_pin_kv[l];
                const uint8_t *ringb = (const uint8_t *)rs->kv_cache[l];
                size_t pstride = (size_t)hcm_ctx_cap * KVQ_SLOT;
                for (int kh = 0; kh < 2 * N_HEAD_KV; kh++) {
                    const uint8_t *srcp = ringb + (size_t)kh * pstride
                                                + (size_t)src * KVQ_SLOT;
                    uint8_t *dstp = dst + (size_t)kh * HCM_H_PIN * KVQ_SLOT
                                         + (size_t)slot * KVQ_SLOT;
                    memcpy(dstp, srcp, KVQ_SLOT);
                }
            }
            for (int l = 0; l < N_ATTN_LAYER; l++) hcm_pin_sal[l][slot] = hcm_sal[l][src];
            hcm_pin_pos[slot] = heap[i].pos;
        }
        hcm_pin_n = 0;
        for (int i = 0; i < hn; i++)
            if (new_arena[i] >= 0) hcm_pin_n++;
        for (int i = 0; i < hn; i++) {
            if (new_arena[i] < 0) continue;
            hcm_live[n++] = HCM_ARENA_BASE + new_arena[i];
        }
    }

    /* 4) agent archive: memory the model pinned itself — always q4, always
       attended, never evicted. Emitted into the q4 segment. */
    if (kvq) {
        for (int e = 0; e < hcm_arc_nent; e++) {
            if (hcm_arc_ent[e].n <= 0) continue;
            for (int k = 0; k < hcm_arc_ent[e].n; k++)
                hcm_live[n++] = HCM_ARC_BASE + hcm_arc_ent[e].pos + k;
        }
    }

    /* 2) Everything else still physically inside the ring buffer is q2,
       ALWAYS live — no top-K salience competition, no silent dropping.
       The EMA salience is too noisy and causes random tokens to be lost.
       Once a token ages past ring capacity (n_ctx) it is genuinely gone
       (standard sliding-window), not heuristically evicted. */
    int cand_lo = HCM_SINK;
    if (oldest > cand_lo) cand_lo = oldest;
    for (int p = cand_lo; p < hot_lo; p++) {
        const int s = hcm_slot(p, hcm_ctx_cap);
        hcm_live[n++] = s;
        hcm_tier[s] = 1;                       /* q2 gist, but LIVE */
    }

    hcm_live_n = n;
    hcm_n4 = n_ring_q4 + (kvq ? hcm_pin_n : 0) + (kvq ? hcm_arc_n : 0);
    hcm_n_ring_q4 = n_ring_q4;
    if (getenv("HCM_DBG")) fprintf(stderr,
        "[hcm] n_pos=%d live=%d (ring-q4=%d pin=%d arc=%d q2=%d) oldest=%d\n",
        n_pos, hcm_live_n, n_ring_q4, kvq ? hcm_pin_n : 0, kvq ? hcm_arc_n : 0,
        hcm_live_n - n_ring_q4 - (kvq ? hcm_pin_n : 0) - (kvq ? hcm_arc_n : 0), oldest);
}
int hcm_save(const char *path) {
    if (!hcm_on || hcm_ctx_cap <= 0) return 0;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    uint32_t nc = (uint32_t)hcm_ctx_cap;
    fwrite(&nc, 4, 1, f);
    for (int l = 0; l < N_ATTN_LAYER; l++)
        fwrite(hcm_sal[l], sizeof(float), (size_t)hcm_ctx_cap, f);
    fwrite(hcm_pin_pos, sizeof(int32_t), HCM_H_PIN, f);
    for (int l = 0; l < N_ATTN_LAYER; l++)
        fwrite(hcm_pin_sal[l], sizeof(float), HCM_H_PIN, f);
    for (int l = 0; l < N_ATTN_LAYER; l++)
        if (hcm_pin_kv[l])
            fwrite(hcm_pin_kv[l], (size_t)2 * N_HEAD_KV * HCM_H_PIN, KVQ_SLOT, f);
    fclose(f);
    return 0;
}

int hcm_load(const char *path, int n_ctx) {
    if (!hcm_on) return 0;
    hcm_init(n_ctx);
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    uint32_t nc = 0;
    if (fread(&nc, 4, 1, f) != 1 || (int)nc != n_ctx) { fclose(f); return -1; }
    for (int l = 0; l < N_ATTN_LAYER; l++) {
        if (fread(hcm_sal[l], sizeof(float), (size_t)n_ctx, f) != (size_t)n_ctx) {
            fclose(f); return -1;
        }
    }
    /* arena section is optional (older files lack it) */
    long pos0 = ftell(f);
    if (fread(hcm_pin_pos, sizeof(int32_t), HCM_H_PIN, f) != (size_t)HCM_H_PIN) {
        fseek(f, pos0, SEEK_SET); memset(hcm_pin_pos, -1, sizeof(hcm_pin_pos));
        hcm_pin_n = 0; fclose(f); return 0;
    }
    for (int l = 0; l < N_ATTN_LAYER; l++)
        if (fread(hcm_pin_sal[l], sizeof(float), HCM_H_PIN, f) != (size_t)HCM_H_PIN) {
            fclose(f); memset(hcm_pin_pos, -1, sizeof(hcm_pin_pos));
            hcm_pin_n = 0; return 0;
        }
    for (int l = 0; l < N_ATTN_LAYER; l++) {
        if (!hcm_pin_kv[l]) continue;
        if (fread(hcm_pin_kv[l], (size_t)2 * N_HEAD_KV * HCM_H_PIN, KVQ_SLOT, f) !=
            (size_t)(2 * N_HEAD_KV * HCM_H_PIN * KVQ_SLOT)) {
            fclose(f); memset(hcm_pin_pos, -1, sizeof(hcm_pin_pos));
            hcm_pin_n = 0; return 0;
        }
    }
    hcm_pin_n = 0;
    for (int a = 0; a < HCM_H_PIN; a++) if (hcm_pin_pos[a] >= 0) hcm_pin_n++;
    fclose(f);
    return 0;
}

/* ---- agent archive (the manually-edited memory ring) ----
 * The model's memory_* tools pin text it wants permanent: a short eval
 * computes real K/V, copied into the protected hcm_arc_kv arena, attended
 * on every token, never evicted by the salience competition, surviving
 * ring wrap and restarts (rebuilt from memory.json at boot). */

static int hcm_arc_find(const char *key) {
    for (int e = 0; e < hcm_arc_nent; e++)
        if (hcm_arc_ent[e].n > 0 && strcmp(hcm_arc_ent[e].key, key) == 0) return e;
    return -1;
}

/* compact active entries to the front of the agent arena + entry table */
static void hcm_arc_defrag(void) {
    int dst = 0, de = 0;
    for (int e = 0; e < hcm_arc_nent; e++) {
        if (hcm_arc_ent[e].n <= 0) continue;
        if (hcm_arc_ent[e].pos != dst) {
            for (int l = 0; l < N_ATTN_LAYER; l++) {
                uint8_t *base = hcm_arc_kv[l];
                for (int kh = 0; kh < 2 * N_HEAD_KV; kh++)
                    memmove(base + ((size_t)kh * HCM_ARC_MAX + (size_t)dst) * KVQ_SLOT,
                            base + ((size_t)kh * HCM_ARC_MAX + (size_t)hcm_arc_ent[e].pos) * KVQ_SLOT,
                            (size_t)hcm_arc_ent[e].n * KVQ_SLOT);
            }
            hcm_arc_ent[e].pos = dst;
        }
        if (de != e) hcm_arc_ent[de] = hcm_arc_ent[e];
        dst += hcm_arc_ent[e].n;
        de++;
    }
    hcm_arc_n = dst;
    hcm_arc_nent = de;
}

/* evaluate `content` into the ring and copy its K/V into the agent arena
   at [pin_pos, pin_pos+n). The eval advances n_pos and the recurrent
   state — the model becomes aware of what it wrote, exactly like appending
   a system message. Returns token count, or <0 on error. */
static int hcm_arc_embed(qma_t *m, runstate_t *rs, int n_threads,
                         const char *content, int pin_pos) {
    int *ids = malloc(sizeof(int) * 262144);
    if (!ids) return -1;
    int n = qma_tokenize(m, content, ids, 262144);
    if (n <= 0) { free(ids); return -1; }
    /* logits NULL: the LM head is skipped — we only need the K/V */
    int rc = qma_eval(m, rs, ids, n, NULL, n_threads, 0, 0);
    free(ids);
    if (rc != 0) return -1;
    const int abs0 = rs->n_pos - n;
    for (int l = 0; l < N_ATTN_LAYER; l++) {
        uint8_t *dst = hcm_arc_kv[l];
        const uint8_t *ringb = (const uint8_t *)rs->kv_cache[l];
        size_t pstride = (size_t)rs->n_ctx * KVQ_SLOT;
        for (int kh = 0; kh < 2 * N_HEAD_KV; kh++)
            for (int k = 0; k < n; k++) {
                const int s = hcm_slot(abs0 + k, rs->n_ctx);
                memcpy(dst + ((size_t)kh * HCM_ARC_MAX + (size_t)pin_pos + (size_t)k) * KVQ_SLOT,
                       ringb + (size_t)kh * pstride + (size_t)s * KVQ_SLOT, KVQ_SLOT);
            }
    }
    hcm_arc_n = pin_pos + n;
    return n;
}

/* after the entry table is updated, make the archive visible immediately */
static void hcm_arc_touch(runstate_t *rs) {
    hcm_rebuild(rs, rs->n_pos);
    hcm_rebuilt_at = rs->n_pos;
    hcm_next_rebuild = rs->n_pos + HCM_REBUILD;
}

int hcm_archive_write(qma_t *m, runstate_t *rs, const char *key,
                      const char *content, int n_threads) {
    if (!hcm_on || qma_kvq_on() != 1) return -2;   /* archive needs the q4 KV path */
    int old = hcm_arc_find(key);
    if (old >= 0) hcm_arc_ent[old].n = 0;          /* lazy free */
    if (hcm_arc_nent >= HCM_ARC_ENTRIES || hcm_arc_n + 16 > HCM_ARC_MAX) {
        hcm_arc_defrag();
        if (hcm_arc_nent >= HCM_ARC_ENTRIES) return -3;
    }
    int *ids = malloc(sizeof(int) * 262144);
    if (!ids) return -1;
    int n = qma_tokenize(m, content, ids, 262144);
    free(ids);
    if (n <= 0) return -1;
    if (hcm_arc_n + n > HCM_ARC_MAX) { hcm_arc_defrag(); }
    if (hcm_arc_n + n > HCM_ARC_MAX) return -3;    /* archive full */
    int pin_pos = hcm_arc_n;
    int rc = hcm_arc_embed(m, rs, n_threads, content, pin_pos);
    if (rc < 0) return rc;
    hcm_arc_ent[hcm_arc_nent].pos = pin_pos;
    hcm_arc_ent[hcm_arc_nent].n = rc;
    hcm_arc_ent[hcm_arc_nent].abs0 = rs->n_pos - rc;
    hcm_arc_ent[hcm_arc_nent].ts = (uint64_t)time(NULL);
    snprintf(hcm_arc_ent[hcm_arc_nent].key, sizeof(hcm_arc_ent[0].key), "%s", key);
    hcm_arc_nent++;
    hcm_arc_touch(rs);
    return rc;
}

int hcm_archive_append(qma_t *m, runstate_t *rs, const char *key,
                       const char *content, int n_threads) {
    if (!hcm_on || qma_kvq_on() != 1) return -2;
    int e = hcm_arc_find(key);
    if (e < 0) return hcm_archive_write(m, rs, key, content, n_threads);
    /* appending writes at the entry's end — that range must be free, i.e.
       the entry must be the LAST pinned range. Otherwise later entries sit
       there and would be overwritten; the caller re-embeds instead. */
    if (hcm_arc_ent[e].pos + hcm_arc_ent[e].n != hcm_arc_n) return -4;
    int *ids = malloc(sizeof(int) * 262144);
    if (!ids) return -1;
    int n = qma_tokenize(m, content, ids, 262144);
    free(ids);
    if (n <= 0) return -1;
    if (hcm_arc_n + n > HCM_ARC_MAX) hcm_arc_defrag();
    if (hcm_arc_n + n > HCM_ARC_MAX) return -3;
    int pin_pos = hcm_arc_ent[e].pos + hcm_arc_ent[e].n;   /* extend contiguously */
    int rc = hcm_arc_embed(m, rs, n_threads, content, pin_pos);
    if (rc < 0) return rc;
    hcm_arc_ent[e].n += rc;
    hcm_arc_ent[e].ts = (uint64_t)time(NULL);
    hcm_arc_touch(rs);
    return rc;
}

int hcm_archive_delete(runstate_t *rs, const char *key) {
    int e = hcm_arc_find(key);
    if (e < 0) return 0;
    hcm_arc_ent[e].n = 0;
    hcm_arc_defrag();
    hcm_rebuild(rs, rs->n_pos);
    return 1;
}

int hcm_archive_clear(runstate_t *rs) {
    hcm_arc_nent = 0;
    hcm_arc_n = 0;
    hcm_rebuild(rs, rs->n_pos);
    return 0;
}

int hcm_archive_count(void) {
    int c = 0;
    for (int e = 0; e < hcm_arc_nent; e++)
        if (hcm_arc_ent[e].n > 0) c++;
    return c;
}

/* per-thread persistent scores buffer (reused across heads/tokens) */
static __thread float *t_attn_scores = NULL;
static __thread size_t t_attn_scores_cap = 0;

/* fp32 K dot with 4 independent accumulators — breaks the 64-FMA latency
   chain of the single-accumulator form in the non-kvq attention paths */
static inline float f32_kvdot(const float *q, const float *kp, int hd) {
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f),
                a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
    int d = 0;
    for (; d + 16 <= hd; d += 16) {
        a0 = vmlaq_f32(a0, vld1q_f32(q + d),      vld1q_f32(kp + d));
        a1 = vmlaq_f32(a1, vld1q_f32(q + d + 4),  vld1q_f32(kp + d + 4));
        a2 = vmlaq_f32(a2, vld1q_f32(q + d + 8),  vld1q_f32(kp + d + 8));
        a3 = vmlaq_f32(a3, vld1q_f32(q + d + 12), vld1q_f32(kp + d + 12));
    }
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    for (; d < hd; d++) s += q[d] * kp[d];
    return s;
}

static void attn_worker(void *arg, int i0, int i1) {
    attn_ctx *c = arg;
    size_t need_scores = (size_t)(c->n_ctx + 512);
    if (need_scores > t_attn_scores_cap) {
        free(t_attn_scores);
        t_attn_scores = malloc(sizeof(float) * need_scores);
        t_attn_scores_cap = need_scores;
    }
    float *scores = t_attn_scores;
    const int hd = N_EMBD_HEAD;
    const float scale = 1.0f / sqrtf((float)hd);
    const int lay = c->lay;   /* attention layer index for salience */
    const int kvq = (kvq_on == 1);   /* quantized KV path */
    const uint8_t *cbase = (const uint8_t *)c->cache;
    const uint8_t *ring_lo = cbase;
    const uint8_t *ring_hi = cbase + (size_t)2 * N_HEAD_KV * c->n_ctx * KVQ_SLOT;
    const uint8_t *pin_lo = c->pin_kv;
    const uint8_t *pin_hi = c->pin_kv ? c->pin_kv + (size_t)2 * N_HEAD_KV * HCM_H_PIN * KVQ_SLOT : NULL;
    const uint8_t *arc_lo = c->arc_kv;
    const uint8_t *arc_hi = c->arc_kv ? c->arc_kv + (size_t)2 * N_HEAD_KV * HCM_ARC_MAX * KVQ_SLOT : NULL;
#define KP_CHECK(kp, lo, hi, label, idxval) do { \
        if ((kp) < (lo) || (hi) == NULL || (kp) + KVQ_SLOT > (hi)) { \
            fprintf(stderr, "[BOUNDS] %s OOB: kp=%p lo=%p hi=%p idx=%d h=%d t=%d\n", \
                    label, (void*)(kp), (void*)(lo), (void*)(hi), (int)(idxval), h, t); \
            fflush(stderr); \
        } \
    } while (0)
    for (int h = i0; h < i1; h++) {
        const int kh = h / (N_HEAD / N_HEAD_KV);
        float *Kc = c->cache + (size_t)kh * c->n_ctx * hd;
        float *Vc = c->cache + (size_t)(N_HEAD_KV + kh) * c->n_ctx * hd;
        /* kvq byte slabs: K head kh at kh*n_ctx*KVQ_SLOT, V after */
        const uint8_t *Ks = kvq ? cbase + (size_t)kh * c->n_ctx * KVQ_SLOT : NULL;
        const uint8_t *Vs = kvq ? cbase + (size_t)(N_HEAD_KV + kh) * c->n_ctx * KVQ_SLOT : NULL;
        /* H2O arena (kvq): K head kh at kh*HCM_H_PIN*KVQ_SLOT, V after */
        const uint8_t *ApK = kvq && c->pin_kv ? c->pin_kv + (size_t)kh * HCM_H_PIN * KVQ_SLOT : NULL;
        const uint8_t *ApV = kvq && c->pin_kv ? c->pin_kv + (size_t)(N_HEAD_KV + kh) * HCM_H_PIN * KVQ_SLOT : NULL;
        /* agent archive (always q4): K head kh at kh*HCM_ARC_MAX*KVQ_SLOT */
        const uint8_t *ArK = kvq && c->arc_kv ? c->arc_kv + (size_t)kh * HCM_ARC_MAX * KVQ_SLOT : NULL;
        const uint8_t *ArV = kvq && c->arc_kv ? c->arc_kv + (size_t)(N_HEAD_KV + kh) * HCM_ARC_MAX * KVQ_SLOT : NULL;
        float *sal = hcm_on ? hcm_sal[lay] : NULL;
        if (kvqi8_on < 0)
            kvqi8_on = getenv("QMA_KVQI8") == NULL || getenv("QMA_KVQI8")[0] != '0';
        /* int8-quantized query scratch (one per head-token; the SDOT score
           kernel then runs per slot without re-widening K nibbles) */
        static _Thread_local int8_t  t_qi8[N_EMBD_HEAD];
        static _Thread_local float   t_qxs[N_EMBD_HEAD / 32];
        static _Thread_local int     t_qsum[N_EMBD_HEAD / 32];
        const int use_i8 = kvq && kvqi8_on;
        /* i8 scores win once enough slots amortize the one-off query
           quantization; at tiny attended counts fp32 expansion is cheaper */
        enum { KVQI8_MIN_SLOTS = 64 };
        for (int t = 0; t < c->T; t++) {
            const int abs = c->n_pos + t;
            const float *q = c->Q + (size_t)t * WQ_DIM + (size_t)h * (hd * 2);
            const int nl_est = (hcm_on && c->n_live > 0) ? c->n_live : (abs + 1);
            const int nk_est = nl_est + (abs - (hcm_on ? hcm_rebuilt_at : 0) + 1);
            const int i8_here = use_i8 && nk_est >= KVQI8_MIN_SLOTS;
            if (i8_here) kvq_q_quant(q, hd, t_qi8, t_qxs, t_qsum);
            const int8_t *qi8 = i8_here ? t_qi8 : NULL;
            const float   *qsx = i8_here ? t_qxs : NULL;
            const int     *qss = i8_here ? t_qsum : NULL;
            /* K/V were appended to the cache by the caller (see
               kv_append below) before the parallel head loop ran.
               Attention reads the LIVE index (sink + hot + pinned + gist)
               instead of every position 0..abs — this is what caps the
               O(n) growth. New positions in this batch (>= n_pos) are
               always live. */
            const int nl = (hcm_on && c->n_live > 0) ? c->n_live : (abs + 1);
            const int use_live = (hcm_on && c->n_live > 0);
            float maxs = -1e30f;
            if (use_live) {
                /* three contiguous segments, no per-position branch:
                   [0,n_ring_q4) ring q4, [n_ring_q4,n4) arena q4 (H2O
                   protected heavy hitters), [n4,nl) ring q2 gist. */
                const int n4 = c->n4;
                const int n_rq = c->n_ring_q4;
                if (kvq) {
                    int j = 0;
                    for (; j < n_rq && j < nl; j++) {
                        const int sl = hcm_live[j];   /* ring slot */
                        const uint8_t *kp = Ks + (size_t)sl * KVQ_SLOT;
                        KP_CHECK(kp, ring_lo, ring_hi, "ring-q4", sl);
                        float s = (qi8 ? kvq4_dot_q8(kp, qi8, qsx, qss, hd)
                                   : kvq4_dot(kp, q, hd)) * scale;
                        scores[j] = s;
                        if (s > maxs) maxs = s;
                    }
                    for (; j < n4 && j < nl; j++) {
                        /* arena segment: H2O pin entries (HCM_ARENA_BASE) and
                           agent-archive entries (HCM_ARC_BASE) — one predictable
                           branch per position, both q4 */
                        const int v = hcm_live[j];
                        const uint8_t *ak = v >= HCM_ARC_BASE ? ArK : ApK;
                        const int a = v - (v >= HCM_ARC_BASE ? HCM_ARC_BASE : HCM_ARENA_BASE);
                        const uint8_t *kp = ak + (size_t)a * KVQ_SLOT;
                        KP_CHECK(kp, (v >= HCM_ARC_BASE ? arc_lo : pin_lo),
                                     (v >= HCM_ARC_BASE ? arc_hi : pin_hi),
                                     (v >= HCM_ARC_BASE ? "archive" : "pin-arena"), a);
                        float s = (qi8 ? kvq4_dot_q8(kp, qi8, qsx, qss, hd)
                                   : kvq4_dot(kp, q, hd)) * scale;
                        scores[j] = s;
                        if (s > maxs) maxs = s;
                    }
                    for (; j < nl; j++) {
                        const int sl = hcm_live[j];   /* ring slot */
                        const uint8_t *kp = Ks + (size_t)sl * KVQ_SLOT + KVQ2_OFF;
                        KP_CHECK(kp - KVQ2_OFF, ring_lo, ring_hi, "ring-q2", sl);
                        float s = kvq2_dot(kp, q, hd) * scale;
                        scores[j] = s;
                        if (s > maxs) maxs = s;
                    }
                } else {
                    for (int j = 0; j < nl; j++) {
                        const int sl = hcm_live[j];   /* ring slot (fp32: no arena) */
                        const float *kp = Kc + (size_t)sl * hd;
                        float s = f32_kvdot(q, kp, hd) * scale;
                        scores[j] = s;
                        if (s > maxs) maxs = s;
                    }
                }
                /* positions appended since the last rebuild — the live
                   index does not cover [rebuilt_at, n_pos), so attend them
                   directly. Includes the current batch AND any tokens
                   generated between rebuilds (e.g. the previous turn's
                   reply). Without this, multi-turn would lose everything
                   written since the last 128-token rebuild. */
                const int batch_lo = c->rebuilt_at;
                for (int p = batch_lo; p <= abs; p++) {
                    float s;
                    if (kvq) {
                        const uint8_t *kp = Ks + (size_t)hcm_slot(p, c->n_ctx) * KVQ_SLOT;
                        KP_CHECK(kp, ring_lo, ring_hi, "batch-lo", hcm_slot(p, c->n_ctx));
                        s = (qi8 ? kvq4_dot_q8(kp, qi8, qsx, qss, hd)
                               : kvq4_dot(kp, q, hd)) * scale;
                    } else {
                        const float *kp = Kc + (size_t)hcm_slot(p, c->n_ctx) * hd;
                        s = f32_kvdot(q, kp, hd) * scale;
                    }
                    scores[nl + (p - batch_lo)] = s;
                    if (s > maxs) maxs = s;
                }
            } else {
                /* dense: every position 0..abs (HCM disabled or pre-first
                   rebuild). kvq slots are all-q4 here (new tokens). */
                for (int p = 0; p <= abs; p++) {
                    float s;
                    if (kvq) {
                        const uint8_t *kp = Ks + (size_t)hcm_slot(p, c->n_ctx) * KVQ_SLOT;
                        KP_CHECK(kp, ring_lo, ring_hi, "dense", hcm_slot(p, c->n_ctx));
                        s = (qi8 ? kvq4_dot_q8(kp, qi8, qsx, qss, hd)
                               : kvq4_dot(kp, q, hd)) * scale;
                    } else {
                        const float *kp = Kc + (size_t)hcm_slot(p, c->n_ctx) * hd;
                        s = f32_kvdot(q, kp, hd) * scale;
                    }
                    scores[p] = s;
                    if (s > maxs) maxs = s;
                }
            }
            const int nk = use_live ? (nl + (abs - c->rebuilt_at + 1)) : (abs + 1);
            float sum = 0.0f;
            {
                /* vectorized softmax: scores are contiguous; subtract max,
                   exp in NEON, accumulate the normalizer */
                const float32x4_t vmax = vdupq_n_f32(maxs);
                float32x4_t sacc = vdupq_n_f32(0.0f);
                int j = 0;
                for (; j + 4 <= nk; j += 4) {
                    float32x4_t s = vsubq_f32(vld1q_f32(scores + j), vmax);
                    s = v_expf32x4(s);
                    vst1q_f32(scores + j, s);
                    sacc = vaddq_f32(sacc, s);
                }
                sum = vaddvq_f32(sacc);
                for (; j < nk; j++) { scores[j] = expf(scores[j] - maxs); sum += scores[j]; }
            }
            float *o = c->out + (size_t)t * WO_DIM + (size_t)h * hd;
            /* NEON weighted V accumulation + salience EMA fusion: one walk
               over the live segments does both the weighted V sum AND the
               H2O heavy-hitter update (previously a second full segment
               walk with a division per entry; now 1/sum precomputed). */
            {
                const float inv = 1.0f / sum;
                const int do_sal = sal && use_live;
                float32x4_t oacc[64];  /* hd/4 = 64 */
                for (int d4 = 0; d4 < hd / 4; d4++) oacc[d4] = vdupq_n_f32(0.0f);
                float *oaccf = (float *)oacc;
                if (use_live) {
                    const int n4 = c->n4;
                    const int n_rq = c->n_ring_q4;
                    int j = 0;
                    if (kvq) {
                        for (; j < n_rq && j < nl; j++) {
                            const int sl = hcm_live[j];   /* ring slot */
                            kvq4_vacc(Vs + (size_t)sl * KVQ_SLOT,
                                      scores[j], oaccf, hd);
                            if (do_sal)
                                sal[sl] += HCM_ALPHA * (scores[j] * inv - sal[sl]);
                        }
                        for (; j < n4 && j < nl; j++) {
                            const int v = hcm_live[j];   /* pin or agent arena */
                            const uint8_t *av = v >= HCM_ARC_BASE ? ArV : ApV;
                            const int a = v - (v >= HCM_ARC_BASE ? HCM_ARC_BASE : HCM_ARENA_BASE);
                            kvq4_vacc(av + (size_t)a * KVQ_SLOT,
                                      scores[j], oaccf, hd);
                            /* pinned facts keep competing for the top set;
                               agent-archive entries have no salience */
                            if (do_sal && v < HCM_ARC_BASE)
                                hcm_pin_sal[lay][a] +=
                                    HCM_ALPHA * (scores[j] * inv - hcm_pin_sal[lay][a]);
                        }
                        for (; j < nl; j++) {
                            const int sl = hcm_live[j];   /* ring slot */
                            kvq2_vacc(Vs + (size_t)sl * KVQ_SLOT + KVQ2_OFF,
                                      scores[j], oaccf, hd);
                            if (do_sal)
                                sal[sl] += HCM_ALPHA * (scores[j] * inv - sal[sl]);
                        }
                    } else {
                        for (int j = 0; j < nl; j++) {
                            const int sl = hcm_live[j];   /* ring slot (fp32) */
                            const float sp = scores[j];
                            const float *vp = Vc + (size_t)sl * hd;
                            float32x4_t spv = vdupq_n_f32(sp);
                            for (int d4 = 0; d4 < hd / 4; d4++)
                                oacc[d4] = vmlaq_f32(oacc[d4], spv, vld1q_f32(vp + d4 * 4));
                            if (do_sal)
                                sal[sl] += HCM_ALPHA * (sp * inv - sal[sl]);
                        }
                    }
                    /* post-rebuild tail + current batch: all q4 */
                    const int batch_lo = c->rebuilt_at;
                    for (int p = batch_lo; p <= abs; p++) {
                        const float sp = scores[nl + (p - batch_lo)];
                        const int sl = hcm_slot(p, c->n_ctx);
                        if (kvq) {
                            kvq4_vacc(Vs + (size_t)sl * KVQ_SLOT,
                                      sp, oaccf, hd);
                        } else {
                            const float *vp = Vc + (size_t)sl * hd;
                            float32x4_t spv = vdupq_n_f32(sp);
                            for (int d4 = 0; d4 < hd / 4; d4++)
                                oacc[d4] = vmlaq_f32(oacc[d4], spv, vld1q_f32(vp + d4 * 4));
                        }
                        if (do_sal)
                            sal[sl] += HCM_ALPHA * (sp * inv - sal[sl]);
                    }
                } else {
                    for (int p = 0; p <= abs; p++) {
                        const float sp = scores[p];
                        if (kvq) {
                            kvq4_vacc(Vs + (size_t)hcm_slot(p, c->n_ctx) * KVQ_SLOT,
                                      sp, oaccf, hd);
                        } else {
                            const float *vp = Vc + (size_t)hcm_slot(p, c->n_ctx) * hd;
                            float32x4_t spv = vdupq_n_f32(sp);
                            for (int d4 = 0; d4 < hd / 4; d4++)
                                oacc[d4] = vmlaq_f32(oacc[d4], spv, vld1q_f32(vp + d4 * 4));
                        }
                    }
                }
                for (int d4 = 0; d4 < hd / 4; d4++)
                    vst1q_f32(o + d4 * 4, vmulq_n_f32(oacc[d4], inv));
            }
        }
    }
#undef KP_CHECK
}

/* append K/V for all kv heads at the given positions (must run before the
   parallel attention so GQA readers never race the writer). With kvq on,
   each position slot stores BOTH the q4 (offset 0) and q2 (offset KVQ2_OFF)
   records, quantized from the same fp32 — tier transitions are tag flips.
   RING: the KV is a static ring; positions map to slots via hcm_slot. When
   a slot is recycled (new position overwrites an old one), its salience
   and tier are reset — the ring turning IS the eviction. */
static void kv_append(const attn_ctx *c) {
    const int hd = N_EMBD_HEAD;
    const int kvq = (kvq_on == 1);
    uint8_t *cbase = (uint8_t *)c->cache;
    for (int t = 0; t < c->T; t++) {
        const int abs = c->n_pos + t;
        const int slot = hcm_slot(abs, c->n_ctx);
        /* Q-Hitter QF: average relative q4 error of K and V for this token
           (lower = quantizes better). Stored per slot; the H2O pin selection
           prefers quantization-friendly tokens so the q4/q2 tiers hold
           higher-fidelity KV. Zero on recycled slots = unknown, treated as
           neutral in selection. */
        if (kvq && hcm_on) {
            float qf_k = 0, qf_v = 0;
            for (int kh = 0; kh < N_HEAD_KV; kh++) {
                const uint8_t *Kslot = cbase + (size_t)kh * c->n_ctx * KVQ_SLOT
                                             + (size_t)slot * KVQ_SLOT;
                const uint8_t *Vslot = cbase + (size_t)(N_HEAD_KV + kh) * c->n_ctx * KVQ_SLOT
                                             + (size_t)slot * KVQ_SLOT;
                const float *kv = c->K + (size_t)t * WKV_DIM + (size_t)kh * hd;
                const float *vv = c->V + (size_t)t * WKV_DIM + (size_t)kh * hd;
                qf_k += kvq4_qf(Kslot, kv, hd);
                qf_v += kvq4_qf(Vslot, vv, hd);
            }
            hcm_qf[slot] = (qf_k + qf_v) / (2.0f * N_HEAD_KV);
        }
        for (int kh = 0; kh < N_HEAD_KV; kh++) {
            if (kvq) {
                uint8_t *Kslot = cbase + (size_t)kh * c->n_ctx * KVQ_SLOT
                                         + (size_t)slot * KVQ_SLOT;
                uint8_t *Vslot = cbase + (size_t)(N_HEAD_KV + kh) * c->n_ctx * KVQ_SLOT
                                         + (size_t)slot * KVQ_SLOT;
                const float *kv = c->K + (size_t)t * WKV_DIM + (size_t)kh * hd;
                kvq4_quant(kv, hd, Kslot);
                kvq2_quant(kv, hd, Kslot);
                const float *vv = c->V + (size_t)t * WKV_DIM + (size_t)kh * hd;
                kvq4_quant(vv, hd, Vslot);
                kvq2_quant(vv, hd, Vslot);
            } else {
                float *Kc = c->cache + (size_t)kh * c->n_ctx * hd;
                float *Vc = c->cache + (size_t)(N_HEAD_KV + kh) * c->n_ctx * hd;
                const float *kv = c->K + (size_t)t * WKV_DIM + (size_t)kh * hd;
                memcpy(Kc + (size_t)slot * hd, kv, hd * sizeof(float));
                const float *vv = c->V + (size_t)t * WKV_DIM + (size_t)kh * hd;
                memcpy(Vc + (size_t)slot * hd, vv, hd * sizeof(float));
            }
        }
        /* recycled slot: this position now owns the slot, so the old
           occupant's salience/tier must not leak into the new token */
        if (hcm_on) {
            hcm_tier[slot] = 2;   /* new tokens start q4 */
            for (int l = 0; l < N_ATTN_LAYER; l++) hcm_sal[l][slot] = 0.0f;
        }
    }
}


/* ---------------- expert streaming (waste-style bounded LFRU cache) ------- */
/*
 * Faithful port of waste's expert streaming:
 *   - a bounded user-space cache (LFRU policy) holds expert records;
 *     one record == one expert's gate+up+down slabs concatenated;
 *   - reader threads keep reads in flight while the matmuls run:
 *     hint() names the layer's selected experts before the expert loop,
 *     each get() releases one more read into the pipe;
 *   - prefetch() speculatively fills the NEXT layer's predicted experts
 *     (real next-layer router on an approximate input), at low priority.
 * The trunk (norms, attn, conv, shexp, gate_inp) is still madvise'd a few
 * layers ahead — it is small and always used (waste wires it resident).
 */

/* one expert record = gate slab + up slab + down slab, concatenated.
   gate/up are Q4_K; down is Q4_K or Q6_K depending on the layer. */
/* per-layer expert slab byte sizes (one expert's worth of each tensor).
   With UD mixes gate/up/down can each be a different quant. */
static void layer_slab_bytes(qma_t *m, int il,
                             size_t *ge, size_t *ue, size_t *dn) {
    const size_t slab = (size_t)N_EMBD * N_FF_EXP;
    *ge = slab * qma_blk_size(m->layers[il].t_gate_exps) / QK_K;
    *ue = slab * qma_blk_size(m->layers[il].t_up_exps) / QK_K;
    *dn = slab * qma_blk_size(m->layers[il].t_down_exps) / QK_K;
}

static size_t layer_rec_bytes(qma_t *m, int il) {
    size_t ge, ue, dn;
    layer_slab_bytes(m, il, &ge, &ue, &dn);
    return ge + ue + dn;
}

/* pread-all: a short read is legal, loop until the whole range lands. */
static int pread_all(int fd, void *dst, size_t n, size_t off)
{
    uint8_t *p = (uint8_t *)dst;
    while (n) {
        const ssize_t r = pread(fd, p, n, (off_t)off);
        if (r <= 0) return -1;
        p += r; n -= (size_t)r; off += (size_t)r;
    }
    return 0;
}

/* The cache's fetch callback: pread this expert's three slabs from the
   GGUF into the record buffer. Callable from reader threads concurrently. */
/* Prefetch an expert's three slabs into page cache via readahead(2).
   The model is mmap'd; madvise(WILLNEED) is a no-op on this device (FBE),
   but readahead() actually pulls the pages. The matmul then reads them
   zero-copy through the mapping with page faults instead of disk seeks. */
static void expert_readahead(qma_t *m, int layer, int expert) {
    if (layer < 0 || layer >= N_LAYER || expert < 0 || expert >= N_EXPERT) return;
    size_t ge, ue, dn; layer_slab_bytes(m, layer, &ge, &ue, &dn);
    readahead(m->fd, m->layers[layer].off_gate_exps + (size_t)expert * ge, ge);
    readahead(m->fd, m->layers[layer].off_up_exps   + (size_t)expert * ue, ue);
    readahead(m->fd, m->layers[layer].off_down_exps + (size_t)expert * dn, dn);
}

static int expert_fetch(void *user, int layer, int expert, uint8_t *dst)
{
    qma_t *m = (qma_t *)user;
    if (layer < 0 || layer >= N_LAYER || expert < 0 || expert >= N_EXPERT)
        return -1;
    size_t ge, ue, dn; layer_slab_bytes(m, layer, &ge, &ue, &dn);
    /* O_DIRECT path: expert bytes land straight in the pool — no page-cache
       copy at all. Only when the fd exists AND the destination is 4K-aligned
       (O_DIRECT requires an aligned buffer; pool slots are 16K-aligned). */
    /* Measured on-device: buffered reads + warm page cache beat O_DIRECT
       by ~15% end-to-end (locality turns the cache into a free tier).
       QMA_DIO=1 restores direct I/O. */
    static int use_dio_env = -1;
    if (use_dio_env < 0) use_dio_env = getenv("QMA_DIO") != NULL;
    const int use_dio = use_dio_env && (m->dio_fd >= 0) && (((uintptr_t)dst & 4095u) == 0);
    const int fd = use_dio ? m->dio_fd : m->fd;
    if (pread_all(fd, dst, ge, m->layers[layer].off_gate_exps + (size_t)expert * ge) != 0)
        return -1;
    if (pread_all(fd, dst + ge, ue, m->layers[layer].off_up_exps + (size_t)expert * ue) != 0)
        return -1;
    if (pread_all(fd, dst + ge + ue, dn, m->layers[layer].off_down_exps + (size_t)expert * dn) != 0)
        return -1;
    /* Buffered preads just populated the page cache with expert pages. The
       ecache pool now owns these bytes and experts are NEVER read via the
       mmap in this mode (trunk only), so drop the file-side copies: no
       double-buffer, no cache churn evicting trunk pages. No-op if the
       pages aren't resident; skipped entirely on the O_DIRECT path. */
    /* O_DIRECT reads never touch page cache -> nothing to drop. For the
       buffered fallback we still drop pages to keep RAM bounded (the old
       unconditional behavior); QMA_KEEPEXP=1 keeps them instead when the
       user prefers speed over a strict RAM ceiling. */
    if (!use_dio && m->map) {
        static int drop_exp = -1;
        if (drop_exp < 0) drop_exp = getenv("QMA_EXPDROP") != NULL;
        if (drop_exp) {            madvise(m->map + m->layers[layer].off_gate_exps + (size_t)expert * ge, ge, MADV_DONTNEED);
            madvise(m->map + m->layers[layer].off_up_exps   + (size_t)expert * ue, ue, MADV_DONTNEED);
            madvise(m->map + m->layers[layer].off_down_exps + (size_t)expert * dn, dn, MADV_DONTNEED);
        }
    }
    return 0;
}

/* Arm the expert cache: budget bytes across the whole model, reader
   threads. Call after load, once. budget 0 or nthreads 0 = off. */
void qma_ecache_arm(qma_t *m, size_t budget_bytes, int nthreads)
{
    size_t rec = 0;
    for (int il = 0; il < N_LAYER; il++) {
        size_t r = layer_rec_bytes(m, il);
        if (r > rec) rec = r;
    }
    if (qma_ecache_init(&m->ecache, budget_bytes, rec, 0) != 0) return;
    if (m->ecache.n_slots > 0) {
        if (qma_ecache_io_start(&m->ecache, expert_fetch, m, nthreads,
                                   nthreads > 0 ? 8 : 0) != 0)
            nthreads = 0;
        m->ecache_on = 1;
        if (getenv("QMA_EXPOFFS")) {
            fprintf(stderr, "expoffs:");
            for (int i = 0; i < N_LAYER; i++) fprintf(stderr, " %zu/%zu/%zu",
                m->layers[i].off_gate_exps, m->layers[i].off_up_exps, m->layers[i].off_down_exps);
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "qma: expert cache %d slots x %.1f MB (rec %zu B, io %d)\n",
                m->ecache.n_slots, (double)m->ecache.slot_bytes / 1048576.0,
                rec, nthreads);
    }
}

void qma_ecache_teardown(qma_t *m)
{
    if (m->ecache_on) {
        const qma_ecache *c = &m->ecache;
        fprintf(stderr, "qma: ecache hits=%llu misses=%llu (%.1f%%) spec=%llu bytes=%.2fMB\n",
                (unsigned long long)c->hits, (unsigned long long)c->misses,
                c->hits + c->misses ? 100.0 * c->hits / (c->hits + c->misses) : 0.0,
                (unsigned long long)c->spec_issued,
                (double)c->bytes_read / 1048576.0);
    }
    qma_ecache_free(&m->ecache);
    m->ecache_on = 0;
    free(g_def_sum); g_def_sum = NULL;
    free(g_def_cnt); g_def_cnt = NULL;
}

/* YALIS default vectors (waste gqdef): enable capture; optionally load
   precomputed per-(layer,expert) averages. File format: "BNDF" magic,
   N_LAYER, N_EXPERT, N_EMBD as u32, then counts u32[N_LAYER*N_EXPERT] and
   sums f32[N_LAYER*N_EXPERT*N_EMBD] (same layout as waste's WDF1). */
int qma_defaults_enable(qma_t *m, const char *load_path)
{
    (void)m;
    const size_t tot = (size_t)N_LAYER * N_EXPERT * N_EMBD;
    if (!g_def_sum) {
        g_def_sum = calloc(tot, sizeof(float));
        g_def_cnt = calloc((size_t)N_LAYER * N_EXPERT, sizeof(uint32_t));
        if (!g_def_sum || !g_def_cnt) { free(g_def_sum); free(g_def_cnt); g_def_sum = NULL; g_def_cnt = NULL; return -1; }
    }
    if (load_path) {
        FILE *f = fopen(load_path, "rb");
        if (!f) return -1;
        uint32_t magic, nl, ne, nh;
        if (fread(&magic, 4, 1, f) == 1 && fread(&nl, 4, 1, f) == 1 &&
            fread(&ne, 4, 1, f) == 1 && fread(&nh, 4, 1, f) == 1 &&
            magic == 0x46444e42 && nl == N_LAYER && ne == N_EXPERT && nh == N_EMBD) {
            if (fread(g_def_cnt, 4, (size_t)N_LAYER * N_EXPERT, f) ==
                    (size_t)N_LAYER * N_EXPERT &&
                fread(g_def_sum, 4, tot, f) == tot) {
                fprintf(stderr, "qma: loaded YALIS defaults from %s\n", load_path);
                fclose(f);
                return 0;
            }
        }
        fclose(f);
        return -1;
    }
    return 0;
}

static size_t lr_min[N_LAYER], lr_max[N_LAYER];
static size_t lr_embd_min, lr_embd_max, lr_out_min, lr_out_max;
static int lr_ready = 0;

void qma_prefetch_init(qma_t *m) {
    for (int i = 0; i < N_LAYER; i++) { lr_min[i] = SIZE_MAX; lr_max[i] = 0; }
    lr_embd_min = lr_out_min = SIZE_MAX;
    lr_embd_max = lr_out_max = 0;
    for (uint32_t i = 0; i < m->n_tensors; i++) {
        tensor_t *t = &m->tensors[i];
        size_t lo = t->off, hi = t->off + t->nbytes;
        /* the three expert tensors are streamed by the router-guided path;
           exclude them from the coarse trunk ranges */
        if (strstr(t->name, "ffn_gate_exps") || strstr(t->name, "ffn_up_exps") ||
            strstr(t->name, "ffn_down_exps"))
            continue;
        if (strncmp(t->name, "blk.", 4) == 0) {
            int il = atoi(t->name + 4);
            if (il >= 0 && il < N_LAYER) {
                if (lo < lr_min[il]) lr_min[il] = lo;
                if (hi > lr_max[il]) lr_max[il] = hi;
            }
        } else if (strcmp(t->name, "token_embd.weight") == 0) { lr_embd_min = lo; lr_embd_max = hi; }
        else if (strcmp(t->name, "output.weight") == 0) { lr_out_min = lo; lr_out_max = hi; }
    }
    lr_ready = 1;
}

static void prefetch_range(size_t lo, size_t hi, qma_t *m) {
    if (lo >= hi) return;
    /* Measured: posix_madvise(WILLNEED) is a no-op on FBE, but replacing it
       with readahead() here was a net LOSS at decode (4.5 -> 3.8 tok/s) —
       the trunk pages are already cached at short ctx and the extra
       readahead syscalls + page-cache churn cost more than they save.
       Keep the madvise (harmless no-op) rather than the harmful readahead. */
    posix_madvise(m->data + lo, hi - lo, POSIX_MADV_WILLNEED);
}

void qma_prefetch_layer(int il, int dist, qma_t *m) {
    if (!lr_ready) return;
    for (int d = 1; d <= dist; d++) {
        int j = il + d;
        if (j < N_LAYER) prefetch_range(lr_min[j], lr_max[j], m);
    }
}

/* Hand this layer's selected (dedup'd) expert ids to the cache: the first
   `depth` reads start now, each get() releases the next. Shared expert is
   small and always used, so it stays resident via the trunk madvise. */
static void qma_prefetch_experts(qma_t *m, int il, const int *ids, int n) {
    if (n <= 0) return;
    if (m->mmap_exps) {
        for (int i = 0; i < n; i++) expert_readahead(m, il, ids[i]);
        return;
    }
    if (!m->ecache_on) return;
    qma_ecache_hint(&m->ecache, il, ids, n);
}

/* Predict the next MoE layer's top experts and prefetch them (waste
   predict_next_moe + ecache_prefetch). `x` is this layer's post-norm FFN
   input, `r_l` the unnormalized post-attention residual, `sel` the current
   layer's routing. With YALIS defaults the prediction runs on the
   quasi-hidden state q = LN_{L+1}(r_l + d_l), d_l = Σ w_j·defaults[L][idx_j]
   — the paper's "one residual short" recovery. Without defaults it
   degrades to the plain rms_norm(x) baseline. */
static void qma_predict_prefetch(qma_t *m, int il, const float *x,
                                    const float *r_l, const moe_sel_t *sel) {
    if (!lr_ready || il + 1 >= N_LAYER) return;
    const int nl = il + 1;
    const float *gi = (const float *)m->layers[nl].ffn_gate_inp;
    const float *pn = (const float *)m->layers[nl].attn_post_norm;
    if (g_pf_trace < 0) g_pf_trace = getenv("QMA_PFTRACE") != NULL;
    if (!gi || !pn) return;
    /* predicted next input: q = LN_{L+1}(r_l + d_l), or rms_norm(x) */
    float q[N_EMBD];
    if (g_def_sum && g_def_cnt && r_l && sel) {
        const float *dL = g_def_sum + (size_t)il * N_EXPERT * N_EMBD;
        const uint32_t *cL = g_def_cnt + (size_t)il * N_EXPERT;
        for (int i = 0; i < N_EMBD; i++) q[i] = r_l[i];
        for (int k = 0; k < N_EXPERT_USED; k++) {
            const int ex = sel[0].sel[k];   /* token 0 drives the guess */
            const uint32_t c = cL[ex];
            if (!c) continue;
            const float *d = dL + (size_t)ex * N_EMBD;
            const float wj = sel[0].w[k] / (float)c;
            for (int i = 0; i < N_EMBD; i++) q[i] += wj * d[i];
        }
        double s = 0;
        for (int i = 0; i < N_EMBD; i++) s += (double)q[i] * q[i];
        float r = 1.0f / sqrtf((float)(s / N_EMBD) + RMS_EPS);
        for (int i = 0; i < N_EMBD; i++) q[i] = q[i] * r * pn[i];
    } else {
        double s = 0;
        for (int i = 0; i < N_EMBD; i++) s += (double)x[i] * x[i];
        float r = 1.0f / sqrtf((float)(s / N_EMBD) + RMS_EPS);
        for (int i = 0; i < N_EMBD; i++) q[i] = x[i] * r * pn[i];
    }
    /* score with the real next router */
    float mx = -1e30f;
    float sc[N_EXPERT];
    for (int e = 0; e < N_EXPERT; e++) {
        const float *g = gi + (size_t)e * N_EMBD;
        float v = 0;
        for (int i = 0; i < N_EMBD; i++) v += g[i] * q[i];
        sc[e] = v;
        if (v > mx) mx = v;
    }
    int ids[N_EXPERT_USED];
    for (int k = 0; k < N_EXPERT_USED; k++) {
        int best = 0; float bv = -1e30f;
        for (int e = 0; e < N_EXPERT; e++) if (sc[e] > bv) { bv = sc[e]; best = e; }
        ids[k] = best; sc[best] = -1e30f;
    }
    /* Speculative next-layer fill. Measured NET-NEGATIVE at ~54% routing
       prediction accuracy in every implementation tried (slot claims
       evicted good records; synchronous readahead stalled the layer).
       Off by default; QMA_SPECPREFETCH=1 re-enables readahead warming.
       Revisit when prediction accuracy improves (YALIS defaults load,
       or a trained light predictor). */
    static int spec_pf = -1;
    if (spec_pf < 0) spec_pf = getenv("QMA_SPECPREFETCH") != NULL;
    if (spec_pf)
        for (int k = 0; k < N_EXPERT_USED; k++)
            expert_readahead(m, nl, ids[k]);
    if (g_pf_trace) {
        for (int k = 0; k < N_EXPERT_USED; k++) g_pred_ids[nl][k] = ids[k];
        g_pred_gen[nl]++;
    }
}

/* ---------------- forward ---------------- */
#define MAX_CHUNK 512

typedef struct {
    float *x;     /* [T][5120]   hidden state */
    float *xn;    /* [T][5120]   norm output */
    float *x2;    /* [T][5120]   residual */
    float *ra;    /* [T][5120]   attn/ffn output */
    float *qfull; /* [T][12288]  Q+gate (attn) */
    float *qkv;   /* [T][10240]  wqkv output (recr) */
    float *kvK;   /* [T][1024]   K (attn) */
    float *kvV;   /* [T][1024]   V (attn) */
    float *z;     /* [T][6144]   gate (attn) / z (recr) / gated norm out */
    float *gkv;   /* [T][18432]  q48/k48/v48 (recr) */
    float *gdout; /* [T][6144]   GDN output */
    float *beta;  /* [T][48]     recr */
    float *gt;    /* [T][48]     recr gate */
    float *cin;   /* [10240][3+T] conv input (channel-major) */
    float *cout;  /* [10240][T]  conv output (channel-major) */
    int T;
} scratch_t;

static void scratch_alloc(runstate_t *rs, scratch_t *s) {
    /* persistent arena: allocate once, reuse across tokens. The old version
       malloc'd ~18 buffers (up to 10MB each) and freed them on EVERY eval
       call -- per-token ~180MB of heap churn, page faults and TLB thrash
       right in the weight-streaming hot path. */
    if (!rs->scratch) {
        size_t sz =
            sizeof(float)*(size_t)N_EMBD*MAX_CHUNK * 4 +            /* x xn x2 ra */
            sizeof(float)*(size_t)WQ_DIM*MAX_CHUNK +                 /* qfull */
            sizeof(float)*(size_t)QKV_DIM*MAX_CHUNK +                /* qkv */
            sizeof(float)*(size_t)WKV_DIM*MAX_CHUNK * 2 +            /* kvK kvV */
            sizeof(float)*(size_t)VAL_DIM*MAX_CHUNK * 2 +            /* z gdout */
            sizeof(float)*(size_t)(S_D_STATE*S_DT_RANK*3)*MAX_CHUNK +/* gkv */
            sizeof(float)*(size_t)S_DT_RANK*MAX_CHUNK * 2 +          /* beta gt */
            sizeof(float)*(size_t)CONV_DIM*(3+MAX_CHUNK) +           /* cin */
            sizeof(float)*(size_t)CONV_DIM*MAX_CHUNK;                /* cout */
        rs->scratch = malloc(sz);
        if (!rs->scratch) { fprintf(stderr, "qma: out of memory\n"); exit(1); }
        float *p = rs->scratch;
        s->x     = p; p += (size_t)N_EMBD*MAX_CHUNK;
        s->xn    = p; p += (size_t)N_EMBD*MAX_CHUNK;
        s->x2    = p; p += (size_t)N_EMBD*MAX_CHUNK;
        s->ra    = p; p += (size_t)N_EMBD*MAX_CHUNK;
        s->qfull = p; p += (size_t)WQ_DIM*MAX_CHUNK;
        s->qkv   = p; p += (size_t)QKV_DIM*MAX_CHUNK;
        s->kvK   = p; p += (size_t)WKV_DIM*MAX_CHUNK;
        s->kvV   = p; p += (size_t)WKV_DIM*MAX_CHUNK;
        s->z     = p; p += (size_t)VAL_DIM*MAX_CHUNK;
        s->gkv   = p; p += (size_t)(S_D_STATE*S_DT_RANK*3)*MAX_CHUNK;
        s->gdout = p; p += (size_t)VAL_DIM*MAX_CHUNK;
        s->beta  = p; p += (size_t)S_DT_RANK*MAX_CHUNK;
        s->gt    = p; p += (size_t)S_DT_RANK*MAX_CHUNK;
        s->cin   = p; p += (size_t)CONV_DIM*(3+MAX_CHUNK);
        s->cout  = p; p += (size_t)CONV_DIM*MAX_CHUNK;
        /* stash for later reset() reuse */
        rs->sx=s->x; rs->sxn=s->xn; rs->sx2=s->x2; rs->sra=s->ra;
        rs->sqfull=s->qfull; rs->sqkv=s->qkv; rs->skvK=s->kvK; rs->skvV=s->kvV;
        rs->sz=s->z; rs->sgkv=s->gkv; rs->sgdout=s->gdout;
        rs->sbeta=s->beta; rs->sgt=s->gt;
        rs->scin=s->cin; rs->scout=s->cout;
    } else {
        s->x=rs->sx; s->xn=rs->sxn; s->x2=rs->sx2; s->ra=rs->sra;
        s->qfull=rs->sqfull; s->qkv=rs->sqkv; s->kvK=rs->skvK; s->kvV=rs->skvV;
        s->z=rs->sz; s->gkv=rs->sgkv; s->gdout=rs->sgdout;
        s->beta=rs->sbeta; s->gt=rs->sgt;
        s->cin=rs->scin; s->cout=rs->scout;
    }
}

int runstate_init(runstate_t *rs, int n_ctx) {
    return runstate_init_kv(rs, n_ctx, NULL, 0);
}

/* File-backed KV: the KV cache is the only context state that grows with
   n_ctx (4KB/token x 10 attn layers). Heap-calloc'ing it caps context at
   ~30K tokens on this device; a sparse file + MAP_SHARED lets n_ctx go to
   1M+ with RAM cost = only the pages attention currently touches.
   ssm/conv state stays in heap (fixed ~89MB, independent of n_ctx). */
int runstate_init_kv(runstate_t *rs, int n_ctx, const char *path, int persist) {
    memset(rs, 0, sizeof(*rs));
    rs->n_ctx = n_ctx;
    rs->kv_fd = -1;
    rs->kv_map = NULL;
    rs->kv_persist = persist;
    rs->kv_path = path;

    const int kvq = qma_kvq_on();
    /* KV total size: N_ATTN_LAYER slabs x 2*N_HEAD_KV x n_ctx x (fp32 256
       floats, or KVQ_SLOT bytes per position in quantized mode) */
    const size_t pos_bytes = kvq ? KVQ_SLOT : (N_EMBD_HEAD * sizeof(float));
    const size_t kv_per_layer = (size_t)2 * N_HEAD_KV * n_ctx * pos_bytes;
    const size_t kv_total = kv_per_layer * N_ATTN_LAYER;

    int have_kv_file = 0;
    int want_anon = 0;   /* small non-persistent ctx: anonymous, no writeback */
    if (path && path[0]) {
        rs->kv_fd = open(path, O_RDWR | O_CREAT, 0600);
        if (rs->kv_fd >= 0) {
            /* format guard: an existing file from the other layout would be
               misread (fp32 slab != kvq slot). If it doesn't match this
               mode's expected size, truncate fresh. */
            struct stat st;
            if (fstat(rs->kv_fd, &st) == 0 && st.st_size > 0 &&
                (size_t)st.st_size != kv_total) {
                fprintf(stderr, "[kv] format mismatch (file %lld B, want %lld B "
                                "%s) — starting fresh\n",
                        (long long)st.st_size, (long long)kv_total,
                        kvq ? "kvq" : "fp32");
                if (ftruncate(rs->kv_fd, 0) != 0) { close(rs->kv_fd); rs->kv_fd = -1; }
            }
            if (rs->kv_fd >= 0 && ftruncate(rs->kv_fd, (off_t)kv_total) == 0)
                have_kv_file = 1;
        }
    } else if (kv_total < (size_t)1 << 30) {
        /* <1GB and not persisted: anonymous mmap. No temp file, no dirty
           page writeback pressure; pages are discarded under pressure. */
        want_anon = 1;
    } else {
        /* anonymous temp file in the state dir (large ctx needs the sparse
           file so storage pages it in/out instead of heap) */
        char tmp[512];
        snprintf(tmp, sizeof(tmp), "/data/data/com.termux/files/home/projects/qma/inference-engine-moe/.kv-%d", (int)getpid());
        rs->kv_fd = open(tmp, O_RDWR | O_CREAT | O_TRUNC, 0600);
        if (rs->kv_fd >= 0) {
            if (ftruncate(rs->kv_fd, (off_t)kv_total) == 0) {
                have_kv_file = 1;
                rs->kv_path = strdup(tmp);
            }
        }
    }
    if (want_anon) {
        void *base = mmap(NULL, kv_total, PROT_READ | PROT_WRITE,
                          MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        if (base == MAP_FAILED) return -1;
        rs->kv_map = base;
        rs->kv_map_len = kv_total;
        for (int il = 0; il < N_LAYER; il++) {
            if (IS_ATTN(il)) {
                rs->kv_cache[il / 4] = (float *)((uint8_t *)base + (size_t)(il / 4) * kv_per_layer);
            } else {
                rs->conv_state[il] = calloc(1, (size_t)CONV_DIM * 3 * sizeof(float));
                rs->ssm_state[il]  = calloc(1, (size_t)S_DT_RANK * S_D_STATE * S_D_STATE * sizeof(float));
                if (!rs->conv_state[il] || !rs->ssm_state[il]) return -1;
            }
        }
        return 0;
    }
    if (!have_kv_file) {
        /* fall back to heap (small n_ctx) */
        for (int il = 0; il < N_LAYER; il++) {
            if (IS_ATTN(il)) {
                rs->kv_cache[il / 4] = calloc(1, kv_per_layer);
                if (!rs->kv_cache[il / 4]) return -1;
            } else {
                rs->conv_state[il] = calloc(1, (size_t)CONV_DIM * 3 * sizeof(float));
                rs->ssm_state[il]  = calloc(1, (size_t)S_DT_RANK * S_D_STATE * S_D_STATE * sizeof(float));
                if (!rs->conv_state[il] || !rs->ssm_state[il]) return -1;
            }
        }
        return 0;
    }

    /* mmap the whole KV arena (file-backed for persist/large ctx) */
    void *base = mmap(NULL, kv_total, PROT_READ | PROT_WRITE, MAP_SHARED, rs->kv_fd, 0);
    if (base == MAP_FAILED) { close(rs->kv_fd); rs->kv_fd = -1; return -1; }
    rs->kv_map = base;
    rs->kv_map_len = kv_total;

    /* slice per-layer slabs */
    for (int il = 0; il < N_LAYER; il++) {
        if (IS_ATTN(il)) {
            rs->kv_cache[il / 4] = (float *)((uint8_t *)base + (size_t)(il / 4) * kv_per_layer);
        } else {
            rs->conv_state[il] = calloc(1, (size_t)CONV_DIM * 3 * sizeof(float));
            rs->ssm_state[il]  = calloc(1, (size_t)S_DT_RANK * S_D_STATE * S_D_STATE * sizeof(float));
            if (!rs->conv_state[il] || !rs->ssm_state[il]) return -1;
        }
    }
    return 0;
}

void runstate_free(runstate_t *rs) {
    /* file-backed KV: unmap + close (and unlink unless persisted) */
    if (rs->kv_map) {
        munmap(rs->kv_map, rs->kv_map_len);
        rs->kv_map = NULL;
    }
    if (rs->kv_fd >= 0) {
        close(rs->kv_fd);
        rs->kv_fd = -1;
        if (!rs->kv_persist && rs->kv_path) {
            unlink(rs->kv_path);
            free((void *)rs->kv_path);
            rs->kv_path = NULL;
        }
    }
    /* ssm/conv state stays heap-allocated in both modes */
    for (int il = 0; il < N_LAYER; il++) {
        if (!IS_ATTN(il)) {
            free(rs->conv_state[il]);
            free(rs->ssm_state[il]);
            rs->conv_state[il] = NULL;
            rs->ssm_state[il] = NULL;
        }
    }
    free(rs->scratch);
    rs->scratch = NULL;
    memset(rs, 0, sizeof(*rs));
}

/* ---- state persistence (ssm + conv + n_pos; KV lives in the mapped file) ---- */
int runstate_save(runstate_t *rs, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int32_t np = rs->n_pos;
    fwrite(&np, 4, 1, f);
    for (int il = 0; il < N_LAYER; il++) {
        if (IS_ATTN(il)) continue;
        fwrite(rs->conv_state[il], sizeof(float), (size_t)CONV_DIM * 3, f);
        fwrite(rs->ssm_state[il], sizeof(float),
               (size_t)S_DT_RANK * S_D_STATE * S_D_STATE, f);
    }
    fclose(f);
    return 0;
}

int runstate_load(runstate_t *rs, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    int32_t np = 0;
    if (fread(&np, 4, 1, f) != 1) { fclose(f); return -1; }
    rs->n_pos = np;
    for (int il = 0; il < N_LAYER; il++) {
        if (IS_ATTN(il)) continue;
        if (fread(rs->conv_state[il], sizeof(float), (size_t)CONV_DIM * 3, f) !=
                (size_t)CONV_DIM * 3) { fclose(f); return -1; }
        if (fread(rs->ssm_state[il], sizeof(float),
                  (size_t)S_DT_RANK * S_D_STATE * S_D_STATE, f) !=
                (size_t)S_DT_RANK * S_D_STATE * S_D_STATE) { fclose(f); return -1; }
    }
    fclose(f);
    return 0;
}

static void embed_row(qma_t *m, int id, float *out) {
    /* token_embd row id is a contiguous slab of N_EMBD weights (the vocab
       index is the slowest dim). Type is resolved at load time (m->t_token_embd)
       -- must NOT be assumed to be Q4_K, it may be Q6_K in this GGUF. */
    size_t bs = qma_blk_size(m->t_token_embd);
    const uint8_t *row = m->token_embd + (size_t)id * (size_t)(N_EMBD / QK_K) * bs;
    switch (m->t_token_embd) {
    case GGML_TYPE_Q4_K:   dequantize_row_q4_K((const block_q4_K *)row, out, N_EMBD); break;
    case GGML_TYPE_Q5_K:   dequantize_row_q5_K((const block_q5_K *)row, out, N_EMBD); break;
    case GGML_TYPE_Q6_K:   dequantize_row_q6_K((const block_q6_K *)row, out, N_EMBD); break;
    case GGML_TYPE_Q3_K:   dequantize_row_q3_K((const block_q3_K *)row, out, N_EMBD); break;
    case GGML_TYPE_IQ2_XS: dequantize_row_iq2_xs((const block_iq2_xs *)row, out, N_EMBD); break;
    case GGML_TYPE_IQ2_S:  dequantize_row_iq2_s((const block_iq2_s *)row, out, N_EMBD); break;
    case GGML_TYPE_Q8_0:   dequantize_row_q8_0(row, out, N_EMBD); break;
    default:               memset(out, 0, N_EMBD * sizeof(float)); break;
    }
}

int qma_eval(qma_t *m, runstate_t *rs, const int *tokens, int n_tokens,
                float *logits, int n_threads, int prefetch, int logits_all) {
    g_pf_dist = prefetch;
    if (g_pf_trace < 0) g_pf_trace = getenv("QMA_PFTRACE") != NULL;
    pool_init(n_threads);
    g_prof_cache = &m->ecache;
    timing_init();
    scratch_t s; scratch_alloc(rs, &s);
    const int T_MAX = MAX_CHUNK;

    /* GPU offload is a prefill-only feature: arm it once when a multi-token
       chunk is coming (decode stays on the CPU). */
    if (n_tokens > 1) cl_maybe_init(m);

    int done = 0;
    while (done < n_tokens) {
        const int T = (n_tokens - done < T_MAX) ? n_tokens - done : T_MAX;
        const int *tok = tokens + done;

        for (int t = 0; t < T; t++) embed_row(m, tok[t], s.x + (size_t)t * N_EMBD);
        if (g_timing) { timing_init(); }

        /* hierarchical context memory: init once per runstate, rebuild the
           live index every HCM_REBUILD tokens (amortized re-rank). The
           trigger uses the RUNNING position (rs->n_pos + done), not the
           pre-call value — rs->n_pos is frozen until this eval returns, so
           a multi-thousand-token prefill used to never re-fire mid-call and
           the dense "since last rebuild" tail grew to the whole prefill
           (O(n^2) attention). Rebuilds now re-fire at chunk boundaries, so
           that tail stays bounded by MAX_CHUNK. */
        if (hcm_on < 0) hcm_init(rs->n_ctx);
        const int pos_now = rs->n_pos + done;
        if (hcm_on && pos_now >= hcm_next_rebuild) {
            hcm_rebuild(rs, pos_now);
            hcm_rebuilt_at = pos_now;
            hcm_next_rebuild = pos_now + HCM_REBUILD;
        }

        for (int il = 0; il < N_LAYER; il++) {
            double tL0 = 0, tM = 0, tN = 0, tA = 0, tC = 0, tG = 0;
            if (g_timing) { tL0 = now_s(); tM = t_acc[0]; }
            /* decode: layer il will demand ~the same experts as last token.
               readahead() pulls those pages into page cache NOW (full layer
               of lead time) so the mmap matmul faults on resident pages. */
            if (m->mmap_exps && g_hist_valid[il]) {
                for (int k = 0; k < N_EXPERT_USED; k++)
                    if (g_hist_ids[il][k] >= 0)
                        expert_readahead(m, il, g_hist_ids[il][k]);
            } else if (g_hist_valid[il] && m->ecache_on) {
                extern void qma_ecache_prefetch(qma_ecache*, int, const int*, int);
                qma_ecache_prefetch(&m->ecache, il, g_hist_ids[il], N_EXPERT_USED);
            }
            qma_prefetch_layer(il, prefetch, m);

            if (g_timing) tN = now_s();
            rms_norm_m(s.xn, s.x, (const float *)m->layers[il].attn_norm, N_EMBD, T, RMS_EPS);
            if (g_timing) tN = now_s() - tN;

            if (IS_ATTN(il)) {
                matmul(s.xn, m->layers[il].wq, N_EMBD, WQ_DIM, T, m->layers[il].t_wq, s.qfull);
                matmul(s.xn, m->layers[il].wk, N_EMBD, WKV_DIM, T, m->layers[il].t_wk, s.kvK);
                matmul(s.xn, m->layers[il].wv, N_EMBD, WKV_DIM, T, m->layers[il].t_wv, s.kvV);

                /* q norm on the strided Q view; copy gate to z */
                {
                    const float *qw = (const float *)m->layers[il].q_norm;
                    for (int t = 0; t < T; t++)
                        for (int h = 0; h < N_HEAD; h++) {
                            float *v = s.qfull + (size_t)t * WQ_DIM + (size_t)h * (N_EMBD_HEAD * 2);
                            double sum = 0;
                            for (int i = 0; i < N_EMBD_HEAD; i++) sum += (double)v[i] * v[i];
                            float r = 1.0f / sqrtf((float)(sum / N_EMBD_HEAD) + RMS_EPS);
                            for (int i = 0; i < N_EMBD_HEAD; i++) v[i] *= r * qw[i];
                            memcpy(s.z + (size_t)t * WO_DIM + (size_t)h * N_EMBD_HEAD,
                                   v + N_EMBD_HEAD, N_EMBD_HEAD * sizeof(float));
                        }
                    const float *kw = (const float *)m->layers[il].k_norm;
                    for (int t = 0; t < T; t++)
                        for (int h = 0; h < N_HEAD_KV; h++) {
                            float *v = s.kvK + (size_t)t * WKV_DIM + (size_t)h * N_EMBD_HEAD;
                            double sum = 0;
                            for (int i = 0; i < N_EMBD_HEAD; i++) sum += (double)v[i] * v[i];
                            float r = 1.0f / sqrtf((float)(sum / N_EMBD_HEAD) + RMS_EPS);
                            for (int i = 0; i < N_EMBD_HEAD; i++) v[i] *= r * kw[i];
                        }
                }

                {
                    int *pos = malloc(sizeof(int) * (T + 1));
                    for (int t = 0; t < T; t++) pos[t] = rs->n_pos + done + t;
                    rope_imrope(s.qfull, N_HEAD, N_EMBD_HEAD, N_EMBD_HEAD * 2, T, pos);
                    rope_imrope(s.kvK, N_HEAD_KV, N_EMBD_HEAD, N_EMBD_HEAD, T, pos);
                    free(pos);
                }

                /* attention: Q strided [T][12288], K/V [T][1024], out [T][6144] into s.z? 
                   z holds the gate — out must go elsewhere: use s.gkv region? gkv [T][18432] free here → out at offset 0. */
                {
                    float *aout = s.gkv; /* [T][6144] */
                    attn_ctx ac = { s.qfull, s.kvK, s.kvV, s.z, aout, rs->kv_cache[il / 4],
                                    T, rs->n_ctx, rs->n_pos + done,
                                    il / 4, hcm_live_n, hcm_n4, hcm_n_ring_q4,
                                    hcm_rebuilt_at, hcm_pin_kv[il / 4], hcm_arc_kv[il / 4] };
                    if (g_timing) tA = now_s();
                    kv_append(&ac);
                    pool_run(attn_worker, &ac, N_HEAD);
                    if (g_timing) tA = now_s() - tA;
                    /* gate multiply (ref: attn * sigmoid(gate)) */
                    {
                        const size_t tot = (size_t)WO_DIM * T, vtot = tot & ~(size_t)3;
                        for (size_t i = 0; i < vtot; i += 4) {
                            float32x4_t g = vld1q_f32(aout + i);
                            g = vmulq_f32(g, v_sigmoid32x4(vld1q_f32(s.z + i)));
                            vst1q_f32(aout + i, g);
                        }
                        for (size_t i = vtot; i < tot; i++)
                            aout[i] *= 1.0f / (1.0f + expf(-s.z[i]));
                    }
                    matmul(aout, m->layers[il].wo, WO_DIM, N_EMBD, T, m->layers[il].t_wo, s.ra);
                }
            } else {
                matmul(s.xn, m->layers[il].wqkv, N_EMBD, QKV_DIM, T, m->layers[il].t_wqkv, s.qkv);
                matmul(s.xn, m->layers[il].attn_gate, N_EMBD, VAL_DIM, T, m->layers[il].t_gate, s.z);
                matmul(s.xn, m->layers[il].ssm_beta, N_EMBD, S_DT_RANK, T, m->layers[il].t_beta, s.beta);
                {
                    const size_t btot = (size_t)S_DT_RANK * T, bvt = btot & ~(size_t)3;
                    for (size_t i = 0; i < bvt; i += 4)
                        vst1q_f32(s.beta + i, v_sigmoid32x4(vld1q_f32(s.beta + i)));
                    for (size_t i = bvt; i < btot; i++)
                        s.beta[i] = 1.0f / (1.0f + expf(-s.beta[i]));
                }
                matmul(s.xn, m->layers[il].ssm_alpha, N_EMBD, S_DT_RANK, T, m->layers[il].t_alpha, s.gt);
                {
                    const float *dt = (const float *)m->layers[il].ssm_dt;
                    const float *sa = (const float *)m->layers[il].ssm_a;
                    for (int t = 0; t < T; t++)
                        for (int h = 0; h < S_DT_RANK; h++) {
                            float a = s.gt[(size_t)t * S_DT_RANK + h] + dt[h];
                            float sp = a > 20.0f ? a : log1pf(expf(a));
                            s.gt[(size_t)t * S_DT_RANK + h] = sp * sa[h];
                        }
                }
                if (g_timing) tC = now_s();
                conv1d_layer(rs->conv_state[il], s.qkv, (const float *)m->layers[il].ssm_conv1d, s.cout, T);
                { /* trace qkv/conv magnitudes at the failing layer */
                    static int tr2 = -1;
                    if (tr2 < 0) tr2 = getenv("QMA_TRACE") != NULL;
                    if (tr2 && il == 5) {
                        double qs=0, ks=0, vs=0, cs=0;
                        for (int i=0;i<KEY_DIM*T;i++) qs += fabs(s.qkv[i]);
                        for (int i=KEY_DIM*T;i<2*KEY_DIM*T;i++) ks += fabs(s.qkv[i]);
                        for (int i=2*KEY_DIM*T;i<QKV_DIM*T;i++) vs += fabs(s.qkv[i]);
                        for (int i=0;i<CONV_DIM*T;i++) cs += fabs(s.cout[i]);
                        fprintf(stderr, "[L5] qkv q=%.3e k=%.3e v=%.3e cout=%.3e convw0=%.3e\n",
                                qs, ks, vs, cs, ((const float*)m->layers[5].ssm_conv1d)[0]);
                    }
                }
                silu_m(s.cout, CONV_DIM, T);
                if (g_timing) tC = now_s() - tC;

                /* split+expand q/k/v (16 -> 48 heads) from channel-major cout into gkv */
                {
                    const int S_v = S_D_STATE;
                    for (int t = 0; t < T; t++) {
                        for (int h = 0; h < S_DT_RANK; h++) {
                            const int qh = h % S_N_GROUP;
                            float *o = s.gkv + (size_t)t * S_v * S_DT_RANK * 3 + (size_t)h * S_v;
                            for (int i = 0; i < S_v; i++) {
                                o[i] = s.cout[(size_t)(qh * S_v + i) * T + t];
                                o[S_v * S_DT_RANK + i] = s.cout[(size_t)(KEY_DIM + qh * S_v + i) * T + t];
                                o[2 * S_v * S_DT_RANK + i] = s.cout[(size_t)(KEY_DIM * 2 + h * S_v + i) * T + t];
                            }
                        }
                    }
                    /* l2 norm on q48 and k48 (per head over S_v); k48 starts at
                       the per-token offset VAL_DIM (not VAL_DIM*T) inside the
                       interleaved gkv rows */
                    l2norm_skv(s.gkv, S_DT_RANK, T);
                    l2norm_skv(s.gkv + (size_t)VAL_DIM, S_DT_RANK, T);
                }

                /* gated delta net */
                {
                    const int ts = S_D_STATE * S_DT_RANK * 3; /* 18432 */
                    /* k/v live at fixed per-token offsets inside each 18432-wide
                       token block (t*18432 + 6144 / + 12288), NOT at *T offsets:
                       the split above writes them interleaved per token. The
                       old gkv + VAL_DIM*T / + 2*VAL_DIM*T bases were only
                       correct for T==1, silently corrupting k/v for T>1. */
                    gdn_ctx gc = { rs->ssm_state[il],
                                   s.gkv,
                                   s.gkv + (size_t)VAL_DIM,
                                   s.gkv + (size_t)(VAL_DIM * 2),
                                   s.gt, s.beta, s.gdout, T, ts };
                    if (g_timing) tG = now_s();
                    pool_run(gdn_worker, &gc, S_DT_RANK);
                    if (g_timing) tG = now_s() - tG;
                }
                /* gated norm: rms per head then * silu(z), in place */
                rms_norm_head(s.gdout, (const float *)m->layers[il].ssm_norm, S_DT_RANK, T);
                {
                    const size_t ztot = (size_t)VAL_DIM * T, zvt = ztot & ~(size_t)3;
                    for (size_t i = 0; i < zvt; i += 4) {
                        /* gdout * silu(z), silu(z) = z * sigmoid(z) -- the z
                           factor matters: this site is SILU, not plain sigmoid */
                        float32x4_t zv = vld1q_f32(s.z + i);
                        float32x4_t g = vmulq_f32(zv, v_sigmoid32x4(zv));
                        vst1q_f32(s.z + i, vmulq_f32(vld1q_f32(s.gdout + i), g));
                    }
                    for (size_t i = zvt; i < ztot; i++)
                        s.z[i] = s.gdout[i] * (s.z[i] / (1.0f + expf(-s.z[i])));
                }
                matmul(s.z, m->layers[il].ssm_out, VAL_DIM, N_EMBD, T, m->layers[il].t_out, s.ra);
            }

            /* residual + MoE FFN (256 experts top-8 + shared expert) */
            add_m(s.x2, s.ra, s.x, N_EMBD, T);
            rms_norm_m(s.xn, s.x2, (const float *)m->layers[il].attn_post_norm, N_EMBD, T, RMS_EPS);
            { double tM0 = now_s();
              moe_ffn(m, il, s.xn, s.x2, s.ra, T);
              timing_add(8, now_s() - tM0); }
            add_m(s.x, s.ra, s.x2, N_EMBD, T);
            if (il == 0 && getenv("QMA_DUMP_L1")) {
                static int dumped1 = 0;
                if (!dumped1) {
                    FILE *f = fopen(getenv("QMA_DUMP_L1"), "wb");
                    if (f) {
                        fwrite(s.xn, sizeof(float), N_EMBD * T, f);   /* ffn input (post_norm) */
                        fwrite(s.ra, sizeof(float), N_EMBD * T, f);   /* ffn output */
                        fwrite(s.x, sizeof(float), N_EMBD * T, f);    /* layer out */
                        fclose(f);
                        fprintf(stderr, "qma: dumped layer-0 FFN data to %s\n", getenv("QMA_DUMP_L1"));
                    }
                    dumped1 = 1;
                }
            }
            if (g_timing) {
                timing_add(2, tG); timing_add(3, tC); timing_add(4, tA); timing_add(5, tN);
                /* misc = layer time - (matmul + quant + everything else); matmul is
                   captured via the global accumulator delta so it is not double
                   counted */
                double dtM = t_acc[0] - tM;   /* matmul ms added during this layer */
                tM = t_acc[0];
                timing_add(6, now_s() - tL0 - dtM - tN - tA - tC - tG);
            }
        }

        if (g_timing) timing_report("eval");
        /* final norm + lm head (output.weight is Q4_K [N_EMBD, N_VOCAB]) */
        rms_norm_m(s.xn, s.x, (const float *)m->output_norm, N_EMBD, T, RMS_EPS);
        if (logits_all) {
            if (logits) {
                matmul(s.xn, m->output, N_EMBD, N_VOCAB, T, m->t_output, logits + (size_t)done * N_VOCAB);
            }
        } else if (logits && done + T == n_tokens) {
            matmul(s.xn + (size_t)(T - 1) * N_EMBD, m->output, N_EMBD, N_VOCAB, 1, m->t_output, logits);
        }
        done += T;
    }

    if (g_timing) timing_summary();
    rs->n_pos += n_tokens;
    if (g_pf_trace) {
        int tot = 0, hits = 0;
        for (int il = 1; il < N_LAYER; il++) {
            if (g_pred_gen[il] && g_act_gen[il]) {
                int h = 0;
                for (int k = 0; k < N_EXPERT_USED; k++)
                    for (int j = 0; j < N_EXPERT_USED; j++)
                        if (g_pred_ids[il][k] == g_act_ids[il][j]) { h++; break; }
                fprintf(stderr, "[pf] L%d: %d/8 pred-act overlap\n", il, h);
                tot += 8; hits += h;
            }
        }
        if (tot) fprintf(stderr, "[pf] total overlap %d/%d (%.1f%%)\n", hits, tot, 100.0*hits/tot);
    }
    return 0;
}


