/*
 * qma.h - Qwen3.6-35B-A3B (qwen35moe) inference engine core
 *
 * A minimal C99 inference engine for qwen3.6:35b:a3b-q4km.gguf
 * (qwen35moe hybrid architecture: 30 recurrent "gated delta net" layers +
 * 10 full-attention layers, MoE FFN with 256 experts top-8 + shared expert)
 * on this specific device (aarch64, 8 cores).
 *
 * with the MoE hyperparameters hardcoded. Ground truth:
 * PrismML-Eng/llama.cpp @ 9ca265a (branch prism), src/models/qwen35moe.cpp.
 */
#ifndef QMA_H
#define QMA_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ecache.h"
#include <math.h>

#define QMA_VERSION "1.0.0-moe"

/* ---------- qwen35moe hyperparameters (from qwen3.6:35b:a3b-q4km.gguf) --- */
#define N_EMBD        2048
#define N_LAYER       40
#define N_HEAD        16
#define N_HEAD_KV     2
#define N_EMBD_HEAD   256
#define N_VOCAB       248320
#define N_ROT         64          /* rope.dimension_count */
#define ROPE_SECT0    11          /* rope.dimension_sections [11,11,10,0] */
#define ROPE_SECT1    11
#define ROPE_SECT2    10
#define FREQ_BASE     10000000.0f
#define RMS_EPS       1e-6f

/* recurrent (linear attention / gated delta net) block dims */
#define S_D_CONV      4           /* ssm.conv_kernel */
#define S_D_INNER     4096        /* ssm.inner_size = value_dim */
#define S_D_STATE     128         /* ssm.state_size   = head_k_dim = head_v_dim */
#define S_DT_RANK     32          /* ssm.time_step_rank = num_v_heads */
#define S_N_GROUP     16          /* ssm.group_count   = num_k_heads */

#define CONV_DIM      (S_D_STATE*S_N_GROUP*2 + S_D_INNER)  /* 8192 */
#define KEY_DIM       (S_D_STATE*S_N_GROUP)                /* 2048 */
#define VAL_DIM       (S_D_STATE*S_DT_RANK)                /* 4096 */
#define QKV_DIM       (KEY_DIM*2 + VAL_DIM)                /* 8192 */
#define WQ_DIM        (N_EMBD_HEAD*N_HEAD*2)               /* 8192 */
#define WKV_DIM       (N_EMBD_HEAD*N_HEAD_KV)              /* 512 */
#define WO_DIM        (N_EMBD_HEAD*N_HEAD)                 /* 4096 */

/* MoE FFN: 256 experts, top-8, expert ffn 512, shared expert 512 */
#define N_EXPERT      256
#define N_EXPERT_USED 8
#define N_FF_EXP      512
#define N_FF_SHEXP    512

/* 4/6: layer is attention if (il+1) % 4 == 0 (10 attn, 30 delta) */
#define IS_ATTN(il)   (((il) + 1) % 4 == 0)
#define N_ATTN_LAYER  10

/* ---------- quant type constants ---------- */
/* NOTE: these ids MUST match the GGUF/llama.cpp enum exactly — they are
   file format ids, not internal numbers. Earlier this table had IQ2_S=23
   and "Q3_KXS"=18 (Q3_K layout); the real enum is IQ3_XXS=18 (98 B),
   IQ2_S=22 (82 B), IQ4_XS=23 (136 B). The Unsloth UD-Q2_K_XL file uses
   IQ3_XXS and IQ4_XS for its down-exps; the wrong mapping made qma decode
   those slabs with the wrong block layout and wrong expert stride
   (gibberish/NaN). */
#define GGML_TYPE_F32   0
#define GGML_TYPE_Q4_K  12
#define GGML_TYPE_Q5_K  13
#define GGML_TYPE_Q6_K  14
#define GGML_TYPE_Q3_K  11
#define GGML_TYPE_Q2_K  10
#define GGML_TYPE_BF16  30
#define GGML_TYPE_IQ2_XS 17
#define GGML_TYPE_IQ3_XXS 18   /* 98 B/block: d(2) + qs[3*QK_K/8] */
#define GGML_TYPE_IQ2_S 22     /* 82 B/block */
#define GGML_TYPE_IQ4_XS 23    /* 136 B/block: d + scales_h + scales_l + qs */
#define GGML_TYPE_Q8_0   8

/* Q4_K: 256 weights per super-block, 8 sub-blocks of 32, 6-bit scales.
 * weight = (q + b) * d, with per-sub-block scale/mins packed in 6 bits.
 * d + dmin fp16, 12 B 6-bit scales, 128 B nibbles = 144 B / 256 w. */
#define QK_K 256
#define K_SCALE_SIZE 12

typedef struct {
    uint16_t d;      /* super-block scale for quantized scales */
    uint16_t dmin;   /* super-block scale for quantized mins */
    uint8_t  scales[K_SCALE_SIZE]; /* scales+mins, 6-bit quantized */
    uint8_t  qs[QK_K/2];           /* 4-bit quants */
} block_q4_K;

/* Q6_K: 16 blocks of 16, weight = q * d, 8-bit scales */
typedef struct {
    uint8_t  ql[QK_K/2];      /* lower 4 bits */
    uint8_t  qh[QK_K/4];      /* upper 2 bits */
    int8_t   scales[QK_K/16]; /* 8-bit */
    uint16_t d;               /* fp16 super-block scale */
} block_q6_K;

/* q8_0: fp16 scale + 32 int8 (activation quant) */
#define QK8_0 32
typedef struct { uint16_t d; int8_t qs[QK8_0]; } block_q8_0;

/* Q5_K: Q4_K layout + 1 extra high bit per weight (qh).
 * weight = d*sc*q - dmin*m, q in [0,31]. 176 B / 256 w. */
typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[K_SCALE_SIZE];
    uint8_t  qh[QK_K/8];          /* high bit of each weight */
    uint8_t  qs[QK_K/2];          /* low 4 bits */
} block_q5_K;

/* Q2_K: 16 groups of 16, 4-bit quants, per-group scale/min in 5-bit
 * packed nibble pairs. weight = d*sc*q - dmin*m. 84 B / 256 w. */
typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[QK_K/16];     /* 16 x 4-bit scale/min pairs */
    uint8_t  qs[QK_K/4];          /* 4-bit quants */
} block_q2_K;

/* Q3_K: 16 groups of 16, weight = d*(sc-32)*(q - 4*[high bit clear]),
 * q in [0,3] from 2-bit fields + hmask high bits. 110 B / 256 w. */
typedef struct {
    uint8_t  hmask[QK_K/8];       /* high bit per weight (bit g -> lanes) */
    uint8_t  qs[QK_K/4];          /* low 2 bits, 4 fields per byte */
    uint8_t  scales[12];          /* 6-bit, shuffled layout (see kernel) */
    uint16_t d;
} block_q3_K;

/* IQ2_XS: codebook quants. Per 256 w: 8 x 32-lane groups, each group has
 * one scale byte (lo nibble -> first 16 lanes' db, hi nibble -> last 16)
 * and four u16 entries; low 9 bits index iq2xs_grid (8 bytes), bits 9-15
 * index ksigns_iq2xs for the sign of each of those 8 weights.
 * weight = d*(0.5+sc)*0.25 * grid[j] * sign. 74 B / 256 w. */
typedef struct {
    uint16_t d;
    uint16_t qs[QK_K/8];          /* 32 grid+sign indices */
    uint8_t  scales[QK_K/32];     /* 8 scale bytes */
} block_iq2_xs;

/* IQ2_S: like IQ2_XS but grid index is 10-bit (qs low 8b + 2 bits from qh),
 * signs stored as plain bytes. 82 B / 256 w. */
typedef struct {
    uint16_t d;
    uint8_t  qs[QK_K/4];          /* [0..31] grid idx lo, [32..63] signs */
    uint8_t  qh[QK_K/32];         /* grid idx bits 8-9, packed */
    uint8_t  scales[QK_K/32];
} block_iq2_s;

/* IQ3_XXS: 98 B / 256 w — d(2) + 96 bytes of 3-bit grid indices.
 * Dequant reads 8-bit indices into iq3xxs_grid + 4-bit scales packed in
 * the top nibble of each 32-bit group of the qs tail. */
typedef struct {
    uint16_t d;
    uint8_t  qs[3*QK_K/8];        /* 96 bytes */
} block_iq3_xxs;

/* IQ4_XS: 136 B / 256 w — d(2) + scales_h(2) + scales_l(4) + qs(128).
 * 32 4-bit weights per sub-block, nonlinear 4-bit codebook (kvalues_iq4nl). */
typedef struct {
    uint16_t d;
    uint16_t scales_h;
    uint8_t  scales_l[QK_K/64];   /* 4 bytes */
    uint8_t  qs[QK_K/2];          /* 128 bytes */
} block_iq4_xs;

/* byte size of one QK_K super-block of a supported weight type
 * (0 = unsupported). Single source of truth — all row/slab math routes
 * through this so mixed-quant models stay consistent. */
static inline size_t qma_blk_size(int type) {
    switch (type) {
    case GGML_TYPE_Q4_K:  return sizeof(block_q4_K);   /* 144 */
    case GGML_TYPE_Q5_K:  return sizeof(block_q5_K);   /* 176 */
    case GGML_TYPE_Q6_K:  return sizeof(block_q6_K);   /* 210 */
    case GGML_TYPE_Q3_K:  return sizeof(block_q3_K);   /* 110 */
    case GGML_TYPE_IQ3_XXS:return sizeof(block_iq3_xxs);/* 98  */
    case GGML_TYPE_Q2_K:  return sizeof(block_q2_K);   /* 84  */
    case GGML_TYPE_Q8_0:  return sizeof(block_q8_0) * (QK_K/QK8_0); /* 272 */
    case GGML_TYPE_IQ2_XS:return sizeof(block_iq2_xs); /* 74  */
    case GGML_TYPE_IQ2_S: return sizeof(block_iq2_s);  /* 82  */
    case GGML_TYPE_IQ4_XS:return sizeof(block_iq4_xs); /* 136 */
    default:              return 0;
    }
}

/* ---------- fp16 helpers ---------- */
/* ARM64 converts fp16<->fp32 in a single FCVT instruction. The portable
   bit-twiddle below was ~20 cycles/call with a subnormal while-loop and is
   called 5x per 128 weights (1 q1_0 scale + 4 q8_0 scales) ~1G times per
   token — it was eating ~60% of runtime. __fp16 is native on aarch64. */
static inline float half_to_float(uint16_t h) {
#if defined(__aarch64__)
    __fp16 f16;
    memcpy(&f16, &h, 2);
    return (float)f16;          /* single FCVT Sd, Hn */
#else
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t man  = h & 0x3ff;
    uint32_t f;
    if (exp == 0) {
        if (man == 0) f = sign;
        else { /* subnormal */
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) { man <<= 1; exp--; }
            man &= 0x3ff;
            f = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 0x1f) {
        f = sign | 0x7f800000 | (man << 13); /* inf/nan */
    } else {
        f = sign | ((exp + (127 - 15)) << 23) | (man << 13);
    }
    float r; memcpy(&r, &f, 4); return r;
#endif
}
static inline uint16_t float_to_half(float x) {
#if defined(__aarch64__)
    __fp16 f16 = (__fp16)x;     /* single FCVT Hd, Sn */
    uint16_t h; memcpy(&h, &f16, 2);
    return h;
#else
    uint32_t f; memcpy(&f, &x, 4);
    uint32_t sign = (f >> 16) & 0x8000;
    int32_t  e    = (int32_t)((f >> 23) & 0xff) - 127 + 15;
    uint32_t m    = f & 0x7fffff;
    if (((f >> 23) & 0xff) == 0xff) return (uint16_t)(sign | 0x7c00 | (m ? 0x200 : 0));
    if (e >= 31) return (uint16_t)(sign | 0x7c00);
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;
        m |= 0x800000;
        uint32_t shift = (uint32_t)(14 - e);
        uint32_t half_m = m >> shift;
        uint32_t rem = m & ((1u << shift) - 1);
        if (rem > (1u << (shift - 1)) || (rem == (1u << (shift - 1)) && (half_m & 1))) half_m++;
        return (uint16_t)(sign | half_m);
    }
    uint32_t half_m = (m + 0x1000 + ((m >> 13) & 1)) >> 13; /* round to nearest even */
    if (half_m & 0x400) { half_m = 0; e++; }
    return (uint16_t)(sign | ((uint32_t)e << 10) | half_m);
#endif
}

/* ---------- tensor descriptor ---------- */
typedef struct {
    char     name[64];
    uint32_t type;
    int64_t  ne[3];
    size_t   off;       /* offset from start of tensor data section */
    size_t   nbytes;
} tensor_t;

/* ---------- model ---------- */
typedef struct {
    /* mmap */
    int      fd;
    int      dio_fd;   /* >= 0: O_DIRECT fd for expert reads (4K-aligned file) */
    uint8_t *map;
    size_t   map_size;
    uint8_t *data;      /* start of tensor data section */

    /* header */
    uint32_t n_tensors;
    tensor_t *tensors;

    /* resolved weights */
    uint8_t *token_embd;   /* Q1_0 [5120, 248320] */
    uint8_t *output_norm;  /* F32  [5120] */
    uint8_t *output;       /* Q1_0 [5120, 248320] */
    uint32_t t_token_embd, t_output; /* resolved GGUF types (do not assume) */

    struct {
        uint8_t *attn_norm;      /* F32 [2048] */
        uint8_t *attn_post_norm; /* F32 [2048] */
        /* recurrent layers */
        uint8_t *wqkv;           /* Q4_K/Q6_K [2048, 8192] (type varies) */
        uint8_t *attn_gate;      /* Q4_K [2048, 4096]  (wqkv_gate) */
        uint8_t *ssm_conv1d;     /* F32  [4, 8192] */
        uint8_t *ssm_dt;         /* F32  [32] */
        uint8_t *ssm_a;          /* F32  [32] */
        uint8_t *ssm_beta;       /* Q4_K [2048, 32] */
        uint8_t *ssm_alpha;      /* Q4_K [2048, 32] */
        uint8_t *ssm_norm;       /* F32  [128] */
        uint8_t *ssm_out;        /* Q4_K [4096, 2048] */
        /* attention layers */
        uint8_t *wq;             /* Q4_K [2048, 8192] */
        uint8_t *wk;             /* Q4_K [2048, 512] */
        uint8_t *wv;             /* Q4_K/Q6_K [2048, 512] (type varies) */
        uint8_t *wo;             /* Q4_K [4096, 2048] */
        uint8_t *q_norm;         /* F32  [256] */
        uint8_t *k_norm;         /* F32  [256] */
        /* ffn (all layers) — MoE: 256 experts top-8 + shared expert */
        uint8_t *ffn_gate_inp;      /* F32   [2048, 256] router logits */
        uint8_t *ffn_gate_exps;     /* Q4_K  [2048, 512, 256] per-expert gate */
        uint8_t *ffn_up_exps;       /* Q4_K  [2048, 512, 256] per-expert up */
        uint8_t *ffn_down_exps;     /* Q4_K/Q6_K [512, 2048, 256] (type varies) */
        uint8_t *ffn_gate_inp_shexp;/* F32   [2048] shared expert gate */
        uint8_t *ffn_gate_shexp;    /* Q4_K  [2048, 512] */
        uint8_t *ffn_up_shexp;      /* Q4_K  [2048, 512] */
        uint8_t *ffn_down_shexp;    /* Q4_K/Q6_K [512, 2048] (type varies) */
        /* resolved weight types (GGUF type varies per layer for some) */
        uint32_t t_wqkv, t_wv, t_down_exps, t_down_shexp;
        uint32_t t_wq, t_wk, t_wo, t_gate, t_alpha, t_beta, t_out;
        uint32_t t_gate_exps, t_up_exps, t_gate_shexp, t_up_shexp;
        /* absolute file offsets of the expert tensors (for the streaming
           cache's pread; = data_section_abs + tensor.off) */
        size_t   off_gate_exps, off_up_exps, off_down_exps;
        size_t   off_gate_shexp, off_up_shexp, off_down_shexp;
    } layers[N_LAYER];

    /* expert streaming cache (waste-style bounded LFRU + reader threads) */
    qma_ecache ecache;
    int ecache_on;
        int mmap_exps;   /* experts read zero-copy via m->data (no ecache) */         /* cache armed (budget > 0) */
    size_t data_section_abs; /* absolute file offset of the data section */

    /* tokenizer */    int      n_vocab;
    char    *tok_text[N_VOCAB];   /* byte-encoded token strings */
    uint8_t  tok_type[N_VOCAB];   /* llama token attr */
    uint32_t byte_to_id[256];     /* byte -> token id */
    uint8_t  tok_is_special[N_VOCAB];
    uint32_t special_ids[N_VOCAB];
    int      n_special;

    /* token hash: open addressing */
    uint32_t tok_cap;
    uint64_t *tok_keys;    /* 64-bit hash of token string */
    uint32_t *tok_ids;     /* token id or UINT32_MAX */

    /* merge table: (left_id, right_id) -> rank */
    uint32_t mg_cap;
    uint64_t *mg_keys;     /* hash of (left_id, right_id) */
    uint32_t *mg_rank;     /* rank or UINT32_MAX */

    /* sampler defaults from GGUF */
    float    s_temp, s_top_p;
    int      s_top_k;

    /* chat template special ids */
    uint32_t id_im_start, id_im_end;
} qma_t;

/* ---------- runtime state ---------- */
typedef struct {
    /* recurrent state per layer */
    float *conv_state[N_LAYER];   /* [3][10240] */
    float *ssm_state[N_LAYER];    /* [128][128][48] */
    /* kv cache: [16 attn layers][2][4 heads][ctx][256]  (K then V) */
    float *kv_cache[N_ATTN_LAYER];
    /* persistent eval scratch (one arena, sliced) -- avoids ~180MB of
       malloc/mmap churn per token */
    void  *scratch;               /* single contiguous block */
    float *sx, *sxn, *sx2, *sra;
    float *sqfull, *sqkv, *skvK, *skvV;
    float *sz, *sgkv, *sgdout, *sbeta, *sgt;
    float *scin, *scout;
    int    n_ctx;                 /* capacity per sequence */
    int    n_pos;                 /* current absolute position */
    /* file-backed KV (large context without heap): one sparse file mapped
       MAP_SHARED; OS pages in/out only what attention actually touches */
    int    kv_fd;
    void  *kv_map;                /* base of the whole mapping */
    size_t kv_map_len;
    const char *kv_path;          /* file path (unlinked unless --kv-save) */
    int    kv_persist;            /* 1 = keep file after free (session resume) */
} runstate_t;

/* ---------- sampler ---------- */
typedef struct {
    uint64_t seed;
    float    temp;
    int      top_k;
    float    top_p;
    float    repeat_penalty;
    int      last_tokens[64];
    int      n_last;
} sampler_t;

/* ---------- engine api ---------- */
int      qma_load(qma_t *m, const char *path, char *err, size_t errlen);
void     qma_free(qma_t *m);

int      runstate_init(runstate_t *rs, int n_ctx);
/* file-backed KV: create a sparse file at `path` and mmap it so context can
   grow far beyond heap (1M tokens = ~40GB KV). path NULL = use a temp file
   that is unlinked on free. Returns 0 ok. */
void     qma_pool_set_max(int n);
int      runstate_init_kv(runstate_t *rs, int n_ctx, const char *path, int persist);
int      qma_kvq_on(void);
/* persist/restore the non-KV state (ssm+conv+n_pos) so a session can resume
   without re-prefilling the prompt. KV itself lives in the mapped file.
   Returns 0 ok. */
int      runstate_save(runstate_t *rs, const char *path);
int      runstate_load(runstate_t *rs, const char *path);
/* HCM salience + heavy-hitter arena persistence (restart survival) */
int      hcm_save(const char *path);
int      hcm_load(const char *path, int n_ctx);
/* agent archive (the manually-edited memory ring): memory_* tools pin
   real K/V into a protected arena that attention always reads. Survives
   ring wrap; rebuilt from memory.json at boot. Returns token count or <0
   on error (-2 = archive needs quantized KV). */
int      hcm_archive_write(qma_t *m, runstate_t *rs, const char *key,
                           const char *content, int n_threads);
int      hcm_archive_append(qma_t *m, runstate_t *rs, const char *key,
                            const char *content, int n_threads);
int      hcm_archive_delete(runstate_t *rs, const char *key);
int      hcm_archive_clear(runstate_t *rs);
int      hcm_archive_count(void);
void     runstate_free(runstate_t *rs);

/* weight streaming: prefetch setup + per-layer lookahead (dist layers) */
void     qma_prefetch_init(qma_t *m);
void     qma_prefetch_layer(int il, int dist, qma_t *m);

/* expert streaming cache (waste-style): arm with a byte budget + reader
   threads after load; teardown before free. budget 0 = off. */
void     qma_ecache_arm(qma_t *m, size_t budget_bytes, int nthreads);
void     qma_ecache_teardown(qma_t *m);

/* YALIS default vectors (waste gqdef): enable the predict_next d_l
   correction. If path is non-NULL, load precomputed averages from a file
   written by an earlier --defaults-dump run; else start capturing now.
   Returns 0 on ok. */
int      qma_defaults_enable(qma_t *m, const char *load_path);

/* eval: process tokens; writes the LAST token's logits to `logits` (or all
   tokens if logits_all, in which case logits must hold n_tokens*N_VOCAB) */
int      qma_eval(qma_t *m, runstate_t *rs, const int *tokens, int n_tokens,
                     float *logits, int n_threads, int prefetch, int logits_all);

/* tokenizer */
int      qma_tokenize(qma_t *m, const char *text, int *out, int max_out);
int      qma_detokenize(qma_t *m, const int *tokens, int n, char *out, int max_out);

/* sampler */
void     sampler_init(sampler_t *s, uint64_t seed, float temp, int top_k, float top_p, float repeat_penalty);
/* two-phase sampling (llama.cpp chain shape): candidates, then pick.
   No grammar filter between them (see agent.c) — tool-call validity is
   enforced by policy, not token masking. */
int      sampler_candidates(int n_vocab, const float *logits, sampler_t *s,
                           int *ids, float *lgs, int max_k);
int      sampler_pick(sampler_t *s, const int *ids, float *lgs, int n, float top_p);

/* chat template (text-only path of the model's Jinja template) */

/* utils */
uint64_t qma_hash64(const void *data, size_t len);

/* q4_k dot with q8_0 activation: s = sum over k of W[k] * y[k] */
void     dequantize_row_q4_K(const block_q4_K *x, float *y, int64_t k);
void     dequantize_row_q6_K(const block_q6_K *x, float *y, int64_t k);
void     dequantize_row_q5_K(const block_q5_K *x, float *y, int64_t k);
void     dequantize_row_q3_K(const block_q3_K *x, float *y, int64_t k);
void     dequantize_row_iq2_xs(const block_iq2_xs *x, float *y, int64_t k);
void     dequantize_row_iq2_s(const block_iq2_s *x, float *y, int64_t k);
void     dequantize_row_iq3_xxs(const block_iq3_xxs *x, float *y, int64_t k);
void     dequantize_row_iq4_xs(const block_iq4_xs *x, float *y, int64_t k);
float    dot_q4_K_f32(const block_q4_K *W, const float *x, int n);
float    dot_q6_K_f32(const block_q6_K *W, const float *x, int n);
void     dequantize_row_q8_0(const void *x, float *y, int64_t k);
float    dot_q8_0_f32(const void *W, const float *x, int n);
float    qma_dot_q8_0_q8k(const void *b, const int8_t *xq,
                          const float *xd, int n);
float    dot_q5_K_f32(const block_q5_K *W, const float *x, int n);
float    dot_q3_K_f32(const block_q3_K *W, const float *x, int n);
float    dot_iq2_xs_f32(const block_iq2_xs *W, const float *x, int n);
float    dot_iq2_s_f32(const block_iq2_s *W, const float *x, int n);
float    dot_iq3_xxs_f32(const block_iq3_xxs *W, const float *x, int n);
float    dot_iq4_xs_f32(const block_iq4_xs *W, const float *x, int n);

/* quantize row of floats to q8_0 blocks (k must be multiple of 32) */
void     quantize_row_q8_0(const float *x, block_q8_0 *y, int64_t k);

/* int8 quantized-activation path (waste q8k / llama.cpp production):
   quantize x once per layer, then fused SDOT dots against Q4_K/Q6_K. */
int      qma_q8k_available(void);
void     qma_q8k_quant(const float *x, int n, int8_t *xq, float *xd, int16_t *xsum);
float    qma_q8k_dot(const void *row, int wtype, const int8_t *xq,
                        const float *xd, const int16_t *xsum, int n);
/* new-type SDOT kernels (qkerns.c); xsum unused by Q3_K/IQ2 (no bias) */
float    qma_dot_q5_K_q8k(const block_q5_K *b, const int8_t *xq,
                          const float *xd, const int16_t *xsum, int n);
float    qma_dot_q3_K_q8k(const block_q3_K *b, const int8_t *xq,
                          const float *xd, int n);
float    qma_dot_iq2_xs_q8k(const block_iq2_xs *b, const int8_t *xq,
                            const float *xd, int n);
float    qma_dot_iq2_s_q8k(const block_iq2_s *b, const int8_t *xq,
                           const float *xd, int n);
float    qma_dot_iq3_xxs_q8k(const block_iq3_xxs *b, const int8_t *xq,
                             const float *xd, int n);
float    qma_dot_iq4_xs_q8k(const block_iq4_xs *b, const int8_t *xq,
                            const float *xd, int n);
/* fused gate+up: two Q4_K rows, one pass over the shared activation */
void     qma_q8k_gateup(const void *g, const void *u, const int8_t *xq,
                           const float *xd, const int16_t *xsum, int n,
                           float *gate, float *up);
/* i8mm GEMM: Q4_K weights x q8 activations, T >= 2 tokens at once.
 * Computes rows [r0, r0+nrows), nrows must be even. */
int      qma_q8k_gemm_available(void);
void     qma_q8k_gemm_q4k(const uint8_t *W, size_t wrow, int r0, int nrows,
                          const int8_t *xq, const float *xd,
                          const int16_t *xsum, int n_in, int n_out,
                          int T, float *out);

/* one-time model alignment (see gguf.c): 1 = needs 4K repack, 0 = aligned,
   -1 = error */
int      qma_model_needs_align(const char *path, char *err, size_t errlen);
/* ensure O_DIRECT-able file: repacks to <src>.4k if needed; *use_path gets
   the file to load (== src when already aligned) */
int      qma_align_model(const char *src, char *use_path, size_t use_len,
                           char *err, size_t errlen);

#endif /* QMA_H */
