/* qma.h — mobile-agent-3 engine + agent. Pure C99.
 *
 * Target model: LFM2.5-2.6B (arch "lfm2") — Liquid AI hybrid:
 *   30 layers = 8 GQA attention + 22 double-gated short-conv.
 * The engine is single-architecture (no qwen35/GDN/HCM baggage) but the
 * GGUF metadata parser is generic: it discovers <arch>.<key> by prefix so
 * other archs fail with a clear "unsupported" error instead of garbage.
 *
 * Reference for exact formulas: llama.cpp src/models/lfm2.cpp.
 */
#ifndef QMA_H
#define QMA_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define N_VOCAB   131072      /* logits buffer cap (lfm2: 128000) */
#define N_LAYER   64          /* max layers (lfm2: 30) */
#define N_CTX_MAX 262144      /* context cap (lfm2: 131072) */

/* matmul weight slot indices (parallel to layers[].wtype[]) */
#define LFM_W_GATE 0
#define LFM_W_UP   1
#define LFM_W_DOWN 2
#define LFM_W_Q    3
#define LFM_W_K    4
#define LFM_W_V    5
#define LFM_W_O    6
#define LFM_W_CIN  7
#define LFM_W_COUT 8
#define LFM_W_NUM  9

/* ---------- quant type constants (GGML enum ids) ---------- */
#define GGML_TYPE_F32  0
#define GGML_TYPE_F16  1
#define GGML_TYPE_Q4_0 2
#define GGML_TYPE_Q4_K 12
#define GGML_TYPE_Q6_K 14

/* Q4_0: fp16 scale + 32 x 4-bit weights per block (16 B of nibbles) */
#define QK4_0 32
typedef struct { uint16_t d; uint8_t qs[QK4_0/2]; } block_q4_0;

/* Q4_K: 256 weights per block, repacked layout (144 B — same as qma's
 * qwen model): d + dmin fp16, scales[12] holding 8 scale/min pairs packed
 * in 6 bits (get_scale_min_k4), qs[128] nibbles. The LFM GGUF on disk is
 * repacked with the same tool, so qma's q8k kernels could consume it too. */
#define QK_K 256
#define K_SCALE_SIZE 12
typedef struct {
    uint16_t d;               /* fp16 super-block scale */
    uint16_t dmin;            /* fp16 super-block min */
    uint8_t  scales[K_SCALE_SIZE]; /* scales+mins, 6-bit packed (8 pairs) */
    uint8_t  qs[QK_K/2];      /* nibbles */
} block_q4_K;

/* Q6_K: 256 weights per block: ql[128] low nibbles, qh[64] high bits,
 * scales[16] int8 (16 groups of 16), one fp16 super-scale. 210 bytes. */
#define QK6_K 256
typedef struct {
    uint8_t  ql[QK6_K/2];      /* lower 4 bits of each weight */
    uint8_t  qh[QK6_K/4];      /* upper 2 bits */
    int8_t   scales[QK6_K/16]; /* per-16 scale (int8), 16 entries */
    uint16_t d;                /* fp16 super-block scale */
} block_q6_k;

/* Q8_0: fp16 scale + 32 int8 (activation quant) */
#define QK8_0 32
typedef struct { uint16_t d; int8_t qs[QK8_0]; } block_q8_0;

/* ---------- fp16 helpers (aarch64 native, portable fallback) ---------- */
static inline float half_to_float(uint16_t h) {
#if defined(__aarch64__)
    __fp16 f16;
    memcpy(&f16, &h, 2);
    return (float)f16;
#else
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t man  = h & 0x3ff;
    if (exp == 0) {
        if (man == 0) return 0.0f;
        /* subnormal */
        exp = 127 - 15 + 1;
        while (!(man & 0x400)) { man <<= 1; exp--; }
        man &= 0x3ff;
        uint32_t bits = sign | (exp << 23) | (man << 13);
        float f; memcpy(&f, &bits, 4); return f;
    }
    if (exp == 31) return sign ? -__builtin_huge_valf() : __builtin_huge_valf();
    exp = exp - 15 + 127;
    uint32_t bits = sign | (exp << 23) | (man << 13);
    float f; memcpy(&f, &bits, 4); return f;
#endif
}
static inline uint16_t float_to_half(float f) {
#if defined(__aarch64__)
    __fp16 f16 = (__fp16)f;
    uint16_t h; memcpy(&h, &f16, 2);
    return h;
#else
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  e    = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t m    = x & 0x7fffff;
    if (((x >> 23) & 0xff) == 0xff) return (uint16_t)(sign | 0x7c00); /* inf/nan */
    if (e >= 31) return (uint16_t)(sign | 0x7c00);
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;
        m |= 0x800000;
        uint32_t shift = (uint32_t)(14 - e);
        uint32_t half_m = (m >> shift) + ((m >> (shift - 1)) & 1); /* round */
        return (uint16_t)(sign | half_m);
    }
    uint32_t half_m = (m >> 13) + ((m >> 12) & 1);
    if (half_m & 0x400) { half_m = 0; e++; }
    return (uint16_t)(sign | ((uint32_t)e << 10) | half_m);
#endif
}

/* ---------- runtime hyperparameters parsed from GGUF ---------- */
typedef struct {
    char     arch[32];              /* general.architecture (lfm2) */
    uint32_t n_embd, n_layer, n_head, n_vocab, n_ff;
    uint32_t n_embd_head;           /* head dim (64) */
    uint32_t n_rot;                 /* rope dim (default = n_embd_head) */
    float    freq_base;             /* rope base (1e7) */
    float    rms_eps;               /* 1e-5 */
    uint32_t head_count_kv[N_LAYER];/* per-layer KV heads; 0 = short-conv layer */
    uint32_t shortconv_l_cache;     /* conv kernel length (3) */
    uint32_t n_ctx;                 /* context_length */
    int      n_attn;                /* count of attention layers */
    int      n_conv;                /* count of short-conv layers */
    int      tied;                  /* output = token_embd (no output tensor) */
} model_config_t;

/* ---------- tensor descriptor ---------- */
typedef struct {
    char     name[64];
    uint32_t type;                  /* GGML_TYPE_* */
    int64_t  ne[2];
    size_t   off;                   /* offset from start of tensor data section */
    size_t   nbytes;
} tensor_t;

/* ---------- model ---------- */
typedef struct {
    /* weights backing: anonymous resident buffer (map_malloc=1) or mmap */
    int      fd;
    int      map_malloc;
    uint8_t *map;
    size_t   map_size;
    uint64_t g_base;    /* GGUF region base within map (0 = whole file) */
    uint64_t g_size;
    uint8_t *data;      /* start of tensor data section */

    uint32_t n_tensors;
    tensor_t *tensors;

    model_config_t cfg;

    /* resolved weights (all point into the mmap) */
    uint8_t *token_embd;       /* Q6_K [n_embd, n_vocab] */
    uint8_t *token_embd_norm;  /* F32  [n_embd] */

    struct {
        uint8_t *attn_norm;      /* F32 [n_embd]   (operator_norm) */
        uint8_t *ffn_norm;       /* F32 [n_embd] */
        uint8_t *ffn_gate;       /* Q4_0/Q4_K [n_embd, n_ff] */
        uint8_t *ffn_up;         /* Q4_0/Q4_K [n_embd, n_ff] */
        uint8_t *ffn_down;       /* Q4_0/Q4_K/Q6_K [n_ff, n_embd] */
        /* GQA attention layers (cfg.head_count_kv[il] > 0) */
        uint8_t *wq;             /* Q4_0/Q4_K [n_embd, n_embd]   (32 h x 64) */
        uint8_t *wk;             /* Q4_0/Q4_K [n_embd, kv*64]    ( 8 h x 64) */
        uint8_t *wv;             /* Q4_0/Q4_K/Q6_K [n_embd, kv*64] */
        uint8_t *wo;             /* Q4_0/Q4_K [n_embd, n_embd] */
        uint8_t *q_norm;         /* F32  [64] */
        uint8_t *k_norm;         /* F32  [64] */
        /* short-conv layers (cfg.head_count_kv[il] == 0) */
        uint8_t *conv_in_proj;   /* Q4_0/Q4_K [n_embd, 3*n_embd] */
        uint8_t *conv;           /* F32  [l_cache, n_embd] */
        uint8_t *conv_out_proj;  /* Q4_0/Q4_K [n_embd, n_embd] */
        uint32_t wtype[LFM_W_NUM]; /* GGML_TYPE per matmul weight (index LFM_W_*) */
    } layers[N_LAYER];

    /* tokenizer */
    int      n_vocab;
    char    *tok_text[N_VOCAB];   /* byte-encoded token strings */
    uint8_t  tok_type[N_VOCAB];   /* llama token attr bits */
    uint32_t byte_to_id[256];     /* byte -> token id */
    uint8_t  tok_is_special[N_VOCAB];
    uint32_t special_ids[N_VOCAB];
    int      n_special;

    uint32_t tok_cap;
    uint64_t *tok_keys;
    uint32_t *tok_ids;

    uint32_t mg_cap;
    uint64_t *mg_keys;     /* hash of (left_id, right_id) */
    uint32_t *mg_rank;     /* rank or UINT32_MAX */

    float    s_temp, s_top_p;
    int      s_top_k;

    /* chat template special ids */
    uint32_t id_bos, id_eos;
    uint32_t id_im_start, id_im_end;
    uint32_t id_think_start, id_think_end;
    uint32_t id_tool_start, id_tool_end;
} lfm_t;

/* ---------- runtime state ---------- */
typedef struct {
    /* KV cache (f16), only the 8 attention layers:
       [lay][2 heads? no: flat] layout: per attn layer, per token:
       K: 8 heads x 64 f16, then V: 8 heads x 64 f16  (= 2048 B/token)
       kv_cache[il] = arena slice of 2*n_kv*64*n_ctx f16 */
    uint16_t *kv_cache[N_LAYER];
    /* short-conv rolling state (f32): [22 conv layers][2][n_embd]
       (l_cache-1 past gated values per channel per layer) */
    float *conv_state[N_LAYER];
    /* arena backing (file-backed kv.bin or anonymous mmap) */
    int    kv_fd;
    void  *kv_map;
    size_t kv_map_len;
    size_t kv_persist_bytes;   /* arena head to persist (KV + conv state) */
    const char *kv_path;
    int    kv_persist;
    /* persistent eval scratch (one arena, sliced) */
    void  *scratch;
    float *sx, *sxn, *sra;         /* [n_ctx][n_embd] x, normed, residual */
    float *sq, *sk, *sv;           /* attention projections */
    float *sscores, *satt_out;     /* [n_ctx] scores, attn output */
    float *sffn_u, *sffn_g, *sffn_out;
    float *sconv_in, *sconv_b, *sconv_c, *sconv_x, *sconv_y;
    void  *sq8;                    /* q8_0 activation blocks */
    int    n_ctx;
    int    n_pos;
} lfm_rs_t;

/* ---------- sampler ---------- */
typedef struct {
    uint64_t seed;
    float    temp;
    int      top_k;
    float    top_p;
    float    repeat_penalty;
    int      last_tokens[64];
    int      n_last;
} lfm_samp_t;

/* ---------- engine api (stable surface the agent loop calls) ---------- */
int      lfm_load(lfm_t *m, const char *path, uint64_t base, char *err, size_t errlen);
int      lfm_load_map(lfm_t *m, uint8_t *map, size_t map_size, uint64_t base,
                      uint64_t model_size, char *err, size_t errlen);
void     lfm_free(lfm_t *m);

int      lfm_rs_init(lfm_rs_t *rs, int n_ctx);
int      lfm_rs_init_kv(lfm_rs_t *rs, int n_ctx, const char *path, int persist);
void     lfm_rs_free(lfm_rs_t *rs);
int      lfm_rs_save(lfm_rs_t *rs, const char *path);
int      lfm_rs_load(lfm_rs_t *rs, const char *path);

/* HCM no-ops (agent loop calls these; LFM2's sparse KV needs no salience) */
int      lfm_hcm_save(const char *path);
int      lfm_hcm_load(const char *path, int n_ctx);

void     lfm_pool_set_max(int n);
void     lfm_pool_stop(void);
/* whole model is resident in RAM: prefetch is a no-op */
void     lfm_prefetch_init(lfm_t *m);
void     lfm_prefetch_layer(int il, int dist, lfm_t *m);
void     lfm_prefetch_whole(lfm_t *m);

int      lfm_eval(lfm_t *m, lfm_rs_t *rs, const int *tokens, int n_tokens,
                     float *logits, int n_threads, int prefetch, int logits_all);

/* emotion / conditioning / coalescence systems stripped — the worker
 * is a bare question-answering engine. */

int      lfm_tokenize(lfm_t *m, const char *text, int *out, int max_out);
int      lfm_detokenize(lfm_t *m, const int *tokens, int n, char *out, int max_out);

void     lfm_samp_init(lfm_samp_t *s, uint64_t seed, float temp, int top_k, float top_p, float repeat_penalty);
int      lfm_samp_candidates(int n_vocab, const float *logits, lfm_samp_t *s,
                            int *ids, float *lgs, int max_k);
int      lfm_samp_pick(lfm_samp_t *s, const int *ids, float *lgs, int n, float top_p);

void     lfm_log(const char *fmt, ...);
uint64_t lfm_hash64(const void *data, size_t len);

/* dequant + dot kernels (nn.c) */
void     dequant_row_q4_0(const block_q4_0 *x, float *y, int64_t k);
void     dequant_row_q6_k(const block_q6_k *x, float *y, int64_t k);
void     dot_q4_0_q8_0(int n, const block_q4_0 *x, const block_q8_0 *y, float *s);
float    dot_q6_k_q8_0_256(const block_q6_k *x, const block_q8_0 *y);
void     quantize_row_q8_0(const float *x, block_q8_0 *y, int64_t k);

#endif /* QMA_H */
