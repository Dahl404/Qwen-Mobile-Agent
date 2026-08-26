/*
 * gguf.c — GGUF v3 loader for LFM2.5-2.6B (arch "lfm2").
 *
 * Parses the header (metadata + tensor infos), mmaps the file, resolves
 * every weight the lfm2 model needs (Q4_0 lines, Q6_K embedding, F32
 * norms), and builds the BPE tokenizer tables. The metadata reader is
 * architecture-generic (discovers <arch>.<key> by prefix) but the tensor
 * binding and the arch gate are lfm2-only — ma3 is single-model.
 *
 * Tensor data offsets in GGUF v3 are relative to the start of the tensor
 * data section (right after the last tensor info).
 */
#include "lfm.h"
/* embedded-load stripped */
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

void lfm_log(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    vfprintf(stderr, fmt, ap); va_end(ap);
}

uint64_t lfm_hash64(const void *data, size_t len) {
    /* FNV-1a 64-bit */
    const uint8_t *p = data;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < len; i++) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

/* ---------------- GGUF binary reader ---------------- */
typedef struct {
    const uint8_t *p;
    const uint8_t *end;
    int err;
} reader_t;

static uint64_t rd_u64(reader_t *r) {
    if (r->err || r->p + 8 > r->end) { r->err = 1; return 0; }
    uint64_t v; memcpy(&v, r->p, 8); r->p += 8; return v;
}
static uint32_t rd_u32(reader_t *r) {
    if (r->err || r->p + 4 > r->end) { r->err = 1; return 0; }
    uint32_t v; memcpy(&v, r->p, 4); r->p += 4; return v;
}
static const uint8_t *rd_str(reader_t *r, size_t *len) {
    uint64_t n = rd_u64(r);
    if (r->err || r->p + n > r->end) { r->err = 1; *len = 0; return NULL; }
    const uint8_t *s = r->p; r->p += n; *len = (size_t)n;
    return s;
}

/* GGUF value types */
#define GVAL_UINT8 0
#define GVAL_INT8  1
#define GVAL_UINT16 2
#define GVAL_INT16 3
#define GVAL_UINT32 4
#define GVAL_INT32 5
#define GVAL_FLOAT32 6
#define GVAL_BOOL 7
#define GVAL_STRING 8
#define GVAL_ARRAY 9
#define GVAL_UINT64 10
#define GVAL_INT64 11
#define GVAL_FLOAT64 12

static void rd_skip_val(reader_t *r, uint32_t t) {
    switch (t) {
        case GVAL_UINT8: case GVAL_INT8: r->p += 1; break;
        case GVAL_UINT16: case GVAL_INT16: r->p += 2; break;
        case GVAL_UINT32: case GVAL_INT32: case GVAL_FLOAT32: r->p += 4; break;
        case GVAL_UINT64: case GVAL_INT64: case GVAL_FLOAT64: r->p += 8; break;
        case GVAL_BOOL: r->p += 1; break;
        case GVAL_STRING: { size_t l; rd_str(r, &l); break; }
        case GVAL_ARRAY: {
            uint32_t at = rd_u32(r);
            uint64_t n = rd_u64(r);
            for (uint64_t i = 0; i < n && !r->err; i++) rd_skip_val(r, at);
            break;
        }
        default: r->err = 1;
    }
}

static int rd_uint_val(reader_t *r, uint32_t t, uint64_t *out) {
    switch (t) {
        case GVAL_UINT8:  if (r->err || r->p + 1 > r->end) { r->err = 1; return 0; } *out = (uint64_t)r->p[0]; r->p += 1; return 1;
        case GVAL_UINT16: if (r->err || r->p + 2 > r->end) { r->err = 1; return 0; } { uint16_t v; memcpy(&v, r->p, 2); r->p += 2; *out = v; return 1; }
        case GVAL_UINT32: if (r->err || r->p + 4 > r->end) { r->err = 1; return 0; } { uint32_t v; memcpy(&v, r->p, 4); r->p += 4; *out = v; return 1; }
        case GVAL_UINT64: if (r->err || r->p + 8 > r->end) { r->err = 1; return 0; } { uint64_t v; memcpy(&v, r->p, 8); r->p += 8; *out = v; return 1; }
        case GVAL_INT32:  if (r->err || r->p + 4 > r->end) { r->err = 1; return 0; } { int32_t v; memcpy(&v, r->p, 4); r->p += 4; *out = (uint64_t)(int64_t)v; return 1; }
        default: return 0;
    }
}

/* ---------------- vocab hash ---------------- */
static void tok_hash_insert(lfm_t *m, const char *str, size_t len, uint32_t id) {
    uint64_t h = lfm_hash64(str, len);
    uint32_t i = (uint32_t)(h & (m->tok_cap - 1));
    while (m->tok_ids[i] != UINT32_MAX) i = (i + 1) & (m->tok_cap - 1);
    m->tok_keys[i] = h;
    m->tok_ids[i] = id;
}

/* quant block sizes: (block bytes, elements per block) per GGML_TYPE */
static void quant_block_size(uint32_t type, size_t *bs, size_t *blk) {
    switch (type) {
        case GGML_TYPE_F32:  *bs = 4;   *blk = 1;   break;
        case GGML_TYPE_Q4_0: *bs = sizeof(block_q4_0); *blk = QK4_0; break;
        case GGML_TYPE_Q4_K: *bs = sizeof(block_q4_K); *blk = QK_K;  break;
        case GGML_TYPE_Q6_K: *bs = sizeof(block_q6_k); *blk = QK6_K; break;
        default:             *bs = 0;   *blk = 0;   break;
    }
}

/* ---------------- load ---------------- */
static int load_from_map(lfm_t *m, uint8_t *map, size_t map_size,
                         uint64_t base, uint64_t model_size,
                         char *err, size_t errlen) {
    m->map = map;
    m->map_size = map_size;
    m->g_base = base;
    m->g_size = model_size;
#if defined(MADV_HUGEPAGE)
    madvise(m->map, m->map_size, MADV_HUGEPAGE);
#endif

    reader_t r = { m->map + base, m->map + base + model_size, 0 };

    if (memcmp(r.p, "GGUF", 4) != 0) { snprintf(err, errlen, "not a GGUF file"); return -1; }
    r.p += 4;
    rd_u32(&r);   /* version */
    uint64_t n_tensors = rd_u64(&r), n_kv = rd_u64(&r);

    /* ------- metadata: pass 1 — discover architecture ------- */
    const uint8_t *kv_start = r.p;
    char arch[32] = "lfm2";
    for (uint64_t i = 0; i < n_kv && !r.err; i++) {
        size_t klen; const uint8_t *key = rd_str(&r, &klen);
        uint32_t t = rd_u32(&r);
        if (r.err) break;
        char kbuf[128]; size_t kl = klen < sizeof(kbuf)-1 ? klen : sizeof(kbuf)-1;
        memcpy(kbuf, key, kl); kbuf[kl] = 0;
        if (strcmp(kbuf, "general.architecture") == 0 && t == GVAL_STRING) {
            size_t alen; const uint8_t *av = rd_str(&r, &alen);
            if (alen > 0 && alen < sizeof(arch)) { memcpy(arch, av, alen); arch[alen] = 0; }
        } else rd_skip_val(&r, t);
    }
    if (r.err) { snprintf(err, errlen, "corrupt metadata"); return -1; }
    r.p = kv_start;
    snprintf(m->cfg.arch, sizeof(m->cfg.arch), "%s", arch);
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "%s.", arch);
    model_config_t *cfg = &m->cfg;

    float s_temp = 1.0f, s_top_p = 0.95f; int s_top_k = 20;
    uint32_t n_vocab = 0, bos_id = 0, eos_id = 0, pad_id = UINT32_MAX;
    size_t alignment = 32; /* GGUF_DEFAULT_ALIGNMENT */

    for (uint64_t i = 0; i < n_kv && !r.err; i++) {
        size_t klen; const uint8_t *key = rd_str(&r, &klen);
        uint32_t t = rd_u32(&r);
        if (r.err) break;
        char kbuf[128]; size_t kl = klen < sizeof(kbuf)-1 ? klen : sizeof(kbuf)-1;
        memcpy(kbuf, key, kl); kbuf[kl] = 0;

        const char *prop = NULL;
        if (strncmp(kbuf, prefix, strlen(prefix)) == 0) prop = kbuf + strlen(prefix);

        if (prop && !strcmp(prop, "block_count")) { uint64_t v; if (rd_uint_val(&r, t, &v)) cfg->n_layer = (uint32_t)v; else rd_skip_val(&r, t); }
        else if (prop && !strcmp(prop, "embedding_length")) { uint64_t v; if (rd_uint_val(&r, t, &v)) cfg->n_embd = (uint32_t)v; else rd_skip_val(&r, t); }
        else if (prop && !strcmp(prop, "feed_forward_length")) { uint64_t v; if (rd_uint_val(&r, t, &v)) cfg->n_ff = (uint32_t)v; else rd_skip_val(&r, t); }
        else if (prop && !strcmp(prop, "context_length")) { uint64_t v; if (rd_uint_val(&r, t, &v)) cfg->n_ctx = (uint32_t)v; else rd_skip_val(&r, t); }
        else if (prop && !strcmp(prop, "attention.head_count")) { uint64_t v; if (rd_uint_val(&r, t, &v)) cfg->n_head = (uint32_t)v; else rd_skip_val(&r, t); }
        else if (prop && !strcmp(prop, "attention.head_count_kv")) {
            /* lfm2: per-layer ARRAY (0 = short-conv, 8 = GQA). Standard
               models: scalar (same for all layers) — accept both. */
            if (t == GVAL_ARRAY) {
                uint32_t at = rd_u32(&r); uint64_t n = rd_u64(&r);
                for (uint64_t j = 0; j < n && j < N_LAYER && !r.err; j++) {
                    uint64_t v; if (rd_uint_val(&r, at, &v)) cfg->head_count_kv[j] = (uint32_t)v;
                }
                for (uint64_t j = n; j < N_LAYER && !r.err; j++) cfg->head_count_kv[j] = 0;
            } else if (t == GVAL_UINT32 || t == GVAL_UINT64) {
                uint64_t v; if (rd_uint_val(&r, t, &v))
                    for (int j = 0; j < N_LAYER; j++) cfg->head_count_kv[j] = (uint32_t)v;
                else rd_skip_val(&r, t);
            } else rd_skip_val(&r, t);
        }
        else if (prop && !strcmp(prop, "attention.layer_norm_rms_epsilon")) { if (t == GVAL_FLOAT32) { memcpy(&cfg->rms_eps, r.p, 4); r.p += 4; } else rd_skip_val(&r, t); }
        else if (prop && !strcmp(prop, "rope.dimension_count")) { uint64_t v; if (rd_uint_val(&r, t, &v)) cfg->n_rot = (uint32_t)v; else rd_skip_val(&r, t); }
        else if (prop && !strcmp(prop, "rope.freq_base")) { if (t == GVAL_FLOAT32) { memcpy(&cfg->freq_base, r.p, 4); r.p += 4; } else rd_skip_val(&r, t); }
        else if (prop && !strcmp(prop, "shortconv.l_cache")) { uint64_t v; if (rd_uint_val(&r, t, &v)) cfg->shortconv_l_cache = (uint32_t)v; else rd_skip_val(&r, t); }
        else if (strcmp(kbuf, "general.alignment") == 0) { uint64_t v; if (rd_uint_val(&r, t, &v)) alignment = (size_t)v; else rd_skip_val(&r, t); }
        else if (strcmp(kbuf, "general.sampling.temp") == 0) { if (t == GVAL_FLOAT32) { memcpy(&s_temp, r.p, 4); r.p += 4; } else rd_skip_val(&r, t); }
        else if (strcmp(kbuf, "general.sampling.top_p") == 0) { if (t == GVAL_FLOAT32) { memcpy(&s_top_p, r.p, 4); r.p += 4; } else rd_skip_val(&r, t); }
        else if (strcmp(kbuf, "general.sampling.top_k") == 0) { uint64_t v; if (rd_uint_val(&r, t, &v)) s_top_k = (int)v; else rd_skip_val(&r, t); }
        else if (strcmp(kbuf, "tokenizer.ggml.tokens") == 0) {
            if (t == GVAL_ARRAY) {
                rd_u32(&r); uint64_t n = rd_u64(&r);
                n_vocab = (uint32_t)n;
                if (n_vocab > N_VOCAB) { snprintf(err, errlen, "vocab too big: %u", n_vocab); return -1; }
                for (uint64_t j = 0; j < n && !r.err; j++) {
                    size_t l; const uint8_t *s = rd_str(&r, &l);
                    if (!r.err) {
                        m->tok_text[j] = malloc(l + 1);
                        memcpy(m->tok_text[j], s, l); m->tok_text[j][l] = 0;
                    }
                }
            } else rd_skip_val(&r, t);
        }
        else if (strcmp(kbuf, "tokenizer.ggml.token_type") == 0) {
            if (t == GVAL_ARRAY) {
                uint32_t at = rd_u32(&r); uint64_t n = rd_u64(&r);
                for (uint64_t j = 0; j < n && !r.err; j++) {
                    uint64_t v; if (rd_uint_val(&r, at, &v)) {
                        if (j < N_VOCAB) {
                            uint32_t g = (uint32_t)v;
                            m->tok_type[j] = (uint8_t)(g == 0 ? 0 : (1u << (g - 1)));
                        }
                    }
                }
            } else rd_skip_val(&r, t);
        }
        else if (strcmp(kbuf, "tokenizer.ggml.merges") == 0) { rd_skip_val(&r, t); } /* re-walk later */
        else if (strcmp(kbuf, "tokenizer.ggml.bos_token_id") == 0) { uint64_t v; if (rd_uint_val(&r, t, &v)) bos_id = (uint32_t)v; else rd_skip_val(&r, t); }
        else if (strcmp(kbuf, "tokenizer.ggml.eos_token_id") == 0) { uint64_t v; if (rd_uint_val(&r, t, &v)) eos_id = (uint32_t)v; else rd_skip_val(&r, t); }
        else if (strcmp(kbuf, "tokenizer.ggml.padding_token_id") == 0) { uint64_t v; if (rd_uint_val(&r, t, &v)) pad_id = (uint32_t)v; else rd_skip_val(&r, t); }
        else if (strcmp(kbuf, "tokenizer.chat_template") == 0) { if (t == GVAL_STRING) { size_t l; rd_str(&r, &l); } else rd_skip_val(&r, t); }
        else rd_skip_val(&r, t);
    }
    if (r.err) { snprintf(err, errlen, "corrupt metadata"); return -1; }

    /* derived config */
    cfg->n_vocab = n_vocab;
    cfg->n_embd_head = cfg->n_head ? cfg->n_embd / cfg->n_head : 0;
    if (cfg->n_rot == 0) cfg->n_rot = cfg->n_embd_head;
    if (cfg->freq_base == 0.0f) cfg->freq_base = 10000.0f;
    if (cfg->rms_eps == 0.0f) cfg->rms_eps = 1e-5f;
    if (cfg->n_ctx == 0 || cfg->n_ctx > N_CTX_MAX) cfg->n_ctx = N_CTX_MAX;
    cfg->tied = 1;   /* lfm2: no output tensor, logits = x · token_embd */
    cfg->n_attn = cfg->n_conv = 0;
    for (int i = 0; i < (int)cfg->n_layer && i < N_LAYER; i++) {
        if (cfg->head_count_kv[i] > 0) cfg->n_attn++; else cfg->n_conv++;
    }

    /* arch gate: ma3 is single-model (lfm2) */
    if (strcmp(cfg->arch, "lfm2") != 0) {
        snprintf(err, errlen, "unsupported architecture '%s' — ma3 targets lfm2 (LFM2.5-2.6B)",
                 cfg->arch);
        return -1;
    }
    if (cfg->n_layer == 0 || cfg->n_embd == 0 || cfg->n_head == 0 || n_vocab == 0 ||
        cfg->n_attn == 0 || cfg->shortconv_l_cache == 0) {
        snprintf(err, errlen, "lfm2 metadata incomplete (layers %u embd %u heads %u vocab %u)",
                 cfg->n_layer, cfg->n_embd, cfg->n_head, n_vocab);
        return -1;
    }

    m->s_temp = s_temp; m->s_top_p = s_top_p; m->s_top_k = s_top_k;
    m->n_vocab = (int)n_vocab;
    m->id_bos = bos_id; m->id_eos = eos_id;

    /* ------- tensor infos ------- */
    m->n_tensors = (uint32_t)n_tensors;
    m->tensors = malloc(sizeof(tensor_t) * n_tensors);
    for (uint64_t i = 0; i < n_tensors && !r.err; i++) {
        tensor_t *t = &m->tensors[i];
        size_t nlen; const uint8_t *nm = rd_str(&r, &nlen);
        size_t cl = nlen < sizeof(t->name)-1 ? nlen : sizeof(t->name)-1;
        memcpy(t->name, nm, cl); t->name[cl] = 0;
        uint32_t nd = rd_u32(&r);
        int64_t dims[2] = {1, 1};
        for (uint32_t d = 0; d < nd && d < 2 && !r.err; d++) dims[d] = (int64_t)rd_u64(&r);
        for (uint32_t d = 2; d < nd && !r.err; d++) rd_u64(&r);
        t->type = rd_u32(&r);
        t->off  = (size_t)rd_u64(&r);
        t->ne[0] = dims[0]; t->ne[1] = dims[1];
        t->nbytes = 0;
    }
    if (r.err) { snprintf(err, errlen, "corrupt tensor info"); return -1; }

    /* data section starts aligned after tensor infos */
    {
        uintptr_t p = (uintptr_t)r.p;
        size_t pad = (size_t)((alignment - (p % alignment)) % alignment);
        r.p += pad;
    }
    m->data = (uint8_t *)r.p;

    /* sizes + range check */
    for (uint32_t i = 0; i < m->n_tensors; i++) {
        tensor_t *t = &m->tensors[i];
        size_t bs, blk;
        quant_block_size(t->type, &bs, &blk);
        if (bs == 0) { snprintf(err, errlen, "tensor %s: unsupported quant type %u", t->name, t->type); return -1; }
        if (t->ne[0] % (int64_t)blk != 0) { snprintf(err, errlen, "tensor %s: ne0 not block multiple", t->name); return -1; }
        t->nbytes = (size_t)((t->ne[0] / (int64_t)blk) * t->ne[1]) * bs;
        if (t->off + t->nbytes > (size_t)(m->map + m->map_size - m->data)) {
            snprintf(err, errlen, "tensor %s out of file range", t->name); return -1;
        }
    }

    /* ------- resolve weights by name (lfm2 binding) ------- */
    #define FIND(dst, nm) do { \
        dst = NULL; \
        for (uint32_t i = 0; i < m->n_tensors; i++) \
            if (strcmp(m->tensors[i].name, nm) == 0) { dst = m->data + m->tensors[i].off; break; } \
        if (!dst) { snprintf(err, errlen, "missing tensor: %s", nm); return -1; } \
    } while (0)
    /* FIND_T(dst, nm, wslot): like FIND but records the weight's GGML type
       for the matmul dispatcher (Q4_0 / Q4_K / Q6_K). */
    #define FIND_T(dst, nm, wslot) do { \
        dst = NULL; \
        for (uint32_t i = 0; i < m->n_tensors; i++) \
            if (strcmp(m->tensors[i].name, nm) == 0) { dst = m->data + m->tensors[i].off; \
                m->layers[il].wtype[wslot] = m->tensors[i].type; break; } \
        if (!dst) { snprintf(err, errlen, "missing tensor: %s", nm); return -1; } \
    } while (0)

    FIND(m->token_embd, "token_embd.weight");
    FIND(m->token_embd_norm, "token_embd_norm.weight");

    char nm[96];
    for (int il = 0; il < (int)cfg->n_layer; il++) {
        int at = cfg->head_count_kv[il] > 0;
        snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight", il); FIND(m->layers[il].attn_norm, nm);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_norm.weight", il);  FIND(m->layers[il].ffn_norm, nm);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_gate.weight", il);  FIND_T(m->layers[il].ffn_gate, nm, LFM_W_GATE);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_up.weight", il);    FIND_T(m->layers[il].ffn_up, nm, LFM_W_UP);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_down.weight", il);  FIND_T(m->layers[il].ffn_down, nm, LFM_W_DOWN);
        if (at) {
            snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight", il);     FIND_T(m->layers[il].wq, nm, LFM_W_Q);
            snprintf(nm, sizeof(nm), "blk.%d.attn_k.weight", il);     FIND_T(m->layers[il].wk, nm, LFM_W_K);
            snprintf(nm, sizeof(nm), "blk.%d.attn_v.weight", il);     FIND_T(m->layers[il].wv, nm, LFM_W_V);
            snprintf(nm, sizeof(nm), "blk.%d.attn_output.weight", il); FIND_T(m->layers[il].wo, nm, LFM_W_O);
            snprintf(nm, sizeof(nm), "blk.%d.attn_q_norm.weight", il); FIND(m->layers[il].q_norm, nm);
            snprintf(nm, sizeof(nm), "blk.%d.attn_k_norm.weight", il); FIND(m->layers[il].k_norm, nm);
        } else {
            snprintf(nm, sizeof(nm), "blk.%d.shortconv.in_proj.weight", il); FIND_T(m->layers[il].conv_in_proj, nm, LFM_W_CIN);
            snprintf(nm, sizeof(nm), "blk.%d.shortconv.conv.weight", il);     FIND(m->layers[il].conv, nm);
            snprintf(nm, sizeof(nm), "blk.%d.shortconv.out_proj.weight", il); FIND_T(m->layers[il].conv_out_proj, nm, LFM_W_COUT);
        }
    }
    #undef FIND

    /* ------- tokenizer tables ------- */
    m->tok_cap = 1;
    while (m->tok_cap < (uint32_t)(m->n_vocab * 2)) m->tok_cap <<= 1;
    m->tok_keys = malloc(sizeof(uint64_t) * m->tok_cap);
    m->tok_ids  = malloc(sizeof(uint32_t) * m->tok_cap);
    for (uint32_t i = 0; i < m->tok_cap; i++) m->tok_ids[i] = UINT32_MAX;
    for (uint32_t i = 0; i < (uint32_t)m->n_vocab; i++) {
        if (m->tok_text[i]) tok_hash_insert(m, m->tok_text[i], strlen(m->tok_text[i]), i);
    }

    /* byte -> token id (gpt2 byte<->unicode map) */
    {
        uint32_t bmap[256];
        for (int b = 0; b < 256; b++) {
            int self = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
            bmap[b] = self ? (uint32_t)b : 0;
        }
        uint32_t n = 0;
        for (int b = 0; b < 256; b++) if (!bmap[b]) bmap[b] = 256 + n++;
        for (int b = 0; b < 256; b++) m->byte_to_id[b] = UINT32_MAX;
        for (int b = 0; b < 256; b++) {
            char utf8[8];
            uint32_t c = bmap[b];
            if (c < 0x80) { utf8[0] = (char)c; utf8[1] = 0; }
            else if (c < 0x800) { utf8[0] = (char)(0xC0 | (c >> 6)); utf8[1] = (char)(0x80 | (c & 0x3F)); utf8[2] = 0; }
            else { utf8[0] = (char)(0xE0 | (c >> 12)); utf8[1] = (char)(0x80 | ((c >> 6) & 0x3F)); utf8[2] = (char)(0x80 | (c & 0x3F)); utf8[3] = 0; }
            uint64_t h = lfm_hash64(utf8, strlen(utf8));
            uint32_t i = (uint32_t)(h & (m->tok_cap - 1));
            while (m->tok_ids[i] != UINT32_MAX) {
                if (m->tok_keys[i] == h && strcmp(m->tok_text[m->tok_ids[i]], utf8) == 0) {
                    m->byte_to_id[b] = m->tok_ids[i]; break;
                }
                i = (i + 1) & (m->tok_cap - 1);
            }
        }
    }

    /* special tokens: CONTROL(4) | USER_DEFINED(8) flag bits, len desc */
    for (int i = 0; i < m->n_vocab; i++) {
        if (m->tok_type[i] & (4 | 8)) m->special_ids[m->n_special++] = (uint32_t)i;
    }
    for (int i = 0; i < m->n_special; i++)
        for (int j = i + 1; j < m->n_special; j++)
            if (strlen(m->tok_text[m->special_ids[j]]) > strlen(m->tok_text[m->special_ids[i]])) {
                uint32_t tmp = m->special_ids[i]; m->special_ids[i] = m->special_ids[j]; m->special_ids[j] = tmp;
            }
    for (int i = 0; i < m->n_special; i++) m->tok_is_special[m->special_ids[i]] = 1;

    /* chat template ids */
    m->id_im_start = m->id_im_end = m->id_think_start = m->id_think_end = UINT32_MAX;
    m->id_tool_start = m->id_tool_end = UINT32_MAX;
    for (int i = 0; i < m->n_vocab; i++) {
        if (!m->tok_text[i]) continue;
        if (strcmp(m->tok_text[i], "<|im_start|>") == 0)       m->id_im_start = (uint32_t)i;
        else if (strcmp(m->tok_text[i], "<|im_end|>") == 0)    m->id_im_end = (uint32_t)i;
        else if (strcmp(m->tok_text[i], "<think>") == 0)       m->id_think_start = (uint32_t)i;
        else if (strcmp(m->tok_text[i], "</think>") == 0)      m->id_think_end = (uint32_t)i;
        else if (strcmp(m->tok_text[i], "<|tool_call_start|>") == 0) m->id_tool_start = (uint32_t)i;
        else if (strcmp(m->tok_text[i], "<|tool_call_end|>") == 0)   m->id_tool_end = (uint32_t)i;
    }
    if (m->id_eos == UINT32_MAX || eos_id == 0) {
        /* eos = <|im_end|> for lfm2 (124900) */
        m->id_eos = m->id_im_end;
    }

    /* ------- merges (the model's own table — matches HF tokenizer) ------- */
    {
        reader_t r2 = { m->map + m->g_base, m->map + m->g_base + m->g_size, 0 };
        r2.p += 4 + 4 + 8 + 8;
        const uint8_t *merges = NULL; size_t merges_len = 0; uint64_t n_merges = 0;
        for (uint64_t i = 0; i < n_kv && !r2.err; i++) {
            size_t klen; const uint8_t *key = rd_str(&r2, &klen);
            uint32_t t = rd_u32(&r2);
            char kbuf[128]; size_t kl = klen < sizeof(kbuf)-1 ? klen : sizeof(kbuf)-1;
            memcpy(kbuf, key, kl); kbuf[kl] = 0;
            if (strcmp(kbuf, "tokenizer.ggml.merges") == 0) {
                if (t == GVAL_ARRAY) {
                    rd_u32(&r2);
                    n_merges = rd_u64(&r2);
                    merges = r2.p;
                    reader_t r3 = r2;
                    for (uint64_t j = 0; j < n_merges && !r3.err; j++) { size_t l; rd_str(&r3, &l); }
                    merges_len = (size_t)(r3.p - r2.p);
                }
                break;
            }
            rd_skip_val(&r2, t);
        }
        if (!merges) { snprintf(err, errlen, "missing tokenizer.ggml.merges"); return -1; }

        m->mg_cap = 1;
        while (m->mg_cap < 2 * (uint32_t)n_merges) m->mg_cap <<= 1;
        m->mg_keys = malloc(sizeof(uint64_t) * m->mg_cap);
        m->mg_rank = malloc(sizeof(uint32_t) * m->mg_cap);
        for (uint32_t i = 0; i < m->mg_cap; i++) m->mg_rank[i] = UINT32_MAX;

        reader_t rm = { merges, merges + merges_len, 0 };
        uint32_t rank = 0;
        while (!rm.err && rm.p + 2 <= rm.end) {
            size_t l; const uint8_t *s = rd_str(&rm, &l);
            if (rm.err) break;
            const uint8_t *sp = NULL;
            for (size_t i = 1; i < l; i++) if (s[i] == ' ') { sp = s + i; break; }
            if (!sp) { rm.err = 1; break; }
            size_t l1 = (size_t)(sp - s);
            size_t l2 = l - l1 - 1;
            const uint8_t *s1 = s, *s2 = sp + 1;
            uint32_t id1 = UINT32_MAX, id2 = UINT32_MAX;
            {
                uint64_t h1 = lfm_hash64(s1, l1);
                uint32_t i1 = (uint32_t)(h1 & (m->tok_cap - 1));
                while (m->tok_ids[i1] != UINT32_MAX) {
                    if (m->tok_keys[i1] == h1 && strncmp(m->tok_text[m->tok_ids[i1]], (const char*)s1, l1) == 0 &&
                        strlen(m->tok_text[m->tok_ids[i1]]) == l1) { id1 = m->tok_ids[i1]; break; }
                    i1 = (i1 + 1) & (m->tok_cap - 1);
                }
                uint64_t h2 = lfm_hash64(s2, l2);
                uint32_t i2 = (uint32_t)(h2 & (m->tok_cap - 1));
                while (m->tok_ids[i2] != UINT32_MAX) {
                    if (m->tok_keys[i2] == h2 && strncmp(m->tok_text[m->tok_ids[i2]], (const char*)s2, l2) == 0 &&
                        strlen(m->tok_text[m->tok_ids[i2]]) == l2) { id2 = m->tok_ids[i2]; break; }
                    i2 = (i2 + 1) & (m->tok_cap - 1);
                }
            }
            if (id1 == UINT32_MAX || id2 == UINT32_MAX) {
                fprintf(stderr, "merge %u lookup fail: '%.*s' id1=%u '%.*s' id2=%u\n",
                        rank, (int)l1, s1, id1, (int)l2, s2, id2);
                rm.err = 1; break;
            }
            uint8_t pair[8]; memcpy(pair, &id1, 4); memcpy(pair + 4, &id2, 4);
            uint64_t h = lfm_hash64(pair, 8);
            uint32_t i = (uint32_t)(h & (m->mg_cap - 1));
            while (m->mg_rank[i] != UINT32_MAX) i = (i + 1) & (m->mg_cap - 1);
            m->mg_keys[i] = h;
            m->mg_rank[i] = rank;
            rank++;
        }
        if (rm.err) { snprintf(err, errlen, "corrupt merges list"); return -1; }
    }

    (void)pad_id;
    return 0;
}

void lfm_free(lfm_t *m) {
    if (m->map_malloc) {
        free(m->map);
    } else if (m->fd >= 0) {
        munmap(m->map, m->map_size);
        close(m->fd);
    }
    for (int i = 0; i < N_VOCAB && i < m->n_vocab; i++) free(m->tok_text[i]);
    free(m->tensors);
    free(m->tok_keys);
    free(m->tok_ids);
    free(m->mg_keys);
    free(m->mg_rank);
    memset(m, 0, sizeof(*m));
}

/* Load a GGUF from an external file. base=0 for a standalone model file. */
int lfm_load(lfm_t *m, const char *path, uint64_t base, char *err, size_t errlen) {
    memset(m, 0, sizeof(*m));
    m->fd = -1;
    int fd = open(path, O_RDONLY);
    if (fd < 0) { snprintf(err, errlen, "cannot open %s", path); return -1; }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 16) {
        close(fd);
        snprintf(err, errlen, "cannot stat %s", path);
        return -1;
    }
    /* resident mode (default): read the whole file into anonymous RAM so
       the weights are NEVER re-read from storage — every token pass is a
       RAM read. mmap streams from flash under memory pressure (battery +
       heat on a phone). Set MA3_MMAP=1 for the old streaming behavior. */
    const size_t fsz = (size_t)st.st_size;
    uint8_t *buf = NULL;
    if (getenv("MA3_MMAP") == NULL) {
        buf = malloc(fsz);
        if (buf) {
            size_t done = 0;
            while (done < fsz) {
                ssize_t r = read(fd, buf + done, fsz - done);
                if (r <= 0) break;
                done += (size_t)r;
            }
            if (done == fsz) {
                close(fd);
                m->fd = -1;
                m->map_malloc = 1;
                fprintf(stderr, "qma: model resident in RAM (%zu MiB)\n", fsz >> 20);
                return load_from_map(m, buf, fsz, base, fsz - base, err, errlen);
            }
            free(buf);
            buf = NULL;
        }
    }
    m->fd = fd;
    uint8_t *map = mmap(NULL, fsz, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) { close(fd); m->fd = -1; snprintf(err, errlen, "mmap failed"); return -1; }
    return load_from_map(m, map, fsz, base, fsz - base, err, errlen);
}

int lfm_load_map(lfm_t *m, uint8_t *map, size_t map_size, uint64_t base,
                 uint64_t model_size, char *err, size_t errlen) {
    return load_from_map(m, map, map_size, base, model_size, err, errlen);
}
