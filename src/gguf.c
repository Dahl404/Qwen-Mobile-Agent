/*
 * gguf.c - GGUF v3 loader for qma-27B-Q1_0.gguf
 *
 * Parses the header (metadata + tensor infos), mmaps the file, resolves
 * every weight the qwen35 model needs, and builds the BPE tokenizer tables.
 * Tensor data offsets in GGUF v3 are relative to the start of the tensor
 * data section (which begins right after the last tensor info).
 */
#include "qma.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>


uint64_t qma_hash64(const void *data, size_t len) {
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

/* read a single value; only used for the few keys we consume */
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
static void tok_hash_insert(qma_t *m, const char *str, size_t len, uint32_t id) {
    uint64_t h = qma_hash64(str, len);
    uint32_t i = (uint32_t)(h & (m->tok_cap - 1));
    while (m->tok_ids[i] != UINT32_MAX) i = (i + 1) & (m->tok_cap - 1);
    m->tok_keys[i] = h;
    m->tok_ids[i] = id;
}

/* ---------------- load ---------------- */
int qma_load(qma_t *m, const char *path, char *err, size_t errlen) {
    memset(m, 0, sizeof(*m));
    m->fd = -1;
    m->dio_fd = -1;

    m->fd = open(path, O_RDONLY);
    if (m->fd < 0) { snprintf(err, errlen, "cannot open %s", path); return -1; }

    struct stat st;
    if (fstat(m->fd, &st) != 0 || st.st_size < 16) {
        snprintf(err, errlen, "cannot stat %s", path); return -1;
    }
    m->map_size = (size_t)st.st_size;
    m->map = mmap(NULL, m->map_size, PROT_READ, MAP_SHARED, m->fd, 0);
    if (m->map == MAP_FAILED) { snprintf(err, errlen, "mmap failed"); return -1; }
    /* huge pages: 3.8GB at 4KB pages = ~950k PTEs per core; THP cuts TLB
       pressure during weight streaming (best-effort, no-op if unsupported) */
    madvise(m->map, m->map_size, MADV_HUGEPAGE);
    /* the whole file is hot: pre-warm the page cache so first-token eval
       doesn't fault weights in from flash mid-generation */
    madvise(m->map, m->map_size, MADV_WILLNEED);

    reader_t r = { m->map, m->map + m->map_size, 0 };

    /* magic + version */
    if (memcmp(r.p, "GGUF", 4) != 0) { snprintf(err, errlen, "not a GGUF file"); return -1; }
    r.p += 4;
    uint32_t version = rd_u32(&r);
    uint64_t n_tensors = rd_u64(&r), n_kv = rd_u64(&r);

    /* ------- metadata ------- */
    uint32_t block_count = 0, embd = 0, head = 0, head_kv = 0;
    uint32_t n_rot = 0, rope_s0 = 0, rope_s1 = 0, rope_s2 = 0;
    float freq_base = 0, rms_eps = 0;
    float s_temp = 1.0f, s_top_p = 0.95f; int s_top_k = 20;
    uint32_t n_vocab = 0, bos_id = 0, eos_id = 0;
    int add_bos = -1;
    size_t alignment = 32; /* GGUF_DEFAULT_ALIGNMENT */
    const uint8_t *chat_tmpl = NULL; size_t chat_tmpl_len = 0;

    for (uint64_t i = 0; i < n_kv && !r.err; i++) {
        size_t klen; const uint8_t *key = rd_str(&r, &klen);
        uint32_t t = rd_u32(&r);
        if (r.err) break;
        char kbuf[128]; size_t kl = klen < sizeof(kbuf)-1 ? klen : sizeof(kbuf)-1;
        memcpy(kbuf, key, kl); kbuf[kl] = 0;

        /* strip the architecture prefix so one reader handles the
           metadata (this engine runs the qwen35moe MoE model) */
        if (strncmp(kbuf, "qwen35moe.", 10) == 0) {
            memmove(kbuf, kbuf + 10, kl - 10 + 1);
        } else if (strncmp(kbuf, "qwen35.", 7) == 0) {
            memmove(kbuf, kbuf + 7, kl - 7 + 1);
        }

        if (strcmp(kbuf, "block_count") == 0) { uint64_t v; if (rd_uint_val(&r, t, &v)) block_count = (uint32_t)v; else rd_skip_val(&r, t); }
        else if (strcmp(kbuf, "embedding_length") == 0) { uint64_t v; if (rd_uint_val(&r, t, &v)) embd = (uint32_t)v; else rd_skip_val(&r, t); }
        else if (strcmp(kbuf, "attention.head_count") == 0) { uint64_t v; if (rd_uint_val(&r, t, &v)) head = (uint32_t)v; else rd_skip_val(&r, t); }
        else if (strcmp(kbuf, "attention.head_count_kv") == 0) { uint64_t v; if (rd_uint_val(&r, t, &v)) head_kv = (uint32_t)v; else rd_skip_val(&r, t); }
        else if (strcmp(kbuf, "rope.dimension_count") == 0) { uint64_t v; if (rd_uint_val(&r, t, &v)) n_rot = (uint32_t)v; else rd_skip_val(&r, t); }
        else if (strcmp(kbuf, "rope.freq_base") == 0) { if (t == GVAL_FLOAT32) { memcpy(&freq_base, r.p, 4); r.p += 4; } else rd_skip_val(&r, t); }
        else if (strcmp(kbuf, "attention.layer_norm_rms_epsilon") == 0) { if (t == GVAL_FLOAT32) { memcpy(&rms_eps, r.p, 4); r.p += 4; } else rd_skip_val(&r, t); }
        else if (strcmp(kbuf, "rope.dimension_sections") == 0) {
            if (t == GVAL_ARRAY) {
                uint32_t at = rd_u32(&r); uint64_t n = rd_u64(&r);
                uint32_t vals[4] = {0,0,0,0};
                for (uint64_t j = 0; j < n && j < 4 && !r.err; j++) { uint64_t v; if (rd_uint_val(&r, at, &v)) vals[j] = (uint32_t)v; }
                rope_s0 = vals[0]; rope_s1 = vals[1]; rope_s2 = vals[2];
            } else rd_skip_val(&r, t);
        }
        else if (strcmp(kbuf, "general.alignment") == 0) { uint64_t v; if (rd_uint_val(&r, t, &v)) alignment = (size_t)v; else rd_skip_val(&r, t); }
        else if (strcmp(kbuf, "general.sampling.temp") == 0) { if (t == GVAL_FLOAT32) { memcpy(&s_temp, r.p, 4); r.p += 4; } else rd_skip_val(&r, t); }
        else if (strcmp(kbuf, "general.sampling.top_p") == 0) { if (t == GVAL_FLOAT32) { memcpy(&s_top_p, r.p, 4); r.p += 4; } else rd_skip_val(&r, t); }
        else if (strcmp(kbuf, "general.sampling.top_k") == 0) { uint64_t v; if (rd_uint_val(&r, t, &v)) s_top_k = (int)v; else rd_skip_val(&r, t); }
        else if (strcmp(kbuf, "tokenizer.ggml.tokens") == 0) {
            if (t == GVAL_ARRAY) {
                rd_u32(&r); /* array elem type (unused) */ uint64_t n = rd_u64(&r);
                n_vocab = (uint32_t)n;
                if (n_vocab > N_VOCAB) { snprintf(err, errlen, "vocab too big: %u", n_vocab); return -1; }
                for (uint64_t j = 0; j < n && !r.err; j++) {
                    size_t l; const uint8_t *s = rd_str(&r, &l);
                    if (!r.err) {
                        /* no length cap: the fork reads token strings of any
                           length (max_token_len is unbounded) */
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
                            /* GGUF token type -> llama token attr: 1<<(t-1) */
                            uint32_t g = (uint32_t)v;
                            m->tok_type[j] = (uint8_t)(g == 0 ? 0 : (1u << (g - 1)));
                        }
                    }
                }
            } else rd_skip_val(&r, t);
        }
        else if (strcmp(kbuf, "tokenizer.ggml.merges") == 0) {
            /* consumed later: we stash the offsets */
            /* (re-parse from raw bytes below instead) */
            rd_skip_val(&r, t);
        }
        else if (strcmp(kbuf, "tokenizer.ggml.add_bos_token") == 0) {
            if (t == GVAL_BOOL) { add_bos = r.p[0] ? 1 : 0; r.p += 1; } else rd_skip_val(&r, t);
        }
        else if (strcmp(kbuf, "tokenizer.ggml.bos_token_id") == 0) { uint64_t v; if (rd_uint_val(&r, t, &v)) bos_id = (uint32_t)v; else rd_skip_val(&r, t); }
        else if (strcmp(kbuf, "tokenizer.ggml.eos_token_id") == 0) { uint64_t v; if (rd_uint_val(&r, t, &v)) eos_id = (uint32_t)v; else rd_skip_val(&r, t); }
        else if (strcmp(kbuf, "tokenizer.chat_template") == 0) {
            if (t == GVAL_STRING) { chat_tmpl = rd_str(&r, &chat_tmpl_len); }
            else rd_skip_val(&r, t);
        }
        else rd_skip_val(&r, t);
    }
    if (r.err) { snprintf(err, errlen, "corrupt metadata"); return -1; }

    if (block_count != N_LAYER || embd != N_EMBD || head != N_HEAD || head_kv != N_HEAD_KV ||
        n_rot != N_ROT || rope_s0 != ROPE_SECT0 || rope_s1 != ROPE_SECT1 || rope_s2 != ROPE_SECT2) {
        snprintf(err, errlen, "model hyperparameters mismatch: this engine targets "
                 "qwen35 %d layers, embd %d, heads %d/%d, n_rot %d, sections %d/%d/%d "
                 "(file has %u/%u/%u/%u/%u/%u/%u/%u)",
                 N_LAYER, N_EMBD, N_HEAD, N_HEAD_KV, N_ROT, ROPE_SECT0, ROPE_SECT1, ROPE_SECT2,
                 block_count, embd, head, head_kv, n_rot, rope_s0, rope_s1, rope_s2);
        return -1;
    }
    (void)freq_base; (void)rms_eps; (void)bos_id; (void)eos_id; (void)add_bos;
    (void)chat_tmpl; (void)chat_tmpl_len;
    m->s_temp = s_temp; m->s_top_p = s_top_p; m->s_top_k = s_top_k;
    m->n_vocab = (int)n_vocab;

    /* ------- tensor infos ------- */
    m->n_tensors = (uint32_t)n_tensors;
    m->tensors = malloc(sizeof(tensor_t) * n_tensors);
    for (uint64_t i = 0; i < n_tensors && !r.err; i++) {
        tensor_t *t = &m->tensors[i];
        size_t nlen; const uint8_t *nm = rd_str(&r, &nlen);
        size_t cl = nlen < sizeof(t->name)-1 ? nlen : sizeof(t->name)-1;
        memcpy(t->name, nm, cl); t->name[cl] = 0;
        uint32_t nd = rd_u32(&r);
        int64_t dims[3] = {1, 1, 1};
        for (uint32_t d = 0; d < nd && d < 3 && !r.err; d++) dims[d] = (int64_t)rd_u64(&r);
        for (uint32_t d = 3; d < nd && !r.err; d++) rd_u64(&r); /* ignore extra dims */
        t->type = rd_u32(&r);
        t->off  = (size_t)rd_u64(&r);
        t->ne[0] = dims[0]; t->ne[1] = dims[1]; t->ne[2] = dims[2];
        t->nbytes = 0;
    }
    if (r.err) { snprintf(err, errlen, "corrupt tensor info"); return -1; }

    /* data section starts at an ALIGNED position after the tensor infos
       (ref: gguf_init_from_file: gr.seek(GGML_PAD(gr.tell(), alignment))) */
    {
        uintptr_t p = (uintptr_t)r.p;
        size_t pad = (size_t)((alignment - (p % alignment)) % alignment);
        r.p += pad;
    }
    m->data = (uint8_t *)r.p;

    /* compute tensor sizes + check offsets in range. The MoE expert tensors
       are 3D [n_embd, n_ff, n_expert]: the expert index is the SLOWEST dim,
       so expert e is a contiguous slab of ne0*ne1 elements -- this is what
       makes direct GGUF expert streaming possible (no repack). */
    for (uint32_t i = 0; i < m->n_tensors; i++) {
        tensor_t *t = &m->tensors[i];
        size_t bs, blk;
        switch (t->type) {
        case GGML_TYPE_F32: bs = 4; blk = 1; break;
        case GGML_TYPE_Q4_K: bs = sizeof(block_q4_K); blk = QK_K; break;
        case GGML_TYPE_Q6_K: bs = sizeof(block_q6_K); blk = QK_K; break;
        default:
            snprintf(err, errlen, "tensor %s: unsupported type %u", t->name, t->type); return -1;
        }
        if (t->ne[0] % blk != 0) { snprintf(err, errlen, "tensor %s: ne0 not block multiple", t->name); return -1; }
        t->nbytes = (size_t)((t->ne[0] / blk) * t->ne[1]) * bs * (size_t)t->ne[2];
        if (t->off + t->nbytes > (size_t)(m->map + m->map_size - m->data)) {
            snprintf(err, errlen, "tensor %s out of file range", t->name); return -1;
        }
    }

    /* ------- resolve weights by name ------- */
    /* FIND2 captures the tensor's GGUF type alongside the pointer; the type
       varies per layer for some tensors (attn_qkv, attn_v, ffn_down_exps,
       ffn_down_shexp are Q4_K or Q6_K depending on the layer). */
    #define FIND_T(dst, tt, nm) do { \
        dst = NULL; tt = 0; \
        for (uint32_t i = 0; i < m->n_tensors; i++) \
            if (strcmp(m->tensors[i].name, nm) == 0) { dst = m->data + m->tensors[i].off; tt = m->tensors[i].type; break; } \
        if (!dst) { snprintf(err, errlen, "missing tensor: %s", nm); return -1; } \
    } while (0)
    #define FIND(dst, nm) do { uint32_t _t; FIND_T(dst, _t, nm); } while (0)
    /* like FIND_T but also records the tensor's absolute file offset */
    #define FIND_TOFF(dst, tt, offp, nm) do { \
        dst = NULL; tt = 0; offp = 0; \
        for (uint32_t i = 0; i < m->n_tensors; i++) \
            if (strcmp(m->tensors[i].name, nm) == 0) { dst = m->data + m->tensors[i].off; tt = m->tensors[i].type; offp = m->data_section_abs + m->tensors[i].off; break; } \
        if (!dst) { snprintf(err, errlen, "missing tensor: %s", nm); return -1; } \
    } while (0)

    FIND_T(m->token_embd, m->t_token_embd, "token_embd.weight");
    FIND(m->output_norm, "output_norm.weight");
    FIND_T(m->output, m->t_output, "output.weight");
    m->data_section_abs = (size_t)(m->data - m->map);  /* abs file offset of data section */

    char nm[96];
    for (int il = 0; il < N_LAYER; il++) {
        int at = IS_ATTN(il);
        snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight", il);      FIND(m->layers[il].attn_norm, nm);
        snprintf(nm, sizeof(nm), "blk.%d.post_attention_norm.weight", il); FIND(m->layers[il].attn_post_norm, nm);
        /* MoE FFN (all layers) */
        snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_inp.weight", il);      FIND(m->layers[il].ffn_gate_inp, nm);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_exps.weight", il);     FIND_TOFF(m->layers[il].ffn_gate_exps, m->layers[il].t_gate_exps, m->layers[il].off_gate_exps, nm);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_up_exps.weight", il);       FIND_TOFF(m->layers[il].ffn_up_exps, m->layers[il].t_up_exps, m->layers[il].off_up_exps, nm);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_down_exps.weight", il);     FIND_TOFF(m->layers[il].ffn_down_exps, m->layers[il].t_down_exps, m->layers[il].off_down_exps, nm);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_inp_shexp.weight", il);FIND(m->layers[il].ffn_gate_inp_shexp, nm);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_shexp.weight", il);    FIND_TOFF(m->layers[il].ffn_gate_shexp, m->layers[il].t_gate_shexp, m->layers[il].off_gate_shexp, nm);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_up_shexp.weight", il);      FIND_TOFF(m->layers[il].ffn_up_shexp, m->layers[il].t_up_shexp, m->layers[il].off_up_shexp, nm);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_down_shexp.weight", il);    FIND_TOFF(m->layers[il].ffn_down_shexp, m->layers[il].t_down_shexp, m->layers[il].off_down_shexp, nm);
        if (at) {
            snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight", il);     FIND_T(m->layers[il].wq, m->layers[il].t_wq, nm);
            snprintf(nm, sizeof(nm), "blk.%d.attn_k.weight", il);     FIND_T(m->layers[il].wk, m->layers[il].t_wk, nm);
            snprintf(nm, sizeof(nm), "blk.%d.attn_v.weight", il);     FIND_T(m->layers[il].wv, m->layers[il].t_wv, nm);
            snprintf(nm, sizeof(nm), "blk.%d.attn_output.weight", il); FIND_T(m->layers[il].wo, m->layers[il].t_wo, nm);
            snprintf(nm, sizeof(nm), "blk.%d.attn_q_norm.weight", il); FIND(m->layers[il].q_norm, nm);
            snprintf(nm, sizeof(nm), "blk.%d.attn_k_norm.weight", il); FIND(m->layers[il].k_norm, nm);
        } else {
            snprintf(nm, sizeof(nm), "blk.%d.attn_qkv.weight", il);   FIND_T(m->layers[il].wqkv, m->layers[il].t_wqkv, nm);
            snprintf(nm, sizeof(nm), "blk.%d.attn_gate.weight", il);  FIND_T(m->layers[il].attn_gate, m->layers[il].t_gate, nm);
            snprintf(nm, sizeof(nm), "blk.%d.ssm_conv1d.weight", il); FIND(m->layers[il].ssm_conv1d, nm);
            snprintf(nm, sizeof(nm), "blk.%d.ssm_dt.bias", il);       FIND(m->layers[il].ssm_dt, nm);
            snprintf(nm, sizeof(nm), "blk.%d.ssm_a", il);             FIND(m->layers[il].ssm_a, nm);
            snprintf(nm, sizeof(nm), "blk.%d.ssm_beta.weight", il);   FIND_T(m->layers[il].ssm_beta, m->layers[il].t_beta, nm);
            snprintf(nm, sizeof(nm), "blk.%d.ssm_alpha.weight", il);  FIND_T(m->layers[il].ssm_alpha, m->layers[il].t_alpha, nm);
            snprintf(nm, sizeof(nm), "blk.%d.ssm_norm.weight", il);   FIND(m->layers[il].ssm_norm, nm);
            snprintf(nm, sizeof(nm), "blk.%d.ssm_out.weight", il);    FIND_T(m->layers[il].ssm_out, m->layers[il].t_out, nm);
        }
    }
    fprintf(stderr, "[types] L0 wqkv=%u wv=%u down_exps=%u down_shexp=%u | L5 wqkv=%u wv=%u\n",
            m->layers[0].t_wqkv, m->layers[0].t_wv, m->layers[0].t_down_exps, m->layers[0].t_down_shexp,
            m->layers[5].t_wqkv, m->layers[5].t_wv);
    {   /* sanity: first bytes of L0 conv1d */
        const float *cw = (const float *)m->layers[0].ssm_conv1d;
        fprintf(stderr, "[conv0] engine sees w[0..3]: %g %g %g %g\n", cw[0], cw[1], cw[2], cw[3]);
    }
    {   /* sanity: first bytes of L0 wqkv as seen by the engine */
        const unsigned char *p = m->layers[0].wqkv;
        fprintf(stderr, "[wqkv0] engine sees: %02x %02x %02x %02x %02x %02x %02x %02x\n",
                p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]);
        for (uint32_t i = 0; i < m->n_tensors; i++)
            if (strcmp(m->tensors[i].name, "blk.0.ffn_gate_exps.weight") == 0) {
                fprintf(stderr, "[gateexps0] tensor off=%zu type=%u ne=[%lld %lld %lld] nbytes=%llu\n",
                        m->tensors[i].off, m->tensors[i].type,
                        (long long)m->tensors[i].ne[0], (long long)m->tensors[i].ne[1], (long long)m->tensors[i].ne[2],
                        (unsigned long long)m->tensors[i].nbytes);
            }
    }
    #undef FIND
    #undef FIND_T

    /* ------- tokenizer tables ------- */
    m->tok_cap = 1;
    while (m->tok_cap < (uint32_t)(m->n_vocab * 2)) m->tok_cap <<= 1;
    m->tok_keys = malloc(sizeof(uint64_t) * m->tok_cap);
    m->tok_ids  = malloc(sizeof(uint32_t) * m->tok_cap);
    for (uint32_t i = 0; i < m->tok_cap; i++) m->tok_ids[i] = UINT32_MAX;
    for (uint32_t i = 0; i < (uint32_t)m->n_vocab; i++) {
        if (m->tok_text[i]) tok_hash_insert(m, m->tok_text[i], strlen(m->tok_text[i]), i);
    }

    /* byte -> token id: the byte tokens are the byte-encoded single chars.
       build the byte->unicode map (GPT2 cs order) and look up each in the hash. */
    {
        uint32_t bmap[256];
        for (int b = 0; b < 256; b++) {
            int self = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
            bmap[b] = self ? (uint32_t)b : 0;
        }
        uint32_t n = 0;
        for (int b = 0; b < 256; b++) if (!bmap[b]) bmap[b] = 256 + n++;
        for (int b = 0; b < 256; b++) m->byte_to_id[b] = UINT32_MAX;
        /* insert into a temp hash then resolve; simpler: build strings and look up */
        for (int b = 0; b < 256; b++) {
            char utf8[8];
            uint32_t c = bmap[b];
            if (c < 0x80) { utf8[0] = (char)c; utf8[1] = 0; }
            else if (c < 0x800) { utf8[0] = (char)(0xC0 | (c >> 6)); utf8[1] = (char)(0x80 | (c & 0x3F)); utf8[2] = 0; }
            else { utf8[0] = (char)(0xE0 | (c >> 12)); utf8[1] = (char)(0x80 | ((c >> 6) & 0x3F)); utf8[2] = (char)(0x80 | (c & 0x3F)); utf8[3] = 0; }
            uint64_t h = qma_hash64(utf8, strlen(utf8));
            uint32_t i = (uint32_t)(h & (m->tok_cap - 1));
            while (m->tok_ids[i] != UINT32_MAX) {
                if (m->tok_keys[i] == h && strcmp(m->tok_text[m->tok_ids[i]], utf8) == 0) {
                    m->byte_to_id[b] = m->tok_ids[i]; break;
                }
                i = (i + 1) & (m->tok_cap - 1);
            }
        }
    }

    /* special tokens: CONTROL(4) | USER_DEFINED(8), sorted by length desc */
    for (int i = 0; i < m->n_vocab; i++) {
        if (m->tok_type[i] & (4 | 8)) m->special_ids[m->n_special++] = (uint32_t)i;
    }
    for (int i = 0; i < m->n_special; i++)
        for (int j = i + 1; j < m->n_special; j++)
            if (strlen(m->tok_text[m->special_ids[j]]) > strlen(m->tok_text[m->special_ids[i]])) {
                uint32_t tmp = m->special_ids[i]; m->special_ids[i] = m->special_ids[j]; m->special_ids[j] = tmp;
            }
    for (int i = 0; i < m->n_special; i++) m->tok_is_special[m->special_ids[i]] = 1;

    m->id_im_start = UINT32_MAX; m->id_im_end = UINT32_MAX;
    for (int i = 0; i < m->n_vocab; i++) {
        if (m->tok_text[i] && strcmp(m->tok_text[i], "<|im_start|>") == 0) m->id_im_start = (uint32_t)i;
        if (m->tok_text[i] && strcmp(m->tok_text[i], "<|im_end|>") == 0) m->id_im_end = (uint32_t)i;
    }

    /* ------- merges ------- */
    /* re-walk metadata to read the merges array (we skipped it before) */
    {
        reader_t r2 = { m->map, m->map + m->map_size, 0 };
        r2.p += 4 + 4 + 8 + 8;
        /* skip kv pairs until merges */
        const uint8_t *merges = NULL; size_t merges_len = 0;
        for (uint64_t i = 0; i < n_kv && !r2.err; i++) {
            size_t klen; const uint8_t *key = rd_str(&r2, &klen);
            uint32_t t = rd_u32(&r2);
            char kbuf[128]; size_t kl = klen < sizeof(kbuf)-1 ? klen : sizeof(kbuf)-1;
            memcpy(kbuf, key, kl); kbuf[kl] = 0;
            if (strcmp(kbuf, "tokenizer.ggml.merges") == 0) {
                if (t == GVAL_ARRAY) {
                    uint32_t at = rd_u32(&r2);
                    uint64_t n = rd_u64(&r2);
                    merges = r2.p; /* array elements: strings */
                    merges_len = 0;
                    /* measure */
                    reader_t r3 = r2;
                    for (uint64_t j = 0; j < n && !r3.err; j++) { size_t l; rd_str(&r3, &l); }
                    merges_len = (size_t)(r3.p - r2.p);
                    (void)at;
                }
                break;
            }
            rd_skip_val(&r2, t);
        }
        if (!merges) { snprintf(err, errlen, "missing tokenizer.ggml.merges"); return -1; }

        m->mg_cap = 1;
        while (m->mg_cap < 2 * 247587) m->mg_cap <<= 1; /* n_merges known to be 247587 */
        m->mg_keys = malloc(sizeof(uint64_t) * m->mg_cap);
        m->mg_rank = malloc(sizeof(uint32_t) * m->mg_cap);
        for (uint32_t i = 0; i < m->mg_cap; i++) m->mg_rank[i] = UINT32_MAX;

        reader_t rm = { merges, merges + merges_len, 0 };
        uint32_t rank = 0;
        while (!rm.err && rm.p + 2 <= rm.end) {
            size_t l; const uint8_t *s = rd_str(&rm, &l);
            if (rm.err) break;
            /* split on first space from position 1 (ref: word.find(' ', 1)) */
            const uint8_t *sp = NULL;
            for (size_t i = 1; i < l; i++) if (s[i] == ' ') { sp = s + i; break; }
            if (!sp) { rm.err = 1; break; }
            size_t l1 = (size_t)(sp - s);
            size_t l2 = l - l1 - 1;
            const uint8_t *s1 = s, *s2 = sp + 1;
            /* find token ids of both strings */
            uint32_t id1 = UINT32_MAX, id2 = UINT32_MAX;
            {
                uint64_t h1 = qma_hash64(s1, l1);
                uint32_t i1 = (uint32_t)(h1 & (m->tok_cap - 1));
                while (m->tok_ids[i1] != UINT32_MAX) {
                    if (m->tok_keys[i1] == h1 && strncmp(m->tok_text[m->tok_ids[i1]], (const char*)s1, l1) == 0 &&
                        strlen(m->tok_text[m->tok_ids[i1]]) == l1) { id1 = m->tok_ids[i1]; break; }
                    i1 = (i1 + 1) & (m->tok_cap - 1);
                }
                uint64_t h2 = qma_hash64(s2, l2);
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
            /* key = hash of (id1, id2) */
            uint8_t pair[8]; memcpy(pair, &id1, 4); memcpy(pair + 4, &id2, 4);
            uint64_t h = qma_hash64(pair, 8);
            uint32_t i = (uint32_t)(h & (m->mg_cap - 1));
            while (m->mg_rank[i] != UINT32_MAX) i = (i + 1) & (m->mg_cap - 1);
            m->mg_keys[i] = h;
            m->mg_rank[i] = rank;
            rank++;
        }
        if (rm.err) { snprintf(err, errlen, "corrupt merges list"); return -1; }
    }

    /* O_DIRECT expert reads: if every expert tensor is 4K-aligned AND the
       per-expert record strides are 4K-multiples, open a second fd with
       O_DIRECT so expert preads bypass the page cache entirely — no
       double-buffering with the ecache pool, and expert pages never churn
       the trunk's page cache. Falls back to buffered preads otherwise
       (unaligned file or filesystem without O_DIRECT). The file is made
       4K-aligned by qma_align_model() at startup. */
    {
        const size_t slab = (size_t)N_EMBD * N_FF_EXP;
        const size_t gu = slab * sizeof(block_q4_K) / QK_K;
        int ok = 1;
        for (int il = 0; il < N_LAYER && ok; il++) {
            const size_t dn = slab * (m->layers[il].t_down_exps == GGML_TYPE_Q4_K
                                          ? sizeof(block_q4_K) : sizeof(block_q6_K)) / QK_K;
            if (m->layers[il].off_gate_exps % 4096 || m->layers[il].off_up_exps % 4096 ||
                m->layers[il].off_down_exps % 4096 || gu % 4096 || dn % 4096)
                ok = 0;
        }
        if (ok) {
            int dfd = open(path, O_RDONLY | O_DIRECT);
            if (dfd >= 0) m->dio_fd = dfd;
        }
    }

    return 0;
}

void qma_free(qma_t *m) {
    if (m->fd >= 0) {
        munmap(m->map, m->map_size);
        close(m->fd);
    }
    if (m->dio_fd >= 0) close(m->dio_fd);
    for (int i = 0; i < N_VOCAB && i < m->n_vocab; i++) free(m->tok_text[i]);
    free(m->tensors);
    free(m->tok_keys);
    free(m->tok_ids);
    free(m->mg_keys);
    free(m->mg_rank);
    memset(m, 0, sizeof(*m));
}

/* ---------------- one-time model alignment (4K tensor starts) ----------------
 * O_DIRECT expert reads require every tensor's file offset to be 4K-aligned.
 * GGUF writers use alignment=32, so tensor starts drift out of 4K. This
 * repacks the file once (streaming, same data, +a few MB of padding) so every
 * tensor starts on a 4096 boundary and sets general.alignment=4096. The result
 * is written to <src>.4k next to the original; the caller then loads THAT file
 * and O_DIRECT expert reads become possible. Disable the automatic repack at
 * startup with QMA_NOALIGN=1. */

typedef struct {
    size_t   alignment;
    off_t    align_pos;    /* file offset of general.alignment VALUE, -1 if absent */
    int      align_width;  /* 4 or 8 */
    size_t   meta_end;     /* just past the last metadata KV */
    size_t   infos_end;    /* just past the last tensor info */
    uint32_t n_tensors;
    size_t  *off;          /* relative tensor offsets (from data section start) */
    size_t  *nbytes;
    off_t   *off_pos;      /* file offset of each offset field in the info table */
} gguf_scan_t;

static void gguf_scan_free(gguf_scan_t *s) {
    free(s->off); free(s->nbytes); free(s->off_pos);
    memset(s, 0, sizeof(*s));
}

/* header-only scan: magic/version, metadata (skipped except general.alignment),
   and the tensor info table. Mirrors qma_load's layout math exactly. */
static int gguf_scan(const char *path, gguf_scan_t *s, char *err, size_t errlen) {
    memset(s, 0, sizeof(*s));
    s->align_pos = -1;
    int fd = open(path, O_RDONLY);
    if (fd < 0) { snprintf(err, errlen, "cannot open %s", path); return -1; }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 16) {
        close(fd); snprintf(err, errlen, "bad file %s", path); return -1;
    }
    size_t msize = (size_t)st.st_size;
    uint8_t *map = mmap(NULL, msize, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) { close(fd); snprintf(err, errlen, "mmap %s", path); return -1; }
    int rc = -1;
    reader_t r = { map, map + msize, 0 };
    if (memcmp(r.p, "GGUF", 4) != 0) { snprintf(err, errlen, "%s not a GGUF", path); goto out; }
    r.p += 4;
    rd_u32(&r); /* version */
    uint64_t n_tensors = rd_u64(&r), n_kv = rd_u64(&r);
    size_t alignment = 32;
    for (uint64_t i = 0; i < n_kv && !r.err; i++) {
        size_t klen; const uint8_t *key = rd_str(&r, &klen);
        uint32_t t = rd_u32(&r);
        if (r.err) break;
        if (klen == 17 && memcmp(key, "general.alignment", 17) == 0) {
            s->align_pos = (off_t)(r.p - map);
            if (t == 4)      { s->align_width = 4; alignment = (size_t)rd_u32(&r); }
            else if (t == 10 || t == 11) { s->align_width = 8; alignment = (size_t)rd_u64(&r); }
            else             { s->align_pos = -1; rd_skip_val(&r, t); }
        } else {
            rd_skip_val(&r, t);
        }
    }
    if (r.err) { snprintf(err, errlen, "%s: corrupt metadata", path); goto out; }
    s->meta_end = (size_t)(r.p - map);
    /* NOTE: this engine (and this file) place the tensor info table
       IMMEDIATELY after the last metadata KV — no alignment padding before
       it (the padding applies to the data section only). Match that. */
    if (n_tensors > (1u << 20)) { snprintf(err, errlen, "%s: absurd tensor count", path); goto out; }
    s->n_tensors = (uint32_t)n_tensors;
    s->off      = malloc(sizeof(size_t) * s->n_tensors);
    s->nbytes   = malloc(sizeof(size_t) * s->n_tensors);
    s->off_pos  = malloc(sizeof(off_t) * s->n_tensors);
    if (!s->off || !s->nbytes || !s->off_pos) { snprintf(err, errlen, "oom"); goto out; }
    for (uint32_t i = 0; i < s->n_tensors && !r.err; i++) {
        size_t nlen; rd_str(&r, &nlen);          /* name: skip */
        uint32_t nd = rd_u32(&r);
        int64_t dims[3] = { 1, 1, 1 };
        for (uint32_t d = 0; d < nd && d < 3 && !r.err; d++) dims[d] = (int64_t)rd_u64(&r);
        for (uint32_t d = 3; d < nd && !r.err; d++) rd_u64(&r);
        uint32_t typ = rd_u32(&r);
        if (r.err) break;
        s->off_pos[i] = (off_t)(r.p - map);
        s->off[i] = (size_t)rd_u64(&r);
        if (r.err) break;
        size_t bs, blk;
        switch (typ) {
        case GGML_TYPE_F32:  bs = 4;              blk = 1;   break;
        case GGML_TYPE_Q4_K: bs = sizeof(block_q4_K); blk = QK_K; break;
        case GGML_TYPE_Q6_K: bs = sizeof(block_q6_K); blk = QK_K; break;
        default:
            snprintf(err, errlen, "%s: tensor %u unsupported type %u", path, i, typ);
            goto out;
        }
        if (dims[0] % blk != 0) { snprintf(err, errlen, "%s: tensor %u ne0 not block multiple", path, i); goto out; }
        s->nbytes[i] = (size_t)((dims[0] / blk) * dims[1]) * bs * (size_t)dims[2];
    }
    if (r.err) { snprintf(err, errlen, "%s: corrupt tensor info", path); goto out; }
    s->infos_end = (size_t)(r.p - map);
    s->alignment = alignment;
    rc = 0;
out:
    munmap(map, msize);
    close(fd);
    if (rc != 0) gguf_scan_free(s);
    return rc;
}

/* 1 = every tensor start is 4K-aligned (O_DIRECT-able), 0 = needs repack */
static int gguf_all_aligned(const gguf_scan_t *s, size_t *data_start_out) {
    const size_t ds = (s->infos_end + s->alignment - 1) / s->alignment * s->alignment;
    int ok = 1;
    for (uint32_t i = 0; i < s->n_tensors && ok; i++)
        if ((ds + s->off[i]) % 4096 != 0) ok = 0;
    if (data_start_out) *data_start_out = ds;   /* ALWAYS: caller needs it */
    return ok;
}

static double align_now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* 1 = model needs the 4K repack, 0 = already aligned, -1 = error */
int qma_model_needs_align(const char *path, char *err, size_t errlen) {
    gguf_scan_t s;
    if (gguf_scan(path, &s, err, errlen) != 0) return -1;
    int need = !gguf_all_aligned(&s, NULL);
    gguf_scan_free(&s);
    return need;
}

static int write_all(int fd, const void *buf, size_t n) {
    const uint8_t *p = buf;
    while (n > 0) {
        ssize_t w = write(fd, p, n);
        if (w <= 0) return -1;
        p += w; n -= (size_t)w;
    }
    return 0;
}

/* Ensure `src` is O_DIRECT-able: repack to <src>.4k if tensor starts aren't
   4K-aligned. On success *use_path holds the file to load (== src when already
   aligned). Failure to repack (no space, corrupt file) falls back to src with
   a warning on stderr — never a hard error. */
int qma_align_model(const char *src, char *use_path, size_t use_len, char *err, size_t errlen) {
    const double t0 = align_now_s();
    gguf_scan_t s;
    char ebuf[256];
    if (gguf_scan(src, &s, ebuf, sizeof(ebuf)) != 0) {
        snprintf(err, errlen, "%s", ebuf);
        return -1;
    }
    size_t old_ds;
    if (gguf_all_aligned(&s, &old_ds)) {
        snprintf(use_path, use_len, "%s", src);
        gguf_scan_free(&s);
        return 0;
    }
    const size_t infos_start_old = s.meta_end;   /* infos follow metadata directly */

    char cand[4096];
    snprintf(cand, sizeof(cand), "%s.4k", src);
    /* reuse a previously aligned copy if it's valid */
    {
        gguf_scan_t c;
        if (gguf_scan(cand, &c, ebuf, sizeof(ebuf)) == 0) {
            if (gguf_all_aligned(&c, NULL)) {
                struct stat cst;
                double gb = 0;
                if (stat(cand, &cst) == 0) gb = (double)cst.st_size / 1e9;
                fprintf(stderr, "qma: aligned model found: %s (%.2f GiB, %.1fs)\n",
                        cand, gb, align_now_s() - t0);
                snprintf(use_path, use_len, "%s", cand);
                gguf_scan_free(&c);
                gguf_scan_free(&s);
                return 0;
            }
            gguf_scan_free(&c);
        }
        unlink(cand);
    }

    /* free space check (need file + a 64MB buffer headroom) */
    struct stat st;
    if (stat(src, &st) != 0) { gguf_scan_free(&s); snprintf(err, errlen, "stat %s", src); return -1; }
    const uint64_t need = (uint64_t)st.st_size + (1ull << 26);
    const uint64_t total = (uint64_t)st.st_size;
    uint64_t freeb = 0;
    {
        char dir[4096]; size_t n = strlen(src);
        while (n > 0 && src[n-1] != '/') n--;
        if (n == 0) snprintf(dir, sizeof(dir), ".");
        else        snprintf(dir, sizeof(dir), "%.*s", (int)(n - 1), src);
        struct statvfs vfs;
        if (statvfs(dir, &vfs) == 0) freeb = (uint64_t)vfs.f_bavail * vfs.f_frsize;
        if (freeb && freeb < need) {
            fprintf(stderr, "qma: not enough free space for aligned model "
                    "(need %.1f GiB, have %.1f GiB) — using %s\n",
                    (double)need / 1e9, (double)freeb / 1e9, src);
            snprintf(use_path, use_len, "%s", src);
            gguf_scan_free(&s);
            return 0;
        }
    }

    /* detailed boot log: what we're doing and why */
    fprintf(stderr, "qma: model %s is NOT 4K-aligned — repacking (one-time)\n", src);
    fprintf(stderr, "qma:   file %.2f GiB · %u tensors · alignment %zu · free %.1f GiB\n",
            (double)total / 1e9, s.n_tensors, s.alignment,
            freeb ? (double)freeb / 1e9 : 0.0);
    fprintf(stderr, "qma:   writing %s\n", cand);

    char tmp[4100];
    snprintf(tmp, sizeof(tmp), "%s.tmp", cand);
    unlink(tmp);
    int out = open(tmp, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (out < 0) {
        snprintf(err, errlen, "cannot create %s: %s", tmp, strerror(errno));
        gguf_scan_free(&s);
        return -1;
    }
    int in = open(src, O_RDONLY);
    if (in < 0) { close(out); unlink(tmp); snprintf(err, errlen, "cannot open %s", src); gguf_scan_free(&s); return -1; }

    uint8_t *buf = malloc(1u << 26);
    if (!buf) { close(in); close(out); unlink(tmp); snprintf(err, errlen, "oom"); gguf_scan_free(&s); return -1; }
    int rc = -1;

    /* 1) [0, meta_end) verbatim */
    {
        size_t left = s.meta_end, off = 0;
        while (left > 0) {
            size_t c = left < (1u << 26) ? left : (1u << 26);
            if (pread(in, buf, c, (off_t)off) != (ssize_t)c) goto fail;
            if (write_all(out, buf, c) != 0) goto fail;
            off += c; left -= c;
        }
    }
    /* 2) general.alignment missing? append the KV (type UINT32) + bump n_kv */
    if (s.align_pos < 0) {
        uint8_t kv[8 + 17 + 4 + 4];
        size_t o = 0;
        uint64_t kl = 17;   /* "general.alignment" is 17 chars — a 16-byte copy
                               truncates to "general.alignmen", which qma_load's
                               strcmp misses and silently keeps alignment=32 */
        memcpy(kv + o, &kl, 8); o += 8;
        memcpy(kv + o, "general.alignment", 17); o += 17;
        uint32_t t = 4, v = 4096;
        memcpy(kv + o, &t, 4); o += 4;
        memcpy(kv + o, &v, 4); o += 4;
        if (write_all(out, kv, o) != 0) goto fail;
        /* n_kv lives at file offset 16 (magic 4 + version 4 + n_tensors 8) */
        uint64_t nk = 0;
        if (pread(in, &nk, 8, 16) != 8) goto fail;
        nk += 1;
        if (pwrite(out, &nk, 8, 16) != 8) goto fail;
    }
    /* 3) copy the info table verbatim — IMMEDIATELY after metadata, matching
       the engine's reader (no padding before infos) */
    off_t new_infos_start = 0;
    {
        new_infos_start = lseek(out, 0, SEEK_CUR);
        size_t left = s.infos_end - infos_start_old, off = infos_start_old;
        while (left > 0) {
            size_t c = left < (1u << 26) ? left : (1u << 26);
            if (pread(in, buf, c, (off_t)off) != (ssize_t)c) goto fail;
            if (write_all(out, buf, c) != 0) goto fail;
            off += c; left -= c;
        }
    }
    /* 4) tensor data: pad every tensor start to 4096, copy bytes, patch offsets */
    uint64_t pad_written = 0;
    {
        off_t pos = lseek(out, 0, SEEK_CUR);
        size_t pad = (size_t)((4096 - ((size_t)pos % 4096)) % 4096);
        if (pad) { memset(buf, 0, pad); if (write_all(out, buf, pad) != 0) goto fail; pad_written += pad; }
        const off_t new_ds = lseek(out, 0, SEEK_CUR);
        uint64_t copied = 0, last_report = 0;
        for (uint32_t i = 0; i < s.n_tensors; i++) {
            off_t cur = lseek(out, 0, SEEK_CUR);
            size_t p2 = (size_t)((4096 - ((size_t)cur % 4096)) % 4096);
            if (p2) { memset(buf, 0, p2); if (write_all(out, buf, p2) != 0) goto fail; pad_written += p2; }
            off_t tstart = lseek(out, 0, SEEK_CUR);
            size_t left = s.nbytes[i], off = old_ds + s.off[i];
            while (left > 0) {
                size_t c = left < (1u << 26) ? left : (1u << 26);
                if (pread(in, buf, c, (off_t)off) != (ssize_t)c) goto fail;
                if (write_all(out, buf, c) != 0) goto fail;
                off += c; left -= c;
            }
            uint64_t nrel = (uint64_t)(tstart - new_ds);
            off_t fpos = new_infos_start + (s.off_pos[i] - (off_t)infos_start_old);
            if (pwrite(out, &nrel, 8, fpos) != 8) goto fail;
            copied += s.nbytes[i];
            if (copied - last_report >= (2ull << 30) || i + 1 == s.n_tensors) {
                const double dt = align_now_s() - t0;
                fprintf(stderr, "qma:   align %5.1f%%  %5.1f/%5.1f GiB  (%4.0f MiB/s)\n",
                        (double)copied / (double)total * 100.0,
                        (double)copied / 1e9, (double)total / 1e9,
                        dt > 0.1 ? (double)copied / 1048576.0 / dt : 0.0);
                last_report = copied;
            }
        }
    }
    /* 5) patch general.alignment to 4096 */
    if (s.align_pos >= 0) {
        if (s.align_width == 4) {
            uint32_t v = 4096;
            if (pwrite(out, &v, 4, s.align_pos) != 4) goto fail;
        } else {
            uint64_t v = 4096;
            if (pwrite(out, &v, 8, s.align_pos) != 8) goto fail;
        }
    }
    if (fsync(out) != 0) goto fail;
    close(out);
    if (rename(tmp, cand) != 0) {
        close(in);
        snprintf(err, errlen, "rename: %s", strerror(errno));
        free(buf); gguf_scan_free(&s);
        return -1;
    }
    /* 6) verify the result: alignment + tensor-byte spot-check
       (source fd `in` is still open — closing it earlier made the check
       read from a closed fd and falsely reject every repack) */
    {
        gguf_scan_t v;
        if (gguf_scan(cand, &v, ebuf, sizeof(ebuf)) != 0 || !gguf_all_aligned(&v, NULL)) {
            snprintf(err, errlen, "aligned model failed verification — keeping %s", src);
            unlink(cand);
            gguf_scan_free(&v);
            close(in);
            free(buf); gguf_scan_free(&s);
            return -1;
        }
        /* compare the head of tensor 0 and the last tensor, byte-for-byte,
           between source and repacked file (catches wrong-source copies) */
        int bad = 0;
        int vfd = open(cand, O_RDONLY);
        if (vfd < 0) { bad = 1; }
        else {
            const size_t vds = (v.infos_end + v.alignment - 1) / v.alignment * v.alignment;
            const uint32_t picks[2] = { 0, s.n_tensors - 1 };
            for (int k = 0; k < 2 && !bad; k++) {
                const uint32_t i = picks[k];
                const size_t nb = s.nbytes[i];
                const size_t chk = nb < (1u << 20) ? nb : (1u << 20);
                if (nb == 0) continue;
                if (pread(in, buf, chk, (off_t)(old_ds + s.off[i])) != (ssize_t)chk) { bad = 1; break; }
                uint8_t *b2 = malloc(chk);
                if (!b2) { bad = 1; break; }
                if (pread(vfd, b2, chk, (off_t)(vds + v.off[i])) != (ssize_t)chk) { free(b2); bad = 1; break; }
                if (memcmp(buf, b2, chk) != 0) { free(b2); bad = 1; break; }
                free(b2);
            }
            close(vfd);
        }
        if (bad) {
            snprintf(err, errlen, "aligned model data mismatch — keeping %s", src);
            unlink(cand);
            gguf_scan_free(&v);
            close(in);
            free(buf); gguf_scan_free(&s);
            return -1;
        }
        gguf_scan_free(&v);
    }
    close(in);
    snprintf(use_path, use_len, "%s", cand);
    {
        const double dt = align_now_s() - t0;
        fprintf(stderr, "qma: model aligned: %s\n", cand);
        fprintf(stderr, "qma:   %.2f GiB written in %.1fs (%.0f MiB/s) · +%.1f MB padding\n",
                (double)total / 1e9, dt,
                dt > 0.1 ? (double)total / 1048576.0 / dt : 0.0,
                (double)pad_written / 1e6);
        fprintf(stderr, "qma:   next launches will load %s and use O_DIRECT expert reads\n", cand);
    }
    rc = 0;
fail:
    if (rc != 0) { close(out); close(in); unlink(tmp); }
    free(buf);
    gguf_scan_free(&s);
    return rc;
}
