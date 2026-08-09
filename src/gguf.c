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
#include <fcntl.h>
#include <unistd.h>


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

    return 0;
}

void qma_free(qma_t *m) {
    if (m->fd >= 0) {
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
