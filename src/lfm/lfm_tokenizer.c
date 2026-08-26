/* tokenizer.c — LFM2.5 (lfm2) tokenizer: byte-level BPE, llama3 rules.
 *
 * The GGUF's tokenizer.ggml.pre = "lfm2" maps to llama.cpp's
 * LLAMA_VOCAB_PRE_TYPE_LLAMA3: byte-encoded BPE with the llama3
 * pre-tokenization regex and BOS added by the caller. We use the model's
 * own merge table from the file (identical to the HF tokenizer).
 *
 * Differences vs the qwen35 tokenizer this was ported from:
 *   - \p{N}{1,3}  (digit runs capped at 3) instead of single digits
 *   - pure \p{L} letter class (no \p{M} combining marks)
 *   - symbol class without \p{M}
 *   - detokenize keeps user-defined/unused tokens (think tags, tool
 *     sentinels, im markers) so the agent's stream scanners can see them;
 *     only control tokens are skipped.
 */
#include "lfm.h"
#include "unicode_tables.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------- byte <-> unicode map (GPT2) ---------------- */
static uint16_t byte_to_cpt[256];   /* byte -> unicode codepoint */
static uint8_t  cpt_to_byte[512];   /* codepoint -> byte (for the mapped range) */
static int      byte_map_ready = 0;

static void byte_map_init(void) {
    if (byte_map_ready) return;
    byte_map_ready = 1;
    int self[256];
    for (int b = 0; b < 256; b++) {
        self[b] = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
    }
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (self[b]) byte_to_cpt[b] = (uint16_t)b;
        else byte_to_cpt[b] = (uint16_t)(256 + n++);
    }
    memset(cpt_to_byte, 0xFF, sizeof(cpt_to_byte));
    for (int b = 0; b < 256; b++) cpt_to_byte[byte_to_cpt[b]] = (uint8_t)b;
}

/* utf8 helpers (see unicode_tables.h for the range tables) */
static uint32_t utf8_to_cpt(const char *s, int *len) {
    uint8_t c = (uint8_t)s[0];
    if (c < 0x80) { *len = 1; return c; }
    if ((c & 0xE0) == 0xC0) { *len = 2; return ((uint32_t)(c & 0x1F) << 6) | ((uint8_t)s[1] & 0x3F); }
    if ((c & 0xF0) == 0xE0) { *len = 3; return ((uint32_t)(c & 0x0F) << 12) | (((uint8_t)s[1] & 0x3F) << 6) | ((uint8_t)s[2] & 0x3F); }
    if ((c & 0xF8) == 0xF0) { *len = 4; return ((uint32_t)(c & 0x07) << 18) | (((uint8_t)s[1] & 0x3F) << 12) | (((uint8_t)s[2] & 0x3F) << 6) | ((uint8_t)s[3] & 0x3F); }
    *len = 1; return c;
}

static void cpt_to_utf8(uint32_t c, char *out) {
    if (c < 0x80) { out[0] = (char)c; out[1] = 0; }
    else if (c < 0x800) { out[0] = (char)(0xC0 | (c >> 6)); out[1] = (char)(0x80 | (c & 0x3F)); out[2] = 0; }
    else if (c < 0x10000) { out[0] = (char)(0xE0 | (c >> 12)); out[1] = (char)(0x80 | ((c >> 6) & 0x3F)); out[2] = (char)(0x80 | (c & 0x3F)); out[3] = 0; }
    else { out[0] = (char)(0xF0 | (c >> 18)); out[1] = (char)(0x80 | ((c >> 12) & 0x3F)); out[2] = (char)(0x80 | ((c >> 6) & 0x3F)); out[3] = (char)(0x80 | (c & 0x3F)); out[4] = 0; }
}

/* flags bits (llama): 1=undefined 2=number 4=letter 8=separator 0x10=accent
   0x20=punct 0x40=symbol 0x80=control 0x100=whitespace */
static uint16_t cpt_flags(uint32_t cpt) {
    int lo = 0, hi = (int)U_FLAGS_COUNT - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (U_FLAGS[mid].start <= cpt) {
            uint32_t last = (mid + 1 < (int)U_FLAGS_COUNT) ? U_FLAGS[mid + 1].start - 1 : 0x10FFFF;
            if (cpt <= last) {
                uint16_t f = U_FLAGS[mid].flags;
                int lo2 = 0, hi2 = (int)U_WHITESPACE_COUNT - 1;
                while (lo2 <= hi2) {
                    int mid2 = (lo2 + hi2) >> 1;
                    if (U_WHITESPACE[mid2] == cpt) { f |= 0x100; break; }
                    if (U_WHITESPACE[mid2] < cpt) lo2 = mid2 + 1; else hi2 = mid2 - 1;
                }
                return f;
            }
            lo = mid + 1;
        } else hi = mid - 1;
    }
    return 0;
}

static uint32_t cpt_tolower(uint32_t cpt) {
    int lo = 0, hi = (int)U_LOWER_COUNT - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (U_LOWER[mid].cpt == cpt) return U_LOWER[mid].lower;
        if (U_LOWER[mid].cpt < cpt) lo = mid + 1; else hi = mid - 1;
    }
    return cpt;
}

static int utf8_cpt_len(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

/* byte-encode a UTF-8 string into a buffer; returns encoded length.
   Each raw BYTE of the input maps through the GPT2 byte->unicode table. */
static int byte_encode(const char *in, int inlen, char *out) {
    byte_map_init();
    int o = 0;
    for (int i = 0; i < inlen; i++) {
        char tmp[8];
        cpt_to_utf8(byte_to_cpt[(uint8_t)in[i]], tmp);
        int tl = (int)strlen(tmp);
        memcpy(out + o, tmp, tl);
        o += tl;
    }
    return o;
}

/* ---------------- llama3 pre-tokenizer split ----------------
 * regex:
 *   (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
 *   | [^\r\n\p{L}\p{N}]?\p{L}+
 *   | \p{N}{1,3}
 *   | ?[^\s\p{L}\p{N}]+[\r\n]*
 *   | \s*[\r\n]+
 *   | \s+(?!\S)
 *   | \s+
 */
static int split_llama3(const uint32_t *cpts, int n, int *words, int max_words) {
    int nw = 0;
    size_t pos = 0;
    while (pos < (size_t)n) {
        uint32_t cpt = cpts[pos];
        uint16_t flags = cpt_flags(cpt);
        int wstart = (int)pos;

        /* contractions: 's 't 're 've 'm 'll 'd (case-insensitive) */
        if (cpt == '\'' && pos + 1 < (size_t)n) {
            uint32_t nx = cpt_tolower(cpts[pos + 1]);
            if (nx == 's' || nx == 't' || nx == 'm' || nx == 'd') {
                pos += 2; goto emit;
            }
            if (pos + 2 < (size_t)n) {
                uint32_t nn = cpt_tolower(cpts[pos + 2]);
                if ((nx == 'r' && nn == 'e') || (nx == 'v' && nn == 'e') || (nx == 'l' && nn == 'l')) {
                    pos += 3; goto emit;
                }
            }
        }

        /* [^\r\n\p{L}\p{N}]?\p{L}+  — optional non-letter/non-digit prefix,
           then one or more letters. Mark (\p{M}) is NOT a letter here. */
        if (!(cpt == '\r' || cpt == '\n' || (flags & 2))) {
            uint16_t f1 = (pos + 1 < (size_t)n) ? cpt_flags(cpts[pos + 1]) : 0;
            if ((flags & 4) || (f1 & 4)) {
                if (!(flags & 4)) pos++;               /* optional prefix */
                while (pos < (size_t)n && (cpt_flags(cpts[pos]) & 4)) pos++;
                goto emit;
            }
        }

        /* \p{N}{1,3} — digit runs capped at 3 */
        if (flags & 2) {
            int nd = 0;
            while (pos < (size_t)n && (cpt_flags(cpts[pos]) & 2) && nd < 3) { pos++; nd++; }
            goto emit;
        }

        /* ?[^\s\p{L}\p{N}]+[\r\n]* — symbols */
        {
            uint16_t f2 = (cpt == ' ' && pos + 1 < (size_t)n) ? cpt_flags(cpts[pos + 1]) : flags;
            if (!(f2 & (0x100 | 4 | 2)) && flags) {
                pos += (cpt == ' ');
                while (pos < (size_t)n && !(cpt_flags(cpts[pos]) & (0x100 | 4 | 2))) pos++;
                while (pos < (size_t)n && (cpts[pos] == '\r' || cpts[pos] == '\n')) pos++;
                goto emit;
            }
        }

        /* whitespace run analysis */
        {
            size_t nws = 0;
            size_t last_rn = 0;
            while (pos + nws < (size_t)n && (cpt_flags(cpts[pos + nws]) & 0x100)) {
                uint32_t c2 = cpts[pos + nws];
                if (c2 == '\r' || c2 == '\n') last_rn = pos + nws + 1;
                nws++;
            }

            /* \s*[\r\n]+ */
            if (last_rn > 0) { pos = last_rn; goto emit; }

            /* \s+(?!\S) */
            if (nws > 1 && pos + nws < (size_t)n) { pos += nws - 1; goto emit; }

            /* \s+ */
            if (nws > 0) { pos += nws; goto emit; }
        }

        /* no match: single char */
        pos++;
    emit:
        if (nw < max_words) words[nw++] = (int)pos - wstart;
    }
    return nw;
}

/* ---------------- BPE heap ---------------- */
typedef struct { int left, right; uint32_t rank; uint32_t li, ri; } bigram_t;

typedef struct {
    bigram_t *a;
    int n, cap;
} heap_t;

static void heap_push(heap_t *h, bigram_t b) {
    if (h->n == h->cap) { h->cap = h->cap ? h->cap * 2 : 64; h->a = realloc(h->a, h->cap * sizeof(bigram_t)); }
    int i = h->n++;
    while (i > 0) {
        int p = (i - 1) / 2;
        bigram_t *par = &h->a[p];
        if (par->rank < b.rank || (par->rank == b.rank && par->left <= b.left)) break;
        h->a[i] = *par; i = p;
    }
    h->a[i] = b;
}

static bigram_t heap_pop(heap_t *h) {
    bigram_t top = h->a[0];
    bigram_t last = h->a[--h->n];
    int i = 0;
    while (1) {
        int l = 2 * i + 1, r = l + 1, m = -1;
        if (l < h->n) {
            m = l;
            if (r < h->n && (h->a[r].rank < h->a[l].rank ||
                (h->a[r].rank == h->a[l].rank && h->a[r].left < h->a[l].left))) m = r;
        }
        if (m < 0 || (last.rank < h->a[m].rank || (last.rank == h->a[m].rank && last.left <= h->a[m].left))) break;
        h->a[i] = h->a[m]; i = m;
    }
    h->a[i] = last;
    return top;
}

/* ---------------- BPE symbol table ---------------- */
typedef struct {
    int start, len;   /* byte range into word buffer */
    int prev, next;
    uint32_t id;      /* token id of symbol string (UINT32_MAX = not in vocab) */
    int alive;
} bpe_sym_t;

typedef struct {
    bpe_sym_t *syms;
    int cap, n;
    heap_t heap;
    char *word;
    int wlen;
} bpe_ctx_t;

static uint32_t tok_lookup(lfm_t *m, const char *s, size_t len) {
    uint64_t h = lfm_hash64(s, len);
    uint32_t i = (uint32_t)(h & (m->tok_cap - 1));
    while (m->tok_ids[i] != UINT32_MAX) {
        if (m->tok_keys[i] == h) {
            const char *t = m->tok_text[m->tok_ids[i]];
            if (strlen(t) == len && memcmp(t, s, len) == 0) return m->tok_ids[i];
        }
        i = (i + 1) & (m->tok_cap - 1);
    }
    return UINT32_MAX;
}

static uint32_t merge_rank(lfm_t *m, uint32_t l, uint32_t r) {
    uint8_t pair[8];
    memcpy(pair, &l, 4); memcpy(pair + 4, &r, 4);
    uint64_t h = lfm_hash64(pair, 8);
    uint32_t i = (uint32_t)(h & (m->mg_cap - 1));
    while (m->mg_rank[i] != UINT32_MAX) {
        if (m->mg_keys[i] == h) return m->mg_rank[i];
        i = (i + 1) & (m->mg_cap - 1);
    }
    return UINT32_MAX;
}

static void bpe_add_bigram(lfm_t *m, bpe_ctx_t *b, int left, int right) {
    if (left < 0 || right < 0) return;
    bpe_sym_t *L = &b->syms[left], *R = &b->syms[right];
    if (!L->alive || !R->alive) return;
    if (L->next != right) return;
    if (L->id == UINT32_MAX || R->id == UINT32_MAX) return;
    uint32_t rank = merge_rank(m, L->id, R->id);
    if (rank == UINT32_MAX) return;
    bigram_t bg = { left, right, rank, L->id, R->id };
    heap_push(&b->heap, bg);
}

static void bpe_word(lfm_t *m, bpe_ctx_t *b, const char *word, int wlen, int *out, int *nout, int max_out) {
    b->n = 0;
    b->heap.n = 0;
    b->word = (char *)word;
    b->wlen = wlen;

    int nchars = 0;
    for (int off = 0; off < wlen; ) { off += utf8_cpt_len((uint8_t)word[off]); nchars++; }
    if (nchars + 1 > b->cap) {
        b->cap = nchars + 8;
        b->syms = realloc(b->syms, b->cap * sizeof(bpe_sym_t));
    }

    int off = 0, idx = 0;
    while (off < wlen) {
        int cl = utf8_cpt_len((uint8_t)word[off]);
        bpe_sym_t *s = &b->syms[idx];
        s->start = off; s->len = cl;
        s->prev = idx - 1;
        s->next = (off + cl >= wlen) ? -1 : idx + 1;
        s->id = tok_lookup(m, word + off, cl);
        s->alive = 1;
        idx++;
        off += cl;
    }
    b->n = idx;

    for (int i = 1; i < b->n; i++) bpe_add_bigram(m, b, i - 1, i);

    while (b->heap.n > 0) {
        bigram_t bg = heap_pop(&b->heap);
        bpe_sym_t *L = &b->syms[bg.left], *R = &b->syms[bg.right];
        if (!L->alive || !R->alive) continue;
        if (L->id != bg.li || R->id != bg.ri) continue;
        if (L->next != bg.right) continue;
        L->len += R->len;
        R->alive = 0;
        L->next = R->next;
        if (R->next >= 0) b->syms[R->next].prev = bg.left;
        L->id = tok_lookup(m, word + L->start, L->len);
        bpe_add_bigram(m, b, L->prev, bg.left);
        bpe_add_bigram(m, b, bg.left, L->next);
    }

    for (int i = 0; i < b->n; i++) {
        bpe_sym_t *s = &b->syms[i];
        if (!s->alive || s->len == 0) continue;
        if (s->id != UINT32_MAX) {
            if (*nout < max_out) out[(*nout)++] = (int)s->id;
        } else {
            for (int j = s->start; j < s->start + s->len; ) {
                int cl = utf8_cpt_len((uint8_t)word[j]);
                uint32_t id = tok_lookup(m, word + j, cl);
                if (id != UINT32_MAX && *nout < max_out) out[(*nout)++] = (int)id;
                j += cl;
            }
        }
    }
}

/* ---------------- tokenize with special partition ---------------- */
typedef struct {
    int kind;          /* 0 = text, 1 = special id */
    int64_t start, len;
    uint32_t id;
} frag_t;

#define MAX_FRAGS 4096

int lfm_tokenize(lfm_t *m, const char *text, int *out, int max_out) {
    size_t tlen = strlen(text);

    frag_t frags[MAX_FRAGS];
    int nfrag = 1;
    frags[0].kind = 0; frags[0].start = 0; frags[0].len = (int64_t)tlen; frags[0].id = 0;

    for (int si = 0; si < m->n_special; si++) {
        uint32_t sid = m->special_ids[si];
        const char *st = m->tok_text[sid];
        size_t sl = strlen(st);
        if (sl == 0) continue;
        int i = 0;
        while (i < nfrag) {
            if (frags[i].kind != 0) { i++; continue; }
            const char *base = text + frags[i].start;
            int64_t flen = frags[i].len;
            int64_t fpos = 0;
            int changed = 0;
            while (fpos + (int64_t)sl <= flen) {
                if (memcmp(base + fpos, st, sl) == 0) {
                    if (nfrag + 2 >= MAX_FRAGS) return -2;
                    int64_t left_len = fpos, right_start = fpos + (int64_t)sl;
                    for (int k = nfrag - 1; k > i; k--) frags[k + 2] = frags[k];
                    frags[i].len = left_len;
                    frags[i + 1].kind = 1; frags[i + 1].id = sid; frags[i + 1].len = 0;
                    frags[i + 2].kind = 0;
                    frags[i + 2].start = frags[i].start + right_start;
                    frags[i + 2].len = flen - right_start;
                    nfrag += 2;
                    frags[i + 1].start = frags[i].start + fpos;
                    i += 2;
                    changed = 1;
                    break;
                }
                fpos++;
            }
            if (!changed) i++;
        }
    }

    int nout = 0;
    uint32_t *cpts = malloc(sizeof(uint32_t) * (tlen + 1));
    int *words = malloc(sizeof(int) * (tlen + 1));
    char *enc = malloc(tlen * 4 + 16);
    if (!cpts || !words || !enc) { free(cpts); free(words); free(enc); return -1; }
    bpe_ctx_t bctx; memset(&bctx, 0, sizeof(bctx));

    for (int i = 0; i < nfrag; i++) {
        if (frags[i].kind == 1) {
            if (nout < max_out) out[nout++] = (int)frags[i].id;
            continue;
        }
        if (frags[i].len == 0) continue;
        const char *ftext = text + frags[i].start;
        int64_t flen = frags[i].len;
        int n = 0;
        for (int64_t p = 0; p < flen; ) {
            int l; cpts[n] = utf8_to_cpt(ftext + p, &l);
            n++; p += l;
        }
        int nw = split_llama3(cpts, n, words, (int)tlen + 1);
        int wstart = 0;
        for (int w = 0; w < nw; w++) {
            int wlen_cpt = words[w];
            char wutf8[4096];
            int wu = 0;
            for (int k = 0; k < wlen_cpt; k++) {
                char tmp[8]; cpt_to_utf8(cpts[wstart + k], tmp);
                int tl = (int)strlen(tmp);
                if (wu + tl >= (int)sizeof(wutf8)) break;
                memcpy(wutf8 + wu, tmp, tl); wu += tl;
            }
            wstart += wlen_cpt;
            int elen = byte_encode(wutf8, wu, enc);
            if (elen > 0) bpe_word(m, &bctx, enc, elen, out, &nout, max_out);
        }
    }
    free(cpts); free(words); free(enc);
    free(bctx.heap.a);
    free(bctx.syms);
    return nout;
}

/* ---------------- detokenize ----------------
 * Control tokens (pad/startoftext/endoftext/fim) are skipped. User-defined
 * and unused tokens — <|im_start|>, <|im_end|>, <think>, </think>,
 * <|tool_call_start|>, <|tool_call_end|> — are KEPT so the agent's stream
 * scanners (think/tool detection) can see the markers. */
int lfm_detokenize(lfm_t *m, const int *tokens, int n, char *out, int max_out) {
    byte_map_init();
    int o = 0;
    for (int i = 0; i < n && o < max_out - 1; i++) {
        int id = tokens[i];
        if (id < 0 || id >= m->n_vocab || !m->tok_text[id]) continue;
        const char *t = m->tok_text[id];
        if (m->tok_type[id] & 2) continue;   /* control only */
        for (const char *p = t; *p && o < max_out - 1; ) {
            int l; uint32_t c = utf8_to_cpt(p, &l);
            uint8_t b = (c < 512) ? cpt_to_byte[c] : 0xFF;
            if (b == 0xFF) { p += l; continue; }
            out[o++] = (char)b;
            p += l;
        }
    }
    out[o] = 0;
    return o;
}
