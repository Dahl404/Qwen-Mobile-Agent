/* sampler.c - temperature / top-k / top-p / repeat-penalty sampling.
 * Split into two phases (llama.cpp sampler-chain shape):
 *   sampler_candidates() — repeat penalty + temp + top-k  -> candidate list
 *   sampler_pick()       — top-p + sample
 * There is no grammar filter anymore (see agent.c): tool-call validity is
 * enforced by policy (one complete call per turn, then the engine
 * force-ends the turn and folds the result back). */
#include "qma.h"

static uint64_t rng_next(uint64_t *s) {
    *s ^= *s >> 12;
    *s ^= *s << 25;
    *s ^= *s >> 27;
    return *s * 0x2545F4914F6CDD1DULL;
}

void sampler_init(sampler_t *s, uint64_t seed, float temp, int top_k, float top_p,
                  float repeat_penalty) {
    s->seed = seed ? seed : 0x9E3779B97F4A7C15ULL;
    s->temp = temp;
    s->top_k = top_k;
    s->top_p = top_p;
    s->repeat_penalty = repeat_penalty;
    s->n_last = 0;
}

typedef struct { float p; int id; } cand_t;

static int cand_cmp(const void *a, const void *b) {
    const cand_t *x = a, *y = b;
    if (x->p > y->p) return -1;
    if (x->p < y->p) return 1;
    return 0;
}

/* Phase 1: top-k candidates (repeat penalty + temp applied). Returns the
   count; ids[i]/lgs[i] are the token id and scaled logit, sorted desc. */
int sampler_candidates(int n_vocab, const float *logits, sampler_t *s,
                       int *ids, float *lgs, int max_k) {
    const int n = n_vocab;
    float *l = (float *)logits;
    float *buf = NULL;
    if (s->repeat_penalty != 1.0f || s->temp > 0.0f) {
        buf = malloc(sizeof(float) * (size_t)n);
        if (!buf) return 0;
        memcpy(buf, logits, sizeof(float) * (size_t)n);
        l = buf;
    }
    if (s->repeat_penalty != 1.0f) {
        for (int i = 0; i < s->n_last; i++) {
            int id = s->last_tokens[i];
            if (id >= 0 && id < n) {
                if (l[id] < 0.0f) l[id] *= s->repeat_penalty;
                else l[id] /= s->repeat_penalty;
            }
        }
    }
    if (s->temp > 0.0f) {
        for (int i = 0; i < n; i++) l[i] /= s->temp;
    }
    int k = s->top_k;
    if (k < 1) k = 1;
    if (k > n) k = n;
    if (k > max_k) k = max_k;
    /* top-k: min-heap of size k over (logit, id) */
    cand_t *c = malloc(sizeof(cand_t) * (size_t)n);
    if (!c) { free(buf); return 0; }
    for (int i = 0; i < k; i++) { c[i].p = l[i]; c[i].id = i; }
    for (int i = k / 2 - 1; i >= 0; i--) {
        int j = i;
        for (;;) {
            int lc = 2 * j + 1, rc = lc + 1, mn = j;
            if (lc < k && c[lc].p < c[mn].p) mn = lc;
            if (rc < k && c[rc].p < c[mn].p) mn = rc;
            if (mn == j) break;
            cand_t tmp = c[j]; c[j] = c[mn]; c[mn] = tmp; j = mn;
        }
    }
    for (int i = k; i < n; i++) {
        if (l[i] > c[0].p) {
            c[0].p = l[i]; c[0].id = i;
            int j = 0;
            for (;;) {
                int lc = 2 * j + 1, rc = lc + 1, mn = j;
                if (lc < k && c[lc].p < c[mn].p) mn = lc;
                if (rc < k && c[rc].p < c[mn].p) mn = rc;
                if (mn == j) break;
                cand_t tmp = c[j]; c[j] = c[mn]; c[mn] = tmp; j = mn;
            }
        }
    }
    qsort(c, (size_t)k, sizeof(cand_t), cand_cmp);
    for (int i = 0; i < k; i++) { ids[i] = c[i].id; lgs[i] = c[i].p; }
    free(c);
    free(buf);
    return k;
}

/* Phase 2: top-p + sample from the candidates. Masked candidates carry
   logit -1e30f (control tokens pre-masked upstream). If every candidate is
   masked, fall back to the top one (the parser backstop catches any bad
   output). */
int sampler_pick(sampler_t *s, const int *ids, float *lgs, int n, float top_p) {
    if (n <= 0) return 0;
    int first_live = -1;
    for (int i = 0; i < n; i++) if (lgs[i] > -1e29f) { first_live = i; break; }
    if (first_live < 0) return ids[0];          /* all masked: fall back */
    if (s->temp <= 0.0f) return ids[first_live];/* greedy: best live */
    /* softmax over live candidates */
    float mx = lgs[first_live];
    for (int i = first_live; i < n; i++) if (lgs[i] > -1e29f && lgs[i] > mx) mx = lgs[i];
    double z = 0.0;
    for (int i = first_live; i < n; i++)
        if (lgs[i] > -1e29f) { float p = expf(lgs[i] - mx); lgs[i] = p; z += p; }
    float inv = (float)(1.0 / z);
    for (int i = first_live; i < n; i++) if (lgs[i] > -1e29f) lgs[i] *= inv;
    float acc = 0.0f;
    int kept = first_live;
    for (int i = first_live; i < n; i++) {
        if (lgs[i] <= -1e29f) continue;
        acc += lgs[i];
        kept = i + 1;
        if (acc >= top_p) break;
    }
    uint64_t r = rng_next(&s->seed);
    float rv = (float)((r >> 11) * (1.0 / 9007199254740992.0));
    float cum = 0.0f;
    int pick = ids[kept - 1];
    for (int i = first_live; i < kept; i++) {
        if (lgs[i] <= -1e29f) continue;
        cum += lgs[i];
        if (rv < cum) { pick = ids[i]; break; }
    }
    if (s->n_last < 64) s->last_tokens[s->n_last++] = pick;
    else { memmove(s->last_tokens, s->last_tokens + 1, 63 * sizeof(int)); s->last_tokens[63] = pick; }
    return pick;
}
