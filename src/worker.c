/* worker.c — LFM2.5 document-worker pool for qma.
 * The main qwen35 agent spawns workers: small LFM2.5 instances (weights
 * loaded ONCE and shared) that hold a document in their KV and answer
 * questions about it. Each worker is a KV runstate with a base (worker
 * system prompt + document, processed once) and ephemeral query layers on
 * top — a clean KV per query, or a persistent conversation in long-term
 * mode. Only KV is per-worker; the weights are shared.
 *
 * The worker source (src/lfm/) lives in qma's internal tree, so the
 * self-hosting agent can read and modify its own sub-agents.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "worker.h"
#include "lfm.h"

#define WORKER_MAX    4          /* simultaneous workers (RAM guard) */
#define WORKER_CTX    128000       /* per-worker context (tokens) */
#define WORKER_DOC_MAX 262144    /* document text cap */
#define IDS_CAP       131072

static const char *WORKER_DEFAULT_ROLE =
    "You are a document analyst. You have been given a document and you "
    "answer questions about it. You have NO tools and cannot take actions, "
    "read files, or call functions — answer directly from the document "
    "text in this conversation only. Answer precisely and concisely, "
    "quoting exact text and line numbers when relevant. If the answer is "
    "not in the document, say so directly — do not guess or invent.";

typedef struct {
    int active;
    char name[64];
    char role[2048];
    lfm_rs_t rs;
    int base_n_pos;          /* position after role+doc (the base) */
    int n_queries;           /* queries served */
    float *base_conv;        /* conv_state checkpoint at base (ephemeral reset) */
    int *ids;
    float *logits;
} worker_t;

static worker_t g_workers[WORKER_MAX];
static lfm_t g_wm;
static int g_wm_loaded = 0;
static int g_wm_threads = 4;

/* ---- shared model ---- */
int worker_model_load(const char *path) {
    if (g_wm_loaded) return 0;
    char err[512] = "";
    memset(&g_wm, 0, sizeof(g_wm));
    if (lfm_load(&g_wm, path, 0, err, sizeof(err)) != 0) {
        fprintf(stderr, "[worker] LFM load failed: %s\n", err);
        return -1;
    }
    g_wm_threads = 4;
    if (getenv("QMA_WORKER_THREADS")) {
        int t = atoi(getenv("QMA_WORKER_THREADS"));
        if (t > 0 && t <= 8) g_wm_threads = t;
    }
    g_wm_loaded = 1;
    fprintf(stderr, "[worker] LFM2.5 worker model resident (%s)\n", path);
    return 0;
}

/* ---- worker lifecycle ---- */
static int worker_find(int id) {
    if (id < 0 || id >= WORKER_MAX || !g_workers[id].active) return -1;
    return id;
}

int worker_count(void) {
    int n = 0;
    for (int i = 0; i < WORKER_MAX; i++) if (g_workers[i].active) n++;
    return n;
}

static void worker_reset_base(worker_t *w) {
    w->rs.n_pos = w->base_n_pos;
    /* restore the recurrent conv state checkpoint (overwrites the query
       effects; the KV tail [base, n_pos) gets overwritten naturally on the
       next append) */
    if (w->base_conv) {
        for (int il = 0; il < N_LAYER; il++)
            if (w->rs.conv_state[il])
                memcpy(w->rs.conv_state[il],
                       w->base_conv + (size_t)il * 2 * 2048,
                       (size_t)2 * 2048 * sizeof(float));
    }
}

int worker_spawn(const char *name, const char *sysprompt, const char *doc) {
    if (!g_wm_loaded) return -1;
    if (!doc || !doc[0]) return -1;
    int slot = -1;
    for (int i = 0; i < WORKER_MAX; i++)
        if (!g_workers[i].active) { slot = i; break; }
    if (slot < 0) return -1;             /* pool full */

    worker_t *w = &g_workers[slot];
    memset(w, 0, sizeof(*w));
    if (name && name[0]) snprintf(w->name, sizeof(w->name), "%s", name);
    else snprintf(w->name, sizeof(w->name), "worker-%d", slot);
    if (sysprompt && sysprompt[0])
        snprintf(w->role, sizeof(w->role), "%s", sysprompt);
    else
        snprintf(w->role, sizeof(w->role), "%s", WORKER_DEFAULT_ROLE);

    if (lfm_rs_init_kv(&w->rs, WORKER_CTX, NULL, 0) != 0) return -1;
    w->ids = malloc(sizeof(int) * IDS_CAP);
    w->logits = malloc(sizeof(float) * g_wm.n_vocab);
    if (!w->ids || !w->logits) { lfm_rs_free(&w->rs); free(w->ids); free(w->logits); return -1; }

    /* base: role + document (processed once) */
    char base[WORKER_DOC_MAX + 4096];
    int bl = snprintf(base, sizeof(base),
        "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n",
        w->role, doc);
    if (bl >= (int)sizeof(base)) bl = (int)sizeof(base) - 1;
    int n = lfm_tokenize(&g_wm, base, w->ids, IDS_CAP);
    if (n <= 0 || n > WORKER_CTX - 1024) {
        lfm_rs_free(&w->rs); free(w->ids); free(w->logits);
        memset(w, 0, sizeof(*w));
        return -2;                         /* doc too big for the context */
    }
    if (lfm_eval(&g_wm, &w->rs, w->ids, n, NULL, g_wm_threads, 0, 0) != 0) {
        lfm_rs_free(&w->rs); free(w->ids); free(w->logits);
        memset(w, 0, sizeof(*w));
        return -1;
    }
    w->base_n_pos = w->rs.n_pos;
    /* checkpoint conv state for ephemeral resets */
    w->base_conv = malloc((size_t)N_LAYER * 2 * 2048 * sizeof(float));
    if (w->base_conv) {
        for (int il = 0; il < N_LAYER; il++)
            if (w->rs.conv_state[il])
                memcpy(w->base_conv + (size_t)il * 2 * 2048,
                       w->rs.conv_state[il], (size_t)2 * 2048 * sizeof(float));
    }
    w->active = 1;
    fprintf(stderr, "[worker] %s spawned (id=%d, base=%d tokens)\n",
            w->name, slot, w->rs.n_pos);
    return slot;
}

char *worker_ask(int id, const char *question, int ephemeral, int max_tokens) {
    worker_t *w = worker_find(id) < 0 ? NULL : &g_workers[id];
    if (!w) return strdup("ERROR: no such worker (worker_list to see ids)");
    if (!question || !question[0]) return strdup("ERROR: empty question");

    char prompt[65536];
    int pl = snprintf(prompt, sizeof(prompt),
        "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", question);
    if (pl >= (int)sizeof(prompt)) pl = (int)sizeof(prompt) - 1;
    int n = lfm_tokenize(&g_wm, prompt, w->ids, IDS_CAP);
    if (n <= 0) return strdup("ERROR: could not tokenize question");
    if (w->rs.n_pos + n > w->rs.n_ctx - 64) {
        if (ephemeral) {
            /* try to reclaim space by resetting to base first */
            worker_reset_base(w);
            if (w->rs.n_pos + n > w->rs.n_ctx - 64)
                return strdup("ERROR: worker context full — the document alone fills it; close and respawn with a shorter document");
        } else {
            return strdup("ERROR: worker context full — close this worker or use ephemeral mode (ephemeral=true)");
        }
    }
    /* prefill the question WITH logits (the sampler needs the last-token
       row; passing NULL leaves stale logits that immediately sample im_end) */
    if (lfm_eval(&g_wm, &w->rs, w->ids, n, w->logits, g_wm_threads, 0, 0) != 0)
        return strdup("ERROR: worker eval failed");

    /* generation budget is a RUNAWAY GUARD only — the model ends the turn
       itself at <|im_end|>/eos. No artificial thinking/answer limit: a doc
       question that needs deep reasoning gets it. */
    if (max_tokens <= 0) max_tokens = 65536;
    else if (max_tokens > 65536) max_tokens = 65536;
    lfm_samp_t sp;
    lfm_samp_init(&sp, (uint64_t)time(NULL) ^ (uint64_t)id, 0.6f, 40, 0.9f, 1.15f);
    char *out = malloc(65536);
    size_t o = 0;
    out[0] = 0;
    for (int i = 0; i < max_tokens; i++) {
        int cand[40]; float lgs[40];
        int nc = lfm_samp_candidates(g_wm.n_vocab, w->logits, &sp, cand, lgs, 40);
        int tok = nc > 0 ? lfm_samp_pick(&sp, cand, lgs, nc, 0.9f) : 0;
        if (tok == (int)g_wm.id_im_end || tok == (int)g_wm.id_eos ||
            tok == (int)g_wm.id_bos) break;
        char piece[128];
        int pl2 = lfm_detokenize(&g_wm, &tok, 1, piece, sizeof(piece));
        if (pl2 > 0 && o + (size_t)pl2 < 65535) {
            memcpy(out + o, piece, (size_t)pl2);
            o += (size_t)pl2;
            out[o] = 0;
        }
        if (lfm_eval(&g_wm, &w->rs, &tok, 1, w->logits, g_wm_threads, 0, 0) != 0)
            break;
    }
    w->n_queries++;
    if (ephemeral) worker_reset_base(w);
    /* strip COMPLETE <think>...</think> reasoning blocks (presentation:
       the main agent wants the answer, not the draft) and any tool-call
       blocks the model tries despite having no tools. An unclosed block
       means the model was still reasoning when it stopped — keep it as-is
       rather than destroying real output. */
    {
        static const struct { const char *open, *close; } blocks[] = {
            { "<think>", "</think>" },
            { "<|tool_call_start|>", "<|tool_call_end|>" },
        };
        char *dst = out, *src = out;
        while (*src) {
            int hit = 0;
            for (int b = 0; b < 2; b++) {
                size_t ol = strlen(blocks[b].open);
                if (strncmp(src, blocks[b].open, ol) == 0) {
                    char *cl = strstr(src + ol, blocks[b].close);
                    if (cl) { src = cl + strlen(blocks[b].close); hit = 1; break; }
                }
            }
            if (hit) continue;
            *dst++ = *src++;
        }
        *dst = 0;
    }
    /* trim leading whitespace/newlines */
    while (out[0] == '\n' || out[0] == '\r' || out[0] == ' ' || out[0] == '\t')
        memmove(out, out + 1, strlen(out));
    return out;
}

int worker_close(int id) {
    worker_t *w = worker_find(id) < 0 ? NULL : &g_workers[id];
    if (!w) return -1;
    free(w->base_conv);
    free(w->ids);
    free(w->logits);
    lfm_rs_free(&w->rs);
    memset(w, 0, sizeof(*w));
    return 0;
}

char *worker_list(void) {
    char *out = malloc(8192);
    size_t o = 0;
    out[0] = 0;
    int any = 0;
    for (int i = 0; i < WORKER_MAX; i++) {
        if (!g_workers[i].active) continue;
        any = 1;
        int w = snprintf(out + o, 8192 - o, "%d\t%-20s base=%d tokens, %d queries served\n",
                         i, g_workers[i].name, g_workers[i].base_n_pos,
                         g_workers[i].n_queries);
        if (w < 0 || (size_t)w >= 8192 - o) break;
        o += (size_t)w;
    }
    if (!any) {
        snprintf(out, 8192, "(no workers — use worker_spawn with a document path)");
    }
    return out;
}

