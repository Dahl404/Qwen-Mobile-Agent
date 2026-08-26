/* mini.c — bare one-shot generation tester.
 *
 * NO agent, NO system prompt, NO tools, NO session dirs. Loads the model,
 * tokenizes a prompt, greedy-decodes N tokens, prints the text as it
 * comes + top-5 logits per step to stderr. Use it to sanity-check a model
 * file in seconds instead of booting the full agent.
 *
 * Usage: mini -m <model> [-p "<prompt>"] [-n <tokens>] [-t <threads>]
 *              [--pf <prefetch>] [--prewarm]
 *   default prompt: "The capital of France is"
 *   default: NOPREWARM (fast start, page-faults during eval) unless --prewarm
 *
 * Env: QMA_MODEL as fallback for -m.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "qma.h"

static const char *def_prompt = "The capital of France is";

int main(int argc, char **argv) {
    const char *model = NULL;
    const char *prompt = def_prompt;
    int n_gen = 24, threads = 8, prefetch = 4, prewarm = 0;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-m") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) n_gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--pf") && i + 1 < argc) prefetch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--prewarm")) prewarm = 1;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!model) model = getenv("QMA_MODEL");
    if (!model) { fprintf(stderr, "no model (-m or QMA_MODEL)\n"); return 2; }

    if (!prewarm) setenv("QMA_NOPREWARM", "1", 1);
    setenv("QMA_NO_CL", "1", 1);       /* deterministic CPU eval */

    static qma_t m;
    char err[256] = "";
    if (qma_load(&m, model, err, sizeof(err)) != 0) {
        fprintf(stderr, "load failed: %s\n", err); return 1;
    }
    /* tiered mode: QMA_AUX=<q2 path> loads the secondary source; runtime
       expert misses then fill from the Q2 slab (smaller/faster) */
    const char *auxp = getenv("QMA_AUX");
    if (auxp && auxp[0]) {
        static qma_aux_t aux;
        if (qma_load_aux(&aux, auxp, err, sizeof(err)) != 0) {
            fprintf(stderr, "aux load failed: %s\n", err); return 1;
        }
        m.aux = &aux;
        fprintf(stderr, "tiered mode: aux=%s\n", auxp);
    }
    size_t eb = getenv("QMA_ECMB") ? (size_t)atoi(getenv("QMA_ECMB")) << 20 : 1024ull << 20;
    qma_ecache_arm(&m, eb, 4);
    qma_prefetch_init(&m);

    runstate_t rs;
    int ctx = getenv("QMA_CTX") ? atoi(getenv("QMA_CTX")) : 8192;
    if (runstate_init(&rs, ctx) != 0) { fprintf(stderr, "rs init failed\n"); return 1; }

    int ids[4096];
    int n = qma_tokenize(&m, prompt, ids, 4096);
    if (n <= 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    fprintf(stderr, "prompt tokens(%d):", n);
    for (int i = 0; i < n && i < 24; i++)
        fprintf(stderr, " [%s]", m.tok_text[ids[i]]);
    fprintf(stderr, "\n");

    static float logits[N_VOCAB];
    if (qma_eval(&m, &rs, ids, n, logits, threads, prefetch, 0) != 0) {
        fprintf(stderr, "prefill failed\n"); return 1;
    }
    if (getenv("QMA_LOGDGB")) {
        float mn=1e30f, mx=-1e30f; double s=0; int nz=0;
        for (int v=0; v<m.n_vocab; v++) { if(logits[v]<mn)mn=logits[v]; if(logits[v]>mx)mx=logits[v]; s+=logits[v]; if(logits[v]!=0.0f)nz++; }
        fprintf(stderr, "LOGDGB: n_vocab=%d output=%p outnorm=%p logits[min=%.3g max=%.3g sum=%.3g nz=%d/%d]\n",
                m.n_vocab, (void*)m.output, (void*)m.output_norm, mn, mx, s, nz, m.n_vocab);
    }
    printf("== %s\n", prompt);
    for (int step = 0; step < n_gen; step++) {
        int best = 0; float bv = -1e30f;
        for (int v = 0; v < m.n_vocab; v++)
            if (logits[v] > bv) { bv = logits[v]; best = v; }
        int t5[5]; float v5[5];
        for (int k = 0; k < 5; k++) { t5[k] = -1; v5[k] = -1e30f; }
        for (int v = 0; v < m.n_vocab; v++) {
            float lv = logits[v];
            if (lv > v5[4]) {
                int p = 4;
                while (p > 0 && v5[p - 1] < lv) { v5[p] = v5[p - 1]; t5[p] = t5[p - 1]; p--; }
                v5[p] = lv; t5[p] = v;
            }
        }
        fprintf(stderr, "step %2d tok=%6d top5:", step, best);
        for (int k = 0; k < 5; k++)
            fprintf(stderr, " %s(%.2f)", m.tok_text[t5[k]], v5[k]);
        fprintf(stderr, "\n");
        const char *txt = m.tok_text[best];
        if (m.tok_is_special[best]) {
            fprintf(stderr, "[special token %s — stopping]\n", txt);
            break;
        }
        fputs(txt, stdout);
        fflush(stdout);
        if (step + 1 < n_gen) {
            if (qma_eval(&m, &rs, &best, 1, logits, threads, prefetch, 0) != 0) {
                fprintf(stderr, "decode failed at %d\n", step); return 1;
            }
        }
    }
    printf("\n");
    return 0;
}
