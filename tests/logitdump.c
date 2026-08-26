/* differential logit dump: fixed tokens -> qma_eval -> top logits per step.
 * CPU-only (QMA_NO_CL=1 set here). Build against REF and CURRENT trees and
 * diff the outputs to locate numerical divergence. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "qma.h"

int main(int argc, char **argv) {
    const char *model = argc > 1 ? argv[1]
        : "/data/data/com.termux/files/home/projects/models/qwen3.6:35b:a3b-q4km.gguf";
    setenv("QMA_NO_CL", "1", 1);
    setenv("QMA_NOEVICT", "1", 1);   /* dense attention, no hcm state */

    static qma_t m;
    char err[256] = "";
    if (qma_load(&m, model, err, sizeof(err)) != 0) {
        fprintf(stderr, "load failed: %s\n", err);
        return 1;
    }
    runstate_t rs;
    if (runstate_init(&rs, 4096) != 0) { fprintf(stderr, "rs init failed\n"); return 1; }

    int ids[64];
    int n = qma_tokenize(&m, "The capital of France is", ids, 64);
    fprintf(stderr, "tokens(%d):", n);
    for (int i = 0; i < n; i++) fprintf(stderr, " %d", ids[i]);
    fprintf(stderr, "\n");

    static float logits[N_VOCAB];
    /* prefill prompt, then greedy-decode 4 tokens, dumping top-8 each step */
    for (int step = 0; step < 4 + 1; step++) {
        int nt = (step == 0) ? n : 1;
        if (qma_eval(&m, &rs, ids + (step ? n + step - 1 : 0), nt,
                     logits, 8, 0, 0) != 0) {
            fprintf(stderr, "eval failed at step %d\n", step);
            return 1;
        }
        /* top-8 by simple selection */
        float best[8]; int bid[8];
        for (int k = 0; k < 8; k++) { best[k] = -1e30f; bid[k] = -1; }
        for (int v = 0; v < m.n_vocab && v < N_VOCAB; v++) {
            float lv = logits[v];
            if (lv > best[7]) {
                int p = 7;
                while (p > 0 && best[p-1] < lv) {
                    best[p] = best[p-1]; bid[p] = bid[p-1]; p--;
                }
                best[p] = lv; bid[p] = v;
            }
        }
        printf("step %d:", step);
        for (int k = 0; k < 8; k++)
            printf(" %d:%.4f", bid[k], best[k]);
        printf("\n");
        fflush(stdout);
        if (step < 4) ids[n + step] = bid[0];   /* greedy feed */
    }
    return 0;
}
