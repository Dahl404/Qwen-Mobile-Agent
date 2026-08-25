/* thread-scaling of the two dominant matmuls at decode shape */
#include "nn.c"
#include <stdio.h>
#include <stdlib.h>

static double t_now(void) { return now_s(); }

int main(int argc, char **argv) {
    const char *model = argc > 1 ? argv[1]
        : "/data/data/com.termux/files/home/projects/models/qwen3.6:35b:a3b-q4km.gguf.4k";
    setenv("QMA_NO_CL", "1", 1);
    static qma_t m; char err[256];
    if (qma_load(&m, model, err, sizeof(err)) != 0) { fprintf(stderr, "load %s\n", err); return 1; }

    static float x[N_EMBD], xn[N_EMBD];
    static float lout[N_VOCAB], qout[WQ_DIM];
    for (int i = 0; i < N_EMBD; i++) x[i] = sinf((float)i * 0.01f) * 0.5f;
    rms_norm_m(xn, x, (const float *)m.layers[3].attn_norm, N_EMBD, 1, RMS_EPS);

    pool_init(8);
    for (int th = 1; th <= 8; th += th == 4 ? 4 : (th == 2 ? 2 : 1)) {
        qma_pool_set_max(th);   /* throttle, don't destroy */
        /* warm */
        matmul(xn, m.output, N_EMBD, N_VOCAB, 1, m.t_output, lout);
        double t0 = t_now();
        const int R = 10;
        for (int i = 0; i < R; i++)
            matmul(xn, m.output, N_EMBD, N_VOCAB, 1, m.t_output, lout);
        double lm_ms = (t_now() - t0) * 1e3 / R;
        double bw = (double)N_VOCAB * ((size_t)N_EMBD / QK_K * sizeof(block_q4_K)) / (lm_ms * 1e-3) / 1e9;
        t0 = t_now();
        for (int i = 0; i < 100; i++)
            matmul(xn, m.layers[3].wq, N_EMBD, WQ_DIM, 1, m.layers[3].t_wq, qout);
        double wq_ms = (t_now() - t0) * 1e3 / 100;
        printf("threads=%d  lm_head %.1f ms (%.1f GB/s)   wq %.3f ms\n",
               th, lm_ms, bw, wq_ms);
        fflush(stdout);
    }
    return 0;
}
