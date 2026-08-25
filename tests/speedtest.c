/* Component-level speed harness for the qma decode pass.
 * Loads the real model, times each operation class at T=1 (decode) shape,
 * prints per-op cost + a projected per-token budget using real layer counts.
 * No agent loop, no system prompt, no sampling.
 */
#include "nn.c"   /* static kernels become visible: matmul, rms_norm_m, ... */
#include <stdio.h>

static double t_now(void) { return now_s(); }

#define BENCH(name, iters, stmt) do { \
    const int IT_ = (iters); \
    double t0_ = t_now(); \
    for (int i_ = 0; i_ < IT_; i_++) { stmt; } \
    double ms_ = (t_now() - t0_) * 1e3 / IT_; \
    printf("%-28s %10.2f us/call  (%d iters)\n", name, ms_ * 1e3, IT_); \
} while (0)

int main(int argc, char **argv) {
    const char *model = argc > 1 ? argv[1]
        : "/data/data/com.termux/files/home/projects/models/qwen3.6:35b:a3b-q4km.gguf.4k";
    setenv("QMA_NO_CL", "1", 1);

    static qma_t m;
    char err[256];
    if (qma_load(&m, model, err, sizeof(err)) != 0) { fprintf(stderr, "load: %s\n", err); return 1; }
    runstate_t rs;
    if (runstate_init(&rs, 8192) != 0) { fprintf(stderr, "rs failed\n"); return 1; }
    pool_init(8);
    hcm_on = 0;   /* dense attention paths for stable measurement */

    /* find an attn layer and a recr layer */
    int la = -1, lr = -1;
    for (int il = 0; il < N_LAYER; il++) {
        if (IS_ATTN(il) && la < 0) la = il;
        if (!IS_ATTN(il) && lr < 0) lr = il;
    }
    printf("attn layer=%d recr layer=%d  (N_LAYER=%d)\n", la, lr, N_LAYER);

    /* ---- shared fixtures ---- */
    static float x[N_EMBD], xn[N_EMBD], out[N_VOCAB > 8192 ? 8192 : N_VOCAB];
    for (int i = 0; i < N_EMBD; i++) x[i] = sinf((float)i * 0.01f) * 0.5f;
    static float logits[N_VOCAB];

    /* ---- 1. norms / elementwise ---- */
    BENCH("rms_norm_m(2048)", 20000, rms_norm_m(xn, x, (const float *)m.layers[la].attn_norm, N_EMBD, 1, RMS_EPS));
    BENCH("silu_m(8192)", 20000, silu_m(out, CONV_DIM < 8192 ? CONV_DIM : 8192, 1));

    /* ---- 2. activation quant ---- */
    static int8_t xq[N_EMBD]; static float xd[N_EMBD / QK_K]; static int16_t xs[N_EMBD / 16];
    BENCH("q8k_quant(2048)", 20000, qma_q8k_quant(x, N_EMBD, xq, xd, xs));

    /* ---- 3. dense matmuls (decode T=1, CPU q8k SDOT) ---- */
    {
        tensor_t *wq_t = NULL;
        for (uint32_t i = 0; i < m.n_tensors; i++)
            if (strcmp(m.tensors[i].name, "blk.3.wq") == 0) { wq_t = &m.tensors[i]; break; }
        (void)wq_t;
        static float qout[WQ_DIM];
        BENCH("matmul wq 2048->8192 T1", 300,
              matmul(xn, m.layers[3].wq, N_EMBD, WQ_DIM, 1, m.layers[3].t_wq, qout));
        static float lout[N_VOCAB];
        BENCH("matmul lm_head ->N_VOCAB", 5,
              matmul(xn, m.output, N_EMBD, N_VOCAB, 1, m.t_output, lout));
    }

    /* ---- 4. MoE expert gate+up (fused gateup, real shared-expert slabs) ---- */
    {
        const uint8_t *g = (const uint8_t *)m.layers[la < 0 ? 0 : lr].ffn_gate_shexp;
        const uint8_t *u = (const uint8_t *)m.layers[lr].ffn_up_shexp;
        float go, uo;
        BENCH("q8k_gateup 2048x512", 5000,
              qma_q8k_gateup(g, u, xq, xd, xs, N_EMBD, &go, &uo));
    }

    /* ---- 5. attention score kernels at nk=512 live slots ---- */
    {
        const int NK = 512;
        static uint8_t slots[NK][KVQ4_REC];
        for (int s = 0; s < NK; s++) {
            for (int i = 0; i < N_EMBD_HEAD; i++) {
                float v = sinf((float)(i + s) * 0.037f) * 2.0f;
                /* kvq4_quant wants full vector; write per-slot temp */
                float tmp[N_EMBD_HEAD];
                for (int j = 0; j < N_EMBD_HEAD; j++)
                    tmp[j] = sinf((float)(j + s * 7) * 0.037f) * 2.0f;
                (void)v;
                kvq4_quant(tmp, N_EMBD_HEAD, slots[s]);
                break;
            }
        }
        float qv[N_EMBD_HEAD];
        for (int i = 0; i < N_EMBD_HEAD; i++) qv[i] = cosf((float)i * 0.041f) * 1.5f;
        static int8_t qi8[N_EMBD_HEAD]; static float qsx[N_EMBD_HEAD / 32]; static int qsum[N_EMBD_HEAD / 32];
        kvq_q_quant(qv, N_EMBD_HEAD, qi8, qsx, qsum);
        double acc = 0;
        double t0 = t_now();
        for (int r = 0; r < 50; r++)
            for (int s = 0; s < NK; s++) acc += kvq4_dot(slots[s], qv, N_EMBD_HEAD);
        double f32ms = (t_now() - t0) * 1e3 / (50 * NK);
        t0 = t_now();
        for (int r = 0; r < 50; r++)
            for (int s = 0; s < NK; s++) acc += kvq4_dot_q8(slots[s], qi8, qsx, qsum, N_EMBD_HEAD);
        double i8ms = (t_now() - t0) * 1e3 / (50 * NK);
        printf("%-28s %10.2f us/dot   (nk=512 pass: %.2f ms)\n", "kvq4_dot fp32-expand", f32ms * 1e3, f32ms * NK);
        printf("%-28s %10.2f us/dot   (nk=512 pass: %.2f ms)\n", "kvq4_dot_q8 sdot", i8ms * 1e3, i8ms * NK);
        /* V accumulate */
        static float oacc[N_EMBD_HEAD];
        t0 = t_now();
        for (int r = 0; r < 50; r++)
            for (int s = 0; s < NK; s++) kvq4_vacc(slots[s], 0.01f, oacc, N_EMBD_HEAD);
        double vms = (t_now() - t0) * 1e3 / (50 * NK);
        printf("%-28s %10.2f us/call   (nk=512 pass: %.2f ms)\n", "kvq4_vacc", vms * 1e3, vms * NK);
    }

    /* ---- 6. GDN worker (one recr layer, all heads) ---- */
    {
        static float state[S_DT_RANK * S_D_STATE * S_D_STATE];
        static float qq[VAL_DIM], kk[VAL_DIM], vv[VAL_DIM], oo[VAL_DIM];
        static float gt[S_DT_RANK], bb[S_DT_RANK];
        for (int i = 0; i < S_DT_RANK * S_D_STATE * S_D_STATE; i++) state[i] = 0.001f;
        for (int i = 0; i < VAL_DIM; i++) { qq[i] = 0.1f; kk[i] = 0.1f; vv[i] = 0.1f; }
        for (int i = 0; i < S_DT_RANK; i++) { gt[i] = -0.1f; bb[i] = 0.9f; }
        gdn_ctx gc = { state, qq, kk, vv, gt, bb, oo, 1, VAL_DIM };
        /* gdn_worker signature: (arg, i0, i1) over heads; q/k/v strides are
           token-major [t][tstride]; for T=1 tstride irrelevant */
        BENCH("gdn_worker T=1 (all heads)", 300, gdn_worker(&gc, 0, S_DT_RANK));
    }

    /* ---- 7. conv1d ---- */
    {
        static float cstate[CONV_DIM * 3], cw[CONV_DIM * S_D_CONV];
        static float cin[CONV_DIM * 4], cout2[CONV_DIM];
        for (int i = 0; i < CONV_DIM * 3; i++) cstate[i] = 0.01f;
        for (int i = 0; i < CONV_DIM * S_D_CONV; i++) cw[i] = 0.01f;
        for (int i = 0; i < CONV_DIM; i++) cin[i] = 0.02f;
        BENCH("conv1d_layer T=1", 2000, conv1d_layer(cstate, cin, cw, cout2, 1));
    }

    /* ---- 8. router (gate_inp dot over N_EXPERT) ---- */
    {
        const float *gi = (const float *)m.layers[lr].ffn_gate_inp;
        volatile float sink = 0;
        double t0 = t_now();
        for (int r = 0; r < 2000; r++)
            for (int e = 0; e < N_EXPERT; e++) {
                const float *g = gi + (size_t)e * N_EMBD;
                float s = 0;
                for (int i = 0; i < N_EMBD; i++) s += g[i] * x[i];
                sink += s;
            }
        double ms = (t_now() - t0) * 1e3 / 2000;
        printf("%-28s %10.2f us/token\n", "router 256x2048 dots", ms * 1e3);
    }

    /* ---- 9. rope ---- */
    {
        static float qr[WQ_DIM]; static int pos[1] = { 0 };
        for (int i = 0; i < WQ_DIM; i++) qr[i] = 0.1f;
        BENCH("rope_imrope WQ_DIM T=1", 20000, rope_imrope(qr, N_HEAD, N_EMBD_HEAD, N_EMBD_HEAD * 2, 1, pos));
    }

    /* ---- projected per-token budget ---- */
    printf("\n-- projected decode budget (T=1, %d layers, %d attn) --\n",
           N_LAYER, N_ATTN_LAYER);
    struct { const char *nm; double us; int per_tok; } items[] = {
        { "wq/wk/wv/wo matmuls", 0, 0 },   /* filled below if desired */
    };
    (void)items;

    printf("\nnote: matmul figures INCLUDE thread-pool fan-out; multiply\n"
           "attn-layer counts yourself: scores+vacc scale with live-slot count.\n");
    return 0;
}
