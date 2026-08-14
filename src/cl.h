/* cl.h — OpenCL GPU engine for qma (Adreno 830 via the vendor ICD).
 *
 * Scope: PREFILL matmuls (T > 1) run on the GPU; decode stays on the CPU
 * (bandwidth-bound anyway, and it spreads the heat). The streamed MoE
 * experts are NOT offloaded — only the dense projections (attention + ssm)
 * and the Q6_K lm head.
 *
 * Math: Q4_K/Q6_K weights are repacked ONCE (host-side, lazily on first
 * use) into the llama.cpp "noshuffle" layout — the exact layout the
 * production kernels gemm_noshuffle_q4_k_f32 / gemm_noshuffle_q6_k_f32
 * consume (ma3's method: take the llama.cpp kernels, adapt for prefill):
 *   q4_K: q[(k/4)*N][n] ushort nibbles, s[(sb*12+i)*N][n] scale bytes,
 *         d/dm[sb*N][n] half super-block scales
 *   q6_K: ql[(k/4)*N][n] ushort low nibbles, qh[(k/4)*N][n] uchar high
 *         2-bit, s[(sb*8+b)*N][n] ushort scale pairs, d[sb*N][n] half
 * The GEMM is the llama.cpp 8x4 thread-tile shape (vector-scalar dequant,
 * half8 accumulators, zero barriers, Adreno full-wave subgroup). The
 * activations are transposed fp32->fp16 to [K][Tpad] by a prep kernel;
 * output is y[T][N] fp32 directly (no host transpose).
 *
 * The weight buffers use CL_MEM_USE_HOST_PTR (zero-copy into the resident
 * anonymous RAM — shared LPDDR5X, no GPU upload).
 *
 * Loader: the Android linker namespace blocks /vendor/lib64, so the vendor
 * libs are copied into work/cl/ and dlopen'd from there (driver primed
 * first, then the ICD shim, then dlsym every cl* symbol).
 */
#ifndef QMA_CL_H
#define QMA_CL_H

#include "qma.h"
#include <CL/cl.h>

typedef struct cl_weight cl_weight_t;
struct cl_weight {
    uint8_t *w;                /* source pointer (lookup key) */
    int      type;             /* GGML_TYPE_Q4_K or GGML_TYPE_Q6_K */
    cl_mem   m_q;              /* q4k nibbles / q6k low nibbles [K/4][N] */
    cl_mem   m_qh;             /* q6k only: high 2-bit [K/4][N] uchars */
    cl_mem   m_s;              /* q4k: scales [K/256*12][N] uchars
                                  q6k: scale pairs [K/32][N] ushorts */
    cl_mem   m_d;              /* super-block scales [K/256][N] halves */
    cl_mem   m_dm;             /* q4k only: mins [K/256][N] halves */
    cl_mem   m_sc, m_mv;       /* q4k: precomputed per-sub-block scale/mval
                                  [K/32][N] halves (host-computed so the GPU
                                  does no scale arithmetic — the Adreno
                                  compiler corrupts those loops) */
    void     *q, *qh, *s, *d, *dm, *sc, *mv;  /* host copies */
    int      M, N;
};
typedef struct cl_engine {
    int ok;
    cl_context ctx;
    cl_command_queue q;
    cl_program prog;
    cl_kernel k_prep, k_q4k, k_q6k, k_transpose;
    cl_weight_t *wtab;
    int n_wtab, wtab_cap;
    size_t x_cap, xt_cap, out_cap, tile_cap;   /* bytes */
    cl_mem m_x, m_xt, m_out, m_tile;
} cl_engine_t;

/* init: load the ICD, pick the GPU, build the kernels. Returns 0 on
 * success, -1 if OpenCL is unavailable (caller falls back to CPU).
 * Weights are registered lazily on first use (no init-time repack). */
int      cl_init(cl_engine_t *e, qma_t *m);
void     cl_free(cl_engine_t *e);
int      cl_ok(const cl_engine_t *e);

/* Q4_K/Q6_K matmul on the GPU: y[T][N] = x[T][M] · W. W is the engine's
 * column-major layout (output row r = n_in/QK_K contiguous blocks). x is
 * f32; converted to fp16 for the upload. y is f32. Returns 0 on success,
 * -1 if the GPU cannot take this call. */
int      cl_matmul_qk(cl_engine_t *e, int wtype, const uint8_t *w,
                      int M, int N, int T, const float *x, float *y);

#endif
