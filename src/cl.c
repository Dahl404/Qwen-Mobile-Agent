/* cl.c — OpenCL GPU engine (see cl.h). Kernels adapted from llama.cpp's
 * ggml-opencl backend (gemm_noshuffle_q4_k_f32 / gemm_noshuffle_q6_k_f32),
 * with the activation image replaced by a plain fp16 buffer (ma3's prep
 * kernel shape) so the data movement matches qma's prefill flow. */
#include "cl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <CL/cl.h>

/* ---------------- ICD loader (dlopen + dlsym) ---------------- */
#define CLFN(ret, name, args) ret (*clf_##name) args
CLFN(cl_int, clGetPlatformIDs, (cl_uint, cl_platform_id*, cl_uint*));
CLFN(cl_int, clGetDeviceIDs, (cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*));
CLFN(cl_int, clGetDeviceInfo, (cl_device_id, cl_device_info, size_t, void*, size_t*));
CLFN(cl_context, clCreateContext, (const cl_context_properties*, cl_uint, const cl_device_id*, void (*)(const char*, const void*, size_t, void*), void*, cl_int*));
CLFN(cl_command_queue, clCreateCommandQueue, (cl_context, cl_device_id, cl_command_queue_properties, cl_int*));
CLFN(cl_program, clCreateProgramWithSource, (cl_context, cl_uint, const char* const*, const size_t*, cl_int*));
CLFN(cl_int, clBuildProgram, (cl_program, cl_uint, const cl_device_id*, const char*, void (*)(cl_program, void*), void*));
CLFN(cl_kernel, clCreateKernel, (cl_program, const char*, cl_int*));
CLFN(cl_mem, clCreateBuffer, (cl_context, cl_mem_flags, size_t, void*, cl_int*));
CLFN(cl_int, clSetKernelArg, (cl_kernel, cl_uint, size_t, const void*));
CLFN(cl_int, clEnqueueNDRangeKernel, (cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*));
CLFN(cl_int, clFinish, (cl_command_queue));
CLFN(cl_int, clEnqueueWriteBuffer, (cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*));
CLFN(cl_int, clEnqueueReadBuffer, (cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*));
CLFN(cl_int, clGetProgramBuildInfo, (cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*));
CLFN(cl_int, clReleaseMemObject, (cl_mem));
CLFN(cl_int, clReleaseKernel, (cl_kernel));
CLFN(cl_int, clReleaseProgram, (cl_program));
CLFN(cl_int, clReleaseCommandQueue, (cl_command_queue));
CLFN(cl_int, clReleaseContext, (cl_context));
CLFN(void*, clGetExtensionFunctionAddress, (const char*));
static void *clh = NULL;
static int cl_loaded = 0;
#define LOAD(s) do { *(void**)&clf_##s = dlsym(clh, #s); if (!*(void**)&clf_##s) { cl_loaded = 0; return -1; } } while (0)
#define clGetPlatformIDs clf_clGetPlatformIDs
#define clGetDeviceIDs clf_clGetDeviceIDs
#define clGetDeviceInfo clf_clGetDeviceInfo
#define clCreateContext clf_clCreateContext
#define clCreateCommandQueue clf_clCreateCommandQueue
#define clCreateProgramWithSource clf_clCreateProgramWithSource
#define clBuildProgram clf_clBuildProgram
#define clCreateKernel clf_clCreateKernel
#define clCreateBuffer clf_clCreateBuffer
#define clSetKernelArg clf_clSetKernelArg
#define clEnqueueNDRangeKernel clf_clEnqueueNDRangeKernel
#define clFinish clf_clFinish
#define clEnqueueWriteBuffer clf_clEnqueueWriteBuffer
#define clEnqueueReadBuffer clf_clEnqueueReadBuffer
#define clGetProgramBuildInfo clf_clGetProgramBuildInfo
#define clReleaseMemObject clf_clReleaseMemObject
#define clReleaseKernel clf_clReleaseKernel
#define clReleaseProgram clf_clReleaseProgram
#define clReleaseCommandQueue clf_clReleaseCommandQueue
#define clReleaseContext clf_clReleaseContext
#define clGetExtensionFunctionAddress clf_clGetExtensionFunctionAddress

/* the vendor libs are copied into work/cl/ at build time (the Android
   linker namespace blocks /vendor/lib64) */
#define CL_LIB_DIR "/data/data/com.termux/files/home/projects/qma/work/cl"

static int cl_load_icd(void) {
    if (cl_loaded) return 0;
    char drv[512], icd[512];
    snprintf(drv, sizeof(drv), "%s/libOpenCL_adreno.so", CL_LIB_DIR);
    snprintf(icd, sizeof(icd), "%s/libOpenCL.so", CL_LIB_DIR);
    dlopen(drv, RTLD_NOW | RTLD_GLOBAL);   /* prime the driver first */
    clh = dlopen(icd, RTLD_NOW | RTLD_GLOBAL);
    if (!clh) return -1;
    cl_loaded = 1;
    LOAD(clGetPlatformIDs); LOAD(clGetDeviceIDs); LOAD(clGetDeviceInfo);
    LOAD(clCreateContext); LOAD(clCreateCommandQueue); LOAD(clCreateProgramWithSource);
    LOAD(clBuildProgram); LOAD(clCreateKernel); LOAD(clCreateBuffer); LOAD(clSetKernelArg);
    LOAD(clEnqueueNDRangeKernel); LOAD(clFinish); LOAD(clEnqueueWriteBuffer); LOAD(clEnqueueReadBuffer);
    LOAD(clGetProgramBuildInfo); LOAD(clReleaseMemObject); LOAD(clReleaseKernel);
    LOAD(clReleaseProgram); LOAD(clReleaseCommandQueue); LOAD(clReleaseContext);
    LOAD(clGetExtensionFunctionAddress);
    return 0;
}

/* ---------------- kernels ----------------
 * llama.cpp production noshuffle GEMM shape: 8x4 thread tile, vector-scalar
 * dequant, half8 accumulators, zero barriers, Adreno full-wave subgroup.
 * B is [K][Tpad] fp16 halves (transposed activations, prep_act). Output
 * C[T][N] fp32 row-major. */
static const char *CL_SRC =
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"#ifdef cl_qcom_reqd_sub_group_size\n"
"#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable\n"
"#endif\n"
"#define REQD_SUBGROUP_SIZE_128\n"
/* fp32 [T][M] -> fp16 [M][Tpad] (transposed, zero-padded) */
"__kernel void prep_act(__global const float *x, __global half *xt, int T, int M, int Tpad) {\n"
"  int id = get_global_id(0);\n"
"  if (id >= M * Tpad) return;\n"
"  int t = id % Tpad;\n"
"  int m = id / Tpad;\n"
"  xt[id] = (t < T) ? (half)x[(size_t)t * M + m] : 0.0h;\n"
"}\n"
/* The Adreno compiler miscompiles component reads of vector accumulators
   (vstore4((float4)(c0.s0, c1.s0, ...)) corrupts the accumulation), so the
   GEMMs store WHOLE vectors into a tile buffer [gy][gx][c][r] and this
   kernel transposes to the natural C[T][N] row-major layout (scalar
   reads/writes only). */
"__kernel void transpose_out(__global const float *in, __global float *out,\n"
"                            int N, int gx_count, int T) {\n"
"  int id = get_global_id(0);\n"
"  if (id >= T * N) return;\n"
"  int t = id / N;\n"
"  int n = id % N;\n"
"  int gy = t >> 3;\n"
"  int r = t & 7;\n"
"  int gx = n >> 2;\n"
"  int c = n & 3;\n"
"  out[id] = in[((size_t)(gy * gx_count + gx) * 32) + c * 8 + r];\n"
"}\n"
/* Q4_K: scales/mins are 6-bit packed; this is llama.cpp's get_scale_min_k4
   on the transposed scale bytes (stride = N). */
"inline void get_scale_min_k4(int j, __global const uchar *q, int stride, uchar *d, uchar *m) {\n"
"  if (j < 4) {\n"
"    *d = q[j*stride]     & 0x3F;\n"
"    *m = q[(j+4)*stride] & 0x3F;\n"
"  } else {\n"
"    *d = (q[(j+4)*stride] & 0x0F) | ((q[(j-4)*stride] & 0xC0) >> 2);\n"
"    *m = ((q[(j+4)*stride] >> 4) & 0x0F) | ((q[j*stride] & 0xC0) >> 2);\n"
"  }\n"
"}\n"
"REQD_SUBGROUP_SIZE_128\n"
"__kernel void gemm_q4k(__global const ushort *q, __global const uchar *s,\n"
"                      __global const half *d, __global const half *dm,\n"
"                      __global const half *scb, __global const half *mvb,\n"
"                      __global const half *B, __global float *Ctile,\n"
"                      int N, int Tpad, int K, int T, int gx_count) {\n"
"  int gy = get_global_id(0);\n"
"  int gx = get_global_id(1);\n"
"  int gx_2 = gx << 2;\n"
"  float8 c0 = (float8)0.0f, c1 = (float8)0.0f, c2 = (float8)0.0f, c3 = (float8)0.0f;\n"
"  __global const half *Bk = B + (size_t)(gy << 3);\n"
"  __global const ushort *qk = q + gx_2;\n"
"  __global const half *sck = scb + gx_2;\n"
"  __global const half *mvk = mvb + gx_2;\n"
"  for (int i = 0; i < K; i += 32) {\n"
"    int sb_idx  = i / 256;\n"
"    int sub_idx = (i / 32) % 8;\n"
"    int sidx = (sb_idx * 8 + sub_idx) * N;\n"
"    float4 scl = convert_float4(vload4(0, sck + sidx));\n"
"    float4 mval = convert_float4(vload4(0, mvk + sidx));\n"
"    for (int ki = i; ki < i + 32; ki += 4) {\n"
"      ushort4 bits = vload4(0, qk + (size_t)(ki / 4) * N);\n"
"      for (int j = 0; j < 4; j++) {\n"
"        float8 Bf = convert_float8(vload8(0, Bk + (size_t)(ki + j) * Tpad));\n"
"        float w0 = (float)((bits.s0 >> (4*j)) & 0xF) * scl.s0 - mval.s0;\n"
"        float w1 = (float)((bits.s1 >> (4*j)) & 0xF) * scl.s1 - mval.s1;\n"
"        float w2 = (float)((bits.s2 >> (4*j)) & 0xF) * scl.s2 - mval.s2;\n"
"        float w3 = (float)((bits.s3 >> (4*j)) & 0xF) * scl.s3 - mval.s3;\n"
"        c0 += Bf * w0;\n"
"        c1 += Bf * w1;\n"
"        c2 += Bf * w2;\n"
"        c3 += Bf * w3;\n"
"      }\n"
"    }\n"
"  }\n"
"  int base = (gy * gx_count + gx) * 32;\n"
"  vstore8(c0, 0, Ctile + base);\n"
"  vstore8(c1, 0, Ctile + base + 8);\n"
"  vstore8(c2, 0, Ctile + base + 16);\n"
"  vstore8(c3, 0, Ctile + base + 24);\n"
"}\n"
"__kernel void gemm_q6k(__global const ushort *ql, __global const uchar *qh,\n"
"                      __global const ushort *s, __global const half *d,\n"
"                      __global const half *B, __global float *Ctile,\n"
"                      int N, int Tpad, int K, int T, int gx_count) {\n"
"  int gy = get_global_id(0);\n"
"  int gx = get_global_id(1);\n"
"  int gx_2 = gx << 2;\n"
"  float8 c0 = (float8)0.0f, c1 = (float8)0.0f, c2 = (float8)0.0f, c3 = (float8)0.0f;\n"
"  __global const half *Bk = B + (size_t)(gy << 3);\n"
"  __global const ushort *qlk = ql + gx_2;\n"
"  __global const uchar  *qhk = qh + gx_2;\n"
"  __global const ushort *sk  = s + gx_2;\n"
"  __global const half   *dk  = d + gx_2;\n"
"  for (int i = 0; i < K; i += 4) {\n"
"    ushort4 bits4 = vload4(0, qlk + (size_t)(i / 4) * N);\n"
"    uchar4  bits2 = vload4(0, qhk + (size_t)(i / 4) * N);\n"
"    char8 scale_s_8 = as_char8(vload4(0, sk + (size_t)(i / 16 / 2) * N));\n"
"    char4 scale_s = ((i / 16) % 2) == 0 ? scale_s_8.s0246 : scale_s_8.s1357;\n"
"    half4 scale_d = vload4(0, dk + (size_t)(i / 256) * N);\n"
"    for (int j = 0; j < 4; j++) {\n"
"      float8 Bf = convert_float8(vload8(0, Bk + (size_t)(i + j) * Tpad));\n"
"      float q0 = (float)(((bits4.s0 >> (4*j)) & 0xF) | (((bits2.s0 >> (2*j)) & 0x3) << 4));\n"
"      float q1 = (float)(((bits4.s1 >> (4*j)) & 0xF) | (((bits2.s1 >> (2*j)) & 0x3) << 4));\n"
"      float q2 = (float)(((bits4.s2 >> (4*j)) & 0xF) | (((bits2.s2 >> (2*j)) & 0x3) << 4));\n"
"      float q3 = (float)(((bits4.s3 >> (4*j)) & 0xF) | (((bits2.s3 >> (2*j)) & 0x3) << 4));\n"
"      float w0 = (q0 - 32.0f) * (float)scale_s.s0 * (float)scale_d.s0;\n"
"      float w1 = (q1 - 32.0f) * (float)scale_s.s1 * (float)scale_d.s1;\n"
"      float w2 = (q2 - 32.0f) * (float)scale_s.s2 * (float)scale_d.s2;\n"
"      float w3 = (q3 - 32.0f) * (float)scale_s.s3 * (float)scale_d.s3;\n"
"      c0 += Bf * w0;\n"
"      c1 += Bf * w1;\n"
"      c2 += Bf * w2;\n"
"      c3 += Bf * w3;\n"
"    }\n"
"  }\n"
"  int base = (gy * gx_count + gx) * 32;\n"
"  vstore8(c0, 0, Ctile + base);\n"
"  vstore8(c1, 0, Ctile + base + 8);\n"
"  vstore8(c2, 0, Ctile + base + 16);\n"
"  vstore8(c3, 0, Ctile + base + 24);\n"
"}\n";
/* ---------------- host-side noshuffle repack ----------------
 * qma's weights are column-major: output row r = n_in/QK_K contiguous
 * blocks. Repack per column into the llama.cpp noshuffle layout the
 * kernels above consume. Sizes match the source tensor (no bloat). */
/* host twin of the kernel's get_scale_min_k4 (6-bit scale/min unpack) */
static void get_scale_min_k4_host(int j, const uint8_t *q,
                                  uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
           *m = (q[j+4] >> 4) | ((q[j-0] >> 6) << 4); }
}

static void prep_q4k_noshuffle(const uint8_t *w, int M, int N,
                               unsigned short *q, unsigned char *s,
                               unsigned short *d, unsigned short *dm,
                               unsigned short *sc, unsigned short *mv) {
    const int n_sb = M / 256;
    for (int n = 0; n < N; n++) {
        const block_q4_K *col = (const block_q4_K *)w + (size_t)n * n_sb;
        for (int sb = 0; sb < n_sb; sb++) {
            const block_q4_K *blk = col + sb;
            d[(size_t)sb * N + n]  = blk->d;
            dm[(size_t)sb * N + n] = blk->dmin;
            for (int i = 0; i < 12; i++)
                s[((size_t)sb * 12 + i) * N + n] = blk->scales[i];
            /* precompute per-sub-block scale = d*sv and mval = dmin*mn as
               fp16 (host side), so the GEMM kernel only reads them — the
               Adreno compiler miscompiles the scale arithmetic loops */
            {
                const float d4 = half_to_float(blk->d);
                const float dm4 = half_to_float(blk->dmin);
                for (int sub = 0; sub < 8; sub++) {
                    uint8_t dsc, dmn;
                    get_scale_min_k4_host(sub, blk->scales, &dsc, &dmn);
                    sc[((size_t)sb * 8 + sub) * N + n] =
                        float_to_half(d4 * (float)dsc);
                    mv[((size_t)sb * 8 + sub) * N + n] =
                        float_to_half(dm4 * (float)dmn);
                    if (getenv("CL_DEBUG") && sb == 0 && sub == 0 && n == 0)
                        fprintf(stderr, "prep sc=%f mv=%f (d=%f dm=%f sv=%u mn=%u)\n",
                                half_to_float(sc[0]), half_to_float(mv[0]), d4, dm4, (unsigned)dsc, (unsigned)dmn);
                }
            }
            for (int k = 0; k < 256; k++) {
                /* qma's Q4_K qs is i8mm-ordered (see dot_q4_K_f32): byte
                   p*32 + l holds weight p*64+l (low nibble) and
                   p*64+32+l (high nibble), p = k>>6, l = k&31 */
                int byte = (k >> 6) * 32 + (k & 31);
                int nib = (k >> 5) & 1;
                int qrow = k >> 2, qnib = k & 3;
                int v = (blk->qs[byte] >> (4 * nib)) & 0xF;
                q[((size_t)sb * 64 + qrow) * N + n] |=
                    (unsigned short)(v << (4 * qnib));
            }
        }
    }
}

/* Q6_K in this engine's GGUFs uses the LFM/i8mm-optimized layout (see
   dot_q6_K_f32 in quants.c), NOT the standard GGML packing:
   - half h = k>>7; idx = k&127; l = idx&31; lane = idx>>5
   - ql byte = h*64 + l + 32*(lane&1); nibble = lane>=2 ? high : low
   - qh byte = h*32 + l, bits (2*lane, 2*lane+1)
   - scale = scales[h*8 + lane*2 + l/16]  (natural sub-block order) */
static void prep_q6k_noshuffle(const uint8_t *w, int M, int N,
                               unsigned short *ql, unsigned char *qh,
                               unsigned short *s, unsigned short *d) {
    const int n_sb = M / 256;
    for (int n = 0; n < N; n++) {
        const block_q6_K *col = (const block_q6_K *)w + (size_t)n * n_sb;
        for (int sb = 0; sb < n_sb; sb++) {
            const block_q6_K *blk = col + sb;
            d[(size_t)sb * N + n] = blk->d;
            for (int b = 0; b < 8; b++)
                s[((size_t)sb * 8 + b) * N + n] =
                    (unsigned short)((unsigned char)blk->scales[2 * b] |
                                     ((unsigned char)blk->scales[2 * b + 1] << 8));
            for (int k = 0; k < 256; k++) {
                int h = k >> 7, idx = k & 127;
                int l = idx & 31, lane = idx >> 5;
                int byte = h * 64 + l + 32 * (lane & 1);
                int nib = (lane >= 2) ? 4 : 0;
                int qv = (blk->ql[byte] >> nib) & 0xF;
                int hi = (blk->qh[h * 32 + l] >> (2 * lane)) & 0x3;
                int qrow = k >> 2, qnib = k & 3;
                ql[((size_t)sb * 64 + qrow) * N + n] |=
                    (unsigned short)(qv << (4 * qnib));
                qh[((size_t)sb * 64 + qrow) * N + n] |=
                    (unsigned char)(hi << (2 * qnib));
            }
        }
    }
}

/* ---------------- init ---------------- */
static int wtab_add(cl_engine_t *e, uint8_t *w, int type, int M, int N) {
    if (e->n_wtab >= e->wtab_cap) {
        e->wtab_cap = e->wtab_cap ? e->wtab_cap * 2 : 64;
        e->wtab = realloc(e->wtab, sizeof(cl_weight_t) * e->wtab_cap);
    }
    cl_weight_t *cw = &e->wtab[e->n_wtab++];
    memset(cw, 0, sizeof(*cw));
    cw->w = w; cw->type = type; cw->M = M; cw->N = N;
    if (type == GGML_TYPE_Q4_K) {
        if (getenv("CL_DEBUG")) fprintf(stderr, "cl: wtab q4k alloc\n");
        cw->q = calloc((size_t)(M / 4) * N, 2);
        cw->s = calloc((size_t)(M / 256) * 12 * N, 1);
        cw->d = calloc((size_t)(M / 256) * N, 2);
        cw->dm = calloc((size_t)(M / 256) * N, 2);
        cw->sc = calloc((size_t)(M / 32) * N, 2);
        cw->mv = calloc((size_t)(M / 32) * N, 2);
        if (!cw->q || !cw->s || !cw->d || !cw->dm || !cw->sc || !cw->mv) { fprintf(stderr, "cl: alloc fail\n"); return -1; }
        if (getenv("CL_DEBUG")) fprintf(stderr, "cl: wtab prep\n");
        prep_q4k_noshuffle(w, M, N, cw->q, cw->s, cw->d, cw->dm, cw->sc, cw->mv);
        if (getenv("CL_DEBUG")) fprintf(stderr, "cl: wtab buffers\n");
        cw->m_q = clCreateBuffer(e->ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                 (size_t)(M / 4) * N * 2, cw->q, NULL);
        cw->m_s = clCreateBuffer(e->ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                 (size_t)(M / 256) * 12 * N, cw->s, NULL);
        cw->m_d = clCreateBuffer(e->ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                 (size_t)(M / 256) * N * 2, cw->d, NULL);
        cw->m_dm = clCreateBuffer(e->ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                  (size_t)(M / 256) * N * 2, cw->dm, NULL);
        cw->m_sc = clCreateBuffer(e->ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                  (size_t)(M / 32) * N * 2, cw->sc, NULL);
        cw->m_mv = clCreateBuffer(e->ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                  (size_t)(M / 32) * N * 2, cw->mv, NULL);
    } else {
        cw->q = calloc((size_t)(M / 4) * N, 2);
        cw->qh = calloc((size_t)(M / 4) * N, 1);
        cw->s = calloc((size_t)(M / 32) * N, 2);
        cw->d = calloc((size_t)(M / 256) * N, 2);
        if (!cw->q || !cw->qh || !cw->s || !cw->d) return -1;
        prep_q6k_noshuffle(w, M, N, cw->q, cw->qh, cw->s, cw->d);
        cw->m_q = clCreateBuffer(e->ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                 (size_t)(M / 4) * N * 2, cw->q, NULL);
        cw->m_qh = clCreateBuffer(e->ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                  (size_t)(M / 4) * N, cw->qh, NULL);
        cw->m_s = clCreateBuffer(e->ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                 (size_t)(M / 32) * N * 2, cw->s, NULL);
        cw->m_d = clCreateBuffer(e->ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                 (size_t)(M / 256) * N * 2, cw->d, NULL);
    }
    return 0;
}

int cl_init(cl_engine_t *e, qma_t *m) {
    (void)m;
    memset(e, 0, sizeof(*e));
    e->ok = 0;
    if (cl_load_icd() != 0) { fprintf(stderr, "cl: ICD load failed\n"); return -1; }
    cl_platform_id plat; cl_uint np = 0;
    cl_int prc = clGetPlatformIDs(1, &plat, &np);
    if (prc != CL_SUCCESS || np < 1) { fprintf(stderr, "cl: no platform (rc=%d)\n", prc); return -1; }
    cl_device_id dev; cl_uint nd = 0;
    if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, &nd) != CL_SUCCESS || nd < 1) return -1;
    e->ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, NULL);
    if (!e->ctx) return -1;
    e->q = clCreateCommandQueue(e->ctx, dev, 0, NULL);
    if (!e->q) return -1;
    e->prog = clCreateProgramWithSource(e->ctx, 1, &CL_SRC, NULL, NULL);
    cl_int rc = clBuildProgram(e->prog, 1, &dev, NULL, NULL, NULL);
    if (rc != CL_SUCCESS) {
        char log[8192]; clGetProgramBuildInfo(e->prog, dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        fprintf(stderr, "cl: kernel build failed:\n%s\n", log);
        return -1;
    }
    e->k_prep = clCreateKernel(e->prog, "prep_act", &rc);
    e->k_q4k = clCreateKernel(e->prog, "gemm_q4k", &rc);
    e->k_q6k = clCreateKernel(e->prog, "gemm_q6k", &rc);
    e->k_transpose = clCreateKernel(e->prog, "transpose_out", &rc);
    if (!e->k_prep || !e->k_q4k || !e->k_q6k || !e->k_transpose) return -1;
    /* NO weight preprocessing at init: tensors are registered lazily on
       first use (cl_matmul_qk). Scratch buffers grow on demand. */
    fprintf(stderr, "qma: OpenCL GPU ready (Adreno) — Q4_K/Q6_K prefill offload, weights registered lazily\n");
    e->ok = 1;
    return 0;
}

int cl_ok(const cl_engine_t *e) { return e && e->ok; }

void cl_free(cl_engine_t *e) {
    if (!e) return;
    if (e->m_tile) clReleaseMemObject(e->m_tile);
    if (e->m_out) clReleaseMemObject(e->m_out);
    if (e->m_xt) clReleaseMemObject(e->m_xt);
    if (e->m_x) clReleaseMemObject(e->m_x);
    for (int i = 0; i < e->n_wtab; i++) {
        if (e->wtab[i].m_q) clReleaseMemObject(e->wtab[i].m_q);
        if (e->wtab[i].m_qh) clReleaseMemObject(e->wtab[i].m_qh);
        if (e->wtab[i].m_s) clReleaseMemObject(e->wtab[i].m_s);
        if (e->wtab[i].m_d) clReleaseMemObject(e->wtab[i].m_d);
        if (e->wtab[i].m_dm) clReleaseMemObject(e->wtab[i].m_dm);
        if (e->wtab[i].m_sc) clReleaseMemObject(e->wtab[i].m_sc);
        if (e->wtab[i].m_mv) clReleaseMemObject(e->wtab[i].m_mv);
        free(e->wtab[i].sc);
        free(e->wtab[i].mv);
        free(e->wtab[i].q);
        free(e->wtab[i].qh);
        free(e->wtab[i].s);
        free(e->wtab[i].d);
        free(e->wtab[i].dm);
    }
    free(e->wtab);
    if (e->k_transpose) clReleaseKernel(e->k_transpose);
    if (e->k_q6k) clReleaseKernel(e->k_q6k);
    if (e->k_q4k) clReleaseKernel(e->k_q4k);
    if (e->k_prep) clReleaseKernel(e->k_prep);
    if (e->prog) clReleaseProgram(e->prog);
    if (e->q) clReleaseCommandQueue(e->q);
    if (e->ctx) clReleaseContext(e->ctx);
    memset(e, 0, sizeof(*e));
}

/* ---------------- matmul ---------------- */
static cl_weight_t *wtab_find(cl_engine_t *e, const uint8_t *w) {
    for (int i = 0; i < e->n_wtab; i++)
        if (e->wtab[i].w == w) return &e->wtab[i];
    return NULL;
}

/* grow a scratch buffer; releases the old one on growth */
static cl_mem grow_buffer(cl_engine_t *e, cl_mem old, size_t need,
                          size_t *cap) {
    if (need <= *cap) return old;
    cl_mem nb = clCreateBuffer(e->ctx, CL_MEM_READ_WRITE, need, NULL, NULL);
    if (!nb) return old;                    /* keep the old (too small) */
    if (old) clReleaseMemObject(old);
    *cap = need;
    return nb;
}

int cl_matmul_qk(cl_engine_t *e, int wtype, const uint8_t *w,
                 int M, int N, int T, const float *x, float *y) {
    if (getenv("CL_DEBUG")) fprintf(stderr, "cl: matmul M=%d N=%d T=%d type=%d\n", M, N, T, wtype);
    if (!e->ok || M % 32 != 0 || T <= 1 || (wtype != GGML_TYPE_Q4_K && wtype != GGML_TYPE_Q6_K))
        return -1;
    cl_weight_t *cw = wtab_find(e, w);
    if (!cw) {
        /* lazy registration: prep this tensor on first use */
        if (getenv("CL_DEBUG")) fprintf(stderr, "cl: registering %d x %d\n", M, N);
        if (wtab_add(e, (uint8_t *)w, wtype, M, N) != 0) return -1;
        cw = wtab_find(e, w);
        if (!cw) return -1;
    }
    if (cw->M != M || cw->N != N || cw->type != wtype) return -1;
    if (getenv("CL_DEBUG")) fprintf(stderr, "cl: wtab ok, growing buffers\n");
    int Tpad = (T + 7) & ~7;
    e->m_x = grow_buffer(e, e->m_x, (size_t)T * M * 4, &e->x_cap);
    e->m_xt = grow_buffer(e, e->m_xt, (size_t)M * Tpad * 2, &e->xt_cap);
    e->m_out = grow_buffer(e, e->m_out, (size_t)T * N * 4, &e->out_cap);
    size_t gx_count = ((size_t)N + 3) / 4;
    size_t tile_need = ((size_t)T + 7) / 8 * gx_count * 32 * 4;
    e->m_tile = grow_buffer(e, e->m_tile, tile_need, &e->tile_cap);
    if (!e->m_x || !e->m_xt || !e->m_out || !e->m_tile) return -1;
    if (getenv("CL_DEBUG")) fprintf(stderr, "cl: buffers ok, write x\n");
    cl_int rc;
    rc = clEnqueueWriteBuffer(e->q, e->m_x, CL_FALSE, 0, (size_t)T * M * 4, x, 0, NULL, NULL);
    if (rc != CL_SUCCESS) { if (getenv("CL_DEBUG")) fprintf(stderr, "cl: write x rc=%d\n", (int)rc); return -1; }
    /* fp32 [T][M] -> fp16 [M][Tpad] (transposed, zero-padded) */
    clSetKernelArg(e->k_prep, 0, sizeof(cl_mem), &e->m_x);
    clSetKernelArg(e->k_prep, 1, sizeof(cl_mem), &e->m_xt);
    clSetKernelArg(e->k_prep, 2, sizeof(int), &T);
    clSetKernelArg(e->k_prep, 3, sizeof(int), &M);
    clSetKernelArg(e->k_prep, 4, sizeof(int), &Tpad);
    size_t pgs = (size_t)M * Tpad;
    rc = clEnqueueNDRangeKernel(e->q, e->k_prep, 1, NULL, &pgs, NULL, 0, NULL, NULL);
    if (rc != CL_SUCCESS) { if (getenv("CL_DEBUG")) fprintf(stderr, "cl: prep rc=%d\n", (int)rc); return -1; }
    if (getenv("CL_DEBUG")) fprintf(stderr, "cl: prep done\n");
    /* noshuffle GEMM: whole-vector tile output (see transpose_out).
       kernel args: 0-3 weights, 4=B(xt), 5=Ctile, 6=N, 7=Tpad, 8=K, 9=T, 10=gx_count */
    cl_kernel k = wtype == GGML_TYPE_Q4_K ? e->k_q4k : e->k_q6k;
    int gxc = (int)gx_count;
    if (wtype == GGML_TYPE_Q4_K) {
        /* q4k args: 0=q 1=s 2=d 3=dm 4=scb 5=mvb 6=B 7=Ctile 8=N 9=Tpad 10=K 11=T 12=gx_count */
        clSetKernelArg(k, 0, sizeof(cl_mem), &cw->m_q);
        clSetKernelArg(k, 1, sizeof(cl_mem), &cw->m_s);
        clSetKernelArg(k, 2, sizeof(cl_mem), &cw->m_d);
        clSetKernelArg(k, 3, sizeof(cl_mem), &cw->m_dm);
        clSetKernelArg(k, 4, sizeof(cl_mem), &cw->m_sc);
        clSetKernelArg(k, 5, sizeof(cl_mem), &cw->m_mv);
        clSetKernelArg(k, 6, sizeof(cl_mem), &e->m_xt);
        clSetKernelArg(k, 7, sizeof(cl_mem), &e->m_tile);
        clSetKernelArg(k, 8, sizeof(int), &N);
        clSetKernelArg(k, 9, sizeof(int), &Tpad);
        clSetKernelArg(k, 10, sizeof(int), &M);
        clSetKernelArg(k, 11, sizeof(int), &T);
        clSetKernelArg(k, 12, sizeof(int), &gxc);
    } else {
        /* q6k args: 0=ql 1=qh 2=s 3=d 4=B 5=Ctile 6=N 7=Tpad 8=K 9=T 10=gx_count */
        clSetKernelArg(k, 0, sizeof(cl_mem), &cw->m_q);
        clSetKernelArg(k, 1, sizeof(cl_mem), &cw->m_qh);
        clSetKernelArg(k, 2, sizeof(cl_mem), &cw->m_s);
        clSetKernelArg(k, 3, sizeof(cl_mem), &cw->m_d);
        clSetKernelArg(k, 4, sizeof(cl_mem), &e->m_xt);
        clSetKernelArg(k, 5, sizeof(cl_mem), &e->m_tile);
        clSetKernelArg(k, 6, sizeof(int), &N);
        clSetKernelArg(k, 7, sizeof(int), &Tpad);
        clSetKernelArg(k, 8, sizeof(int), &M);
        clSetKernelArg(k, 9, sizeof(int), &T);
        clSetKernelArg(k, 10, sizeof(int), &gxc);
    }
    size_t gsz[2] = { ((size_t)T + 7) / 8, ((size_t)N + 3) / 4 };
    size_t gx_lsz = (N / 4 >= 128) ? 128 : ((N / 4 >= 64) ? 64 : ((size_t)N / 4 > 0 ? (size_t)N / 4 : 1));
    size_t lsz[2] = { 1, gx_lsz };
    rc = clEnqueueNDRangeKernel(e->q, k, 2, NULL, gsz, lsz, 0, NULL, NULL);
    if (rc != CL_SUCCESS) { if (getenv("CL_DEBUG")) fprintf(stderr, "cl: gemm rc=%d\n", (int)rc); return -1; }
    /* transpose tile -> C[T][N] */
    clSetKernelArg(e->k_transpose, 0, sizeof(cl_mem), &e->m_tile);
    clSetKernelArg(e->k_transpose, 1, sizeof(cl_mem), &e->m_out);
    clSetKernelArg(e->k_transpose, 2, sizeof(int), &N);
    clSetKernelArg(e->k_transpose, 3, sizeof(int), &gxc);
    clSetKernelArg(e->k_transpose, 4, sizeof(int), &T);
    size_t tgs = (size_t)T * N;
    rc = clEnqueueNDRangeKernel(e->q, e->k_transpose, 1, NULL, &tgs, NULL, 0, NULL, NULL);
    if (rc != CL_SUCCESS) { if (getenv("CL_DEBUG")) fprintf(stderr, "cl: transpose rc=%d\n", (int)rc); return -1; }
    clFinish(e->q);
    if (getenv("QMA_CL_NOTRANS")) {
        /* debug: read the raw tile instead of the transposed output */
        size_t gy_count = ((size_t)T + 7) / 8;
        clEnqueueReadBuffer(e->q, e->m_tile, CL_TRUE, 0, gy_count * gx_count * 32 * 4, y, 0, NULL, NULL);
        return 0;
    }
    clEnqueueReadBuffer(e->q, e->m_out, CL_TRUE, 0, (size_t)T * N * 4, y, 0, NULL, NULL);
    return 0;
}
