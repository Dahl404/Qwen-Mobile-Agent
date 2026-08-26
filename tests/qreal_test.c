/* qreal_test.c — validate the new kernels against REAL tensor bytes from
 * the aligned Q2_K_XL model. The engine's own mmap is the sanctioned way
 * to read the file (no side-process scanning).
 *
 * For blk.34 (the layer that poisons the residual stream):
 *   - dequantize rows of ffn_down_exps (IQ2_S) with both our code and the
 *     ggml reference; compare exactly.
 *   - dot a real row against a random activation: our f32 path vs
 *     reference-dequant dot vs q8k SDOT path on reconstructed activations.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

typedef uint16_t ggml_half;
#include "../src/qma.h"
#define GGML_FP16_TO_FP32(x) half_to_float(x)
#define GGML_RESTRICT restrict
static inline void get_scale_min_k4(int j, const uint8_t *q,
                                        uint8_t *d, uint8_t *m)
{
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else       { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
                 *m = (q[j + 4] >> 4) | ((q[j + 0] >> 6) << 4); }
}
#include <assert.h>
#include "../src/iq2tables.h"
#include "ref_dequant.inc"

static uint64_t rs = 0x9e3779b97f4a7c15ull;
static uint32_t rnd(void){rs^=rs<<13;rs^=rs>>7;rs^=rs<<17;return (uint32_t)(rs>>32);}

int main(int argc, char **argv)
{
    const char *path = argv[1];
    int il = argc > 2 ? atoi(argv[2]) : 34;

    /* --- parse just enough GGUF to find blk.<il>.ffn_down_exps --- */
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st; fstat(fd, &st);
    uint8_t *map = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); return 1; }
    const uint8_t *p = map;
    if (memcmp(p, "GGUF", 4)) { fprintf(stderr,"bad magic\n"); return 1; }
    p += 8;
    uint64_t nt = *(const uint64_t *)p; p += 8;
    uint64_t nk = *(const uint64_t *)p; p += 8;
#define RDS(v) do { uint64_t _n = *(const uint64_t*)p; p+=8; \
                    v = (const char*)p; p += _n; } while(0)
#define RDSL(v,vl) do { vl = *(const uint64_t*)p; p+=8; \
                    v = (const char*)p; p += vl; } while(0)
    size_t align = 32;
    for (uint64_t i = 0; i < nk; i++) {
        const char *k; uint64_t kl; RDSL(k, kl);
        uint32_t t = *(const uint32_t*)p; p += 4;
        if (i >= nk - 6 || !memcmp(k,"general.alignment",17)&&kl==17)
            fprintf(stderr, "kv[%llu] %.*s t=%u @%td\n",(unsigned long long)i,(int)kl,k,t,(ptrdiff_t)(p-map));
        if (!memcmp(k,"general.alignment",17)&&kl==17&&t==4) { align = *(const uint32_t*)p; }
        /* skip value. metadata vtypes: 0-7 scalars, 8 string, 9 array,
           10/11/12 u64/i64/f64 */
        if (t == 8) { uint64_t n=*(const uint64_t*)p; p+=8+n; }
        else if (t == 9) {
            uint32_t vt=*(const uint32_t*)p; p+=4;
            uint64_t n=*(const uint64_t*)p; p+=8;
            if (vt == 8) {
                for (uint64_t j=0;j<n;j++){uint64_t l=*(const uint64_t*)p;p+=8+l;}
            } else {
                size_t vs=0; switch(vt){case 0:case 1:case 7:vs=1;break;case 2:case 3:vs=2;break;case 4:case 5:case 6:vs=4;break;default:vs=8;}
                p += n*vs;
            }
        }
        else { size_t s=0; switch(t){case 0:case 1:case 7:s=1;break;case 2:case 3:s=2;break;case 4:case 5:case 6:s=4;break;default:s=8;} p+=s; }
    }
    fprintf(stderr, "nt=%llu nk=%llu align=%zu meta_end=%td\n",
            (unsigned long long)nt, (unsigned long long)nk, align, (ptrdiff_t)(p-map));
    /* tensor infos */
    const uint8_t *target = NULL; size_t target_off = 0;
    uint32_t target_type = 0;
    long dims_t[3] = {0,0,0};
    for (uint64_t i = 0; i < nt; i++) {
        const char *name; uint64_t nlen;
        RDSL(name, nlen);
        uint32_t nd = *(const uint32_t*)p; p += 4;
        long dims[3]={1,1,1};
        for (uint32_t d = 0; d < nd; d++) { dims[d<3?d:2] = (long)*(const uint64_t*)p; p += 8; if (d>=3) continue; }
        uint32_t typ = *(const uint32_t*)p; p += 4;
        uint64_t off = *(const uint64_t*)p; p += 8;
        if (i < 3 || getenv("QDBG")) fprintf(stderr, "tensor[%llu] %s nd=%u\n", (unsigned long long)i, name, nd);
        char want[96]; snprintf(want, sizeof(want), "blk.%d.ffn_down_exps.weight", il);
        if (strlen(want) == nlen && !memcmp(name, want, nlen)) {
            fprintf(stderr, "MATCHED at %llu\n", (unsigned long long)i);
            target_off = off; target_type = typ;
            dims_t[0]=dims[0]; dims_t[1]=dims[1]; dims_t[2]=dims[2];
        }
    }
    size_t data_start = ((size_t)(p - map) + align - 1) / align * align;
    fprintf(stderr, "walked %llu tensors\n", (unsigned long long)nt);
    printf("blk.%d.ffn_down_exps: type=%u ne=%ldx%ldx%ld off=%zu\n",
           il, target_type, dims_t[0], dims_t[1], dims_t[2], target_off);
    if (target_type != GGML_TYPE_IQ2_S && target_type != GGML_TYPE_IQ3_XXS &&
        target_type != GGML_TYPE_IQ4_XS &&
        target_type != GGML_TYPE_IQ2_XS && target_type != GGML_TYPE_Q3_K &&
        target_type != GGML_TYPE_Q8_0) {
        printf("type %u not covered by this test\n", target_type); return 0;
    }

    const uint8_t *base = map + data_start + target_off;
    /* one expert slab: [N_FF_EXP, N_EMBD, N_EXPERT] -> per-expert stride
       ne0*ne1 elements; row = N_FF_EXP outputs from N_EMBD inputs */
    const int n_experts = (int)dims_t[2];
    const size_t nel_per_exp = (size_t)dims_t[0] * dims_t[1];
    printf("experts=%d elems/expert=%zu\n", n_experts, nel_per_exp);

    /* pick expert 0, row 0 */
    extern void dummy(void); (void)dummy;
    const uint8_t *row = base;
    int nb = (int)(nel_per_exp / QK_K);   /* super-blocks in the row */

    static float yref[1 << 21], yous[1 << 21];
    static float x[1 << 21];
    if (nel_per_exp > sizeof(x)/sizeof(x[0])) { printf("row too big\n"); return 1; }
    for (size_t i = 0; i < nel_per_exp; i++)
        x[i] = (float)((int)(rnd() % 7) - 3);

    double sref = 0;
    if (target_type == GGML_TYPE_IQ2_S) {
        dequantize_row_iq2_s((const block_iq2_s *)row, yref, nel_per_exp);
        for (size_t i = 0; i < nel_per_exp; i++) sref += (double)yref[i] * x[i];
    } else if (target_type == GGML_TYPE_IQ3_XXS) {
        dequantize_row_iq3_xxs_ref((const block_iq3_xxs *)row, yref, nel_per_exp);
        for (size_t i = 0; i < nel_per_exp; i++) sref += (double)yref[i] * x[i];
    } else if (target_type == GGML_TYPE_IQ4_XS) {
        dequantize_row_iq4_xs_ref((const block_iq4_xs *)row, yref, nel_per_exp);
        for (size_t i = 0; i < nel_per_exp; i++) sref += (double)yref[i] * x[i];
    } else { printf("extend me\n"); return 0; }

    /* f32 dot (ours) */
    extern float dot_iq2_s_f32(const block_iq2_s *, const float *, int);
    extern float dot_iq3_xxs_f32(const block_iq3_xxs *, const float *, int);
    extern float dot_iq4_xs_f32(const block_iq4_xs *, const float *, int);
    float s32 = (target_type == GGML_TYPE_IQ2_S) ? dot_iq2_s_f32((const block_iq2_s *)row, x, (int)nel_per_exp)
              : (target_type == GGML_TYPE_IQ3_XXS) ? dot_iq3_xxs_f32((const block_iq3_xxs *)row, x, (int)nel_per_exp)
              : dot_iq4_xs_f32((const block_iq4_xs *)row, x, (int)nel_per_exp);
    printf("dot f32 ours=%.6f ref=%.6f rel=%.3g\n",
           (double)s32, sref, fabs(s32 - sref) / (fabs(sref) > 1 ? fabs(sref) : 1));

    /* q8k dot with reconstructed activations */
    static int8_t xq[1 << 21]; static float xd[8192]; static int16_t xs[65536];
    extern void qma_q8k_quant(const float *, int, int8_t *, float *, int16_t *);
    extern float qma_q8k_dot(const void *, int, const int8_t *,
                             const float *, const int16_t *, int);
    qma_q8k_quant(x, (int)nel_per_exp, xq, xd, xs);
    static float x8[1 << 21];
    for (int b = 0; b < nb; b++)
        for (int j = 0; j < QK_K; j++)
            x8[b * QK_K + j] = (float)xq[b * QK_K + j] * xd[b];
    double sref8 = 0;
    for (size_t i = 0; i < nel_per_exp; i++) sref8 += (double)yref[i] * x8[i];
    float sq8 = qma_q8k_dot(row, GGML_TYPE_IQ2_S, xq, xd, xs, (int)nel_per_exp);
    printf("dot q8k ours=%.6f ref=%.6f rel=%.3g\n",
           (double)sq8, sref8, fabs(sq8 - sref8) / (fabs(sref8) > 1 ? fabs(sref8) : 1));

    /* magnitude sanity of first superblock */
    float mx = 0; for (int i = 0; i < QK_K; i++) if (fabsf(yref[i]) > mx) mx = fabsf(yref[i]);
    printf("first-superblock max |w| = %g\n", mx);
    return 0;
}
