/* Standalone parity: OUR iq3_xxs/iq4_xs kernels vs ggml ref, on REAL bytes.
 * Uses the engine's own header parse (header-only) to find slabs. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "../src/qma.h"
#include "../src/iq2tables.h"
#define GGML_FP16_TO_FP32(x) half_to_float(x)
#define GGML_RESTRICT restrict
typedef struct { const uint8_t *p; const uint8_t *end; int err; } rd_t;
static uint64_t rdu64(rd_t *r){ if(r->err||r->p+8>r->end){r->err=1;return 0;} uint64_t v; memcpy(&v,r->p,8); r->p+=8; return v; }
static uint32_t rdu32(rd_t *r){ if(r->err||r->p+4>r->end){r->err=1;return 0;} uint32_t v; memcpy(&v,r->p,4); r->p+=4; return v; }
static void rdskip(rd_t *r, uint32_t t){
    switch(t){
        case 0: case 1: r->p+=1; break;
        case 2: case 3: r->p+=2; break;
        case 4: case 5: case 6: r->p+=4; break;
        case 10: case 11: case 12: r->p+=8; break;
        case 7: r->p+=1; break;
        case 8: { uint64_t n=rdu64(r); r->p+=(size_t)n; break; }
        case 9: { uint32_t at=rdu32(r); uint64_t n=rdu64(r); for(uint64_t i=0;i<n&&!r->err;i++) rdskip(r,at); break; }
        default: r->err=1;
    }
}
/* ref dequants (ggml verbatim) */
void dequantize_row_iq3_xxs_ref(const block_iq3_xxs *x, float *y, int64_t k) {
    const int64_t nb = k / QK_K; uint32_t aux32;
    for (int64_t i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t *qs = x[i].qs;
        const uint8_t *ss = qs + QK_K/4;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(&aux32, ss + 4*ib32, 4);
            const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
                const uint8_t *g1 = (const uint8_t *)(iq3xxs_grid + qs[2*l+0]);
                const uint8_t *g2 = (const uint8_t *)(iq3xxs_grid + qs[2*l+1]);
                for (int j = 0; j < 4; ++j) {
                    y[j+0] = db * g1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    y[j+4] = db * g2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
        }
    }
}
void dequantize_row_iq4_xs_ref(const block_iq4_xs *x, float *y, int64_t k) {
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t *qs = x[i].qs;
        const float d = GGML_FP16_TO_FP32(x[i].d);
        for (int ib = 0; ib < QK_K/32; ++ib) {
            const int ls = ((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[i].scales_h >> 2*ib) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) { y[j+0] = dl * kvalues_iq4nl[qs[j]&0xf]; y[j+16] = dl * kvalues_iq4nl[qs[j]>>4]; }
            y += 32; qs += 16;
        }
    }
}
/* our kernels (declared in qma.h, defined in qkerns.c) */
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <model> <layer>\n", argv[0]); return 2; }
    int fd = open(argv[1], O_RDONLY);
    struct stat st; fstat(fd, &st);
    size_t win = st.st_size < (64u<<20) ? st.st_size : (64u<<20);
    const uint8_t *map = mmap(NULL, win, PROT_READ, MAP_SHARED, fd, 0);
    rd_t r = { map, map+win, 0 };
    r.p += 4; rdu32(&r);
    uint64_t n_tensors = rdu64(&r), n_kv = rdu64(&r);
    size_t alignment = 32;
    for (uint64_t i = 0; i < n_kv && !r.err; i++) {
        uint64_t klen = rdu64(&r);
        if (r.err || r.p + klen > r.end) return 1;
        r.p += klen;
        uint32_t t = rdu32(&r);
        if (klen == 17 && memcmp(r.p - klen - 4, "general.alignment", 17) == 0) {
            if (t == 4) alignment = rdu32(&r); else if (t==10||t==11) alignment=(size_t)rdu64(&r); else rdskip(&r,t);
        } else rdskip(&r, t);
    }
    char buf[256]; size_t down_off=0; uint32_t down_typ=0; int found=0;
    char want[64]; snprintf(want, sizeof(want), "blk.%s.ffn_down_exps.weight", argv[2]);
    for (uint64_t i = 0; i < n_tensors && !r.err; i++) {
        uint64_t nlen = rdu64(&r);
        size_t cl = nlen < 255 ? nlen : 255;
        memcpy(buf, r.p, cl); buf[cl] = 0; r.p += nlen;
        uint32_t nd = rdu32(&r);
        for (uint32_t d = 0; d < nd && d < 4 && !r.err; d++) rdu64(&r);
        for (uint32_t d = 4; d < nd && !r.err; d++) rdu64(&r);
        uint32_t typ = rdu32(&r);
        uint64_t off = rdu64(&r);
        if (strcmp(buf, want) == 0) { down_off = (size_t)off; down_typ = typ; found = 1; }
    }
    if (!found) { fprintf(stderr, "tensor %s not found\n", want); return 1; }
    size_t p = (size_t)(r.p - map);
    size_t dsa = p + ((alignment - (p % alignment)) % alignment);
    size_t slab = 512*2048/256 * qma_blk_size(down_typ);
    int ex = atoi(getenv("EXP") ? getenv("EXP") : "42");
    uint8_t *b = malloc(slab);
    pread(fd, b, slab, (off_t)(dsa + down_off + (size_t)ex * slab));
    printf("layer %s down type=%u slab=%zu expert=%d\n", argv[2], down_typ, slab, ex);
    static float yref[1<<20], yous[1<<20];
    static float x[1<<20];
    for (size_t i = 0; i < slab / sizeof(block_iq4_xs) * QK_K; i++) x[i] = (float)((int)(i*2654435761u % 7) - 3);
    double sref = 0;
    if (down_typ == GGML_TYPE_IQ3_XXS) {
        dequantize_row_iq3_xxs_ref((const block_iq3_xxs*)b, yref, 512*2048);
        dequantize_row_iq3_xxs((const block_iq3_xxs*)b, yous, 512*2048);
    } else if (down_typ == GGML_TYPE_IQ4_XS) {
        dequantize_row_iq4_xs_ref((const block_iq4_xs*)b, yref, 512*2048);
        dequantize_row_iq4_xs((const block_iq4_xs*)b, yous, 512*2048);
    } else { printf("type %u not iq3xxs/iq4xs\n", down_typ); return 0; }
    double maxdiff = 0; int nmis = 0;
    for (int i = 0; i < 512*2048; i++) {
        double d = fabs((double)yref[i] - yous[i]);
        if (d > maxdiff) maxdiff = d;
        if (d > 1e-5) nmis++;
        sref += (double)yref[i] * x[i];
    }
    printf("dequant: maxdiff=%.3g mismatched=%d / %d\n", maxdiff, nmis, 512*2048);
    /* f32 dot vs ref-dot */
    double sref2 = 0; for (int i = 0; i < 512*2048; i++) sref2 += (double)yref[i]*x[i];
    float ours = (down_typ == GGML_TYPE_IQ3_XXS)
        ? dot_iq3_xxs_f32((const block_iq3_xxs*)b, x, 512*2048)
        : dot_iq4_xs_f32((const block_iq4_xs*)b, x, 512*2048);
    printf("f32 dot: ours=%.6f ref=%.6f rel=%.3g\n", ours, sref2,
           fabs(ours - sref2)/(fabs(sref2)>1?fabs(sref2):1));
    return 0;
}
