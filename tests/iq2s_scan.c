/* iq2s_scan.c — read blk.34/38/39 ffn_down_exps slabs straight from the
 * file at HEADER offsets and dequant with the ref (llama-identical) code.
 *
 * Goal: is the corruption in the FILE bytes, or does qma read wrong bytes?
 * Decode == ref/ggml-quants.c dequantize_row_iq2_s (verified byte-identical
 * to current llama.cpp). Slab base = data_section_abs + header off; expert
 * stride = 512*2048/256*82 = 335872 B (QK_K=256).
 *
 * Device rule: header-only mmap + one pread per slab, ~336 KB each. Only
 * run when nothing else touches the file (no qma running).
 *
 * Usage: iq2s_scan <model.gguf> [expert1 expert2 ...]   (default 0 42 63 85 128 255)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "../src/qma.h"

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

/* engine's own fp16 helper (qma.h half_to_float — native FCVT on aarch64) */

int main(int argc, char **argv){
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf> [experts...]\n", argv[0]); return 2; }
    const char *path = argv[1];
    int exps[8] = {0, 42, 63, 85, 128, 255, 7, 200};
    int n_exps = argc - 2;
    if (n_exps > 0) { for (int i = 0; i < n_exps && i < 8; i++) exps[i] = atoi(argv[2 + i]); n_exps = n_exps < 8 ? n_exps : 8; }
    else n_exps = 6;

    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st; if (fstat(fd, &st) != 0) { perror("stat"); return 1; }
    size_t win = st.st_size < (64u<<20) ? (size_t)st.st_size : (64u<<20);
    const uint8_t *map = mmap(NULL, win, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); return 1; }
    rd_t r = { map, map + win, 0 };
    if (memcmp(r.p, "GGUF", 4) != 0) { fprintf(stderr, "not GGUF\n"); return 1; }
    r.p += 4; rdu32(&r);
    uint64_t n_tensors = rdu64(&r), n_kv = rdu64(&r);
    size_t alignment = 32;
    for (uint64_t i = 0; i < n_kv && !r.err; i++) {
        uint64_t klen = rdu64(&r);
        if (r.err || r.p + klen > r.end) { fprintf(stderr, "kv oob\n"); return 1; }
        r.p += klen;
        uint32_t t = rdu32(&r);
        if (klen == 17 && memcmp(r.p - klen, "general.alignment", 17) == 0) {
            if (t == 4) alignment = rdu32(&r);
            else if (t == 10 || t == 11) alignment = (size_t)rdu64(&r);
            else rdskip(&r, t);
        } else rdskip(&r, t);
    }
    if (r.err) { fprintf(stderr, "metadata corrupt\n"); return 1; }
    /* tensor infos */
    size_t off_down[40] = {0}, off_gate[40] = {0};
    uint32_t t_down[40] = {0}, t_gate[40] = {0};
    char buf[256];
    for (uint64_t i = 0; i < n_tensors && !r.err; i++) {
        uint64_t nlen = rdu64(&r);
        if (r.err || r.p + nlen > r.end) { fprintf(stderr, "name oob\n"); return 1; }
        size_t cl = nlen < sizeof(buf)-1 ? (size_t)nlen : sizeof(buf)-1;
        memcpy(buf, r.p, cl); buf[cl] = 0; r.p += nlen;
        uint32_t nd = rdu32(&r);
        for (uint32_t d = 0; d < nd && d < 4 && !r.err; d++) rdu64(&r);
        for (uint32_t d = 4; d < nd && !r.err; d++) rdu64(&r);
        uint32_t typ = rdu32(&r);
        uint64_t off = rdu64(&r);
        if (r.err) { fprintf(stderr, "info corrupt\n"); return 1; }
        int il = -1;
        int n1 = 0, n2 = 0;
        if (i < 3 || strstr(buf,"blk.0.ffn_gate_exps") || strstr(buf,"blk.34.ffn_down_exps"))
            printf("t %llu %-40s typ=%u off=%llu nd=%u\n",(unsigned long long)i,buf,typ,(unsigned long long)off,nd);
        if (sscanf(buf, "blk.%d.ffn_down_exps.weight%n", &il, &n1) == 1 && il < 40 && (size_t)n1 == strlen(buf)) {
            off_down[il] = (size_t)off; t_down[il] = typ;
        }
        if (sscanf(buf, "blk.%d.ffn_gate_exps.weight%n", &il, &n2) == 1 && il < 40 && (size_t)n2 == strlen(buf)) {
            off_gate[il] = (size_t)off; t_gate[il] = typ;
        }
    }
    if (r.err) { fprintf(stderr, "tensor table corrupt\n"); return 1; }
    /* data section start (same math as qma_load) */
    size_t p = (size_t)(r.p - map);
    size_t pad = (alignment - (p % alignment)) % alignment;
    size_t dsa = p + pad;
    printf("file=%s dsa=%zu alignment=%zu\n", path, dsa, alignment);
    printf("DBG stored: t_down[34]=%u t_gate[0]=%u t_down[0]=%u t_gate[34]=%u\n",
           t_down[34], t_gate[0], t_down[0], t_gate[34]);

    int layers[] = {0, 34, 38, 39};
    for (int li = 0; li < 4; li++) {
        int il = layers[li];
        size_t dn_off = dsa + off_down[il];
        size_t ge_off = dsa + off_gate[il];
        size_t dn_slab = 512 * 2048 / 256 * qma_blk_size(t_down[il]);
        size_t ge_slab = 2048 * 512 / 256 * qma_blk_size(t_gate[il]);
        printf("\n== blk.%d  down type=%u slab=%zu B  gate type=%u slab=%zu B\n",
               il, t_down[il], dn_slab, t_gate[il], ge_slab);
        for (int ei = 0; ei < n_exps; ei++) {
            int ex = exps[ei];
            if (ex < 0 || ex >= 256) continue;
            /* decode down slab of expert ex */
            size_t nb = dn_slab;
            uint8_t *buf2 = malloc(nb ? nb : 1);
            if (pread(fd, buf2, nb, (off_t)(dn_off + (size_t)ex * dn_slab)) != (ssize_t)nb) {
                printf("  ex %3d: pread down FAILED\n", ex); free(buf2); continue;
            }
            if (t_down[il] != GGML_TYPE_IQ2_S) {
                printf("  ex %3d: (down not IQ2_S, skipped)\n", ex); free(buf2); continue;
            }
            int nblocks = (int)(nb / sizeof(block_iq2_s));
            double dmin = 1e30, dmax = -1e30;
            int n_big = 0, n_big1 = 0, n_nan = 0;
            for (int b = 0; b < nblocks; b++) {
                float d = half_to_float(((block_iq2_s*)buf2)[b].d);
                if (d < dmin) dmin = d;
                if (d > dmax) dmax = d;
                if (d > 100.0) n_big++;
                if (d > 1.0) n_big1++;
                if (!(d == d)) n_nan++;
            }
            printf("  ex %3d: down dmin=%.4g dmax=%.4g  d>100:%d d>1:%d nan:%d /%d blocks\n",
                   ex, dmin, dmax, n_big, n_big1, n_nan, nblocks);
            free(buf2);
        }
    }
    munmap((void*)map, win);
    close(fd);
    return 0;
}
