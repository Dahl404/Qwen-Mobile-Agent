/* mtp_probe.c — HEADER-ONLY GGUF probe, uses the ENGINE's exact quant ids.
 *
 * Maps only the first 64 MiB of a GGUF (metadata + tensor-info table live
 * there; the tensor DATA section is never touched). Prints block_count and
 * every tensor whose name mentions mtp / nextn / blk.<NLAYER-1> / output.
 * Env:
 *   ALL=1  print every tensor (name+dims only)
 *   EXPT=1 print per-layer expert/attn type table (name+dims only)
 *
 * Device rule: never full-file scan the multi-GB models with side
 * processes. This mirrors gguf_scan() (the engine's own loader) — header
 * pages only, ~tens of MB faulted in, and only safe when nothing else is
 * using the file. Check ps/free before running.
 *
 * Usage: mtp_probe <model.gguf> [<nlayers>]   (nlayers default 40)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "../src/qma.h"   /* engine's exact GGML_TYPE ids + block sizes */

#define WINDOW (64u << 20)   /* header-only window: 64 MiB */

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

static const char *tname(uint32_t t){
    switch (t) {
        case GGML_TYPE_F32:    return "F32";
        case GGML_TYPE_Q4_K:   return "Q4_K";
        case GGML_TYPE_Q5_K:   return "Q5_K";
        case GGML_TYPE_Q6_K:   return "Q6_K";
        case GGML_TYPE_Q3_K:   return "Q3_K";
        case GGML_TYPE_Q3_KXS: return "Q3_KXS";
        case GGML_TYPE_Q2_K:   return "Q2_K";
        case GGML_TYPE_BF16:   return "BF16";
        case GGML_TYPE_IQ2_XS: return "IQ2_XS";
        case GGML_TYPE_IQ2_S:  return "IQ2_S";
        case GGML_TYPE_Q8_0:   return "Q8_0";
        default: { static char b[16]; snprintf(b, sizeof(b), "id%u", t); return b; }
    }
}
/* byte size for one QK_K super-block via the ENGINE's own table */
static size_t blksize(uint32_t t){ return qma_blk_size((int)t); }
static int qk(uint32_t t){ return (t == GGML_TYPE_Q8_0) ? QK8_0 : QK_K; }

int main(int argc, char **argv){
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf> [nlayers]\n", argv[0]); return 2; }
    int nlayers = argc > 2 ? atoi(argv[2]) : 40;
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st; if (fstat(fd, &st) != 0) { perror("stat"); return 1; }
    size_t win = st.st_size < WINDOW ? (size_t)st.st_size : WINDOW;
    const uint8_t *map = mmap(NULL, win, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); return 1; }
    close(fd);
    rd_t r = { map, map + win, 0 };
    if (memcmp(r.p, "GGUF", 4) != 0) { fprintf(stderr, "not GGUF\n"); return 1; }
    r.p += 4;
    uint32_t ver = rdu32(&r);
    uint64_t n_tensors = rdu64(&r), n_kv = rdu64(&r);
    printf("file=%s size=%.2f GiB GGUFv%u tensors=%llu kv=%llu\n",
           argv[1], (double)st.st_size/1e9, ver,
           (unsigned long long)n_tensors, (unsigned long long)n_kv);

    uint32_t block_count = 0; int arch[4] = {0};
    for (uint64_t i = 0; i < n_kv && !r.err; i++) {
        uint64_t klen = rdu64(&r);
        if (r.err || r.p + klen > r.end) { fprintf(stderr, "kv name oob\n"); return 1; }
        const char *k = (const char*)r.p; r.p += klen;
        uint32_t t = rdu32(&r);
        if (klen >= 12 && memcmp(k, "block_count", 11) == 0) {
            if (t == 4) block_count = rdu32(&r);
            else if (t == 10 || t == 11) block_count = (uint32_t)rdu64(&r);
            else rdskip(&r, t);
        } else if (klen >= 18 && memcmp(k, "general.architecture", 20) == 0 && t == 8) {
            uint64_t l = rdu64(&r);
            if (!r.err && r.p + l <= r.end) { memcpy(arch, r.p, l < 16 ? l : 16); r.p += l; }
        } else rdskip(&r, t);
    }
    if (r.err) { fprintf(stderr, "metadata corrupt\n"); return 1; }
    printf("arch=%.16s block_count=%u\n", (char*)arch, block_count);

    /* tensor info table: pass 1 collect names for dup detection */
    rd_t r1 = r;
    char (*names)[96] = calloc(n_tensors, 96);
    for (uint64_t i = 0; i < n_tensors && !r1.err; i++) {
        uint64_t nlen = rdu64(&r1);
        if (r1.err || r1.p + nlen > r1.end) break;
        size_t cl = nlen < 95 ? (size_t)nlen : 95;
        memcpy(names[i], r1.p, cl); names[i][cl] = 0;
        r1.p += nlen;
        uint32_t nd = rdu32(&r1);
        for (uint32_t d = 0; d < nd && d < 4 && !r1.err; d++) rdu64(&r1);
        for (uint32_t d = 4; d < nd && !r1.err; d++) rdu64(&r1);
        rdu32(&r1); rdu64(&r1);
    }
    int dups = 0;
    for (uint64_t i = 0; i < n_tensors && !r1.err; i++)
        for (uint64_t j = i + 1; j < n_tensors; j++)
            if (strcmp(names[i], names[j]) == 0) {
                if (dups < 20) printf("  DUP: %s (idx %llu, %llu)\n", names[i],
                                      (unsigned long long)i, (unsigned long long)j);
                dups++;
            }
    printf("dup tensor names: %d\n", dups);
    free(names);

    /* tensor info table */
    char buf[256];
    for (uint64_t i = 0; i < n_tensors && !r.err; i++) {
        uint64_t nlen = rdu64(&r);
        if (r.err || r.p + nlen > r.end) { fprintf(stderr, "tensor name oob at %llu\n", (unsigned long long)i); return 1; }
        size_t cl = nlen < sizeof(buf)-1 ? (size_t)nlen : sizeof(buf)-1;
        memcpy(buf, r.p, cl); buf[cl] = 0;
        r.p += nlen;
        uint32_t nd = rdu32(&r);
        int64_t dims[4] = {1,1,1,1};
        for (uint32_t d = 0; d < nd && d < 4 && !r.err; d++) dims[d] = (int64_t)rdu64(&r);
        for (uint32_t d = 4; d < nd && !r.err; d++) rdu64(&r);
        uint32_t typ = rdu32(&r);
        uint64_t off = rdu64(&r);
        if (r.err) { fprintf(stderr, "tensor info corrupt at %llu\n", (unsigned long long)i); return 1; }
        int want = 0;
        if (getenv("ALL")) want = 1;
        if (strstr(buf, "mtp") || strstr(buf, "nextn") || strstr(buf, "output") ||
            strstr(buf, "token_embd") || strstr(buf, "norm.weight")) want = 1;
        else {
            /* blk.<nlayers> and above = beyond main stack */
            int il = -1;
            if (sscanf(buf, "blk.%d.", &il) == 1 && il >= nlayers) want = 1;
        }
        /* expert-type table for ALL layers (name+dims only, no type) */
        if (getenv("EXPT")) {
            int il = -1;
            if (sscanf(buf, "blk.%d.", &il) == 1 && il < nlayers) {
                if (strstr(buf, "ffn_gate_exps.weight") || strstr(buf, "ffn_up_exps.weight") ||
                    strstr(buf, "ffn_down_exps.weight") || strstr(buf, "ffn_gate_inp.weight") ||
                    strstr(buf, "ffn_gate_shexp.weight") || strstr(buf, "ffn_up_shexp.weight") ||
                    strstr(buf, "ffn_down_shexp.weight") || strstr(buf, "attn_q.weight") ||
                    strstr(buf, "attn_qkv.weight") || strstr(buf, "attn_output.weight") ||
                    strstr(buf, "attn_k.weight") || strstr(buf, "attn_v.weight") ||
                    strstr(buf, "ssm_out.weight")) {
                    printf("E %d %-32s %-7s %lldx%lldx%lld\n", il, buf + 5,
                           tname(typ), (long long)dims[0], (long long)dims[1], (long long)dims[2]);
                    want = 0;
                }
            }
        }
        if (want) {
            size_t bs = blksize(typ), q = qk(typ);
            size_t nbytes = bs ? (size_t)((dims[0]/q) * dims[1]) * bs * (size_t)dims[2] : 0;
            printf("  %-56s type=%-7s dims=%lldx%lldx%lldx%lld off=%llu nbytes=%zu\n",
                   buf, tname(typ), (long long)dims[0], (long long)dims[1],
                   (long long)dims[2], (long long)dims[3],
                   (unsigned long long)off, nbytes);
        }
    }
    if (r.err) { fprintf(stderr, "tensor table corrupt\n"); return 1; }
    printf("tensor table consumed %zu bytes of header\n", (size_t)(r.p - map));
    munmap((void*)map, win);
    return 0;
}
