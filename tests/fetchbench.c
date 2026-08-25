/* Weight-fetch cost harness: how much does getting expert bytes into the
 * compute path actually cost?
 *   1. buffered pread (warm page cache)
 *   2. buffered pread after POSIX_FADV_DONTNEED (true cold UFS read)
 *   3. mmap page-fault path (what zero-copy matmul hits when pages are out)
 *   4. readahead(2) then mmap access
 *   5. O_DIRECT-style aligned pread if dio_fd exists
 */
#include "nn.c"
#include <stdio.h>
#include <fcntl.h>

static double ms_since(double t0) { return (now_s() - t0) * 1e3; }

int main(int argc, char **argv) {
    const char *model = argc > 1 ? argv[1]
        : "/data/data/com.termux/files/home/projects/models/qwen3.6:35b:a3b-q4km.gguf.4k";
    setenv("QMA_NO_CL", "1", 1);

    static qma_t m;
    char err[256];
    if (qma_load(&m, model, err, sizeof(err)) != 0) { fprintf(stderr, "load: %s\n", err); return 1; }

    /* find a recr (MoE) layer */
    int il = -1;
    for (int l = 0; l < N_LAYER && il < 0; l++)
        if (!IS_ATTN(l)) il = l;
    const size_t rec = layer_rec_bytes(&m, il);
    const size_t gu = (size_t)N_EMBD * N_FF_EXP * sizeof(block_q4_K) / QK_K;
    const size_t dn = (size_t)N_EMBD * N_FF_EXP *
        (m.layers[il].t_down_exps == GGML_TYPE_Q4_K ? sizeof(block_q4_K)
                                                    : sizeof(block_q6_K)) / QK_K;
    printf("layer %d record %.2f MB (gu=%.2f dn=%.2f, down=%s)\n", il,
           rec / 1048576.0, gu / 1048576.0, dn / 1048576.0,
           m.layers[il].t_down_exps == GGML_TYPE_Q4_K ? "Q4" : "Q6");
    printf("mmap_exps=%d ecache_on=%d dio_fd=%d map=%s\n",
           m.mmap_exps, m.ecache_on, m.dio_fd, m.map ? "yes" : "no");

    static uint8_t dst[8 << 20];   /* >= one record */
    const int EXP = 137;           /* arbitrary mid expert */

    const size_t off_g = m.layers[il].off_gate_exps + (size_t)EXP * gu;
    const size_t off_u = m.layers[il].off_up_exps + (size_t)EXP * gu;
    const size_t off_d = m.layers[il].off_down_exps + (size_t)EXP * dn;

    /* ---- 1. warm buffered pread (page cache hot) ---- */
    {
        /* warm once */
        pread_all(m.fd, dst, rec / 3, off_g);
        double best = 1e9;
        for (int r = 0; r < 5; r++) {
            double t0 = now_s();
            int rc = expert_fetch(&m, il, EXP, dst);
            double ms = ms_since(t0);
            if (rc != 0) { printf("fetch failed\n"); return 1; }
            if (ms < best) best = ms;
        }
        printf("%-34s %8.2f ms  (%.1f us/KB, %2.1f MB/s)\n",
               "pread warm", best, best * 1000 / (rec / 1024.0), rec / 1048576.0 / best * 1e3);
    }

    /* ---- 2. cold buffered pread (evict first) ---- */
    {
        double best = 1e9;
        for (int r = 0; r < 3; r++) {
            /* evict this expert's pages from page cache */
            posix_fadvise(m.fd, off_g, gu, POSIX_FADV_DONTNEED);
            posix_fadvise(m.fd, off_u, gu, POSIX_FADV_DONTNEED);
            posix_fadvise(m.fd, off_d, dn, POSIX_FADV_DONTNEED);
            double t0 = now_s();
            expert_fetch(&m, il, EXP, dst);
            double ms = ms_since(t0);
            if (ms < best) best = ms;
        }
        printf("%-34s %8.2f ms  (%.1f us/KB, %2.1f MB/s)\n",
               "pread cold (fadv-evicted)", best, best * 1000 / (rec / 1024.0), rec / 1048576.0 / best * 1e3);
    }

    if (m.map) {
        volatile uint64_t sink = 0;
        /* ---- 3. mmap fault-in (zero-copy matmul's cold path) ---- */
        {
            double best = 1e9;
            for (int r = 0; r < 3; r++) {
                posix_fadvise(m.fd, off_g, gu, POSIX_FADV_DONTNEED);
                posix_fadvise(m.fd, off_u, gu, POSIX_FADV_DONTNEED);
                posix_fadvise(m.fd, off_d, dn, POSIX_FADV_DONTNEED);
                const uint8_t *p = m.map + off_g;
                const uint8_t *p2 = m.map + off_u;
                const uint8_t *p3 = m.map + off_d;
                double t0 = now_s();
                for (size_t o = 0; o < gu; o += 4096) sink += p[o] + p2[o];
                for (size_t o = 0; o < dn; o += 4096) sink += p3[o];
                double ms = ms_since(t0);
                if (ms < best) best = ms;
            }
            printf("%-34s %8.2f ms  (per %.2f MB expert slab set)\n",
                   "mmap page-fault touch", best, rec / 1048576.0);
        }

        /* ---- 4. readahead then touch ---- */
        {
            double best = 1e9;
            for (int r = 0; r < 3; r++) {
                posix_fadvise(m.fd, off_g, gu, POSIX_FADV_DONTNEED);
                posix_fadvise(m.fd, off_u, gu, POSIX_FADV_DONTNEED);
                posix_fadvise(m.fd, off_d, dn, POSIX_FADV_DONTNEED);
                double t0 = now_s();
                readahead(m.fd, off_g, gu);
                readahead(m.fd, off_u, gu);
                readahead(m.fd, off_d, dn);
                const uint8_t *p = m.map + off_g;
                const uint8_t *p2 = m.map + off_u;
                const uint8_t *p3 = m.map + off_d;
                for (size_t o = 0; o < gu; o += 4096) sink += p[o] + p2[o];
                for (size_t o = 0; o < dn; o += 4096) sink += p3[o];
                double ms = ms_since(t0);
                if (ms < best) best = ms;
            }
            printf("%-34s %8.2f ms\n", "readahead + mmap touch", best);
        }

        /* ---- 5. resident mmap touch (all hot) ---- */
        {
            const uint8_t *p = m.map + off_g;
            const uint8_t *p2 = m.map + off_u;
            const uint8_t *p3 = m.map + off_d;
            for (size_t o = 0; o < rec / 3; o += 4096) sink += p[o];
            double t0 = now_s();
            for (int r = 0; r < 20; r++) {
                for (size_t o = 0; o < gu; o += 64) sink += p[o] + p2[o];
                for (size_t o = 0; o < dn; o += 64) sink += p3[o];
            }
            printf("%-34s %8.2f ms  (full-sweep, resident)\n",
                   "mmap resident sweep", ms_since(t0) / 20);
        }
    }

    /* ---- 6. O_DIRECT fd if present ---- */
    if (m.dio_fd >= 0) {
        double best = 1e9;
        for (int r = 0; r < 3; r++) {
            double t0 = now_s();
            expert_fetch(&m, il, EXP, dst);   /* uses dio_fd when dst aligned */
            best = (now_s() - t0) * 1e3 < best ? (now_s() - t0) * 1e3 : best;
        }
        printf("%-34s %8.2f ms\n", "expert_fetch (O_DIRECT path)", best);
    } else {
        printf("dio_fd not open (no O_DIRECT path armed)\n");
    }

    /* ---- 7. per-token projection ---- */
    printf("\nper decode token: 8 experts x 40 layers = 320 records\n");
    printf("  -> cold-fetch all: %.0f ms | warm all: %.0f ms (upper bounds;\n"
           "     ecache hits and routing locality reduce this heavily)\n",
           320.0 * rec / 1048576.0 / 1000.0 / 2.0 /* assume ~2GB/s UFS */,
           320.0 * rec / 1048576.0 / 1000.0 / 8.0 /* assume ~8GB/s RAM */);
    return 0;
}
