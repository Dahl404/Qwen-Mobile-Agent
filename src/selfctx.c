#define _GNU_SOURCE
/* selfctx.c — self-contained context (see selfctx.h).
 * The session (KV + recurrent state + salience) is embedded in a dated
 * copy of the executable. Reading the running binary's tail is allowed;
 * writing a NEW file at exit sidesteps Android's "Text file busy".
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <errno.h>
#include "selfctx.h"
#include "intern.h"

#define PAGE 4096

/* layout of the pre-self-hosting (QMASNAP2) header, for reading old snapshots */
typedef struct {
    char     magic[8];
    uint64_t base_size;
    uint64_t cfg_off, cfg_size;
    uint64_t kv_off, kv_size;
    uint64_t st_off, st_size;
    uint64_t sal_off, sal_size;
    uint64_t sysfp;
    uint64_t n_pos;
    uint64_t flags;
} hdr_old_t;

static uint64_t align_up(uint64_t v, uint64_t a) {
    return (v + a - 1) & ~(a - 1);
}

/* copy [src_off, src_off+size) of src_fd into dst_fd at dst_off,
   preserving sparsity (SEEK_DATA/SEEK_HOLE walk relative to src_off) so an
   unused KV region stays a hole on disk. */
static int copy_sparse_at(int src_fd, off_t src_off, int dst_fd, off_t dst_off, size_t size) {
    off_t pos = 0;
    char buf[65536];
    while (pos < (off_t)size) {
        off_t data = lseek(src_fd, src_off + pos, SEEK_DATA);
        if (data < 0 || data - src_off >= (off_t)size) break;   /* rest is a hole */
        off_t hole = lseek(src_fd, data, SEEK_HOLE);
        if (hole < 0) hole = src_off + (off_t)size;
        if (hole > src_off + (off_t)size) hole = src_off + (off_t)size;
        off_t d = data - src_off;
        while (d < hole - src_off) {
            size_t want = (size_t)(hole - src_off - d);
            if (want > sizeof(buf)) want = sizeof(buf);
            ssize_t n = pread(src_fd, buf, want, src_off + d);
            if (n <= 0) return -1;
            if (pwrite(dst_fd, buf, (size_t)n, dst_off + d) != n) return -1;
            d += n;
        }
        pos = hole - src_off;
    }
    return 0;
}

static int copy_plain_at(int src_fd, off_t src_off, int dst_fd, off_t dst_off, size_t size) {
    off_t pos = 0;
    char buf[65536];
    while (pos < (off_t)size) {
        size_t want = size - (size_t)pos;
        if (want > sizeof(buf)) want = sizeof(buf);
        ssize_t n = pread(src_fd, buf, want, src_off + pos);
        if (n <= 0) return -1;
        if (pwrite(dst_fd, buf, (size_t)n, dst_off + pos) != n) return -1;
        pos += n;
    }
    return 0;
}

int selfctx_detect(const char *path, selfctx_hdr_t *hdr) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < (off_t)SELFCTX_HDR_LEN_OLD) {
        close(fd);
        return 0;
    }
    /* current format first (QMASNAP3, 136-byte header) */
    if (st.st_size >= (off_t)SELFCTX_HDR_LEN_NEW) {
        selfctx_hdr_t h;
        ssize_t n = pread(fd, &h, sizeof(h), st.st_size - (off_t)sizeof(h));
        if (n == (ssize_t)sizeof(h) && memcmp(h.magic, SELFCTX_MAGIC, 8) == 0) {
            close(fd);
            if (h.base_size == 0 || h.base_size > (uint64_t)st.st_size) return 0;
            if (h.int_size > 0) {
                if (h.int_off < h.base_size) return 0;
                if (h.cfg_off > 0 && h.int_off + h.int_size > h.cfg_off) return 0;
            } else if (h.cfg_off < h.base_size) {
                return 0;
            }
            if (h.cfg_size > 0 && h.cfg_off + h.cfg_size > h.kv_off) return 0;
            if (h.kv_size > 0 && h.kv_off + h.kv_size > (uint64_t)st.st_size) return 0;
            if (hdr) *hdr = h;
            return 1;
        }
    }
    /* legacy format (QMASNAP2, 104-byte header): internal fields read as 0 */
    {
        hdr_old_t o;
        ssize_t n = pread(fd, &o, sizeof(o), st.st_size - (off_t)sizeof(o));
        if (n == (ssize_t)sizeof(o) && memcmp(o.magic, SELFCTX_MAGIC2, 8) == 0) {
            close(fd);
            if (o.base_size == 0 || o.base_size > (uint64_t)st.st_size) return 0;
            if (o.cfg_off < o.base_size) return 0;
            if (o.cfg_off + o.cfg_size > o.kv_off) return 0;
            if (o.kv_off + o.kv_size > (uint64_t)st.st_size) return 0;
            if (hdr) {
                memset(hdr, 0, sizeof(*hdr));
                memcpy(hdr->magic, o.magic, 8);
                hdr->base_size = o.base_size;
                hdr->cfg_off = o.cfg_off;  hdr->cfg_size = o.cfg_size;
                hdr->kv_off = o.kv_off;    hdr->kv_size = o.kv_size;
                hdr->st_off = o.st_off;    hdr->st_size = o.st_size;
                hdr->sal_off = o.sal_off;  hdr->sal_size = o.sal_size;
                hdr->sysfp = o.sysfp;      hdr->n_pos = o.n_pos;
                hdr->flags = o.flags;
            }
            return 1;
        }
    }
    close(fd);
    return 0;
}

int selfctx_extract(const char *path, const selfctx_hdr_t *hdr, const char *dir) {
    int src = open(path, O_RDONLY);
    if (src < 0) return -1;
    char p[1200];
    int rc = -1;
    /* kv.bin — sparse copy of the region */
    snprintf(p, sizeof(p), "%s/kv.bin", dir);
    int dst = open(p, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (dst < 0) goto out1;
    if (ftruncate(dst, (off_t)hdr->kv_size) != 0) goto out2;
    if (copy_sparse_at(src, (off_t)hdr->kv_off, dst, 0, (size_t)hdr->kv_size) != 0) goto out2;
    close(dst);
    /* state.bin */
    snprintf(p, sizeof(p), "%s/state.bin", dir);
    dst = open(p, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (dst < 0) goto out1;
    if (copy_plain_at(src, (off_t)hdr->st_off, dst, 0, (size_t)hdr->st_size) != 0) goto out2;
    close(dst);
    /* salience.bin */
    snprintf(p, sizeof(p), "%s/salience.bin", dir);
    dst = open(p, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (dst < 0) goto out1;
    if (copy_plain_at(src, (off_t)hdr->sal_off, dst, 0, (size_t)hdr->sal_size) != 0) goto out2;
    close(dst);
    /* sysfp.txt + sysready.txt */
    snprintf(p, sizeof(p), "%s/sysfp.txt", dir);
    FILE *f = fopen(p, "w");
    if (!f) goto out1;
    fprintf(f, "%llx\n", (unsigned long long)hdr->sysfp);
    fclose(f);
    if (hdr->flags & SELFCTX_READY) {
        snprintf(p, sizeof(p), "%s/sysready.txt", dir);
        f = fopen(p, "w");
        if (!f) goto out1;
        fprintf(f, "%llx\n", (unsigned long long)hdr->sysfp);
        fclose(f);
    }
    rc = 0;
out2:
    close(dst);
out1:
    close(src);
    return rc;
}

char *selfctx_get_config(const char *path, const selfctx_hdr_t *hdr) {
    if (!hdr || hdr->cfg_size == 0) return NULL;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    char *buf = malloc((size_t)hdr->cfg_size + 1);
    if (!buf) { close(fd); return NULL; }
    ssize_t n = pread(fd, buf, (size_t)hdr->cfg_size, (off_t)hdr->cfg_off);
    close(fd);
    if (n != (ssize_t)hdr->cfg_size) { free(buf); return NULL; }
    buf[hdr->cfg_size] = 0;
    return buf;
}

/* 1 if s is a valid YYYYMMDD-HHMM timestamp (13 chars: 8 digits, '-', 4 digits) */
static int is_ts_suffix(const char *s) {
    if (strlen(s) != 13) return 0;
    for (int i = 0; i < 13; i++) {
        if (i == 8) { if (s[i] != '-') return 0; }
        else if (s[i] < '0' || s[i] > '9') return 0;
    }
    return 1;
}

int selfctx_append_bare(int fd, uint64_t base_size, uint64_t int_off,
                        uint64_t int_size, uint64_t gen, uint64_t build_time) {
    selfctx_hdr_t h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, SELFCTX_MAGIC, 8);
    h.base_size = base_size;
    h.int_off = int_off;
    h.int_size = int_size;
    h.gen = gen;
    h.build_time = build_time;
    /* no session sections (cfg/kv/st/sal = 0) */
    ssize_t n = write(fd, &h, sizeof(h));
    return n == (ssize_t)sizeof(h) ? 0 : -1;
}

char *selfctx_snapshot_gen(const char *src, const char *dir, const char *out_dir,
                           const char *cfg, size_t cfg_len,
                           const char *intern_dir, uint64_t gen) {
    selfctx_hdr_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, SELFCTX_MAGIC, 8);
    hdr.gen = gen;
    hdr.build_time = (uint64_t)time(NULL);

    int src_fd = open(src, O_RDONLY);
    if (src_fd < 0) return NULL;
    struct stat st;
    if (fstat(src_fd, &st) != 0) { close(src_fd); return NULL; }
    hdr.base_size = (uint64_t)st.st_size;
    /* if src is itself a snapshot, the base is its recorded base_size */
    {
        selfctx_hdr_t old;
        if (selfctx_detect(src, &old)) hdr.base_size = old.base_size;
    }

    /* session files from the temp dir */
    char kv_path[1200], st_path[1200], sal_path[1200], fp_path[1200], rdy_path[1200];
    snprintf(kv_path, sizeof(kv_path), "%s/kv.bin", dir);
    snprintf(st_path, sizeof(st_path), "%s/state.bin", dir);
    snprintf(sal_path, sizeof(sal_path), "%s/salience.bin", dir);
    snprintf(fp_path, sizeof(fp_path), "%s/sysfp.txt", dir);
    snprintf(rdy_path, sizeof(rdy_path), "%s/sysready.txt", dir);

    struct stat kst, sst, ast;
    if (stat(kv_path, &kst) != 0) { close(src_fd); return NULL; }
    int have_st = (stat(st_path, &sst) == 0);
    int have_sal = (stat(sal_path, &ast) == 0);

    /* layout: [base][internal][pad][cfg][kv][state][salience][header] */
    hdr.int_off = align_up(hdr.base_size, PAGE);
    /* int_size is filled after embedding (needs the blob's actual size) */

    /* n_pos + fp from the temp dir */
    {
        FILE *f = fopen(fp_path, "r");
        if (f) { if (fscanf(f, "%llx", (unsigned long long *)&hdr.sysfp) != 1) hdr.sysfp = 0; fclose(f); }
        struct stat rd;
        if (stat(rdy_path, &rd) == 0) hdr.flags |= SELFCTX_READY;
    }

    /* output name: <base>-gen<N>-YYYYMMDD-HHMMSS (strip previous suffixes) */
    char name[512];
    {
        const char *slash = strrchr(src, '/');
        const char *base = slash ? slash + 1 : src;
        char clean[512];
        snprintf(clean, sizeof(clean), "%s", base);
        size_t l = strlen(clean);
        /* strip a -YYYYMMDD-HHMMSS tail, then a -gen<N> tail */
        if (l > 14 && clean[l-14] == '-' && is_ts_suffix(clean + l - 13)) {
            clean[l-14] = 0;
            l = strlen(clean);
            size_t d = l;
            while (d > 0 && clean[d-1] >= '0' && clean[d-1] <= '9') d--;
            if (d > 0 && d >= 5 && strncmp(clean + d - 5, "-gen", 4) == 0)
                clean[d - 5] = 0;
        }
        time_t t = time(NULL);
        struct tm tm;
        localtime_r(&t, &tm);
        char ts[32];
        snprintf(ts, sizeof(ts), "%04d%02d%02d-%02d%02d",
                 tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min);
        if (gen > 0)
            snprintf(name, sizeof(name), "%s-gen%llu-%s", clean, (unsigned long long)gen, ts);
        else
            snprintf(name, sizeof(name), "%s-%s", clean, ts);
    }
    char out_path[1200];
    snprintf(out_path, sizeof(out_path), "%s/%s", out_dir, name);

    int dst = open(out_path, O_RDWR | O_CREAT | O_TRUNC, 0755);
    if (dst < 0) { close(src_fd); return NULL; }
    /* base code */
    if (copy_plain_at(src_fd, 0, dst, 0, (size_t)hdr.base_size) != 0) goto fail;
    /* pad base → int_off, then embed the internal tree */
    if (ftruncate(dst, (off_t)hdr.int_off) != 0) goto fail;
    if (intern_dir && intern_dir[0]) {
        if (lseek(dst, (off_t)hdr.int_off, SEEK_SET) < 0) goto fail;
        if (intern_embed_dir(dst, intern_dir, 0, hdr.gen, hdr.build_time) != 0) goto fail;
        off_t end = lseek(dst, 0, SEEK_END);
        if (end < 0) goto fail;
        hdr.int_size = (uint64_t)(end - (off_t)hdr.int_off);
    }
    /* remaining sections */
    hdr.cfg_off = align_up(hdr.int_off + hdr.int_size, PAGE);
    hdr.cfg_size = cfg ? (uint64_t)cfg_len : 0;
    hdr.kv_off = hdr.cfg_off + hdr.cfg_size;
    hdr.kv_size = (uint64_t)kst.st_size;
    hdr.st_off = hdr.kv_off + hdr.kv_size;
    hdr.st_size = have_st ? (uint64_t)sst.st_size : 0;
    hdr.sal_off = hdr.st_off + hdr.st_size;
    hdr.sal_size = have_sal ? (uint64_t)ast.st_size : 0;
    uint64_t total = hdr.sal_off + hdr.sal_size + sizeof(selfctx_hdr_t);
    if (ftruncate(dst, (off_t)total) != 0) goto fail;
    /* model-path config blob */
    if (hdr.cfg_size > 0) {
        if (pwrite(dst, cfg, (size_t)hdr.cfg_size, (off_t)hdr.cfg_off) != (ssize_t)hdr.cfg_size)
            goto fail;
    }
    /* KV (sparse) */
    {
        int kv = open(kv_path, O_RDONLY);
        if (kv < 0) goto fail;
        int r = copy_sparse_at(kv, 0, dst, (off_t)hdr.kv_off, (size_t)hdr.kv_size);
        close(kv);
        if (r != 0) goto fail;
    }
    /* state + salience */
    if (have_st) {
        int sf = open(st_path, O_RDONLY);
        if (sf < 0) goto fail;
        int r = copy_plain_at(sf, 0, dst, (off_t)hdr.st_off, (size_t)hdr.st_size);
        close(sf);
        if (r != 0) goto fail;
    }
    if (have_sal) {
        int af = open(sal_path, O_RDONLY);
        if (af < 0) goto fail;
        int r = copy_plain_at(af, 0, dst, (off_t)hdr.sal_off, (size_t)hdr.sal_size);
        close(af);
        if (r != 0) goto fail;
    }
    /* header at the very end */
    if (pwrite(dst, &hdr, sizeof(hdr), (off_t)(total - sizeof(hdr))) != (ssize_t)sizeof(hdr))
        goto fail;
    fsync(dst);
    close(dst);
    close(src_fd);
    return strdup(out_path);
fail:
    close(dst);
    close(src_fd);
    unlink(out_path);
    return NULL;
}
