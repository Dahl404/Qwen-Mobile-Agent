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

#define PAGE 4096

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
    if (fstat(fd, &st) != 0 || st.st_size < (off_t)sizeof(selfctx_hdr_t)) {
        close(fd);
        return 0;
    }
    selfctx_hdr_t h;
    ssize_t n = pread(fd, &h, sizeof(h), st.st_size - (off_t)sizeof(h));
    close(fd);
    if (n != (ssize_t)sizeof(h)) return 0;
    if (memcmp(h.magic, SELFCTX_MAGIC, 8) != 0) return 0;
    /* sanity: offsets must look sane */
    if (h.base_size == 0 || h.base_size > (uint64_t)st.st_size) return 0;
    if (h.cfg_off < h.base_size) return 0;
    if (h.cfg_off + h.cfg_size > h.kv_off) return 0;
    if (h.kv_off + h.kv_size > (uint64_t)st.st_size) return 0;
    if (hdr) *hdr = h;
    return 1;
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

char *selfctx_snapshot(const char *src, const char *dir, const char *out_dir,
                       const char *cfg, size_t cfg_len) {
    selfctx_hdr_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, SELFCTX_MAGIC, 8);

    int src_fd = open(src, O_RDONLY);
    if (src_fd < 0) return NULL;
    struct stat st;
    if (fstat(src_fd, &st) != 0) { close(src_fd); return NULL; }
    uint64_t src_size = (uint64_t)st.st_size;
    hdr.base_size = src_size;
    if (selfctx_detect(src, &hdr)) hdr.base_size = hdr.base_size; /* already set */
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

    /* layout: [base][pad][cfg][kv][state][salience][header] */
    hdr.cfg_off = align_up(hdr.base_size, PAGE);
    hdr.cfg_size = cfg ? (uint64_t)cfg_len : 0;
    hdr.kv_off = hdr.cfg_off + hdr.cfg_size;
    hdr.kv_size = (uint64_t)kst.st_size;
    hdr.st_off = hdr.kv_off + hdr.kv_size;
    hdr.st_size = have_st ? (uint64_t)sst.st_size : 0;
    hdr.sal_off = hdr.st_off + hdr.st_size;
    hdr.sal_size = have_sal ? (uint64_t)ast.st_size : 0;
    uint64_t total = hdr.sal_off + hdr.sal_size + sizeof(selfctx_hdr_t);

    /* n_pos + fp from the temp dir */
    {
        FILE *f = fopen(fp_path, "r");
        if (f) { if (fscanf(f, "%llx", (unsigned long long *)&hdr.sysfp) != 1) hdr.sysfp = 0; fclose(f); }
        struct stat rd;
        if (stat(rdy_path, &rd) == 0) hdr.flags |= SELFCTX_READY;
    }

    /* output name: <base>-YYYYMMDD-HHMMSS (strip a previous suffix) */
    char name[512];
    {
        const char *slash = strrchr(src, '/');
        const char *base = slash ? slash + 1 : src;
        char clean[512];
        snprintf(clean, sizeof(clean), "%s", base);
        size_t l = strlen(clean);
        if (l > 14 && clean[l-14] == '-' && is_ts_suffix(clean + l - 13))
            clean[l-14] = 0;   /* running a snapshot: next one keeps the base name */
        time_t t = time(NULL);
        struct tm tm;
        localtime_r(&t, &tm);
        snprintf(name, sizeof(name), "%s-%04d%02d%02d-%02d%02d", clean,
                 tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min);
    }
    char out_path[1200];
    snprintf(out_path, sizeof(out_path), "%s/%s", out_dir, name);

    int dst = open(out_path, O_RDWR | O_CREAT | O_TRUNC, 0755);
    if (dst < 0) { close(src_fd); return NULL; }
    if (ftruncate(dst, (off_t)total) != 0) goto fail;
    /* base code */
    if (copy_plain_at(src_fd, 0, dst, 0, (size_t)hdr.base_size) != 0) goto fail;
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
