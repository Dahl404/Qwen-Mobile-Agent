/* intern.c — embedded internal tree (see intern.h).
 * Self-contained: no engine dependencies, so it can also be built into
 * standalone tests (work/test_intern.c).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <poll.h>
#include <signal.h>
#include "intern.h"
#include "selfctx.h"

#define INTERN_VERSION 1u
#define HDR_LEN 40   /* magic8 + ver4 + gen8 + bt8 + n4 + size8 */
#define PAGE 4096

static uint64_t align_up(uint64_t v, uint64_t a) {
    return (v + a - 1) & ~(a - 1);
}

/* ---------------- writing ---------------- */

static int wr_all(int fd, const void *buf, size_t n) {
    const uint8_t *p = buf;
    while (n > 0) {
        ssize_t w = write(fd, p, n);
        if (w <= 0) return -1;
        p += w; n -= (size_t)w;
    }
    return 0;
}

static int wr_u16(int fd, uint16_t v) { uint16_t x = v; return wr_all(fd, &x, 2); }
static int wr_u32(int fd, uint32_t v) { uint32_t x = v; return wr_all(fd, &x, 4); }
static int wr_u64(int fd, uint64_t v) { uint64_t x = v; return wr_all(fd, &x, 8); }

/* build artifacts / non-source files to skip when embedding the repo at
   build time (and anything the agent shouldn't carry: binaries, tmp, the
   internal blob itself). Checks EVERY path component, not just the base. */
static int junk_path(const char *rel) {
    /* component-level skips (whole dirs / prefixes) */
    const char *p = rel;
    while (*p) {
        const char *e = strchr(p, '/');
        size_t cl = e ? (size_t)(e - p) : strlen(p);
        char comp[256];
        if (cl >= sizeof(comp)) { p = e ? e + 1 : p + cl; continue; }
        memcpy(comp, p, cl);
        comp[cl] = 0;
        if (strcmp(comp, ".git") == 0) return 1;
        if (strcmp(comp, "work") == 0) return 1;
        if (strncmp(comp, "qma-", 4) == 0) return 1;          /* snapshots */
        if (strncmp(comp, "qma-gen", 7) == 0) return 1;
        if (strncmp(comp, ".internal-", 10) == 0) return 1;
        if (strcmp(comp, "internal.blob") == 0) return 1;
        if (strcmp(comp, "selfbuild") == 0) return 1;
        p = e ? e + 1 : p + cl;
    }
    /* basename checks */
    const char *base = strrchr(rel, '/');
    base = base ? base + 1 : rel;
    size_t l = strlen(base);
    if (strcmp(base, "qma") == 0) return 1;                   /* the binary */
    if (l > 3 && strcmp(base + l - 3, ".4k") == 0) return 1;
    if (l > 4 && strcmp(base + l - 4, ".4k.tmp") == 0) return 1;
    if (l > 2 && strcmp(base + l - 2, ".o") == 0) return 1;
    if (l > 2 && strcmp(base + l - 2, ".a") == 0) return 1;
    if (l > 4 && strcmp(base + l - 4, ".tmp") == 0) return 1;
    return 0;
}

static int embed_one(int fd, const char *root, const char *rel,
                     int skip_junk, uint32_t *n_files) {
    char path[4096];
    if (rel[0])
        snprintf(path, sizeof(path), "%s/%s", root, rel);
    else
        snprintf(path, sizeof(path), "%s", root);
    struct stat st;
    if (lstat(path, &st) != 0) return 0;
    if (S_ISDIR(st.st_mode)) {
        DIR *d = opendir(path);
        if (!d) return 0;
        struct dirent *e;
        while ((e = readdir(d)) != NULL) {
            if (strcmp(e->d_name, ".") == 0 || strcmp(e->d_name, "..") == 0) continue;
            char sub[4096];
            if (rel[0]) snprintf(sub, sizeof(sub), "%s/%s", rel, e->d_name);
            else        snprintf(sub, sizeof(sub), "%s", e->d_name);
            embed_one(fd, root, sub, skip_junk, n_files);
        }
        closedir(d);
        return 0;
    }
    if (!S_ISREG(st.st_mode)) return 0;   /* symlinks, fifos: skip */
    if (skip_junk && junk_path(rel)) return 0;
    size_t pl = strlen(rel);
    if (pl == 0 || pl > 65535) return 0;
    uint64_t dl = (uint64_t)st.st_size;
    if (wr_u16(fd, (uint16_t)pl) != 0) return -1;
    if (wr_all(fd, rel, pl) != 0) return -1;
    if (wr_u64(fd, dl) != 0) return -1;
    int in = open(path, O_RDONLY);
    if (in < 0) return -1;
    char buf[65536];
    uint64_t left = dl;
    while (left > 0) {
        size_t want = left < sizeof(buf) ? (size_t)left : sizeof(buf);
        ssize_t n = read(in, buf, want);
        if (n <= 0) { close(in); return -1; }
        if (wr_all(fd, buf, (size_t)n) != 0) { close(in); return -1; }
        left -= (size_t)n;
    }
    close(in);
    (*n_files)++;
    return 0;
}

int intern_embed_dir(int fd, const char *root, int skip_junk,
                     uint64_t gen, uint64_t build_time) {
    off_t start = lseek(fd, 0, SEEK_CUR);
    if (start < 0) return -1;
    if (wr_all(fd, INTERN_MAGIC, 8) != 0) return -1;
    if (wr_u32(fd, INTERN_VERSION) != 0) return -1;
    if (wr_u64(fd, gen) != 0) return -1;
    if (wr_u64(fd, build_time) != 0) return -1;
    if (wr_u32(fd, 0) != 0) return -1;                 /* n_files (patched) */
    if (wr_u64(fd, 0) != 0) return -1;                 /* blob_size (patched) */
    uint32_t n_files = 0;
    if (embed_one(fd, root, "", skip_junk, &n_files) != 0) return -1;
    off_t end = lseek(fd, 0, SEEK_CUR);
    if (end < 0) return -1;
    uint64_t blob_size = (uint64_t)(end - start);
    /* patch n_files (at start+28) and blob_size (at start+32) */
    if (pwrite(fd, &n_files, 4, start + 28) != 4) return -1;
    if (pwrite(fd, &blob_size, 8, start + 32) != 8) return -1;
    return 0;
}

static int copy_file(const char *src, const char *dst) {
    int in = open(src, O_RDONLY);
    if (in < 0) return -1;
    int out = open(dst, O_WRONLY | O_CREAT | O_TRUNC, 0755);
    if (out < 0) { close(in); return -1; }
    char buf[65536];
    for (;;) {
        ssize_t n = read(in, buf, sizeof(buf));
        if (n <= 0) break;
        if (wr_all(out, buf, (size_t)n) != 0) { close(in); close(out); return -1; }
    }
    close(in); close(out);
    return 0;
}

int intern_embed_binary(const char *tree, const char *outfile,
                        uint64_t gen, uint64_t build_time, int skip_junk) {
    char tmp[1200];
    snprintf(tmp, sizeof(tmp), "%s.embedded.tmp", outfile);
    if (copy_file(outfile, tmp) != 0) return -1;
    struct stat st;
    if (stat(tmp, &st) != 0) { unlink(tmp); return -1; }
    uint64_t base_size = (uint64_t)st.st_size;
    uint64_t int_off = align_up(base_size, PAGE);
    int fd = open(tmp, O_WRONLY);   /* NOT O_APPEND: it would force pwrite patches to the end */
    if (fd < 0) { unlink(tmp); return -1; }
    if (ftruncate(fd, (off_t)int_off) != 0) { close(fd); unlink(tmp); return -1; }
    if (lseek(fd, (off_t)int_off, SEEK_SET) < 0) { close(fd); unlink(tmp); return -1; }
    int rc = intern_embed_dir(fd, tree, skip_junk, gen, build_time);
    if (rc != 0) { close(fd); unlink(tmp); return -1; }
    off_t end = lseek(fd, 0, SEEK_END);
    uint64_t int_size = (uint64_t)(end - (off_t)int_off);
    /* bare selfctx header at the tail so detect() finds the internal tree */
    if (selfctx_append_bare(fd, base_size, int_off, int_size, gen, build_time) != 0) {
        close(fd); unlink(tmp); return -1;
    }
    fsync(fd);
    close(fd);
    if (rename(tmp, outfile) != 0) { unlink(tmp); return -1; }
    return 0;
}

/* ---------------- reading ---------------- */

static int rd_all(int fd, void *buf, size_t n, off_t *cur) {
    uint8_t *p = buf;
    while (n > 0) {
        ssize_t r = pread(fd, p, n, *cur);
        if (r <= 0) return -1;
        p += r; n -= (size_t)r; *cur += r;
    }
    return 0;
}
static int rd_u16(int fd, uint16_t *v, off_t *cur) {
    uint8_t b[2];
    if (rd_all(fd, b, 2, cur) != 0) return -1;
    *v = (uint16_t)(b[0] | (b[1] << 8));
    return 0;
}
static int rd_u32(int fd, uint32_t *v, off_t *cur) {
    uint8_t b[4];
    if (rd_all(fd, b, 4, cur) != 0) return -1;
    *v = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return 0;
}
static int rd_u64(int fd, uint64_t *v, off_t *cur) {
    uint32_t lo, hi;
    if (rd_u32(fd, &lo, cur) != 0) return -1;
    if (rd_u32(fd, &hi, cur) != 0) return -1;
    *v = (uint64_t)lo | ((uint64_t)hi << 32);
    return 0;
}

/* path safety: reject absolute paths and any ".." component */
static int safe_path(const char *p, size_t n) {
    if (n == 0 || p[0] == '/') return 0;
    size_t i = 0;
    while (i < n) {
        size_t j = i;
        while (j < n && p[j] != '/') j++;
        size_t cl = j - i;
        if (cl == 2 && p[i] == '.' && p[i + 1] == '.') return 0;
        i = j + 1;
    }
    return 1;
}

static void mkdirs(const char *path) {
    char tmp[4096];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char *q = tmp + 1; *q; q++) {
        if (*q == '/') { *q = 0; mkdir(tmp, 0700); *q = '/'; }
    }
    mkdir(tmp, 0700);
}

int intern_extract_region(int fd, off_t off, uint64_t size, const char *dest_dir) {
    if (size < HDR_LEN) return -1;
    off_t cur = off;
    char magic[8];
    if (rd_all(fd, magic, 8, &cur) != 0) return -1;
    if (memcmp(magic, INTERN_MAGIC, 8) != 0) return -1;
    uint32_t version, n_files;
    uint64_t gen, build_time, blob_size;
    if (rd_u32(fd, &version, &cur) != 0) return -1;
    if (rd_u64(fd, &gen, &cur) != 0) return -1;
    if (rd_u64(fd, &build_time, &cur) != 0) return -1;
    if (rd_u32(fd, &n_files, &cur) != 0) return -1;
    if (rd_u64(fd, &blob_size, &cur) != 0) return -1;
    if (version != INTERN_VERSION || blob_size != size) return -1;
    char path[4096];
    char buf[65536];
    for (uint32_t i = 0; i < n_files; i++) {
        uint16_t pl;
        uint64_t dl;
        if (rd_u16(fd, &pl, &cur) != 0) return -1;
        if (pl == 0 || pl >= sizeof(path)) return -1;
        if (rd_all(fd, path, pl, &cur) != 0) return -1;
        path[pl] = 0;
        if (!safe_path(path, pl)) return -1;
        if (rd_u64(fd, &dl, &cur) != 0) return -1;
        /* full path under dest_dir */
        char full[4200];
        snprintf(full, sizeof(full), "%s/%s", dest_dir, path);
        /* parent dirs */
        char *slash = strrchr(full, '/');
        if (slash) { *slash = 0; mkdirs(full); *slash = '/'; }
        int out = open(full, O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (out < 0) return -1;
        uint64_t left = dl;
        int bad = 0;
        while (left > 0) {
            size_t want = left < sizeof(buf) ? (size_t)left : sizeof(buf);
            if (rd_all(fd, buf, want, &cur) != 0) { bad = 1; break; }
            if (wr_all(out, buf, want) != 0) { bad = 1; break; }
            left -= want;
        }
        close(out);
        if (bad) return -1;
    }
    return 0;
}

int intern_extract_file(const char *blob_path, const char *dest_dir) {
    int fd = open(blob_path, O_RDONLY);
    if (fd < 0) return -1;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < (off_t)HDR_LEN) { close(fd); return -1; }
    int rc = intern_extract_region(fd, 0, (uint64_t)st.st_size, dest_dir);
    close(fd);
    return rc;
}

int intern_probe(const char *blob_path, uint64_t *gen, uint64_t *build_time) {
    int fd = open(blob_path, O_RDONLY);
    if (fd < 0) return 0;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < (off_t)HDR_LEN) { close(fd); return 0; }
    off_t cur = 0;
    char magic[8];
    uint32_t version;
    uint64_t g, bt;
    int ok = rd_all(fd, magic, 8, &cur) == 0 &&
             memcmp(magic, INTERN_MAGIC, 8) == 0 &&
             rd_u32(fd, &version, &cur) == 0 &&
             rd_u64(fd, &g, &cur) == 0 &&
             rd_u64(fd, &bt, &cur) == 0;
    close(fd);
    if (!ok) return 0;
    if (gen) *gen = g;
    if (build_time) *build_time = bt;
    return 1;
}

/* ---------------- manifest + stats ---------------- */

static int stat_one(const char *root, const char *rel, uint32_t *n, uint64_t *bytes) {
    char path[4096];
    if (rel[0]) snprintf(path, sizeof(path), "%s/%s", root, rel);
    else        snprintf(path, sizeof(path), "%s", root);
    struct stat st;
    if (lstat(path, &st) != 0) return 0;
    if (S_ISDIR(st.st_mode)) {
        DIR *d = opendir(path);
        if (!d) return 0;
        struct dirent *e;
        while ((e = readdir(d)) != NULL) {
            if (strcmp(e->d_name, ".") == 0 || strcmp(e->d_name, "..") == 0) continue;
            char sub[4096];
            if (rel[0]) snprintf(sub, sizeof(sub), "%s/%s", rel, e->d_name);
            else        snprintf(sub, sizeof(sub), "%s", e->d_name);
            stat_one(root, sub, n, bytes);
        }
        closedir(d);
        return 0;
    }
    if (S_ISREG(st.st_mode)) { (*n)++; *bytes += (uint64_t)st.st_size; }
    return 0;
}

int intern_stats(const char *root, uint32_t *n_files, uint64_t *bytes) {
    uint32_t n = 0;
    uint64_t b = 0;
    stat_one(root, "", &n, &b);
    if (n_files) *n_files = n;
    if (bytes) *bytes = b;
    return 0;
}

int intern_log(const char *root, const char *fmt, ...) {
    char path[4200];
    snprintf(path, sizeof(path), "%s/VERSIONS.md", root);
    FILE *f = fopen(path, "a");
    if (!f) return -1;
    va_list ap;
    va_start(ap, fmt);
    vfprintf(f, fmt, ap);
    va_end(ap);
    fclose(f);
    return 0;
}

/* ---------------- rebuild ---------------- */

void intern_build_cmd(char *buf, size_t cap, const char *src, const char *out_bin) {
    static const char *files[] = {
        "gguf.c", "tokenizer.c", "nn.c", "quants.c", "sampler.c",
        "ecache.c", "q8k.c", "kvq.c", "intern.c",
        "json.c", "toolparse.c", "tools.c",
        "thermal.c", "selfctx.c", "agent.c",
    };
    snprintf(buf, cap,
        "clang -O3 -std=c99 -Wall -Wextra -Wno-unused-parameter "
        "-mcpu=native -ffast-math -fomit-frame-pointer -pthread -o %s",
        out_bin);
    size_t o = strlen(buf);
    for (size_t i = 0; i < sizeof(files) / sizeof(files[0]); i++) {
        int w = snprintf(buf + o, cap - o, " %s/%s", src, files[i]);
        if (w < 0 || (size_t)w >= cap - o) break;
        o += (size_t)w;
    }
    snprintf(buf + o, cap - o, " -lm -pthread");
}

int sys_run_capture(const char *cmd, int timeout_s, char *out, size_t cap) {
    int pipefd[2];
    if (pipe(pipefd) != 0) return -1;
    pid_t pid = fork();
    if (pid < 0) { close(pipefd[0]); close(pipefd[1]); return -1; }
    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], 1);
        dup2(pipefd[1], 2);
        close(pipefd[1]);
        execl("/data/data/com.termux/files/usr/bin/bash", "bash", "-c", cmd, (char *)NULL);
        _exit(127);
    }
    close(pipefd[1]);
    size_t o = 0;
    out[0] = 0;
    struct pollfd pfd = { pipefd[0], POLLIN, 0 };
    double deadline = (double)time(NULL) + (double)timeout_s;
    int timed_out = 0;
    for (;;) {
        double now = (double)time(NULL);
        int remaining = now >= deadline ? 0 : (int)(deadline - now);
        int pr = poll(&pfd, 1, remaining * 1000);
        if (pr < 0) { if (errno == EINTR) continue; break; }
        if (pr == 0) { timed_out = 1; break; }
        ssize_t n = read(pipefd[0], out + o, cap - 1 - o);
        if (n <= 0) break;
        o += (size_t)n;
        out[o] = 0;
    }
    close(pipefd[0]);
    int status = 0;
    if (timed_out) {
        kill(pid, SIGKILL);
        waitpid(pid, NULL, 0);
        status = -1;
    } else {
        waitpid(pid, &status, 0);
    }
    out[cap - 1] = 0;
    return timed_out ? -1 : (WIFEXITED(status) ? WEXITSTATUS(status) : 127);
}
