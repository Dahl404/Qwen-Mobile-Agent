/* selfctx.h — self-contained context: the session lives INSIDE a copy of
 * the executable. A snapshot binary is:
 *
 *     [base ELF][internal blob][pad][cfg blob][KV][state][salience][header]
 *
 * The running process reads its own tail via /proc/self/exe (read-only is
 * allowed — only WRITE gets "Text file busy" on Android), and on graceful
 * exit writes a NEW snapshot. The running binary is never modified.
 * The internal blob carries the agent's source tree + vault (see intern.h);
 * the cfg blob carries the model path so a snapshot is portable: copy the
 * binary to another phone and it knows what model to load (and reprompts
 * if that path is missing there).
 *
 * Format: QMASNAP3 headers carry the internal blob + generation serial.
 * QMASNAP2 (pre-self-hosting) snapshots are still readable — the internal
 * fields read as zero. New snapshots are always QMASNAP3.
 */
#ifndef SELFCTX_H
#define SELFCTX_H

#include <stdint.h>
#include <stddef.h>

#define SELFCTX_MAGIC  "QMASNAP3"
#define SELFCTX_MAGIC2 "QMASNAP2"   /* older snapshots: no internal blob */
#define SELFCTX_READY  1u            /* flag: system-prompt ingest completed */

#define SELFCTX_HDR_LEN_NEW 136      /* sizeof the QMASNAP3 header */
#define SELFCTX_HDR_LEN_OLD 104      /* sizeof the QMASNAP2 header */

typedef struct {
    char     magic[8];
    uint64_t base_size;     /* bare executable size (code ends here) */
    uint64_t int_off, int_size;  /* internal tree blob (may be 0)     */
    uint64_t cfg_off, cfg_size;  /* model-path config blob (may be 0) */
    uint64_t kv_off, kv_size;    /* KV region (sparse, fixed size)   */
    uint64_t st_off, st_size;    /* state.bin copy                  */
    uint64_t sal_off, sal_size;  /* salience.bin copy               */
    uint64_t gen;               /* generation serial                */
    uint64_t build_time;        /* unix time of this build          */
    uint64_t sysfp;             /* fnv1a of the system prompt       */
    uint64_t n_pos;             /* session position                 */
    uint64_t flags;             /* SELFCTX_READY = ingest complete  */
} selfctx_hdr_t;

/* 1 if path has a valid session tail; fills *hdr (both QMASNAP3 and the
   older QMASNAP2 are recognized; missing fields read as zero). */
int  selfctx_detect(const char *path, selfctx_hdr_t *hdr);

/* Extract the embedded session from path into dir as kv.bin, state.bin,
 * salience.bin, sysfp.txt, sysready.txt. Returns 0 on success. */
int  selfctx_extract(const char *path, const selfctx_hdr_t *hdr, const char *dir);

/* Read the embedded model-path config blob. Returns a heap-allocated
 * NUL-terminated string, or NULL if there is none. Caller frees. */
char *selfctx_get_config(const char *path, const selfctx_hdr_t *hdr);

/* Write a dated snapshot of src with the session taken from dir and the
 * model-path cfg blob (NUL-terminated, cfg_len bytes) embedded. The
 * internal tree at intern_dir (may be NULL) is embedded as the internal
 * blob with the given generation serial. Snapshot goes to out_dir as
 * <base>-gen<N>-YYYYMMDD-HHMMSS (gen 0: <base>-YYYYMMDD-HHMMSS).
 * Returns a heap-allocated output path, or NULL on failure. */
char *selfctx_snapshot_gen(const char *src, const char *dir, const char *out_dir,
                           const char *cfg, size_t cfg_len,
                           const char *intern_dir, uint64_t gen);

/* Append a bare session header to fd (no session sections) describing the
 * internal blob already appended at [int_off, int_off+int_size). Used by
 * the build-time embed so selfctx_detect finds the internal tree. */
int  selfctx_append_bare(int fd, uint64_t base_size, uint64_t int_off,
                         uint64_t int_size, uint64_t gen, uint64_t build_time);

#endif /* SELFCTX_H */
