/* selfctx.h — self-contained context: the session lives INSIDE a copy of
 * the executable. A snapshot binary is:
 *
 *     [base ELF][pad][cfg blob][KV region][state][salience][header]
 *
 * The running process reads its own tail via /proc/self/exe (read-only is
 * allowed — only WRITE gets "Text file busy" on Android), and on graceful
 * exit writes a NEW dated snapshot. The running binary is never modified.
 * The cfg blob carries the model path so a snapshot is portable: copy the
 * binary to another phone and it knows what model to load (and reprompts
 * if that path is missing there).
 */
#ifndef SELFCTX_H
#define SELFCTX_H

#include <stdint.h>
#include <stddef.h>

#define SELFCTX_MAGIC  "QMASNAP2"
#define SELFCTX_READY  1u     /* flag: system-prompt ingest completed */

typedef struct {
    char     magic[8];
    uint64_t base_size;     /* bare executable size (code ends here) */
    uint64_t cfg_off, cfg_size;  /* model-path config blob (may be 0) */
    uint64_t kv_off, kv_size;    /* KV region (sparse, fixed size)   */
    uint64_t st_off, st_size;    /* state.bin copy                  */
    uint64_t sal_off, sal_size;  /* salience.bin copy               */
    uint64_t sysfp;             /* fnv1a of the system prompt       */
    uint64_t n_pos;             /* session position                 */
    uint64_t flags;             /* SELFCTX_READY = ingest complete  */
} selfctx_hdr_t;

/* 1 if path has a valid session tail; fills *hdr. */
int  selfctx_detect(const char *path, selfctx_hdr_t *hdr);

/* Extract the embedded session from path into dir as kv.bin, state.bin,
 * salience.bin, sysfp.txt, sysready.txt. Returns 0 on success. */
int  selfctx_extract(const char *path, const selfctx_hdr_t *hdr, const char *dir);

/* Read the embedded model-path config blob. Returns a heap-allocated
 * NUL-terminated string, or NULL if there is none. Caller frees. */
char *selfctx_get_config(const char *path, const selfctx_hdr_t *hdr);

/* Write a dated snapshot of src (the running binary) with the session
 * taken from dir and the model-path cfg blob (NUL-terminated, cfg_len
 * bytes) embedded. Snapshot goes to out_dir as <base>-YYYYMMDD-HHMMSS.
 * Returns a heap-allocated output path, or NULL on failure. */
char *selfctx_snapshot(const char *src, const char *dir, const char *out_dir,
                       const char *cfg, size_t cfg_len);

#endif /* SELFCTX_H */
