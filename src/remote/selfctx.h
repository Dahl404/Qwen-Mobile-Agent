/* selfctx.h — self-contained context: the session lives INSIDE a copy of
 * the executable. mobile-agent-2's binary carries the ENTIRE agent:
 *
 *   [base ELF][model blob][internal blob][pad][cfg][KV][state][salience][header]
 *
 * The model blob is the 4K-aligned GGUF (the neural net) — the binary IS
 * the model. The internal blob carries the agent's source tree + vault;
 * the cfg blob carries the model path (used only for the initial breed).
 * The running process reads its own tail via /proc/self/exe.
 *
 * Format: QMASNAP4. The model section is embedded by the breed step on
 * first boot (no model in the binary → align the config's model → breed a
 * new base executable → exec it), and copied forward by every generation.
 */
#ifndef SELFCTX_H
#define SELFCTX_H

#include <stdint.h>
#include <stddef.h>

#define SELFCTX_MAGIC  "QMASNAP4"
#define SELFCTX_READY  1u            /* flag: system-prompt ingest completed */

#define SELFCTX_HDR_LEN 152          /* sizeof the QMASNAP4 header */

typedef struct {
    char     magic[8];
    uint64_t base_size;     /* bare executable size (code ends here) */
    uint64_t model_off, model_size;  /* embedded neural net (aligned GGUF) */
    uint64_t int_off, int_size;      /* internal tree blob                */
    uint64_t cfg_off, cfg_size;      /* model-path config blob            */
    uint64_t kv_off, kv_size;        /* KV region                         */
    uint64_t st_off, st_size;        /* state.bin copy                    */
    uint64_t sal_off, sal_size;      /* salience.bin copy                 */
    uint64_t gen;               /* generation serial                */
    uint64_t build_time;        /* unix time of this build          */
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

/* Write a dated snapshot of src with the session taken from dir and the
 * model-path cfg blob (NUL-terminated, cfg_len bytes) embedded. The
 * internal tree at intern_dir (may be NULL) is embedded as the internal
 * blob with the given generation serial. Snapshot goes to out_dir as
 * <base>-gen<N>-YYYYMMDD-HHMMSS (gen 0: <base>-YYYYMMDD-HHMMSS).
 * Returns a heap-allocated output path, or NULL on failure. */
char *selfctx_snapshot_gen(const char *src, const char *dir, const char *out_dir,
                           const char *cfg, size_t cfg_len,
                           const char *intern_dir, uint64_t gen,
                           const char *model_src);
/* model_src: binary to copy the embedded model section from (NULL = src).
   The exit pipeline passes the RUNNING binary (which carries the net) so a
   freshly rebuilt generation keeps the model embedded. */

/* Append a bare session header to fd (no session sections) describing the
 * internal blob already appended at [int_off, int_off+int_size). Used by
 * the build-time embed so selfctx_detect finds the internal tree. */
int  selfctx_append_bare(int fd, uint64_t base_size, uint64_t int_off,
                         uint64_t int_size, uint64_t gen, uint64_t build_time);

/* Breed a new base executable: copy `base` (a freshly built binary with an
 * internal blob but no embedded model), insert the 4K-aligned GGUF as the
 * model section, rewrite the tail header, and rename over `outfile`.
 * Returns 0 on success. Used once on first boot before exec'ing. */
int  selfctx_embed_model(const char *base, const char *model_file,
                         const char *outfile);

#endif /* SELFCTX_H */
