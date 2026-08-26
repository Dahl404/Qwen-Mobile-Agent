/* intern.h — the embedded internal tree (self-hosting source + agent vault).
 *
 * qma carries its own source (and anything the agent collects) INSIDE the
 * binary as a QMAINT01 blob appended to the executable tail (see selfctx).
 * At build time `make` embeds the repo tree; at boot the blob is extracted
 * to <session>/internal/ (mounted as /internal/ for the agent's tools); on
 * clean exit qma recompiles itself from <session>/internal/src and re-embeds
 * the whole tree — source, collected tools/data, VERSIONS.md, rebuild.log —
 * into the next generation.
 *
 * Blob layout (little-endian, no padding):
 *   magic[8] "QMAINT01"
 *   u32 version
 *   u64 gen            serial of the generation that produced this tree
 *   u64 build_time     unix time of that build
 *   u32 n_files
 *   u64 blob_size      total blob bytes (header + entries + data)
 *   entries: u16 path_len, path bytes, u64 data_len, data bytes
 */
#ifndef INTERN_H
#define INTERN_H

#include <stdint.h>
#include <stddef.h>
#include <sys/types.h>

#define INTERN_MAGIC "QMAINT01"

/* Append the internal tree at root (recursive) to fd at its current end as a
   QMAINT01 blob. skip_junk filters build artifacts (.git, work/, binaries). */
int intern_embed_dir(int fd, const char *root, int skip_junk,
                     uint64_t gen, uint64_t build_time);

/* Copy `outfile` to a temp, append the internal blob, rename over outfile.
   Used by the build step: a running executable cannot be opened for writing
   on Linux (ETXTBSY), so we never touch the running file's inode directly. */
int intern_embed_binary(const char *tree, const char *outfile,
                        uint64_t gen, uint64_t build_time, int skip_junk);

/* Parse + extract a blob from fd at [off, off+size) into dest_dir, recreating
   the tree (mkdir -p). Rejects absolute and ".." paths. */
int intern_extract_region(int fd, off_t off, uint64_t size, const char *dest_dir);

/* Extract a blob file into dest_dir. */
int intern_extract_file(const char *blob_path, const char *dest_dir);

/* Read a blob header: gen + build_time. Returns 1 on success. */
int intern_probe(const char *blob_path, uint64_t *gen, uint64_t *build_time);

/* Count regular files + total bytes under root. */
int intern_stats(const char *root, uint32_t *n_files, uint64_t *bytes);

/* Append a formatted line to <root>/VERSIONS.md (creates it if missing). */
int intern_log(const char *root, const char *fmt, ...);

/* Recursively delete a directory tree (prune stale mounts). */
int intern_rmtree(const char *path);

/* Build the clang command that compiles src_dir's .c files into out_bin
   (same flags and file list as the Makefile, plus intern.c itself). */
void intern_build_cmd(char *buf, size_t cap, const char *src_dir, const char *out_bin);

/* Run cmd via bash -c, capturing stdout+stderr into out (cap-1 max bytes),
   killing the child after timeout_s. Returns exit status, -1 on timeout,
   127 on spawn failure. */
int sys_run_capture(const char *cmd, int timeout_s, char *out, size_t cap);

#endif /* INTERN_H */
