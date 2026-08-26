/* remote_main.c — headless boot + mission loop for qma's remote agent.
 * No UI, no chat: reads a task spec, runs the model as an autonomous tool
 * loop, collects data into the internal tree, packs a return binary.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <time.h>
#include <dirent.h>
#include "remote.h"
#include "lfm.h"
#include "selfctx.h"
#include "intern.h"

#define MAX_TURNS 128
#define IDS_CAP 131072

static lfm_t g_model;
static lfm_rs_t g_rs;
static int g_rs_ready = 0;
static char g_self_exe[1024] = "";
static char g_workdir[4096] = "";
static char g_internal_root[4096] = "";
static int g_model_loaded = 0;
static float *g_logits = NULL;   /* [N_VOCAB] persistent lm-head buffer */

static char *read_whole_file_remote(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return strdup("ERROR: cannot open");
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)sz + 1);
    size_t rd = buf ? fread(buf, 1, (size_t)sz, f) : 0;
    fclose(f);
    if (!buf) return strdup("ERROR: OOM");
    buf[rd] = 0;
    return buf;
}

const char *remote_internal_root(void) { return g_internal_root; }
const char *remote_collect_dir(void) {
    static char p[4096];
    snprintf(p, sizeof(p), "%s/collect", g_internal_root);
    return p;
}
void remote_log(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "[remote] "); vfprintf(stderr, fmt, ap); fprintf(stderr, "\n");
    va_end(ap);
}

/* boot: resolve self, detect platform, self-rebuild if the arch differs,
   extract the internal tree, load the embedded model. */
static int boot(int argc, char **argv) {
    ssize_t n = readlink("/proc/self/exe", g_self_exe, sizeof(g_self_exe) - 1);
    if (n > 0) g_self_exe[n] = 0;
    else snprintf(g_self_exe, sizeof(g_self_exe), "%s", argv[0]);

    /* platform id: stamp + rebuild if the binary was built elsewhere */
    {
        char pid[64] = "";
        qma_platform_id(pid, sizeof(pid));
        remote_log("platform: %s", pid);
        char stamped[96] = "";
        snprintf(stamped, sizeof(stamped), "%s.plat", g_self_exe);
        FILE *sf = fopen(stamped, "r");
        char have[64] = "";
        if (sf) { fgets(have, sizeof(have), sf); fclose(sf); }
        size_t hl = strlen(have);
        while (hl && (have[hl-1]=='\n'||have[hl-1]=='\r')) have[--hl] = 0;
        if (strcmp(have, pid) != 0) {
            remote_log("platform changed (%s != %s) — self-rebuilding", have[0]?have:"none", pid);
            /* rebuild from the embedded source, then exec the fresh binary */
            char root[1200], cmd[8192], out[65536];
            snprintf(root, sizeof(root), "%s.internal", g_self_exe);
            selfctx_hdr_t h;
            int fd = open(g_self_exe, O_RDONLY);
            if (fd >= 0 && selfctx_detect(g_self_exe, &h) && h.int_size > 0 &&
                intern_extract_region(fd, (off_t)h.int_off, h.int_size, root) == 0) {
                intern_build_cmd(cmd, sizeof(cmd), root, g_self_exe);
                int rc = sys_run_capture(cmd, 600, out, sizeof(out));
                if (rc == 0) {
                    FILE *sf = fopen(stamped, "w");
                    if (sf) { fprintf(sf, "%s\n", pid); fclose(sf); }
                    remote_log("rebuild ok — execing");
                    /* re-exec with the ORIGINAL argv (the fresh binary's
                       .plat stamp matches, so no rebuild loop) */
                    char **a2 = malloc(sizeof(char *) * (size_t)(argc + 1));
                    for (int i = 0; i < argc; i++) a2[i] = argv[i];
                    a2[0] = g_self_exe;
                    a2[argc] = NULL;
                    execv(g_self_exe, a2);
                }
                remote_log("rebuild failed (rc=%d): %.300s", rc, out);
            }
        }
    }

    /* extract internal tree (source + tools + collect/) */
    {
        selfctx_hdr_t h;
        if (selfctx_detect(g_self_exe, &h) && h.int_size > 0) {
            int fd = open(g_self_exe, O_RDONLY);
            if (fd >= 0) {
                if (intern_extract_region(fd, (off_t)h.int_off, h.int_size, g_internal_root) == 0) {
                    char c[1200];
                    snprintf(c, sizeof(c), "%s/collect", g_internal_root);
                    mkdir(c, 0755);
                }
                close(fd);
            }
        }
    }
    return 0;
}

/* load the embedded LFM weights (the binary IS the model) */
static int load_embedded_model(void) {
    selfctx_hdr_t h;
    if (!selfctx_detect(g_self_exe, &h) || h.model_size == 0) {
        remote_log("no embedded model — this is a bare base binary");
        return -1;
    }
    int fd = open(g_self_exe, O_RDONLY);
    if (fd < 0) return -1;
    size_t mapsz = (size_t)h.model_off + (size_t)h.model_size;
    uint8_t *map = mmap(NULL, mapsz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) return -1;
    char err[512] = "";
    memset(&g_model, 0, sizeof(g_model));
    if (lfm_load_map(&g_model, map, mapsz, h.model_off, h.model_size,
                     err, sizeof(err)) != 0) {
        remote_log("embedded model load failed: %s", err);
        munmap(map, mapsz);
        return -1;
    }
    g_model_loaded = 1;
    remote_log("embedded model loaded (%llu bytes): %u layers, %u embd",
               (unsigned long long)h.model_size, g_model.cfg.n_layer, g_model.cfg.n_embd);
    return 0;
}

/* generate a response from the current KV state, starting after the given
   prompt tokens. headless: the model emits text until im_end/eos. */
static float *logits_buf(void) {
    if (!g_logits) g_logits = malloc(sizeof(float) * g_model.n_vocab);
    return g_logits;
}

static void generate(char *out, size_t cap, int max_tokens) {
    lfm_samp_t sp;
    lfm_samp_init(&sp, (uint64_t)time(NULL), 0.7f, 40, 0.9f, 1.1f);
    size_t o = 0;
    out[0] = 0;
    for (int i = 0; i < max_tokens && o + 64 < cap; i++) {
        int cand[40]; float lgs[40];
        int nc = lfm_samp_candidates(g_model.n_vocab, logits_buf(), &sp, cand, lgs, 40);
        int tok = nc > 0 ? lfm_samp_pick(&sp, cand, lgs, nc, 0.9f) : 0;
        if (tok == (int)g_model.id_im_end || tok == (int)g_model.id_eos) break;
        char piece[128];
        int pl = lfm_detokenize(&g_model, &tok, 1, piece, sizeof(piece));
        if (pl > 0 && o + (size_t)pl < cap - 1) { memcpy(out + o, piece, (size_t)pl); o += (size_t)pl; out[o] = 0; }
        if (lfm_eval(&g_model, &g_rs, &tok, 1, logits_buf(), 4, 0, 0) != 0) break;
    }
}

/* pack the return binary: current binary + internal tree + KV/state, per
   the return directive. Returns the output path or NULL. */
static char *pack_return(const char *return_path) {
    /* build a snapshot: src=this binary, session dir carries KV/state */
    char sess[1200], kv[1200], st[1200];
    snprintf(sess, sizeof(sess), "%s.sess", g_self_exe);
    mkdir(sess, 0755);
    /* the LFM runstate persists via lfm_rs_save-style manual write: write
       the KV arena + n_pos to files (the runstate has no save fn in the
       lfm copy — serialize the mapped arena + n_pos here). */
    snprintf(kv, sizeof(kv), "%s/kv.bin", sess);
    snprintf(st, sizeof(st), "%s/state.bin", sess);
    {
        FILE *f = fopen(kv, "wb");
        if (f) {
            fwrite(g_rs.kv_map, 1, g_rs.kv_persist_bytes, f);
            fclose(f);
        }
        f = fopen(st, "wb");
        if (f) { int32_t np = g_rs.n_pos; fwrite(&np, 4, 1, f); fclose(f); }
    }
    char *out = selfctx_snapshot_gen(g_self_exe, sess, return_path, NULL, 0,
                                     g_internal_root[0] ? g_internal_root : NULL, 1,
                                     g_self_exe);
    remote_log("return binary packed: %s", out ? out : "(failed)");
    return out;
}


/* ---- memory: the agent's own always-attended KV (same corrected model
   as qma — linear LFM cache, nothing to evict; content eval'd into the KV
   persists and travels home in the return binary). Keys tracked in a small
   memory.json in the internal tree. ---- */
static char *memory_json_path(void) {
    static char p[1200];
    snprintf(p, sizeof(p), "%s/memory.json", g_internal_root);
    return p;
}

int remote_memory_write(const char *key, const char *content) {
    if (!key || !key[0] || !content || !content[0]) return -2;
    char msg[262144];
    snprintf(msg, sizeof(msg),
             "<|im_start|>system\n[MEMORY %s]\n%s<|im_end|>\n", key, content);
    int ids[IDS_CAP];
    int n = lfm_tokenize(&g_model, msg, ids, IDS_CAP);
    if (n <= 0) return -1;
    if (lfm_eval(&g_model, &g_rs, ids, n, NULL, 4, 0, 0) != 0) return -1;
    /* track the key (dedup) */
    FILE *f = fopen(memory_json_path(), "a+");
    if (f) {
        char line[512]; int found = 0;
        while (fgets(line, sizeof(line), f))
            if (strstr(line, key)) { found = 1; break; }
        if (!found) fprintf(f, "%s\n", key);
        fclose(f);
    }
    return n;
}

int remote_memory_append(const char *key, const char *content) {
    return remote_memory_write(key, content);   /* append == write for the KV */
}

char *remote_memory_list(void) {
    FILE *f = fopen(memory_json_path(), "r");
    if (!f) return strdup("(no memory entries)");
    char *out = malloc(8192); size_t o = 0; out[0] = 0;
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        size_t l = strlen(line);
        while (l && (line[l-1]=='\n'||line[l-1]=='\r')) line[--l] = 0;
        if (l && o + l + 2 < 8192) { memcpy(out + o, line, l); o += l; out[o++] = '\n'; }
    }
    fclose(f);
    return out;
}

/* generate tokens until a complete <|tool_call_start|>...</tool_call_end|>
   block lands in out, OR the model ends the turn (im_end/eos), OR the
   buffer fills. Returns 1 if a complete call is present. */
static int gen_until_call(char *out, size_t cap, size_t *olen) {
    lfm_samp_t sp;
    lfm_samp_init(&sp, (uint64_t)time(NULL) ^ (uint64_t)getpid(), 0.7f, 40, 0.9f, 1.15f);
    size_t o = 0;
    out[0] = 0;
    for (int i = 0; i < 8192 && o + 128 < cap; i++) {
        int cand[40]; float lgs[40];
        int nc = lfm_samp_candidates(g_model.n_vocab, logits_buf(), &sp, cand, lgs, 40);
        int tok = nc > 0 ? lfm_samp_pick(&sp, cand, lgs, nc, 0.9f) : 0;
        if (tok == (int)g_model.id_im_end || tok == (int)g_model.id_eos) break;
        char piece[128];
        int pl = lfm_detokenize(&g_model, &tok, 1, piece, sizeof(piece));
        if (pl > 0 && o + (size_t)pl < cap - 1) { memcpy(out + o, piece, (size_t)pl); o += (size_t)pl; out[o] = 0; }
        if (strstr(out, "<|tool_call_end|>")) { *olen = o; return 1; }
        if (lfm_eval(&g_model, &g_rs, &tok, 1, logits_buf(), 4, 0, 0) != 0) break;
    }
    *olen = o;
    return strstr(out, "<|tool_call_end|>") != NULL;
}

/* the autonomous mission loop: the model calls tools until done(). */
static void run_mission(const char *spec, int max_turns) {
    /* system prompt: role + mission + tool list */
    {
        char sysp[131072];
        snprintf(sysp, sizeof(sysp),
                 "<|im_start|>system\nYou are a remote agent of qma operating "
                 "autonomously on a foreign device. You have a mission and must "
                 "complete it without any human interaction.\n\n%s\n%s<|im_end|>\n",
                 remote_tools_header(), spec);
        int ids[IDS_CAP];
        int n = lfm_tokenize(&g_model, sysp, ids, IDS_CAP);
        if (n > 0) lfm_eval(&g_model, &g_rs, ids, n, NULL, 4, 0, 0);
    }
    const char *kick = "<|im_start|>user\nBegin. Work autonomously — call tools as "
                       "needed. When the mission is complete, call done() and give "
                       "your final report.<|im_end|>\n<|im_start|>assistant\n";
    {
        int ids[IDS_CAP];
        int n = lfm_tokenize(&g_model, kick, ids, IDS_CAP);
        if (n > 0) lfm_eval(&g_model, &g_rs, ids, n, logits_buf(), 4, 0, 0);
    }
    for (int turn = 0; turn < max_turns; turn++) {
        char buf[65536]; size_t bl = 0;
        int has_call = gen_until_call(buf, sizeof(buf), &bl);
        if (has_call) {
            const char *after = NULL;
            char name[128], args[262144];
            int pr = remote_parse_call(buf, &after, name, sizeof(name), args, sizeof(args));
            if (pr == 1) {
                remote_log("tool call: %s(%s)", name, args);
                char *result = NULL;
                remote_tool_dispatch(name, args, &result);
                /* fold the result back in LFM's native format */
                char resp[270000];
                snprintf(resp, sizeof(resp),
                         "<|tool_response_start|>\n%s\n<|tool_response_end|>\n"
                         "<|im_start|>assistant\n", result);
                free(result);
                int ids[IDS_CAP];
                int n = lfm_tokenize(&g_model, resp, ids, IDS_CAP);
                if (n > 0) lfm_eval(&g_model, &g_rs, ids, n, logits_buf(), 4, 0, 0);
                if (strcmp(name, "done") == 0) { remote_log("mission done"); break; }
                continue;
            }
        }
        /* no (complete) call and the turn ended: the model finished */
        if (bl > 0) remote_log("mission turn ended without a tool call (%.200s)", buf);
        break;
    }
}


int main(int argc, char **argv) {
    const char *task_file = NULL;
    const char *return_path = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--task") == 0 && i + 1 < argc) task_file = argv[++i];
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) return_path = argv[++i];
        else if (strcmp(argv[i], "--rebooted") == 0) { /* after self-rebuild */ }
    }
    boot(argc, argv);
    if (load_embedded_model() != 0) {
        fprintf(stderr, "remote: bare binary — use qma to dispatch (embed model + task)\n");
        return 1;
    }

    if (lfm_rs_init_kv(&g_rs, 8192, NULL, 0) != 0) { fprintf(stderr, "rs init\n"); return 1; }
    logits_buf();
    g_rs_ready = 1;

    /* read the task spec: embedded cfg blob or --task file */
    char *spec = NULL;
    selfctx_hdr_t h;
    if (selfctx_detect(g_self_exe, &h) && h.cfg_size > 0) {
        spec = selfctx_get_config(g_self_exe, &h);
    }
    if (!spec && task_file) {
        FILE *f = fopen(task_file, "rb");
        if (f) { fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
                 spec = malloc((size_t)sz + 1); size_t rd = fread(spec, 1, (size_t)sz, f);
                 spec[rd] = 0; fclose(f); }
    }
    if (!spec) { fprintf(stderr, "remote: no task spec\n"); return 1; }
    remote_log("task spec: %.200s", spec);

    /* base KV: role + mission (the system prompt stays for the whole run) */
    {
        char base[65536];
        snprintf(base, sizeof(base), "<|im_start|>system\nYou are a remote agent of qma, operating autonomously. %s<|im_end|>\n",
                 spec);
        int ids[IDS_CAP];
        int n = lfm_tokenize(&g_model, base, ids, IDS_CAP);
        if (n > 0) lfm_eval(&g_model, &g_rs, ids, n, NULL, 4, 0, 0);
    }

    /* mission: autonomous tool loop (headless) */
    run_mission(spec, 64);

    /* collect directive: grab remote paths into /internal/collect/ so they
       travel home (a direct copy — no model needed) */
    {
        const char *cd = remote_collect_dir();
        const char *cl = strstr(spec, "\"collect\"");
        if (cl) {
            const char *p = cl;
            while ((p = strstr(p, "\"")) != NULL) {
                p++;
                const char *e = strchr(p, '\"');
                if (!e) break;
                if ((size_t)(e - p) > 4 && strncmp(e - 4, ".txt", 4) != 0 &&
                    !strchr(p, ':')) {
                    char path[1024];
                    size_t l = (size_t)(e - p);
                    if (l >= sizeof(path)) l = sizeof(path) - 1;
                    memcpy(path, p, l); path[l] = 0;
                    if (path[0] == '/') {
                        char dst[1200];
                        const char *slash = strrchr(path, '/');
                        snprintf(dst, sizeof(dst), "%s/%s", cd, slash ? slash + 1 : path);
                        char *content = read_whole_file_remote(path);
                        if (content && strncmp(content, "ERROR:", 6) != 0) {
                            FILE *f = fopen(dst, "w");
                            if (f) { fwrite(content, 1, strlen(content), f); fclose(f);
                                     remote_log("collected %s", path); }
                        }
                        free(content);
                    }
                }
                p = e;
            }
        }
    }
    /* final report: whatever the assistant last said, plus the collect
       manifest — written into the internal tree so it carries home */
    {
        char rp[1200];
        snprintf(rp, sizeof(rp), "%s/collect/report.txt", g_internal_root);
        FILE *f = fopen(rp, "w");
        if (f) {
            fprintf(f, "REMOTE MISSION REPORT\n====================\n");
            fprintf(f, "host: ");
            char pid[64]; qma_platform_id(pid, sizeof(pid)); fprintf(f, "%s\n", pid);
            fprintf(f, "mission spec: %.400s\n\n", spec);
            fprintf(f, "collected files: \n");
            char ml[1200];
            snprintf(ml, sizeof(ml), "%s", remote_collect_dir());
            DIR *d = opendir(ml);
            if (d) { struct dirent *e; while ((e = readdir(d)) != NULL)
                        if (e->d_name[0] != '.') fprintf(f, "  %s\n", e->d_name);
                     closedir(d); }
            fclose(f);
        }
    }

    /* return per directive (shell: --out is a DIRECTORY the packed binary
       is written into, or default next to the binary) */
    char defpath[1200];
    if (!return_path) {
        snprintf(defpath, sizeof(defpath), "%s.returned", g_self_exe);
        return_path = defpath;
    }
    mkdir(return_path, 0755);
    char *out = pack_return(return_path);
    free(spec);
    return out ? 0 : 1;
}
