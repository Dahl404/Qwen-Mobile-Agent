/* agent.c — qwen mobile agent: engine + agent loop + minimal UI in ONE binary.
 *
 *   qma [options]              interactive agent (pi-agent style)
 *   qma -p "prompt"            one-shot
 *
 * Turn flow (per user message):
 *   render delta -> eval into the continual KV -> generate (streaming:
 *   thinking dim, content normal, tool calls visible) -> if tool calls:
 *   execute each, fold results back as <tool_response>, generate again;
 *   else: text-only response — the turn ends (pi loop: wait on text).
 *
 * The session is a single persistent KV:
 *   <session>/kv.bin (file-backed) + state.bin + salience.bin + sysfp.txt.
 * On resume, if the system prompt hash changed, the stale KV is wiped —
 * old sessions can never pollute a new one.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <ctype.h>
#include <time.h>
#include <sys/stat.h>
#include <errno.h>
#include <termios.h>
#include <poll.h>
#include <pthread.h>
#include "qma.h"
#include "json.h"
#include "toolparse.h"
#include "tools.h"
#include "grammar.h"
#include "thermal.h"
#include "selfctx.h"

static const char *DEFAULT_MODEL = "/data/data/com.termux/files/home/projects/models/qwen3.6:35b:a3b-q4km.gguf";

/* ---- config ---- */
static int g_threads = 8, g_ctx = 65536, g_prefetch = 4;
static float g_temp = 0.7f, g_top_p = 0.95f, g_repeat = 1.05f;
static int g_top_k = 20;
static float g_eos_penalty = 5.0f;
static int g_enable_thinking = 1;
static int g_ecache_mb = 1024;
static int g_use_color = 0;
static int g_reset = 0;

static double now_s2(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static int g_selfctx = 0;         /* snapshot mode (no --session): context lives in a dated copy of the binary */
static char g_self_exe[1024] = ""; /* running binary path (readlink /proc/self/exe) */
static char g_session_dir[1024] = "";
static char g_workdir[4096] = "";
static const char *g_model_path = NULL;
static char g_cfg_path[1024] = "";       /* ~/.qma/config — model path memory */
static char g_model_path_buf[4096] = ""; /* resolved model path (stable storage) */

/* ---- engine state ---- */
static qma_t g_model;
static runstate_t g_rs;
static int g_rs_ready = 0;
static volatile sig_atomic_t g_shutdown = 0;
static volatile sig_atomic_t g_stop_turn = 0;
static volatile sig_atomic_t g_snoop_reset = 0;   /* /reset typed mid-generation */
static int g_stdin_tty = 0;   /* ESC snoop only on a real terminal */

#define MAX_TURNS  50   /* pi-style turn budget: the only runaway guard */

static char g_system_prompt[262144];   /* tools header + agent behavior */

/* ---- terminal helpers ---- */
static void color(const char *code) {
    if (g_use_color) fputs(code, stdout);
}
static void c_reset(void)   { color("\x1b[0m"); }
static void c_dim(void)     { color("\x1b[2m"); }
static void c_cyan(void)    { color("\x1b[1;36m"); }
static void c_green(void)   { color("\x1b[1;32m"); }
static void c_yellow(void)  { color("\x1b[1;33m"); }
static void c_red(void)     { color("\x1b[1;31m"); }

static void out(const char *s) {
    fputs(s, stdout);
    fflush(stdout);
}

/* ---- ESC snoop: cancel generation and return to the user turn ----
 * While the model is generating, stdin is switched to a raw-ish mode
 * (ICANON|ECHO off, ISIG kept so Ctrl-C still delivers SIGINT) and a
 * background thread polls it. A bare ESC sets g_stop_turn; the generation
 * loop breaks, agent_generate closes the assistant frame, and agent_turn
 * returns to the prompt. Any other keypress is dropped silently. The
 * terminal is restored before tools execute, so ask_user()'s fgets is
 * unaffected. */
static struct termios g_save_tio;
static int g_tio_saved = 0;
static pthread_t g_snoop;
static volatile int g_snoop_on = 0;

static void *snoop_thread(void *arg) {
    (void)arg;
    struct pollfd pfd = { .fd = 0, .events = POLLIN };
    char line[64];
    int nline = 0;
    while (g_snoop_on) {
        if (poll(&pfd, 1, 50) <= 0) continue;
        unsigned char c;
        ssize_t n = read(0, &c, 1);
        if (n != 1) continue;
        if (c == 0x1b) { g_stop_turn = 1; break; }   /* ESC: cancel the turn */
        if (c == '\n' || c == '\r') {
            /* a full line typed during generation: honor the slash commands
               so typing /exit isn't silently eaten by the raw-mode snoop */
            if (nline >= 1 && line[0] == '/') {
                line[nline] = 0;
                if (strcmp(line, "/exit") == 0 || strcmp(line, "/quit") == 0) {
                    g_shutdown = 1;
                    g_stop_turn = 1;
                    break;
                }
                if (strcmp(line, "/reset") == 0 || strcmp(line, "/clear") == 0) {
                    g_snoop_reset = 1;
                    g_stop_turn = 1;
                    break;
                }
            }
            nline = 0;
        } else if (nline < (int)sizeof(line) - 1) {
            line[nline++] = (char)c;
        }
    }
    return NULL;
}

static void key_snoop_start(void) {
    if (!g_stdin_tty || g_snoop_on) return;
    struct termios t;
    if (tcgetattr(0, &t) != 0) return;
    g_save_tio = t;
    g_tio_saved = 1;
    t.c_lflag &= ~(ICANON | ECHO);   /* raw-ish; ISIG stays on */
    t.c_cc[VMIN] = 0;
    t.c_cc[VTIME] = 0;
    tcsetattr(0, TCSANOW, &t);
    g_snoop_on = 1;
    if (pthread_create(&g_snoop, NULL, snoop_thread, NULL) != 0) {
        g_snoop_on = 0;
        tcsetattr(0, TCSANOW, &g_save_tio);
        g_tio_saved = 0;
    }
}

static void key_snoop_stop(void) {
    if (!g_snoop_on) return;
    g_snoop_on = 0;
    pthread_join(g_snoop, NULL);
    if (g_tio_saved) {
        tcsetattr(0, TCSANOW, &g_save_tio);
        g_tio_saved = 0;
    }
}

/* UTF-8 piece buffering (byte BPE can split a char across tokens) */
static char g_piece[64];
static int g_piece_len = 0;
static void flush_piece(void) {
    if (g_piece_len == 0) return;
    fwrite(g_piece, 1, (size_t)g_piece_len, stdout);
    g_piece_len = 0;
}
static void print_piece(const char *s) {
    /* Never drop bytes: the old code skipped the copy when a piece would
       overflow the 60-byte budget, silently losing words ("Either way" →
       "Ei") and mangling tags ("</tool_call>" → "</tool>"). Flush the
       buffer first whenever it is full, then copy byte-by-byte. */
    size_t len = strlen(s);
    for (size_t i = 0; i < len; i++) {
        if (g_piece_len >= 56) flush_piece();
        g_piece[g_piece_len++] = s[i];
    }
}

/* ---- session (continual KV) ---- */
static uint64_t fnv1a(const char *s) {
    uint64_t h = 1469598103934665603ULL;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        h ^= *p; h *= 1099511628211ULL;
    }
    return h;
}

static void session_save(void) {
    if (!g_session_dir[0] || !g_rs_ready) return;
    char p[1200];
    snprintf(p, sizeof(p), "%s/state.bin", g_session_dir);
    runstate_save(&g_rs, p);
    snprintf(p, sizeof(p), "%s/salience.bin", g_session_dir);
    hcm_save(p);
}

static void session_wipe_files(void);   /* fwd (defined below) */

/* snapshot the session into a dated copy of the binary, then remove the
   temp session dir. Only in snapshot mode (no --session). */
static void finish_session(void) {
    session_save();
    if (!g_selfctx) return;
    const char *slash = strrchr(g_self_exe, '/');
    char out_dir[1024];
    if (slash && slash != g_self_exe) {
        size_t n = (size_t)(slash - g_self_exe);
        if (n >= sizeof(out_dir)) n = sizeof(out_dir) - 1;
        memcpy(out_dir, g_self_exe, n);
        out_dir[n] = 0;
    } else {
        snprintf(out_dir, sizeof(out_dir), ".");
    }
    char *out = selfctx_snapshot(g_self_exe, g_session_dir, out_dir,
                                 g_model_path, strlen(g_model_path) + 1);
    if (!out && strcmp(out_dir, ".") != 0)
        out = selfctx_snapshot(g_self_exe, g_session_dir, ".",
                               g_model_path, strlen(g_model_path) + 1);
    if (out) {
        fprintf(stderr, "qma: context snapshot → %s\n", out);
        free(out);
    } else {
        fprintf(stderr, "qma: snapshot failed (session kept in %s)\n", g_session_dir);
    }
    session_wipe_files();
    rmdir(g_session_dir);
}

/* unlink every session file (used by session_reset and --reset; safe to
   call before the runstate exists) */
static void session_wipe_files(void) {
    if (!g_session_dir[0]) return;
    char p[1200];
    snprintf(p, sizeof(p), "%s/kv.bin", g_session_dir); unlink(p);
    snprintf(p, sizeof(p), "%s/state.bin", g_session_dir); unlink(p);
    snprintf(p, sizeof(p), "%s/salience.bin", g_session_dir); unlink(p);
    snprintf(p, sizeof(p), "%s/sysfp.txt", g_session_dir); unlink(p);
    snprintf(p, sizeof(p), "%s/sysready.txt", g_session_dir); unlink(p);
}

static void session_reset(void) {
    if (!g_session_dir[0]) return;
    runstate_free(&g_rs);
    session_wipe_files();
    char kv_path[1200];
    snprintf(kv_path, sizeof(kv_path), "%s/kv.bin", g_session_dir);
    if (runstate_init_kv(&g_rs, g_ctx, kv_path, 1) != 0) {
        fprintf(stderr, "qma: session reset failed\n");
        exit(1);
    }
}

/* mkdir -p: create every path component (the default session lives under
   ~/.qma/session — the parent ~/.qma must exist or the KV file silently
   falls back to memory and nothing ever persists) */
static void mkdirs(const char *path) {
    char tmp[4096];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            mkdir(tmp, 0700);
            *p = '/';
        }
    }
    mkdir(tmp, 0700);
}

/* ---- model-path config (~/.qma/config) ----
 * The model path lives in a small config file so the binary is portable:
 * copy a snapshot to another phone and it knows what to load (and
 * reprompts if that path is missing there). The choice is also embedded in
 * the snapshot binary itself (see selfctx) and rewritten here on launch. */
static void cfg_path(char *buf, size_t n) {
    const char *home = getenv("HOME");
    if (home && home[0]) snprintf(buf, n, "%s/.qma/config", home);
    else snprintf(buf, n, ".qma/config");
}

/* read the first model path from the config (comments/#, "key = value"
   or a bare path line) */
static int cfg_read_model(char *out, size_t n) {
    FILE *f = fopen(g_cfg_path, "r");
    if (!f) return 0;
    int got = 0;
    char line[4096];
    while (fgets(line, sizeof(line), f)) {
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == '\r' || *p == 0) continue;
        char *eq = strchr(p, '=');
        if (eq) {
            p = eq + 1;
            while (*p == ' ' || *p == '\t') p++;
        }
        size_t l = strlen(p);
        while (l > 0 && (p[l-1] == '\n' || p[l-1] == '\r' ||
                         p[l-1] == ' ' || p[l-1] == '\t')) p[--l] = 0;
        if (l > 0) { snprintf(out, n, "%s", p); got = 1; break; }
    }
    fclose(f);
    return got;
}

static void cfg_write_model(const char *path) {
    char d[4096];
    snprintf(d, sizeof(d), "%s", g_cfg_path);
    char *sl = strrchr(d, '/');
    if (sl) { *sl = 0; mkdirs(d); }
    FILE *f = fopen(g_cfg_path, "w");
    if (!f) return;
    fprintf(f, "# qma config — model path (GGUF). Rewritten by qma on launch.\n"
              "model = %s\n", path);
    fclose(f);
}

static int path_is_model(const char *p) {
    struct stat st;
    return stat(p, &st) == 0 && S_ISREG(st.st_mode);
}

/* Resolve the model path. Sources, in order: CLI -m, embedded snapshot
   config, config file, interactive prompt. Validate existence and
   reprompt on failure (non-interactive: error out). Persist the choice. */
static void resolve_model_path(int from_cli) {
    char cand[4096] = "";
    int have = 0;

    if (from_cli) {
        snprintf(cand, sizeof(cand), "%s", g_model_path);
        have = 1;
    } else {
        /* embedded in this (snapshot) binary? */
        selfctx_hdr_t hdr;
        if (g_self_exe[0] && selfctx_detect(g_self_exe, &hdr)) {
            char *cfg = selfctx_get_config(g_self_exe, &hdr);
            if (cfg && cfg[0]) { snprintf(cand, sizeof(cand), "%s", cfg); have = 1; }
            free(cfg);
        }
        if (!have) have = cfg_read_model(cand, sizeof(cand));
        if (!have && !g_stdin_tty) {
            fprintf(stderr, "qma: no model path configured — using default %s\n",
                    DEFAULT_MODEL);
            snprintf(cand, sizeof(cand), "%s", DEFAULT_MODEL);
            have = 1;
        }
    }

    int eof = 0;
    for (;;) {
        if (!have) {
            c_cyan();
            printf("model path (GGUF file, empty = default) > ");
            c_reset();
            fflush(stdout);
            if (!fgets(cand, sizeof(cand), stdin)) {
                snprintf(cand, sizeof(cand), "%s", DEFAULT_MODEL);
                eof = 1;
            } else {
                size_t l = strlen(cand);
                while (l > 0 && (cand[l-1] == '\n' || cand[l-1] == '\r')) cand[--l] = 0;
                if (l == 0) snprintf(cand, sizeof(cand), "%s", DEFAULT_MODEL);
            }
            have = 1;
        }
        if (path_is_model(cand)) break;
        if (!g_stdin_tty || eof) {
            fprintf(stderr, "qma: model not found: %s\n", cand);
            exit(1);
        }
        fprintf(stderr, "qma: model not found: %s — try again\n", cand);
        have = 0;
    }

    snprintf(g_model_path_buf, sizeof(g_model_path_buf), "%s", cand);
    g_model_path = g_model_path_buf;
    cfg_write_model(g_model_path);
}

static void session_init(void) {
    if (!g_session_dir[0]) {
        /* memory-only session: still initialize the runstate (the old
           code returned here without init, so interactive ./qma with no
           --session segfaulted inside the first eval) */
        if (runstate_init_kv(&g_rs, g_ctx, NULL, 0) != 0) {
            fprintf(stderr, "qma: state alloc failed\n");
            exit(1);
        }
        g_rs_ready = 1;
        return;
    }
    mkdirs(g_session_dir);
    char kv_path[1200], st_path[1200], sal_path[1200], fp_path[1200];
    snprintf(kv_path, sizeof(kv_path), "%s/kv.bin", g_session_dir);
    snprintf(st_path, sizeof(st_path), "%s/state.bin", g_session_dir);
    snprintf(sal_path, sizeof(sal_path), "%s/salience.bin", g_session_dir);
    snprintf(fp_path, sizeof(fp_path), "%s/sysfp.txt", g_session_dir);
    struct stat kvst;
    int kv_pre_existed = (stat(kv_path, &kvst) == 0);
    if (runstate_init_kv(&g_rs, g_ctx, kv_path, 1) != 0) {
        fprintf(stderr, "qma: state alloc failed\n");
        exit(1);
    }
    g_rs_ready = 1;
    if (runstate_load(&g_rs, st_path) == 0) {
        /* resumed: the context is only valid if the system prompt matches
           AND its one-time ingestion actually completed (a process killed
           mid-warm-up leaves a partial system prompt in the KV — sysfp
           would match, but the context is corrupt) */
        uint64_t fp = fnv1a(g_system_prompt);
        FILE *f = fopen(fp_path, "r");
        uint64_t stored = 0;
        if (f) {
            if (fscanf(f, "%llx", (unsigned long long *)&stored) != 1) stored = 0;
            fclose(f);
        }
        char rp[1200];
        snprintf(rp, sizeof(rp), "%s/sysready.txt", g_session_dir);
        struct stat rst;
        int ready = (stat(rp, &rst) == 0);
        if (stored != fp || !ready) {
            fprintf(stderr, "qma: %s — resetting stale session\n",
                    stored != fp ? "system prompt changed"
                                 : "system prompt not fully ingested (interrupted)");
            session_reset();
            g_rs_ready = 1;
        }
    } else if (kv_pre_existed) {
        /* a KV file with no matching saved state is half-written garbage */
        fprintf(stderr, "qma: stale session state — resetting\n");
        session_reset();
        g_rs_ready = 1;
    }
    /* restore HCM salience (pinned heavy-hitters) when a session was
       resumed — it was saved on exit but never loaded back */
    {
        char sal[1200];
        snprintf(sal, sizeof(sal), "%s/salience.bin", g_session_dir);
        struct stat salst;
        if (stat(sal, &salst) == 0 && salst.st_size > 0)
            hcm_load(sal, g_ctx);
    }
    /* write the system-prompt fingerprint for next launch (the sysready
       marker is written only after the ingestion completes) */
    uint64_t fp = fnv1a(g_system_prompt);
    FILE *f = fopen(fp_path, "w");
    if (f) { fprintf(f, "%llx\n", (unsigned long long)fp); fclose(f); }
}

/* evaluate the system prompt into the KV (fresh sessions and /reset only)
   and mark the ingestion complete so an interrupted warm-up is detected
   on the next launch */
static void ingest_system_prompt(void) {
    char sys_wrapped[270000];
    snprintf(sys_wrapped, sizeof(sys_wrapped),
             "<|im_start|>system\n%s<|im_end|>\n", g_system_prompt);
    int *ids = malloc(sizeof(int) * 262144);
    float *logits = malloc(sizeof(float) * N_VOCAB);
    if (ids && logits) {
        int nids = qma_tokenize(&g_model, sys_wrapped, ids, 262144);
        if (nids > 0) qma_eval(&g_model, &g_rs, ids, nids, logits, g_threads, g_prefetch, 0);
    }
    free(ids); free(logits);
    if (g_session_dir[0]) {
        char p[1200];
        snprintf(p, sizeof(p), "%s/sysready.txt", g_session_dir);
        FILE *f = fopen(p, "w");
        if (f) { fprintf(f, "%llx\n", (unsigned long long)fnv1a(g_system_prompt)); fclose(f); }
    }
}

static void on_signal(int sig) {
    (void)sig;
    g_shutdown = 1;
    g_stop_turn = 1;
}

/* ---- generation driver (ported from the engine's stream_generate) ---- */
#define RAW_CAP 524288
#define SSE_HOLDBACK 32

typedef struct {
    char content[131072];   /* text content (tool XML excluded) */
    size_t content_len;
    tool_call_t calls[MAX_TOOL_CALLS];
    int n_calls;
    int had_tool_mode;      /* model attempted a call (even if dropped) */
    int cancelled;          /* generation stopped by the user (ESC) */
} gen_result_t;

static void apply_eos_penalty(const qma_t *m, float *logits) {
    if (g_eos_penalty <= 0.0f) return;
    for (int i = 0; i < m->n_vocab; i++)
        if (m->tok_type[i] & 4) logits[i] -= g_eos_penalty;
}

static void content_append(gen_result_t *res, const char *s, size_t n) {
    if (res->content_len + n + 1 < sizeof(res->content)) {
        memcpy(res->content + res->content_len, s, n);
        res->content_len += n;
        res->content[res->content_len] = 0;
    }
}

static void print_content(gen_result_t *res, const char *s, size_t n) {
    content_append(res, s, n);
    char tmp[4096];
    while (n > 0) {
        size_t c = n > sizeof(tmp) - 1 ? sizeof(tmp) - 1 : n;
        memcpy(tmp, s, c); tmp[c] = 0;
        out(tmp);
        s += c; n -= c;
    }
}

static int agent_generate(qma_t *m, const char *prompt, int enable_thinking,
                          gen_result_t *res) {
    memset(res, 0, sizeof(*res));
    g_stop_turn = 0;   /* fresh turn: clear any earlier soft stop (ESC) */
    double t_pre0 = now_s2();
    int *ids = malloc(sizeof(int) * 262144);
    if (!ids) return -1;
    int nids = qma_tokenize(m, prompt, ids, 262144);
    if (nids < 0) { free(ids); return -1; }
    float *logits = malloc(sizeof(float) * N_VOCAB);
    if (!logits) { free(ids); return -1; }
    if (qma_eval(m, &g_rs, ids, nids, logits, g_threads, g_prefetch, 0) != 0) {
        free(ids); free(logits); return -1;
    }
    double t_pre1 = now_s2();
    apply_eos_penalty(m, logits);

    sampler_t sp;
    sampler_init(&sp, (uint64_t)time(NULL), g_temp, g_top_k, g_top_p, g_repeat);

    char *raw = malloc(RAW_CAP);
    char *ans = malloc(RAW_CAP);
    if (!raw || !ans) { free(raw); free(ans); free(ids); free(logits); return -1; }
    size_t raw_len = 0, ans_len = 0, reasoning_emitted = 0;
    int thinking_done = enable_thinking ? 0 : 1;
    int tool_mode = 0;
    size_t tool_scan = 0;
    int ngen = 0;

    int cap = 262144;
    c_dim();
    int cancelled = 0;
    for (int i = 0; i < cap; i++) {
        if (g_stop_turn) { cancelled = 1; break; }
        /* two-phase sampling with grammar constraint (llama.cpp approach):
           get the top-k candidates, mask any that cannot legally extend the
           tool-call grammar, then sample. This makes malformed calls
           impossible while a call is open. */
        int cand_ids[40];
        float cand_lgs[40];
        int nc = sampler_candidates(N_VOCAB, logits, &sp, cand_ids, cand_lgs, 40);
        if (nc > 0 && tool_mode && tool_block_open(ans, ans_len)) {
            for (int ci = 0; ci < nc; ci++) {
                if (cand_lgs[ci] <= -1e29f) continue;             /* already masked */
                if (m->tok_type[cand_ids[ci]] & 4) {              /* control tokens */
                    cand_lgs[ci] = -1e30f;
                    continue;
                }
                if (!grammar_tool_token_ok(m, ans, ans_len, cand_ids[ci]))
                    cand_lgs[ci] = -1e30f;                        /* grammar rejects */
            }
        }
        int id = nc > 0 ? sampler_pick(&sp, cand_ids, cand_lgs, nc, g_top_p) : 0;
        if (id < 0 || id >= N_VOCAB) break;
        if (id == (int)m->id_im_end || (m->tok_type[id] & 4)) {
            if (id == (int)m->id_im_end) {
                qma_eval(m, &g_rs, &id, 1, logits, g_threads, g_prefetch, 0);
            } else {
                /* non-im_end control token: close the assistant frame so the
                   next delta appends a clean <|im_start|>user boundary */
                int ie = (int)m->id_im_end;
                if (ie > 0) qma_eval(m, &g_rs, &ie, 1, logits, g_threads, g_prefetch, 0);
            }
            break;
        }
        char piece[4096];
        int pl = qma_detokenize(m, &id, 1, piece, sizeof(piece));
        if (pl > 0 && raw_len + (size_t)pl < RAW_CAP) {
            memcpy(raw + raw_len, piece, (size_t)pl);
            raw_len += (size_t)pl;
            raw[raw_len] = 0;
        }

        if (!thinking_done) {
            /* reasoning phase */
            char *close = strstr(raw, "</think>");
            if (close) {
                size_t pos = (size_t)(close - raw) + strlen("</think>");
                if (pos > reasoning_emitted) {
                    size_t rbl = pos - reasoning_emitted;
                    char buf[4200];
                    size_t n0 = rbl < sizeof(buf) - 1 ? rbl : sizeof(buf) - 1;
                    memcpy(buf, raw + reasoning_emitted, n0);
                    buf[n0] = 0;
                    size_t tlen = strlen(buf);
                    if (tlen >= 8 && strcmp(buf + tlen - 8, "</think>") == 0)
                        buf[tlen - 8] = 0;
                    char *op = strstr(buf, "<think>");
                    if (op) {
                        flush_piece();   /* older thinking bytes first */
                        fputs(op + 7, stdout);
                    } else {
                        flush_piece();
                        fputs(buf, stdout);
                    }
                }
                reasoning_emitted = pos;
                thinking_done = 1;
                ans_len = raw_len - pos;
                if (ans_len > 0) memcpy(ans, raw + pos, ans_len);
                ans[ans_len] = 0;
                c_reset();
            } else if (pl > 0) {
                print_piece(piece);
                reasoning_emitted = raw_len;
            }
        } else if (tool_mode) {
            /* tool phase: show raw tokens dim; emit completed valid calls */
            if (ans_len + (size_t)pl < RAW_CAP) {
                memcpy(ans + ans_len, piece, (size_t)pl);
                ans_len += (size_t)pl;
                ans[ans_len] = 0;
            }
            print_piece(piece);
            /* process blocks */
            for (;;) {
                const char *base = ans + (tool_scan <= ans_len ? tool_scan : ans_len);
                const char *t0 = strstr(base, "<tool_call>");
                if (!t0) break;
                const char *tend = strstr(t0 + strlen("<tool_call>"), "</tool_call>");
                if (!tend) break;              /* open: wait for more tokens */
                const char *after = t0;
                tool_call_t tc;
                int r = parse_one_tool_call(ans, &after, &tc);
                if (r == 1) {
                    res->had_tool_mode = 1;
                    if (res->n_calls < MAX_TOOL_CALLS) {
                        snprintf(res->calls[res->n_calls].name,
                                 sizeof(res->calls[res->n_calls].name), "%s", tc.name);
                        snprintf(res->calls[res->n_calls].args,
                                 sizeof(res->calls[res->n_calls].args), "%s", tc.args);
                        res->n_calls++;
                        c_reset(); c_cyan();
                        flush_piece();   /* pending raw tokens before the ⬡ line */
                        printf("\n⬡ %s(%s)\n", tc.name, tc.args);
                        c_reset(); c_dim();
                    }
                }
                /* r == -1: malformed — dropped (never executed) */
                tool_scan = (size_t)(after - ans);
                if (r != 0) continue;
                break;
            }
        } else {
            /* answer phase: content with holdback; detect <tool_call> */
            if (ans_len + (size_t)pl < RAW_CAP) {
                memcpy(ans + ans_len, piece, (size_t)pl);
                ans_len += (size_t)pl;
                ans[ans_len] = 0;
            }
            /* stray </think> rewind (Qwen3 drafts the answer twice) */
            char *stray = strstr(ans, "</think>");
            if (stray) {
                size_t rest = ans_len - (size_t)(stray - ans) - 8;
                memmove(ans, stray + 8, rest);
                ans_len = rest;
                ans[ans_len] = 0;
                /* the first draft was already streamed: mark the rewind so
                   the duplication is explained, not read as engine output */
                flush_piece();
                c_dim();
                fputs("\n[draft rewound — the model re-drafted its answer]\n", stdout);
                c_reset();
            }
            char *tc = strstr(ans, "<tool_call>");
            if (tc) {
                size_t pre = (size_t)(tc - ans);
                if (pre > 0) {
                    size_t el = u8_safe_len(ans, pre);
                    c_reset();
                    print_content(res, ans, el);
                }
                tool_mode = 1;
                res->had_tool_mode = 1;   /* model attempted a call */
                tool_scan = 0;
                size_t rest = ans_len - pre;
                memmove(ans, ans + pre, rest);
                ans_len = rest;
                ans[ans_len] = 0;
                c_dim();
            } else if (ans_len > SSE_HOLDBACK) {
                size_t emit = ans_len - SSE_HOLDBACK;
                size_t el = u8_safe_len(ans, emit);
                if (el > 0) {
                    c_reset();
                    print_content(res, ans, el);
                    c_dim();
                }
                size_t held = ans_len - el;
                memmove(ans, ans + el, held);
                ans_len = held;
                ans[ans_len] = 0;
            }
        }
        if (qma_eval(m, &g_rs, &id, 1, logits, g_threads, g_prefetch, 0) != 0) break;
        apply_eos_penalty(m, logits);
        ngen++;
    }
    c_reset();
    flush_piece();

    if (cancelled) {
        /* ESC: stop here. The partial assistant text is already in the KV;
           eval <|im_end|> to close the frame so the next user turn starts
           from a clean boundary. Partial tool results are discarded — the
           turn ends and control returns to the user prompt. */
        int ie = (int)m->id_im_end;
        if (ie > 0) qma_eval(m, &g_rs, &ie, 1, logits, g_threads, g_prefetch, 0);
        res->cancelled = 1;
        free(raw); free(ans); free(ids); free(logits);
        return 0;
    }

    /* finalize: flush any complete blocks, then remaining content */
    if (tool_mode) {
        for (;;) {
            const char *base = ans + (tool_scan <= ans_len ? tool_scan : ans_len);
            const char *t0 = strstr(base, "<tool_call>");
            if (!t0) break;
            const char *after = t0;
            tool_call_t tc;
            int r = parse_one_tool_call(ans, &after, &tc);
            if (r == 1) {
                res->had_tool_mode = 1;
                if (res->n_calls < MAX_TOOL_CALLS) {
                    snprintf(res->calls[res->n_calls].name, sizeof(res->calls[res->n_calls].name), "%s", tc.name);
                    snprintf(res->calls[res->n_calls].args, sizeof(res->calls[res->n_calls].args), "%s", tc.args);
                    res->n_calls++;
                    c_cyan();
                    flush_piece();   /* pending raw tokens before the ⬡ line */
                    printf("\n⬡ %s(%s)\n", tc.name, tc.args);
                    c_reset();
                }
            }
            tool_scan = (size_t)(after - ans);
        }
    } else if (thinking_done && ans_len > 0) {
        size_t el = u8_safe_len(ans, ans_len);
        if (el > 0) print_content(res, ans, el);
    }
    c_reset();

    /* TPS stats: prefill + generation */
    {
        double t_end = now_s2();
        double pre_s = t_pre1 - t_pre0, gen_s = t_end - t_pre1;
        if (pre_s > 0.05 && gen_s > 0.05)
            fprintf(stderr, "[prefill %d tok @ %.1f tok/s · gen %d tok @ %.1f tok/s · %.1fs]\n",
                    nids, nids / pre_s, ngen, ngen / gen_s, t_end - t_pre0);
    }

    free(raw); free(ans); free(ids); free(logits);
    return 0;
}

/* ---- delta rendering ---- */
static int build_user_delta(char *buf, size_t buflen, const char *input) {
    return snprintf(buf, buflen,
                    "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n",
                    input);
}

static int build_tool_delta(char *buf, size_t buflen,
                            const char *const *results, int n,
                            const char *note) {
    size_t o = 0;
    #define APP(...) do { int w = snprintf(buf + o, buflen - o, __VA_ARGS__); \
        if (w < 0 || (size_t)w >= buflen - o) return -1; o += (size_t)w; } while (0)
    APP("<|im_start|>user\n");
    if (note && note[0]) APP("<tool_response>\n%s\n</tool_response>", note);
    for (int i = 0; i < n; i++)
        APP("<tool_response>\n%s\n</tool_response>", results[i]);
    APP("<|im_end|>\n<|im_start|>assistant\n<think>\n");
    #undef APP
    return (int)o;
}

/* ---- agent turn ---- */
static int agent_turn(const char *input) {
    char delta[524288];
    char *cur = strdup(input);
    int pre_rendered = 0;   /* cur is already a rendered template delta */
    int turns = 0;
    gen_result_t res;

    /* pi-agent loop: call the model, execute any tool calls, fold the
       results back and call again — the loop continues ONLY while the
       model calls tools. A text-only response ends the turn and control
       returns to the user prompt; the agent never nudges the model to
       keep going (no [continue], no DONE parsing). MAX_TURNS is the only
       runaway guard, exactly like pi's turn budget. */
    for (;;) {
        if (g_shutdown) break;
        if (++turns > MAX_TURNS) {
            c_yellow(); printf("\n[stopping: turn budget reached]\n"); c_reset();
            break;
        }
        if (pre_rendered) {
            snprintf(delta, sizeof(delta), "%s", cur);
            pre_rendered = 0;
        } else {
            build_user_delta(delta, sizeof(delta), cur);
        }
        key_snoop_start();
        int gr = agent_generate(&g_model, delta, g_enable_thinking, &res);
        key_snoop_stop();
        if (gr != 0) {
            c_red(); printf("\n[generation failed]\n"); c_reset();
            free(cur);
            return -1;
        }
        printf("\n");

        if (res.cancelled) {
            c_yellow(); printf("[cancelled — back to you]\n"); c_reset();
            free(cur);
            return 0;
        }

        if (res.n_calls > 0) {
            /* execute every call, fold results back, call the model again */
            char *results[MAX_TOOL_CALLS];
            int n_res = 0;
            for (int i = 0; i < res.n_calls; i++) {
                char *r = NULL;
                tool_dispatch(res.calls[i].name, res.calls[i].args, &r);
                if (strlen(r) > 6000) {
                    /* heap overflow fix: the "...\n[output truncated]" suffix
                       is 26 bytes incl. NUL; the old code allocated 6004 and
                       wrote 6026 — any output over 6000 chars overran the
                       heap. Size to the actual suffix length instead. */
                    static const char trunc[] = "...\n[output truncated]";
                    size_t need = 6000 + sizeof(trunc);
                    char *tmp = malloc(need);
                    if (tmp) {
                        memcpy(tmp, r, 6000);
                        memcpy(tmp + 6000, trunc, sizeof(trunc)); /* incl. NUL */
                        free(r);
                        r = tmp;
                    }
                }
                results[n_res++] = r;
                if (strncmp(r, "ERROR:", 6) == 0) {
                    c_red(); printf("  ✕ %s\n", r); c_reset();
                } else {
                    c_green(); printf("  ✓ %s\n", r); c_reset();
                }
            }
            char td[524288];
            build_tool_delta(td, sizeof(td), (const char **)results, n_res, NULL);
            for (int i = 0; i < n_res; i++) free(results[i]);
            free(cur);
            cur = strdup(td);
            pre_rendered = 1;
            continue;
        }

        if (res.had_tool_mode && res.n_calls == 0) {
            /* the model's calls were all malformed and the engine dropped
               them — feed that back as a tool error (exactly what pi does
               for a failed tool) and continue; bounded by MAX_TURNS */
            char td[524288];
            build_tool_delta(td, sizeof(td), NULL, 0,
                "ERROR: your tool call was malformed and discarded — it "
                "NEVER executed. Re-issue it as ONE complete call with "
                "valid arguments.");
            free(cur);
            cur = strdup(td);
            pre_rendered = 1;
            continue;
        }

        /* text-only response: the model decided it's done — end the turn */
        break;
    }
    free(cur);
    return 0;
}

/* ---- system prompt ---- */
static void build_system_prompt(char *buf, size_t buflen) {
    char header[70000];
    tools_render_header(header, sizeof(header));
    snprintf(buf, buflen, "%s\n\n## Agent Behavior\n\n"
        "You are an autonomous coding agent working on a phone. The user gives you a "
        "goal; YOU drive the work to completion.\n"
        "- Work step by step. Call tools as many times as needed. Do NOT stop after a "
        "single tool call — keep going until the task is genuinely finished.\n"
        "- For multi-step work, call todo_write() to create subtasks, then work through "
        "them and mark them done with todo_complete().\n"
        "- After changing files, VERIFY your work: re-read the file, run the command, "
        "check the output. Don't declare success you haven't checked.\n"
        "- When you need input the user must decide, use ask_user().\n"
        "- When the task is truly complete, give a short summary and stop calling tools.\n"
        "- If asked a simple question with no work needed, just answer it.\n",
        header);
}

/* ---- main ---- */
static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s [options]\n"
        "  -m <path>     model file (default: %s, else ~/.qma/config)\n"
        "  -p <prompt>   one-shot prompt (default: interactive)\n"
        "  -t <n>        threads (default 8)\n"
        "  -c <n>        ring context size (default 65536)\n"
        "  --temp <f>    temperature (default 0.7)\n"
        "  --repeat <f>  repeat penalty (default 1.05)\n"
        "  --eos-penalty <f>  control-token logit penalty (default 5.0)\n"
        "  --no-think    disable the <think> block\n"
        "  --ecache <mb> bounded expert cache (default 1024; 0 = mmap)\n"
        "  --session <dir>  persistent continual session\n"
        "  --workdir <dir>  working directory for tools (default: cwd)\n"
        "  --reset        wipe the persistent context and start fresh\n"
        "  --no-color     disable ANSI colors\n"
        "  -h            help\n"
        "\ninteractive commands: /exit /quit /reset /clear\n",
        prog, DEFAULT_MODEL);
}

int main(int argc, char **argv) {
    const char *one_shot = NULL;
    int model_from_cli = 0;
    g_model_path = DEFAULT_MODEL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            snprintf(g_model_path_buf, sizeof(g_model_path_buf), "%s", argv[++i]);
            g_model_path = g_model_path_buf;
            model_from_cli = 1;
        }
        else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) one_shot = argv[++i];
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) g_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) g_ctx = atoi(argv[++i]);
        else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) g_temp = atof(argv[++i]);
        else if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) g_repeat = atof(argv[++i]);
        else if (strcmp(argv[i], "--eos-penalty") == 0 && i + 1 < argc) g_eos_penalty = atof(argv[++i]);
        else if (strcmp(argv[i], "--no-think") == 0) g_enable_thinking = 0;
        else if (strcmp(argv[i], "--ecache") == 0 && i + 1 < argc) g_ecache_mb = atoi(argv[++i]);
        else if (strcmp(argv[i], "--session") == 0 && i + 1 < argc) {
            snprintf(g_session_dir, sizeof(g_session_dir), "%s", argv[++i]);
        }
        else if (strcmp(argv[i], "--workdir") == 0 && i + 1 < argc) {
            snprintf(g_workdir, sizeof(g_workdir), "%s", argv[++i]);
        }
        else if (strcmp(argv[i], "--reset") == 0) g_reset = 1;
        else if (strcmp(argv[i], "--no-color") == 0) g_use_color = 0;
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); usage(argv[0]); return 1; }
    }

    g_use_color = isatty(1) && isatty(0);  /* ANSI only on a real terminal */
    g_stdin_tty = isatty(0);               /* ESC snoop + prompt only on a tty */
    if (g_workdir[0]) {
        /* create the workdir if missing */
        char tmp[4096];
        snprintf(tmp, sizeof(tmp), "%s", g_workdir);
        for (char *p = tmp + 1; *p; p++) {
            if (*p == '/') { *p = 0; mkdir(tmp, 0755); *p = '/'; }
        }
        mkdir(tmp, 0755);
        if (chdir(g_workdir) != 0)
            fprintf(stderr, "qma: workdir %s: %s\n", g_workdir, strerror(errno));
    }

    /* resolve the running binary (read-only self-read is allowed; writing
       to it is not — "Text file busy" — so snapshots are NEW files) */
    {
        ssize_t n = readlink("/proc/self/exe", g_self_exe, sizeof(g_self_exe) - 1);
        if (n > 0) g_self_exe[n] = 0;
        else snprintf(g_self_exe, sizeof(g_self_exe), "%s", argv[0]);
    }
    /* model path: CLI -m > embedded snapshot config > ~/.qma/config > prompt */
    cfg_path(g_cfg_path, sizeof(g_cfg_path));
    resolve_model_path(model_from_cli);

    /* silent thermal governor (off for debugging via QMA_NOTHERMAL=1) */
    if (getenv("QMA_NOTHERMAL") == NULL)
        thermal_start();

    fprintf(stderr, "qma: loading %s ...\n", g_model_path);
    char err[512];
    if (qma_load(&g_model, g_model_path, err, sizeof(err)) != 0) {
        fprintf(stderr, "qma: load failed: %s\n", err);
        return 1;
    }
    qma_prefetch_init(&g_model);
    /* YALIS: accumulate per-expert output averages to predict the next
       layer's input for prefetching. The machinery existed but was never
       activated — qma_defaults_enable was never called, so g_def_sum
       stayed NULL and every branch was dead. Wire it here (disable with
       QMA_NOYALIS=1 if it ever hurts cache hits). */
    if (getenv("QMA_NOYALIS") == NULL) {
        if (qma_defaults_enable(&g_model, NULL) == 0)
            fprintf(stderr, "qma: YALIS expert-defaults active (predictive prefetch)\n");
    }
    if (g_ecache_mb > 0) {
        g_model.mmap_exps = 0;
        qma_ecache_arm(&g_model, (size_t)g_ecache_mb * 1024 * 1024, 4);
        fprintf(stderr, "qma: expert cache armed (%d MB, bounded)\n", g_ecache_mb);
    } else {
        g_model.mmap_exps = 1;
    }

    build_system_prompt(g_system_prompt, sizeof(g_system_prompt));
    /* session mode: --session <dir> = classic persistent dir (no
       snapshots); default = temp dir + dated context snapshot on exit */
    if (!g_session_dir[0]) {
        const char *tmp = getenv("TMPDIR");
        char tmpl[1200];
        snprintf(tmpl, sizeof(tmpl), "%s/qma-XXXXXX",
                 tmp && tmp[0] ? tmp : "/data/data/com.termux/files/usr/tmp");
        char *d = mkdtemp(tmpl);
        if (d) {
            snprintf(g_session_dir, sizeof(g_session_dir), "%s", d);
        } else {
            mkdirs(".qma_tmp");
            snprintf(g_session_dir, sizeof(g_session_dir), ".qma_tmp");
        }
        g_selfctx = 1;
        fprintf(stderr, "qma: snapshot mode — context embedded in a dated copy of the binary on exit\n");
    }
    if (g_reset) {
        fprintf(stderr, "qma: reset — ignoring any embedded context\n");
        session_wipe_files();
    } else if (g_selfctx) {
        /* resume the embedded context from THIS binary's tail, if any */
        selfctx_hdr_t hdr;
        if (selfctx_detect(g_self_exe, &hdr)) {
            if (selfctx_extract(g_self_exe, &hdr, g_session_dir) == 0) {
                fprintf(stderr, "qma: embedded context found (n_pos=%llu) — extracting\n",
                        (unsigned long long)hdr.n_pos);
            }
        }
    }
    session_init();

    /* the system prompt is evaluated ONCE per context lifetime: only when
       the KV is truly empty (fresh session or after a wipe). A resumed
       session appends just the new user turn — nothing is re-sent. */
    if (g_rs.n_pos == 0 && g_rs_ready) {
        fprintf(stderr, "qma: fresh context — ingesting system prompt (one-time, a few minutes)…\n");
        fflush(stderr);
        ingest_system_prompt();
        fprintf(stderr, "qma: session started (system prompt ingested, n_pos=%d)\n", g_rs.n_pos);
    } else {
        fprintf(stderr, "qma: resumed session at n_pos=%d (context kept)\n", g_rs.n_pos);
    }

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    char cwd[4096];
    getcwd(cwd, sizeof(cwd));
    fprintf(stderr, "qma: unified agent | model %s | workdir %s | session %s\n",
            g_model_path, cwd, g_session_dir[0] ? g_session_dir : "(memory)");
    fprintf(stderr, "> type /exit to quit, /reset to clear the session\n\n");

    if (one_shot) {
        agent_turn(one_shot);
        thermal_stop();
        finish_session();
        if (g_model.ecache_on) qma_ecache_teardown(&g_model);
        runstate_free(&g_rs);
        qma_free(&g_model);
        return 0;
    }

    char line[65536];
    for (;;) {
        if (g_shutdown) break;
        c_cyan();
        printf("you> ");
        c_reset();
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) break;
        size_t ll = strlen(line);
        while (ll > 0 && (line[ll-1] == '\n' || line[ll-1] == '\r')) line[--ll] = 0;
        if (ll == 0) continue;
        if (strcmp(line, "/exit") == 0 || strcmp(line, "/quit") == 0) break;
        if (strcmp(line, "/reset") == 0 || strcmp(line, "/clear") == 0) {
            session_reset();
            fprintf(stderr, "qma: context wiped — ingesting system prompt…\n");
            ingest_system_prompt();
            fprintf(stderr, "session reset (n_pos=%d)\n", g_rs.n_pos);
            continue;
        }
        agent_turn(line);
        if (g_snoop_reset) {
            g_snoop_reset = 0;
            session_reset();
            fprintf(stderr, "qma: context wiped — ingesting system prompt…\n");
            ingest_system_prompt();
            fprintf(stderr, "session reset (n_pos=%d)\n", g_rs.n_pos);
            continue;
        }
    }
    thermal_stop();
    finish_session();
    if (g_model.ecache_on) qma_ecache_teardown(&g_model);
    runstate_free(&g_rs);
    qma_free(&g_model);
    return 0;
}
