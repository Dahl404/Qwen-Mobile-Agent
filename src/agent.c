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
#include <sys/ioctl.h>
#include <dirent.h>
#include "qma.h"
#include "json.h"
#include "toolparse.h"
#include "tools.h"
#include "thermal.h"
#include "selfctx.h"
#include "intern.h"

static const char *DEFAULT_MODEL = "/data/data/com.termux/files/home/projects/models/qwen3.6:35b:a3b-q4km.gguf.4k";

/* ---- config ---- */
static int g_threads = 8, g_ctx = 65536, g_prefetch = 4;
static float g_temp = 0.8f, g_top_p = 0.95f, g_repeat = 1.1f;
static int g_top_k = 20;
static float g_eos_penalty = 1.5f;
static int g_enable_thinking = 1;
static int g_ecache_mb = 1024;
static int g_use_color = 0;
static int g_reset = 0;
static int g_check_align = 0;

static double now_s2(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static int g_selfctx = 0;         /* snapshot mode (no --session): context lives in a dated copy of the binary */
char g_self_exe[1024] = ""; /* running binary path (readlink /proc/self/exe); shared with tools.c so enter() can carry it */
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
static char g_delta[524288];           /* turn delta AND tool-response delta — never live at once */

/* ---- self-hosting: the embedded internal tree ---- */
static char    g_intern_root[4096] = "";   /* <session>/internal (the /internal/ fs) */
static uint64_t g_intern_gen = 0;           /* generation serial of the running binary */

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

/* ---- line editor for the `you> ` prompt ----
   Raw-mode input with arrow-key editing: Left/Right move the cursor
   (utf8-aware), Up/Down walk the prompt history, Backspace/Delete edit at
   the cursor, Home/End jump, Enter submits. Ctrl-C still works (ISIG stays
   on). Only used on a real tty; non-tty input stays on plain fgets. The
   terminal is restored before returning, so the generation-time ESC snoop
   (key_snoop_start/stop) always sees canonical mode. */

#define EDIT_HIST_MAX 16
#define EDIT_HIST_LEN 4096
static char  g_ehist[EDIT_HIST_MAX][EDIT_HIST_LEN];  /* prompts, newest last */
static int   g_ehist_n = 0;
static int   g_ehist_pos = -1;                       /* -1 = fresh line */
static char  g_ehist_draft[65536];                   /* in-progress line */

/* display columns of s[0..n): non-continuation bytes count 1 each */
static size_t edit_cols(const char *s, size_t n) {
    size_t c = 0;
    for (size_t i = 0; i < n; i++)
        if (((unsigned char)s[i] & 0xC0) != 0x80) c++;
    return c;
}

/* byte position one utf8 char left / right of the cursor */
static size_t edit_left(const char *s, size_t cur) {
    if (cur == 0) return 0;
    size_t i = cur - 1;
    while (i > 0 && ((unsigned char)s[i] & 0xC0) == 0x80) i--;
    return i;
}
static size_t edit_right(const char *s, size_t len, size_t cur) {
    if (cur >= len) return len;
    size_t i = cur + 1;
    while (i < len && ((unsigned char)s[i] & 0xC0) == 0x80) i++;
    return i;
}

/* redraw the whole input line: back to the prompt's first row (wrapped
   lines), \r + prompt + buffer, clear the tail, park the cursor at `cur`
   (display columns, so utf8 keeps the cursor aligned) */
static void edit_redraw(const char *buf, size_t len, size_t cur) {
    struct winsize ws;
    int cols = 80;
    if (ioctl(1, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) cols = ws.ws_col;
    const size_t total = 5 + edit_cols(buf, len);        /* "you> " = 5 cols */
    const int rows = (int)((total + (size_t)cols - 1) / (size_t)cols);
    if (rows > 1) printf("\x1b[%dA", rows - 1);        /* up to the prompt row */
    printf("\r");
    c_cyan(); printf("you> "); c_reset();
    fputs(buf, stdout);
    fputs("\x1b[K", stdout);                            /* clear row tail */
    size_t back = edit_cols(buf, len) - edit_cols(buf, cur);
    if (back > 0) printf("\x1b[%zuD", back);
    fflush(stdout);
}

/* pull history entry into the buffer (newest last: index n-1) */
static void edit_load_hist(char *buf, size_t cap, int idx) {
    size_t n = strlen(g_ehist[idx]);
    if (n >= cap) n = cap - 1;
    memcpy(buf, g_ehist[idx], n);
    buf[n] = 0;
}

/* record the submitted line in the history ring (dedup against the last) */
static void edit_save_hist(const char *buf) {
    if (buf[0] == 0) return;
    if (g_ehist_n > 0 && strcmp(g_ehist[g_ehist_n - 1], buf) == 0) return;
    if (g_ehist_n == EDIT_HIST_MAX) {
        memmove(g_ehist[0], g_ehist[1], (size_t)(EDIT_HIST_MAX - 1) * EDIT_HIST_LEN);
    } else {
        g_ehist_n++;
    }
    snprintf(g_ehist[g_ehist_n - 1], EDIT_HIST_LEN, "%s", buf);
}

/* read one line with editing. Returns length (>= 0) or -1 on EOF/shutdown. */
static int read_line_editor(char *buf, size_t cap) {
    struct termios save, raw;
    if (tcgetattr(0, &save) != 0) return -1;
    raw = save;
    raw.c_lflag &= ~(ICANON | ECHO);   /* raw-ish; ISIG stays on for Ctrl-C */
    raw.c_cc[VMIN] = 1;
    raw.c_cc[VTIME] = 0;
    tcsetattr(0, TCSANOW, &raw);

    size_t len = 0, cur = 0;
    buf[0] = 0;
    edit_redraw(buf, 0, 0);
    struct pollfd pfd = { .fd = 0, .events = POLLIN };
    int rc = 0;
    while (!rc) {
        if (g_shutdown) { rc = -1; break; }
        unsigned char c;
        ssize_t n = read(0, &c, 1);
        if (n != 1) continue;                      /* EINTR */
        if (c == '\r' || c == '\n') { rc = (int)len; break; }
        if (c == 0x1b) {
            /* ESC [ x / ESC O x, or a bare ESC (ignored) */
            if (poll(&pfd, 1, 30) > 0) {
                unsigned char c2;
                if (read(0, &c2, 1) == 1 && (c2 == '[' || c2 == 'O')) {
                    if (poll(&pfd, 1, 30) > 0) {
                        unsigned char c3;
                        if (read(0, &c3, 1) == 1) {
                            if (c2 == '[' && c3 >= '0' && c3 <= '9') {
                                /* CSI n ~ : 1~/7~ home, 3~ delete, 4~/8~ end */
                                if (poll(&pfd, 1, 30) > 0) {
                                    unsigned char c4;
                                    if (read(0, &c4, 1) == 1 && c4 == '~') {
                                        if (c3 == '1' || c3 == '7') cur = 0;
                                        else if (c3 == '4' || c3 == '8') cur = len;
                                        else if (c3 == '3') {   /* delete */
                                            size_t e = edit_right(buf, len, cur);
                                            memmove(buf + cur, buf + e, len - e);
                                            len -= e - cur;
                                        }
                                    }
                                }
                            } else if (c2 == '[') {
                                switch (c3) {
                                case 'A':   /* up: older */
                                    if (g_ehist_n > 0) {
                                        if (g_ehist_pos < 0) {
                                            snprintf(g_ehist_draft, sizeof(g_ehist_draft), "%s", buf);
                                            g_ehist_pos = g_ehist_n - 1;
                                        } else if (g_ehist_pos > 0) {
                                            g_ehist_pos--;
                                        }
                                        edit_load_hist(buf, cap, g_ehist_pos);
                                        len = strlen(buf); cur = len;
                                    }
                                    break;
                                case 'B':   /* down: newer */
                                    if (g_ehist_pos >= 0) {
                                        if (g_ehist_pos < g_ehist_n - 1) g_ehist_pos++;
                                        else { g_ehist_pos = -1; len = strlen(g_ehist_draft);
                                               if (len >= cap) len = cap - 1;
                                               memcpy(buf, g_ehist_draft, len); buf[len] = 0; }
                                        if (g_ehist_pos >= 0) {
                                            edit_load_hist(buf, cap, g_ehist_pos);
                                            len = strlen(buf);
                                        }
                                        cur = len;
                                    }
                                    break;
                                case 'C':   /* right */
                                    cur = edit_right(buf, len, cur);
                                    break;
                                case 'D':   /* left */
                                    cur = edit_left(buf, cur);
                                    break;
                                case 'H':   /* home */
                                    cur = 0;
                                    break;
                                case 'F':   /* end */
                                    cur = len;
                                    break;
                                }
                            } else if (c2 == 'O') {   /* SS3: H/F arrows */
                                if (c3 == 'H') cur = 0;
                                else if (c3 == 'F') cur = len;
                                else if (c3 == 'C') cur = edit_right(buf, len, cur);
                                else if (c3 == 'D') cur = edit_left(buf, cur);
                            }
                        }
                    }
                }
            }
        } else if (c == 0x7f || c == 0x08) {
            /* backspace: delete the char before the cursor */
            if (cur > 0) {
                size_t s = edit_left(buf, cur);
                memmove(buf + s, buf + cur, len - cur);
                len -= cur - s;
                cur = s;
            }
        } else if (c >= 0x20) {
            /* printable (incl. utf8 continuation bytes) — insert at cursor */
            if (len + 1 < cap) {
                memmove(buf + cur + 1, buf + cur, len - cur);
                buf[cur++] = (char)c;
                len++;
            }
        }
        buf[len] = 0;
        edit_redraw(buf, len, cur);
    }
    tcsetattr(0, TCSANOW, &save);
    if (rc > 0) { buf[len] = 0; edit_save_hist(buf); }
    printf("\n");
    fflush(stdout);
    return rc;
}

/* input line: line editor on a tty, plain fgets otherwise */
static int read_input_line(char *line, size_t cap) {
    if (!g_stdin_tty) {
        if (!fgets(line, cap, stdin)) return -1;
        return (int)strlen(line);
    }
    return read_line_editor(line, cap);
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
    /* persist salience + heavy-hitter arena so pinned facts survive a
       restart (the agent archive itself is rebuilt from memory.json) */
    snprintf(p, sizeof(p), "%s/salience.bin", g_session_dir);
    hcm_save(p);
}

static void session_wipe_files(void);   /* fwd (defined below) */

/* ---- self-hosting: mount the embedded internal tree ---- */
static void mkdirs(const char *path);   /* defined below (session helpers) */
static void intern_init(void) {
    g_intern_gen = 0;
    g_intern_root[0] = 0;
    selfctx_hdr_t h;
    if (!g_self_exe[0] || !selfctx_detect(g_self_exe, &h) || h.int_size == 0) {
        fprintf(stderr, "qma: no embedded internal tree (pre-self-hosting binary)\n");
        return;
    }
    int fd = open(g_self_exe, O_RDONLY);
    if (fd < 0) return;
    char root[1200];
    snprintf(root, sizeof(root), "%s/internal", g_session_dir);
    mkdirs(root);
    int rc = intern_extract_region(fd, (off_t)h.int_off, h.int_size, root);
    close(fd);
    if (rc != 0) { fprintf(stderr, "qma: internal tree extraction failed\n"); return; }
    g_intern_gen = h.gen;
    snprintf(g_intern_root, sizeof(g_intern_root), "%s", root);
    intern_set_root(g_intern_root);          /* tools.c /internal/ mapping */
    setenv("QMA_INTERNAL", g_intern_root, 1);
    /* bootstrap the version log if this tree has none yet */
    {
        char v[1200];
        snprintf(v, sizeof(v), "%s/VERSIONS.md", root);
        struct stat st;
        if (stat(v, &st) != 0) {
            char ts[32];
            time_t t = time(NULL); struct tm tm; localtime_r(&t, &tm);
            snprintf(ts, sizeof(ts), "%04d-%02d-%02d %02d:%02d",
                     tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min);
            intern_log(root, "# qma generation log\n\n## gen %llu — %s — built\n",
                       (unsigned long long)h.gen, ts);
        }
    }
    uint32_t nf = 0; uint64_t nb = 0;
    intern_stats(root, &nf, &nb);
    fprintf(stderr, "qma: internal tree mounted at %s [gen %llu, %u files, %.1f KB]\n",
            root, (unsigned long long)g_intern_gen, nf, (double)nb / 1024.0);
}

static void write_file_at(const char *dir, const char *name, const char *content) {
    char p[4200];
    snprintf(p, sizeof(p), "%s/%s", dir, name);
    FILE *f = fopen(p, "w");
    if (f) { fputs(content, f); fclose(f); }
}

/* delete older generation binaries (qma-gen<N>-...) in out_dir, keeping the
   newest gen <= keep_gen. The bare base binary is never touched. */
static void intern_retain(const char *out_dir, uint64_t keep_gen) {
    DIR *d = opendir(out_dir);
    if (!d) return;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (strncmp(e->d_name, "qma-gen", 7) != 0) continue;
        const char *p = e->d_name + 7;
        uint64_t g = 0;
        while (*p >= '0' && *p <= '9') { g = g * 10 + (uint64_t)(*p - '0'); p++; }
        if (p == e->d_name + 7 || *p != '-') continue;   /* not a gen name */
        if (g < keep_gen) {
            char fp[1200];
            snprintf(fp, sizeof(fp), "%s/%s", out_dir, e->d_name);
            unlink(fp);
            fprintf(stderr, "qma: pruned old generation %s\n", e->d_name);
        }
    }
    closedir(d);
}

/* snapshot the session into a dated copy of the binary, then remove the
   temp session dir. Only in snapshot mode (no --session).
   Self-hosting: on a clean exit qma recompiles itself from the (possibly
   agent-edited) internal src, smoke-tests the fresh binary, and if it
   passes, embeds the whole internal tree + context into the next
   generation. On any failure it falls back to the current binary + the
   edited internal tree + rebuild.log, so the agent can fix its own code
   next boot. */
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

    char *out = NULL;
    const uint64_t gen = g_intern_gen;

    if (g_intern_root[0]) {
        /* ---- recompile from the edited internal source ---- */
        char src_dir[1200], tmpbin[1200], cmd[8192], log[65536];
        snprintf(src_dir, sizeof(src_dir), "%s/src", g_intern_root);
        snprintf(tmpbin, sizeof(tmpbin), "%s/selfbuild", g_session_dir);
        char ts[32];
        {
            time_t t = time(NULL); struct tm tm; localtime_r(&t, &tm);
            snprintf(ts, sizeof(ts), "%04d-%02d-%02d %02d:%02d",
                     tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min);
        }
        fprintf(stderr, "qma: rebuilding self from %s (gen %llu → %llu)…\n",
                src_dir, (unsigned long long)gen, (unsigned long long)(gen + 1));
        intern_build_cmd(cmd, sizeof(cmd), src_dir, tmpbin);
        int rc = sys_run_capture(cmd, 240, log, sizeof(log));
        if (rc == 0) {
            /* smoke-test the fresh binary */
            char smoke[8192], s2[4096];
            snprintf(smoke, sizeof(smoke), "%s --check-align 2>&1", tmpbin);
            int src = sys_run_capture(smoke, 30, s2, sizeof(s2));
            if (src == 0) {
                out = selfctx_snapshot_gen(tmpbin, g_session_dir, out_dir,
                                           g_model_path, strlen(g_model_path) + 1,
                                           g_intern_root, gen + 1);
                if (out) {
                    intern_log(g_intern_root,
                        "## gen %llu — %s — build OK (smoke passed) — new binary %s\n",
                        (unsigned long long)(gen + 1), ts, out);
                    intern_retain(out_dir, gen + 1);
                } else {
                    intern_log(g_intern_root,
                        "## gen %llu — %s — build OK but snapshot failed\n",
                        (unsigned long long)(gen + 1), ts);
                }
            } else {
                intern_log(g_intern_root,
                    "## gen %llu — %s — build OK but SMOKE FAILED (see rebuild.log)\n",
                    (unsigned long long)(gen + 1), ts);
                write_file_at(g_intern_root, "rebuild.log", s2);
                fprintf(stderr, "qma: self-build compiled but failed smoke test — keeping current binary\n");
            }
        } else {
            intern_log(g_intern_root,
                "## gen %llu — %s — build FAILED (see rebuild.log)\n",
                (unsigned long long)(gen + 1), ts);
            write_file_at(g_intern_root, "rebuild.log", log);
            fprintf(stderr, "qma: self-build failed (rc=%d) — keeping current binary, errors → /internal/rebuild.log\n", rc);
        }
        unlink(tmpbin);
    }

    if (!out) {
        /* fallback: current binary + (edited) internal tree + context, same gen */
        out = selfctx_snapshot_gen(g_self_exe, g_session_dir, out_dir,
                                   g_model_path, strlen(g_model_path) + 1,
                                   g_intern_root[0] ? g_intern_root : NULL, gen);
        if (out) intern_retain(out_dir, gen);
    }
    if (out) {
        fprintf(stderr, "qma: generation → %s\n", out);
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

    /* one-time: repack the model with 4K-aligned tensor starts so expert
       reads can use O_DIRECT (see gguf.c). The aligned copy lives next to
       the original (<model>.4k) and the config is updated to point at it.
       Skipped with QMA_NOALIGN=1; falls back to the unaligned file when
       there isn't room. */
    if (!g_check_align && getenv("QMA_NOALIGN") == NULL) {
        char aligned[4096];
        char aerr[512];
        if (qma_align_model(cand, aligned, sizeof(aligned), aerr, sizeof(aerr)) == 0) {
            if (strcmp(aligned, cand) != 0)
                fprintf(stderr, "qma: using aligned model %s\n", aligned);
            snprintf(cand, sizeof(cand), "%s", aligned);
        } else if (aerr[0]) {
            fprintf(stderr, "qma: model alignment skipped: %s\n", aerr);
        }
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
    /* Restore HCM salience + heavy-hitter arena so pinned facts survive a
       restart. (The agent archive itself is rebuilt from memory.json after
       the system prompt is ingested — see agent_memory_restore().) */
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
    static char sys_wrapped[270000];   /* one-time boot ingest; BSS, not stack */
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
    char *content;          /* text content (tool XML excluded) */
    size_t content_len;
    size_t content_cap;
    tool_call_t *calls;     /* dynamically grown array of tool calls */
    int n_calls;
    int calls_cap;
    int had_tool_mode;      /* model attempted a call (even if dropped) */
    int cancelled;          /* generation stopped by the user (ESC) */
} gen_result_t;

static void apply_eos_penalty(const qma_t *m, float *logits) {
    if (g_eos_penalty <= 0.0f) return;
    for (int i = 0; i < m->n_vocab; i++)
        if (m->tok_type[i] & 4) logits[i] -= g_eos_penalty;
}

static void content_append(gen_result_t *res, const char *s, size_t n) {
    if (res->content_len + n + 1 > res->content_cap) {
        size_t new_cap = res->content_cap * 2;
        if (new_cap < res->content_len + n + 1) {
            new_cap = res->content_len + n + 1 + 4096;
        }
        char *new_content = realloc(res->content, new_cap);
        if (!new_content) return;
        res->content = new_content;
        res->content_cap = new_cap;
    }
    memcpy(res->content + res->content_len, s, n);
    res->content_len += n;
    res->content[res->content_len] = 0;
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

/* 1 if `call` (a tool-call marker) sits inside an unclosed <think> block
   in `ans` — i.e. the last <think> opener before it has no </think>
   before it. The model plans tool calls inside its thinking; those must
   never execute. The real call (emitted after </think>) is unaffected. */
static int inside_think_block(const char *ans, const char *call) {
    const char *scan = ans;
    const char *last_open = NULL;
    while ((scan = strstr(scan, "<think>")) != NULL && scan < call) {
        last_open = scan;
        scan += strlen("<think>");
    }
    if (!last_open) return 0;
    const char *cl = strstr(last_open + strlen("<think>"), "</think>");
    if (!cl) return 1;              /* think block never closed: still planning */
    return cl > call;               /* closed after the call: call is inside it */
}

static int agent_generate(qma_t *m, const char *prompt, int enable_thinking,
                          gen_result_t *res) {
    memset(res, 0, sizeof(*res));
    res->content_cap = 131072;
    res->content = malloc(res->content_cap);
    if (!res->content) return -1;
    res->content[0] = '\0';
    res->content_len = 0;
    res->calls = NULL;
    res->n_calls = 0;
    res->calls_cap = 0;

    g_stop_turn = 0;   /* fresh turn: clear any earlier soft stop (ESC) */
    double t_pre0 = now_s2();
    int *ids = malloc(sizeof(int) * 262144);
    if (!ids) { free(res->content); return -1; }
    int nids = qma_tokenize(m, prompt, ids, 262144);
    if (nids < 0) { free(res->content); free(ids); return -1; }
    float *logits = malloc(sizeof(float) * N_VOCAB);
    if (!logits) { free(res->content); free(ids); return -1; }
    if (qma_eval(m, &g_rs, ids, nids, logits, g_threads, g_prefetch, 0) != 0) {
        free(res->content); free(ids); free(logits); return -1;
    }
    double t_pre1 = now_s2();
    apply_eos_penalty(m, logits);

    sampler_t sp;
    sampler_init(&sp, (uint64_t)time(NULL), g_temp, g_top_k, g_top_p, g_repeat);

    char *raw = malloc(RAW_CAP);
    char *ans = malloc(RAW_CAP);
    if (!raw || !ans) { free(res->content); free(raw); free(ans); free(ids); free(logits); return -1; }
    size_t raw_len = 0, ans_len = 0, reasoning_emitted = 0;
    int thinking_done = enable_thinking ? 0 : 1;
    int tool_mode = 0;
    int ended_clean = 0;   /* the model closed the turn itself (<|im_end|>) */
    size_t tool_scan = 0;
    /* scan cursors: raw/ans grow by appending, but ans is also front-trimmed
       (holdback emit, draft rewind, tool-call extraction). Each scan restarts
       11 bytes before the cursor so a tag split across an append boundary is
       still caught (all tags are <= 12 bytes; the overlap guarantees full
       coverage of every position). */
    size_t think_scan = 0;   /* "</think>" scan position in raw  */
    size_t ans_scan = 0;     /* "</think>"/"<tool_call>" scan position in ans */
    int ngen = 0;

    int cap = 262144;
    c_dim();
    int cancelled = 0;
    for (int i = 0; i < cap; i++) {
        if (g_stop_turn) { cancelled = 1; break; }
        /* NO tool-call grammar mask (ma3 scope decision — the mask CAUSED
           failures): when every top-k candidate was masked, sampler_pick
           fell back to the masked top token, injecting an illegal char that
           corrupted the open call — and the model then ran away emitting
           garbage forever. Calls are constrained by POLICY instead: the
           turn is force-ended after ONE complete call (force_end below),
           the result is folded back, and the model continues — one
           grounded step at a time. */
        int cand_ids[40];
        float cand_lgs[40];
        int nc = sampler_candidates(N_VOCAB, logits, &sp, cand_ids, cand_lgs, 40);
        int id = nc > 0 ? sampler_pick(&sp, cand_ids, cand_lgs, nc, g_top_p) : 0;
        if (id < 0 || id >= N_VOCAB) break;
        if (id == (int)m->id_im_end || (m->tok_type[id] & 4)) {
            ended_clean = 1;
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
            /* reasoning phase — scan only the bytes appended since the last
               token (plus an 11-byte overlap for tags split across pieces).
               Reasoning ends at </think> OR at an unpaired <tool_call> —
               qwen3.x natively stops thinking the moment it starts a call
               (vLLM is_reasoning_end), it does NOT always emit </think>.
               Without this, calls drafted at the end of thinking streamed
               as reasoning and never executed. */
            size_t ts = think_scan > 11 ? think_scan - 11 : 0;
            char *close = strstr(raw + ts, "</think>");
            char *tcall = strstr(raw + ts, "<tool_call>");
            const char *endtag = NULL;
            if (close && (!tcall || close <= tcall)) endtag = "</think>";
            else if (tcall) endtag = "<tool_call>";
            think_scan = raw_len;
            if (endtag) {
                const size_t etl = strlen(endtag);
                char *mk = strstr(raw + ts, endtag);
                size_t pos = (size_t)(mk - raw) + etl;
                if (pos > reasoning_emitted) {
                    size_t rbl = pos - reasoning_emitted;
                    char buf[4200];
                    size_t n0 = rbl < sizeof(buf) - 1 ? rbl : sizeof(buf) - 1;
                    memcpy(buf, raw + reasoning_emitted, n0);
                    buf[n0] = 0;
                    size_t tlen = strlen(buf);
                    if (tlen >= etl && strcmp(buf + tlen - etl, endtag) == 0)
                        buf[tlen - etl] = 0;
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
                ans_scan = 0;   /* fresh ans buffer — rescan from the top */
                c_reset();
            } else if (pl > 0) {
                print_piece(piece);
                reasoning_emitted = raw_len;
            }
        } else if (tool_mode) {
            /* tool phase: show raw tokens dim; emit completed valid calls.
               Native qwen3.x behavior (multi_step_tool): the model may emit
               SEVERAL complete <tool_call> blocks in one assistant message;
               all of them execute, results fold back as one <tool_response>
               turn, and the model continues. We scan for complete blocks
               and record each; generation stops only when the model closes
               the turn itself (<|im_end|>). The parser accepts bare
               <function= blocks (dropped <tool_call> opener) and the
               model's improvised close tags. */
            if (ans_len + (size_t)pl < RAW_CAP) {
                memcpy(ans + ans_len, piece, (size_t)pl);
                ans_len += (size_t)pl;
                ans[ans_len] = 0;
            }
            print_piece(piece);
            for (;;) {
                const char *base = ans + (tool_scan <= ans_len ? tool_scan : ans_len);
                const char *t0 = strstr(base, "<tool_call>");
                const char *f0 = strstr(base, "<function=");
                const char *blk = t0 && (!f0 || t0 <= f0) ? t0 : f0;
                if (!blk) break;
                const char *after = blk;
                tool_call_t tc;
                int r = parse_one_tool_call(ans, &after, &tc);
                if (r == 0) break;             /* open: wait for more tokens */
                if (r == 1) {
                    res->had_tool_mode = 1;
                    if (res->n_calls >= res->calls_cap) {
                        int new_cap = res->calls_cap == 0 ? 4 : res->calls_cap * 2;
                        tool_call_t *new_calls = realloc(res->calls, sizeof(tool_call_t) * new_cap);
                        if (!new_calls) break;
                        res->calls = new_calls;
                        res->calls_cap = new_cap;
                    }
                    snprintf(res->calls[res->n_calls].name,
                             sizeof(res->calls[res->n_calls].name), "%s", tc.name);
                    snprintf(res->calls[res->n_calls].args,
                             sizeof(res->calls[res->n_calls].args), "%s", tc.args);
                    res->n_calls++;
                    c_reset(); c_cyan();
                    flush_piece();   /* pending raw tokens before the ⬡ line */
                    printf("\n⬡ %s(%s)\n", tc.name, tc.args);
                    c_reset(); c_dim();
                } else {
                    /* malformed complete block — dropped; the caller's
                       had_tool_mode && n_calls==0 path reports it */
                    res->had_tool_mode = 1;
                    tool_scan = (size_t)(after - ans);
                    break;             /* malformed: stop scanning this turn */
                }
                tool_scan = (size_t)(after - ans);
                /* continue scanning for the NEXT complete block (multi-call) */
            }
        } else {
            /* answer phase: content with holdback; detect <tool_call> */
            if (ans_len + (size_t)pl < RAW_CAP) {
                memcpy(ans + ans_len, piece, (size_t)pl);
                ans_len += (size_t)pl;
                ans[ans_len] = 0;
            }
            /* stray </think> rewind (Qwen3 drafts the answer twice) — cursor
               scans only the new region (11-byte overlap for split tags) */
            size_t as = ans_scan > 11 ? ans_scan - 11 : 0;
            char *stray = strstr(ans + as, "</think>");
            if (stray) {
                size_t rest = ans_len - (size_t)(stray - ans) - 8;
                memmove(ans, stray + 8, rest);
                ans_len = rest;
                ans[ans_len] = 0;
                ans_scan = 0;   /* front trimmed through the tag — rescan from top */
                /* the first draft was already streamed: mark the rewind so
                   the duplication is explained, not read as engine output */
                flush_piece();
                c_dim();
                fputs("\n[draft rewound — the model re-drafted its answer]\n", stdout);
                c_reset();
            } else {
                ans_scan = ans_len;   /* no rewind: cursor = current end */
            }
            char *tc = strstr(ans + (stray ? 0 : as), "<tool_call>");
            char *fn = strstr(ans + (stray ? 0 : as), "<function=");
            char *tc0 = tc && (!fn || tc <= fn) ? tc : fn;
            if (tc0 && inside_think_block(ans, tc0)) {
                /* the model is PLANNING — the call sits inside an unclosed
                   <think> block (it drafted the call as part of reasoning).
                   Do NOT execute it. Stream it dim like thinking and keep
                   scanning; the real call (after </think>) executes normally. */
                size_t pre = (size_t)(tc0 - ans);
                if (pre > 0) {
                    size_t el = u8_safe_len(ans, pre);
                    c_reset();
                    print_content(res, ans, el);
                }
                size_t rest = ans_len - pre;
                memmove(ans, ans + pre, rest);
                ans_len = rest;
                ans[ans_len] = 0;
                ans_scan = 0;   /* rescan from the top; skip the planned call */
                c_dim();
            } else if (tc0) {
                size_t pre = (size_t)(tc0 - ans);
                if (pre > 0) {
                    size_t el = u8_safe_len(ans, pre);
                    c_reset();
                    print_content(res, ans, el);
                }
                tool_mode = 1;
                res->had_tool_mode = 1;   /* model attempted a call */
                /* repeat-penalty exemption: the tag skeleton (<parameter=,
                   </parameter>, <function=, </function>) MUST repeat
                   verbatim for every parameter in a multi-arg call. With
                   the penalty still active it actively suppresses the
                   model's own required tag tokens on the 2nd+ parameter —
                   the sampler fighting the schema it's supposed to be
                   producing. Disable it for the rest of this call. */
                sp.repeat_penalty = 1.0f;
                tool_scan = 0;
                ans_scan = 0;   /* ans now starts at the call */
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
                ans_scan = ans_len;   /* holdback trimmed el bytes from the front */
            }
        }
        if (qma_eval(m, &g_rs, &id, 1, logits, g_threads, g_prefetch, 0) != 0) break;
        apply_eos_penalty(m, logits);
        ngen++;
    }
    c_reset();
    flush_piece();

    /* multi-call tool turns: if the model never emitted <|im_end|> itself
       (token cap), close the assistant frame so the caller's tool-response
       delta appends to a clean <|im_start|>user boundary. The calls' raw
       text is already in the KV from generation. */
    if (tool_mode && !ended_clean) {
        int ie = (int)m->id_im_end;
        if (ie > 0) qma_eval(m, &g_rs, &ie, 1, logits, g_threads, g_prefetch, 0);
    }

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

    /* finalize: flush any complete blocks, then remaining content. The
       streaming path already recorded complete blocks; this catches any
       block that completed right at the turn end. */
    if (tool_mode) {
        for (;;) {
            const char *base = ans + (tool_scan <= ans_len ? tool_scan : ans_len);
            const char *t0 = strstr(base, "<tool_call>");
            const char *f0 = strstr(base, "<function=");
            const char *blk = t0 && (!f0 || t0 <= f0) ? t0 : f0;
            if (!blk) break;
            const char *after = blk;
            tool_call_t tc;
            /* at_end = 1: accept a block whose parameters all closed even
               if the model dropped </function> (the turn is over — it is
               not going to write more) */
            int r = parse_one_tool_call_full(ans, &after, &tc, 1);
            if (r == 1) {
                res->had_tool_mode = 1;
                if (res->n_calls >= res->calls_cap) {
                    int new_cap = res->calls_cap == 0 ? 4 : res->calls_cap * 2;
                    tool_call_t *new_calls = realloc(res->calls, sizeof(tool_call_t) * new_cap);
                    if (!new_calls) break;
                    res->calls = new_calls;
                    res->calls_cap = new_cap;
                }
                snprintf(res->calls[res->n_calls].name, sizeof(res->calls[res->n_calls].name), "%s", tc.name);
                snprintf(res->calls[res->n_calls].args, sizeof(res->calls[res->n_calls].args), "%s", tc.args);
                res->n_calls++;
                c_cyan();
                flush_piece();   /* pending raw tokens before the ⬡ line */
                printf("\n⬡ %s(%s)\n", tc.name, tc.args);
                c_reset();
            }
            /* r == 0 (block still open at turn end) or -1 (malformed):
               nothing more to ship. r==0 does NOT advance *ppos, so we
               MUST break here — looping would rescan the same open block
               forever (a hang made reachable now that the grammar mask no
               longer forces calls to complete). */
            tool_scan = (size_t)(after - ans);
            if (r != 1) break;
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
    /* ensure buf is at least null-terminated even if we return early */
    if (buflen > 0) buf[0] = '\0';
    size_t o = 0;
    #define APP(...) do { \
        int w = snprintf(buf + o, buflen - o, __VA_ARGS__); \
        if (w < 0 || (size_t)w >= buflen - o) return -1; \
        o += (size_t)w; \
    } while (0)
    APP("<|im_start|>user\n");
    if (note && note[0]) APP("<tool_response>\n%s\n</tool_response>", note);
    for (int i = 0; i < n; i++)
        APP("<tool_response>\n%s\n</tool_response>", results[i]);
    APP("<|im_end|>\n<|im_start|>assistant\n<think>\n");
    #undef APP
    return (int)o;
}

/* ---- dynamic builder for tool responses (no fixed limit) ---- */
static char *build_tool_delta_dynamic(const char *const *results, int n, const char *note) {
    size_t cap = 4096, len = 0;
    char *buf = malloc(cap);
    if (!buf) return NULL;

    #define APPEND(...) do { \
        int needed = snprintf(NULL, 0, __VA_ARGS__); \
        if (needed < 0) goto error; \
        if ((size_t)needed + len + 1 > cap) { \
            cap = (len + (size_t)needed + 1) * 2; \
            char *newbuf = realloc(buf, cap); \
            if (!newbuf) goto error; \
            buf = newbuf; \
        } \
        int written = snprintf(buf + len, cap - len, __VA_ARGS__); \
        if (written < 0) goto error; \
        len += (size_t)written; \
    } while(0)

    APPEND("<|im_start|>user\n");
    if (note && note[0]) APPEND("<tool_response>\n%s\n</tool_response>", note);
    for (int i = 0; i < n; i++) {
        const char *res = results[i] ? results[i] : "";
        APPEND("<tool_response>\n%s\n</tool_response>", res);
    }
    APPEND("<|im_end|>\n<|im_start|>assistant\n<think>\n");

    buf[len] = '\0';
    return buf;

error:
    free(buf);
    return NULL;
}
#undef APPEND

/* ---- agent turn ---- */
static int agent_turn(const char *input) {
    char *cur = strdup(input);
    int pre_rendered = 0;   /* cur is already a rendered template delta */
    int turns = 0;
    /* ~258KB struct (131KB content + 8 tool slots): keep it off the stack */
    gen_result_t *res = calloc(1, sizeof(*res));
    if (!res) { free(cur); return -1; }

    /* pi-agent loop: call the model, execute any tool calls, fold the
       results back and call again — the loop continues ONLY while the
       model calls tools. A text-only response ends the turn and control
       returns to the user prompt; the agent never nudges the model to
       keep going (no [continue], no DONE parsing). MAX_TURNS is the only
       runaway guard, exactly like pi's turn budget. g_delta is shared
       between the user delta and the tool-response delta — the two are
       never live at the same time (the tool delta is strdup'd into cur
       before the buffer is reused on the next iteration). */
    for (;;) {
        if (g_shutdown) break;
        if (++turns > MAX_TURNS) {
            c_yellow(); printf("\n[stopping: turn budget reached]\n"); c_reset();
            break;
        }
        if (pre_rendered) {
            snprintf(g_delta, sizeof(g_delta), "%s", cur);
            pre_rendered = 0;
        } else {
            build_user_delta(g_delta, sizeof(g_delta), cur);
        }
        key_snoop_start();
        int gr = agent_generate(&g_model, g_delta, g_enable_thinking, res);
        key_snoop_stop();
        if (gr != 0) {
            c_red(); printf("\n[generation failed]\n"); c_reset();
            free(cur);
            free(res->content);
            free(res->calls);
            free(res);
            return -1;
        }
        printf("\n");

        if (res->cancelled) {
            c_yellow(); printf("[cancelled — back to you]\n"); c_reset();
            free(cur);
            free(res->content);
            free(res->calls);
            free(res);
            return 0;
        }

        if (res->n_calls > 0) {
            /* execute every call, fold results back, call the model again */
            /* allocate results array dynamically to avoid overflow */
            int n_res = res->n_calls;
            char **results = malloc(sizeof(char *) * n_res);
            if (!results) {
                c_red(); printf("[error: out of memory for tool results]\n"); c_reset();
                free(cur);
                free(res->content);
                free(res->calls);
                free(res);
                return -1;
            }
            for (int i = 0; i < n_res; i++) {
                char *r = NULL;
                tool_dispatch(res->calls[i].name, res->calls[i].args, &r);
                if (strlen(r) > 6000) {
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
                results[i] = r;
                if (strncmp(r, "ERROR:", 6) == 0) {
                    c_red(); printf("  ✕ %s\n", r); c_reset();
                } else {
                    c_green(); printf("  ✓ %s\n", r); c_reset();
                }
            }

            /* Use the dynamic builder; fallback to static if allocation fails */
            char *delta = build_tool_delta_dynamic((const char **)results, n_res, NULL);
            if (!delta) {
                /* Fallback: use static builder with a note */
                build_tool_delta(g_delta, sizeof(g_delta), (const char **)results, n_res,
                                 "ERROR: internal memory allocation failed");
                delta = strdup(g_delta);
            }
            for (int i = 0; i < n_res; i++) free(results[i]);
            free(results);
            free(cur);
            cur = delta;
            pre_rendered = 1;
            free(res->content);
            free(res->calls);
            res->content = NULL;
            res->calls = NULL;
            res->content_len = 0;
            res->content_cap = 0;
            res->n_calls = 0;
            res->calls_cap = 0;
            continue;
        }

        if (res->had_tool_mode && res->n_calls == 0) {
            /* the model's calls were all malformed and the engine dropped
               them — feed that back as a tool error (exactly what pi does
               for a failed tool) and continue; bounded by MAX_TURNS. Show
               the EXACT format so the model can self-correct instead of
               flailing. ALSO print it to the console — the user must see
               what the model attempted and what it was told. */
            c_red();
            printf("  ✕ malformed tool call (not executed) — fed back to the model:\n");
            c_reset();
            build_tool_delta(g_delta, sizeof(g_delta), NULL, 0,
                "ERROR: your tool call was malformed and discarded — it NEVER "
                "executed. The ONLY accepted format is EXACTLY:\n"
                "<tool_call>\n"
                "<function=TOOL_NAME>\n"
                "<parameter=ARG_NAME>\n"
                "value\n"
                "</parameter>\n"
                "</function>\n"
                "</tool_call>\n"
                "Rules:\n"
                "- wrap each call in <tool_call>...</tool_call> (you may emit "
                "several per message when the steps are independent)\n"
                "- put each value on its own line under its <parameter=...> tag\n"
                "- close every <parameter=...> with </parameter>\n"
                "- END the call with </function> (a call is only complete "
                "when </function> or </tool_call> closes it)\n"
                "Re-issue the call now in this exact format.");
            free(cur);
            cur = strdup(g_delta);
            pre_rendered = 1;
            free(res->content);
            free(res->calls);
            res->content = NULL;
            res->calls = NULL;
            res->content_len = 0;
            res->content_cap = 0;
            res->n_calls = 0;
            res->calls_cap = 0;
            continue;
        }

        /* text-only response: the model decided it's done — end the turn */
        break;
    }
    free(cur);
    free(res->content);
    free(res->calls);
    free(res);
    return 0;
}

/* ---- system prompt ---- */
static void build_system_prompt(char *buf, size_t buflen) {
    /* render the tools header straight into the target buffer — no temp
       header array on the stack */
    tools_render_header(buf, buflen);
    size_t o = strlen(buf);
    if (o + 1 < buflen)
        snprintf(buf + o, buflen - o, "\n\n## Agent Behavior\n\n"
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
            "- If asked a simple question with no work needed, just answer it.\n");
    o = strlen(buf);
    if (g_intern_root[0] && o + 1 < buflen)
        snprintf(buf + o, buflen - o, "\n## Self-Hosting\n"
            "You are a self-hosting agent. Your internal filesystem is mounted at /internal/ "
            "(real path: $QMA_INTERNAL). Your own source code is at /internal/src/ — you may "
            "read and modify it. On a clean exit, qma recompiles itself from /internal/src, "
            "smoke-tests the result, and if it passes, carries the new binary AND the whole "
            "/internal/ tree (plus this conversation) forward as the next generation. "
            "Use self_build() to compile + smoke-test your source changes at any time before "
            "they are committed at exit. Version history: /internal/VERSIONS.md. Failed builds "
            "write their compiler errors to /internal/rebuild.log so you can fix your own code "
            "next boot. Collect tools and data under /internal/tools/ and /internal/data/ to "
            "keep them forever.\n");
    o = strlen(buf);
    if (o + 1 < buflen)
        snprintf(buf + o, buflen - o, "\n## Long-Term Memory — YOUR OWN ACTIVE MEMORY\n"
            "This is not a file system, a diary, or external storage. It is YOUR OWN memory:\n"
            "text you write there is embedded into your KV cache as real key/value vectors\n"
            "that you attend to on EVERY token you generate. You never need a tool to read\n"
            "it — you already see it every step, always, forever (it survives the rolling\n"
            "context window and restarts).\n"
            "The tools only MUTATE that memory:\n"
            "- memory_write(\"key\", text) pins new text into it.\n"
            "- memory_append(\"key\", text) adds to an existing entry.\n"
            "- memory_delete(\"key\") forgets an entry.\n"
            "- memory_list() shows what you currently have pinned (keys + sizes).\n"
            "Use it actively, like a scratchpad that stays open in front of you:\n"
            "- Before a multi-step task: memory_write(\"task\", ...) the user's goal and your plan.\n"
            "- After each step: memory_append(\"task\", ...) what you did, what you found, what's next.\n"
            "- Keep anything you must not forget (paths, decisions, constraints, open questions).\n"
            "Because you see it on every token, keeping state there keeps you coherent across\n"
            "the whole task AND across sessions — consolidate early and often.\n");
    o = strlen(buf);
    if (o + 1 < buflen)
        snprintf(buf + o, buflen - o, "\n## Document Workers (LFM sub-agents)\n"
            "You can spawn small worker agents (a 2.6B LFM2.5 model, weights shared) to read\n"
            "documents and answer questions about them — instead of reading a whole file into\n"
            "your own context, load it into a worker and interrogate it:\n"
            "- worker_spawn(\"name\", path=\"...\") loads a file into a new worker (returns an id).\n"
            "- worker_ask(id, question) asks it about the document. Default is EPHEMERAL: each\n"
            "  question gets a clean KV on the same document base (the worker forgets the prior\n"
            "  question). For follow-up chains, pass ephemeral=false — the worker keeps the\n"
            "  conversation so follow-up questions have context.\n"
            "- worker_close(id) frees the worker; worker_list() shows active ids.\n"
            "Use workers for: finding exact lines, checking whether something is in a file,\n"
            "summarizing sections, or comparing multiple documents (spawn one per doc).\n");
}

/* ---- agent memory: the manually-edited archive ring ----
 * memory.json in the session dir is the single source of truth (you can
 * edit it by hand). Each entry {key, content, ts} is mirrored into the KV
 * archive arena (real K/V, always attended) via hcm_archive_*. On boot the
 * arena is rebuilt from the file so memory survives restarts. */

typedef struct { char key[64]; char *content; int64_t ts; } mem_entry_t;

static const char *memory_path(void) {
    static char p[1200];
    if (g_session_dir[0]) snprintf(p, sizeof(p), "%s/memory.json", g_session_dir);
    else snprintf(p, sizeof(p), "./.qma-memory.json");
    return p;
}

/* parse memory.json into a heap array; returns count (0 = none/error) */
static int mem_load(mem_entry_t **out) {
    *out = NULL;
    char *raw = read_whole_file(memory_path());
    if (!raw || strncmp(raw, "ERROR:", 6) == 0) { free(raw); return 0; }
    jval_t *j = json_parse(raw);
    free(raw);
    if (!j || j->type != J_ARR) { if (j) json_free(j); return 0; }
    int n = 0;
    mem_entry_t *ent = calloc(j->n ? j->n : 1, sizeof(mem_entry_t));
    for (size_t i = 0; i < j->n; i++) {
        const jval_t *o = json_arr_get(j, i);
        const char *k = json_str(json_obj_get(o, "key"));
        const char *c = json_str(json_obj_get(o, "content"));
        if (!k[0]) continue;
        snprintf(ent[n].key, sizeof(ent[n].key), "%s", k);
        ent[n].content = strdup(c);
        ent[n].ts = (int64_t)json_num(json_obj_get(o, "ts"));
        n++;
    }
    json_free(j);
    *out = ent;
    return n;
}

static void mem_free(mem_entry_t *ent, int n) {
    for (int i = 0; i < n; i++) free(ent[i].content);
    free(ent);
}

static int mem_find(mem_entry_t *ent, int n, const char *key) {
    for (int i = 0; i < n; i++) if (strcmp(ent[i].key, key) == 0) return i;
    return -1;
}

static int mem_save(mem_entry_t *ent, int n) {
    size_t cap = 4096;
    for (int i = 0; i < n; i++) cap += strlen(ent[i].key) + strlen(ent[i].content) * 2 + 64;
    char *buf = malloc(cap);
    if (!buf) return -1;
    size_t o = 0;
    o += (size_t)snprintf(buf + o, cap - o, "[");
    for (int i = 0; i < n; i++) {
        char ke[160], ce[131072];
        json_quote_escape(ent[i].key, ke, sizeof(ke));
        json_quote_escape(ent[i].content, ce, sizeof(ce));
        int w = snprintf(buf + o, cap - o, "%s{\"key\":%s,\"content\":%s,\"ts\":%lld}",
                         i ? "," : "", ke, ce, (long long)ent[i].ts);
        if (w < 0 || (size_t)w >= cap - o) { free(buf); return -1; }
        o += (size_t)w;
    }
    o += (size_t)snprintf(buf + o, cap - o, "]");
    FILE *f = fopen(memory_path(), "w");
    if (!f) { free(buf); return -1; }
    fwrite(buf, 1, o, f);
    fclose(f);
    free(buf);
    return 0;
}

char *agent_memory_write(const jval_t *args) {
    const char *key = json_str(json_obj_get(args, "key"));
    const char *content = json_str(json_obj_get(args, "content"));
    if (!key[0]) return xstrdup("ERROR: memory_write requires a non-empty key");
    if (!content[0]) return xstrdup("ERROR: memory_write requires non-empty content");
    if (!g_rs_ready) return xstrdup("ERROR: model not ready");
    int rc = hcm_archive_write(&g_model, &g_rs, key, content, g_threads);
    if (rc < 0) {
        if (rc == -2) return xstrdup("ERROR: memory archive requires quantized KV (QMA_KVQ unset — the default)");
        if (rc == -3) return xstrdup("ERROR: memory archive is full — memory_delete() something first");
        return xstrdup("ERROR: failed to embed memory entry (eval error)");
    }
    mem_entry_t *ent = NULL; int n = mem_load(&ent);
    int i = mem_find(ent, n, key);
    if (i < 0) {
        ent = realloc(ent, sizeof(mem_entry_t) * (n + 1));
        if (!ent) return xstrdup("ERROR: OOM");
        i = n;
        snprintf(ent[i].key, sizeof(ent[i].key), "%s", key);
        ent[i].content = NULL;
        n++;
    } else {
        free(ent[i].content);
    }
    ent[i].content = strdup(content);
    ent[i].ts = (int64_t)time(NULL);
    if (mem_save(ent, n) != 0) { mem_free(ent, n); return xstrdup("ERROR: cannot write memory.json"); }
    char *r = fmt("✓ memory '%s' pinned (%d tokens, always attended)", key, rc);
    mem_free(ent, n);
    return r;
}

char *agent_memory_append(const jval_t *args) {
    const char *key = json_str(json_obj_get(args, "key"));
    const char *content = json_str(json_obj_get(args, "content"));
    if (!key[0]) return xstrdup("ERROR: memory_append requires a non-empty key");
    if (!content[0]) return xstrdup("ERROR: memory_append requires non-empty content");
    if (!g_rs_ready) return xstrdup("ERROR: model not ready");
    mem_entry_t *ent = NULL; int n = mem_load(&ent);
    int i = mem_find(ent, n, key);
    /* for a fresh key memory_append behaves like memory_write */
    if (i < 0) {
        mem_free(ent, n);
        return agent_memory_write(args);
    }
    int rc = hcm_archive_append(&g_model, &g_rs, key, content, g_threads);
    if (rc == -4) {
        /* the entry is not the last pinned range (other entries were
           written after it) — appending in place would overwrite them, so
           re-embed the full combined text instead */
        size_t oldl = strlen(ent[i].content), newl = oldl + strlen(content) + 2;
        char *combined = malloc(newl);
        if (!combined) { mem_free(ent, n); return xstrdup("ERROR: OOM"); }
        snprintf(combined, newl, "%s\n%s", ent[i].content, content);
        rc = hcm_archive_write(&g_model, &g_rs, key, combined, g_threads);
        free(combined);
    }
    if (rc < 0) { mem_free(ent, n); return xstrdup("ERROR: failed to append memory entry (eval error)"); }
    size_t oldl = strlen(ent[i].content), newl = oldl + strlen(content) + 2;
    char *nc = malloc(newl);
    if (!nc) { mem_free(ent, n); return xstrdup("ERROR: OOM"); }
    snprintf(nc, newl, "%s\n%s", ent[i].content, content);
    free(ent[i].content);
    ent[i].content = nc;
    ent[i].ts = (int64_t)time(NULL);
    if (mem_save(ent, n) != 0) { mem_free(ent, n); return xstrdup("ERROR: cannot write memory.json"); }
    char *r = fmt("✓ memory '%s' extended (%d tokens)", key, rc);
    mem_free(ent, n);
    return r;
}

char *agent_memory_list(const jval_t *args) {
    (void)args;
    mem_entry_t *ent = NULL; int n = mem_load(&ent);
    if (n == 0) { mem_free(ent, n); return xstrdup("(memory archive is empty)"); }
    size_t cap = 4096 + n * 128;
    char *out = malloc(cap); size_t o = 0;
    for (int i = 0; i < n; i++) {
        int w = snprintf(out + o, cap - o, "%s%-32s %zu tokens\n",
                         i ? "" : "", ent[i].key, strlen(ent[i].content) / 4);
        if (w < 0 || (size_t)w >= cap - o) break;
        o += (size_t)w;
    }
    mem_free(ent, n);
    return out;
}

char *agent_memory_delete(const jval_t *args) {
    const char *key = json_str(json_obj_get(args, "key"));
    if (!key[0]) return xstrdup("ERROR: memory_delete requires a key");
    hcm_archive_delete(&g_rs, key);
    mem_entry_t *ent = NULL; int n = mem_load(&ent);
    int i = mem_find(ent, n, key);
    if (i < 0) { mem_free(ent, n); return fmt("ERROR: no memory entry named '%s'", key); }
    free(ent[i].content);
    for (int j = i; j < n - 1; j++) ent[j] = ent[j + 1];
    n--;
    if (mem_save(ent, n) != 0) { mem_free(ent, n); return xstrdup("ERROR: cannot write memory.json"); }
    mem_free(ent, n);
    return fmt("✓ memory '%s' deleted (KV slots freed)", key);
}

char *agent_memory_clear(void) {
    hcm_archive_clear(&g_rs);
    if (mem_save(NULL, 0) != 0) return xstrdup("ERROR: cannot write memory.json");
    return xstrdup("✓ memory archive cleared");
}

/* boot: re-embed every memory.json entry into the archive arena (in file
   order) so pinned memories survive a restart. Called after the system
   prompt is ingested and the runstate is ready. */
void agent_memory_restore(void) {
    if (!g_rs_ready) return;
    mem_entry_t *ent = NULL; int n = mem_load(&ent);
    for (int i = 0; i < n; i++) {
        int rc = hcm_archive_write(&g_model, &g_rs, ent[i].key, ent[i].content, g_threads);
        if (rc < 0)
            fprintf(stderr, "[memory] failed to restore '%s' (rc=%d)\n", ent[i].key, rc);
        else
            fprintf(stderr, "[memory] restored '%s' (%d tokens)\n", ent[i].key, rc);
    }
    mem_free(ent, n);
}

/* ---- main ---- */
static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s [options]\n"
        "  -m <path>     model file (default: %s, else ~/.qma/config)\n"
        "  -p <prompt>   one-shot prompt (default: interactive)\n"
        "  -t <n>        threads (default 8)\n"
        "  -c <n>        ring context size (default 65536)\n"
        "  --temp <f>    temperature (default 0.8)\n"
        "  --repeat <f>  repeat penalty (default 1.1)\n"
        "  --eos-penalty <f>  control-token logit penalty (default 1.5)\n"
        "  --no-think    disable the <think> block\n"
        "  --check-align print whether the model is 4K-aligned (no repack, no load)\n"
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
    const char *embed_tree = NULL, *embed_out = NULL;
    const char *export_dir = NULL;
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
        else if (strcmp(argv[i], "--check-align") == 0) g_check_align = 1;
        else if (strcmp(argv[i], "--embed-internal") == 0 && i + 2 < argc) {
            embed_tree = argv[++i];
            embed_out = argv[++i];
        }
        else if (strcmp(argv[i], "--export-internal") == 0 && i + 1 < argc) {
            export_dir = argv[++i];
        }
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

    /* self-hosting tool modes (no model needed) */
    if (embed_tree && embed_out) {
        int rc = intern_embed_binary(embed_tree, embed_out, 0,
                                     (uint64_t)time(NULL), 1);
        fprintf(stderr, "qma: internal tree embedded into %s (%s)\n",
                embed_out, rc == 0 ? "ok" : "FAILED");
        return rc == 0 ? 0 : 1;
    }
    if (export_dir) {
        selfctx_hdr_t h;
        if (!selfctx_detect(g_self_exe, &h) || h.int_size == 0) {
            fprintf(stderr, "qma: no embedded internal tree in %s\n", g_self_exe);
            return 1;
        }
        mkdirs(export_dir);
        int fd = open(g_self_exe, O_RDONLY);
        int rc = -1;
        if (fd >= 0) {
            rc = intern_extract_region(fd, (off_t)h.int_off, h.int_size, export_dir);
            close(fd);
        }
        fprintf(stderr, "qma: internal tree exported to %s (%s)\n",
                export_dir, rc == 0 ? "ok" : "FAILED");
        return rc == 0 ? 0 : 1;
    }

    resolve_model_path(model_from_cli);

    /* silent thermal governor (off for debugging via QMA_NOTHERMAL=1) */
    if (getenv("QMA_NOTHERMAL") == NULL)
        thermal_start();

    if (g_check_align) {
        char err[512] = "";
        int need = qma_model_needs_align(g_model_path, err, sizeof(err));
        if (need < 0) {
            fprintf(stderr, "qma: check-align: %s\n", err);
            return 2;
        }
        fprintf(stderr, "qma: model %s is %s\n", g_model_path,
                need ? "NOT 4K-aligned (repack on next normal launch)"
                     : "4K-aligned (O_DIRECT ready)");
        return 0;
    }

    fprintf(stderr, "qma: loading %s ...\n", g_model_path);
    char err[512];
    if (qma_load(&g_model, g_model_path, err, sizeof(err)) != 0) {
        fprintf(stderr, "qma: load failed: %s\n", err);
        return 1;
    }
    fprintf(stderr, "qma: expert reads: %s\n",
            g_model.dio_fd >= 0 ? "O_DIRECT (4K-aligned, no page cache)" : "buffered (page cache)");
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

    /* session mode: --session <dir> = classic persistent dir (no
       snapshots); default = a hidden session dir in the LOCAL folder (the
       cwd qma runs from — never a system tmp: Termux's usr/tmp is a plain
       directory, not a real tmpfs, and can be cleaned or removed out from
       under us) + dated context snapshot on exit. The session dir is
       transient: wiped on a clean exit after the context is embedded. */
    if (!g_session_dir[0]) {
        char tmpl[1200];
        snprintf(tmpl, sizeof(tmpl), ".qma-XXXXXX");
        char *d = mkdtemp(tmpl);
        if (d) {
            snprintf(g_session_dir, sizeof(g_session_dir), "%s", d);
        } else {
            mkdirs(".qma_tmp");
            snprintf(g_session_dir, sizeof(g_session_dir), ".qma_tmp");
        }
        g_selfctx = 1;
        fprintf(stderr, "qma: snapshot mode — session dir in the local folder, context embedded in a dated copy of the binary on exit\n");
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
    /* mount the embedded internal tree (the /internal/ filesystem) */
    intern_init();

    /* the system prompt depends on the internal path, so it is built after
       the session dir + internal tree are in place */
    build_system_prompt(g_system_prompt, sizeof(g_system_prompt));
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

    /* re-embed the agent's archive (memory.json) into the KV arena so
       pinned memories survive restarts */
    if (g_rs_ready) agent_memory_restore();

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    char cwd[4096];
    getcwd(cwd, sizeof(cwd));
    fprintf(stderr, "qma: unified agent | model %s | workdir %s | session %s\n",
            g_model_path, cwd, g_session_dir[0] ? g_session_dir : "(memory)");
    fprintf(stderr, "> type /exit to quit, /reset to clear the session\n\n");

    if (one_shot) {
        agent_turn(one_shot);
        thermal_drain();   /* flush any queued heat-status line */
        thermal_stop();
        finish_session();
        if (g_model.ecache_on) qma_ecache_teardown(&g_model);
        runstate_free(&g_rs);
        qma_free(&g_model);
        return 0;
    }

    static char line[65536];   /* REPL line buffer — BSS, not stack */
    for (;;) {
        if (g_shutdown) break;
        thermal_drain();   /* queued heat updates land here, not mid-chat */
        if (!g_stdin_tty) {
            c_cyan();
            printf("you> ");
            c_reset();
            fflush(stdout);
        }
        int rl = read_input_line(line, sizeof(line));
        if (rl < 0) break;
        size_t ll = (size_t)rl;
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


