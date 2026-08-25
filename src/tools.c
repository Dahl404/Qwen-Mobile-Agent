/* tools.c — tool schemas + implementations for the unified agent.
 * Ported from jerry2-py (executor.py / tools_minimal.py) minus the
 * experimental layers (coins, roles, self-modifying tools, screen control).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/poll.h>
#include <errno.h>
#include <time.h>
#include <pwd.h>
#include <fnmatch.h>
#include <fcntl.h>
#include "tools.h"
#include "json.h"
#include "intern.h"
#include "worker.h"

/* ---------------- schemas ---------------- */

const tool_schema_t g_tools[] = {
    {"pwd",
     "Show current working directory",
     "{\"type\":\"object\",\"properties\":{}}"},
    {"ls",
     "List files in a directory",
     "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\",\"default\":\".\"},\"show_hidden\":{\"type\":\"boolean\"},\"long_format\":{\"type\":\"boolean\"}}}"},
    {"find",
     "Find files by glob pattern under a directory (e.g. *.c)",
     "{\"type\":\"object\",\"properties\":{\"pattern\":{\"type\":\"string\"},\"path\":{\"type\":\"string\",\"default\":\".\"}},\"required\":[\"pattern\"]}"},
    {"grep",
     "Search files with grep",
     "{\"type\":\"object\",\"properties\":{\"pattern\":{\"type\":\"string\"},\"path\":{\"type\":\"string\",\"default\":\".\"},\"recursive\":{\"type\":\"boolean\"}},\"required\":[\"pattern\"]}"},
    {"read",
     "Read the contents of a file. path is REQUIRED and must be a file path, never empty.",
     "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"start_line\":{\"type\":\"integer\",\"default\":1},\"max_lines\":{\"type\":\"integer\",\"default\":500}},\"required\":[\"path\"]}"},
    {"write",
     "Write content to a file. path is REQUIRED and must be a file path (e.g. 'main.c' or 'src/main.c'), NEVER empty and NEVER a directory.",
     "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}},\"required\":[\"path\",\"content\"]}"},
    {"edit",
     "Edit file by exact text replacement (pi-style). Multiple disjoint {oldText, newText} pairs; each oldText must match exactly once. path is REQUIRED.",
     "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"edits\":{\"type\":\"array\",\"items\":{\"type\":\"object\",\"properties\":{\"oldText\":{\"type\":\"string\"},\"newText\":{\"type\":\"string\"}},\"required\":[\"oldText\",\"newText\"]}}},\"required\":[\"path\",\"edits\"]}"},
    {"replace_lines",
     "Replace line range in file (use after read_file). path is REQUIRED.",
     "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"start_line\":{\"type\":\"integer\"},\"end_line\":{\"type\":\"integer\"},\"new_content\":{\"type\":\"string\"}},\"required\":[\"path\",\"start_line\",\"end_line\",\"new_content\"]}"},
    {"insert_lines",
     "Insert lines after given line number. path is REQUIRED.",
     "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"after_line\":{\"type\":\"integer\"},\"content\":{\"type\":\"string\"}},\"required\":[\"path\",\"after_line\",\"content\"]}"},
    {"delete_lines",
     "Delete line range from file. path is REQUIRED.",
     "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"start_line\":{\"type\":\"integer\"},\"end_line\":{\"type\":\"integer\"}},\"required\":[\"path\",\"start_line\",\"end_line\"]}"},
    {"bash",
     "Execute bash commands in the current working directory",
     "{\"type\":\"object\",\"properties\":{\"command\":{\"type\":\"string\"},\"timeout\":{\"type\":\"integer\",\"default\":60}},\"required\":[\"command\"]}"},
    {"enter",
     "Change current working directory; the qma binary physically moves to the new directory (so the next self-hosted generation is written there)",
     "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}},\"required\":[\"path\"]}"},
    {"todo_write",
     "Replace entire todo list. Pass array of {content, priority, completed}.",
     "{\"type\":\"object\",\"properties\":{\"todos\":{\"type\":\"array\",\"items\":{\"type\":\"object\",\"properties\":{\"content\":{\"type\":\"string\"},\"priority\":{\"type\":\"string\"},\"completed\":{\"type\":\"boolean\"}}}}},\"required\":[\"todos\"]}"},
    {"todo_complete",
     "Mark todo as done by index (0-based)",
     "{\"type\":\"object\",\"properties\":{\"index\":{\"type\":\"integer\"}},\"required\":[\"index\"]}"},
    {"todo_remove",
     "Remove todo by index (0-based)",
     "{\"type\":\"object\",\"properties\":{\"index\":{\"type\":\"integer\"}},\"required\":[\"index\"]}"},
    {"write_diary",
     "Write reflection to diary",
     "{\"type\":\"object\",\"properties\":{\"entry\":{\"type\":\"string\"},\"mood\":{\"type\":\"string\",\"default\":\"neutral\"}},\"required\":[\"entry\"]}"},
    {"read_diary",
     "Read past diary entries",
     "{\"type\":\"object\",\"properties\":{\"days_back\":{\"type\":\"integer\",\"default\":7}}}"},
    {"ask_user",
     "Ask the user a question when you need clarification or decisions",
     "{\"type\":\"object\",\"properties\":{\"question\":{\"type\":\"string\"}},\"required\":[\"question\"]}"},
    {"speak",
     "Speak text aloud through the phone speaker (Termux TTS). Use when you want to actually talk to the user instead of only writing text.",
     "{\"type\":\"object\",\"properties\":{\"text\":{\"type\":\"string\"},\"rate\":{\"type\":\"integer\",\"default\":150}},\"required\":[\"text\"]}"},
    {"notify",
     "Send a notification to the phone status bar (Termux:API). Use to alert the user when a long task finishes or something needs their attention.",
     "{\"type\":\"object\",\"properties\":{\"title\":{\"type\":\"string\"},\"content\":{\"type\":\"string\",\"default\":\"\"}},\"required\":[\"title\"]}"},
    {"vibrate",
     "Vibrate the phone for a duration in milliseconds.",
     "{\"type\":\"object\",\"properties\":{\"duration_ms\":{\"type\":\"integer\",\"default\":500}}}"},
    {"battery",
     "Report phone battery status (level, charging state, health, temperature) as JSON.",
     "{\"type\":\"object\",\"properties\":{}}"},
    {"self_build",
     "Compile the internal source tree (/internal/src) with the same flags as the Makefile and smoke-test the result. Call after editing your own code to verify it before qma commits it at exit.",
     "{\"type\":\"object\",\"properties\":{}}"},
    {"memory_write",
     "Write (create or replace) a long-term memory entry. Memory is pinned into your KV cache and attended on EVERY token forever — it survives the rolling window and restarts. Use it for consolidation: save the user's goal, your plan, decisions, and facts you must not forget. Returns the token count pinned.",
     "{\"type\":\"object\",\"properties\":{\"key\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}},\"required\":[\"key\",\"content\"]}"},
    {"memory_append",
     "Append text to an existing memory entry (or create it). Use after each step of a multi-step task to record progress: what you did and what's next.",
     "{\"type\":\"object\",\"properties\":{\"key\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}},\"required\":[\"key\",\"content\"]}"},
    {"memory_list",
     "List all memory entry keys with their token counts.",
     "{\"type\":\"object\",\"properties\":{}}"},
    {"memory_delete",
     "Delete a memory entry and free its pinned KV slots.",
     "{\"type\":\"object\",\"properties\":{\"key\":{\"type\":\"string\"}},\"required\":[\"key\"]}"},
    {"memory_clear",
     "Delete ALL memory entries (frees the whole pinned archive).",
     "{\"type\":\"object\",\"properties\":{}}"},
    {"worker_spawn",
     "Spawn an LFM document-analyst worker. Give it a document (path preferred — the worker reads the file itself; or doc for inline text). The worker holds the document in its KV cache and answers questions about it. Returns a worker id.",
     "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},\"path\":{\"type\":\"string\"},\"doc\":{\"type\":\"string\"},\"role\":{\"type\":\"string\"}}}"},
    {"worker_ask",
     "Ask a spawned worker a question about its document. ephemeral=true (default) gives a clean KV per query — the worker forgets the question after answering, so every question sees the same fresh document. ephemeral=false keeps the conversation (follow-up questions work). Returns the worker's answer.",
     "{\"type\":\"object\",\"properties\":{\"id\":{\"type\":\"integer\"},\"question\":{\"type\":\"string\"},\"ephemeral\":{\"type\":\"boolean\",\"default\":true},\"max_tokens\":{\"type\":\"integer\",\"default\":0}},\"required\":[\"id\",\"question\"]}"},
    {"worker_close",
     "Close a worker and free its KV cache.",
     "{\"type\":\"object\",\"properties\":{\"id\":{\"type\":\"integer\"}},\"required\":[\"id\"]}"},
    {"worker_list",
     "List active workers with their ids.",
     "{\"type\":\"object\",\"properties\":{}}"},
};

const int g_n_tools = (int)(sizeof(g_tools) / sizeof(g_tools[0]));

/* ---------------- safe helpers ---------------- */

char *xstrdup(const char *s) {
    return strdup(s ? s : "");
}

/* Safe dynamic formatting (uses vasprintf, falls back to snprintf) */
static char *fmt_v(const char *fmt, va_list ap) {
    char *buf = NULL;
    int len = vasprintf(&buf, fmt, ap);
    if (len < 0) return xstrdup("(formatting error)");
    return buf;
}

char *fmt(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char *r = fmt_v(fmt, ap);
    va_end(ap);
    return r;
}

/* Expand leading ~/ and ~user/ to home directory using getpwnam if needed.
   Returns a newly allocated string (caller must free). */
static char *path_expand_home_alloc(const char *p) {
    if (!p || !p[0]) return xstrdup(p);
    if (p[0] != '~') return xstrdup(p);
    const char *slash = strchr(p, '/');
    const char *user = NULL;
    size_t user_len = 0;
    if (slash && slash > p + 1) {
        user = p + 1;
        user_len = (size_t)(slash - user);
    } else if (!slash && p[1] != '\0') {
        user = p + 1;
        user_len = strlen(user);
    }
    const char *home = NULL;
    if (user_len > 0) {
        char *uname = malloc(user_len + 1);
        if (!uname) return xstrdup(p);
        memcpy(uname, user, user_len);
        uname[user_len] = 0;
        struct passwd *pw = getpwnam(uname);
        free(uname);
        if (pw) home = pw->pw_dir;
    } else {
        home = getenv("HOME");
    }
    if (!home) home = "/";
    if (!slash) return fmt("%s", home);
    return fmt("%s%s", home, slash);
}

/* Path guard: returns NULL if ok, else heap error string */
static char *path_guard(const char *path) {
    char *ep = path_expand_home_alloc(path);
    if (!ep || !ep[0]) {
        free(ep);
        return fmt("ERROR: path is REQUIRED and must be a file path, never empty.");
    }
    if (ep[strlen(ep)-1] == '/') {
        free(ep);
        return fmt("ERROR: '%s' is a directory. path must be a file path.", ep);
    }
    struct stat st;
    if (stat(ep, &st) == 0 && S_ISDIR(st.st_mode)) {
        free(ep);
        return fmt("ERROR: '%s' is a directory. path must be a file path.", ep);
    }
    free(ep);
    return NULL;
}

char *read_whole_file(const char *path) {
    fprintf(stderr, "[TRACE] read_whole_file: enter, path=%p\n", (void*)path); fflush(stderr);
    char *ep = path_expand_home_alloc(path);
    fprintf(stderr, "[TRACE] read_whole_file: expanded ep=%s\n", ep ? ep : "(null)"); fflush(stderr);
    if (!ep) return fmt("ERROR: invalid path");
    FILE *f = fopen(ep, "rb");
    fprintf(stderr, "[TRACE] read_whole_file: fopen -> %p\n", (void*)f); fflush(stderr);
    if (!f) {
        char *err = fmt("ERROR: cannot open '%s': %s", ep, strerror(errno));
        free(ep);
        return err;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    fprintf(stderr, "[TRACE] read_whole_file: sz=%ld\n", sz); fflush(stderr);
    if (sz < 0) {
        fclose(f);
        free(ep);
        return fmt("ERROR: cannot read '%s'", ep);
    }
    char *buf = malloc((size_t)sz + 1);
    fprintf(stderr, "[TRACE] read_whole_file: malloc(%ld+1) -> %p\n", sz, (void*)buf); fflush(stderr);
    if (!buf) {
        fclose(f);
        free(ep);
        return fmt("ERROR: OOM reading '%s'", ep);
    }
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fprintf(stderr, "[TRACE] read_whole_file: fread -> rd=%zu\n", rd); fflush(stderr);
    fclose(f);
    free(ep);
    buf[rd] = 0;
    fprintf(stderr, "[TRACE] read_whole_file: returning buf=%p, len=%zu\n", (void*)buf, strlen(buf)); fflush(stderr);
    return buf;
}

/* ---------------- internal filesystem mapping ----
   The agent's internal tree (its own source + collected tools/data) lives
   at $QMA_INTERNAL, extracted from the binary at boot. Structured tools
   accept the logical path /internal/... and we rewrite it to the real
   location; bash uses the $QMA_INTERNAL env var instead. */
static char *g_intern_root = NULL;
void intern_set_root(const char *root) {
    free(g_intern_root);
    g_intern_root = root ? strdup(root) : NULL;
}

/* running binary path — shared with agent.c */
extern char g_self_exe[1024];

/* map_path returns a newly allocated string (caller must free) */
static char *map_path_alloc(const char *p) {
    if (!p || !g_intern_root) return xstrdup(p);
    const char *rest = NULL;
    if (strncmp(p, "/internal/", 10) == 0)    rest = p + 10;
    else if (strcmp(p, "/internal") == 0)     rest = "";
    else if (strncmp(p, "internal/", 9) == 0) rest = p + 9;
    else if (strcmp(p, "internal") == 0)      rest = "";
    if (rest) return fmt("%s/%s", g_intern_root, rest);
    return xstrdup(p);
}

/* ---- tool implementations ---- */

static char *t_pwd(void) {
    char buf[4096];
    if (!getcwd(buf, sizeof(buf))) return fmt("ERROR: getcwd: %s", strerror(errno));
    return xstrdup(buf);
}

static char *t_list_dir(const jval_t *args) {
    const char *path_raw = json_str(json_obj_get(args, "path"));
    char *path = map_path_alloc(path_expand_home_alloc(path_raw));
    if (!path[0]) { free(path); path = xstrdup("."); }
    int hidden = json_bool(json_obj_get(args, "show_hidden"));
    int lng = json_bool(json_obj_get(args, "long_format"));
    DIR *d = opendir(path);
    if (!d) {
        char *err = fmt("ERROR: cannot open '%s': %s", path, strerror(errno));
        free(path);
        return err;
    }
    char *out = malloc(65536);
    size_t o = 0;
    out[0] = 0;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (!hidden && e->d_name[0] == '.') continue;
        char line[1024];
        if (lng) {
            struct stat st;
            char full[4096];
            snprintf(full, sizeof(full), "%s/%s", path, e->d_name);
            if (stat(full, &st) == 0)
                snprintf(line, sizeof(line), "%s%s  %8lld  %s\n",
                         S_ISDIR(st.st_mode) ? "d" : "-",
                         S_ISDIR(st.st_mode) ? "" : "r",
                         (long long)st.st_size, e->d_name);
            else
                snprintf(line, sizeof(line), "%s\n", e->d_name);
        } else {
            snprintf(line, sizeof(line), "%s%s\n", e->d_name, e->d_type == DT_DIR ? "/" : "");
        }
        if (o + strlen(line) + 1 < 65536) {
            strcpy(out + o, line);
            o += strlen(line);
        }
    }
    closedir(d);
    free(path);
    return out;
}

static char *t_search(const jval_t *args) {
    const char *pattern = json_str(json_obj_get(args, "pattern"));
    const char *path_raw = json_str(json_obj_get(args, "path"));
    char *path = map_path_alloc(path_expand_home_alloc(path_raw));
    int recursive = json_bool(json_obj_get(args, "recursive"));
    if (!pattern[0]) { free(path); return fmt("ERROR: pattern is required"); }
    if (!path[0]) { free(path); path = xstrdup("."); }
    char cmd[8192];
    char qpat[2048], qpath[2048];
    snprintf(qpat, sizeof(qpat), "'%s'", pattern);
    snprintf(qpath, sizeof(qpath), "'%s'", path);
    snprintf(cmd, sizeof(cmd), "grep -n %s %s %s 2>&1 | head -200",
             recursive ? "-r" : "", qpat, qpath);
    FILE *f = popen(cmd, "r");
    if (!f) { free(path); return fmt("ERROR: grep failed"); }
    char *out = malloc(65536);
    size_t o = 0;
    out[0] = 0;
    char line[2048];
    while (fgets(line, sizeof(line), f)) {
        if (o + strlen(line) + 1 < 65536) {
            strcpy(out + o, line);
            o += strlen(line);
        }
    }
    pclose(f);
    free(path);
    if (o == 0) { free(out); return xstrdup("no matches"); }
    return out;
}

#define LARGE_FILE_THRESHOLD 2000   // estimated tokens

static char *t_read_file(const jval_t *args) {
    fprintf(stderr, "[TRACE] t_read_file: enter\n"); fflush(stderr);
    const char *path = json_str(json_obj_get(args, "path"));
    fprintf(stderr, "[TRACE] t_read_file: path='%s'\n", path ? path : "(null)"); fflush(stderr);
    int start = (int)json_num(json_obj_get(args, "start_line"));
    int maxl = (int)json_num(json_obj_get(args, "max_lines"));
    char *err = path_guard(path);
    fprintf(stderr, "[TRACE] t_read_file: path_guard -> %s\n", err ? "ERROR" : "ok"); fflush(stderr);
    if (err) return err;
    char *ep = path_expand_home_alloc(path);
    fprintf(stderr, "[TRACE] t_read_file: ep='%s'\n", ep ? ep : "(null)"); fflush(stderr);
    if (start < 1) start = 1;
    if (maxl <= 0) maxl = 500;

    char *content = read_whole_file(ep);
    fprintf(stderr, "[TRACE] t_read_file: read_whole_file returned %p\n", (void*)content); fflush(stderr);
    free(ep);
    if (!content || strncmp(content, "ERROR:", 6) == 0) {
        fprintf(stderr, "[TRACE] t_read_file: content error path\n"); fflush(stderr);
        return content ? content : xstrdup("ERROR: read failed");
    }

    size_t len = strlen(content);
    int est_tokens = (int)(len / 4);
    fprintf(stderr, "[TRACE] t_read_file: len=%zu est_tokens=%d\n", len, est_tokens); fflush(stderr);

    // If file is large AND not a specific range, show preview + worker suggestion
    if (est_tokens > LARGE_FILE_THRESHOLD && start == 1 && maxl >= 500) {
        fprintf(stderr, "[TRACE] t_read_file: entering PREVIEW branch\n"); fflush(stderr);
        char *out = malloc(16384);
        fprintf(stderr, "[TRACE] t_read_file: preview malloc -> %p\n", (void*)out); fflush(stderr);
        size_t o = 0;
        out[0] = 0;
        int line = 1;
        char *save = NULL;
        char *tok = strtok_r(content, "\n", &save);
        while (tok && line <= 50) {
            int w = snprintf(out + o, 16384 - o, "%6d\t%s\n", line, tok);
            if (w < 0 || (size_t)w >= 16384 - o) break;   // would overflow out[]; stop the preview here
            o += (size_t)w;
            line++;
            tok = strtok_r(NULL, "\n", &save);
        }
        fprintf(stderr, "[TRACE] t_read_file: preview loop done, o=%zu line=%d\n", o, line); fflush(stderr);
        free(content);
        snprintf(out + o, 16384 - o,
                 "\n... file has ~%d tokens (full content not shown). "
                 "Use worker_spawn(path=\"%s\") then worker_ask(id, \"...\") to "
                 "analyse it without loading it into your own context.\n",
                 est_tokens, path);
        fprintf(stderr, "[TRACE] t_read_file: preview branch returning\n"); fflush(stderr);
        return out;
    }
    fprintf(stderr, "[TRACE] t_read_file: entering FULL branch\n"); fflush(stderr);

    // Full content with line numbers using dynamic buffer
    char *out = NULL;
    size_t out_cap = 0, out_len = 0;
    int line = 1;
    char *save = NULL;
    char *tok = strtok_r(content, "\n", &save);
    while (tok) {
        if (line > start - 1 + maxl) break;
        if (line >= start) {
            int need = snprintf(NULL, 0, "%6d\t%s\n", line, tok);
            if (need < 0) { free(content); free(out); return xstrdup("ERROR: formatting failed"); }
            if (out_len + (size_t)need + 1 > out_cap) {
                size_t new_cap = out_cap ? out_cap * 2 : 16384;
                while (new_cap < out_len + (size_t)need + 1) new_cap *= 2;
                char *new_out = realloc(out, new_cap);
                if (!new_out) { free(content); free(out); return xstrdup("ERROR: OOM"); }
                out = new_out;
                out_cap = new_cap;
            }
            int w = snprintf(out + out_len, out_cap - out_len, "%6d\t%s\n", line, tok);
            if (w > 0) out_len += (size_t)w;
        }
        line++;
        tok = strtok_r(NULL, "\n", &save);
    }
    fprintf(stderr, "[TRACE] t_read_file: full loop done, out_len=%zu\n", out_len); fflush(stderr);
    free(content);
    if (!out) { out = xstrdup(""); out_len = 0; } else out[out_len] = 0;
    fprintf(stderr, "[TRACE] t_read_file: full branch returning\n"); fflush(stderr);
    return out;
}

static char *t_write_file(const jval_t *args) {
    const char *path = json_str(json_obj_get(args, "path"));
    const char *content = json_str(json_obj_get(args, "content"));
    char *err = path_guard(path);
    if (err) return err;
    char *ep = path_expand_home_alloc(path);
    if (!content || !content[0]) {
        free(ep);
        return xstrdup("ERROR: content is empty — nothing was written. write() requires a non-empty content string.");
    }
    char copy[4096];
    snprintf(copy, sizeof(copy), "%s", ep);
    for (char *p = copy; (p = strchr(p, '/')) != NULL; p++) {
        *p = 0;
        if (copy[0]) mkdir(copy, 0755);
        *p = '/';
    }
    FILE *f = fopen(ep, "w");
    if (!f) {
        char *err2 = fmt("ERROR: cannot write '%s': %s", ep, strerror(errno));
        free(ep);
        return err2;
    }
    fwrite(content, 1, strlen(content), f);
    fclose(f);
    free(ep);
    return fmt("✓ wrote %zu bytes to %s", strlen(content), path);
}

static char *t_edit(const jval_t *args) {
    const char *path = json_str(json_obj_get(args, "path"));
    char *err = path_guard(path);
    if (err) return err;
    char *ep = path_expand_home_alloc(path);
    const jval_t *edits = json_obj_get(args, "edits");
    if (!edits || edits->type != J_ARR || edits->n == 0) {
        free(ep);
        return xstrdup("ERROR: edit requires an edits array of {oldText,newText} pairs");
    }
    char *content = read_whole_file(ep);
    if (!content || strncmp(content, "ERROR:", 6) == 0) {
        free(ep);
        return content ? content : xstrdup("ERROR: read failed");
    }
    char *cur = content;
    for (size_t i = 0; i < edits->n; i++) {
        const jval_t *pr = json_arr_get(edits, i);
        if (!pr || pr->type != J_OBJ) { free(content); free(ep); return fmt("ERROR: edits[%zu] is not an object", i); }
        const char *oldT = json_str(json_obj_get(pr, "oldText"));
        const char *newT = json_str(json_obj_get(pr, "newText"));
        if (!oldT[0]) { free(content); free(ep); return fmt("ERROR: edits[%zu] oldText is empty", i); }
        int count = 0;
        for (const char *p = cur; (p = strstr(p, oldT)) != NULL; p += strlen(oldT)) count++;
        if (count != 1) { free(content); free(ep); return fmt("ERROR: oldText must match exactly once (found %d): %.80s", count, oldT); }
        size_t cl = strlen(cur), ol = strlen(oldT), nl = strlen(newT);
        char *next = malloc(cl - ol + nl + 1);
        if (!next) { free(content); free(ep); return fmt("ERROR: OOM"); }
        char *hit = strstr(cur, oldT);
        size_t pre = (size_t)(hit - cur);
        memcpy(next, cur, pre);
        memcpy(next + pre, newT, nl);
        memcpy(next + pre + nl, hit + ol, cl - pre - ol + 1);
        free(cur);
        cur = next;
    }
    FILE *f = fopen(ep, "w");
    if (!f) {
        free(content); free(cur); free(ep);
        return fmt("ERROR: cannot write '%s': %s", ep, strerror(errno));
    }
    fwrite(cur, 1, strlen(cur), f);
    fclose(f);
    size_t n = strlen(cur);
    free(cur); free(content); free(ep);
    return fmt("✓ edited %s (%zu bytes)", path, n);
}

/* line-range helpers */
typedef struct { char **lines; int n; int cap; } lines_t;
static void lines_free(lines_t *ls) {
    for (int i = 0; i < ls->n; i++) free(ls->lines[i]);
    free(ls->lines);
}
static int lines_split(lines_t *ls, const char *content) {
    ls->n = 0; ls->cap = 0; ls->lines = NULL;
    const char *p = content;
    while (*p) {
        const char *nl = strchr(p, '\n');
        size_t len = nl ? (size_t)(nl - p) : strlen(p);
        if (ls->n >= ls->cap) {
            ls->cap = ls->cap ? ls->cap * 2 : 16;
            char **nl2 = realloc(ls->lines, (size_t)ls->cap * sizeof(char *));
            if (!nl2) return -1;
            ls->lines = nl2;
        }
        ls->lines[ls->n] = malloc(len + 1);
        if (!ls->lines[ls->n]) return -1;
        memcpy(ls->lines[ls->n], p, len);
        ls->lines[ls->n][len] = 0;
        ls->n++;
        if (!nl) break;
        p = nl + 1;
    }
    return 0;
}
static char *lines_join(lines_t *ls) {
    size_t total = 1;
    for (int i = 0; i < ls->n; i++) total += strlen(ls->lines[i]) + 1;
    char *out = malloc(total);
    size_t o = 0;
    for (int i = 0; i < ls->n; i++) {
        size_t l = strlen(ls->lines[i]);
        memcpy(out + o, ls->lines[i], l); o += l;
        out[o++] = '\n';
    }
    out[o] = 0;
    return out;
}

static char *t_replace_lines(const jval_t *args) {
    const char *path = json_str(json_obj_get(args, "path"));
    int s = (int)json_num(json_obj_get(args, "start_line"));
    int e = (int)json_num(json_obj_get(args, "end_line"));
    const char *nc = json_str(json_obj_get(args, "new_content"));
    char *err = path_guard(path);
    if (err) return err;
    char *ep = path_expand_home_alloc(path);
    char *content = read_whole_file(ep);
    if (!content || strncmp(content, "ERROR:", 6) == 0) {
        free(ep);
        return content ? content : xstrdup("ERROR: read failed");
    }
    lines_t ls;
    if (lines_split(&ls, content) != 0) { free(content); free(ep); return xstrdup("ERROR: OOM"); }
    if (s < 1 || e > ls.n || e < s) {
        lines_free(&ls); free(content); free(ep);
        return fmt("ERROR: bad range %d-%d (file has %d lines)", s, e, ls.n);
    }
    int nl = 1;
    const char *p = nc;
    while (*p) { if (*p == '\n') nl++; p++; }
    char **newlines = malloc((size_t)(ls.n - (e - s + 1) + nl) * sizeof(char *));
    int k = 0;
    for (int i = 0; i < s - 1; i++) newlines[k++] = ls.lines[i];
    {
        char tmp[65536];
        snprintf(tmp, sizeof(tmp), "%s", nc);
        char *save = NULL;
        for (char *t = strtok_r(tmp, "\n", &save); t; t = strtok_r(NULL, "\n", &save)) {
            newlines[k] = xstrdup(t);
            k++;
        }
    }
    for (int i = e; i < ls.n; i++) newlines[k++] = ls.lines[i];
    lines_t nl2 = { newlines, k, k };
    char *out = lines_join(&nl2);
    FILE *f = fopen(ep, "w");
    if (!f) {
        free(out); free(newlines); lines_free(&ls); free(content); free(ep);
        return fmt("ERROR: cannot write '%s': %s", ep, strerror(errno));
    }
    fwrite(out, 1, strlen(out), f);
    fclose(f);
    free(out); free(newlines); lines_free(&ls); free(content); free(ep);
    return fmt("✓ replaced lines %d-%d in %s", s, e, path);
}

static char *t_insert_lines(const jval_t *args) {
    const char *path = json_str(json_obj_get(args, "path"));
    int after = (int)json_num(json_obj_get(args, "after_line"));
    const char *content = json_str(json_obj_get(args, "content"));
    char *err = path_guard(path);
    if (err) return err;
    char *ep = path_expand_home_alloc(path);
    char *file = read_whole_file(ep);
    if (!file || strncmp(file, "ERROR:", 6) == 0) {
        free(ep);
        return file ? file : xstrdup("ERROR: read failed");
    }
    lines_t ls;
    if (lines_split(&ls, file) != 0) { free(file); free(ep); return xstrdup("ERROR: OOM"); }
    if (after < 0 || after > ls.n) {
        lines_free(&ls); free(file); free(ep);
        return fmt("ERROR: after_line %d out of range (file has %d lines)", after, ls.n);
    }
    int nl = 1;
    const char *p = content;
    while (*p) { if (*p == '\n') nl++; p++; }
    char **newlines = malloc((size_t)(ls.n + nl) * sizeof(char *));
    int k = 0;
    for (int i = 0; i < after; i++) newlines[k++] = ls.lines[i];
    {
        char tmp[65536];
        snprintf(tmp, sizeof(tmp), "%s", content);
        char *save = NULL;
        for (char *t = strtok_r(tmp, "\n", &save); t; t = strtok_r(NULL, "\n", &save)) {
            newlines[k] = xstrdup(t);
            k++;
        }
    }
    for (int i = after; i < ls.n; i++) newlines[k++] = ls.lines[i];
    lines_t nl2 = { newlines, k, k };
    char *out = lines_join(&nl2);
    FILE *f = fopen(ep, "w");
    if (!f) {
        free(out); free(newlines); lines_free(&ls); free(file); free(ep);
        return fmt("ERROR: cannot write '%s': %s", ep, strerror(errno));
    }
    fwrite(out, 1, strlen(out), f);
    fclose(f);
    free(out); free(newlines); lines_free(&ls); free(file); free(ep);
    return fmt("✓ inserted %d lines after line %d in %s", nl, after, path);
}

static char *t_delete_lines(const jval_t *args) {
    const char *path = json_str(json_obj_get(args, "path"));
    int s = (int)json_num(json_obj_get(args, "start_line"));
    int e = (int)json_num(json_obj_get(args, "end_line"));
    char *err = path_guard(path);
    if (err) return err;
    char *ep = path_expand_home_alloc(path);
    char *file = read_whole_file(ep);
    if (!file || strncmp(file, "ERROR:", 6) == 0) {
        free(ep);
        return file ? file : xstrdup("ERROR: read failed");
    }
    lines_t ls;
    if (lines_split(&ls, file) != 0) { free(file); free(ep); return xstrdup("ERROR: OOM"); }
    if (s < 1 || e > ls.n || e < s) {
        lines_free(&ls); free(file); free(ep);
        return fmt("ERROR: bad range %d-%d (file has %d lines)", s, e, ls.n);
    }
    int removed = e - s + 1;
    char **newlines = malloc((size_t)(ls.n - removed) * sizeof(char *));
    int k = 0;
    for (int i = 0; i < s - 1; i++) newlines[k++] = ls.lines[i];
    for (int i = e; i < ls.n; i++) newlines[k++] = ls.lines[i];
    lines_t nl2 = { newlines, k, k };
    char *out = lines_join(&nl2);
    FILE *f = fopen(ep, "w");
    if (!f) {
        free(out); free(newlines); lines_free(&ls); free(file); free(ep);
        return fmt("ERROR: cannot write '%s': %s", ep, strerror(errno));
    }
    fwrite(out, 1, strlen(out), f);
    fclose(f);
    free(out); free(newlines); lines_free(&ls); free(file); free(ep);
    return fmt("✓ deleted lines %d-%d from %s", s, e, path);
}

static char *t_execute_command(const jval_t *args) {
    const char *cmd = json_str(json_obj_get(args, "command"));
    int timeout = (int)json_num(json_obj_get(args, "timeout"));
    if (!cmd[0]) return xstrdup("ERROR: command is required");
    if (timeout <= 0) timeout = 60;
    int pipefd[2];
    if (pipe(pipefd) != 0) return fmt("ERROR: pipe: %s", strerror(errno));
    pid_t pid = fork();
    if (pid < 0) return fmt("ERROR: fork: %s", strerror(errno));
    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], 1);
        dup2(pipefd[1], 2);
        close(pipefd[1]);
        execl("/data/data/com.termux/files/usr/bin/bash", "bash", "-c", cmd, (char *)NULL);
        _exit(127);
    }
    close(pipefd[1]);
    char *cap = malloc(262144);
    size_t ccap = 0;
    char buf[4096];
    struct pollfd pfd = { pipefd[0], POLLIN, 0 };
    double deadline = (double)time(NULL) + timeout;
    int timed_out = 0;
    for (;;) {
        double now = (double)time(NULL);
        int remaining = now >= deadline ? 0 : (int)(deadline - now);
        int pr = poll(&pfd, 1, remaining * 1000);
        if (pr < 0) { if (errno == EINTR) continue; break; }
        if (pr == 0) { timed_out = 1; break; }
        ssize_t n = read(pipefd[0], buf, sizeof(buf));
        if (n <= 0) break;
        fwrite(buf, 1, (size_t)n, stdout);
        fflush(stdout);
        if (ccap + (size_t)n < 262144) {
            memcpy(cap + ccap, buf, (size_t)n);
            ccap += (size_t)n;
        }
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
    cap[ccap] = 0;
    int nlines = 0;
    for (size_t i = 0; i < ccap; i++) if (cap[i] == '\n') nlines++;
    char *tail = cap;
    if (nlines > 200) {
        int skip = nlines - 200;
        const char *p = cap;
        while (skip-- > 0 && (p = strchr(p, '\n')) != NULL) p++;
        if (p && *p) tail = (char *)p + 1;
    }
    int rc = WIFEXITED(status) ? WEXITSTATUS(status) : 127;
    char *out;
    if (timed_out) {
        out = fmt("ERROR: timed out after %ds\n%s", timeout, tail);
    } else if (WIFEXITED(status) && rc == 0) {
        out = (tail[0]) ? xstrdup(tail) : fmt("✓ command succeeded");
    } else {
        out = fmt("ERROR: command exited %d\n%s", rc, tail);
    }
    free(cap);
    return out;
}

static char *t_enter(const jval_t *args) {
    const char *path = json_str(json_obj_get(args, "path"));
    char *expanded = path_expand_home_alloc(path);
    if (!expanded || !expanded[0]) {
        free(expanded);
        return xstrdup("ERROR: path is required");
    }
    if (chdir(expanded) != 0) {
        char *err = fmt("ERROR: cannot enter '%s': %s (try using ~/ or the full path from pwd)", expanded, strerror(errno));
        free(expanded);
        return err;
    }
    free(expanded);
    char buf[4096];
    getcwd(buf, sizeof(buf));
    char note[1400] = "";
    if (g_self_exe[0]) {
        const char *slash = strrchr(g_self_exe, '/');
        const char *name = slash ? slash + 1 : g_self_exe;
        char bindir[1024];
        if (slash && slash != g_self_exe) {
            size_t n = (size_t)(slash - g_self_exe);
            if (n >= sizeof(bindir)) n = sizeof(bindir) - 1;
            memcpy(bindir, g_self_exe, n); bindir[n] = 0;
        } else snprintf(bindir, sizeof(bindir), ".");
        int into_internal = g_intern_root && strncmp(buf, g_intern_root, strlen(g_intern_root)) == 0;
        if (strcmp(bindir, buf) != 0 && !into_internal) {
            char dest[1400];
            snprintf(dest, sizeof(dest), "%s/%s", buf, name);
            if (rename(g_self_exe, dest) == 0) {
                snprintf(g_self_exe, sizeof(g_self_exe), "%s", dest);
                snprintf(note, sizeof(note), " — binary moved to %s", dest);
            } else {
                snprintf(note, sizeof(note), " — binary NOT moved (%s)", strerror(errno));
            }
        }
    }
    return fmt("✓ now in %s%s", buf, note);
}

/* ---- todos (in-memory) ---- */
typedef struct { char content[512]; char priority[16]; int done; } todo_t;
static todo_t g_todos[64];
static int g_n_todos = 0;

static char *t_todo_write(const jval_t *args) {
    const jval_t *todos = json_obj_get(args, "todos");
    if (!todos || todos->type != J_ARR) return xstrdup("ERROR: todo_write requires a list of todos");
    g_n_todos = 0;
    for (size_t i = 0; i < todos->n && g_n_todos < 64; i++) {
        const jval_t *t = json_arr_get(todos, i);
        if (!t || t->type != J_OBJ) continue;
        snprintf(g_todos[g_n_todos].content, sizeof(g_todos[g_n_todos].content), "%s", json_str(json_obj_get(t, "content")));
        snprintf(g_todos[g_n_todos].priority, sizeof(g_todos[g_n_todos].priority), "%s", json_str(json_obj_get(t, "priority")));
        g_todos[g_n_todos].done = json_bool(json_obj_get(t, "completed"));
        g_n_todos++;
    }
    char *out = malloc(16384);
    size_t o = 0;
    for (int i = 0; i < g_n_todos; i++) {
        int w = snprintf(out + o, 16384 - o, "%s #%d: %s [%s]\n",
                         g_todos[i].done ? "✓" : "○", i, g_todos[i].content,
                         g_todos[i].priority[0] ? g_todos[i].priority : "medium");
        if (w < 0 || (size_t)w >= 16384 - o) break;
        o += (size_t)w;
    }
    if (o == 0) snprintf(out, 16384, "(todo list cleared)\n");
    return out;
}

static char *t_todo_complete(const jval_t *args) {
    int idx = (int)json_num(json_obj_get(args, "index"));
    if (idx < 0 || idx >= g_n_todos) return fmt("ERROR: index %d out of range (0-%d)", idx, g_n_todos - 1);
    g_todos[idx].done = 1;
    return fmt("✓ marked done: %s", g_todos[idx].content);
}

static char *t_todo_remove(const jval_t *args) {
    int idx = (int)json_num(json_obj_get(args, "index"));
    if (idx < 0 || idx >= g_n_todos) return fmt("ERROR: index %d out of range (0-%d)", idx, g_n_todos - 1);
    char c[512];
    snprintf(c, sizeof(c), "%s", g_todos[idx].content);
    for (int i = idx; i < g_n_todos - 1; i++) g_todos[i] = g_todos[i + 1];
    g_n_todos--;
    return fmt("✓ removed: %s", c);
}

/* ---- diary ---- */
static char *diary_path(char *buf, size_t buflen, int days_back) {
    time_t t = time(NULL) - (time_t)days_back * 86400;
    struct tm tm;
    localtime_r(&t, &tm);
    mkdir("diary", 0755);
    snprintf(buf, buflen, "diary/%04d-%02d-%02d.md",
             tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday);
    return buf;
}

static char *t_write_diary(const jval_t *args) {
    const char *entry = json_str(json_obj_get(args, "entry"));
    const char *mood = json_str(json_obj_get(args, "mood"));
    char path[4096];
    diary_path(path, sizeof(path), 0);
    FILE *f = fopen(path, "a");
    if (!f) return fmt("ERROR: cannot open diary: %s", strerror(errno));
    time_t t = time(NULL);
    struct tm tm;
    localtime_r(&t, &tm);
    fprintf(f, "\n## %02d:%02d [%s]\n%s\n", tm.tm_hour, tm.tm_min,
            mood[0] ? mood : "neutral", entry);
    fclose(f);
    return fmt("✓ diary entry saved to %s", path);
}

static char *t_read_diary(const jval_t *args) {
    int days = (int)json_num(json_obj_get(args, "days_back"));
    if (days <= 0) days = 7;
    char *out = malloc(65536);
    size_t o = 0;
    out[0] = 0;
    for (int d = days; d >= 0; d--) {
        char path[4096];
        diary_path(path, sizeof(path), d);
        FILE *f = fopen(path, "r");
        if (!f) continue;
        char *content = malloc(32768);
        size_t n = fread(content, 1, 32767, f);
        content[n] = 0;
        fclose(f);
        int w = snprintf(out + o, 65536 - o, "=== %s ===\n%s\n", path, content);
        free(content);
        if (w < 0 || (size_t)w >= 65536 - o) break;
        o += (size_t)w;
        if (o > 60000) break;
    }
    if (o == 0) return xstrdup("(no diary entries in range)");
    return out;
}

/* ---- ask_user ---- */
char *tools_ask_user(const char *question) {
    printf("\n\x1b[1;36m[ask you] %s\x1b[0m\n> ", question ? question : "?");
    fflush(stdout);
    char line[4096];
    if (!fgets(line, sizeof(line), stdin)) return xstrdup("(no answer)");
    size_t l = strlen(line);
    while (l && (line[l-1] == '\n' || line[l-1] == '\r')) line[--l] = 0;
    return xstrdup(line);
}

/* ---- find (pi-style: glob under a directory tree) ---- */
static void find_walk(const char *base, const char *pattern, char *out, size_t *o, int depth) {
    if (depth > 10 || *o > 60000) return;
    DIR *d = opendir(base);
    if (!d) return;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] == '.') continue;
        char full[4096];
        snprintf(full, sizeof(full), "%s/%s", base, e->d_name);
        if (e->d_type == DT_DIR) {
            find_walk(full, pattern, out, o, depth + 1);
        } else {
            if (fnmatch(pattern, e->d_name, 0) == 0) {
                int w = snprintf(out + *o, 65536 - *o, "%s\n", full);
                if (w < 0 || (size_t)w >= 65536 - *o) { closedir(d); return; }
                *o += (size_t)w;
            }
        }
    }
    closedir(d);
}

static char *t_find(const jval_t *args) {
    const char *pattern = json_str(json_obj_get(args, "pattern"));
    const char *path_raw = json_str(json_obj_get(args, "path"));
    char *path = map_path_alloc(path_expand_home_alloc(path_raw));
    if (!pattern[0]) { free(path); return xstrdup("ERROR: pattern is required"); }
    if (!path[0]) { free(path); path = xstrdup("."); }
    char *out = malloc(65536);
    size_t o = 0;
    out[0] = 0;
    find_walk(path, pattern, out, &o, 0);
    free(path);
    if (o == 0) { free(out); return xstrdup("no files matched"); }
    return out;
}

/* ---------------- Termux:API tools (phone integration) ----
 * Shell out to the termux-api binaries (pkg install termux-api). Every
 * command runs via execv — no shell involved, so argument quoting is
 * never an issue. Captures stdout/stderr; returns a heap result string
 * ("ERROR:" prefix marks failures so the model sees them). */
static int termux_find(const char *name, char *out, size_t n) {
    if (strchr(name, '/')) {
        if (access(name, X_OK) != 0) return 0;
        snprintf(out, n, "%s", name);
        return 1;
    }
    const char *path = getenv("PATH");
    if (!path) path = "/data/data/com.termux/files/usr/bin:/usr/bin:/bin";
    const char *p = path;
    while (p && *p) {
        const char *colon = strchr(p, ':');
        size_t plen = colon ? (size_t)(colon - p) : strlen(p);
        if (plen > 0 && plen + strlen(name) + 2 < n) {
            memcpy(out, p, plen);
            out[plen] = '/';
            strcpy(out + plen + 1, name);
            if (access(out, X_OK) == 0) return 1;
        }
        p = colon ? colon + 1 : NULL;
    }
    return 0;
}

static char *termux_exec(char *const argv[], int timeout_s) {
    static char resolved[4096];
    if (!termux_find(argv[0], resolved, sizeof(resolved)))
        return fmt("ERROR: '%s' not found — install it with: pkg install termux-api", argv[0]);
    int pipefd[2];
    if (pipe(pipefd) != 0) return fmt("ERROR: pipe: %s", strerror(errno));
    pid_t pid = fork();
    if (pid < 0) return fmt("ERROR: fork: %s", strerror(errno));
    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], 1);
        dup2(pipefd[1], 2);
        close(pipefd[1]);
        execv(resolved, argv);
        _exit(127);
    }
    close(pipefd[1]);
    char *cap = malloc(16384);
    size_t ccap = 0;
    char buf[4096];
    struct pollfd pfd = { pipefd[0], POLLIN, 0 };
    double deadline = (double)time(NULL) + (timeout_s > 0 ? timeout_s : 30);
    int timed_out = 0;
    for (;;) {
        double now = (double)time(NULL);
        int remaining = now >= deadline ? 0 : (int)(deadline - now);
        int pr = poll(&pfd, 1, remaining * 1000);
        if (pr < 0) { if (errno == EINTR) continue; break; }
        if (pr == 0) { timed_out = 1; break; }
        ssize_t n = read(pipefd[0], buf, sizeof(buf));
        if (n <= 0) break;
        if (ccap + (size_t)n < 16384) { memcpy(cap + ccap, buf, (size_t)n); ccap += (size_t)n; }
    }
    close(pipefd[0]);
    int status = 0;
    if (timed_out) { kill(pid, SIGKILL); waitpid(pid, NULL, 0); status = -1; }
    else waitpid(pid, &status, 0);
    cap[ccap] = 0;
    while (ccap > 0 && (cap[ccap-1] == '\n' || cap[ccap-1] == '\r' || cap[ccap-1] == ' ')) cap[--ccap] = 0;
    char *res;
    if (timed_out)
        res = fmt("ERROR: %s timed out", argv[0]);
    else if (WIFEXITED(status) && WEXITSTATUS(status) == 0)
        res = (cap[0]) ? xstrdup(cap) : fmt("✓ %s", argv[0]);
    else if (WIFEXITED(status))
        res = fmt("ERROR: %s exited %d%s%s", argv[0], WEXITSTATUS(status),
                  cap[0] ? ": " : "", cap);
    else
        res = fmt("ERROR: %s %s", argv[0], WIFSIGNALED(status) ? "killed by signal" : "failed");
    free(cap);
    return res;
}

static char *t_speak(const jval_t *args) {
    const char *text = json_str(json_obj_get(args, "text"));
    if (!text[0]) return xstrdup("ERROR: text is required");
    double rate = json_num(json_obj_get(args, "rate"));
    char ratebuf[32];
    char *argv[6];
    int ai = 0;
    argv[ai++] = (char *)"termux-tts-speak";
    if (rate > 0) {
        snprintf(ratebuf, sizeof(ratebuf), "%.0f", rate);
        argv[ai++] = (char *)"-r";
        argv[ai++] = ratebuf;
    }
    argv[ai++] = (char *)text;
    argv[ai] = NULL;
    char *r = termux_exec(argv, 30);
    if (r && strncmp(r, "ERROR:", 6) == 0) return r;
    free(r);
    return fmt("✓ speaking: %.60s%s", text, strlen(text) > 60 ? "…" : "");
}

static char *t_notify(const jval_t *args) {
    const char *title = json_str(json_obj_get(args, "title"));
    const char *content = json_str(json_obj_get(args, "content"));
    if (!title[0]) return xstrdup("ERROR: title is required");
    char *argv[6];
    int ai = 0;
    argv[ai++] = (char *)"termux-notification";
    argv[ai++] = (char *)"--title";
    argv[ai++] = (char *)title;
    if (content[0]) {
        argv[ai++] = (char *)"--content";
        argv[ai++] = (char *)content;
    }
    argv[ai] = NULL;
    char *r = termux_exec(argv, 15);
    if (r && strncmp(r, "ERROR:", 6) == 0) return r;
    free(r);
    return fmt("✓ notification sent: %s", title);
}

static char *t_vibrate(const jval_t *args) {
    double ms = json_num(json_obj_get(args, "duration_ms"));
    if (ms <= 0) ms = 500;
    char msbuf[32];
    snprintf(msbuf, sizeof(msbuf), "%.0f", ms);
    char *argv[] = { (char *)"termux-vibrate", (char *)"-d", msbuf, NULL };
    char *r = termux_exec(argv, 15);
    if (r && strncmp(r, "ERROR:", 6) == 0) return r;
    free(r);
    return fmt("✓ vibrated for %.0f ms", ms);
}

static char *t_battery(const jval_t *args) {
    (void)args;
    char *argv[] = { (char *)"termux-battery-status", NULL };
    return termux_exec(argv, 15);   /* the raw JSON status is the result */
}

/* ---------------- dispatch ---------------- */

static char *t_self_build(const jval_t *args) {
    (void)args;
    if (!g_intern_root)
        return xstrdup("ERROR: no internal tree mounted (this binary has no embedded source)");
    char src_dir[1200], sess[1200], tmpbin[1200], cmd[8192], log[65536];
    snprintf(src_dir, sizeof(src_dir), "%s/src", g_intern_root);
    snprintf(sess, sizeof(sess), "%s", g_intern_root);
    size_t l = strlen(sess);
    if (l > 9 && strcmp(sess + l - 9, "/internal") == 0) sess[l - 9] = 0;
    snprintf(tmpbin, sizeof(tmpbin), "%s/selfbuild-%d", sess, (int)getpid());
    intern_build_cmd(cmd, sizeof(cmd), src_dir, tmpbin);
    int rc = sys_run_capture(cmd, 240, log, sizeof(log));
    if (rc != 0) {
        size_t ll = strlen(log);
        const char *tail = ll > 3000 ? log + ll - 3000 : log;
        unlink(tmpbin);
        return fmt("ERROR: self_build failed (rc=%d):\n%s", rc, tail);
    }
    char smoke[8192], s2[4096];
    snprintf(smoke, sizeof(smoke), "%s --check-align 2>&1", tmpbin);
    int src = sys_run_capture(smoke, 30, s2, sizeof(s2));
    unlink(tmpbin);
    if (src != 0) {
        size_t ll = strlen(s2);
        const char *tail = ll > 2000 ? s2 + ll - 2000 : s2;
        return fmt("ERROR: self_build compiled but failed the smoke test:\n%s", tail);
    }
    return xstrdup("✓ self_build OK — /internal/src compiles and passes the smoke test");
}

/* ---- LFM document workers ----
 * The main agent spawns small LFM2.5 workers to read documents and answer
 * questions about them. Worker source lives at /internal/src/lfm/ and the
 * shared worker model path comes from QMA_WORKER_MODEL (or a default). */
static const char *worker_model_path(void) {
    static char p[1024] = "";
    const char *e = getenv("QMA_WORKER_MODEL");
    if (e && e[0]) return e;
    if (!p[0])
        snprintf(p, sizeof(p), "/data/data/com.termux/files/home/projects/models/lfm2.5:2.6b:Q4_K_M.gguf");
    return p;
}

static char *t_worker_spawn(const jval_t *args) {
    const char *name = json_str(json_obj_get(args, "name"));
    const char *path = json_str(json_obj_get(args, "path"));
    const char *doc = json_str(json_obj_get(args, "doc"));
    const char *role = json_str(json_obj_get(args, "role"));
    if (worker_model_load(worker_model_path()) != 0)
        return xstrdup("ERROR: LFM worker model failed to load — set QMA_WORKER_MODEL to the lfm2.5:2.6b Q4_K_M gguf");
    char *content = NULL;
    if (path && path[0]) {
        content = read_whole_file(path);
        if (!content || strncmp(content, "ERROR:", 6) == 0)
            return content ? content : xstrdup("ERROR: cannot read document path");
        doc = content;
    }
    if (!doc || !doc[0]) { free(content); return xstrdup("ERROR: worker_spawn needs a document — pass path (preferred) or doc"); }
    int id = worker_spawn(name, role, doc);
    free(content);
    if (id < 0)
        return fmt("ERROR: worker_spawn failed (rc=%d) — worker pool full, or the document is too large for the worker context", id);
    return fmt("✓ worker %d spawned (%s) — ask it questions with worker_ask(%d, \"...\")",
               id, name && name[0] ? name : "unnamed", id);
}

static char *t_worker_ask(const jval_t *args) {
    int id = (int)json_num(json_obj_get(args, "id"));
    const char *q = json_str(json_obj_get(args, "question"));
    int ephemeral = json_obj_get(args, "ephemeral") ? json_bool(json_obj_get(args, "ephemeral")) : 1;
    int mt = (int)json_num(json_obj_get(args, "max_tokens"));
    return worker_ask(id, q, ephemeral, mt);
}

static char *t_worker_close(const jval_t *args) {
    int id = (int)json_num(json_obj_get(args, "id"));
    if (worker_close(id) != 0) return fmt("ERROR: no such worker %d", id);
    return fmt("✓ worker %d closed (KV freed)", id);
}

static char *t_worker_list(const jval_t *args) {
    (void)args;
    return worker_list();
}

int tool_dispatch(const char *name, const char *args_json, char **result) {
    jval_t *args = json_parse(args_json);
    if (!args) {
        *result = fmt("ERROR: malformed arguments JSON: %.120s", args_json);
        return 0;
    }
    char *r = NULL;
    if (strcmp(name, "pwd") == 0) r = t_pwd();
    else if (strcmp(name, "ls") == 0) r = t_list_dir(args);
    else if (strcmp(name, "grep") == 0) r = t_search(args);
    else if (strcmp(name, "read") == 0) r = t_read_file(args);
    else if (strcmp(name, "write") == 0) r = t_write_file(args);
    else if (strcmp(name, "find") == 0) r = t_find(args);
    else if (strcmp(name, "edit") == 0) r = t_edit(args);
    else if (strcmp(name, "replace_lines") == 0) r = t_replace_lines(args);
    else if (strcmp(name, "insert_lines") == 0) r = t_insert_lines(args);
    else if (strcmp(name, "delete_lines") == 0) r = t_delete_lines(args);
    else if (strcmp(name, "bash") == 0) r = t_execute_command(args);
    else if (strcmp(name, "enter") == 0) r = t_enter(args);
    else if (strcmp(name, "todo_write") == 0) r = t_todo_write(args);
    else if (strcmp(name, "todo_complete") == 0) r = t_todo_complete(args);
    else if (strcmp(name, "todo_remove") == 0) r = t_todo_remove(args);
    else if (strcmp(name, "write_diary") == 0) r = t_write_diary(args);
    else if (strcmp(name, "read_diary") == 0) r = t_read_diary(args);
    else if (strcmp(name, "ask_user") == 0) r = tools_ask_user(json_str(json_obj_get(args, "question")));
    else if (strcmp(name, "speak") == 0) r = t_speak(args);
    else if (strcmp(name, "notify") == 0) r = t_notify(args);
    else if (strcmp(name, "vibrate") == 0) r = t_vibrate(args);
    else if (strcmp(name, "battery") == 0) r = t_battery(args);
    else if (strcmp(name, "self_build") == 0) r = t_self_build(args);
    else if (strcmp(name, "memory_write") == 0) r = agent_memory_write(args);
    else if (strcmp(name, "memory_append") == 0) r = agent_memory_append(args);
    else if (strcmp(name, "memory_list") == 0) r = agent_memory_list(args);
    else if (strcmp(name, "memory_delete") == 0) r = agent_memory_delete(args);
    else if (strcmp(name, "memory_clear") == 0) r = agent_memory_clear();
    else if (strcmp(name, "worker_spawn") == 0) r = t_worker_spawn(args);
    else if (strcmp(name, "worker_ask") == 0) r = t_worker_ask(args);
    else if (strcmp(name, "worker_close") == 0) r = t_worker_close(args);
    else if (strcmp(name, "worker_list") == 0) r = t_worker_list(args);
    else r = fmt("ERROR: unknown tool '%s'", name);
    json_free(args);
    if (!r) r = xstrdup("(no output)");
    *result = r;
    return 0;
}

void tools_render_header(char *out, size_t outsz) {
    size_t o = 0;
    #define APP(...) do { int w = snprintf(out + o, outsz - o, __VA_ARGS__); \
        if (w < 0 || (size_t)w >= outsz - o) return; o += (size_t)w; } while (0)
    APP("# Tools\n\nYou have access to the following functions:\n\n<tools>\n");
    for (int i = 0; i < g_n_tools; i++) {
        APP("{\"type\":\"function\",\"function\":{\"name\":\"%s\",\"description\":\"%s\",\"parameters\":%s}}\n",
            g_tools[i].name, g_tools[i].description, g_tools[i].params_json);
    }
    APP("</tools>\n\n"
        "If you choose to call a function ONLY reply in the following format with NO suffix:\n"
        "\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
        "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n"
        "</function>\n</tool_call>\n"
        "\n<IMPORTANT>\nReminder:\n"
        "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
        "- You may emit several <tool_call> blocks in one message when the steps are independent — the engine runs them all and returns the results together\n"
        "- Required parameters MUST be specified\n"
        "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
        "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
        "</IMPORTANT>\n");
    #undef APP
}

