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
     "Change current working directory",
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
};

const int g_n_tools = (int)(sizeof(g_tools) / sizeof(g_tools[0]));

/* ---------------- helpers ---------------- */

static char *xstrdup(const char *s) {
    char *p = strdup(s ? s : "");
    return p;
}

static char *fmt(const char *fmt, ...) {
    va_list ap;
    char buf[8192];
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    return xstrdup(buf);
}

#include <stdarg.h>

/* path guard: returns NULL if ok, else a heap error string */
static char *path_guard(const char *path) {
    if (!path || !path[0])
        return fmt("ERROR: path is REQUIRED and must be a file path, never empty. Use list_directory to see existing dirs.");
    if (path[strlen(path)-1] == '/')
        return fmt("ERROR: '%s' is a directory. path must be a file path (e.g. 'main.c' or 'src/main.c').", path);
    struct stat st;
    if (stat(path, &st) == 0 && S_ISDIR(st.st_mode))
        return fmt("ERROR: '%s' is a directory. path must be a file path (e.g. 'main.c' or 'src/main.c').", path);
    return NULL;
}

static char *read_whole_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return fmt("ERROR: cannot open '%s': %s", path, strerror(errno));
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); return fmt("ERROR: cannot read '%s'", path); }
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return fmt("ERROR: OOM reading '%s'", path); }
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[rd] = 0;
    return buf;
}

/* ---------------- tool implementations ---------------- */

static char *t_pwd(void) {
    char buf[4096];
    if (!getcwd(buf, sizeof(buf))) return fmt("ERROR: getcwd: %s", strerror(errno));
    return fmt("%s", buf);
}

static char *t_list_dir(const jval_t *args) {
    const char *path = json_str(json_obj_get(args, "path"));
    if (!path[0]) path = ".";
    int hidden = json_bool(json_obj_get(args, "show_hidden"));
    int lng = json_bool(json_obj_get(args, "long_format"));
    DIR *d = opendir(path);
    if (!d) return fmt("ERROR: cannot open '%s': %s", path, strerror(errno));
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
    return out;
}

static char *t_search(const jval_t *args) {
    const char *pattern = json_str(json_obj_get(args, "pattern"));
    const char *path = json_str(json_obj_get(args, "path"));
    int recursive = json_bool(json_obj_get(args, "recursive"));
    if (!pattern[0]) return fmt("ERROR: pattern is required");
    if (!path[0]) path = ".";
    /* grep -rn (or -n for single file/dir) */
    char cmd[8192];
    /* simple shell-quote the pattern and path */
    char qpat[2048], qpath[2048];
    snprintf(qpat, sizeof(qpat), "'%s'", pattern);
    snprintf(qpath, sizeof(qpath), "'%s'", path);
    snprintf(cmd, sizeof(cmd), "grep -n %s %s %s 2>&1 | head -200",
             recursive ? "-r" : "", qpat, qpath);
    FILE *f = popen(cmd, "r");
    if (!f) return fmt("ERROR: grep failed");
    char *out = malloc(65536);
    size_t o = 0;
    out[0] = 0;
    char line[2048];
    while (fgets(line, sizeof(line), f)) {
        if (o + strlen(line) + 1 < 65536) { strcpy(out + o, line); o += strlen(line); }
    }
    pclose(f);
    if (o == 0) { free(out); return xstrdup("no matches"); }
    return out;
}

static char *t_read_file(const jval_t *args) {
    const char *path = json_str(json_obj_get(args, "path"));
    int start = (int)json_num(json_obj_get(args, "start_line"));
    int maxl = (int)json_num(json_obj_get(args, "max_lines"));
    char *err = path_guard(path);
    if (err) return err;
    if (start < 1) start = 1;
    if (maxl <= 0) maxl = 500;
    char *content = read_whole_file(path);
    if (!content || strncmp(content, "ERROR:", 6) == 0) return content ? content : xstrdup("ERROR: read failed");
    /* split lines, render with numbers */
    char *out = malloc(262144);
    size_t o = 0;
    out[0] = 0;
    int line = 1;
    char *save = NULL;
    char *tok = strtok_r(content, "\n", &save);
    while (tok) {
        if (line > start - 1 + maxl) break;
        if (line >= start) {
            int w = snprintf(out + o, 262144 - o, "%6d\t%s\n", line, tok);
            if (w > 0) o += (size_t)w;
        }
        line++;
        tok = strtok_r(NULL, "\n", &save);
    }
    free(content);
    return out;
}

static char *t_write_file(const jval_t *args) {
    const char *path = json_str(json_obj_get(args, "path"));
    const char *content = json_str(json_obj_get(args, "content"));
    char *err = path_guard(path);
    if (err) return err;
    /* create parent dirs */
    char copy[4096];
    snprintf(copy, sizeof(copy), "%s", path);
    for (char *p = copy; (p = strchr(p, '/')) != NULL; p++) {
        *p = 0;
        if (copy[0]) mkdir(copy, 0755);
        *p = '/';
    }
    FILE *f = fopen(path, "w");
    if (!f) return fmt("ERROR: cannot write '%s': %s", path, strerror(errno));
    fwrite(content, 1, strlen(content), f);
    fclose(f);
    return fmt("✓ wrote %zu bytes to %s", strlen(content), path);
}

static char *t_edit(const jval_t *args) {
    const char *path = json_str(json_obj_get(args, "path"));
    char *err = path_guard(path);
    if (err) return err;
    const jval_t *edits = json_obj_get(args, "edits");
    if (!edits || edits->type != J_ARR || edits->n == 0)
        return xstrdup("ERROR: edit requires an edits array of {oldText,newText} pairs");
    char *content = read_whole_file(path);
    if (!content || strncmp(content, "ERROR:", 6) == 0) return content ? content : xstrdup("ERROR: read failed");
    char *cur = content;
    for (size_t i = 0; i < edits->n; i++) {
        const jval_t *pr = json_arr_get(edits, i);
        if (!pr || pr->type != J_OBJ) { free(content); return fmt("ERROR: edits[%zu] is not an object", i); }
        const char *oldT = json_str(json_obj_get(pr, "oldText"));
        const char *newT = json_str(json_obj_get(pr, "newText"));
        if (!oldT[0]) { free(content); return fmt("ERROR: edits[%zu] oldText is empty", i); }
        /* count occurrences */
        int count = 0;
        for (const char *p = cur; (p = strstr(p, oldT)) != NULL; p += strlen(oldT)) count++;
        if (count != 1) { free(content); return fmt("ERROR: oldText must match exactly once (found %d): %.80s", count, oldT); }
        /* replace */
        size_t cl = strlen(cur), ol = strlen(oldT), nl = strlen(newT);
        char *next = malloc(cl - ol + nl + 1);
        if (!next) { free(content); return fmt("ERROR: OOM"); }
        char *hit = strstr(cur, oldT);
        size_t pre = (size_t)(hit - cur);
        memcpy(next, cur, pre);
        memcpy(next + pre, newT, nl);
        memcpy(next + pre + nl, hit + ol, cl - pre - ol + 1);
        free(cur);
        cur = next;
    }
    FILE *f = fopen(path, "w");
    if (!f) { free(content); free(cur); return fmt("ERROR: cannot write '%s': %s", path, strerror(errno)); }
    fwrite(cur, 1, strlen(cur), f);
    fclose(f);
    size_t n = strlen(cur);
    free(cur); free(content);
    return fmt("✓ edited %s (%zu bytes)", path, n);
}

/* line-range helpers: split content into lines array */
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
    char *content = read_whole_file(path);
    if (!content || strncmp(content, "ERROR:", 6) == 0) return content ? content : xstrdup("ERROR: read failed");
    lines_t ls;
    if (lines_split(&ls, content) != 0) { free(content); return xstrdup("ERROR: OOM"); }
    if (s < 1 || e > ls.n || e < s) { lines_free(&ls); free(content); return fmt("ERROR: bad range %d-%d (file has %d lines)", s, e, ls.n); }
    /* build new lines: 1..s-1, new_content lines, e+1..n */
    int nl = 1;
    const char *p = nc;
    while (*p) { if (*p == '\n') nl++; p++; }
    char **newlines = malloc((size_t)(ls.n - (e - s + 1) + nl) * sizeof(char *));
    int k = 0;
    for (int i = 0; i < s - 1; i++) newlines[k++] = ls.lines[i];
    /* split new_content */
    {
        char tmp[65536];
        snprintf(tmp, sizeof(tmp), "%s", nc);
        char *save = NULL;
        for (char *t = strtok_r(tmp, "\n", &save); t; t = strtok_r(NULL, "\n", &save)) {
            newlines[k] = xstrdup(t);
            k++;
        }
        if (k == 0 || (k == 1 && !newlines[0][0])) { /* empty content */ }
    }
    for (int i = e; i < ls.n; i++) newlines[k++] = ls.lines[i];
    /* rebuild */
    lines_t nl2 = { newlines, k, k };
    char *out = lines_join(&nl2);
    FILE *f = fopen(path, "w");
    if (!f) { free(out); lines_free(&ls); free(content); return fmt("ERROR: cannot write '%s': %s", path, strerror(errno)); }
    fwrite(out, 1, strlen(out), f);
    fclose(f);
    free(out); free(newlines); lines_free(&ls); free(content);
    return fmt("✓ replaced lines %d-%d in %s", s, e, path);
}

static char *t_insert_lines(const jval_t *args) {
    const char *path = json_str(json_obj_get(args, "path"));
    int after = (int)json_num(json_obj_get(args, "after_line"));
    const char *content = json_str(json_obj_get(args, "content"));
    char *err = path_guard(path);
    if (err) return err;
    char *file = read_whole_file(path);
    if (!file || strncmp(file, "ERROR:", 6) == 0) return file ? file : xstrdup("ERROR: read failed");
    lines_t ls;
    if (lines_split(&ls, file) != 0) { free(file); return xstrdup("ERROR: OOM"); }
    if (after < 0 || after > ls.n) { lines_free(&ls); free(file); return fmt("ERROR: after_line %d out of range (file has %d lines)", after, ls.n); }
    /* build */
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
    FILE *f = fopen(path, "w");
    if (!f) { free(out); free(newlines); lines_free(&ls); free(file); return fmt("ERROR: cannot write '%s': %s", path, strerror(errno)); }
    fwrite(out, 1, strlen(out), f);
    fclose(f);
    free(out); free(newlines); lines_free(&ls); free(file);
    return fmt("✓ inserted %d lines after line %d in %s", nl, after, path);
}

static char *t_delete_lines(const jval_t *args) {
    const char *path = json_str(json_obj_get(args, "path"));
    int s = (int)json_num(json_obj_get(args, "start_line"));
    int e = (int)json_num(json_obj_get(args, "end_line"));
    char *err = path_guard(path);
    if (err) return err;
    char *file = read_whole_file(path);
    if (!file || strncmp(file, "ERROR:", 6) == 0) return file ? file : xstrdup("ERROR: read failed");
    lines_t ls;
    if (lines_split(&ls, file) != 0) { free(file); return xstrdup("ERROR: OOM"); }
    if (s < 1 || e > ls.n || e < s) { lines_free(&ls); free(file); return fmt("ERROR: bad range %d-%d (file has %d lines)", s, e, ls.n); }
    int removed = e - s + 1;
    char **newlines = malloc((size_t)(ls.n - removed) * sizeof(char *));
    int k = 0;
    for (int i = 0; i < s - 1; i++) newlines[k++] = ls.lines[i];
    for (int i = e; i < ls.n; i++) newlines[k++] = ls.lines[i];
    lines_t nl2 = { newlines, k, k };
    char *out = lines_join(&nl2);
    FILE *f = fopen(path, "w");
    if (!f) { free(out); free(newlines); lines_free(&ls); free(file); return fmt("ERROR: cannot write '%s': %s", path, strerror(errno)); }
    fwrite(out, 1, strlen(out), f);
    fclose(f);
    free(out); free(newlines); lines_free(&ls); free(file);
    return fmt("✓ deleted lines %d-%d from %s", s, e, path);
}

static char *t_execute_command(const jval_t *args) {
    const char *cmd = json_str(json_obj_get(args, "command"));
    int timeout = (int)json_num(json_obj_get(args, "timeout"));
    if (!cmd[0]) return xstrdup("ERROR: command is required");
    if (timeout <= 0) timeout = 60;
    /* fork + pipe, tee output to stdout, capture last ~200 lines, timeout via poll */
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
    /* keep the last 200 lines */
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
    char *out = fmt("%s%s%s", timed_out ? "ERROR: timed out after 60s\n" : "",
                    tail, rc == 0 ? "" : "");
    free(cap);
    return out;
}

static char *t_enter(const jval_t *args) {
    const char *path = json_str(json_obj_get(args, "path"));
    if (!path[0]) return xstrdup("ERROR: path is required");
    if (chdir(path) != 0) return fmt("ERROR: cannot enter '%s': %s", path, strerror(errno));
    char buf[4096];
    getcwd(buf, sizeof(buf));
    return fmt("✓ now in %s", buf);
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
        if (w > 0) o += (size_t)w;
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
        if (w > 0) o += (size_t)w;
        free(content);
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
                if (w > 0) *o += (size_t)w;
            }
        }
    }
    closedir(d);
}

static char *t_find(const jval_t *args) {
    const char *pattern = json_str(json_obj_get(args, "pattern"));
    const char *path = json_str(json_obj_get(args, "path"));
    if (!pattern[0]) return xstrdup("ERROR: pattern is required");
    if (!path[0]) path = ".";
    char *out = malloc(65536);
    size_t o = 0;
    out[0] = 0;
    find_walk(path, pattern, out, &o, 0);
    if (o == 0) { free(out); return xstrdup("no files matched"); }
    return out;
}

/* ---------------- dispatch ---------------- */


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
        "- Required parameters MUST be specified\n"
        "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
        "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
        "</IMPORTANT>\n");
    #undef APP
}
