/* remote_tools.c — headless tool surface for qma's remote agent.
 * The LFM model's NATIVE tool format is the Hermes/Pythonic style:
 *   <|tool_call_start|>[name(key='value', key2="value2")]<|tool_call_end|>
 * (NOT qma's qwen XML format). Tool definitions go in a
 * <|tool_list_start|>...<|tool_list_end|> block in the system prompt;
 * results fold back as <|tool_response_start|>...<|tool_response_end|>.
 *
 * Tools: ls, read, write, bash, collect (copy a remote path into this
 * agent's internal tree so it carries home), memory_write, memory_append,
 * memory_list, done (declare mission complete).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include "remote.h"
#include "json.h"

/* ---- memory hooks implemented in remote_main.c (they need the model
   + runstate to eval content into the always-attended KV) ---- */
int remote_memory_write(const char *key, const char *content);
int remote_memory_append(const char *key, const char *content);
char *remote_memory_list(void);

static char *xstrdup(const char *s) { return strdup(s ? s : ""); }

static char *fmt(const char *f, ...) {
    va_list ap; va_start(ap, f);
    char buf[8192];
    vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return xstrdup(buf);
}

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return fmt("ERROR: cannot open '%s': %s", path, strerror(errno));
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); return fmt("ERROR: read '%s'", path); }
    char *buf = malloc((size_t)sz + 1);
    size_t rd = buf ? fread(buf, 1, (size_t)sz, f) : 0;
    fclose(f);
    if (!buf) return fmt("ERROR: OOM reading '%s'", path);
    buf[rd] = 0;
    return buf;
}

/* ---- LFM Pythonic tool-call parser ----
 * <|tool_call_start|>[name(key='val', k2="v2", n=3, b=true)]<|tool_call_end|>
 * -> name + JSON {"key":"val","k2":"v2","n":3,"b":true}. Returns 1 + fills
 * name/args; 0 = open/partial; -1 = malformed. */
int remote_parse_call(const char *text, const char **after,
                      char *name, size_t name_cap, char *args, size_t args_cap) {
    const char *start = strstr(text, "<|tool_call_start|>");
    if (!start) { *after = text; return 0; }
    const char *body = start + strlen("<|tool_call_start|>");
    if (*body == '[') body++;
    const char *end = strstr(body, "<|tool_call_end|>");
    const char *close = end ? end : NULL;
    if (!close) return 0;                              /* still open */
    /* function name */
    const char *b = body;
    while (b < close && ((*b >= 'a' && *b <= 'z') || (*b >= 'A' && *b <= 'Z') ||
                         (*b >= '0' && *b <= '9') || *b == '_'))
        b++;
    size_t fn = (size_t)(b - body);
    if (fn == 0 || fn >= name_cap) return -1;
    memcpy(name, body, fn); name[fn] = 0;
    /* args: (key=value, ...) or none */
    while (b < close && *b != '(' && *b != ']') b++;
    if (b < close && *b == '(') {
        const char *p = b + 1;
        size_t o = 0;
        if (args_cap > 1) args[o++] = '{';
        int first = 1;
        while (p < close) {
            while (p < close && (*p == ' ' || *p == '\t' || *p == ',')) p++;
            if (p >= close || *p == ')') break;
            const char *keq = memchr(p, '=', (size_t)(close - p));
            if (!keq || keq >= close) return -1;
            size_t kl = (size_t)(keq - p);
            if (kl == 0 || kl > 500) return -1;
            char kb[512];
            memcpy(kb, p, kl); kb[kl] = 0;
            p = keq + 1;
            while (p < close && (*p == ' ' || *p == '\t')) p++;
            if (p >= close) return -1;
            char vb[131072] = "";
            if (*p == '\'' || *p == '"') {
                char q = *p++;
                const char *vs = p;
                while (p < close && *p != q) p++;
                if (p >= close) return -1;
                size_t vl = (size_t)(p - vs);
                if (vl >= sizeof(vb)) vl = sizeof(vb) - 1;
                memcpy(vb, vs, vl); vb[vl] = 0;
                p++;                       /* past closing quote */
            } else {
                const char *vs = p;
                while (p < close && *p != ',' && *p != ')') p++;
                size_t vl = (size_t)(p - vs);
                if (vl >= sizeof(vb)) vl = sizeof(vb) - 1;
                memcpy(vb, vs, vl); vb[vl] = 0;
            }
            char ke[1024], ve[262144];
            json_quote_escape(kb, ke, sizeof(ke));
            json_quote_escape(vb, ve, sizeof(ve));
            int w = snprintf(args + o, args_cap - o, "%s%s:%s", first ? "" : ",", ke, ve);
            if (w < 0 || (size_t)w >= args_cap - o) return -1;
            o += (size_t)w;
            first = 0;
        }
        if (o + 1 < args_cap) { args[o++] = '}'; args[o] = 0; }
    } else {
        args[0] = '{'; args[1] = '}'; args[2] = 0;
    }
    *after = close + strlen("<|tool_call_end|>");
    return 1;
}

/* ---- tools ---- */
static char *t_ls(const char *path) {
    DIR *d = opendir(path && path[0] ? path : ".");
    if (!d) return fmt("ERROR: cannot open '%s': %s", path ? path : ".", strerror(errno));
    char *out = malloc(65536); size_t o = 0; out[0] = 0;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] == '.') continue;
        int w = snprintf(out + o, 65536 - o, "%s%s\n", e->d_name, e->d_type == DT_DIR ? "/" : "");
        if (w > 0) o += (size_t)w;
    }
    closedir(d);
    return out;
}

static char *t_read(const char *path, int start, int maxl) {
    char *c = read_file(path);
    if (!c || strncmp(c, "ERROR:", 6) == 0) return c ? c : fmt("ERROR: read failed");
    /* render numbered lines [start, start+maxl) */
    char *out = malloc(262144); size_t o = 0; out[0] = 0;
    int line = 1;
    char *save = NULL, *tok = strtok_r(c, "\n", &save);
    while (tok) {
        if (line >= start && line < start + maxl) {
            int w = snprintf(out + o, 262144 - o, "%6d\t%s\n", line, tok);
            if (w > 0) o += (size_t)w;
        }
        line++;
        tok = strtok_r(NULL, "\n", &save);
    }
    free(c);
    return out;
}

static char *t_write(const char *path, const char *content) {
    if (!content || !content[0]) return xstrdup("ERROR: content is empty");
    FILE *f = fopen(path, "w");
    if (!f) return fmt("ERROR: cannot write '%s': %s", path, strerror(errno));
    fwrite(content, 1, strlen(content), f);
    fclose(f);
    return fmt("✓ wrote %zu bytes to %s", strlen(content), path);
}

static char *t_bash(const char *cmd) {
    char out[65536]; out[0] = 0;
    FILE *p = popen(cmd ? cmd : "true", "r");
    if (!p) return xstrdup("ERROR: popen failed");
    size_t o = 0;
    char line[512];
    while (fgets(line, sizeof(line), p) && o + strlen(line) + 1 < 65536) {
        strcpy(out + o, line); o += strlen(line);
    }
    int rc = pclose(p);
    return fmt("%s%s", out, rc != 0 ? "\n[exit nonzero]" : "");
}

/* copy a remote path into /internal/collect/ so it travels home */
static char *t_collect(const char *path) {
    const char *cd = remote_collect_dir();
    char *content = read_file(path);
    if (!content || strncmp(content, "ERROR:", 6) == 0)
        return content ? content : fmt("ERROR: cannot collect '%s'", path);
    const char *slash = strrchr(path, '/');
    const char *base = slash ? slash + 1 : path;
    char dst[1200];
    snprintf(dst, sizeof(dst), "%s/%s", cd, base);
    FILE *f = fopen(dst, "w");
    if (!f) { free(content); return fmt("ERROR: cannot store collect '%s'", dst); }
    fwrite(content, 1, strlen(content), f);
    fclose(f);
    size_t n = strlen(content);
    free(content);
    return fmt("✓ collected '%s' (%zu bytes) — it will travel home in this binary", base, n);
}

/* ---- dispatch ---- */
int remote_tool_dispatch(const char *name, const char *args_json, char **result) {
    jval_t *args = json_parse(args_json);
    if (!args) { *result = fmt("ERROR: bad args JSON: %.120s", args_json); return 0; }
    char *r = NULL;
    if (strcmp(name, "ls") == 0) r = t_ls(json_str(json_obj_get(args, "path")));
    else if (strcmp(name, "read") == 0)
        r = t_read(json_str(json_obj_get(args, "path")),
                   (int)json_num(json_obj_get(args, "start_line")) ?: 1,
                   (int)json_num(json_obj_get(args, "max_lines")) ?: 500);
    else if (strcmp(name, "write") == 0)
        r = t_write(json_str(json_obj_get(args, "path")), json_str(json_obj_get(args, "content")));
    else if (strcmp(name, "bash") == 0) r = t_bash(json_str(json_obj_get(args, "command")));
    else if (strcmp(name, "collect") == 0) r = t_collect(json_str(json_obj_get(args, "path")));
    else if (strcmp(name, "memory_write") == 0)
        r = fmt("memory_write rc=%d", remote_memory_write(
                   json_str(json_obj_get(args, "key")), json_str(json_obj_get(args, "content"))));
    else if (strcmp(name, "memory_append") == 0)
        r = fmt("memory_append rc=%d", remote_memory_append(
                   json_str(json_obj_get(args, "key")), json_str(json_obj_get(args, "content"))));
    else if (strcmp(name, "memory_list") == 0) r = remote_memory_list();
    else if (strcmp(name, "done") == 0) r = xstrdup("✓ mission complete");
    else r = fmt("ERROR: unknown tool '%s'", name);
    json_free(args);
    if (!r) r = xstrdup("(no output)");
    *result = r;
    return 0;
}

/* the tool list rendered into the model's system prompt (LFM native:
   a <|tool_list_start|> block; calls use the Pythonic format). */
const char *remote_tools_header(void) {
    return
        "You are an autonomous remote agent. You have these tools:\n"
        "<|tool_list_start|>\n"
        "ls(path='.')\n"
        "read(path, start_line=1, max_lines=500)\n"
        "write(path, content)\n"
        "bash(command)\n"
        "collect(path)  — copy a file into your internal tree so it travels home\n"
        "memory_write(key, content) — pin a fact into your always-attended memory\n"
        "memory_append(key, content)\n"
        "memory_list()\n"
        "done()  — declare the mission complete\n"
        "<|tool_list_end|>\n"
        "To call a tool, reply EXACTLY:\n"
        "<|tool_call_start|>[tool_name(arg='value')]<|tool_call_end|>\n"
        "The engine runs it and returns <|tool_response_start|>result<|tool_response_end|>.\n"
        "Keep working until the mission is complete, then call done().\n";
}

void remote_tools_init(void) { }
