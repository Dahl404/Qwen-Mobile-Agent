/* json.c — tiny JSON parser (no deps).
 * Recursive descent over a NUL-terminated string. Nodes point into the
 * input (strings are not copied), so the input must outlive the tree.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "json.h"

static const char *jws(const char *p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

/* parse a string at *pp; on success advances *pp past the closing quote
   and returns a malloc'd jval (J_STR) whose s is an OWNED, NUL-terminated
   copy of the unescaped string bytes. */
static jval_t *jparse_string(const char **pp) {
    const char *p = *pp;
    if (*p != '"') return NULL;
    p++;
    /* unescape into an owned buffer (the previous implementation copied
       the raw bytes and left \n \t \" \\ as literal text, so every
       multi-line tool value was written corrupted) */
    size_t cap = 64, len = 0;
    char *copy = malloc(cap);
    if (!copy) return NULL;
    int bad = 0;
    while (*p && *p != '"') {
        if (len + 8 >= cap) {
            size_t ncap = cap * 2;
            char *n = realloc(copy, ncap);
            if (!n) { bad = 1; break; }
            copy = n;
            cap = ncap;
        }
        unsigned char c = (unsigned char)*p;
        if (c == '\\') {
            p++;
            if (!*p) { bad = 1; break; }
            switch (*p) {
            case 'n':  copy[len++] = '\n'; p++; break;
            case 'r':  copy[len++] = '\r'; p++; break;
            case 't':  copy[len++] = '\t'; p++; break;
            case 'b':  copy[len++] = '\b'; p++; break;
            case 'f':  copy[len++] = '\f'; p++; break;
            case '"':  copy[len++] = '"'; p++; break;
            case '\\': copy[len++] = '\\'; p++; break;
            case '/':  copy[len++] = '/'; p++; break;
            case 'u': {
                uint32_t u = 0;
                int ok = 1;
                for (int i = 0; i < 4; i++) {
                    p++;
                    unsigned char h = (unsigned char)*p;
                    uint32_t v;
                    if (h >= '0' && h <= '9') v = (uint32_t)(h - '0');
                    else if (h >= 'a' && h <= 'f') v = (uint32_t)(h - 'a' + 10);
                    else if (h >= 'A' && h <= 'F') v = (uint32_t)(h - 'A' + 10);
                    else { ok = 0; break; }
                    u = (u << 4) | v;
                }
                if (!ok) { bad = 1; break; }
                p++;   /* past the 4th hex digit */
                if (u < 0x80) copy[len++] = (char)u;
                else if (u < 0x800) {
                    copy[len++] = (char)(0xC0 | (u >> 6));
                    copy[len++] = (char)(0x80 | (u & 0x3F));
                } else {
                    copy[len++] = (char)(0xE0 | (u >> 12));
                    copy[len++] = (char)(0x80 | ((u >> 6) & 0x3F));
                    copy[len++] = (char)(0x80 | (u & 0x3F));
                }
                break;
            }
            default: bad = 1; break;   /* unknown escape: refuse */
            }
        } else {
            copy[len++] = (char)c;
            p++;
        }
    }
    if (bad || *p != '"') { free(copy); return NULL; }
    copy[len] = 0;
    jval_t *v = calloc(1, sizeof(jval_t));
    if (!v) { free(copy); return NULL; }
    v->type = J_STR;
    v->slen = len;
    v->s = copy;
    *pp = p + 1;
    return v;
}

static jval_t *jparse_value(const char **pp);

static jval_t *jparse_array(const char **pp) {
    const char *p = *pp;               /* at '[' */
    p++;
    jval_t *v = calloc(1, sizeof(jval_t));
    if (!v) return NULL;
    v->type = J_ARR;
    size_t cap = 4;
    v->items = malloc(cap * sizeof(jval_t));
    if (!v->items) { free(v); return NULL; }
    p = jws(p);
    if (*p == ']') { *pp = p + 1; return v; }
    for (;;) {
        jval_t *it = jparse_value(&p);
        if (!it) { free(v->items); free(v); return NULL; }
        if (v->n >= cap) {
            cap *= 2;
            jval_t *ni = realloc(v->items, cap * sizeof(jval_t));
            if (!ni) { free(it); free(v->items); free(v); return NULL; }
            v->items = ni;
        }
        v->items[v->n++] = *it;
        free(it);
        p = jws(p);
        if (*p == ',') { p = jws(p + 1); continue; }
        if (*p == ']') { *pp = p + 1; return v; }
        free(v->items); free(v); return NULL;
    }
}

static jval_t *jparse_object(const char **pp) {
    const char *p = *pp;               /* at '{' */
    p++;
    jval_t *v = calloc(1, sizeof(jval_t));
    if (!v) return NULL;
    v->type = J_OBJ;
    size_t cap = 4;
    v->items = malloc(cap * sizeof(jval_t));
    v->keys = malloc(cap * sizeof(const char *));
    if (!v->items || !v->keys) { free(v->items); free(v->keys); free(v); return NULL; }
    p = jws(p);
    if (*p == '}') { *pp = p + 1; return v; }
    for (;;) {
        jval_t *k = jparse_string(&p);
        if (!k || k->type != J_STR) { free(k); free(v->items); free(v->keys); free(v); return NULL; }
        p = jws(p);
        if (*p != ':') { free(k); free(v->items); free(v->keys); free(v); return NULL; }
        p = jws(p + 1);
        jval_t *it = jparse_value(&p);
        if (!it) { free(k); free(v->items); free(v->keys); free(v); return NULL; }
        if (v->n >= cap) {
            cap *= 2;
            jval_t *ni = realloc(v->items, cap * sizeof(jval_t));
            const char **nk = realloc(v->keys, cap * sizeof(const char *));
            if (!ni || !nk) { free(ni); free(nk); free(k); free(it); free(v->items); free(v->keys); free(v); return NULL; }
            v->items = ni; v->keys = nk;
        }
        /* keys point into input (not NUL-terminated) — store offset+len via a
           small helper: we keep them NUL-terminated by noting the input is
           NUL-terminated at the end; keys are bounded by the following ':',
           so copy the key into a heap string for safety. */
        {
            char *key = malloc(k->slen + 1);
            if (!key) { free(k); free(it); free(v->items); free(v->keys); free(v); return NULL; }
            memcpy(key, k->s, k->slen);
            key[k->slen] = 0;
            v->keys[v->n] = key;
        }
        v->items[v->n++] = *it;
        free(k); free(it);
        p = jws(p);
        if (*p == ',') { p = jws(p + 1); continue; }
        if (*p == '}') { *pp = p + 1; return v; }
        free(v->items); free(v->keys); free(v); return NULL;
    }
}

static jval_t *jparse_value(const char **pp) {
    const char *p = jws(*pp);
    jval_t *v;
    if (*p == '"') {
        *pp = p;
        return jparse_string(pp);
    }
    if (*p == '[') { *pp = p; return jparse_array(pp); }
    if (*p == '{') { *pp = p; return jparse_object(pp); }
    if (*p == 't' && strncmp(p, "true", 4) == 0) {
        v = calloc(1, sizeof(jval_t)); if (!v) return NULL;
        v->type = J_BOOL; v->boolean = 1; *pp = p + 4; return v;
    }
    if (*p == 'f' && strncmp(p, "false", 5) == 0) {
        v = calloc(1, sizeof(jval_t)); if (!v) return NULL;
        v->type = J_BOOL; v->boolean = 0; *pp = p + 5; return v;
    }
    if (*p == 'n' && strncmp(p, "null", 4) == 0) {
        v = calloc(1, sizeof(jval_t)); if (!v) return NULL;
        v->type = J_NULL; *pp = p + 4; return v;
    }
    /* number */
    {
        const char *q = p;
        if (*q == '-') q++;
        int digits = 0;
        while (*q >= '0' && *q <= '9') { q++; digits++; }
        if (*q == '.') { q++; while (*q >= '0' && *q <= '9') q++; }
        if (*q == 'e' || *q == 'E') {
            q++;
            if (*q == '+' || *q == '-') q++;
            while (*q >= '0' && *q <= '9') q++;
        }
        if (digits == 0) return NULL;
        char tmp[64];
        size_t n = (size_t)(q - p);
        if (n >= sizeof(tmp)) return NULL;
        memcpy(tmp, p, n); tmp[n] = 0;
        v = calloc(1, sizeof(jval_t)); if (!v) return NULL;
        v->type = J_NUM; v->num = atof(tmp); *pp = q; return v;
    }
    return NULL;
}

jval_t *json_parse(const char *text) {
    const char *p = jws(text);
    jval_t *v = jparse_value(&p);
    if (!v) return NULL;
    p = jws(p);
    if (*p) { json_free(v); return NULL; }   /* trailing junk -> invalid */
    return v;
}

int json_valid(const char *text) {
    jval_t *v = json_parse(text);
    if (!v) return 0;
    json_free(v);
    return 1;
}

const jval_t *json_obj_get(const jval_t *obj, const char *key) {
    if (!obj || obj->type != J_OBJ) return NULL;
    for (size_t i = 0; i < obj->n; i++)
        if (strcmp(obj->keys[i], key) == 0) return &obj->items[i];
    return NULL;
}

const jval_t *json_arr_get(const jval_t *arr, size_t i) {
    if (!arr || arr->type != J_ARR || i >= arr->n) return NULL;
    return &arr->items[i];
}

const char *json_str(const jval_t *v) {
    /* string nodes own their NUL-terminated copy — return it directly */
    if (!v || v->type != J_STR) return "";
    return v->s;
}

double json_num(const jval_t *v) {
    if (!v || v->type != J_NUM) return 0.0;
    return v->num;
}

int json_bool(const jval_t *v) {
    if (!v || v->type != J_BOOL) return 0;
    return v->boolean;
}

/* free a node's children/resources (NOT the node itself — nodes may be
   interior slots of a parent's items array) */
static void json_free_children(jval_t *v) {
    if (!v) return;
    if (v->type == J_OBJ) {
        for (size_t i = 0; i < v->n; i++) free((void *)v->keys[i]);
        free(v->keys);
    }
    if (v->type == J_ARR || v->type == J_OBJ) {
        for (size_t i = 0; i < v->n; i++) json_free_children(&v->items[i]);
        free(v->items);
    } else if (v->type == J_STR) {
        free((void *)v->s);   /* owned NUL-terminated copy */
    }
}

/* free a parsed tree: children resources + the top node */
void json_free(jval_t *v) {
    if (!v) return;
    json_free_children(v);
    free(v);
}

void json_escape(const char *in, char *out, size_t outsz) {
    size_t o = 0;
    for (const unsigned char *p = (const unsigned char *)in; *p && o + 7 < outsz; p++) {
        unsigned char c = *p;
        if (c == '"' || c == '\\') { out[o++] = '\\'; out[o++] = (char)c; }
        else if (c == '\n') { out[o++] = '\\'; out[o++] = 'n'; }
        else if (c == '\r') { out[o++] = '\\'; out[o++] = 'r'; }
        else if (c == '\t') { out[o++] = '\\'; out[o++] = 't'; }
        else if (c < 0x20) { snprintf(out + o, outsz - o, "\\u%04x", c); o += 6; }
        else out[o++] = (char)c;
    }
    out[o] = 0;
}

void json_quote_escape(const char *in, char *out, size_t outsz) {
    size_t o = 0;
    if (o + 1 < outsz) out[o++] = '"';
    for (const unsigned char *p = (const unsigned char *)in; *p && o + 7 < outsz; p++) {
        unsigned char c = *p;
        if (c == '"' || c == '\\') { out[o++] = '\\'; out[o++] = (char)c; }
        else if (c == '\n') { out[o++] = '\\'; out[o++] = 'n'; }
        else if (c == '\r') { out[o++] = '\\'; out[o++] = 'r'; }
        else if (c == '\t') { out[o++] = '\\'; out[o++] = 't'; }
        else if (c < 0x20) { snprintf(out + o, outsz - o, "\\u%04x", c); o += 6; }
        else out[o++] = (char)c;
    }
    if (o + 1 < outsz) out[o++] = '"';
    out[o] = 0;
}
