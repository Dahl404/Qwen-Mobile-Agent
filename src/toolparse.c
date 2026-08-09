/* toolparse.c — native <tool_call> XML parsing with JSON validation.
 * See toolparse.h. Ported from the engine's server.c with the robustness
 * fixes: param_close() only accepts a </parameter> when the value so far
 * round-trips to valid JSON (a truncated scalar never closes a parameter),
 * and the assembled args must pass json_valid() before a call is returned.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include "qma.h"
#include "json.h"
#include "toolparse.h"

#define MAX_TOOL_ARGS 64

/* length-bounded scalar lookahead: does s[0..len) look like a bare JSON
   scalar (number / bool / null / string / array / object)? */
static int looks_scalar_bounded(const char *s, size_t len) {
    size_t i = 0;
    while (i < len && (s[i]==' '||s[i]=='\t'||s[i]=='\n'||s[i]=='\r')) i++;
    if (i >= len) return 0;
    if (s[i] == '"' || s[i] == '{' || s[i] == '[') return 1;
    {
        const char *p = s + i;
        const char *e = s + len;
        if (p < e && *p == '-') p++;
        int digits = 0;
        while (p < e && *p >= '0' && *p <= '9') { p++; digits++; }
        if (p < e && *p == '.') { p++; while (p < e && *p >= '0' && *p <= '9') p++; }
        if (p < e && (*p=='e'||*p=='E')) { p++; if (p < e && (*p=='+'||*p=='-')) p++; while (p < e && *p>='0'&&*p<='9') p++; }
        if (digits > 0 && p == e) return 1;
    }
    if (len - i == 4 && strncasecmp(s+i, "true", 4) == 0) return 1;
    if (len - i == 5 && strncasecmp(s+i, "false", 5) == 0) return 1;
    if (len - i == 4 && (strncasecmp(s+i, "null", 4) == 0 ||
                         strncasecmp(s+i, "none", 4) == 0)) return 1;
    return 0;
}

/* A parameter value round-trips to valid JSON iff it is a bare JSON scalar
   (emitted bare) or arbitrary text (emitted quoted). */
/* The model's tool-call XML markers must NEVER appear inside a parameter
   value. When the model re-drafts mid-answer (stray </think>) or cascades
   nested calls, the garbage ends up in a value — accepting it as "text"
   shipped broken calls (observed: a ls call whose path was a whole nested
   <tool_call> block). Any of these markers makes the value invalid, so the
   parameter never closes there and the call is dropped instead. */
static const char *const XML_MARKERS[] = {
    "<tool_call>", "</tool_call>", "<function=", "</function>",
    "<parameter=", "</parameter>", "<think>", "</think>",
};
static int has_xml_marker(const char *s, size_t len) {
    for (int i = 0; i < 8; i++) {
        size_t ml = strlen(XML_MARKERS[i]);
        if (len >= ml) {
            for (size_t j = 0; j + ml <= len; j++)
                if (memcmp(s + j, XML_MARKERS[i], ml) == 0) return 1;
        }
    }
    return 0;
}

static int val_roundtrips_valid(const char *s, size_t len) {
    if (len == 0) return 1;
    if (has_xml_marker(s, len)) return 0;
    if (looks_scalar_bounded(s, len)) {
        /* Python-style bool literals (True/False/None) are normalized to
           JSON at build time — accept them here so the parameter closes */
        if ((len == 4 && (strncasecmp(s, "true", 4) == 0 ||
                          strncasecmp(s, "none", 4) == 0)) ||
            (len == 5 && strncasecmp(s, "false", 5) == 0))
            return 1;
        /* bounded json_valid: reuse the tree parser on a NUL-terminated copy */
        char tmp[16384];
        if (len >= sizeof(tmp)) len = sizeof(tmp) - 1;
        memcpy(tmp, s, len); tmp[len] = 0;
        return json_valid(tmp);
    }
    return 1;
}

/* Find the </parameter> close for the value at vstart: the FIRST close after
 * which the value round-trips to valid JSON. Returns NULL when none. */
static const char *param_close(const char *vstart, const char *end) {
    const char *scan = vstart;
    for (;;) {
        const char *c = strstr(scan, "</parameter>");
        if (!c || (end && c > end)) return NULL;
        size_t vl = (size_t)(c - vstart);
        while (vl > 0 && (vstart[vl-1]=='\n'||vstart[vl-1]=='\r'||
                          vstart[vl-1]==' '||vstart[vl-1]=='\t')) vl--;
        if (val_roundtrips_valid(vstart, vl)) return c;
        scan = c + strlen("</parameter>");
    }
}

int parse_one_tool_call(const char *text, const char **ppos, tool_call_t *tc) {
    const char *p = *ppos;
    const char *body = p + strlen("<tool_call>");
    const char *end = strstr(body, "</tool_call>");
    if (!end) return 0;                     /* still open */
    *ppos = end + strlen("</tool_call>");
    tc->name[0] = 0;
    tc->args[0] = 0;
    const char *fn = strstr(body, "<function=");
    if (!fn || fn >= end) return -1;
    const char *fn_end = strchr(fn + strlen("<function="), '>');
    if (!fn_end || fn_end >= end) return -1;
    size_t nlen = (size_t)(fn_end - (fn + strlen("<function=")));
    if (nlen >= sizeof(tc->name)) nlen = sizeof(tc->name) - 1;
    memcpy(tc->name, fn + strlen("<function="), nlen);
    tc->name[nlen] = 0;
    if (tc->name[0] == 0) return -1;
    /* parameters: <parameter=KEY>\nVALUE\n</parameter> */
    char (*keys)[256] = malloc((size_t)MAX_TOOL_ARGS * 256);
    char (*vals)[16384] = malloc((size_t)MAX_TOOL_ARGS * 16384);
    if (!keys || !vals) { free(keys); free(vals); return -1; }
    int na = 0, bad = 0;
    const char *q = fn_end + 1;
    while (na < MAX_TOOL_ARGS) {
        const char *pm = strstr(q, "<parameter=");
        if (!pm || pm >= end) break;
        const char *pm_end = strchr(pm + strlen("<parameter="), '>');
        if (!pm_end || pm_end >= end) break;
        size_t klen = (size_t)(pm_end - (pm + strlen("<parameter=")));
        if (klen >= 256) klen = 255;
        memcpy(keys[na], pm + strlen("<parameter="), klen);
        keys[na][klen] = 0;
        const char *vstart = pm_end + 1;
        while (*vstart == '\n' || *vstart == '\r') vstart++;
        const char *vclose = param_close(vstart, end);
        if (!vclose) { bad = 1; break; }    /* truncated value: malformed */
        size_t vlen = (size_t)(vclose - vstart);
        while (vlen > 0 && (vstart[vlen-1]=='\n'||vstart[vlen-1]=='\r'||
                            vstart[vlen-1]==' '||vstart[vlen-1]=='\t')) vlen--;
        if (vlen >= 16384) vlen = 16383;
        memcpy(vals[na], vstart, vlen);
        vals[na][vlen] = 0;
        na++;
        q = vclose + strlen("</parameter>");
    }
    if (bad) { free(keys); free(vals); return -1; }
    /* build JSON arguments {"k":v,...} */
    {
        size_t pos = 0, cap = sizeof(tc->args);
        int w = snprintf(tc->args, cap, "{");
        if (w > 0) pos = (size_t)w;
        for (int i = 0; i < na; i++) {
            char key_esc[1600], val_esc[131072];
            json_quote_escape(keys[i], key_esc, sizeof(key_esc));
            if (looks_scalar_bounded(vals[i], strlen(vals[i]))) {
                /* normalize Python-style bool literals to JSON */
                if (strcasecmp(vals[i], "true") == 0)
                    snprintf(val_esc, sizeof(val_esc), "true");
                else if (strcasecmp(vals[i], "false") == 0)
                    snprintf(val_esc, sizeof(val_esc), "false");
                else if (strcasecmp(vals[i], "null") == 0 ||
                         strcasecmp(vals[i], "none") == 0)
                    snprintf(val_esc, sizeof(val_esc), "null");
                else
                    snprintf(val_esc, sizeof(val_esc), "%s", vals[i]);
            } else
                json_quote_escape(vals[i], val_esc, sizeof(val_esc));
            w = snprintf(tc->args + pos, cap - pos, "%s%s:%s", i ? "," : "", key_esc, val_esc);
            if (w < 0 || (size_t)w >= cap - pos) break;
            pos += (size_t)w;
        }
        if (pos + 1 < cap) { tc->args[pos++] = '}'; tc->args[pos] = 0; }
    }
    free(keys); free(vals);
    if (!json_valid(tc->args)) return -1;   /* refuse malformed args */
    return 1;
}

int parse_tool_calls(const char *text, tool_call_t *calls, int max) {
    int n = 0;
    const char *p = text;
    while (n < max && (p = strstr(p, "<tool_call>")) != NULL) {
        int r = parse_one_tool_call(text, &p, &calls[n]);
        if (r == 0) break;
        if (r == 1) n++;
    }
    return n;
}

int tool_block_open(const char *ans, size_t ans_len) {
    const char *p = ans;
    const char *e = ans + ans_len;
    while (p < e) {
        const char *a = strstr(p, "<tool_call>");
        const char *b = strstr(p, "</tool_call>");
        if (!a && !b) break;
        if (!b || (a && a < b)) {
            const char *nb = strstr(a + strlen("<tool_call>"), "</tool_call>");
            if (!nb) {
                const char *fn = strstr(a, "<function=");
                if (fn) return 1;
                break;
            }
            p = a + strlen("<tool_call>");
        } else {
            p = b + strlen("</tool_call>");
        }
    }
    return 0;
}

/* Grammar-constrained sampling during an open tool call (the llama.cpp
   approach: mask illegal tokens at every step so the model CANNOT produce
   a malformed call, instead of parsing garbage after the fact).

   While a <tool_call> block is open:
   - control tokens (<|im_end|> etc) are forbidden (existing fix),
   - the single-token tags <think> </think> <tool_call> <tool_response>
     </tool_response> are forbidden — the observed re-draft/cascade
     failures were exactly these tokens appearing mid-call (</tool_call>
     stays allowed so the call can close),
   - inside an open parameter value, '<' and '>' tokens are forbidden
     while the value is an INCOMPLETE JSON structure (so a nested
     <function=/<parameter= can never appear, and the value must finish
     before </parameter> is reachable). Bare-text values and closed JSON
     keep '<' so the close tag stays reachable — no deadlock. */

/* value constraint: 0 = '<' allowed, 2 = '<'/'>' forbidden */
static int value_lt_state(const char *ans, size_t ans_len) {
    const char *e = ans + ans_len;
    /* the open block: last "<tool_call>" with no "</tool_call>" after */
    const char *tc = NULL, *p = ans;
    while ((p = strstr(p, "<tool_call>")) != NULL) {
        const char *close = strstr(p + strlen("<tool_call>"), "</tool_call>");
        if (!close || close >= e) { tc = p; break; }
        p = close + strlen("</tool_call>");
    }
    if (!tc) return 0;
    /* the last "<parameter=" in the block with no "</parameter>" after */
    const char *pm = NULL, *q = tc;
    while ((q = strstr(q, "<parameter=")) != NULL && q < e) {
        const char *pc = strstr(q + strlen("<parameter="), "</parameter>");
        if (!pc || pc >= e) { pm = q; break; }
        q = pc + strlen("</parameter>");
    }
    if (!pm) return 0;
    const char *vstart = strchr(pm + strlen("<parameter="), '>');
    if (!vstart || vstart >= e) return 0;
    vstart++;
    while (vstart < e && (*vstart == '\n' || *vstart == '\r')) vstart++;
    const char *vend = strstr(vstart, "</parameter>");
    if (vend && vend < e) return 0;   /* value already closed */
    size_t vlen = (size_t)(e - vstart);
    size_t i = 0;
    while (i < vlen && (vstart[i]==' '||vstart[i]=='\t')) i++;
    if (i >= vlen) return 0;          /* empty value */
    char c = vstart[i];
    if (c == '"') {
        /* in-string (odd unescaped quotes): '<' is legal string content */
        int in_str = 0, esc = 0;
        for (size_t j = 0; j < vlen; j++) {
            if (esc) { esc = 0; continue; }
            if (vstart[j] == '\\') { esc = 1; continue; }
            if (vstart[j] == '"') in_str = !in_str;
        }
        if (in_str) return 0;
        /* closed: complete JSON string -> '<' allowed (close reachable) */
        char tmp[16384];
        size_t n = vlen; if (n >= sizeof(tmp)) n = sizeof(tmp) - 1;
        memcpy(tmp, vstart, n); tmp[n] = 0;
        return json_valid(tmp) ? 0 : 2;
    }
    if (c == '[' || c == '{' || c == '-' || (c >= '0' && c <= '9')) {
        char tmp[16384];
        size_t n = vlen; if (n >= sizeof(tmp)) n = sizeof(tmp) - 1;
        memcpy(tmp, vstart, n); tmp[n] = 0;
        return json_valid(tmp) ? 0 : 2;
    }
    return 0;   /* bare text: '<' allowed so the close tag stays reachable */
}

void mask_tool_grammar(const qma_t *m, float *logits, const char *ans, size_t ans_len) {
    int vs = value_lt_state(ans, ans_len);
    for (int i = 0; i < m->n_vocab; i++) {
        if (m->tok_type[i] & 4) { logits[i] = -1e30f; continue; }   /* control */
        if (m->tok_type[i] & 8) {
            /* think/tool tags are single tokens — forbidden mid-call,
               except </tool_call> which closes the block */
            const char *t = m->tok_text[i];
            if (t && (strcmp(t, "<think>") == 0 || strcmp(t, "</think>") == 0 ||
                      strcmp(t, "<tool_call>") == 0 ||
                      strcmp(t, "<tool_response>") == 0 ||
                      strcmp(t, "</tool_response>") == 0))
                logits[i] = -1e30f;
            continue;
        }
        if (vs == 2 && m->tok_text[i] &&
            (strchr(m->tok_text[i], '<') || strchr(m->tok_text[i], '>')))
            logits[i] = -1e30f;
    }
}

size_t u8_safe_len(const char *s, size_t len) {
    size_t hold = 0;
    while (hold < 4 && len > hold) {
        unsigned char c = (unsigned char)s[len - 1 - hold];
        if (c < 0x80) break;
        if ((c & 0xC0) == 0xC0) { hold++; break; }
        hold++;
    }
    return len - hold;
}
