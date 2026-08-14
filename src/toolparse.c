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

/* Find the close for the value at vstart: the FIRST close after which the
 * value round-trips to valid JSON. Accepted closes:
 *   </parameter>             — the canonical tag from the model's template
 *   </KEY>                  — HTML-style close the model improvises
 *                             (e.g. <parameter=start_line> 120</start_line>)
 *   </arg...>               — the "arguments" family (</arg_value>, </args>)
 * Any other '</' sequence does NOT close a value — so a bash command like
 * 'cat < /dev/null' keeps its value intact. Returns the close position and
 * its tag length, or NULL when none (truncated value). */
static const char *value_close(const char *vstart, const char *end,
                               const char *key, size_t keylen, size_t *tlen) {
    const char *scan = vstart;
    for (;;) {
        /* canonical </parameter> */
        const char *c = strstr(scan, "</parameter>");
        if (c && end && c > end) c = NULL;
        /* improvised </KEY> / </arg...> */
        const char *g = NULL; size_t gl = 0;
        for (const char *s = scan; s + 2 <= end; s++) {
            if (s[0] == '<' && s[1] == '/') {
                const char *w = s + 2, *w2 = w;
                while (w2 < end && ((*w2 >= 'a' && *w2 <= 'z') ||
                                    (*w2 >= 'A' && *w2 <= 'Z') ||
                                    (*w2 >= '0' && *w2 <= '9') || *w2 == '_')) w2++;
                if (w2 < end && *w2 == '>' && w2 > w) {
                    size_t wl = (size_t)(w2 - w);
                    if ((keylen && wl == keylen && strncasecmp(w, key, keylen) == 0) ||
                        (wl >= 3 && strncasecmp(w, "arg", 3) == 0)) {
                        g = s; gl = (size_t)(w2 - s) + 1;
                        break;
                    }
                }
                s = w2;   /* skip past this non-close </word> */
            }
        }
        const char *best = NULL; size_t bt = 0;
        if (c && (!g || c <= g)) { best = c; bt = strlen("</parameter>"); }
        else if (g) { best = g; bt = gl; }
        if (!best || (end && best > end)) return NULL;
        size_t vl = (size_t)(best - vstart);
        while (vl > 0 && (vstart[vl-1]=='\n'||vstart[vl-1]=='\r'||
                          vstart[vl-1]==' '||vstart[vl-1]=='\t')) vl--;
        if (val_roundtrips_valid(vstart, vl)) { *tlen = bt; return best; }
        scan = best + 1;
    }
}

int parse_one_tool_call(const char *text, const char **ppos, tool_call_t *tc) {
    const char *p = *ppos;
    /* block start: the <tool_call> opener if it precedes a <function=, else
       a bare <function= (the model without a grammar mask frequently drops
       the opener). */
    const char *opener = strstr(p, "<tool_call>");
    const char *fn = strstr(p, "<function=");
    if (fn && opener && opener < fn)
        fn = strstr(opener + strlen("<tool_call>"), "<function=");
    /* block end: first of </function> or </tool_call> after the block start */
    const char *blk = fn ? fn : (opener ? opener : p);
    const char *fe = strstr(blk, "</function>");
    const char *tce = strstr(blk, "</tool_call>");
    const char *end;
    if (fe && (!tce || fe < tce)) end = fe;
    else if (tce) end = tce;
    else return 0;                          /* still open: wait for more tokens */
    if (fe && end == fe) {
        *ppos = fe + strlen("</function>");
        /* consume trailing whitespace + an adjacent </tool_call> close */
        const char *w = *ppos;
        while (*w == '\n' || *w == '\r' || *w == ' ' || *w == '\t') w++;
        if (strncmp(w, "</tool_call>", strlen("</tool_call>")) == 0)
            *ppos = w + strlen("</tool_call>");
    } else {
        *ppos = tce + strlen("</tool_call>");
    }
    if (!fn) return -1;                     /* block but no function: malformed */
    tc->name[0] = 0;
    tc->args[0] = 0;
    const char *fn_end = strchr(fn + strlen("<function="), '>');
    if (!fn_end || fn_end >= end) return -1;
    size_t nlen = (size_t)(fn_end - (fn + strlen("<function=")));
    if (nlen >= sizeof(tc->name)) nlen = sizeof(tc->name) - 1;
    memcpy(tc->name, fn + strlen("<function="), nlen);
    tc->name[nlen] = 0;
    if (tc->name[0] == 0) return -1;
    /* parameters: <parameter=KEY>VALUE</CLOSE>; the value may sit on the
       same line as its tag (inline) or on following lines */
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
        while (*vstart == '\n' || *vstart == '\r' ||
               *vstart == ' ' || *vstart == '\t') vstart++;
        size_t vtlen = 0;
        const char *vclose = value_close(vstart, end, keys[na], klen, &vtlen);
        if (!vclose) { bad = 1; break; }    /* truncated value: malformed */
        size_t vlen = (size_t)(vclose - vstart);
        while (vlen > 0 && (vstart[vlen-1]=='\n'||vstart[vlen-1]=='\r'||
                            vstart[vlen-1]==' '||vstart[vlen-1]=='\t')) vlen--;
        if (vlen >= 16384) vlen = 16383;
        memcpy(vals[na], vstart, vlen);
        vals[na][vlen] = 0;
        na++;
        q = vclose + vtlen;
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
