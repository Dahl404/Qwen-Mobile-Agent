/* grammar.c — char-level DFA for the model's native tool-call format.
 *
 * The grammar (from the model's chat template):
 *   <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>...
 *   \n</function>\n</tool_call>
 * with VALUE = a JSON value (string/number/bool/null/array/object) or bare
 * text terminated by the closing tag.
 *
 * Every candidate token is simulated through the DFA one char at a time;
 * if any char is not a legal continuation, the token is rejected before
 * sampling (llama.cpp's grammar-sampler approach — the model physically
 * cannot produce a malformed call). JSON values use a real incremental
 * recognizer (string/escape, number grammar, literals, and a structure
 * stack for arrays/objects), so truncated or nested-garbage values are
 * impossible while the grammar is active.
 */
#include <string.h>
#include "qma.h"
#include "grammar.h"

/* ---- structural states ---- */enum {
    ST_CALL_OPEN_TAG,      /* consuming "<tool_call>" */
    ST_CALL_OPEN_END,      /* ws, then '<' */
    ST_FUNCTION_LT,        /* saw '<': expect 'f' of "<function=" */
    ST_FUNCTION_TAG,       /* consuming "<function=" */
    ST_NAME,               /* tool name chars until '>' */
    ST_AFTER_FUNCTION,     /* ws, or '<' */
    ST_AFTER_FUNCTION_LT,  /* saw '<' after </function>-side: 'p' or '/' */
    ST_PARAM_TAG,          /* consuming "parameter=" (after the '<') */
    ST_KEY,                /* key chars until '>' */
    ST_AFTER_KEY,          /* ws, then the value */
    ST_VALUE,              /* parameter value (JSON or bare text) */
    ST_VALUE_CLOSE_TAG,    /* consuming "/parameter>" (after the '<') */
    ST_AFTER_PARAM,        /* ws, or '<' */
    ST_AFTER_PARAM_LT,     /* saw '<' after a param closed */
    ST_FN_CLOSE_TAG,       /* consuming "/function>" (after the '<') */
    ST_CALL_CLOSE_LT,      /* expect '<' (start of </tool_call>) */
    ST_CALL_CLOSE_TAG,     /* consuming "tool_call>" (after the '<') */
    ST_DONE                /* the call is complete */
};

/* ---- JSON value sub-states ---- */
enum {
    VJ_UNKNOWN,            /* no value char yet (leading ws) */
    VJ_STRING,             /* JSON string (in_string/esc) */
    VJ_NUMBER,             /* JSON number (num_state) */
    VJ_LITERAL,            /* true / false / null (lit) */
    VJ_STRUCT,             /* array/object (jstk) */
    VJ_TEXT                /* bare text value */
};

/* number grammar positions */
enum { NUM_START, NUM_INT, NUM_DOT, NUM_FRAC, NUM_EXP_MARK, NUM_EXP_SIGN, NUM_EXP_DIG };

/* structure stack entries */
enum { JE_VALUE_ARR, JE_VALUE_OBJ, JE_KEY, JE_COLON, JE_SEP_ARR, JE_SEP_OBJ };

static int ws_char(char c) { return c==' '||c=='\t'||c=='\n'||c=='\r'; }
static int name_char(char c) {
    return (c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_';
}
static int digit_char(char c) { return c>='0'&&c<='9'; }

static const char *TAG_CALL_OPEN  = "<tool_call>";
static const char *TAG_FUNCTION   = "<function=";
static const char *TAG_FN_CLOSE   = "</function>";
static const char *TAG_PARAM      = "<parameter=";
static const char *TAG_PARAM_CLOSE= "</parameter>";
static const char *TAG_CALL_CLOSE = "</tool_call>";

/* consume one char of a tag; advances tag_pos on match */
static int tag_char(gpos_t *g, const char *tag, char c) {
    if (tag[g->tag_pos] == c) { g->tag_pos++; return 1; }
    return 0;
}

/* pop the structure stack; returns 0 on underflow (bad) */
static int jpop(gpos_t *g) {
    if (g->jd <= 0) return 0;
    g->jd--;
    if (g->jd == 0) g->val_done = 1;   /* top-level array/object closed */
    return 1;
}

static int gchar_number(gpos_t *g, char c);

/* called when a nested JSON value completes: advance the parent container */
static int jvalue_done(gpos_t *g) {
    if (g->jd == 0) { g->val_done = 1; return 1; }
    int top = g->jstk[g->jd - 1];
    if (top == JE_VALUE_ARR)      g->jstk[g->jd - 1] = JE_SEP_ARR;
    else if (top == JE_VALUE_OBJ) g->jstk[g->jd - 1] = JE_SEP_OBJ;
    else return 0;                /* a value ended in a bad expectation */
    return 1;
}

/* one JSON char inside VJ_STRUCT (array/object). Returns 1 if accepted. */
static int jstruct_char(gpos_t *g, char c) {
    if (g->in_string) {
        if (g->esc) { g->esc = 0; return 1; }
        if (c == '\\') { g->esc = 1; return 1; }
        if (c == '"') { g->in_string = 0; return 1; }  /* close (caller handles kind) */
        return 1;
    }
    if (g->jd <= 0) return 0;
    int top = g->jstk[g->jd - 1];
    switch (top) {
    case JE_VALUE_ARR:
    case JE_VALUE_OBJ:
        if (ws_char(c)) return 1;
        if (c == '"') { g->in_string = 1; return 1; }   /* string value */
        if (c == '[') { if (g->jd >= JSTK_CAP) return 0; g->jstk[g->jd++] = JE_VALUE_ARR; return 1; }
        if (c == '{') { if (g->jd >= JSTK_CAP) return 0; g->jstk[g->jd++] = JE_KEY; return 1; }
        if (digit_char(c) || c == '-') {
            g->vj = VJ_NUMBER; g->num_state = NUM_START;
            /* re-run the char through the number machine */
            return gchar_number(g, c);
        }
        if (c == 't') { g->vj = VJ_LITERAL; g->lit = "true"; g->lit_pos = 1; return 1; }
        if (c == 'f') { g->vj = VJ_LITERAL; g->lit = "false"; g->lit_pos = 1; return 1; }
        if (c == 'n') { g->vj = VJ_LITERAL; g->lit = "null"; g->lit_pos = 1; return 1; }
        return 0;
    case JE_KEY:
        if (ws_char(c)) return 1;
        if (c == '}') { return jpop(g); }               /* empty object */
        if (c == '"') { g->in_string = 1; return 1; }   /* key string */
        return 0;
    case JE_COLON:
        if (ws_char(c)) return 1;
        if (c == ':') { g->jstk[g->jd - 1] = JE_VALUE_OBJ; return 1; }
        return 0;
    case JE_SEP_ARR:
        if (ws_char(c)) return 1;
        if (c == ',') { g->jstk[g->jd - 1] = JE_VALUE_ARR; return 1; }
        if (c == ']') return jpop(g);
        return 0;
    case JE_SEP_OBJ:
        if (ws_char(c)) return 1;
        if (c == ',') { g->jstk[g->jd - 1] = JE_KEY; return 1; }
        if (c == '}') return jpop(g);
        return 0;
    }
    return 0;
}

/* one char of the parameter value. Returns 1 if accepted. */
static int gvalue_char(gpos_t *g, char c) {
    if (g->vj == VJ_UNKNOWN) {
        if (ws_char(c)) return 1;                 /* leading whitespace */
        if (c == '"') { g->vj = VJ_STRING; g->in_string = 1; g->esc = 0; return 1; }
        if (c == '[') { g->vj = VJ_STRUCT; g->jd = 1; g->jstk[0] = JE_VALUE_ARR; return 1; }
        if (c == '{') { g->vj = VJ_STRUCT; g->jd = 1; g->jstk[0] = JE_KEY; return 1; }
        if (digit_char(c) || c == '-') { g->vj = VJ_NUMBER; g->num_state = NUM_START; return gchar_number(g, c); }
        if (c == 't') { g->vj = VJ_LITERAL; g->lit = "true";  g->lit_pos = 1; return 1; }
        if (c == 'f') { g->vj = VJ_LITERAL; g->lit = "false"; g->lit_pos = 1; return 1; }
        if (c == 'n') { g->vj = VJ_LITERAL; g->lit = "null";  g->lit_pos = 1; return 1; }
        /* bare text value */
        g->vj = VJ_TEXT;
        if (c == '<') return 0;                   /* '<' starts the close tag */
        g->text_seen = 1;
        return 1;
    }
    switch (g->vj) {
    case VJ_STRING:
        if (g->esc) { g->esc = 0; return 1; }
        if (c == '\\') { g->esc = 1; return 1; }
        if (c == '"') {
            g->in_string = 0;
            if (g->jd == 0) { g->val_done = 1; return 1; }   /* top-level string */
            /* nested string: a struct value or key completed */
            int top = g->jstk[g->jd - 1];
            if (top == JE_KEY)      { g->jstk[g->jd - 1] = JE_COLON; return 1; }
            if (top == JE_VALUE_ARR || top == JE_VALUE_OBJ) return jvalue_done(g);
            return 0;
        }
        return 1;   /* any other char is valid string content */
    case VJ_NUMBER:
        return gchar_number(g, c);
    case VJ_LITERAL:
        if (g->lit[g->lit_pos] == c) {
            g->lit_pos++;
            if (g->lit[g->lit_pos] == 0) {        /* literal complete */
                if (g->jd == 0) { g->val_done = 1; return 1; }
                return jvalue_done(g);
            }
            return 1;
        }
        return 0;
    case VJ_STRUCT:
        if (g->in_string) {
            /* a string inside a struct */
            if (g->esc) { g->esc = 0; return 1; }
            if (c == '\\') { g->esc = 1; return 1; }
            if (c == '"') {
                g->in_string = 0;
                int top = g->jstk[g->jd - 1];
                if (top == JE_KEY)      { g->jstk[g->jd - 1] = JE_COLON; return 1; }
                if (top == JE_VALUE_ARR || top == JE_VALUE_OBJ) return jvalue_done(g);
                return 0;
            }
            return 1;
        }
        return jstruct_char(g, c);
    case VJ_TEXT:
        if (c == '<') {                       /* '<' starts </parameter> */
            g->st = ST_VALUE_CLOSE_TAG;
            g->tag_pos = 1;                   /* we consumed '<' */
            return 1;
        }
        g->text_seen = 1;
        return 1;
    }
    return 0;
}

static int gchar_number(gpos_t *g, char c) {
    switch (g->num_state) {
    case NUM_START:
        if (c == '-') { g->num_state = NUM_INT; return 1; }
        if (digit_char(c)) { g->num_state = NUM_INT; return 1; }
        return 0;
    case NUM_INT:
        if (digit_char(c)) return 1;
        if (c == '.') { g->num_state = NUM_DOT; return 1; }
        if (c == 'e' || c == 'E') { g->num_state = NUM_EXP_MARK; return 1; }
        break;   /* number complete — fall through to the completion handling */
    case NUM_DOT:
        if (digit_char(c)) { g->num_state = NUM_FRAC; return 1; }
        return 0;
    case NUM_FRAC:
        if (digit_char(c)) return 1;
        if (c == 'e' || c == 'E') { g->num_state = NUM_EXP_MARK; return 1; }
        break;
    case NUM_EXP_MARK:
        if (digit_char(c)) { g->num_state = NUM_EXP_DIG; return 1; }
        if (c == '+' || c == '-') { g->num_state = NUM_EXP_SIGN; return 1; }
        return 0;
    case NUM_EXP_SIGN:
        if (digit_char(c)) { g->num_state = NUM_EXP_DIG; return 1; }
        return 0;
    case NUM_EXP_DIG:
        if (digit_char(c)) return 1;
        break;
    }
    /* the number is complete: the next char belongs to the parent */
    if (g->jd == 0) {
        if (ws_char(c)) { g->val_done = 1; return 1; }
        if (c == '<') { g->val_done = 1; g->st = ST_VALUE_CLOSE_TAG; g->tag_pos = 1; return 1; }
        return 0;
    }
    {
        int top = g->jstk[g->jd - 1];
        if (top == JE_VALUE_ARR || top == JE_VALUE_OBJ) {
            if (!jvalue_done(g)) return 0;
            /* re-run the char in the new expectation state */
            if (ws_char(c)) return 1;
            if (c == ',') {
                int t2 = g->jstk[g->jd - 1];
                if (t2 == JE_SEP_ARR) { g->jstk[g->jd - 1] = JE_VALUE_ARR; return 1; }
                if (t2 == JE_SEP_OBJ) { g->jstk[g->jd - 1] = JE_KEY; return 1; }
                return 0;
            }
            if (c == ']') return (g->jstk[g->jd - 1] == JE_SEP_ARR) ? jpop(g) : 0;
            if (c == '}') return (g->jstk[g->jd - 1] == JE_SEP_OBJ) ? jpop(g) : 0;
            return 0;
        }
    }
    return 0;
}

/* the main structural transition. Returns 1 if c is accepted. */
static int gchar(gpos_t *g, char c) {
    switch (g->st) {
    case ST_CALL_OPEN_TAG:
        if (!tag_char(g, TAG_CALL_OPEN, c)) return 0;
        if (g->tag_pos == (int)strlen(TAG_CALL_OPEN)) { g->st = ST_CALL_OPEN_END; g->tag_pos = 0; }
        return 1;
    case ST_CALL_OPEN_END:
        if (ws_char(c)) return 1;
        if (c == '<') { g->st = ST_FUNCTION_LT; return 1; }
        return 0;
    case ST_FUNCTION_LT:
        if (c == 'f') { g->st = ST_FUNCTION_TAG; g->tag_pos = 2; return 1; }
        return 0;
    case ST_FUNCTION_TAG:
        if (!tag_char(g, TAG_FUNCTION, c)) return 0;
        if (g->tag_pos == (int)strlen(TAG_FUNCTION)) { g->st = ST_NAME; g->tag_pos = 0; }
        return 1;
    case ST_NAME:
        if (name_char(c)) return 1;
        if (c == '>') { g->st = ST_AFTER_FUNCTION; return 1; }
        return 0;
    case ST_AFTER_FUNCTION:
    case ST_AFTER_PARAM:
        if (ws_char(c)) return 1;
        if (c == '<') { g->st = (g->st == ST_AFTER_FUNCTION) ? ST_AFTER_FUNCTION_LT : ST_AFTER_PARAM_LT; return 1; }
        return 0;
    case ST_AFTER_FUNCTION_LT:
    case ST_AFTER_PARAM_LT:
        if (c == 'p') { g->st = ST_PARAM_TAG; g->tag_pos = 2; return 1; }      /* "<parameter=" */
        if (c == '/') { g->st = ST_FN_CLOSE_TAG; g->tag_pos = 2; return 1; }   /* "</function>" */
        return 0;
    case ST_PARAM_TAG:
        if (!tag_char(g, TAG_PARAM, c)) return 0;
        if (g->tag_pos == (int)strlen(TAG_PARAM)) { g->st = ST_KEY; g->tag_pos = 0; }
        return 1;
    case ST_KEY:
        if (name_char(c)) return 1;
        if (c == '>') { g->st = ST_AFTER_KEY; return 1; }
        return 0;
    case ST_AFTER_KEY:
        if (ws_char(c)) return 1;
        g->st = ST_VALUE;
        return gvalue_char(g, c);
    case ST_VALUE:
        if (g->val_done) {
            if (ws_char(c)) return 1;
            if (c == '<') { g->st = ST_VALUE_CLOSE_TAG; g->tag_pos = 1; return 1; }
            return 0;
        }
        return gvalue_char(g, c);
    case ST_VALUE_CLOSE_TAG:
        if (!tag_char(g, TAG_PARAM_CLOSE, c)) return 0;
        if (g->tag_pos == (int)strlen(TAG_PARAM_CLOSE)) { g->st = ST_AFTER_PARAM; g->tag_pos = 0; }
        return 1;
    case ST_FN_CLOSE_TAG:
        if (!tag_char(g, TAG_FN_CLOSE, c)) return 0;
        if (g->tag_pos == (int)strlen(TAG_FN_CLOSE)) { g->st = ST_CALL_CLOSE_LT; g->tag_pos = 0; }
        return 1;
    case ST_CALL_CLOSE_LT:
        if (ws_char(c)) return 1;
        if (c == '<') { g->st = ST_CALL_CLOSE_TAG; g->tag_pos = 1; return 1; }
        return 0;
    case ST_CALL_CLOSE_TAG:
        if (!tag_char(g, TAG_CALL_CLOSE, c)) return 0;
        if (g->tag_pos == (int)strlen(TAG_CALL_CLOSE)) { g->st = ST_DONE; }
        return 1;
    case ST_DONE:
        return 0;   /* nothing may follow a completed call */
    }
    return 0;
}

/* replay the open call's text through the grammar. Returns 0 if the text is
   already invalid (shouldn't happen while the mask is active). */
static int grammar_replay(const char *text, size_t len, gpos_t *g) {
    memset(g, 0, sizeof(*g));
    g->st = ST_CALL_OPEN_TAG;
    for (size_t i = 0; i < len; i++) {
        if (!gchar(g, text[i])) return 0;
    }
    return 1;
}

/* find the open call in ans[0..ans_len): the LAST "<tool_call>" with no
   "</tool_call>" after it. Returns a pointer to it or NULL. */
static const char *grammar_open_call(const char *ans, size_t ans_len) {
    const char *e = ans + ans_len;
    const char *start = NULL, *p = ans;
    while ((p = strstr(p, "<tool_call>")) != NULL) {
        const char *close = strstr(p + strlen("<tool_call>"), "</tool_call>");
        if (!close || close >= e) { start = p; break; }
        p = close + strlen("</tool_call>");
    }
    return start;
}

int grammar_tool_open_state(qma_t *m, const char *ans, size_t ans_len,
                            gpos_t *out) {
    (void)m;
    const char *start = grammar_open_call(ans, ans_len);
    if (!start) return 0;   /* no open call: no constraint */
    if (!grammar_replay(start, (size_t)(ans + ans_len - start), out))
        return -1;          /* open call but its text is already invalid */
    return 1;
}

int grammar_tool_token_from_state(const gpos_t *st, qma_t *m, int id) {
    if (id < 0 || id >= m->n_vocab || !m->tok_text[id]) return 0;
    const char *piece = m->tok_text[id];
    if (piece[0] == 0) return 0;      /* byte-0 special token */
    gpos_t g = *st;   /* copy: lit points at a read-only string literal */
    for (const char *q = piece; *q; q++) {
        if (!gchar(&g, *q)) return 0;
    }
    return 1;
}

int grammar_tool_token_ok(qma_t *m, const char *ans, size_t ans_len, int id) {
    gpos_t g;
    int st = grammar_tool_open_state(m, ans, ans_len, &g);
    if (st == 0) return 1;              /* unconstrained */
    if (st < 0) return 0;               /* open but invalid: reject */
    return grammar_tool_token_from_state(&g, m, id);
}
