/* grammar.h — tool-call grammar validation (llama.cpp style).
 * Ported from llama.cpp's grammar sampler: every candidate token is
 * validated char-by-char against the model's native tool-call grammar
 * before sampling, so the model CANNOT emit a malformed call (nested
 * tags, stray </think>, truncated values, out-of-order tags).
 */
#ifndef GRAMMAR_H
#define GRAMMAR_H

#include "qma.h"

/* DFA state for an in-progress tool call. grammar_tool_open_state() builds
   one from the current open <tool_call> text; grammar_tool_token_from_state()
   simulates a candidate token against it. Callers treat it as opaque — it is
   only passed between the two functions, and is safe to copy by value. */
#define JSTK_CAP 16

typedef struct {
    int st;                /* structural state */
    int tag_pos;           /* char index into the active tag */
    /* value state */
    int vj;                /* VJ_* */
    int in_string, esc;    /* inside a JSON string */
    int num_state;
    const char *lit;       /* current literal ("true"/"false"/"null") */
    int lit_pos;
    int jstk[JSTK_CAP];    /* structure stack */
    int jd;                /* depth */
    int val_done;          /* top-level value completed */
    int text_seen;         /* bare-text value has content */
} gpos_t;

/* Build the DFA state for the currently open <tool_call> in ans[0..ans_len).
   Returns:
      1  an open call exists and `out` holds a valid state — mask candidates
         individually via grammar_tool_token_from_state()
      0  no open call — no constraint applies
     -1  an open call exists but its text is already invalid — reject ALL
         candidates
   Build ONCE per decode step, then test each candidate against the state:
   the old grammar_tool_token_ok() replayed the whole call per candidate
   (up to 40 replays per token while a call is being written). */
int grammar_tool_open_state(qma_t *m, const char *ans, size_t ans_len,
                            gpos_t *out);

/* 1 iff token id can legally extend the call from a state built by
   grammar_tool_open_state(). */
int grammar_tool_token_from_state(const gpos_t *st, qma_t *m, int id);

/* Convenience wrapper: open-state + token-from-state in one call. */
int grammar_tool_token_ok(qma_t *m, const char *ans, size_t ans_len, int id);

#endif /* GRAMMAR_H */
