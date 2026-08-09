/* grammar.h — tool-call grammar validation (llama.cpp style).
 * Ported from llama.cpp's grammar sampler: every candidate token is
 * validated char-by-char against the model's native tool-call grammar
 * before sampling, so the model CANNOT emit a malformed call (nested
 * tags, stray </think>, truncated values, out-of-order tags).
 */
#ifndef GRAMMAR_H
#define GRAMMAR_H

#include "qma.h"

/* 1 iff token id can extend the tool-call grammar given the current call
   text (ans[0..ans_len) must contain an OPEN <tool_call> block). */
int grammar_tool_token_ok(qma_t *m, const char *ans, size_t ans_len, int id);

#endif /* GRAMMAR_H */
