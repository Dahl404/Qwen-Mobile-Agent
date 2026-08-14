/* toolparse.h — parsing the model's native <tool_call> XML into OpenAI-style
 * tool calls, with JSON validation. Ported from the engine's server.c (the
 * tested fixes): a parameter only closes at the first </parameter> after
 * which its value round-trips to valid JSON, so truncated/malformed calls
 * are never shipped — parse_one_tool_call returns -1 and the caller drops
 * them. There is NO grammar mask on generation (ma3 scope decision): calls
 * are constrained by policy instead — one complete call per turn, then the
 * engine force-ends the turn and folds the result back.
 */
#ifndef TOOLPARSE_H
#define TOOLPARSE_H

#include "qma.h"

#define MAX_TOOL_CALLS 8

typedef struct {
    char name[256];
    char args[16384];   /* JSON-encoded arguments string */
} tool_call_t;

/* Parse ONE complete <tool_call> block. *ppos must point at '<tool_call>';
 * advanced past '</tool_call>' on return. Returns:
 *    1  block complete AND valid (name + args parse as JSON)
 *    0  block still open
 *   -1  complete but malformed (dropped)
 */
int parse_one_tool_call(const char *text, const char **ppos, tool_call_t *tc);

/* length of s[0..len) that forms complete UTF-8 characters */
size_t u8_safe_len(const char *s, size_t len);

#endif /* TOOLPARSE_H */
