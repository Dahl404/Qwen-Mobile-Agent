/* test_toolparse.c — parser contract for the one-call-per-turn policy.
 *
 * parse_one_tool_call must:
 *   1  return 1 for a complete, valid <tool_call> block (name + JSON args)
 *   0  return 0 for an OPEN block and NOT advance *ppos (the caller must
 *      break on 0 — looping would rescan the same block forever)
 *  -1  return -1 for a complete but malformed block (never executed)
 *
 * Build: make tests/test_toolparse  (or clang -I src -o tests/test_toolparse
 *        tests/test_toolparse.c src/toolparse.c src/json.c -lm)
 */
#include <stdio.h>
#include <string.h>
#include "toolparse.h"

static int n_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s (line %d)\n", msg, __LINE__); n_fail++; } \
    else printf("ok   %s\n", msg); \
} while (0)

int main(void) {
    tool_call_t tc;

    /* 1. valid single call, bare scalar argument */
    {
        const char *txt = "prefix <tool_call>\n<function=ls>\n<parameter=path>\n/internal/src\n</parameter>\n</function>\n</tool_call> suffix";
        const char *p = strstr(txt, "<tool_call>");
        int r = parse_one_tool_call(txt, &p, &tc);
        CHECK(r == 1, "valid call -> 1");
        CHECK(strcmp(tc.name, "ls") == 0, "name == ls");
        CHECK(strcmp(tc.args, "{\"path\":\"/internal/src\"}") == 0, "args JSON bare scalar");
        CHECK(p - txt == (long)(strstr(txt, "</tool_call>") + strlen("</tool_call>") - txt), "ppos advanced past close");
    }

    /* 2. valid call, quoted multi-line argument */
    {
        const char *txt = "<tool_call>\n<function=bash>\n<parameter=command>\n\"ls -la\"\n</parameter>\n</function>\n</tool_call>";
        const char *p = strstr(txt, "<tool_call>");
        int r = parse_one_tool_call(txt, &p, &tc);
        CHECK(r == 1, "valid quoted arg -> 1");
        CHECK(strcmp(tc.args, "{\"command\":\"ls -la\"}") == 0, "args JSON quoted string");
    }

    /* 3. OPEN block: no </tool_call> -> 0 AND *ppos must NOT advance */
    {
        const char *txt = "<tool_call>\n<function=ls>\n<parameter=path>\n/internal";
        const char *p = strstr(txt, "<tool_call>");
        const char *p0 = p;
        int r = parse_one_tool_call(txt, &p, &tc);
        CHECK(r == 0, "open block -> 0");
        CHECK(p == p0, "open block does not advance ppos");
    }

    /* 4. complete but malformed: no <function= -> -1, ppos advanced */
    {
        const char *txt = "<tool_call>\n<parameter=path>\n/internal\n</parameter>\n</tool_call>";
        const char *p = strstr(txt, "<tool_call>");
        const char *p0 = p;
        int r = parse_one_tool_call(txt, &p, &tc);
        CHECK(r == -1, "no function name -> -1");
        CHECK(p != p0, "malformed block advances ppos");
    }

    /* 5. complete but malformed: empty function name -> -1 */
    {
        const char *txt = "<tool_call>\n<function=>\n</function>\n</tool_call>";
        const char *p = strstr(txt, "<tool_call>");
        int r = parse_one_tool_call(txt, &p, &tc);
        CHECK(r == -1, "empty function name -> -1");
    }

    /* 6. complete but malformed: a value that LOOKS like a scalar but does
       not parse as JSON (truncated JSON) never closes the parameter -> -1 */
    {
        const char *txt = "<tool_call>\n<function=ls>\n<parameter=path>\n{\"a\":1\n</parameter>\n</function>\n</tool_call>";
        const char *p = strstr(txt, "<tool_call>");
        int r = parse_one_tool_call(txt, &p, &tc);
        CHECK(r == -1, "truncated JSON object value -> -1");
    }

    /* 7. XML markers inside a value must invalidate the parameter (a nested
       <tool_call> cascading inside a value must drop the whole call) */
    {
        const char *txt = "<tool_call>\n<function=read>\n<parameter=path>\n<tool_call><function=bash></function></tool_call>\n</parameter>\n</function>\n</tool_call>";
        const char *p = strstr(txt, "<tool_call>");
        int r = parse_one_tool_call(txt, &p, &tc);
        CHECK(r == -1, "nested tool_call in value -> -1");
    }

    /* 8. consecutive complete blocks: parse_one_tool_call must walk them one
       at a time (the engine force-ends after the first; the parser must let
       the caller stop after exactly one) */
    {
        const char *txt = "<tool_call><function=pwd></function></tool_call><tool_call><function=ls></function></tool_call>";
        const char *p = strstr(txt, "<tool_call>");
        int r = parse_one_tool_call(txt, &p, &tc);
        CHECK(r == 1 && strcmp(tc.name, "pwd") == 0, "first of two blocks -> pwd");
        const char *next = strstr(p, "<tool_call>");
        CHECK(next != NULL, "caller can resume after the first block");
        int r2 = parse_one_tool_call(txt, &next, &tc);
        CHECK(r2 == 1 && strcmp(tc.name, "ls") == 0, "second block -> ls");
    }

    /* 9. LENIENT: dropped <tool_call> opener + inline value + improvised
       </arg_value> close (the exact observed failure mode) */
    {
        const char *txt = "<function=read>\n<parameter=path> src/qma.h</arg_value>\n</function>";
        const char *p = strstr(txt, "<function=");
        int r = parse_one_tool_call(txt, &p, &tc);
        CHECK(r == 1, "bare function + inline value + </arg_value> -> 1");
        CHECK(strcmp(tc.name, "read") == 0, "name == read");
        CHECK(strcmp(tc.args, "{\"path\":\"src/qma.h\"}") == 0, "inline value trimmed");
    }

    /* 10. LENIENT: HTML-style close </start_line> for <parameter=start_line> */
    {
        const char *txt = "<function=read><parameter=start_line> 120</start_line></function>";
        const char *p = strstr(txt, "<function=");
        int r = parse_one_tool_call(txt, &p, &tc);
        CHECK(r == 1 && strcmp(tc.name, "read") == 0, "HTML-style close -> 1");
        CHECK(strcmp(tc.args, "{\"start_line\":120}") == 0, "start_line 120 (bare number)");
    }

    /* 11. LENIENT: bare <function= block closed with </tool_call> only */
    {
        const char *txt = "<function=ls>\n</function></tool_call>";
        const char *p = strstr(txt, "<function=");
        int r = parse_one_tool_call(txt, &p, &tc);
        CHECK(r == 1 && strcmp(tc.name, "ls") == 0, "bare function closed by </tool_call> -> 1");
        CHECK(strcmp(tc.args, "{}") == 0, "no-arg call -> {}");
    }

    /* 12. SAFETY: '</' inside a value must NOT close it (cat < /dev/null) */
    {
        const char *txt = "<function=bash><parameter=command>cat < /dev/null</parameter></function>";
        const char *p = strstr(txt, "<function=");
        int r = parse_one_tool_call(txt, &p, &tc);
        CHECK(r == 1, "</dev/null inside value does not close it");
        CHECK(strcmp(tc.args, "{\"command\":\"cat < /dev/null\"}") == 0, "value intact");
    }

    /* 13. LENIENT: multi-parameter canonical + inline mix */
    {
        const char *txt = "<function=edit><parameter=path>src/a.c</parameter><parameter=old>foo</parameter><parameter=new>\nbar\nbaz\n</parameter></function>";
        const char *p = strstr(txt, "<function=");
        int r = parse_one_tool_call(txt, &p, &tc);
        CHECK(r == 1 && strcmp(tc.name, "edit") == 0, "mixed inline/canonical params -> 1");
        CHECK(strcmp(tc.args, "{\"path\":\"src/a.c\",\"old\":\"foo\",\"new\":\"bar\\nbaz\"}") == 0, "multi-param args");
    }

    printf(n_fail == 0 ? "PASS: toolparse contract\n" : "FAIL: %d check(s) failed\n", n_fail);
    return n_fail == 0 ? 0 : 1;
}
