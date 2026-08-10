/* test_grammar.c — grammar fast-path parity test.
 *
 * grammar_tool_open_state() + grammar_tool_token_from_state() must agree
 * with the wrapper grammar_tool_token_ok() on every (ans, token) pair:
 * the fast path builds the DFA state ONCE per decode step and simulates
 * each candidate against it, replacing the old per-candidate full replay.
 * This proves the refactor is behavior-preserving.
 *
 * Build: make work/test_grammar   (or clang -I src -o work/test_grammar
 *        work/test_grammar.c src/grammar.c -lm)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "qma.h"
#include "grammar.h"

static int failures = 0;

/* wrapper vs fast path on one (ans, id) */
static void check(qma_t *m, const char *ans, int id) {
    size_t alen = strlen(ans);
    int old = grammar_tool_token_ok(m, ans, alen, id);
    gpos_t g;
    int st = grammar_tool_open_state(m, ans, alen, &g);
    int nw;
    if (st == 0)      nw = 1;              /* no open call: unconstrained */
    else if (st < 0)  nw = 0;              /* open but invalid: reject all */
    else              nw = grammar_tool_token_from_state(&g, m, id);
    if (old != nw) {
        failures++;
        printf("MISMATCH ans=%-70s id=%2d (%-14s) old=%d new=%d st=%d\n",
               ans, id, m->tok_text[id] ? m->tok_text[id] : "?", old, nw, st);
    }
}

int main(void) {
    qma_t *m = calloc(1, sizeof(*m));
    if (!m) { fprintf(stderr, "oom\n"); return 2; }
    m->n_vocab = 248320;
    /* token pieces used by the battery */
    m->tok_text[1]  = (char *)"<";
    m->tok_text[2]  = (char *)"<parameter=";
    m->tok_text[3]  = (char *)"</function>";
    m->tok_text[4]  = (char *)"path";
    m->tok_text[5]  = (char *)"\n";
    m->tok_text[6]  = (char *)"\"hello\"";
    m->tok_text[7]  = (char *)"</think>";
    m->tok_text[8]  = (char *)"<tool_call>";
    m->tok_text[9]  = (char *)"x";
    m->tok_text[10] = (char *)"</parameter>";
    m->tok_text[11] = (char *)"<function=read";
    m->tok_text[12] = (char *)"{ \"a\": 1 }";
    m->tok_text[13] = (char *)" ";
    m->tok_text[14] = (char *)"123";
    m->tok_text[15] = (char *)"</tool_call>";
    m->tok_text[16] = (char *)"\"a";
    m->tok_text[17] = (char *)"\\\"";
    m->tok_text[18] = (char *)"hello world text";

    const char *anses[] = {
        "",                                                         /* no call */
        "plain text, no call",                                      /* no call */
        "<tool_call>\n",                                            /* expect ws or '<' */
        "<tool_call>\n<function=read",                              /* name: chars or '>' */
        "<tool_call>\n<function=read>",                             /* expect ws or '<' */
        "<tool_call>\n<function=read>\n<parameter=path>",           /* expect value */
        "<tool_call>\n<function=read>\n<parameter=path>\n\"hello",  /* in string */
        "<tool_call>\n<function=read>\n<parameter=path>\n123",      /* number */
        "<tool_call>\n<function=read>\n<parameter=path>\n{ \"a\": 1", /* object */
        "<tool_call>\n<function=read>\n<parameter=path>\n\"hello\"\n</parameter>",      /* ws or '<' */
        "<tool_call>\n<function=read>\n<parameter=path>\n\"hello\"\n</parameter>\n</function>", /* ws or '<' */
        "<tool_call>\n<function=read>\n<parameter=path>\n\"hello\"\n</parameter>\n</function>\n</tool_call>", /* complete */
        "<tool_call>\n<function=read\nx",                           /* invalid text */
        "text <tool_call>\n<function=read>",                        /* open after text */
        "<tool_call>\n<function=read>\n<parameter=path>\n\"hello\"\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=write", /* second open */
    };
    const int n_ans = (int)(sizeof(anses) / sizeof(anses[0]));

    for (int a = 0; a < n_ans; a++)
        for (int i = 1; i <= 18; i++)
            check(m, anses[a], i);

    /* spot-check the semantics of the wrapper itself */
    int ok1 = grammar_tool_token_ok(m, "<tool_call>\n<function=read>",
                                    strlen("<tool_call>\n<function=read>"), 2);  /* "<parameter=" */
    int ok2 = grammar_tool_token_ok(m, "<tool_call>\n<function=read>",
                                    strlen("<tool_call>\n<function=read>"), 9);  /* "x" */
    int ok3 = grammar_tool_token_ok(m, "no call here",
                                    strlen("no call here"), 9);                 /* unconstrained */
    printf("sanity: <parameter= ok=%d (want 1), x ok=%d (want 0), no-call ok=%d (want 1)\n",
           ok1, ok2, ok3);
    if (ok1 != 1 || ok2 != 0 || ok3 != 1) failures++;

    free(m);
    if (failures) { printf("FAILED: %d mismatches\n", failures); return 1; }
    printf("PASS: grammar fast path matches wrapper on %d cases x 18 tokens\n", n_ans);
    return 0;
}
