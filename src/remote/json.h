/* json.h — tiny JSON parser + helpers for the unified agent.
 * Parses tool-call arguments ({"k":v,...}) into a typed tree so tool
 * implementations can read named fields, arrays, and nested objects.
 * Also provides full-string validation (json_valid) used to refuse
 * truncated/malformed tool calls.
 */
#ifndef JSON_H
#define JSON_H

#include <stddef.h>

typedef enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } jtype_t;

typedef struct jval jval_t;
struct jval {
    jtype_t type;
    const char *s; size_t slen;   /* J_STR: string bytes (not NUL-terminated) */
    double num;                   /* J_NUM */
    int boolean;                  /* J_BOOL */
    jval_t *items; size_t n;      /* J_ARR: items; J_OBJ: values */
    const char **keys;            /* J_OBJ: member names (parallel to items) */
};

/* Parse one JSON value from text. Returns NULL on error. */
jval_t *json_parse(const char *text);

/* 1 iff text is exactly one valid JSON value (no trailing junk). */
int json_valid(const char *text);

/* Look up a member by name (J_OBJ only). Returns NULL if absent. */
const jval_t *json_obj_get(const jval_t *obj, const char *key);

/* Index an array element (J_ARR only). Returns NULL out of range. */
const jval_t *json_arr_get(const jval_t *arr, size_t i);

/* String value ("" for non-strings), double, bool with defaults. */
const char *json_str(const jval_t *v);      /* NUL-terminated copy? no: points into input */
double      json_num(const jval_t *v);
int         json_bool(const jval_t *v);

/* Free a parsed tree. */
void json_free(jval_t *v);

/* JSON-escape helpers (for building the <tools> header and tool responses). */
void json_escape(const char *in, char *out, size_t outsz);
void json_quote_escape(const char *in, char *out, size_t outsz);

#endif /* JSON_H */
