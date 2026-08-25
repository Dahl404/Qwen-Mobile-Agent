/* tools.h — tool schemas + dispatch for the unified agent.
 * Schemas are rendered into the model's native <tools> system block;
 * tool_dispatch parses the JSON args and runs the implementation, returning
 * a heap-allocated result string (caller frees). "ERROR: ..." prefix marks
 * failures so the model sees them in the <tool_response>.
 */
#ifndef TOOLS_H
#define TOOLS_H

#include "json.h"

typedef struct {
    const char *name;
    const char *description;
    const char *params_json;   /* {"properties":{...},"required":[...]} */
} tool_schema_t;

extern const tool_schema_t g_tools[];
extern const int g_n_tools;

/* Render the <tools>...</tools> header (one tool JSON per line). */
void tools_render_header(char *out, size_t outsz);

/* Set the real path of the mounted internal tree (see map_path). Called by
 * the agent at boot after extracting the embedded blob. */
void intern_set_root(const char *root);

/* Ask the user (blocking stdin read); returns the answer (heap). */
char *tools_ask_user(const char *question);

/* Agent memory archive tools (implemented in agent.c — they need the
   loaded model + runstate to compute real K/V for the pinned arena). */
char *agent_memory_write(const jval_t *args);
char *agent_memory_append(const jval_t *args);
char *agent_memory_list(const jval_t *args);
char *agent_memory_delete(const jval_t *args);
char *agent_memory_clear(void);

/* Re-embed every memory.json entry into the archive arena at boot. */
void agent_memory_restore(void);

/* shared small helpers (also used by agent.c's memory tools) */
char *xstrdup(const char *s);
char *fmt(const char *fmt, ...);
char *read_whole_file(const char *path);

/* Execute one tool call. result is heap-allocated. Returns 0 on success
 * (including tool-level errors, which are prefixed "ERROR:"). */
int tool_dispatch(const char *name, const char *args_json, char **result);

#endif /* TOOLS_H */
