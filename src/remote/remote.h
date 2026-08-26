/* remote.h — qma's headless autonomous remote agent.
 *
 * A dispatched remote agent is a self-contained binary:
 *
 *   [base ELF][LFM model blob][internal tree][task cfg][KV][state][hdr]
 *
 * It boots on a foreign device (no model file needed — the binary IS the
 * model), rebuilds itself for that platform if needed, reads the task spec
 * from the embedded cfg blob (or a task file), executes the mission
 * AUTONOMOUSLY with no UI and no chat, collects data into its internal
 * tree, packs a NEW generation binary with the results + its KV memory,
 * and follows the return directive (write it back to a path / stay /
 * self-destruct). qma re-ingests the returned binary: it extracts the
 * collected data and loads the returned KV into a long-term worker so it
 * can ask the agent questions about the remote environment.
 *
 * Task spec (JSON, embedded as the cfg blob or given with --task):
 *   {
 *     "role":      "system prompt / role for the agent",
 *     "mission":   "what to accomplish (free text the agent autonomously pursues)",
 *     "done_when": "completion protocol — how it knows it is finished",
 *     "collect":   ["/remote/path", ...]  paths/globs to copy into /internal/collect/,
 *     "report":    "optional: what to write to /internal/collect/report.txt",
 *     "return":    {"mode":"path"|"stay"|"selfdestruct", "path":"/return/path"},
 *     "max_turns": 128,
 *     "max_seconds": 3600
 *   }
 *
 * Memory: the agent has the same corrected memory model as qma — the KV
 * is a linear always-attended cache (nothing to evict), the system prompt
 * (role + mission) is processed once and stays, and memory_write/append/
 * list tools pin extra facts the agent wants permanent. The whole KV
 * travels home in the return binary.
 */
#ifndef REMOTE_H
#define REMOTE_H

#include <stdint.h>

/* remote_tools.c — the remote agent's tool surface (headless). */
int  remote_tool_dispatch(const char *name, const char *args_json,
                          char **result);
void remote_tools_init(void);

/* remote_tools.c */
int  remote_parse_call(const char *text, const char **after,
                       char *name, size_t name_cap, char *args, size_t args_cap);
const char *remote_tools_header(void);

/* remote_main.c — lifecycle helpers shared with tools. */
const char *remote_internal_root(void);   /* /internal mount point (real path) */
const char *remote_collect_dir(void);     /* /internal/collect/ */
void        remote_log(const char *fmt, ...);
/* platform_id.c */
void        qma_platform_id(char *buf, size_t cap);

#endif /* REMOTE_H */
