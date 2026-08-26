/* worker.h — LFM2.5 document-worker pool for qma.
 * The main qwen35 agent spawns workers: small LFM2.5 instances (weights
 * loaded ONCE and shared) that hold a document in their KV and answer
 * questions about it. Each worker is a KV runstate with a base (worker
 * system prompt + document, processed once) and ephemeral query layers on
 * top — a clean KV per query, or a persistent conversation in long-term
 * mode. Only KV is per-worker; the weights are shared. */
#ifndef WORKER_H
#define WORKER_H

/* load the shared LFM model (idempotent). Returns 0 ok, -1 error. */
int  worker_model_load(const char *path);

/* spawn a worker holding `doc` under `name` (default role prompt if
   sysprompt is NULL). Returns worker id (>= 0) or -1. */
int  worker_spawn(const char *name, const char *sysprompt, const char *doc);

/* ask the worker a question. ephemeral = 1 resets the KV to the document
   base after answering (clean KV per query); 0 keeps the conversation in
   long-term mode. Returns a heap-allocated answer string. */
char *worker_ask(int id, const char *question, int ephemeral, int max_tokens);

/* close a worker and free its KV. Returns 0 ok. */
int  worker_close(int id);

/* render a worker listing into out (heap-allocated) */
char *worker_list(void);

/* number of active workers */
int  worker_count(void);

#endif /* WORKER_H */
