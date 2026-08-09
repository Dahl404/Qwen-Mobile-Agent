/* SPDX-License-Identifier: Apache-2.0
 * ecache.h — bounded expert cache (faithful port of waste's ecache).
 *
 * The model does not fit in RAM, so expert weights are streamed from disk
 * and a bounded cache decides how often that costs a read. Policy is LFRU
 * — frequency first, recency as tiebreak, victims chosen from a small
 * random sample rather than a full scan.
 *
 * Unlike the madvise hint approach this replaces, the engine owns the
 * cache: slots have an explicit state machine (EMPTY/READY/INFLIGHT/
 * FAILED), a hash index maps (layer, expert) -> slot, and reader threads
 * keep reads in flight while the caller's matmuls run. hint() names the
 * records a layer is about to ask for; each get() releases one more read
 * into the pipe. prefetch() speculatively fills a layer that has not
 * routed yet, at lower priority, without polluting the demand hit rate.
 *
 * One record == one expert: the gate, up and down slabs concatenated.
 * rec_bytes is sized to the largest expert (down_exps is Q6_K in some
 * layers), and the fetch callback preads the three regions into the slot.
 */

#ifndef QMA_ECACHE_H
#define QMA_ECACHE_H

#include <stddef.h>
#include <stdint.h>

/* `fetch(user, layer, expert, dst)` must fill rec_bytes and return 0.
 * With read-ahead on it is called from the reader threads as well as the
 * caller's, so it must be safe to enter concurrently. */
typedef int (*qma_fetch_fn)(void *user, int layer, int expert, uint8_t *dst);

enum {
    EC_EMPTY = 0,        /* no record, and none on the way                  */
    EC_READY,            /* data is the record named by key                 */
    EC_INFLIGHT,         /* an I/O thread is filling data right now         */
    EC_FAILED            /* the read for key failed; get() reports and drops */
};

typedef struct {
    int32_t  key;        /* layer<<16 | expert, or -1 when empty            */
    uint8_t  state;      /* one of the EC_* above                           */
    uint8_t  fresh;      /* read was issued for this batch: not a hit yet   */
    uint8_t  birth;      /* how this slot was claimed: 0 spec, 1 hint, 2 sync */
    uint32_t hits;       /* LFRU frequency term                             */
    uint64_t last;       /* LFRU recency term                               */
    uint64_t pin;        /* hint generation holding it; 0 = evictable       */
    uint64_t birth_gen;  /* pf_gen at claim (debug: which hint owned it)    */
    uint8_t *data;
} qma_eslot;

#define EC_PF_MAX 64     /* most experts one hint may name (top_k <= 8)     */
#define EC_SAMPLE 16     /* victims sampled per eviction (Redis-style)      */
#define EC_QCAP   256    /* outstanding read requests                       */
#define EC_MAXIO  8      /* reader threads                                  */

struct qma_eio;       /* opaque: the reader threads and their queue      */

typedef struct {
    qma_eslot *slot;
    int32_t *hash;       /* open addressing: hash -> slot index, -1 empty   */
    int n_slots, hash_mask;
    size_t rec_bytes, budget_bytes;
    uint64_t clock, hits, misses, bytes_read, evictions;
    uint64_t prefetched; /* misses whose read was started before it blocked */
    uint64_t spec_issued;/* records the lookahead fetched on a guess         */
    uint64_t dbg_hit_spec, dbg_hit_hint_prev, dbg_hit_sync, dbg_hit_hint_this;
    unsigned rng;
    int policy;          /* 0 = LFRU, 1 = LRU                               */

    struct qma_eio *io;
    qma_fetch_fn fetch;
    void *fetch_user;
    int depth;                       /* reads kept in flight               */
    uint64_t pf_gen;                 /* current hint generation, never 0   */
    int pf_ids[EC_PF_MAX];
    int pf_layer, pf_n, pf_issued;

    int held[EC_PF_MAX];
    int n_held;
    int last_used;
    size_t slot_bytes;               /* rec_bytes rounded up to 16 KiB      */
    uint8_t *pool;                   /* one aligned block, sliced into slots */
} qma_ecache;

/* budget_bytes 0 disables caching (every access reads). Returns 0 on ok. */
int  qma_ecache_init(qma_ecache *c, size_t budget_bytes, size_t rec_bytes,
                        int policy);
void qma_ecache_free(qma_ecache *c);
void qma_ecache_clear(qma_ecache *c);

/* Returns a pointer to the expert's record bytes, reading it through
 * `fetch` on a miss, or waiting for the read a hint already started.
 * NULL on failure. */
const uint8_t *qma_ecache_get(qma_ecache *c, int layer, int expert,
                                 qma_fetch_fn fetch, void *user);

/* get(), except the record stays claimed after the next one is asked for
 * (a caller handing k records to k threads needs all of them to outlive
 * the loop that collected them). Holds at most EC_PF_MAX; every hold must
 * be matched by one release, which releases the whole set. */
const uint8_t *qma_ecache_hold(qma_ecache *c, int layer, int expert,
                                  qma_fetch_fn fetch, void *user);
void qma_ecache_release(qma_ecache *c);

/* ---- read-ahead ---------------------------------------------------------
 * A blocking pread stops the arithmetic. A MoE layer knows all of its
 * top-K expert ids before it reads the first one, so the reads can be
 * issued ahead and consumed as the matmuls finish.
 * nthreads 0 leaves every fetch synchronous. Returns 0 on success; failing
 * to start the threads is not fatal, the cache simply stays synchronous. */
int  qma_ecache_io_start(qma_ecache *c, qma_fetch_fn fetch,
                            void *user, int nthreads, int depth);
void qma_ecache_io_stop(qma_ecache *c);

/* Name the records this layer is about to ask for, in the order it will
 * ask. Reads for the first `depth` of them that are not resident start
 * now, and each get() releases one more into the pipe. */
void qma_ecache_hint(qma_ecache *c, int layer, const int *ids, int n);

/* Speculative fill for a layer that has not routed yet. Unlike a hint
 * these are not pinned and are not counted as misses. */
void qma_ecache_prefetch(qma_ecache *c, int layer, const int *ids, int n);

#endif /* QMA_ECACHE_H */
