/* SPDX-License-Identifier: Apache-2.0
 * ecache.c — bounded expert cache. See ecache.h.
 *
 * Faithful port of waste's ecache.c (LFRU policy, reader threads, hint /
 * prefetch pipeline). The Linux-only parts were kept; the macOS purgeable
 * and mlock experiments were dropped — the phone has no F_NOCACHE and the
 * argument here is the read-ahead, not the wiring.
 */

#include "ecache.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#define EC_DIO_ALIGN 16384

/* --- instrumentation (env QMA_ECPROF=1) --- */
static unsigned long long g_ec_sync_cnt, g_ec_wait_cnt, g_ec_hit_cnt;
static double g_ec_sync_ms, g_ec_wait_ms;
static int g_ec_prof = -1;
static double ec_now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
static void ec_prof_init(void){ if (g_ec_prof<0) g_ec_prof = getenv("QMA_ECPROF")!=NULL; }
void qma_ecache_prof_report(void){
    fprintf(stderr, "[ecprof] sync=%llu(%.1fms) wait=%llu(%.1fms) hit=%llu\n",
            g_ec_sync_cnt, g_ec_sync_ms, g_ec_wait_cnt, g_ec_wait_ms, g_ec_hit_cnt);
    g_ec_sync_cnt=g_ec_wait_cnt=g_ec_hit_cnt=0; g_ec_sync_ms=g_ec_wait_ms=0;
}
void qma_ecache_prof_reset(void){ g_ec_sync_cnt=g_ec_wait_cnt=g_ec_hit_cnt=0; g_ec_sync_ms=g_ec_wait_ms=0; }
static unsigned long long p_s, p_w, p_h; static double p_sm, p_wm;
static unsigned long long p_ev, p_sp;
void qma_ecache_prof_delta(const char *tag){
    qma_ecache *c = NULL; /* need the cache — set via hook below */
    extern qma_ecache *g_prof_cache;
    c = g_prof_cache;
    fprintf(stderr, "[ecprof:%s] sync=%llu(%.1fms) wait=%llu(%.1fms) hit=%llu", tag,
            (unsigned long long)(g_ec_sync_cnt-p_s), g_ec_sync_ms-p_sm,
            (unsigned long long)(g_ec_wait_cnt-p_w), g_ec_wait_ms-p_wm,
            (unsigned long long)(g_ec_hit_cnt-p_h));
    if (c) fprintf(stderr, " ev=%llu spec=%llu miss=%llu",
                   (unsigned long long)(c->evictions-p_ev),
                   (unsigned long long)(c->spec_issued-p_sp),
                   (unsigned long long)(c->misses));
    fprintf(stderr, "\n");
    p_s=g_ec_sync_cnt; p_sm=g_ec_sync_ms; p_w=g_ec_wait_cnt; p_wm=g_ec_wait_ms; p_h=g_ec_hit_cnt;
    if (c) { p_ev=c->evictions; p_sp=c->spec_issued; }
}

static void *ec_dio_alloc(size_t n)
{
    const size_t pad = (n + EC_DIO_ALIGN - 1) / EC_DIO_ALIGN * EC_DIO_ALIGN;
    void *p = NULL;
    return posix_memalign(&p, EC_DIO_ALIGN, pad) == 0 ? p : NULL;
}

static void ec_dio_free(void *p) { free(p); }

static int32_t ec_key(int layer, int expert) { return (layer << 16) | expert; }

static uint32_t ec_hash(int32_t k)
{
    uint32_t x = (uint32_t)k;
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

/* ---- reader threads ----------------------------------------------------- */

typedef struct { int slot, layer, expert; int prio; int src; } eio_job;

struct qma_eio {
    pthread_t th[EC_MAXIO];
    int nthreads;
    pthread_mutex_t mu;
    pthread_cond_t  work;   /* a job was queued, or stop was set            */
    pthread_cond_t  done;   /* some slot left EC_INFLIGHT                   */
    eio_job q[EC_QCAP];
    int qhead, qn, stop;
    qma_ecache *c;
};

static void ec_lock(qma_ecache *c)   { if (c->io) pthread_mutex_lock(&c->io->mu); }
static void ec_unlock(qma_ecache *c) { if (c->io) pthread_mutex_unlock(&c->io->mu); }

static void *eio_worker(void *p)
{
    struct qma_eio *io = (struct qma_eio *)p;
    qma_ecache *c = io->c;
    for (;;) {
        pthread_mutex_lock(&io->mu);
        while (!io->stop && io->qn == 0) pthread_cond_wait(&io->work, &io->mu);
        if (io->stop && io->qn == 0) { pthread_mutex_unlock(&io->mu); return NULL; }
        const eio_job j = io->q[io->qhead];
        io->qhead = (io->qhead + 1) % EC_QCAP;
        io->qn--;
        pthread_mutex_unlock(&io->mu);

        const qma_fetch_fn ff = (j.src && c->fetch_q2) ? c->fetch_q2 : c->fetch;
        const int rc = ff(c->fetch_user, j.layer, j.expert,
                          c->slot[j.slot].data);

        pthread_mutex_lock(&io->mu);
        c->slot[j.slot].state = rc == 0 ? (uint8_t)EC_READY : (uint8_t)EC_FAILED;
        c->slot[j.slot].src = (j.src && c->fetch_q2) ? 1 : 0;
        pthread_cond_broadcast(&io->done);
        pthread_mutex_unlock(&io->mu);
    }
}

/* caller holds the lock; prio 1 = demand (hint), 0 = speculative */
static int eio_push(struct qma_eio *io, int slot, int layer, int expert, int prio, int src)
{
    if (io->qn == EC_QCAP) return -1;
    /* demand reads jump the queue ahead of speculative reads */
    if (prio && io->qn > 0) {
        /* find the first speculative job and insert before it */
        int pos = 0;
        for (int i = 0; i < io->qn; i++) {
            const int idx = (io->qhead + i) % EC_QCAP;
            if (!io->q[idx].prio) break;
            pos = i + 1;
        }
        const int ip = (io->qhead + pos) % EC_QCAP;
        for (int i = io->qn; i > pos; i--) {
            const int from = (io->qhead + i - 1) % EC_QCAP;
            const int to = (io->qhead + i) % EC_QCAP;
            io->q[to] = io->q[from];
        }
        io->q[ip] = (eio_job){ slot, layer, expert, prio, src };
    } else {
        io->q[(io->qhead + io->qn) % EC_QCAP] = (eio_job){ slot, layer, expert, prio, src };
    }
    io->qn++;
    pthread_cond_signal(&io->work);
    return 0;
}

int qma_ecache_io_start(qma_ecache *c, qma_fetch_fn fetch, void *user,
                           int nthreads, int depth)
{
    if (c->io || nthreads <= 0 || c->n_slots <= 0 || !fetch) return 0;
    if (nthreads > EC_MAXIO) nthreads = EC_MAXIO;
    if (depth < 1) depth = 1;
    if (depth > c->n_slots / 4) depth = c->n_slots / 4;
    if (depth < 1) return 0;

    struct qma_eio *io = (struct qma_eio *)calloc(1, sizeof *io);
    if (!io) return -1;
    io->c = c;
    if (pthread_mutex_init(&io->mu, NULL)) { free(io); return -1; }
    pthread_cond_init(&io->work, NULL);
    pthread_cond_init(&io->done, NULL);

    c->fetch = fetch;
    c->fetch_user = user;
    c->depth = depth;
    c->io = io;                       /* readers dereference c->io->mu     */

    for (int i = 0; i < nthreads; i++) {
        if (pthread_create(&io->th[i], NULL, eio_worker, io)) break;
        io->nthreads++;
    }
    if (io->nthreads == 0) {          /* nothing started: stay synchronous */
        c->io = NULL;
        pthread_cond_destroy(&io->work);
        pthread_cond_destroy(&io->done);
        pthread_mutex_destroy(&io->mu);
        free(io);
        return -1;
    }
    return 0;
}

void qma_ecache_io_stop(qma_ecache *c)
{
    struct qma_eio *io = c->io;
    if (!io) return;
    pthread_mutex_lock(&io->mu);
    io->stop = 1;
    pthread_cond_broadcast(&io->work);
    pthread_mutex_unlock(&io->mu);
    for (int i = 0; i < io->nthreads; i++) pthread_join(io->th[i], NULL);
    c->io = NULL;
    pthread_cond_destroy(&io->work);
    pthread_cond_destroy(&io->done);
    pthread_mutex_destroy(&io->mu);
    free(io);
}

/* ---- cache -------------------------------------------------------------- */

int qma_ecache_init(qma_ecache *c, size_t budget_bytes, size_t rec_bytes,
                       int policy)
{
    memset(c, 0, sizeof *c);
    c->rec_bytes = rec_bytes;
    c->budget_bytes = budget_bytes;
    c->policy = policy;
    c->rng = 0x9e3779b9u;
    c->pf_gen = 1;
    c->last_used = -1;
    c->n_held = 0;
    if (!rec_bytes || budget_bytes < rec_bytes) return 0;   /* no cache */

    c->n_slots = (int)(budget_bytes / rec_bytes);
    if (c->n_slots < 4) c->n_slots = 4;
    int hs = 1;
    while (hs < c->n_slots * 2) hs <<= 1;
    c->hash_mask = hs - 1;

    c->slot = (qma_eslot *)calloc((size_t)c->n_slots, sizeof *c->slot);
    c->hash = (int32_t *)malloc((size_t)hs * sizeof *c->hash);
    if (!c->slot || !c->hash) { qma_ecache_free(c); return -1; }
    memset(c->hash, 0xff, (size_t)hs * sizeof *c->hash);   /* all -1 */

    /* One aligned allocation, sliced into slot-aligned pieces (Android's
     * scudo refuses thousands of 16 KiB-aligned allocations). */
    c->slot_bytes = (rec_bytes + EC_DIO_ALIGN - 1) / EC_DIO_ALIGN * EC_DIO_ALIGN;
    const size_t total = (size_t)c->n_slots * c->slot_bytes;
    c->pool = (uint8_t *)ec_dio_alloc(total);
    if (!c->pool) { qma_ecache_free(c); return -1; }
    for (int i = 0; i < c->n_slots; i++) {
        c->slot[i].key = -1;
        c->slot[i].state = EC_EMPTY;
        c->slot[i].data = c->pool + (size_t)i * c->slot_bytes;
    }
    return 0;
}

void qma_ecache_free(qma_ecache *c)
{
    qma_ecache_io_stop(c);
    if (c->slot) free(c->slot);
    if (c->hash) free(c->hash);
    if (c->pool) ec_dio_free(c->pool);
    c->slot = NULL; c->hash = NULL; c->pool = NULL; c->n_slots = 0;
}

static int ec_lookup(qma_ecache *c, int32_t key)
{
    uint32_t h = ec_hash(key) & (uint32_t)c->hash_mask;
    for (int probe = 0; probe <= c->hash_mask; probe++) {
        const int32_t si = c->hash[h];
        if (si < 0) return -1;
        if (c->slot[si].key == key) return si;
        h = (h + 1) & (uint32_t)c->hash_mask;
    }
    return -1;
}

static void ec_insert(qma_ecache *c, int32_t key, int slot)
{
    uint32_t h = ec_hash(key) & (uint32_t)c->hash_mask;
    while (c->hash[h] >= 0) h = (h + 1) & (uint32_t)c->hash_mask;
    c->hash[h] = slot;
}

static void ec_rehash(qma_ecache *c)
{
    memset(c->hash, 0xff, ((size_t)c->hash_mask + 1) * sizeof *c->hash);
    for (int i = 0; i < c->n_slots; i++)
        if (c->slot[i].key >= 0) ec_insert(c, c->slot[i].key, i);
}

static int ec_is_held(const qma_ecache *c, int i)
{
    for (int k = 0; k < c->n_held; k++)
        if (c->held[k] == i) return 1;
    return 0;
}

static int ec_pinned(const qma_ecache *c, int i)
{
    return c->slot[i].state == EC_INFLIGHT || c->slot[i].pin == c->pf_gen ||
           i == c->last_used || ec_is_held(c, i);
}

static int ec_victim(qma_ecache *c)
{
    for (int i = 0; i < c->n_slots; i++)
        if (c->slot[i].state == EC_EMPTY) return i;

    int best = -1;
    uint32_t best_h = 0;
    uint64_t best_l = 0;
    for (int s = 0; s < EC_SAMPLE; s++) {
        c->rng = c->rng * 1664525u + 1013904223u;
        const int i = (int)(c->rng % (uint32_t)c->n_slots);
        if (ec_pinned(c, i)) continue;
        const qma_eslot *sl = &c->slot[i];
        int better;
        if (c->policy == 1)                       /* LRU */
            better = (best < 0) || sl->last < best_l;
        else                                      /* LFRU */
            better = (best < 0) || sl->hits < best_h ||
                     (sl->hits == best_h && sl->last < best_l);
        if (better) { best = i; best_h = sl->hits; best_l = sl->last; }
    }
    if (best < 0)
        for (int i = 0; i < c->n_slots; i++)
            if (!ec_pinned(c, i)) { best = i; break; }
    return best;
}

/* Claim `vi` for `key`, counting the read that is about to happen. */
static void ec_claim_spec(qma_ecache *c, int vi, int32_t key)
{
    const int had = c->slot[vi].key >= 0;
    c->slot[vi].key = key;
    c->slot[vi].state = EC_INFLIGHT;
    c->slot[vi].fresh = 0;
    c->slot[vi].pin = c->pf_gen + 1;   /* immune during the layer that demands it */
    c->slot[vi].birth = 0;             /* spec */
    c->slot[vi].birth_gen = c->pf_gen;
    c->slot[vi].hits = 0;              /* LFRU victim of choice when idle */
    c->slot[vi].last = c->clock;
    if (had) { c->evictions++; ec_rehash(c); }
    else ec_insert(c, key, vi);
    c->spec_issued++;
    c->bytes_read += c->rec_bytes;
}

static void ec_claim(qma_ecache *c, int vi, int32_t key, int fresh)
{
    const int had = c->slot[vi].key >= 0;
    c->slot[vi].key = key;
    c->slot[vi].state = EC_INFLIGHT;
    c->slot[vi].fresh = (uint8_t)fresh;
    c->slot[vi].pin = fresh ? c->pf_gen : 0;
    c->slot[vi].birth = fresh ? 1 : 2;  /* 1 hint, 2 sync */
    c->slot[vi].birth_gen = c->pf_gen;
    c->slot[vi].hits = 1;
    c->slot[vi].last = c->clock;
    if (had) { c->evictions++; ec_rehash(c); }
    else ec_insert(c, key, vi);
    c->misses++;
    c->bytes_read += c->rec_bytes;
}

static void ec_drop(qma_ecache *c, int vi)
{
    c->slot[vi].key = -1;
    c->slot[vi].state = EC_EMPTY;
    c->slot[vi].fresh = 0;
    c->slot[vi].pin = 0;
    ec_rehash(c);
}

/* Release one more read into the pipe. Caller holds the lock. */
static void ec_issue_next(qma_ecache *c)
{
    if (!c->io) return;
    while (c->pf_issued < c->pf_n) {
        const int eid = c->pf_ids[c->pf_issued++];
        const int32_t key = ec_key(c->pf_layer, eid);
        const int si = ec_lookup(c, key);
        if (si >= 0) {
            c->slot[si].pin = c->pf_gen;
            continue;
        }
        const int vi = ec_victim(c);
        if (vi < 0) { c->pf_issued--; return; }
        ec_claim(c, vi, key, 1);
        c->prefetched++;
        if (eio_push(c->io, vi, c->pf_layer, eid, 1, 0) != 0) {   /* hint: Q4 */
            c->prefetched--;
            c->misses--;
            c->bytes_read -= c->rec_bytes;
            ec_drop(c, vi);
            c->pf_issued--;
        }
        return;
    }
}

void qma_ecache_prefetch(qma_ecache *c, int layer, const int *ids, int n)
{
    if (!c->io || c->n_slots <= 0 || n <= 0) return;
    ec_lock(c);
    for (int i = 0; i < n; i++) {
        if (ids[i] < 0) continue;
        const int32_t key = ec_key(layer, ids[i]);
        if (ec_lookup(c, key) >= 0) continue;
        const int vi = ec_victim(c);
        if (vi < 0) break;
        ec_claim_spec(c, vi, key);
        if (eio_push(c->io, vi, layer, ids[i], 0, 0) != 0) {   /* spec: Q4 */
            c->spec_issued--;
            c->bytes_read -= c->rec_bytes;
            ec_drop(c, vi);
            break;
        }
    }
    ec_unlock(c);
}

void qma_ecache_hint(qma_ecache *c, int layer, const int *ids, int n)
{
    if (!c->io || c->n_slots <= 0 || n <= 0) return;

    ec_lock(c);
    c->pf_gen++;
    c->pf_layer = layer;
    c->pf_n = n > EC_PF_MAX ? EC_PF_MAX : n;
    if (c->pf_n > c->n_slots / 4) c->pf_n = c->n_slots / 4;
    for (int i = 0; i < c->pf_n; i++) c->pf_ids[i] = ids[i];
    c->pf_issued = 0;
    for (int d = 0; d < c->depth; d++) ec_issue_next(c);
    ec_unlock(c);
}

const uint8_t *qma_ecache_get(qma_ecache *c, int layer, int expert,
                                 qma_fetch_fn fetch, void *user)
{
    const int32_t key = ec_key(layer, expert);

    ec_lock(c);
    c->clock++;
    if (c->last_used >= 0 && c->slot[c->last_used].state == EC_READY)
        c->last_used = -1;   /* released on the next get */

    if (c->n_slots > 0) {
        int si = ec_lookup(c, key);
        if (si >= 0) {
            if (g_ec_prof && c->slot[si].state == EC_INFLIGHT) {
                const double t0 = ec_now_s();
                while (c->io && c->slot[si].state == EC_INFLIGHT)
                    pthread_cond_wait(&c->io->done, &c->io->mu);
                g_ec_wait_ms += (ec_now_s()-t0)*1000.0; g_ec_wait_cnt++;
            } else {
                while (c->io && c->slot[si].state == EC_INFLIGHT)
                    pthread_cond_wait(&c->io->done, &c->io->mu);
            }
            if (c->slot[si].state != EC_READY) {
                ec_drop(c, si);
                ec_issue_next(c);
                ec_unlock(c);
                return NULL;
            }
            /* A record this hint brought in was already counted as a miss
             * when its read was issued; counting it again here would turn
             * every prefetch into a fictitious hit. */
            if (c->slot[si].fresh) c->slot[si].fresh = 0;
            else {
                c->hits++; c->slot[si].hits++;
                if (c->slot[si].birth == 0) c->dbg_hit_spec++;
                else if (c->slot[si].birth == 1 &&
                         c->slot[si].birth_gen < c->pf_gen)
                    c->dbg_hit_hint_prev++;
                else if (c->slot[si].birth == 2) c->dbg_hit_sync++;
                else c->dbg_hit_hint_this++;
            }
            ec_prof_init();
            if (g_ec_prof) { g_ec_hit_cnt++; }
            c->slot[si].last = c->clock;
            uint8_t *d = c->slot[si].data;
            c->last_used = si;
            ec_issue_next(c);
            ec_unlock(c);
            return d;
        }
    }

    if (c->n_slots == 0) {
        c->misses++;
        c->bytes_read += c->rec_bytes;
        ec_unlock(c);
        return NULL;
    }

    /* Not resident and not hinted: read it here, but still through a slot,
     * so a reader thread cannot pick the same one meanwhile. Tiered mode
     * (fetch_q2 set): the runtime miss is the expensive path (unpredicted
     * expert), so fill it with the lighter Q2 slab — half the bytes, and
     * the worker decodes with the Q2 types. The slot is tagged src=1 so
     * the decode picks the right slab split + quant types. */
    const int vi = ec_victim(c);
    if (vi < 0) { ec_unlock(c); return NULL; }
    ec_claim(c, vi, key, 0);
    uint8_t *dst = c->slot[vi].data;
    ec_unlock(c);

    ec_prof_init();
    double t0 = 0; if (g_ec_prof) t0 = ec_now_s();
    const int rc = c->fetch_q2 ? c->fetch_q2(user, layer, expert, dst)
                               : fetch(user, layer, expert, dst);
    if (g_ec_prof) { g_ec_sync_ms += (ec_now_s()-t0)*1000.0; g_ec_sync_cnt++; }

    ec_lock(c);
    if (rc != 0) {
        ec_drop(c, vi);
        ec_unlock(c);
        return NULL;
    }
    c->slot[vi].src = c->fetch_q2 ? 1 : 0;
    c->slot[vi].state = EC_READY;
    c->last_used = vi;
    if (c->io) pthread_cond_broadcast(&c->io->done);
    ec_issue_next(c);
    ec_unlock(c);
    return dst;
}

const uint8_t *qma_ecache_hold(qma_ecache *c, int layer, int expert,
                                  qma_fetch_fn fetch, void *user)
{
    if (c->n_slots <= 0 || c->n_held >= EC_PF_MAX) return NULL;
    const uint8_t *r = qma_ecache_get(c, layer, expert, fetch, user);
    if (!r) return NULL;
    ec_lock(c);
    if (c->last_used >= 0) {
        if (!ec_is_held(c, c->last_used)) c->held[c->n_held++] = c->last_used;
        c->last_used = -1;
    }
    ec_unlock(c);
    return r;
}

void qma_ecache_release(qma_ecache *c)
{
    ec_lock(c);
    c->n_held = 0;
    ec_unlock(c);
}

void qma_ecache_clear(qma_ecache *c)
{
    ec_lock(c);
    for (int i = 0; i < c->n_slots; i++) {
        c->slot[i].key = -1;
        c->slot[i].state = EC_EMPTY;
        c->slot[i].fresh = 0;
        c->slot[i].pin = 0;
        c->slot[i].hits = 0;
        c->slot[i].last = 0;
    }
    if (c->hash) memset(c->hash, 0xff, ((size_t)c->hash_mask + 1) * sizeof *c->hash);
    c->clock = c->hits = c->misses = c->bytes_read = 0;
    c->evictions = c->prefetched = c->spec_issued = 0;
    c->pf_n = c->pf_issued = 0;
    c->last_used = -1;
    c->n_held = 0;
    ec_unlock(c);
}

/* return the source tag (0 = Q4 primary, 1 = Q2 degraded) of the resident
   slot for (layer, expert); -1 if not resident */
int qma_ecache_src(const qma_ecache *c, int layer, int expert)
{
    const int32_t key = ec_key(layer, expert);
    int rc = -1;
    ec_lock((qma_ecache *)c);
    int si = ec_lookup((qma_ecache *)c, key);
    if (si >= 0 && c->slot[si].state == EC_READY)
        rc = c->slot[si].src == 0 ? 0 : 1;   /* 1 and 2 both decode as Q2 */
    ec_unlock((qma_ecache *)c);
    return rc;
}
