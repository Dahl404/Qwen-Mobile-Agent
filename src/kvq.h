/* kvq.h - quantized KV cache records for the HCM tiered attention.
 *
 * Formats (per kv-head, per token, N_EMBD_HEAD = 256 dims):
 *   KVQ4: 8 sub-blocks x 32 lanes. Each sub-block: fp16 scale d, fp16 min m,
 *         32 x 4-bit quants (16 B). 20 B x 8 = 160 B per record.
 *   KVQ2: 4 sub-blocks x 64 lanes. Each sub-block: fp16 d, fp16 m,
 *         64 x 2-bit quants (16 B). 20 B x 4 = 80 B per record.
 *   value = q * d + m,  q in [0,15] (q4) / [0,3] (q2).
 *
 * Storage: every position owns a fixed KVQ_SLOT (240 B) holding BOTH the
 * q4 (offset 0, 160 B) and q2 (offset KVQ4_REC, 80 B) records, both
 * quantized from the original fp32 at append time (best quality for both
 * tiers — no q4->q2 re-quantization error). The HCM tier byte per position
 * selects which record attention reads; demotion/promotion is a tag flip.
 */
#ifndef KVQ_H
#define KVQ_H

#include <stdint.h>
#include <stddef.h>

#define KVQ_SLOT   240            /* bytes per (head, position) slot       */
#define KVQ4_SUB   8              /* 8 sub-blocks x 32 lanes */
#define KVQ2_SUB   4              /* 4 sub-blocks x 64 lanes */

/* record sizes (K or V, one head, one token) */
#define KVQ4_REC   160
#define KVQ2_REC   80
#define KVQ2_OFF   160            /* q2 record offset inside the slot */

/* quantize a 256-float K or V vector into the slot's q4 region (offset 0) */
void kvq4_quant(const float *x, int n, uint8_t *slot);
/* quantize into the slot's q2 region (offset KVQ2_OFF) */
void kvq2_quant(const float *x, int n, uint8_t *slot);

/* dequantize (for tests / debug); rec may point at either region */
void kvq4_dequant(const uint8_t *rec, float *y, int n);
void kvq2_dequant(const uint8_t *rec, float *y, int n);

/* Q.K dot: score += dequant(rec)[i] * x[i] over n=256 lanes (NEON) */
float kvq4_dot(const uint8_t *rec, const float *x, int n);
float kvq2_dot(const uint8_t *rec, const float *x, int n);

/* weighted V accumulate: oacc[i] += w * dequant(rec)[i] over n lanes.
   oacc must hold n floats (the caller's oacc array reinterpreted). */
void kvq4_vacc(const uint8_t *rec, float w, float *oacc, int n);
void kvq2_vacc(const uint8_t *rec, float w, float *oacc, int n);

/* Q-Hitter quantization-friendliness: relative q4 RMS error for a K/V
   vector (0 = perfect, higher = worse quantizer). Computed at append from
   the fp32 source so selection can prefer quantization-friendly tokens. */
float kvq4_qf(const uint8_t *slot, const float *x, int n);

#endif /* KVQ_H */
