/* kvq4_dot vs kvq4_dot_q8: accuracy on quantized-realistic data + perf */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "kvq.h"
#include "qma.h"

static uint64_t rs = 0x9e3779b97f4a7c15ULL;
static uint32_t rnd(void){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return (uint32_t)(rs>>32); }
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return (double)t.tv_sec + (double)t.tv_nsec/1e9; }

int main(void){
    const int n = N_EMBD_HEAD;
    uint8_t slot[KVQ4_REC];
    float x[n]; static int8_t qi8[n]; static float xsb[n/32]; static int xsum[n/32];
    double maxrel = 0; double sumrel = 0;
    const int TRIALS = 2000;
    for (int tr = 0; tr < TRIALS; tr++) {
        /* realistic K-like vector: smooth + noise */
        for (int i = 0; i < n; i++)
            x[i] = sinf((float)i * 0.11f + (float)(tr % 97)) * 1.7f
                 + ((double)rnd()/4294967295.0 - 0.5) * 0.35f;
        kvq4_quant(x, n, slot);            /* build a real record */
        for (int i = 0; i < n; i++)        /* query = another such vector */
            ; /* reuse x as query: score of a slot against its own source */
        kvq_q_quant(x, n, qi8, xsb, xsum);
        float c = kvq4_dot(slot, x, n);
        float a = kvq4_dot_q8(slot, qi8, xsb, xsum, n);
        float ref = 0;                    /* exact fp32 ground truth */
        for (int i = 0; i < n; i++) ref += (float)x[i] * x[i];  /* slot==dequant(x)~self */
        (void)ref;
        double rel = fabs(a - c) / (fabs(c) + 1e-6);
        sumrel += rel;
        if (rel > maxrel) maxrel = rel;
        if (!isfinite(a)) { printf("NONFINITE at trial %d\n", tr); return 1; }
    }
    printf("accuracy over %d trials: mean rel err %.2e, max %.2e\n",
           TRIALS, sumrel / TRIALS, maxrel);

    /* perf */
    for (int i = 0; i < n; i++) x[i] = sinf((float)i * 0.05f);
    kvq4_quant(x, n, slot);
    kvq_q_quant(x, n, qi8, xsb, xsum);
    volatile float sink = 0;
    double t0 = now();
    for (int r = 0; r < 300000; r++) sink += kvq4_dot(slot, x, n);
    double tc = now() - t0;
    t0 = now();
    for (int r = 0; r < 300000; r++) sink += kvq4_dot_q8(slot, qi8, xsb, xsum, n);
    double ta = now() - t0;
    printf("perf: fp32-expand %.2f ms vs i8-sdot %.2f ms (%.2fx per dot)\n",
           tc*1e3, ta*1e3, tc/ta);
    return 0;
}
