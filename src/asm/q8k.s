.arch armv8.2-a+dotprod
/* q8k.s — int8 quantized-activation dot kernels (SDOT/NEON).
 *
 * Faithful asm port of q8k.c (waste gq_neon.c). Activation x is quantized
 * once per layer (xq + per-256-block scale xd + per-16-lane sums xsum),
 * then <dequant(row), x> is computed with NEON SDOT against Q4_K / Q6_K.
 * Requires armv8.2-a+dotprod (this device has it).
 *
 * CRITICAL AARCH64 REGISTER FACT: the scalar FP register sN is the LOW
 * 32-bit LANE of vector register vN — sN aliases vN.s[0]. To read a
 * scalar that a FMLA-by-element consumes (e.g. `fmla v0.4s, v26.4s,
 * v7.s[0]`), the scalar MUST live in v7.s[0]. Keeping d1xd in "s28"
 * while the fmla reads v7.s[0] silently multiplies by garbage.
 *
 * Register discipline (AAPCS64):
 *   - leaf kernels (q4/q6): only caller-saved regs (x0-x18, v0-v7,
 *     v16-v31). Scalars live in s0-s7 (v0-v7 lane 0) and s16-s31 only
 *     AFTER the dot-temp vectors are consumed.
 *   - gateup: saves d8-d15 (v8-v15 are callee-saved) and keeps its
 *     block accumulators on the stack.
 *   - s-lanes are addressed via their sN name or vN.s[0] — never assume
 *     s28 == v7.s[0].
 */
.include "qma_defs.s"

/* ------------------------------------------------------------------ */
/* Scale-decode helpers (q8k_scale_min_k4, 6-bit scale/min unpack).    */
/* Simple form (j < 4):  out = mreg * ((word >> shift) & 63)            */
/* Packed form (j >= 4): out = mreg * (((w1>>s1)&0xF) |                */
/*                                   (((w2>>s2)&3) << 4))              */
/* Scratch: w17, w18 only — never touches the source words.            */
/* out/mreg are s-registers (sN = vN.s[0]).                            */
/* ------------------------------------------------------------------ */
.macro SCALE out, word, shift, mreg
    ubfx w17, \word, #\shift, #6
    ucvtf \out, w17
    fmul \out, \out, \mreg
.endm

.macro SCALE_P out, w1, s1, w2, s2, mreg
    ubfx w18, \w2, #\s2, #2
    ubfx w17, \w1, #\s1, #4
    lsl w18, w18, #4
    orr w17, w17, w18
    ucvtf \out, w17
    fmul \out, \out, \mreg
.endm

/* Q4 dot body (uses qs ptr x5, xp ptr x7, sp ptr x8, loM v6):
   acc += pn*d1xd + ph*d2xd ; bias += (mm1*sx0 + mm2*sx1)*xd_blk
   Scalars (sN = vN.s[0]): d1xd=s7  mm1=s29  d2xd=s28  mm2=s30
   bias=s1  total=s5  xd_blk=s4  sx0=s31  sx1=s26  tmp=s27
   Dot temps v16-v27 only — v28-v31 free. */
.macro Q4_DOT
    ldr q16, [x5]
    ldr q17, [x5, #16]
    and v18.16b, v16.16b, v6.16b    /* nib0 */
    and v19.16b, v17.16b, v6.16b    /* nib1 */
    ushr v20.16b, v16.16b, #4       /* hi0 */
    ushr v21.16b, v17.16b, #4       /* hi1 */
    ldr q22, [x7]
    ldr q23, [x7, #16]
    ldr q24, [x7, #32]
    ldr q25, [x7, #48]
    movi v26.4s, #0
    sdot v26.4s, v18.16b, v22.16b   /* pn = nib0·xa */
    sdot v26.4s, v19.16b, v23.16b   /*    + nib1·xb */
    movi v27.4s, #0
    sdot v27.4s, v20.16b, v24.16b   /* ph = hi0·xc */
    sdot v27.4s, v21.16b, v25.16b   /*    + hi1·xd */
    scvtf v26.4s, v26.4s
    scvtf v27.4s, v27.4s
    fmla v0.4s, v26.4s, v7.s[0]     /* acc += pn * d1xd (s7 = v7.s[0]) */
    fmla v0.4s, v27.4s, v28.s[0]    /* acc += ph * d2xd (s28 = v28.s[0]) */
    /* bias += (mm1*sx0 + mm2*sx1) * xd_blk */
    ldrsh w16, [x8]
    ldrsh w17, [x8, #2]
    add w16, w16, w17
    scvtf s31, w16                   /* sx0 (v31.s[0], free) */
    ldrsh w16, [x8, #4]
    ldrsh w17, [x8, #6]
    add w16, w16, w17
    scvtf s26, w16                   /* sx1 (v26.s[0] = pn, consumed) */
    fmul s27, s29, s31               /* mm1*sx0 */
    fmadd s27, s30, s26, s27         /* + mm2*sx1 */
    fmul s27, s27, s4                /* * xd_blk */
    fadd s1, s1, s27                 /* bias += */
.endm

/* Gate-row dot body (g qs x10, u qs x11, shared xp x12, loM v6):
   gacc += gpn*gd1xd + gph*gd2xd ; then GU_DOT_U adds the u row.
   Scalars: gd1xd=s8 gmm1=s9 gd2xd=s10 gmm2=s11 (v8-v11.s[0], saved)
   Dot temps v16-v25, v28-v31 — v26/v27 + v8-v15 kept. */
.macro GU_DOT_G
    ldr q16, [x10]                   /* g0 */
    ldr q17, [x10, #16]              /* g1 */
    and v24.16b, v16.16b, v6.16b     /* gn0 */
    and v25.16b, v17.16b, v6.16b     /* gn1 */
    ushr v20.16b, v16.16b, #4        /* gh0 */
    ushr v21.16b, v17.16b, #4        /* gh1 */
    ldr q28, [x12]                   /* xa (shared) */
    ldr q29, [x12, #16]              /* xb */
    ldr q30, [x12, #32]              /* xc */
    ldr q31, [x12, #48]              /* xd */
    movi v22.4s, #0
    sdot v22.4s, v24.16b, v28.16b    /* gpn */
    sdot v22.4s, v25.16b, v29.16b
    movi v23.4s, #0
    sdot v23.4s, v20.16b, v30.16b    /* gph */
    sdot v23.4s, v21.16b, v31.16b
    scvtf v22.4s, v22.4s
    scvtf v23.4s, v23.4s
    fmla v0.4s, v22.4s, v8.s[0]      /* gacc += gpn * gd1xd (s8) */
    fmla v0.4s, v23.4s, v10.s[0]     /* gacc += gph * gd2xd (s10) */
.endm

.macro GU_DOT_U
    ldr q18, [x11]                   /* u0 */
    ldr q19, [x11, #16]              /* u1 */
    and v24.16b, v18.16b, v6.16b     /* un0 */
    and v25.16b, v19.16b, v6.16b     /* un1 */
    ushr v20.16b, v18.16b, #4        /* uh0 */
    ushr v21.16b, v19.16b, #4        /* uh1 */
    movi v22.4s, #0
    sdot v22.4s, v24.16b, v28.16b    /* upn */
    sdot v22.4s, v25.16b, v29.16b
    movi v23.4s, #0
    sdot v23.4s, v20.16b, v30.16b    /* uph */
    sdot v23.4s, v21.16b, v31.16b
    scvtf v22.4s, v22.4s
    scvtf v23.4s, v23.4s
    fmla v1.4s, v22.4s, v12.s[0]     /* uacc += upn * ud1xd (s12) */
    fmla v1.4s, v23.4s, v14.s[0]     /* uacc += uph * ud2xd (s14) */
.endm

    .text

/* int qma_q8k_available(void) */
    .global asm_qma_q8k_available
asm_qma_q8k_available:
    mov w0, #1
    ret

/* ------------------------------------------------------------------ */
/* quantize x (n floats, n % 256 == 0):                               */
/*   xd[i] = amax/127 (1.0 when all-zero) ; xq = clamp(round(x/xd))   */
/*   xsum = Σ xq per 16 lanes                                         */
/* in: x0=x, w1=n, x2=xq, x3=xd, x4=xsum                               */
/* ------------------------------------------------------------------ */
    .global asm_qma_q8k_quant
asm_qma_q8k_quant:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    lsr w9, w1, #8                   /* nb */
    cbz w9, .Lq8q_out
    mov x10, x0                      /* xs */
    mov x11, x2                      /* xq */
    mov x12, x3                      /* xd */
    mov x13, x4                      /* xsum */
.Lq8q_blk:
    movi v0.4s, #0
    mov w14, #64                     /* 64 quad loads over 256 lanes */
.Lq8q_amax:
    ldr q1, [x10], #16
    fabs v1.4s, v1.4s
    fmax v0.4s, v0.4s, v1.4s
    subs w14, w14, #1
    b.ne .Lq8q_amax
    fmaxp v0.4s, v0.4s, v0.4s
    fmaxp v0.4s, v0.4s, v0.4s       /* all lanes = amax */
    sub x10, x10, #1024             /* rewind: quant loop re-reads this block */
    mov w15, #0x42fe0000            /* 127.0f */
    fmov s1, w15
    fdiv s2, s0, s1                 /* s = amax/127 */
    mov w16, #0x3f800000            /* 1.0f */
    fmov s3, w16
    fcmp s2, #0.0
    fcsel s2, s2, s3, gt            /* s = amax>0 ? amax/127 : 1 */
    str s2, [x12], #4               /* xd[i] = s */
    fmov s4, s3
    fdiv s4, s4, s2                 /* inv = 1/s */
    dup v4.4s, v4.s[0]
    mov w17, #127
    dup v5.4s, w17
    mov w17, #-127
    dup v6.4s, w17
    mov w17, #16
.Lq8q_16:
    /* 16 lanes: 4 quads: x*inv, fcvtas, clamp [-127,127], narrow */
    ldr q10, [x10], #16
    fmul v10.4s, v10.4s, v4.4s
    fcvtas v10.4s, v10.4s
    ldr q11, [x10], #16
    fmul v11.4s, v11.4s, v4.4s
    fcvtas v11.4s, v11.4s
    ldr q12, [x10], #16
    fmul v12.4s, v12.4s, v4.4s
    fcvtas v12.4s, v12.4s
    ldr q13, [x10], #16
    fmul v13.4s, v13.4s, v4.4s
    fcvtas v13.4s, v13.4s
    smin v10.4s, v10.4s, v5.4s
    smax v10.4s, v10.4s, v6.4s
    smin v11.4s, v11.4s, v5.4s
    smax v11.4s, v11.4s, v6.4s
    smin v12.4s, v12.4s, v5.4s
    smax v12.4s, v12.4s, v6.4s
    smin v13.4s, v13.4s, v5.4s
    smax v13.4s, v13.4s, v6.4s
    sqxtn v14.4h, v10.4s
    sqxtn2 v14.8h, v11.4s
    sqxtn v15.8b, v14.8h
    sqxtn v16.4h, v12.4s
    sqxtn2 v16.8h, v13.4s
    sqxtn2 v15.16b, v16.8h
    str d15, [x11], #8
    mov d17, v15.d[1]
    str d17, [x11], #8
    saddlp v18.8h, v15.16b
    addp v18.8h, v18.8h, v18.8h
    addp v18.8h, v18.8h, v18.8h
    addp v18.8h, v18.8h, v18.8h
    smov w18, v18.h[0]
    strh w18, [x13], #2
    subs w17, w17, #1
    b.ne .Lq8q_16
    subs w9, w9, #1
    b.ne .Lq8q_blk
.Lq8q_out:
    ldp x29, x30, [sp], #16
    ret

/* ================================================================== */
/* Q4_K × q8: asm_q8k_dot_q4(b, xq, xd, xsum, cols)                   */
/*   x0=b, x1=xq, x2=xd, x3=xsum, w4=cols                              */
/* Per 256-lane block: 4 × 64-lane groups. Each group:                 */
/*   scales/min decode (q8k_scale_min_k4), SDOT nibbles, fma with      */
/*   d·sc·xd, min-bias subtracts dmin·m·xd·Σxq per 32-lane half.       */
/* Scalars (sN = vN.s[0]): d=s2 mn=s3 xd_blk=s4 total=s5 bias=s1      */
/*   d1xd=s7 mm1=s29 d2xd=s28 mm2=s30  (v28-v31 free in Q4_DOT)        */
/* ================================================================== */
    .global asm_q8k_dot_q4
asm_q8k_dot_q4:
    lsr w9, w4, #8                   /* nb = cols/256 */
    cbz w9, .Lq4_zero
    movi v6.16b, #0x0F               /* loM */
    fmov s5, wzr                     /* total = 0 (v5.s[0]) */
.Lq4_blk:
    add x5, x0, #16                  /* q = b[i].qs (offset 16) */
    add x6, x0, #4                   /* scales = b[i].scales (offset 4) */
    mov x7, x1                       /* xp = xq + i*256 */
    mov x8, x3                       /* sp = xsum + i*16 */
    ldr h24, [x0]                    /* d fp16 */
    fcvt s2, h24                     /* s2 = d (v2.s[0]) */
    ldr h24, [x0, #2]                /* dmin fp16 */
    fcvt s3, h24                     /* s3 = mn (v3.s[0]) */
    ldr s4, [x2]                     /* s4 = xd_blk (v4.s[0]) */
    ldr w10, [x6]                    /* scales[0..3] */
    ldr w11, [x6, #4]                /* scales[4..7] */
    ldr w12, [x6, #8]                /* scales[8..11] */
    movi v0.4s, #0                   /* acc */
    fmov s1, wzr                     /* bias = 0 */

    /* ---- group 0: j=0,1 (simple form) ---- */
    SCALE s7, w10, 0, s2             /* d1xd = d*s1        (v7.s[0]) */
    SCALE s29, w11, 0, s3            /* mm1 = mn*m1        (v29.s[0]) */
    SCALE s28, w10, 8, s2            /* d2xd = d*s2        (v28.s[0]) */
    SCALE s30, w11, 8, s3            /* mm2 = mn*m2        (v30.s[0]) */
    fmul s7, s7, s4                  /* d1xd *= xd_blk */
    fmul s28, s28, s4                /* d2xd *= xd_blk */
    Q4_DOT
    add x5, x5, #32
    add x7, x7, #64
    add x8, x8, #8

    /* ---- group 1: j=2,3 ---- */
    SCALE s7, w10, 16, s2
    SCALE s29, w11, 16, s3
    SCALE s28, w10, 24, s2
    SCALE s30, w11, 24, s3
    fmul s7, s7, s4
    fmul s28, s28, s4
    Q4_DOT
    add x5, x5, #32
    add x7, x7, #64
    add x8, x8, #8

    /* ---- group 2: j=4,5 (packed form) ---- */
    SCALE_P s7, w12, 0,  w10, 6, s2
    SCALE_P s29, w12, 4,  w11, 6, s3
    SCALE_P s28, w12, 8,  w10, 14, s2
    SCALE_P s30, w12, 12, w11, 14, s3
    fmul s7, s7, s4
    fmul s28, s28, s4
    Q4_DOT
    add x5, x5, #32
    add x7, x7, #64
    add x8, x8, #8

    /* ---- group 3: j=6,7 ---- */
    SCALE_P s7, w12, 16, w10, 22, s2
    SCALE_P s29, w12, 20, w11, 22, s3
    SCALE_P s28, w12, 24, w10, 30, s2
    SCALE_P s30, w12, 28, w11, 30, s3
    fmul s7, s7, s4
    fmul s28, s28, s4
    Q4_DOT
    add x5, x5, #32
    add x7, x7, #64
    add x8, x8, #8

    /* total += vaddvq_f32(acc) - bias */
    faddp v0.4s, v0.4s, v0.4s
    faddp s0, v0.2s
    fsub s0, s0, s1
    fadd s5, s5, s0
    /* advance block */
    add x0, x0, #144                 /* sizeof(block_q4_K) */
    add x1, x1, #256
    add x2, x2, #4
    add x3, x3, #32
    subs w9, w9, #1
    b.ne .Lq4_blk
    fmov s0, s5
    ret
.Lq4_zero:
    fmov s0, wzr
    ret

/* ================================================================== */
/* Q6_K × q8: asm_q8k_dot_q6(b, xq, xd, cols)                         */
/*   x0=b, x1=xq, x2=xd, w3=cols                                       */
/* Per 256-lane block: 2 × 128-lane halves; each half: 8 × 16-lane     */
/* groups of 6-bit weights (raw-32) SDOT against x, scaled by int8 sc. */
/* Scalars: d=s2 xd_blk=s3 total=s4 tmp=s5                            */
/* ================================================================== */
    .global asm_q8k_dot_q6
asm_q8k_dot_q6:
    lsr w3, w3, #8                   /* nb = cols/256 */
    cbz w3, .Lq6_zero
    movi v22.16b, #0x0F              /* loM */
    movi v23.16b, #0x03
    movi v24.16b, #0x0C
    movi v25.16b, #0x30
    movi v26.16b, #0xC0
    movi v27.16b, #0x20              /* 32 for the -32 bias fold */
    fmov s4, wzr                     /* total = 0 (v4.s[0]) */
.Lq6_blk:
    mov x4, x0                       /* ql = b[i].ql (offset 0) */
    add x5, x0, #128                 /* qh = b[i].qh (offset 128) */
    add x6, x0, #192                 /* sc = b[i].scales (offset 192) */
    mov x7, x1                       /* xp = xq + i*256 */
    ldr h10, [x0, #208]              /* d fp16 -- h10: v24 holds the 0x0C
                                        mask constant and must NOT be
                                        clobbered by this load */
    fcvt s2, h10                     /* s2 = d (v2.s[0]) */
    ldr s3, [x2]                     /* s3 = xd_blk (v3.s[0]) */
    movi v0.4s, #0                   /* acc (s32x4) */
    mov w8, #2                       /* half counter */
.Lq6_half:
    ldr q16, [x4]                    /* ql0 */
    ldr q17, [x4, #16]               /* ql1 */
    ldr q18, [x4, #32]               /* ql2 */
    ldr q19, [x4, #48]               /* ql3 */
    ldr q20, [x5]                    /* qh0 */
    ldr q21, [x5, #16]               /* qh1 */
    ldr d31, [x6]                    /* sc[0..7] (8 bytes) */

    /* g0: raw0 = (ql0 & 0x0F) | ((qh0 & 3) << 4) ; w = raw0 - 32 */
    and v28.16b, v16.16b, v22.16b
    and v29.16b, v20.16b, v23.16b
    shl v29.16b, v29.16b, #4
    orr v28.16b, v28.16b, v29.16b
    sub v28.16b, v28.16b, v27.16b
    ldr q29, [x7]
    movi v30.4s, #0
    sdot v30.4s, v28.16b, v29.16b
    smov w9, v31.b[0]
    dup v28.4s, w9
    mla v0.4s, v30.4s, v28.s[0]

    /* g1: raw1 = (ql1 & 0x0F) | ((qh1 & 3) << 4) */
    and v28.16b, v17.16b, v22.16b
    and v29.16b, v21.16b, v23.16b
    shl v29.16b, v29.16b, #4
    orr v28.16b, v28.16b, v29.16b
    sub v28.16b, v28.16b, v27.16b
    ldr q29, [x7, #16]
    movi v30.4s, #0
    sdot v30.4s, v28.16b, v29.16b
    smov w9, v31.b[1]
    dup v28.4s, w9
    mla v0.4s, v30.4s, v28.s[0]

    /* g2: raw2 = (ql2 & 0x0F) | ((qh0 & 0xC) << 2) */
    and v28.16b, v18.16b, v22.16b
    and v29.16b, v20.16b, v24.16b
    shl v29.16b, v29.16b, #2
    orr v28.16b, v28.16b, v29.16b
    sub v28.16b, v28.16b, v27.16b
    ldr q29, [x7, #32]
    movi v30.4s, #0
    sdot v30.4s, v28.16b, v29.16b
    smov w9, v31.b[2]
    dup v28.4s, w9
    mla v0.4s, v30.4s, v28.s[0]

    /* g3: raw3 = (ql3 & 0x0F) | ((qh1 & 0xC) << 2) */
    and v28.16b, v19.16b, v22.16b
    and v29.16b, v21.16b, v24.16b
    shl v29.16b, v29.16b, #2
    orr v28.16b, v28.16b, v29.16b
    sub v28.16b, v28.16b, v27.16b
    ldr q29, [x7, #48]
    movi v30.4s, #0
    sdot v30.4s, v28.16b, v29.16b
    smov w9, v31.b[3]
    dup v28.4s, w9
    mla v0.4s, v30.4s, v28.s[0]

    /* g4: raw4 = (ql0 >> 4) | (qh0 & 0x30) */
    ushr v28.16b, v16.16b, #4
    and v29.16b, v20.16b, v25.16b
    orr v28.16b, v28.16b, v29.16b
    sub v28.16b, v28.16b, v27.16b
    ldr q29, [x7, #64]
    movi v30.4s, #0
    sdot v30.4s, v28.16b, v29.16b
    smov w9, v31.b[4]
    dup v28.4s, w9
    mla v0.4s, v30.4s, v28.s[0]

    /* g5: raw5 = (ql1 >> 4) | (qh1 & 0x30) */
    ushr v28.16b, v17.16b, #4
    and v29.16b, v21.16b, v25.16b
    orr v28.16b, v28.16b, v29.16b
    sub v28.16b, v28.16b, v27.16b
    ldr q29, [x7, #80]
    movi v30.4s, #0
    sdot v30.4s, v28.16b, v29.16b
    smov w9, v31.b[5]
    dup v28.4s, w9
    mla v0.4s, v30.4s, v28.s[0]

    /* g6: raw6 = (ql2 >> 4) | ((qh0 & 0xC0) >> 2) */
    ushr v28.16b, v18.16b, #4
    and v29.16b, v20.16b, v26.16b
    ushr v29.16b, v29.16b, #2
    orr v28.16b, v28.16b, v29.16b
    sub v28.16b, v28.16b, v27.16b
    ldr q29, [x7, #96]
    movi v30.4s, #0
    sdot v30.4s, v28.16b, v29.16b
    smov w9, v31.b[6]
    dup v28.4s, w9
    mla v0.4s, v30.4s, v28.s[0]

    /* g7: raw7 = (ql3 >> 4) | ((qh1 & 0xC0) >> 2) */
    ushr v28.16b, v19.16b, #4
    and v29.16b, v21.16b, v26.16b
    ushr v29.16b, v29.16b, #2
    orr v28.16b, v28.16b, v29.16b
    sub v28.16b, v28.16b, v27.16b
    ldr q29, [x7, #112]
    movi v30.4s, #0
    sdot v30.4s, v28.16b, v29.16b
    smov w9, v31.b[7]
    dup v28.4s, w9
    mla v0.4s, v30.4s, v28.s[0]

    add x4, x4, #64                  /* ql += 64 */
    add x5, x5, #32                  /* qh += 32 */
    add x6, x6, #8                   /* sc += 8 */
    add x7, x7, #128                 /* xp += 128 */
    subs w8, w8, #1
    b.ne .Lq6_half

    /* total += d * xd_blk * (float)vaddvq_s32(acc) */
    addv s0, v0.4s
    fmov w9, s0
    scvtf s5, w9                     /* sum (v5.s[0]) */
    fmul s5, s5, s2                  /* sum * d */
    fmul s5, s5, s3                  /* * xd_blk */
    fadd s4, s4, s5                  /* total += */
    add x0, x0, #210                 /* sizeof(block_q6_K) */
    add x1, x1, #256
    add x2, x2, #4
    subs w3, w3, #1
    b.ne .Lq6_blk
    fmov s0, s4
    ret
.Lq6_zero:
    fmov s0, wzr
    ret

/* ================================================================== */
/* Fused gate+up Q4_K × q8: asm_q8k_dot_gateup(g, u, xq, xd, xsum,    */
/*   n, gate, up) — one pass over the shared quantized activation for  */
/*   both rows. x0=g, x1=u, x2=xq, x3=xd, x4=xsum, w5=n, x6=gate, x7=up */
/*                                                                     */
/* Scalars (sN = vN.s[0]): gd=s2 gmn=s3 ud=s4 umn=s5 xd_blk=s7        */
/*   gd1xd=s8 gmm1=s9 gd2xd=s10 gmm2=s11 (v8-v11, saved)              */
/*   ud1xd=s12 umm1=s13 ud2xd=s14 umm2=s15 (v12-v15, saved)           */
/*   gbias/ubias/gtot/utot on the stack (frame [sp+8..23])            */
/* ================================================================== */
    .global asm_q8k_dot_gateup
asm_q8k_dot_gateup:
    lsr w5, w5, #8                   /* nb = cols/256 */
    cbz w5, .Lgu_zero
    /* prologue: save callee-saved v8-v15 (used as scalar hosts) */
    sub sp, sp, #80
    stp d8, d9, [sp, #16]
    stp d10, d11, [sp, #32]
    stp d12, d13, [sp, #48]
    stp d14, d15, [sp, #64]
    movi v6.16b, #0x0F               /* loM */
    fmov s16, wzr                    /* gtot = 0 (stack slot) */
    fmov s17, wzr                    /* utot = 0 */
    fmov s18, wzr                    /* gbias = 0 */
    fmov s19, wzr                    /* ubias = 0 */
    str s16, [sp, #0]
    str s17, [sp, #4]
    str s18, [sp, #8]
    str s19, [sp, #12]
.Lgu_blk:
    add x8, x0, #4                   /* g scales */
    add x9, x1, #4                   /* u scales */
    add x10, x0, #16                 /* g qs */
    add x11, x1, #16                 /* u qs */
    mov x12, x2                      /* xp */
    mov x13, x4                      /* sp */
    ldr h24, [x0]                    /* gd fp16 */
    fcvt s2, h24                     /* s2 = gd (v2.s[0]) */
    ldr h24, [x0, #2]                /* gmn fp16 */
    fcvt s3, h24                     /* s3 = gmn (v3.s[0]) */
    ldr h24, [x1]                    /* ud fp16 */
    fcvt s4, h24                     /* s4 = ud (v4.s[0]) */
    ldr h24, [x1, #2]                /* umn fp16 */
    fcvt s5, h24                     /* s5 = umn (v5.s[0]) */
    ldr s7, [x3]                     /* s7 = xd_blk (v7.s[0]) */
    movi v0.4s, #0                   /* gacc */
    movi v1.4s, #0                   /* uacc */
    /* per-block bias reset (the C zeroes gbias/ubias per block) */
    fmov s18, wzr
    fmov s19, wzr
    str s18, [sp, #8]
    str s19, [sp, #12]

    /* ---- group 0: j=0,1 ---- */
    /* g row */
    ldr w14, [x8]
    ldr w15, [x8, #4]
    ldr w16, [x8, #8]
    SCALE s8, w14, 0, s2             /* gd1xd (v8.s[0]) */
    SCALE s9, w15, 0, s3             /* gmm1 (v9.s[0]) */
    SCALE s10, w14, 8, s2            /* gd2xd (v10.s[0]) */
    SCALE s11, w15, 8, s3            /* gmm2 (v11.s[0]) */
    fmul s8, s8, s7
    fmul s10, s10, s7
    GU_DOT_G
    /* sx0/sx1 shared, gbias */
    ldrsh w17, [x13]
    ldrsh w18, [x13, #2]
    add w17, w17, w18
    scvtf s26, w17                   /* sx0 (v26.s[0], free) */
    ldrsh w17, [x13, #4]
    ldrsh w18, [x13, #6]
    add w17, w17, w18
    scvtf s27, w17                   /* sx1 (v27.s[0], free) */
    fmul s24, s9, s26                /* gmm1*sx0 */
    fmadd s24, s11, s27, s24         /* + gmm2*sx1 */
    fmul s24, s24, s7
    ldr s16, [sp, #8]                /* gbias */
    fadd s16, s16, s24
    str s16, [sp, #8]
    /* u row */
    ldr w14, [x9]
    ldr w15, [x9, #4]
    ldr w16, [x9, #8]
    SCALE s12, w14, 0, s4            /* ud1xd (v12.s[0]) */
    SCALE s13, w15, 0, s5            /* umm1 (v13.s[0]) */
    SCALE s14, w14, 8, s4            /* ud2xd (v14.s[0]) */
    SCALE s15, w15, 8, s5            /* umm2 (v15.s[0]) */
    fmul s12, s12, s7
    fmul s14, s14, s7
    GU_DOT_U
    fmul s25, s13, s26               /* umm1*sx0 */
    fmadd s25, s15, s27, s25         /* + umm2*sx1 */
    fmul s25, s25, s7
    ldr s16, [sp, #12]               /* ubias */
    fadd s16, s16, s25
    str s16, [sp, #12]
    add x10, x10, #32
    add x11, x11, #32
    add x12, x12, #64
    add x13, x13, #8

    /* ---- group 1: j=2,3 ---- */
    ldr w14, [x8]
    ldr w15, [x8, #4]
    ldr w16, [x8, #8]
    SCALE s8, w14, 16, s2
    SCALE s9, w15, 16, s3
    SCALE s10, w14, 24, s2
    SCALE s11, w15, 24, s3
    fmul s8, s8, s7
    fmul s10, s10, s7
    GU_DOT_G
    ldrsh w17, [x13]
    ldrsh w18, [x13, #2]
    add w17, w17, w18
    scvtf s26, w17
    ldrsh w17, [x13, #4]
    ldrsh w18, [x13, #6]
    add w17, w17, w18
    scvtf s27, w17
    fmul s24, s9, s26
    fmadd s24, s11, s27, s24
    fmul s24, s24, s7
    ldr s16, [sp, #8]
    fadd s16, s16, s24
    str s16, [sp, #8]
    ldr w14, [x9]
    ldr w15, [x9, #4]
    ldr w16, [x9, #8]
    SCALE s12, w14, 16, s4
    SCALE s13, w15, 16, s5
    SCALE s14, w14, 24, s4
    SCALE s15, w15, 24, s5
    fmul s12, s12, s7
    fmul s14, s14, s7
    GU_DOT_U
    fmul s25, s13, s26
    fmadd s25, s15, s27, s25
    fmul s25, s25, s7
    ldr s16, [sp, #12]
    fadd s16, s16, s25
    str s16, [sp, #12]
    add x10, x10, #32
    add x11, x11, #32
    add x12, x12, #64
    add x13, x13, #8

    /* ---- group 2: j=4,5 (packed) ---- */
    ldr w14, [x8]
    ldr w15, [x8, #4]
    ldr w16, [x8, #8]
    SCALE_P s8, w16, 0,  w14, 6, s2
    SCALE_P s9, w16, 4,  w15, 6, s3
    SCALE_P s10, w16, 8,  w14, 14, s2
    SCALE_P s11, w16, 12, w15, 14, s3
    fmul s8, s8, s7
    fmul s10, s10, s7
    GU_DOT_G
    ldrsh w17, [x13]
    ldrsh w18, [x13, #2]
    add w17, w17, w18
    scvtf s26, w17
    ldrsh w17, [x13, #4]
    ldrsh w18, [x13, #6]
    add w17, w17, w18
    scvtf s27, w17
    fmul s24, s9, s26
    fmadd s24, s11, s27, s24
    fmul s24, s24, s7
    ldr s16, [sp, #8]
    fadd s16, s16, s24
    str s16, [sp, #8]
    ldr w14, [x9]
    ldr w15, [x9, #4]
    ldr w16, [x9, #8]
    SCALE_P s12, w16, 0,  w14, 6, s4
    SCALE_P s13, w16, 4,  w15, 6, s5
    SCALE_P s14, w16, 8,  w14, 14, s4
    SCALE_P s15, w16, 12, w15, 14, s5
    fmul s12, s12, s7
    fmul s14, s14, s7
    GU_DOT_U
    fmul s25, s13, s26
    fmadd s25, s15, s27, s25
    fmul s25, s25, s7
    ldr s16, [sp, #12]
    fadd s16, s16, s25
    str s16, [sp, #12]
    add x10, x10, #32
    add x11, x11, #32
    add x12, x12, #64
    add x13, x13, #8

    /* ---- group 3: j=6,7 (packed) ---- */
    ldr w14, [x8]
    ldr w15, [x8, #4]
    ldr w16, [x8, #8]
    SCALE_P s8, w16, 16, w14, 22, s2
    SCALE_P s9, w16, 20, w15, 22, s3
    SCALE_P s10, w16, 24, w14, 30, s2
    SCALE_P s11, w16, 28, w15, 30, s3
    fmul s8, s8, s7
    fmul s10, s10, s7
    GU_DOT_G
    ldrsh w17, [x13]
    ldrsh w18, [x13, #2]
    add w17, w17, w18
    scvtf s26, w17
    ldrsh w17, [x13, #4]
    ldrsh w18, [x13, #6]
    add w17, w17, w18
    scvtf s27, w17
    fmul s24, s9, s26
    fmadd s24, s11, s27, s24
    fmul s24, s24, s7
    ldr s16, [sp, #8]
    fadd s16, s16, s24
    str s16, [sp, #8]
    ldr w14, [x9]
    ldr w15, [x9, #4]
    ldr w16, [x9, #8]
    SCALE_P s12, w16, 16, w14, 22, s4
    SCALE_P s13, w16, 20, w15, 22, s5
    SCALE_P s14, w16, 24, w14, 30, s4
    SCALE_P s15, w16, 28, w15, 30, s5
    fmul s12, s12, s7
    fmul s14, s14, s7
    GU_DOT_U
    fmul s25, s13, s26
    fmadd s25, s15, s27, s25
    fmul s25, s25, s7
    ldr s16, [sp, #12]
    fadd s16, s16, s25
    str s16, [sp, #12]
    add x10, x10, #32
    add x11, x11, #32
    add x12, x12, #64
    add x13, x13, #8

    /* gtot += vaddvq_f32(gacc) - gbias ; utot += vaddvq_f32(uacc) - ubias */
    /* NOTE: s24/s25 (v24/v25.s[0]) are free here — v24/v25 are dot temps,
       and v26/v27 are sx0/sx1 which were consumed. s1 is uacc lane 0 —
       must NOT be used as a temp before uacc reduction! */
    faddp v0.4s, v0.4s, v0.4s
    faddp s0, v0.2s
    ldr s24, [sp, #8]               /* gbias */
    fsub s0, s0, s24
    ldr s24, [sp, #0]               /* gtot */
    fadd s24, s24, s0
    str s24, [sp, #0]
    faddp v1.4s, v1.4s, v1.4s
    faddp s0, v1.2s
    ldr s25, [sp, #12]              /* ubias */
    fsub s0, s0, s25
    ldr s25, [sp, #4]               /* utot */
    fadd s25, s25, s0
    str s25, [sp, #4]
    /* advance block */
    add x0, x0, #144
    add x1, x1, #144
    add x2, x2, #256
    add x3, x3, #4
    add x4, x4, #32
    subs w5, w5, #1
    b.ne .Lgu_blk
    ldr s16, [sp, #0]                /* gtot */
    ldr s17, [sp, #4]                /* utot */
    str s16, [x6]                    /* *gate = gtot */
    str s17, [x7]                    /* *up = utot */
    /* epilogue: restore v8-v15 */
    ldp d14, d15, [sp, #64]
    ldp d12, d13, [sp, #48]
    ldp d10, d11, [sp, #32]
    ldp d8, d9, [sp, #16]
    add sp, sp, #80
    ret
.Lgu_zero:
    movi v0.4s, #0
    str s0, [x6]                     /* *gate = 0 */
    str s0, [x7]                     /* *up = 0 */
    ret

/* ================================================================== */
/* public dispatchers (tail-call into the kernels)                     */
/* ================================================================== */
    .global asm_qma_q8k_dot
/* float asm_qma_q8k_dot(row, wtype, xq, xd, xsum, n) */
asm_qma_q8k_dot:
    cmp w1, #GGML_TYPE_Q4_K
    b.eq 1f
    cmp w1, #GGML_TYPE_Q6_K
    b.eq 2f
    fmov s0, wzr
    ret
1:  mov x1, x2                       /* xq */
    mov x2, x3                       /* xd */
    mov x3, x4                       /* xsum */
    mov w4, w5                       /* n */
    b asm_q8k_dot_q4
2:  mov x1, x2                       /* xq */
    mov x2, x3                       /* xd */
    mov w3, w5                       /* n */
    b asm_q8k_dot_q6

    .global asm_qma_q8k_gateup
/* void asm_qma_q8k_gateup(g, u, xq, xd, xsum, n, gate, up) — same ABI as
   asm_q8k_dot_gateup, so it's a pure tail-call. */
asm_qma_q8k_gateup:
    b asm_q8k_dot_gateup

    .section .note.GNU-stack,"",%progbits
