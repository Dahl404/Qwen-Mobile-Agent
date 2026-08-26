/* quants.s — Q4_K / Q6_K dequant + fp32 dot kernels (aarch64 NEON).
 *
 * Faithful asm port of quants.c. Weight formats (GGUF / llama.cpp ground
 * truth):
 *   Q4_K: 256 weights/super-block. d,dmin fp16 + 12 B 6-bit scales +
 *         128 B nibbles. weight = d*sc*(nibble) - dmin*m, per 32-sub-block.
 *   Q6_K: 256 weights/super-block. 6-bit quants (ql/qh) + 8-bit scales.
 *         weight = d*sc*(q-32).
 *
 * The fp32 dots use NEON fmla with the min-bias fold: for a 32-lane group,
 *   Σ (d*nib - m)*x = d*Σ(nib*x) - m*Σx
 * so each group needs one product-sum and one x-sum accumulator.
 */
.include "qma_defs.s"

    .text

/* ------------------------------------------------------------------ */
/* fp16 helpers (aarch64: single FCVT each way)                        */
/* ------------------------------------------------------------------ */
/* float qma_half_to_float(uint16_t h) */
    .global asm_qma_half_to_float
asm_qma_half_to_float:
    dup v0.4h, w0
    fcvt s0, h0
    ret

/* uint16_t qma_float_to_half(float x) */
    .global asm_qma_float_to_half
asm_qma_float_to_half:
    fcvt h0, s0
    umov w0, v0.h[0]
    ret

/* ------------------------------------------------------------------ */
/* 6-bit scale/min unpack for Q4_K (get_scale_min_k4).                */
/* in:  w0 = j, x1 = scales[12]                                       */
/* out: w0 = d, w1 = m                                                */
/* ------------------------------------------------------------------ */
    .global asm_q4_scale_min_k4
    .type asm_q4_scale_min_k4, %function
asm_q4_scale_min_k4:
    cmp w0, #4
    b.ge .Lq4sm_ge4
    ldrb w2, [x1, w0, uxtw]          /* q[j] */
    add w3, w0, #4
    ldrb w3, [x1, w3, uxtw]          /* q[j+4] */
    and w0, w2, #63
    and w1, w3, #63
    ret
.Lq4sm_ge4:
    /* NOTE: w2,w3,w4,w6 only — w5 is the callers' live `is` reg (caller-saved) */
    add w2, w0, #4
    ldrb w2, [x1, w2, uxtw]          /* q[j+4] */
    sub w3, w0, #4
    ldrb w3, [x1, w3, uxtw]          /* q[j-4] */
    ldrb w4, [x1, w0, uxtw]          /* q[j]   */
    and w6, w2, #0xF
    lsr w3, w3, #6
    orr w0, w6, w3, lsl #4           /* d */
    lsr w6, w2, #4
    lsr w4, w4, #6
    orr w1, w6, w4, lsl #4           /* m */
    ret

/* ------------------------------------------------------------------ */
/* dequantize_row_q4_K:  y[i] = d*sc*(nib)-dmin*m, 32 lanes/group     */
/* in: x0 = W, x1 = y, x2 = k (multiple of 256)                       */
/* ------------------------------------------------------------------ */
    .global asm_dequantize_row_q4_K
asm_dequantize_row_q4_K:
    stp x29, x30, [sp, #-64]!
    mov x29, sp
    stp d8, d9, [sp, #16]
    stp d10, d11, [sp, #32]
    stp d12, d13, [sp, #48]
    lsr w9, w2, #8                   /* nb = k/256 */
    cbz w9, .Ldq4_out
    mov x10, x0
    mov x11, x1
.Ldq4_blk:
    ldrh w3, [x10]                   /* W->d */
    dup v3.4h, w3
    fcvt s8, h3                      /* d */
    ldrh w4, [x10, #2]
    dup v4.4h, w4
    fcvt s9, h4                      /* min */
    add x13, x10, #4                 /* scales */
    add x14, x10, #16                /* qs */
    mov w5, #0                       /* is */
    mov w15, #4                      /* group counter */
.Ldq4_grp:
    mov w0, w5
    mov x1, x13
    bl asm_q4_scale_min_k4               /* w0=s1 w1=m1 */
    ucvtf s10, w0
    fmul s10, s8, s10                /* d1 */
    ucvtf s11, w1
    fmul s11, s9, s11                /* m1 */
    add w0, w5, #1
    mov x1, x13
    bl asm_q4_scale_min_k4               /* w0=s2 w1=m2 */
    ucvtf s12, w0
    fmul s12, s8, s12                /* d2 */
    ucvtf s13, w1
    fmul s13, s9, s13                /* m2 */
    /* 32 lanes: low nibbles (q[0:16], q[16:32]) * d1 - m1, then high * d2 - m2 */
    ldr q1, [x14]
    ldr q2, [x14, #16]
    movi v29.16b, #0x0f
    and v3.16b, v1.16b, v29.16b      /* lo0 */
    and v4.16b, v2.16b, v29.16b      /* lo1 */
    ushr v5.16b, v1.16b, #4          /* hi0 */
    ushr v6.16b, v2.16b, #4          /* hi1 */
    /* --- lo0: 16 nibbles -> f32, write y[0..15] --- */
    uxtl v7.8h, v3.8b
    uxtl2 v16.8h, v3.16b
    uxtl v17.4s, v7.4h
    uxtl2 v18.4s, v7.8h
    uxtl v19.4s, v16.4h
    uxtl2 v20.4s, v16.8h
    ucvtf v17.4s, v17.4s
    ucvtf v18.4s, v18.4s
    ucvtf v19.4s, v19.4s
    ucvtf v20.4s, v20.4s
    dup v21.4s, v10.s[0]             /* d1 */
    dup v22.4s, v11.s[0]             /* m1 */
    fmul v17.4s, v17.4s, v21.4s
    fmul v18.4s, v18.4s, v21.4s
    fmul v19.4s, v19.4s, v21.4s
    fmul v20.4s, v20.4s, v21.4s
    fsub v17.4s, v17.4s, v22.4s
    fsub v18.4s, v18.4s, v22.4s
    fsub v19.4s, v19.4s, v22.4s
    fsub v20.4s, v20.4s, v22.4s
    stp q17, q18, [x11]
    stp q19, q20, [x11, #32]
    /* --- lo1: 16 nibbles -> f32, write y[16..31] --- */
    uxtl v7.8h, v4.8b
    uxtl2 v16.8h, v4.16b
    uxtl v17.4s, v7.4h
    uxtl2 v18.4s, v7.8h
    uxtl v19.4s, v16.4h
    uxtl2 v20.4s, v16.8h
    ucvtf v17.4s, v17.4s
    ucvtf v18.4s, v18.4s
    ucvtf v19.4s, v19.4s
    ucvtf v20.4s, v20.4s
    fmul v17.4s, v17.4s, v21.4s
    fmul v18.4s, v18.4s, v21.4s
    fmul v19.4s, v19.4s, v21.4s
    fmul v20.4s, v20.4s, v21.4s
    fsub v17.4s, v17.4s, v22.4s
    fsub v18.4s, v18.4s, v22.4s
    fsub v19.4s, v19.4s, v22.4s
    fsub v20.4s, v20.4s, v22.4s
    stp q17, q18, [x11, #64]
    stp q19, q20, [x11, #96]
    /* --- hi0: 16 nibbles -> f32, write y[32..47] --- */
    uxtl v7.8h, v5.8b
    uxtl2 v16.8h, v5.16b
    uxtl v17.4s, v7.4h
    uxtl2 v18.4s, v7.8h
    uxtl v19.4s, v16.4h
    uxtl2 v20.4s, v16.8h
    ucvtf v17.4s, v17.4s
    ucvtf v18.4s, v18.4s
    ucvtf v19.4s, v19.4s
    ucvtf v20.4s, v20.4s
    dup v21.4s, v12.s[0]             /* d2 */
    dup v22.4s, v13.s[0]             /* m2 */
    fmul v17.4s, v17.4s, v21.4s
    fmul v18.4s, v18.4s, v21.4s
    fmul v19.4s, v19.4s, v21.4s
    fmul v20.4s, v20.4s, v21.4s
    fsub v17.4s, v17.4s, v22.4s
    fsub v18.4s, v18.4s, v22.4s
    fsub v19.4s, v19.4s, v22.4s
    fsub v20.4s, v20.4s, v22.4s
    stp q17, q18, [x11, #128]
    stp q19, q20, [x11, #160]
    /* --- hi1: 16 nibbles -> f32, write y[48..63] --- */
    uxtl v7.8h, v6.8b
    uxtl2 v16.8h, v6.16b
    uxtl v17.4s, v7.4h
    uxtl2 v18.4s, v7.8h
    uxtl v19.4s, v16.4h
    uxtl2 v20.4s, v16.8h
    ucvtf v17.4s, v17.4s
    ucvtf v18.4s, v18.4s
    ucvtf v19.4s, v19.4s
    ucvtf v20.4s, v20.4s
    fmul v17.4s, v17.4s, v21.4s
    fmul v18.4s, v18.4s, v21.4s
    fmul v19.4s, v19.4s, v21.4s
    fmul v20.4s, v20.4s, v21.4s
    fsub v17.4s, v17.4s, v22.4s
    fsub v18.4s, v18.4s, v22.4s
    fsub v19.4s, v19.4s, v22.4s
    fsub v20.4s, v20.4s, v22.4s
    stp q17, q18, [x11, #192]
    stp q19, q20, [x11, #224]
    add x14, x14, #32
    add w5, w5, #2
    add x11, x11, #256
    subs w15, w15, #1
    b.ne .Ldq4_grp
    add x10, x10, #144               /* next block */
    subs w9, w9, #1
    b.ne .Ldq4_blk
.Ldq4_out:
    ldp d12, d13, [sp, #48]
    ldp d10, d11, [sp, #32]
    ldp d8, d9, [sp, #16]
    ldp x29, x30, [sp], #64
    ret

/* ------------------------------------------------------------------ */
/* dot_q4_K_f32:  out = Σ_i W[i]*x[i], W = Q4_K, x = fp32, n%256==0  */
/* in: x0 = W, x1 = x, w2 = n                                         */
/* out: s0 = result                                                   */
/* ------------------------------------------------------------------ */
    .global asm_dot_q4_K_f32
asm_dot_q4_K_f32:
    stp x29, x30, [sp, #-64]!
    mov x29, sp
    stp d8, d9, [sp, #16]
    stp d10, d11, [sp, #32]
    stp d12, d13, [sp, #48]
    lsr w9, w2, #8                   /* nb */
    movi v0.4s, #0                   /* total acc */
    cbz w9, .Lq4d_out
    mov x10, x0
    mov x11, x1
.Lq4d_blk:
    ldrh w3, [x10]
    dup v3.4h, w3
    fcvt s8, h3                      /* d */
    ldrh w4, [x10, #2]
    dup v4.4h, w4
    fcvt s9, h4                      /* min */
    add x13, x10, #4                 /* scales */
    add x14, x10, #16                /* qs */
    mov w5, #0                       /* is */
    mov w15, #4                      /* groups */
.Lq4d_grp:
    mov w0, w5
    mov x1, x13
    bl asm_q4_scale_min_k4
    ucvtf s10, w0
    fmul s10, s8, s10                /* d1 */
    ucvtf s11, w1
    fmul s11, s9, s11                /* m1v */
    add w0, w5, #1
    mov x1, x13
    bl asm_q4_scale_min_k4
    ucvtf s12, w0
    fmul s12, s8, s12                /* d2 */
    ucvtf s13, w1
    fmul s13, s9, s13                /* m2v */
    ldr q1, [x14]
    ldr q2, [x14, #16]
    movi v29.16b, #0x0f
    and v3.16b, v1.16b, v29.16b
    and v4.16b, v2.16b, v29.16b
    ushr v5.16b, v1.16b, #4
    ushr v6.16b, v2.16b, #4
    /* accA/xsA for lo (xp[0..31]), accB/xsB for hi (xp[32..63]) */
    movi v7.4s, #0
    movi v16.4s, #0
    movi v17.4s, #0
    movi v18.4s, #0
    /* lo0 */
    uxtl v19.8h, v3.8b
    uxtl2 v20.8h, v3.16b
    uxtl v21.4s, v19.4h
    uxtl2 v22.4s, v19.8h
    uxtl v23.4s, v20.4h
    uxtl2 v24.4s, v20.8h
    ucvtf v21.4s, v21.4s
    ucvtf v22.4s, v22.4s
    ucvtf v23.4s, v23.4s
    ucvtf v24.4s, v24.4s
    ldr q25, [x11]
    ldr q26, [x11, #16]
    ldr q27, [x11, #32]
    ldr q28, [x11, #48]
    fmla v7.4s, v21.4s, v25.4s
    fmla v7.4s, v22.4s, v26.4s
    fmla v7.4s, v23.4s, v27.4s
    fmla v7.4s, v24.4s, v28.4s
    fadd v16.4s, v16.4s, v25.4s
    fadd v16.4s, v16.4s, v26.4s
    fadd v16.4s, v16.4s, v27.4s
    fadd v16.4s, v16.4s, v28.4s
    /* lo1 */
    uxtl v19.8h, v4.8b
    uxtl2 v20.8h, v4.16b
    uxtl v21.4s, v19.4h
    uxtl2 v22.4s, v19.8h
    uxtl v23.4s, v20.4h
    uxtl2 v24.4s, v20.8h
    ucvtf v21.4s, v21.4s
    ucvtf v22.4s, v22.4s
    ucvtf v23.4s, v23.4s
    ucvtf v24.4s, v24.4s
    ldr q25, [x11, #64]
    ldr q26, [x11, #80]
    ldr q27, [x11, #96]
    ldr q28, [x11, #112]
    fmla v7.4s, v21.4s, v25.4s
    fmla v7.4s, v22.4s, v26.4s
    fmla v7.4s, v23.4s, v27.4s
    fmla v7.4s, v24.4s, v28.4s
    fadd v16.4s, v16.4s, v25.4s
    fadd v16.4s, v16.4s, v26.4s
    fadd v16.4s, v16.4s, v27.4s
    fadd v16.4s, v16.4s, v28.4s
    /* hi0 (xp[32..63]) */
    uxtl v19.8h, v5.8b
    uxtl2 v20.8h, v5.16b
    uxtl v21.4s, v19.4h
    uxtl2 v22.4s, v19.8h
    uxtl v23.4s, v20.4h
    uxtl2 v24.4s, v20.8h
    ucvtf v21.4s, v21.4s
    ucvtf v22.4s, v22.4s
    ucvtf v23.4s, v23.4s
    ucvtf v24.4s, v24.4s
    ldr q25, [x11, #128]
    ldr q26, [x11, #144]
    ldr q27, [x11, #160]
    ldr q28, [x11, #176]
    fmla v17.4s, v21.4s, v25.4s
    fmla v17.4s, v22.4s, v26.4s
    fmla v17.4s, v23.4s, v27.4s
    fmla v17.4s, v24.4s, v28.4s
    fadd v18.4s, v18.4s, v25.4s
    fadd v18.4s, v18.4s, v26.4s
    fadd v18.4s, v18.4s, v27.4s
    fadd v18.4s, v18.4s, v28.4s
    /* hi1 */
    uxtl v19.8h, v6.8b
    uxtl2 v20.8h, v6.16b
    uxtl v21.4s, v19.4h
    uxtl2 v22.4s, v19.8h
    uxtl v23.4s, v20.4h
    uxtl2 v24.4s, v20.8h
    ucvtf v21.4s, v21.4s
    ucvtf v22.4s, v22.4s
    ucvtf v23.4s, v23.4s
    ucvtf v24.4s, v24.4s
    ldr q25, [x11, #192]
    ldr q26, [x11, #208]
    ldr q27, [x11, #224]
    ldr q28, [x11, #240]
    fmla v17.4s, v21.4s, v25.4s
    fmla v17.4s, v22.4s, v26.4s
    fmla v17.4s, v23.4s, v27.4s
    fmla v17.4s, v24.4s, v28.4s
    fadd v18.4s, v18.4s, v25.4s
    fadd v18.4s, v18.4s, v26.4s
    fadd v18.4s, v18.4s, v27.4s
    fadd v18.4s, v18.4s, v28.4s
    /* group totals: d1*(ΣaccA) - m1v*(ΣxsA) ; d2*(ΣaccB) - m2v*(ΣxsB) */
    faddp v7.4s, v7.4s, v7.4s
    faddp v7.4s, v7.4s, v7.4s
    faddp v16.4s, v16.4s, v16.4s
    faddp v16.4s, v16.4s, v16.4s
    faddp v17.4s, v17.4s, v17.4s
    faddp v17.4s, v17.4s, v17.4s
    faddp v18.4s, v18.4s, v18.4s
    faddp v18.4s, v18.4s, v18.4s
    fmul s7, s7, s10
    fmul s16, s16, s11
    fmul s17, s17, s12
    fmul s18, s18, s13
    fsub s7, s7, s16
    fsub s17, s17, s18
    fadd s7, s7, s17
    fadd s0, s0, s7
    add x14, x14, #32
    add w5, w5, #2
    add x11, x11, #256
    subs w15, w15, #1
    b.ne .Lq4d_grp
    add x10, x10, #144
    subs w9, w9, #1
    b.ne .Lq4d_blk
.Lq4d_out:
    ldp d12, d13, [sp, #48]
    ldp d10, d11, [sp, #32]
    ldp d8, d9, [sp, #16]
    ldp x29, x30, [sp], #64
    ret

/* ------------------------------------------------------------------ */
/* dequantize_row_q6_K:  y[i] = d*sc*(q-32)                           */
/* in: x0 = W, x1 = y, x2 = k (multiple of 256)                       */
/* scalar (correctness-first; the q8k SDOT path is the hot one)       */
/* ------------------------------------------------------------------ */
    .global asm_dequantize_row_q6_K
asm_dequantize_row_q6_K:
    stp x29, x30, [sp, #-64]!
    mov x29, sp
    stp d8, d9, [sp, #16]
    stp x19, x20, [sp, #32]
    stp x21, x22, [sp, #48]
    lsr w7, w2, #8                   /* nb (w7: caller-saved, not used below) */
    cbz w7, .Ldq6_out
    mov x10, x0
    mov x11, x1
.Ldq6_blk:
    ldrh w3, [x10, #208]
    dup v3.4h, w3
    fcvt s8, h3                      /* d */
    add x12, x10, #0                 /* ql */
    add x13, x10, #128               /* qh */
    add x14, x10, #192               /* scales */
    mov w15, #2                      /* 2 chunks of 128 lanes */
.Ldq6_chunk:
    mov w6, #0                       /* l = 0..31 */
.Ldq6_lane:
    /* is = l < 16 ? 0 : 1 */
    cmp w6, #16
    csinc w8, wzr, wzr, lt
    /* q1 = (ql[l]&0xF) | ((qh[l]&3)<<4) - 32 */
    ldrb w9, [x12, w6, uxtw]         /* ql[l] */
    and w16, w9, #0xF
    ldrb w17, [x13, w6, uxtw]        /* qh[l] */
    and w18, w17, #3
    orr w16, w16, w18, lsl #4
    sub w16, w16, #32
    /* q2 = (ql[l+32]&0xF) | (((qh[l]>>2)&3)<<4) - 32 */
    add w19, w6, #32
    ldrb w20, [x12, w19, uxtw]
    and w21, w20, #0xF
    lsr w18, w17, #2
    and w18, w18, #3
    orr w21, w21, w18, lsl #4
    sub w21, w21, #32
    /* q3 = (ql[l]>>4) | (((qh[l]>>4)&3)<<4) - 32 */
    lsr w9, w9, #4
    lsr w18, w17, #4
    and w18, w18, #3
    orr w9, w9, w18, lsl #4
    sub w9, w9, #32
    /* q4 = (ql[l+32]>>4) | (((qh[l]>>6)&3)<<4) - 32 */
    lsr w18, w17, #6
    and w18, w18, #3
    lsr w20, w20, #4
    orr w18, w20, w18, lsl #4
    sub w18, w18, #32
    /* scales sc[is+0], sc[is+2], sc[is+4], sc[is+6] */
    add w22, w8, #0
    ldrsb w0, [x14, w22, uxtw]       /* int8 scale (signed) */
    add w22, w8, #2
    ldrsb w1, [x14, w22, uxtw]
    add w22, w8, #4
    ldrsb w2, [x14, w22, uxtw]
    add w22, w8, #6
    ldrsb w3, [x14, w22, uxtw]
    /* y[l]   = d*sc[is+0]*q1 */
    scvtf s1, w16
    scvtf s2, w0
    fmul s2, s8, s2
    fmul s1, s2, s1
    add x16, x11, w6, uxtw #2
    str s1, [x16]
    /* y[l+32] = d*sc[is+2]*q2 */
    scvtf s3, w21
    scvtf s4, w1
    fmul s4, s8, s4
    fmul s3, s4, s3
    add x16, x11, w19, uxtw #2
    str s3, [x16]
    /* y[l+64] = d*sc[is+4]*q3 */
    scvtf s5, w9
    scvtf s6, w2
    fmul s6, s8, s6
    fmul s5, s6, s5
    add x16, x11, #256
    add x16, x16, w6, uxtw #2
    str s5, [x16]
    /* y[l+96] = d*sc[is+6]*q4 */
    scvtf s7, w18
    scvtf s16, w3
    fmul s16, s8, s16
    fmul s7, s16, s7
    add x16, x11, #384
    add x16, x16, w6, uxtw #2
    str s7, [x16]
    add w6, w6, #1
    cmp w6, #32
    b.lt .Ldq6_lane
    /* chunk done: ql += 64, qh += 32, sc += 8, y += 128 floats */
    add x12, x12, #64
    add x13, x13, #32
    add x14, x14, #8
    add x11, x11, #512
    subs w15, w15, #1
    b.ne .Ldq6_chunk
    add x10, x10, #210
    subs w7, w7, #1
    b.ne .Ldq6_blk
.Ldq6_out:
    ldp x21, x22, [sp, #48]
    ldp x19, x20, [sp, #32]
    ldp d8, d9, [sp, #16]
    ldp x29, x30, [sp], #64
    ret

/* ------------------------------------------------------------------ */
/* dot_q6_K_f32:  out = Σ_i W[i]*x[i], W = Q6_K, x = fp32, n%256==0  */
/* scalar accumulation into s0 (fallback path)                        */
/* ------------------------------------------------------------------ */
    .global asm_dot_q6_K_f32
asm_dot_q6_K_f32:
    stp x29, x30, [sp, #-64]!
    mov x29, sp
    stp d8, d9, [sp, #16]
    stp x19, x20, [sp, #32]
    stp x21, x22, [sp, #48]
    lsr w7, w2, #8
    movi v0.4s, #0
    cbz w7, .Lq6d_out
    mov x10, x0
    mov x11, x1
.Lq6d_blk:
    ldrh w3, [x10, #208]
    dup v3.4h, w3
    fcvt s8, h3                      /* d */
    add x12, x10, #0                 /* ql */
    add x13, x10, #128               /* qh */
    add x14, x10, #192               /* scales */
    mov w15, #2
.Lq6d_chunk:
    mov w6, #0
.Lq6d_lane:
    cmp w6, #16
    cmp w6, #16
    csinc w8, wzr, wzr, lt           /* is = 0/1 */
    ldrb w9, [x12, w6, uxtw]
    and w16, w9, #0xF
    ldrb w17, [x13, w6, uxtw]
    and w18, w17, #3
    orr w16, w16, w18, lsl #4
    sub w16, w16, #32
    add w19, w6, #32
    ldrb w20, [x12, w19, uxtw]
    and w21, w20, #0xF
    lsr w18, w17, #2
    and w18, w18, #3
    orr w21, w21, w18, lsl #4
    sub w21, w21, #32
    lsr w9, w9, #4
    lsr w18, w17, #4
    and w18, w18, #3
    orr w9, w9, w18, lsl #4
    sub w9, w9, #32
    lsr w18, w17, #6
    and w18, w18, #3
    lsr w20, w20, #4
    orr w18, w20, w18, lsl #4
    sub w18, w18, #32
    add w22, w8, #0
    ldrsb w0, [x14, w22, uxtw]       /* int8 scale (signed) */
    add w22, w8, #2
    ldrsb w1, [x14, w22, uxtw]
    add w22, w8, #4
    ldrsb w2, [x14, w22, uxtw]
    add w22, w8, #6
    ldrsb w3, [x14, w22, uxtw]
    /* acc += d*sc*q*xp for the 4 quants */
    scvtf s1, w16
    scvtf s2, w0
    fmul s2, s8, s2
    fmul s1, s2, s1
    ldr s3, [x11, w6, uxtw #2]
    fmadd s0, s1, s3, s0
    scvtf s1, w21
    scvtf s2, w1
    fmul s2, s8, s2
    fmul s1, s2, s1
    add x16, x11, w19, uxtw #2
    ldr s3, [x16]
    fmadd s0, s1, s3, s0
    scvtf s1, w9
    scvtf s2, w2
    fmul s2, s8, s2
    fmul s1, s2, s1
    add x16, x11, #256
    add x16, x16, w6, uxtw #2
    ldr s3, [x16]
    fmadd s0, s1, s3, s0
    scvtf s1, w18
    scvtf s2, w3
    fmul s2, s8, s2
    fmul s1, s2, s1
    add x16, x11, #384
    add x16, x16, w6, uxtw #2
    ldr s3, [x16]
    fmadd s0, s1, s3, s0
    add w6, w6, #1
    cmp w6, #32
    b.lt .Lq6d_lane
    add x12, x12, #64
    add x13, x13, #32
    add x14, x14, #8
    add x11, x11, #512
    subs w15, w15, #1
    b.ne .Lq6d_chunk
    add x10, x10, #210
    subs w7, w7, #1
    b.ne .Lq6d_blk
.Lq6d_out:
    ldp x21, x22, [sp, #48]
    ldp x19, x20, [sp, #32]
    ldp d8, d9, [sp, #16]
    ldp x29, x30, [sp], #64
    ret

    .section .note.GNU-stack,"",%progbits
