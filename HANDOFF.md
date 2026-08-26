# HANDOFF — Q2_K_XL gibberish bug: ROOT CAUSE FOUND (2026-08-26)

## TL;DR — the bug was qma's GGML type-id table, not the file
`Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf` (Unsloth Dynamic UD recipe) was NEVER
corrupt. llama-cli was right all along. qma.h hardcoded the GGML type
enum with two WRONG ids:

| raw id | qma.h believed | REAL llama.cpp | qma block size | real |
|---|---|---|---|---|
| 18 | Q3_KXS (Q3_K layout) | **IQ3_XXS** | 110 B | **98 B** |
| 23 | IQ2_S | **IQ4_XS** | 82 B | **136 B** |

(real enum: IQ2_S=22, IQ4_XS=23, IQ3_XXS=18; verify: ggml/include/ggml.h
in ~/llama.cpp)

The Q2 file's expert DOWN slabs are IQ3_XXS (most layers) and IQ4_XS
(blk.34/38/39). With the wrong block size the engine computed the wrong
per-expert slab stride AND decoded each block with the wrong layout ⇒
garbage weights everywhere. Id-18 layers produced finite-but-wrong
garbage (gibberish text, no NaN); id-23 layers (blk.34/38/39) hit NaN
bit patterns ⇒ the visible "layer-34 poison".

Q4_K_M works because it only uses ids 12/14 (Q4_K/Q6_K), which match in
both tables.

## Decisive byte proof (tests/mtp_probe.c + tests/retype.c, header-only)
Raw file, header offsets, expert 42's slab:
- blk.0  down id 18 as IQ3_XXS(98B): dmin=3.5e-05 dmax=2.5e-04, 0 bad, 0 NaN
- blk.0  down id 18 as Q3_K(110B):   503 blocks d>100, 88 NaN   ← garbage
- blk.34 down id 23 as IQ4_XS(136B): dmin=-1.7e-05 dmax=2.3e-05, 0 bad, 0 NaN
- blk.34 down id 23 as IQ2_S(82B):   848 blocks d>100, 86 NaN   ← garbage

## What was fixed (committed in src/, uncommitted)
1. src/qma.h: enum corrected (IQ3_XXS=18, IQ2_S=22, IQ4_XS=23) + added
   block_iq3_xxs (98 B) + block_iq4_xs (136 B) structs + qma_blk_size
   entries. Old comment claiming "Q3_KXS id 18" removed.
2. src/gguf.c: both type-size switches (loader + gguf_scan) now handle
   IQ3_XXS / IQ4_XS with real block sizes.
3. src/iq2tables.h: added iq3xxs_grid[256] (uint32) + kvalues_iq4nl[16].
4. src/qkerns.c: dequantize_row_iq3_xxs / dequantize_row_iq4_xs,
   dot_iq3_xxs_f32 / dot_iq4_xs_f32 (portable), qma_dot_iq3_xxs_q8k /
   qma_dot_iq4_xs_q8k (NEON SDOT) — ported verbatim from ref/ggml-quants.c
   + ref/ggml-common.h (byte-identical to ~/llama.cpp current).
5. src/nn.c: dot_w_f32 dispatch + token-embd dequant switch got the new
   types. src/q8k.c: qma_q8k_dot dispatch got the new types.
6. QMA_TYPES per-layer print in gguf.c references the new ids.

## Old HANDOFF conclusions that were WRONG (ignore)
- "File is corrupt / ref-dequant also gives huge values": the earlier
  analysis pread'd slabs at the WRONG STRIDE (82/110 B/block vs real
  136/98), so bad_expert.bin was misaligned garbage. Garbage in, garbage out.
- "20% of blocks corrupt": misdecode artifact, not file damage.
- "Poison only at blk.34/38": id-23 layers are 34/38/39 (blk.39 is an
  ATTN layer with IQ4_XS down).

## Verify after rebuild
1. `QMA_NANCHECK=1 QMA_NOALIGN=1 ./qma -m <raw Q2 gguf> -p "Hello"` — expect
   NO NANCHECK lines (previously il=34 poisoned on first token).
2. Generation should be coherent text now.
3. tests/mtp_probe.c (header-only, engine-exact ids) is committed as a tool;
   tests/retype.c decodes a slab as multiple candidate types.
4. Differential: compare top-k logits vs llama-cli on same prompt
   (tests/logitdump.c pattern) once the engine runs clean.

## Open items (from the earlier discussion, unchanged)
- MTP head grafting: blk.40 tensors exist ONLY in Q2 (20 tensors, verified);
   engine still has no MTP execution path.
- Tiered expert serving: indexing parity CONFIRMED (733 shared tensors,
   byte-identical names+dims); now that Q2 decodes correctly the tiered
   plan is unblocked.
- Disk: 10 G free; models 46 GB. Q2 file is fine — no redownload needed.

## 2026-08-26 FINAL — root cause fully closed (the .4k was the garbage source)

The Q2 model through the agent now works: "Hello! How can I help you today?"
(system prompt ingested clean, 0 NANCHECK, 89% ecache hit rate, 2.0 tok/s).

Root cause chain:
1. qma.h had wrong GGML type ids (18=Q3_KXS→real IQ3_XXS 98B; 23=IQ2_S→real
   IQ4_XS 136B). Fixed + kernels added + verified bit-exact vs llama ref.
2. THE ACTUAL GARBAGE SOURCE: the .4k was repacked 2026-08-25 with the OLD
   wrong block sizes — IQ4_XS copied at 82 B/block instead of 136, so every
   expert down-slab in the .4k was truncated/misaligned → NaN at layer 34 →
   zero logits → the 4.240#1%& symbol stream. The raw .gguf was always fine
   (why llama-cli worked). The stale .4k passed the old validator (it checked
   against the same wrong sizes) and was auto-reused by qma_align_model.
3. Fixed: deleted the bad .4k, regenerated with corrected repacker (12.6 GB
   in 31s, verified all tensors), config points at the fresh .4k.

Storage cleanup (user-approved): deleted 21 .qma-* session dirs (~12.75 GB),
core (640 MB), work/remote-* (~3.3 GB), src.zip/log.out. Disk 100%→87% free.
qma_align_model re-run produces a CORRECT .4k now — no manual intervention.

State: model decode fixed + verified (mini: "a city known for its art and
culture"; agent: "Hello! How can I help you today?"), Q4 no regression.
