# qma — qwen mobile agent

qma is a single C99 binary that runs a large MoE language model as an
autonomous coding agent entirely on-device, on a phone. There is no Python
runtime, no HTTP server, no separate inference process — the model engine,
the agent loop, and the tools the agent calls all live in one executable.

qma is also **self-hosting**: the binary embeds its own source tree (and
anything the agent collects). The agent can read and modify its own code
through an internal filesystem (`/internal/`), test changes with
`self_build()`, and on a clean exit qma recompiles itself from the edited
source, smoke-tests the result, and carries the new binary — plus the whole
conversation and the internal tree — forward as the next generation.

It's tuned for `qwen3.6:35b:a3b` (the `qwen35moe` architecture: 30 recurrent
"gated delta net" layers plus 10 full-attention layers, a MoE FFN with 256
experts routed top-8 plus a shared expert) running on a Samsung Galaxy S25
Ultra under Termux. The hyperparameters in `qma.h` are hardcoded for that specific
model/device pairing. Adapting qma to a different large MoE model or a
different phone means editing those constants (embedding size, layer count,
head counts, expert count/top-k, FFN dims, rope settings) to match the new
model's config, and re-tuning the thermal thresholds and worker counts in
`thermal.c` for the new device's sensors and core layout. The overall
approach — streaming experts from disk, tiered KV memory, embedding sessions
in the binary — carries over to any model that's bigger than the device's RAM.

## What it does

Because the model doesn't fit in memory, most of what qma does is manage a
model that's bigger than the device running it:

**Expert streaming.** Only a fraction of the MoE's 256 experts per layer are
active on any given token (top-8 + shared). Rather than mmapping the whole
model and hoping the OS's page cache does something sensible, qma reads
expert weights straight from disk into a bounded cache sized in MB
(`--ecache`, default 1024). The cache tracks how often and how recently each
expert has been used, evicts the coldest ones under pressure, and issues
reads ahead of time on background threads so the matmuls for expert N+1 don't
have to wait for expert N+1's weights to arrive from disk. Setting `--ecache
0` switches to plain mmap instead.

**Tiered context memory.** The KV cache — the running memory of everything
said so far — is stored in a fixed-size ring buffer, but the model behaves as
if its memory is unbounded. Every token position has a running score (an
exponential moving average of how much attention it has received). The first
few tokens are always kept (a small fixed "sink"), the most recent several
thousand tokens are kept at full precision, a wider window behind that is
kept at lower precision, and a small set of high-scoring tokens from
anywhere in the conversation — "heavy hitters" — get copied out of the ring
into a protected slot so they survive being overwritten by newer tokens as
long as they keep scoring well. Tokens that fall out of every tier simply
aren't attended to. In effect, a long conversation keeps only what mattered.

**Agent archive — the model's own active memory.** On top of the tiered
cache, the agent can pin anything it wants permanent: the `memory_write` /
`memory_append` / `memory_list` / `memory_delete` / `memory_clear` tools
compute real K/V for the text and store it in a protected arena that
attention reads on **every token** — immune to eviction, ring wrap, and
restarts (entries are rebuilt from `memory.json` at boot, so the file is
human-editable too). This is the model's own always-visible memory, not a
file system: it never needs a tool to read it (it attends to it every step),
and the tools only mutate it. It gives a long-running agent consolidation
and coherence — it writes its task state and decisions down, so it never
loses the thread when the window rolls.

**Quantized KV storage.** Every key/value vector is stored in two precisions
at once when it's written (4-bit and 2-bit), so moving a token between the
full-precision and reduced-precision tiers is just a pointer change, not a
re-computation.

**Sessions embedded in the binary.** By default, running qma without
`--session` starts from a temp directory and, on a clean exit, writes a new
timestamped copy of the executable with the conversation's KV cache, model
state, and salience scores appended to its tail. Running that new binary
picks the conversation back up — there's no separate session folder to
track, back up, or lose. Passing `--session <dir>` instead uses a normal
persistent directory (`kv.bin`, `state.bin`, `salience.bin`) if you'd rather
manage sessions that way, or run several in parallel.

**Self-hosting.** The same tail that carries the session also carries the
agent's own source tree — `src/`, the Makefile, the README, plus any tools
or data the agent collects — as an embedded blob (`intern.c`). At boot it is
extracted to `<session>/internal/` and mounted as `/internal/` for the
agent's file tools (`$QMA_INTERNAL` holds the real path for `bash`). On a
clean exit qma recompiles itself from `/internal/src` (same flags as the
Makefile), smoke-tests the fresh binary, and if it passes, embeds the whole
internal tree + the conversation into the next generation binary
(`qma-gen<N>-<timestamp>`). If the rebuild fails, the current binary is kept
and the compiler errors land in `/internal/rebuild.log` so the agent can
fix its own code next boot. Generations are pruned to the newest one (the
original base binary is always kept).

**Thermal governor.** A background thread watches CPU and battery
temperature every few seconds and quietly shrinks the worker thread pool
(and, if things get hot enough, restricts work to the phone's efficiency
cores) so a long agent run doesn't throttle or overheat the device. It has
no effect on the model's output — only on how many threads are doing the
math at once — and logs level changes to stderr only.

**Native Qwen tool calling.** qma uses qwen3.6's own tool-call format
(`<tool_call><function=NAME><parameter=KEY>value</parameter></function></tool_call>`,
per the model's chat template and the vLLM qwen3 parser). The engine never
constrains the model's token stream (an early grammar mask backfired — when
every top-k candidate was masked the sampler injected an illegal char and
ran away). Reasoning follows the model's native conventions too: an
unpaired `<tool_call>` token implicitly ends the `<think>` block, and the
model may emit **several** complete calls in one message (its native
multi-step behavior) — the engine runs them all and folds the results back
as one `<tool_response>` turn. The parser is tolerant of the model's known
format drift (dropped `<tool_call>` wrappers, improvised close tags) but
never executes a half-typed call mid-stream; malformed calls are dropped
and fed back as an error so the model re-issues them.

**Document workers (LFM sub-agents).** qma also embeds a second, tiny
engine — LFM2.5-2.6B (arch `lfm2`), ported from ma3 under `src/lfm/` — as
a worker pool the main agent can delegate to. Instead of reading a whole
file into its own context, the agent spawns a worker (`worker_spawn` with a
path or inline text) that holds the document in its KV cache and answers
questions about it (`worker_ask`): what line is something on, is it in
here, summarize a section, compare several documents. The worker weights
are loaded **once and shared**; each worker is only a KV runstate — a base
(the worker's role prompt + document, processed once) plus **ephemeral**
query layers on top (every question gets a clean KV on the same base) or a
**long-term** conversation (follow-ups have context). `worker_close` frees
one, `worker_list` shows them. The worker source lives in `/internal/src/lfm/`
so the agent can read and modify its own sub-agents.

**GPU prefill offload (OpenCL, Adreno).** The dense projections (attention
+ ssm) and the Q6_K lm head run their prefill matmuls (`T > 1`) on the GPU;
decode and the streamed MoE experts stay on the CPU (bandwidth-bound anyway,
and it spreads the heat). The kernels are the llama.cpp production
"noshuffle" Q4_K/Q6_K GEMMs (8×4 thread tile, vector-scalar dequant, float
accumulators) adapted for qma's prefill flow, with qma's weights repacked
once (lazily, per tensor) into the noshuffle layout — including qma's
i8mm-ordered Q4_K nibble packing and the LFM-ordered Q6_K packing. Two
Adreno driver bugs shaped the implementation: vector component stores
corrupt the accumulation, so the GEMMs write whole-vector tiles that a
transpose kernel reorders into `C[T][N]`; and scale arithmetic in-kernel
miscompiles, so per-sub-block scale/min values are precomputed on the host.
The numerical gate (`tests/cltest.c`) proves every shape matches the CPU
path within fp16 noise (~0.1%) before the offload is used. Vendor OpenCL
libs are copied into `work/cl/` (the Android linker namespace blocks
`/vendor/lib64`); `QMA_NO_CL=1` falls back to pure CPU.

## The agent

qma's system prompt frames the model as an autonomous coding agent working
from a phone: given a goal, it keeps working — calling tools, checking its
own work by re-reading files or re-running commands, tracking multi-step work
with a todo list — until the goal is actually done, only stopping to ask the
user when a real decision is needed.

Tools available to the model: `pwd`, `ls`, `find`, `grep`, `read`, `write`,
`edit` (exact-text-match replacement, several edits per call), `replace_lines`,
`insert_lines`, `delete_lines`, `bash`, `enter` (change directory — the
running qma binary physically moves to the new directory, so the next
self-hosted generation is written into the project the agent moved to),
`todo_write` / `todo_complete` / `todo_remove`, `write_diary` / `read_diary`,
`ask_user`, `self_build` (compile + smoke-test the agent's own source),
`memory_write` / `memory_append` / `memory_list` / `memory_delete` /
`memory_clear` (the agent's own always-attended KV memory), and the worker
pool: `worker_spawn` / `worker_ask` / `worker_close` / `worker_list` (LFM
document-analyst sub-agents).

A turn works like this: the user's message is added to the running KV cache,
the model generates a response — thinking shown dimmed (the think block ends
at `</think>` **or** at the first unpaired `<tool_call>`, per the model's
native behavior), regular content shown normally, tool calls shown as they're
emitted. Every complete `<tool_call>` block is parsed (several may appear in
one message — multi-step is native); the engine runs them all and feeds the
results back in as `<tool_response>` blocks before the model generates
again. This repeats until the model responds with plain text and no tool
call, which ends the turn.

The agent is self-aware about its own code: `/internal/` is its internal
filesystem (source in `/internal/src/`, version history in
`/internal/VERSIONS.md`, last failed build's errors in
`/internal/rebuild.log`). It may modify its own source, collect tools and
data under `/internal/tools/` and `/internal/data/` to carry forever, and
verify its edits at any time with `self_build()` before qma commits them at
exit.

## Building

```
make          # builds ./qma
make test     # builds and runs the unit tests
make clean
```

Needs `clang`, C99, links against `-lm -pthread`. The default `CFLAGS`
include `-mcpu=native`, so the build is tuned for whatever machine compiles
it — cross-compiling for a different aarch64 device means adjusting that.
The included `qma` binary is an ARM64 Android executable built for Termux.

## Running

```
qma [options]              interactive agent
qma -p "prompt"            one-shot, non-interactive
```

| Flag | Meaning |
|---|---|
| `-m <path>` | path to the `.gguf` model file (else `~/.qma/config`, else it prompts) |
| `-p <prompt>` | run one prompt and exit, instead of an interactive session |
| `-t <n>` | number of threads (default 8) |
| `-c <n>` | ring context size in tokens (default 65536) |
| `--temp <f>` | sampling temperature (default 0.8) |
| `--repeat <f>` | repeat penalty (default 1.1) |
| `--eos-penalty <f>` | penalty applied to control-token logits (default 1.5) |
| `--no-think` | turn off the model's `<think>` block |
| `--ecache <mb>` | expert cache size in MB (default 1024; `0` uses mmap instead) |
| `--session <dir>` | use a persistent session directory instead of a binary snapshot |
| `--workdir <dir>` | working directory for the agent's file/shell tools |
| `--reset` | clear the current session's context and start over |
| `--no-color` | disable ANSI colors in the terminal output |
| `--check-align` | print whether the model file is 4K-aligned (O_DIRECT-ready) and exit |
| `--embed-internal <tree> <outfile>` | append the internal-tree blob to a freshly built binary (used by `make`) |
| `--export-internal <dir>` | extract this binary's embedded internal tree to a directory |
| `-h` / `--help` | show usage |

Interactive commands: `/exit`, `/quit`, `/reset`, `/clear`.

### Model path & ESC

- The model path is remembered in `~/.qma/config` (written on every launch).
  Resolution order: `-m` flag → config embedded in a snapshot binary →
  `~/.qma/config` → interactive prompt. If the chosen file doesn't exist,
  qma reprompts for a path; in non-interactive mode it errors out.
- The `you> ` prompt is a small line editor on a real terminal: Left/Right
  move the cursor (UTF-8 aware), Up/Down walk your prompt history,
  Backspace/Delete edit at the cursor, Home/End jump, Enter submits.
  Non-tty input (pipes, scripts) falls back to plain line reads.
- Press `ESC` during generation to cancel the current turn and return to the
  `you> ` prompt (the partial reply stays in the context, closed off cleanly).
  `Ctrl-C` still shuts the program down.


## Environment variables

These toggle debug output or opt out of a subsystem that's normally on:

| Variable | Effect |
|---|---|
| `QMA_INTERNAL=<path>` | real path of the mounted internal tree (set at boot; use it in `bash`) |
| `QMA_NOALIGN=1` | skip the one-time 4K-alignment repack of the model file |
| `QMA_NOTHERMAL=1` | turn off the thermal governor |
| `QMA_NO_CL=1` | turn off the OpenCL GPU offload (pure CPU) |
| `QMA_NOYALIS=1` | turn off predictive expert prefetching |
| `QMA_NOEVICT=1` | turn off tiered context memory (attend to everything instead) |
| `QMA_NOQ8K=1` | turn off the int8 quantized-activation fast path |
| `QMA_KVQ=0` | store the KV cache in full precision instead of quantized |
| `QMA_WORKER_MODEL=<path>` | LFM worker model file (default `~/projects/models/lfm2.5:2.6b:Q4_K_M.gguf`) |
| `QMA_WORKER_THREADS=<n>` | worker eval threads (default 4) |
| `QMA_TRACE=1` | print detailed per-layer trace output |
| `QMA_PFTRACE=1` | print prefetch trace output |
| `QMA_ECPROF=1` | print expert-cache hit/miss statistics |
| `QMA_TIMING=1` | print a timing breakdown per token |
| `QMA_EXPOFFS=1` | print resolved expert file offsets |
| `QMA_DUMP_L0=<path>` / `QMA_DUMP_L1=<path>` | dump layer-0 intermediate values to a file |

## Layout

```
src/
  qma.h, gguf.c        model definition, GGUF file loading, hyperparameters
  nn.c                 forward pass: attention, gated delta net, MoE FFN, tiered memory
  quants.c, q8k.c      weight/activation quantization formats and dot products
  cl.c/.h              OpenCL GPU offload (Adreno) — prefill matmuls
  kvq.c/.h             quantized KV cache storage
  ecache.c/.h          expert streaming cache
  sampler.c            temperature / top-k / top-p / repeat-penalty sampling
  tokenizer.c, unicode_tables.h   tokenizer
  agent.c              CLI, session handling, the agent turn loop
  tools.c/.h           tool definitions and implementations
  toolparse.c/.h       parses <tool_call> output into structured calls
  selfctx.c/.h         embeds/extracts a session in the binary's own file
  intern.c/.h          embedded internal tree: embed/extract, rebuild, version log
  thermal.c/.h         thermal governor
  json.c/.h            JSON parser/encoder
  worker.c/.h          LFM document-worker pool (base KV + ephemeral queries)
  lfm/                 LFM2.5-2.6B engine (namespaced port from ma3):
    lfm.h, lfm_gguf.c, lfm_tokenizer.c, lfm_nn.c, lfm_sampler.c
                       Q4_0/Q4_K/Q6_K matmuls, resident RAM, worker model
```

The internal tree embedded in every binary (mounted at `/internal/`):

```
internal/
  src/                  the engine source (what qma recompiles from at exit)
  Makefile, README.md
  tools/, data/         collected by the agent, carried forward forever
  VERSIONS.md           generation history
  rebuild.log           last failed build's compiler errors (if any)
```

## Tests

`make test` builds and runs the unit suites under `tests/` (each links only
the source files it needs, without loading a model):

- `tests/test_toolparse.c` — parser contract: a complete valid call parses
  to a name + JSON args; an open block returns 0 without advancing (the
  engine breaks, never spins); a complete-but-malformed block returns -1 and
  is dropped, never executed. Streaming never auto-closes a half-typed call.
- `tests/test_intern.c` — internal-tree blob round trip: embed a tree,
  extract it, verify every file byte-for-byte, probe the generation serial.
- `tests/cltest.c` — GPU numerical gate: quantizes random weights to
  Q4_K/Q6_K, runs the OpenCL GEMM (T>1) and compares every output against
  the CPU dot path — the Swiftlet-style proof that the GPU matches the CPU
  within fp16 noise (~0.1%). Prints `GPU UNAVAILABLE` and exits 0 when
  OpenCL can't start (the engine falls back to CPU in that case).

There are also headless engine tests under `work/` (they load a real model,
so they are not part of `make test`):
- `work/test_lfm.c` — the namespaced LFM engine loads the Q4_K/Q6_K model
  and generates sanely.
- `work/test_worker.c` — worker pool E2E: spawn a document worker, ask
  ephemeral questions (clean KV each), long-term follow-ups, no-tools role.

The session-snapshot suite is planned.


##DISCLAIMER:
This is a pre-alpha build so it might have some bugs, it does work, gets 
around 3tps steady, and is minimal on battery usage, but past that no guarentees. 

