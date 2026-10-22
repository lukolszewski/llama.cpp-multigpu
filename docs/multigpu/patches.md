# Patch index

Every downstream optimization should be one identifiable commit, so that the branch history reads as
`upstream llama.cpp + patch 1 + patch 2 + ...` and any single patch can be measured, generalized,
upstreamed, or deleted on its own.

Base: `df03399b8` (`opencl: add A8 Q4_0 mm binary kernel support (#28268)` — the upstream revision
that `master` and `multigpu` diverge from). Delta: **47 commits, 39 files, +4707/−441** before the
CI/documentation commits added on top (`multigpu` = `ngram-groups` @ `e4054f726`, the configuration that
runs in production on machine-01 since 2026-10-05). Patches 1–12 are the decode-pipeline seed; 13–30 add
the continuous per-sequence decode groups and prefill-while-decoding; 31–41 the asynchronous prompt-cache
transfers, drain-free checkpoints and output-slot fixes; 42–47 the n-gram lookup speculation that works
with the groups.

| # | commit | date | area | what it does | switch | upstream ref |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `b22bc5b68` | 2026-09-11 | qwen4exp / KV / CUDA FA | fixes CUDA decode depth-decay at long context; reworks the indexer-cache/KV path in `llama-memory-hybrid-idx` (559 lines) and `qwen4exp` attention | built in (no opt-out) | re: [ggml-org/llama.cpp#28734](https://github.com/ggml-org/llama.cpp/issues/28734) |
| 2 | `4d6ef5ad4` | 2026-09-12 | qwen4exp attention | compact-gather attention for decode so a **quantized** KV cache keeps the sparse speedup instead of falling back to dense f16 | built in | re: #28734; relies on the sparse-FA path merged upstream in [#27970](https://github.com/ggml-org/llama.cpp/pull/27970) |
| 3 | `b75e380e0` | 2026-09-20 | server slots | keeps busy slots on contiguous sequence ids by migrating a busy sequence into an idle slot's sequence, so several sessions decode inside one ubatch | `--seq-compact` / `--no-seq-compact` (default on), `LLAMA_ARG_SEQ_COMPACT` | — |
| 4 | `02e136bef` | 2026-09-21 | server scheduling | prefill admission policy: bound how many slots prefill inside one batch and how many "long" prompts run concurrently; exposes waiting metrics | `--prefill-max-partial` (1), `--prefill-long-threshold` (8192), `--prefill-max-long` (1) | — |
| 5 | `c8832a5f0` | 2026-09-21 | scheduler / CUDA | keeps ubatches overlapping across GPUs for the qwen4exp hybrid (pipeline parallelism), plus the small CUDA/alloc/argsort support changes it needs | `LLAMA_PIPELINE_PARALLEL` (`1` keeps PP on through tensor overrides, `0` forces off), `GGML_CUDA_GRAPHS_FORCE`, `GGML_CUDA_SEGMENTED_SORT` | — |
| 6 | `0f5aaf45d` | 2026-09-21 | ggml-backend | bounds pipeline depth so input copies trade against compute-buffer memory | `GGML_SCHED_N_COPIES` (default 4 = `GGML_SCHED_MAX_COPIES`, range 1..4) | — |
| 7 | `a4e856f1a` | 2026-10-03 | graph | builds the causal KQ mask on the device from cell positions instead of materializing it on the host and copying it | `LLAMA_KQ_MASK_DEVICE` (default on; `0` = host mask) | — |
| 8 | `9f5a3202e` | 2026-10-03 | qwen4exp QSA | builds the QSA block bias on the device | `LLAMA_QSA_BIAS_DEVICE` (default on; `0` = host) | — |
| 9 | `4660a295e` | 2026-10-03 | memory-hybrid-idx | splits a pure multi-sequence decode batch into per-sequence ubatches | `LLAMA_DECODE_PIPELINE` (default `0`) | — |
| 10 | `fb8897947` | 2026-10-03 | graph / KV | stream-agnostic decode graph so those per-sequence decode ubatches can run without per-stream graph constraints | `LLAMA_DECODE_PIPELINE` (same switch, this is the graph half) | — |
| 11 | `1567041ef` | 2026-10-03 | diagnostics | per-ubatch reuse/reserve/timing and per-split host-wait trace | `LLAMA_UBATCH_TRACE`, `GGML_SCHED_LOG_REALLOC` (both off unless set) | — |
| 12 | `1ca80a510` | 2026-10-03 | ggml-backend | stages scheduler inputs through rotating pinned buffers and copies them asynchronously on the compute stream | `GGML_SCHED_STAGE_INPUTS` (default on when the scheduler is parallel with `n_copies > 1`) | — |
| 13 | `5110faa51` | 2026-10-03 | ggml-backend | ggml-backend: one staging offset per backend per compute, not per split | — | — |
| 14 | `6b7bb59fe` | 2026-10-03 | ggml-backend | ggml-backend: grow the staging slots geometrically, and a GGML_SCHED_STAGE_WAIT bisection knob | `GGML_SCHED_STAGE_WAIT` | — |
| 15 | `850db8aec` | 2026-10-03 | ggml-backend | ggml-backend: the source stream of a cross-backend hop copy waits for the destination's previous split (reused-graph pipelining fix; GGML... | `GGML_SCHED_NO_HOP_WAIT=1` disables | — |
| 16 | `21910bb3b` | 2026-10-03 | ggml-backend | ggml-backend: stage scheduler inputs only for reused graphs (a freshly allocated graph has a fresh copy slot); GGML_SCHED_STAGE_INPUTS=2 ... | `GGML_SCHED_STAGE_INPUTS` | — |
| 17 | `66a32a7f8` | 2026-10-04 | scheduler | scheduler input staging only for pipelined decode ubatches (ggml_backend_sched_set_stage_inputs, set per ubatch by llama_context); solo d... | — | — |
| 18 | `fa532f399` | 2026-10-04 | server | server: LLAMA_SERVER_TIMINGS=1 enables the per-phase update_slots timing report at runtime (5 s windows, reset after each report) | `LLAMA_SERVER_TIMINGS=1` | — |
| 19 | `53fcc9f87` | 2026-10-04 | decode groups | phase 2: multiple in-flight batches (llama_output_slots_set/llama_output_slot/llama_output_select: per-batch output regions and completio... | `LLAMA_SERVER_GROUPS` (phase 2 core) | — |
| 20 | `d59899454` | 2026-10-04 | server | server: member-wise server_batch::swap for parking pipelined-group batches (std::swap double-freed the llama_batch arrays) | — | — |
| 21 | `9109ac64c` | 2026-10-04 | decode pipeline | LLAMA_DECODE_PIPELINE=2: single-sequence decode batches also take the stream-agnostic class with a cache-wide n_kv floor (one graph for a... | `LLAMA_DECODE_PIPELINE=2` (set by groups) | — |
| 22 | `c89c7fc89` | 2026-10-04 | diagnostics | trace: mem_hybrid reuse reject detail | — | — |
| 23 | `ec1b7bf52` | 2026-10-04 | decode pipeline | decode pipeline level read live from the environment (the static was captured by the warmup decode before the server set level 2) | — | — |
| 24 | `2c9807e15` | 2026-10-04 | diagnostics | trace full synchronize while output slots are in use; timing report must not synchronize in group mode | — | — |
| 25 | `00b3703f0` | 2026-10-04 | sampling | backend-sampling getters wait only for the selected output batch (the common sampler probes them on every token; the full synchronize ser... | — | — |
| 26 | `f8c8a9ce0` | 2026-10-04 | sampling | common sampler waits only for the selected output batch (llama_output_synchronize) instead of a full synchronize per token | — | — |
| 27 | `ef7ffc7e2` | 2026-10-04 | server groups | server groups: skip groups without work, per-stream graph class while a single slot is active | `LLAMA_SERVER_GROUPS` | — |
| 28 | `07728cbfa` | 2026-10-04 | server groups | server groups: chunk prefill to one ubatch and one slot while other slots generate | `LLAMA_SERVER_PREFILL_CHUNK`, `LLAMA_SERVER_PREFILL_MAX_WITH_DECODE` | — |
| 29 | `da3fbb65d` | 2026-10-04 | llama context | context: reserve compute buffers once per ubatch shape class | — | — |
| 30 | `6ccace4e8` | 2026-10-04 | scheduler | sched: do not drain the pipeline when the graph layout changes | `GGML_SCHED_REPLAN_SYNC=1` restores the drain | — |
| 31 | `7419cd87c` | 2026-10-04 | scheduler | server groups: eager prefill chunks; sched: pre-allocated 12-slot staging ring | `LLAMA_SERVER_PREFILL_EAGER` | — |
| 32 | `442b4e349` | 2026-10-04 | scheduler | sched: synchronize before an unstaged host copy after a re-plan; server: eager prefill off by default | — | — |
| 33 | `270b27d53` | 2026-10-04 | scheduler | safe defaults: scheduler drain on re-plan, prefill chunking off | defaults flipped (later re-enabled in 93764d163) | — |
| 34 | `c73c59364` | 2026-10-04 | scheduler | sched: cover the user's asynchronous output read-outs with the completion events; non-draining re-plan and 256-token prefill chunks by de... | — | — |
| 35 | `ca0faad46` | 2026-10-04 | scheduler | sched: keep the re-plan safety window open for several computes | — | — |
| 36 | `93764d163` | 2026-10-04 | scheduler | sched: order re-planned computes after other-plan computes only; server: chunk only while decoders outnumber prefills | chunking policy default (256 tokens while decoders ≥ prefills) | — |
| 37 | `a1405f2f7` | 2026-10-04 | server | WIP (not compiled): asynchronous prompt-cache save/restore — intermediate commit, compiles together with the next one | — | — |
| 38 | `fcb21fe5f` | 2026-10-04 | server | server: drain-free context checkpoints, pinned staging for state transfers, idle saves spread out | `LLAMA_SERVER_CKPT_SYNC=1` (old), `LLAMA_SERVER_XFER_STAGE_MB` | — |
| 39 | `1b85f6429` | 2026-10-04 | server | server: finish asynchronous state transfers even when no slot is processing | — | — |
| 40 | `2866c72a2` | 2026-10-04 | server | server: a task whose best-matching slot is being saved waits for it | — | — |
| 41 | `c8bef79c2` | 2026-10-04 | server | server: cheap prompt-cache save start (shared checkpoint bytes, off-thread frees) | — | — |
| 42 | `4fb700cf5` | 2026-10-04 | llama | llama: staged state copies on several threads (LLAMA_STATE_XFER_THREADS) | `LLAMA_STATE_XFER_THREADS` (4) | — |
| 43 | `6a33bc322` | 2026-10-05 | server | server: release a parked batch's output slot even when its logits were never read | — | — |
| 44 | `efef36982` | 2026-10-05 | server | server: n-gram speculation with decode groups, recurrent-state snapshots instead of host checkpoints | `--spec-type ngram-*`, `LLAMA_SERVER_SPEC_CKPT_SYNC=1` (old) | — |
| 45 | `83f09b865` | 2026-10-05 | server | speculative: fresh n-gram map per new prompt; server: cap on the number of drafting users | `LLAMA_SERVER_SPEC_MAX_USERS` | — |
| 46 | `7c29e21b8` | 2026-10-05 | server | server: acceptance gate for speculative drafts | `LLAMA_SERVER_SPEC_MIN_ACCEPT` | — |
| 47 | `e4054f726` | 2026-10-05 | llama | llama: portable std::max in output_reserve (gcc 13 rejects the explicit-template brace-list form) | — | — |
| 48 | `286d6953b` | 2026-10-22 | CUDA FA | sync of the FlashAttention/MMA files (`fattn*.cu/.cuh`, `mma.cuh`, `cp-async.cuh`) to upstream `a37cc4610`: swizzle refactor, whole-tile scheduling, process_tile race fix, config tuning; prerequisite for 49–52. Upstream already carries the fork's sparse D=256 1×8 case | built in | upstream [#28536](https://github.com/ggml-org/llama.cpp/pull/28536), [#29612](https://github.com/ggml-org/llama.cpp/pull/29612), [#29435](https://github.com/ggml-org/llama.cpp/pull/29435), [#30103](https://github.com/ggml-org/llama.cpp/pull/30103) |
| 49 | `c7d4c6d44`…`607f75c68` | 2026-10-22 | CUDA FA | PR #85 native quantized FA (21 commits, cherry-picked with `-x`): the MMA FlashAttention kernels read a `q8_0`/`q4_0` (and `q4_1`/`q5_0`/`q5_1` via `GGML_CUDA_FA_QUANTS`) K/V cache in place instead of casting it to f16 per call; tiles for decode, speculative batches and 128-column prompt tiles; stream-k; `NATIVE_QUANT_EQUIVALENCE` test, kernel-inventory CI script | `LLAMA_ARG_FLASH_ATTN_NATIVE_QUANTS=1` / `--flash-attn-native-quants` (off) | fork PR #85 (piggidragon, `fattn-native-quants-flag`), on upstream `a37cc4610`; not an upstream PR |
| 50 | `aa294aa24`, `bafe67a7e`, `f4161cbb3` | 2026-10-22 | CUDA FA | PR #30792 q8_0 tile route: native `q8_0` KV (and f16 K / `q8_0` V) in the flash-attn tile kernel, `(f16, q8_0)` test coverage, Turing routing of `q8_0` decode to the tile kernel; off by default, measured slower than 49 on the 2080 Ti, kept for A/B | `GGML_CUDA_FA_Q8_TILE=N` (unset = off) | upstream [#30792](https://github.com/ggml-org/llama.cpp/pull/30792) |
| 51 | `9838c1aa2`, `5de885b4a` | 2026-10-22 | tests / docs | Turing audit of PR #85 for sm_75 (`AUDIT-D5.md`) and `test-backend-ops` cases for the Qwen3.8-27B per-device shape (D=256, 2 KV heads, GQA 6, `q8_0`, nb 1/2/3/256) | — | — |
| 52 | `8ad6d7f11`, `b72668413`, `7cbcafe48`, `c44b327b0`, `b75cf989f`, `47daf3b50` | 2026-10-22 | CUDA FA | Turing tile-width knob: `ggml_cuda_fattn_pick_turing_cols` in `fattn.cu` pins the f16-MMA and native-route tile width on sm_75, the 64-column trap in `fattn-mma-f16.cuh` is lifted so sm_75 compiles tiles of up to 64 columns; logs the pinned width; report in `docs/pm/D6-turing-tile-knob-RESULT.md` | `GGML_CUDA_FA_TURING_COLS=16\|32\|64` (unset = upstream choice) | — |

## Runtime switches, with defaults read from the code

The whole patchset is selected at runtime, which means **one build carries all of it** — there are no
per-feature binaries and no custom CMake options. Defaults below were taken from the source, not
assumed.

| variable / flag | default | effect |
| --- | --- | --- |
| `LLAMA_DECODE_PIPELINE` | `0` (off) | off: upstream-like batch handling. `1`: pure multi-sequence decode batches are split into per-sequence ubatches and run through the stream-agnostic decode graph (patches 9, 10) |
| `LLAMA_PIPELINE_PARALLEL` | unset (upstream heuristic) | `1`: keep scheduler pipeline parallelism even when tensor overrides move input-stage tensors to CPU; `0`: force it off |
| `LLAMA_QSA_BIAS_DEVICE` | on | `0` restores the host-side QSA block bias (patch 8) |
| `LLAMA_KQ_MASK_DEVICE` | on | `0` restores the host-built causal KQ mask; 2-D M-RoPE image ubatches always keep the host mask (patch 7) |
| `GGML_SCHED_STAGE_INPUTS` | on (parallel scheduler) | `0` disables rotating pinned-buffer staging of user inputs (patch 12) |
| `GGML_SCHED_N_COPIES` | `GGML_SCHED_MAX_COPIES` = 4 | lowers pipeline depth and per-input compute-buffer memory (patch 6) |
| `GGML_CUDA_GRAPHS_FORCE` | off | forces CUDA graph capture/instantiate on otherwise-unsuitable calls (patch 5) |
| `GGML_CUDA_SEGMENTED_SORT` | off | `1` restores the previous argsort behaviour; default is the asynchronous segmented sort (patch 5) |
| `LLAMA_RESERVE_ON_DEMAND_DISABLE` | inherits the base default | `1` disables on-demand reserve (patch 5) |
| `LLAMA_GRAPH_REUSE_DISABLE` | upstream behaviour | existing llama.cpp switch, overridden by the patched context init when set (patch 5) |
| `LLAMA_UBATCH_TRACE` | off | set to any value to enable per-ubatch reuse/reserve/timing and host-wait traces (patch 11) |
| `GGML_SCHED_LOG_REALLOC` | off | set to any value to log every scheduler reallocation, each of which synchronizes all backends (patch 11) |
| `--prefill-max-partial N` | `1` | max slots prefilling within one batch; `1` = first slot takes the whole batch, others wait (patch 4) |
| `--prefill-long-threshold N` | `8192` | a prompt with more than N tokens left to prefill counts as "long" (patch 4) |
| `--prefill-max-long N` | `1` | max long prompts prefilling concurrently; the rest wait in arrival order (patch 4) |
| `--seq-compact` / `--no-seq-compact` | enabled | keep busy slots on contiguous sequence ids so they decode in one ubatch (patch 3) |
| `--cache-idle-slots` | enabled | save and clear idle slots when a new task starts (patch 3 area) |
| `LLAMA_ARG_PREFILL_MAX_PARTIAL` / `_LONG_THRESHOLD` / `_MAX_LONG`, `LLAMA_ARG_SEQ_COMPACT`, `LLAMA_ARG_CACHE_IDLE_SLOTS` | — | environment forms of the flags above |
| `LLAMA_SERVER_GROUPS` | `1` (off) | N ≥ 2: slots are split into N pipelined decode groups; each group's batch is issued and parked while the others' are in flight, so decoders of different slots overlap across the GPUs (patches 19–27). Needs `-np ≥ 2` and no draft-model context; sets `LLAMA_DECODE_PIPELINE=2`. Production: 5 |
| `LLAMA_DECODE_PIPELINE=2` | set by groups | single-sequence decode batches also take the stream-agnostic graph class (one graph shared by all groups; patch 21) |
| `LLAMA_SERVER_PREFILL_CHUNK` | `256` (`0` = off) | prompt tokens per round while other slots decode; the decoders' latency is bounded by one chunk per card (patches 28–36) |
| `LLAMA_SERVER_PREFILL_MAX_WITH_DECODE` | `1` | slots allowed to prefill while others decode |
| `LLAMA_SERVER_PREFILL_EAGER` | off | issue prefill chunks eagerly (more prefill throughput, ~2× the decoders' gaps) |
| `GGML_SCHED_REPLAN_SYNC` | off | `1` restores the full drain when the scheduler re-plans a graph layout (patch 30) |
| `GGML_SCHED_NO_HOP_WAIT` / `GGML_SCHED_STAGE_WAIT` | off | bisection knobs for the cross-backend hop-copy ordering fix (patches 15, 14) |
| `LLAMA_SERVER_CKPT_SYNC` | off | `1` = synchronous context checkpoints (drains the pipeline for every user at each prompt end; patch 38) |
| `LLAMA_SERVER_XFER_STAGE_MB` | `64` (`0` = off) | pinned staging buffer per slot for prompt-cache save/restore transfers (patch 38) |
| `LLAMA_STATE_XFER_THREADS` | `4` (1..8) | copy threads for state transfers; 2.4 GiB restore 1.8 → 1.0 s (patch 42) |
| `LLAMA_SERVER_TIMINGS` | off | `1`: per-phase `update_slots` timing report every 5 s (patch 18) |
| `LLAMA_ATTN_ROT_DISABLE` | off | the production configuration sets `1` together with a quantized (`q8_0`) KV cache (patch 1–2 area) |
| `--spec-type ngram-map-k4v` (+ `--spec-ngram-map-k4v-size-m 48`) | off | n-gram lookup speculation: no draft model; the 48 tokens that followed an earlier occurrence of the last 12 tokens are proposed and verified in one batch; output identical to greedy. Checkpoints are recurrent-state snapshots in the memory (patch 44); `LLAMA_SERVER_SPEC_CKPT_SYNC=1` = old host copies |
| `LLAMA_SERVER_SPEC_MAX_USERS` | `0` (no cap) | draft only while ≤ N slots generate — every draft round re-plans the shared graph twice (~60 ms felt by every user); production: 2 (patch 46) |
| `LLAMA_SERVER_SPEC_MIN_ACCEPT` | `2.5` | acceptance gate: keep drafting while the EMA of accepted draft tokens per round ≥ this × generating users (patch 47) |
| `LLAMA_ARG_FLASH_ATTN_NATIVE_QUANTS` / `--flash-attn-native-quants` | off | `1`: quantized-native FlashAttention — the CUDA MMA kernels read a `q8_0`/`q4_0` KV cache in place instead of casting to f16 per call (patch 49); changes rounding order slightly |
| `GGML_CUDA_FA_Q8_TILE` | unset (off) | `N`: on Turing route `q8_0`-KV FlashAttention with up to N query tokens to the tile kernel (patch 50); A/B knob, slower than patch 49 on the 2080 Ti |
| `GGML_CUDA_FA_TURING_COLS` | unset (upstream choice) | `16`, `32` or `64`: tile width of the f16-MMA and native FlashAttention kernels on sm_75; `64` doubles K/V reuse per shared-memory load for prefill (patch 52) |

## Comparing a patch against upstream

Because each patch has a switch (or is a self-contained commit), an A/B measurement does not need two
builds:

```sh
# patched default (production shape: decode groups, prefill while decoding, async prompt cache)
LLAMA_DECODE_PIPELINE=1 LLAMA_SERVER_GROUPS=5 LLAMA_PIPELINE_PARALLEL=1 GGML_CUDA_GRAPHS_FORCE=1 llama-server ...

# closest upstream-equivalent behaviour for the decode path
LLAMA_DECODE_PIPELINE=0 LLAMA_QSA_BIAS_DEVICE=0 LLAMA_KQ_MASK_DEVICE=0 \
GGML_SCHED_STAGE_INPUTS=0 --no-seq-compact llama-server ...
```

Record which switches were flipped, the upstream commit and the multigpu commit with every number.
Env-only comparisons are cheaper than building twice, but they are **not** equivalent to upstream:
patches 1 and 2 have no opt-out, so the honest upstream comparison is always against a real upstream
build of the named commit.

## Obtaining individual patches

```sh
scripts/multigpu/export-patches.sh                 # git format-patch for the whole downstream range
scripts/multigpu/export-patches.sh -n 3            # only the last three patches
```

CI also publishes the range as a `patches-*.tar.gz` artifact on every build, so users can choose
between upstream, upstream + selected patches, or a complete `llama.cpp-multigpu` build. Explicit
`.patch` files are not committed: they would duplicate the history and go stale on the next rebase.

## Removal criteria

A patch is a candidate for deletion when upstream llama.cpp provides equivalent or better behaviour
for the target workload. The procedure per candidate:

1. identify the upstream commit that plausibly supersedes it;
2. re-sync `multigpu` onto a `master` that contains it;
3. benchmark the affected workloads (prefill and generation, 1 slot and 5 slots) at 5k/50k/150k/200k/250k;
4. if the patch no longer measurably helps, revert/drop it and note the drop here with the commit and
   the benchmark run that justified it;
5. if it still helps, keep it and record *why* the upstream version is insufficient, ideally in the
   linked upstream thread.

Dropping a patch is progress, not regression.

## Scope guard

This table is the project. A new commit here should answer: which workload on which hardware got
faster, and what number proves it? Unrelated llama.cpp features belong upstream. See
[../../README.md](../../README.md) non-goals.
