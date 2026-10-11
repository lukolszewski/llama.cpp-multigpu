# Benchmark methodology and results

Benchmarks are the reason this fork exists: a performance claim without a measured number, a named
upstream revision and a named machine is not a result. The machine-01 grid was measured on 2026-10-05
(results below); every remaining `TBD` means "not measured", never "approximately".

- Headline table: [README §1](../../README.md#1-performance-summary)
- This document: suites, protocol, metric definitions, machine records, reproduction steps
- Raw output: `benches/multi-gpu/<machine>/`

---

## Rules

1. **Every comparison names both revisions.** `upstream commit: <sha>` and `multigpu commit: <sha>`.
   Saying "upstream" without a revision is meaningless because llama.cpp moves daily. If an upstream
   change removes a patch's benefit, that must be visible, and the patch gets dropped
   ([patches.md#removal-criteria](patches.md#removal-criteria)).
2. **Every number names its machine and its configuration**: model + quant + file list, context size,
   slot count, batch settings, split mode / tensor split, GPU placement, environment variables, server
   command line, CUDA/driver/toolchain, build flags.
3. **Never mix machines in one table.** A row belongs to exactly one machine record. Cross-machine
   comparison is done by linking two tables, not by merging rows.
4. **No fabricated or interpolated values.** Missing data stays `TBD`.
5. **Unfavourable behaviour is published.** Mixed prefill + generation is documented as its own section
   rather than hidden behind all-prefill / all-generate tables.
6. **A patch may be a net loss elsewhere.** Each result states the tested hardware and workload; claims
   beyond that configuration are not made.

---

## Machines

### machine-01 (primary)

| field | value | source |
| --- | --- | --- |
| hostname | `megaczop` | measured 2026-10-04 |
| CPU | AMD Ryzen 9 7950X, 16C/32T, 1 NUMA node exposed | `lscpu` |
| RAM | 192 GB DDR5, 4 x 48 GB, JEDEC settings (`free` reports 187 GiB) | brief + `free -g` |
| Motherboard | ASUS ProArt X670E-CREATOR WIFI | `/sys/class/dmi/id/board_name` |
| GPUs | 6 x GeForce RTX 3090, 24576 MiB each, 144 GB aggregate | `nvidia-smi` |
| Other GPU | 1 x GeForce RTX 5060 Ti, 16311 MiB (index 6, ~208 MiB used, not in the inference pool) | `nvidia-smi` |
| GPU 0/1 interconnect | NVLink, 4 active links at 14.062 GB/s each (`NV4` in `nvidia-smi topo -m`) | `nvidia-smi nvlink -s`, `topo -m` |
| GPU 2-5 interconnect | `PHB`/`PXB` — behind PCIe host bridges, no active NVLink | `nvidia-smi topo -m` |
| PCIe width per GPU (idle) | GPU0 x8, GPU1 x8, GPU2 x4, GPU3 x2, GPU4 x4, GPU5 x4 | `nvidia-smi --query-gpu=pcie.link.width.current` |
| PCIe generation (idle) | Gen1 reported for all 3090s while in P8 — a power-state artifact, **not** the link capability | `nvidia-smi --query-gpu=pcie.link.gen.current` |
| PCIe LnkCap / LnkSta under load | **TBD** (needs `sudo lspci -vv` while the GPUs are busy) | `scripts/multigpu/env-snapshot.sh` |
| OS / kernel | Debian GNU/Linux 12 (bookworm), 6.12.9 SMP PREEMPT_DYNAMIC | `uname -a` |
| NVIDIA driver | 595.99.02, driver-reported CUDA 13.2 | `nvidia-smi` |
| CUDA toolkit / `nvcc` on host | **TBD** (no `nvcc` in `PATH` at snapshot time) | `command -v nvcc` |
| Compiler / toolchain | **TBD** | |
| CMake / Ninja | **TBD** | |
| ECC / power limits / cooling | **TBD** (3090 has no ECC; power caps affect every number here) | |
| Model | Qwen3.8-Flash-Next, GGUF arch `qwen4exp` | |
| Quantization | `UD-Q4_K_XL` from `unsloth/Qwen3.8-Flash-Next-GGUF` | |
| Model files | `Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` (10.9 MB), `-00002-of-00004.gguf` (49.86 GB), `-00003-of-00004.gguf` (49.38 GB), `-00004-of-00004.gguf` (12.09 GB) — ~111.3 GB total | HF API listing |
| VRAM budget | 144 GB aggregate − ~111.3 GB weights ≈ 33 GB for KV cache + compute buffers across 6 devices, before overhead; measured per-device usage **TBD** | arithmetic on the two rows above |
| Server command line | **TBD (recorded per run)** | |
| Slots / context / batch / split mode | **TBD** (`-np 5` is the benchmark server configuration) | |
| Environment variables | **TBD** (must be recorded, since the patchset is switch-selected) | |

Regenerate this record with `scripts/multigpu/env-snapshot.sh`, which writes the raw tool output to
`benches/multi-gpu/machine-01-7950x-6x3090/`. Fields marked **TBD** require either root (PCIe
capability registers) or a run of the benchmark itself.

### machine-02 and later (rented, Vast.ai)

Rented multi-GPU machines are measured with the tooling in
[`scripts/multigpu/bench/vast/`](../../scripts/multigpu/bench/vast/README.md): the same grid script, the
same server flags and the release's own `cuda-12.9` image, on 6 × RTX 4090, 6–8 × RTX 5090, RTX 6000 /
PRO 6000 and 8 × Tesla V100 class boxes as they become available. Each machine gets its own
`benches/multi-gpu/machine-NN-<n>x<gpu>/` with a generated `hardware.md` and one run directory per
measurement; a section is appended below and a row to README §1 by `vast-bench.sh land`, from the raw JSON,
through a pull request. Until a run exists a machine has no rows anywhere.

Differences from machine-01 that are inherent to a rented container and are stated in every record:

- the hardware record is what `nvidia-smi` reports from inside the container (driver, VRAM, PCIe
  link gen/width as negotiated at collection time, topology matrix); motherboard, RAM modules and PCIe
  capability registers are `TBD` unless `lspci` can see the bus;
- the host driver is whatever the provider runs (R570–R610 in October 2026); PTX-only GPUs (Turing)
  are only accepted on drivers that know CUDA 12.9 PTX;
- the upstream side is the fork's own build of its upstream base commit with the same recipe
  (`baseline-<sha7>` prerelease, see [builds.md](builds.md)), because upstream publishes no Linux CUDA
  tarball or `server-cuda-b<N>` image for these builds; when credit does not allow the 3–4 h upstream
  grid, the run says "upstream: not measured on this machine" and the README row carries no ratio;
- the tensor split is VRAM-proportional with the main GPU reduced by the same ~3.6 GB machine-01 uses
  (`--tensor-split` printed in `bench.log` and stored in `config.json`), the context stays 5 × 262144
  regardless of how much VRAM the box has, so the numbers are comparable to machine-01's;
- `sm_120` (RTX 5090) and `sm_89` (RTX 4090) boxes runtime-validate the `120a`/`89` code of the release
  that machine-01 cannot (its 5060 Ti is the display GPU and outside the inference pool).

---

## Suites

### Single slot (`-np 1`, one active session)

For each context/prompt size **5k, 50k, 150k, 200k, 250k**:

1. **prefill / prompt processing** — the whole prompt processed as one request;
2. **generation** — decode from the resulting context.

Purpose: characterize a single interactive session from short prompts up to very large contexts, where
KV cache growth and cross-device traffic begin to dominate.

### Five slots (`-np 5`, all slots active concurrently)

For each of the same sizes, all five slots concurrently, as two separate workloads:

1. **all five prefill at once**;
2. **all five generate at once**.

Both aggregate throughput and per-slot throughput are reported: a 5x aggregate figure with 5x worse
per-slot latency is not a win for the intended user.

Why 5 slots: that is the benchmark server configuration, and it matches the intended usage — one
person running several agent sessions, coding-agent tasks, independent chats or parallel research
jobs against a single local server.

---

## Metrics

| metric | definition |
| --- | --- |
| prompt processing t/s | tokens/sec over the whole prefill, from request start to first token generated |
| generation t/s | decode tokens/sec, averaged over the measured generation window |
| aggregate t/s | sum across all concurrently active slots |
| per-slot t/s | mean over slots, plus min/max if spread matters |
| time to first token | request received → first token emitted, per slot |
| total prefill time | wall-clock until the last slot finished prefilling |
| total generation wall time | wall-clock for the whole generation window |
| per-request latency | end-to-end per request, including queueing/waiting |
| VRAM per device | `nvidia-smi` sampled during the run, peak and steady-state |
| host RAM | peak RSS of the server process |
| GPU utilization | optional, `nvidia-smi`/DCGM if reproducible |
| PCIe traffic | optional, only if a reproducible measurement method is documented |

The README carries only the headline table; the rest lives here and in `benches/multi-gpu/`.

---

## Results

### machine-01

```
upstream commit:  df03399b885831b2a1603b3abb0d8c156808e363  (df03399b8, built with .devops/cuda.Dockerfile: CUDA 12.4.1, gcc 12, 86-real)
multigpu commit:  1344895820bf3d5ec14764132ac84fd5494b15d0  (code = e4054f726, image llama.cpp:q8c-p6-cuda124, same recipe)
model:            Qwen3.8-Flash-Next, UD-Q4_K_XL (unsloth, 4 shards)
server:           -ngl 99 -ot 'per_layer_token_embd\.weight=CPU' --tensor-split 0.85,1,1,1,1,1 --main-gpu 0
                  -c 1310720 --parallel 5 -fa on --cache-type-k q8_0 --cache-type-v q8_0 -b 2048 -ub 512 --jinja
                  (patched adds --prefill-max-partial 2 and the env LLAMA_DECODE_PIPELINE=1 LLAMA_SERVER_GROUPS=5
                  LLAMA_PIPELINE_PARALLEL=1 GGML_CUDA_GRAPHS_FORCE=1 LLAMA_ATTN_ROT_DISABLE=1; speculation off)
prompts:          synthetic word lists; actual tokens 4992 / 49678 / 148978 / 198628 / 248278 per slot
prefill:          one request per slot, n_predict 1, cache_prompt false; TTFT = request wall time
generate:         128 greedy tokens on the cached prompt (cache_prompt true; prompt_n of the second request = 4)
measured:         2026-10-05 13:05-17:12, patched grid first (13:05-13:38), upstream second (13:38-17:12);
                  0 error lines in both server logs; GPU temperature/power sampled before and after each row
raw data:         benches/multi-gpu/machine-01-7950x-6x3090/2026-10-05-grid-df03399b8-vs-134489582/
```

Time to first token at the deepest rows (prefill wall time, five slots at once): upstream 4684 s vs
patched 603 s at 250k; single slot 945 s vs 118 s.

| Workload | Context | Upstream llama.cpp | llama.cpp-multigpu | Improvement |
| --- | --- | --- | --- | --- |
| 1 slot - prefill | 5k | 715 | 1249 | 1.75x (+534 t/s) |
| 1 slot - generate | 5k | 38.5 | 45.9 | 1.19x (+7.4 t/s) |
| 1 slot - prefill | 50k | 568 | 2127 | 3.75x (+1560 t/s) |
| 1 slot - generate | 50k | 25.7 | 41.9 | 1.63x (+16.2 t/s) |
| 1 slot - prefill | 150k | 368 | 2152 | 5.86x (+1785 t/s) |
| 1 slot - generate | 150k | 14.1 | 37.8 | 2.69x (+23.7 t/s) |
| 1 slot - prefill | 200k | 315 | 2122 | 6.73x (+1806 t/s) |
| 1 slot - generate | 200k | 11.9 | 36.6 | 3.07x (+24.7 t/s) |
| 1 slot - prefill | 250k | 263 | 2111 | 8.02x (+1848 t/s) |
| 1 slot - generate | 250k | 10.2 | 33.7 | 3.29x (+23.5 t/s) |
| 5 slots - concurrent - prefill | 5k | 630 (388/slot) | 2120 (681/slot) | 3.37x (+1491 t/s) |
| 5 slots - concurrent - generate | 5k | 77.3 (18.2/slot) | 175.2 (42.3/slot) | 2.27x (+97.9 t/s) |
| 5 slots - concurrent - prefill | 50k | 574 (526/slot) | 2301 (1511/slot) | 4.01x (+1727 t/s) |
| 5 slots - concurrent - generate | 50k | 34.8 (8.1/slot) | 153.3 (38.3/slot) | 4.41x (+118.5 t/s) |
| 5 slots - concurrent - prefill | 150k | 363 (353/slot) | 2251 (1993/slot) | 6.20x (+1888 t/s) |
| 5 slots - concurrent - generate | 150k | 15.8 (3.6/slot) | 123.1 (32.4/slot) | 7.78x (+107.2 t/s) |
| 5 slots - concurrent - prefill | 200k | 307 (300/slot) | 2180 (1977/slot) | 7.11x (+1873 t/s) |
| 5 slots - concurrent - generate | 200k | 11.6 (2.8/slot) | 105.9 (29.0/slot) | 9.13x (+94.3 t/s) |
| 5 slots - concurrent - prefill | 250k | 265 (261/slot) | 2060 (1928/slot) | 7.77x (+1795 t/s) |
| 5 slots - concurrent - generate | 250k | 9.1 (2.3/slot) | 98.0 (27.3/slot) | 10.79x (+88.9 t/s) |

Units: tokens/s (aggregate for the 5-slot rows; per-slot values recorded alongside). Improvement is
reported as a ratio **and** absolute delta, since a 2x on a bad baseline and a 1.2x on a good one mean
different things to different users.

### machine-01 - vLLM pipeline-parallel column

```
vllm:             0.31.1rc1.dev253+g7d0b4e57a (commit 7d0b4e57a)
image:            vllm-pp6:7d0b4e57a  digest sha256:eed859ad25fddc9ca83593fc315d5b0b63bfbe87b3fd0fd9499d2533c4e9cb8b
run:              docker run -d --name vllm-pp6 --gpus "device=0,1,2,3,4,5" --ipc=host --shm-size 16g -p 8010:8010 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e HF_HUB_OFFLINE=1 -e VLLM_PP_LAYER_PARTITION=8,8,8,8,8,8 -v /home/luk/dev/ai/cache/qwen38-flash-next-gptq4:/model:ro -v /home/luk/dev/airun/vllm-pp6/work/T1-patches/overlay/vllm/model_executor/models/config.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/config.py:ro -v /home/luk/dev/airun/vllm-pp6/work/T1-patches/overlay/vllm/models/qwen4_exp/amd/model_state.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/amd/model_state.py:ro -v /home/luk/dev/airun/vllm-pp6/work/T1-patches/overlay/vllm/models/qwen4_exp/common/ngram_embedding.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/common/ngram_embedding.py:ro -v /home/luk/dev/airun/vllm-pp6/work/T1-patches/overlay/vllm/models/qwen4_exp/nvidia/model.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/model.py:ro -v /home/luk/dev/airun/vllm-pp6/work/T1-patches/overlay/vllm/models/qwen4_exp/nvidia/model_state.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/model_state.py:ro -v /home/luk/dev/airun/vllm-pp6/work/T1-patches/overlay/vllm/models/qwen4_exp/nvidia/qsa.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/qsa.py:ro -v /home/luk/dev/airun/vllm-pp6/work/T1-patches/overlay/vllm/v1/core/kv_cache_utils.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py:ro vllm-pp6:7d0b4e57a /model --served-model-name qwen3.8-flash-next --dtype bfloat16 --pipeline-parallel-size 6 --tensor-parallel-size 1 --distributed-executor-backend mp --max-model-len 262144 --max-num-seqs 5 --gpu-memory-utilization 0.92 --engram-config {"cpu_offload": true} --port 8010 --host 0.0.0.0 --trust-remote-code --no-enable-prefix-caching
pipeline:         part=VLLM_PP_LAYER_PARTITION=8,8,8,8,8,8 (pipeline_parallel_size 6, tensor_parallel_size 1, mp executor); max_model_len=262144; max_num_seqs=5; kv_cache_dtype=auto (bf16); prefix_caching=false
patches:          overlay.diff in this directory (unified diff of the 7 overlaid files against the image originals)
overlay files:    overlay/vllm/model_executor/models/config.py, overlay/vllm/models/qwen4_exp/amd/model_state.py, overlay/vllm/models/qwen4_exp/common/ngram_embedding.py, overlay/vllm/models/qwen4_exp/nvidia/model.py, overlay/vllm/models/qwen4_exp/nvidia/model_state.py, overlay/vllm/models/qwen4_exp/nvidia/qsa.py, overlay/vllm/v1/core/kv_cache_utils.py
env:              CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 VLLM_PP_LAYER_PARTITION=8,8,8,8,8,8
checkpoint:       btbtyler09/Qwen3.8-Flash-Next-GPTQ-4bit (70 files, 188 GB = 175 GiB; per-file sizes in config.json)
measured:         2026-10-11 (grid-vllm.json)
gpus:             6 x RTX 3090 (GPUs 0-5, one pipeline stage each; GPU 6 RTX 5060 Ti unused), AMD 7950X, 187 GB RAM
raw data:         benches/multi-gpu/machine-01-7950x-6x3090/2026-10-11-vllm-pp6-7d0b4e5/
vllm commit: 7d0b4e57a (vllm 0.31.1rc1.dev253+g7d0b4e57a, torch 2.13.0+cu129; base image vllm/vllm-openai:cu129-nightly as of 2026-10-10)
vllm image: vllm-pp6:7d0b4e57a (local derivative: base image with torchcodec removed; Id sha256:eed859ad25fddc9ca83593fc315d5b0b63bfbe87b3fd0fd9499d2533c4e9cb8b, not pushed)
vllm patches: 7 Python files bind-mounted over /usr/local/lib/python3.12/dist-packages (overlay.diff in this directory; file list in config.json overlay_files)
checkpoint: btbtyler09/Qwen3.8-Flash-Next-GPTQ-4bit (GPTQ 4-bit g32 Linear-only, 70 files, 188 GB (175 GiB); PLE table / indexer / linear_attn / norms BF16)
upstream llama.cpp commit: df03399b885831b2a1603b3abb0d8c156808e363 (column copied from 2026-10-05-grid-df03399b8-vs-134489582/grid-upstream.json)
multigpu commit: 1344895820bf3d5ec14764132ac84fd5494b15d0 (column copied from 2026-10-05-grid-df03399b8-vs-134489582/grid-multigpu.json)
```

| Workload | Context | Upstream llama.cpp | llama.cpp-multigpu | vLLM-PP (patched) | multigpu vs upstream | multigpu vs vLLM |
| --- | --- | --- | --- | --- | --- | --- |
| 1 slot - prefill | 5k | 715 | 1249 | 4028 | 1.75x (+534 t/s) | 0.31x (-2778 t/s) |
| 1 slot - generate | 5k | 38.5 | 45.9 | 55.9 | 1.19x (+7.4 t/s) | 0.82x (-10.0 t/s) |
| 1 slot - prefill | 50k | 568 | 2127 | 8365 | 3.75x (+1560 t/s) | 0.25x (-6238 t/s) |
| 1 slot - generate | 50k | 25.7 | 41.9 | 57.6 | 1.63x (+16.2 t/s) | 0.73x (-15.7 t/s) |
| 1 slot - prefill | 150k | 368 | 2152 | 9220 | 5.86x (+1785 t/s) | 0.23x (-7068 t/s) |
| 1 slot - generate | 150k | 14.1 | 37.8 | 58.8 | 2.69x (+23.7 t/s) | 0.64x (-21.0 t/s) |
| 1 slot - prefill | 200k | 315 | 2122 | 9235 | 6.73x (+1806 t/s) | 0.23x (-7113 t/s) |
| 1 slot - generate | 200k | 11.9 | 36.6 | 59.0 | 3.07x (+24.7 t/s) | 0.62x (-22.4 t/s) |
| 1 slot - prefill | 250k | 263 | 2111 | 9046 | 8.02x (+1848 t/s) | 0.23x (-6935 t/s) |
| 1 slot - generate | 250k | 10.2 | 33.7 | 57.8 | 3.29x (+23.5 t/s) | 0.58x (-24.1 t/s) |
| 5 slots - concurrent - prefill | 5k | 630 (388/slot) | 2120 (681/slot) | 6208 (2192/slot) | 3.37x (+1491 t/s) | 0.34x (-4088 t/s) |
| 5 slots - concurrent - generate | 5k | 77.3 (18.2/slot) | 175.2 (42.3/slot) | 107.9 (31.2/slot) | 2.27x (+97.9 t/s) | 1.62x (+67.4 t/s) |
| 5 slots - concurrent - prefill | 50k | 574 (526/slot) | 2301 (1511/slot) | 8452 (3872/slot) | 4.01x (+1727 t/s) | 0.27x (-6151 t/s) |
| 5 slots - concurrent - generate | 50k | 34.8 (8.1/slot) | 153.3 (38.3/slot) | 24.2 (15.2/slot) | 4.41x (+118.5 t/s) | 6.33x (+129.1 t/s) |
| 5 slots - concurrent - prefill | 150k | 363 (353/slot) | 2251 (1993/slot) | 8203 (3752/slot) | 6.20x (+1888 t/s) | 0.27x (-5952 t/s) |
| 5 slots - concurrent - generate | 150k | 15.8 (3.6/slot) | 123.1 (32.4/slot) | 8.5 (11.2/slot) | 7.78x (+107.2 t/s) | 14.43x (+114.5 t/s) |
| 5 slots - concurrent - prefill | 200k | 307 (300/slot) | 2180 (1977/slot) | 8533 (3882/slot) | 7.11x (+1873 t/s) | 0.26x (-6354 t/s) |
| 5 slots - concurrent - generate | 200k | 11.6 (2.8/slot) | 105.9 (29.0/slot) | 6.7 (10.3/slot) | 9.13x (+94.3 t/s) | 15.78x (+99.2 t/s) |
| 5 slots - concurrent - prefill | 250k | 265 (261/slot) | 2060 (1928/slot) | 8111 (3899/slot) | 7.77x (+1795 t/s) | 0.25x (-6051 t/s) |
| 5 slots - concurrent - generate | 250k | 9.1 (2.3/slot) | 98.0 (27.3/slot) | 5.0 (12.6/slot) | 10.79x (+88.9 t/s) | 19.44x (+92.9 t/s) |

Caveats:

- Pipeline parallelism (PP=6, one 8-layer stage per GPU) over 6x RTX 3090 (PCIe: GPU0/1 Gen4 x8, GPU2-5 behind Thunderbolt); the upstream nightly refuses PP for this model, the overlay removes the guards and fixes GPTQ loading and KV-cache allocation under PP
- PLE n-gram table (102 GB BF16) host-offloaded via --engram-config cpu_offload: container RSS ~143 GB, 20 GB swap in use during the run — host RAM is the binding constraint
- KV cache 1,016,755 tokens total (bf16) → 3.88 concurrent 262k sequences; 5 x 250k prompts exceed it, so the 5-session 250k row includes queueing
- Correctness: needle-in-haystack at 50k and 200k passed; byte-equality of 5 concurrent vs solo greedy completions FAILS (outputs identical for the first ~30-40 tokens then diverge in wording, coherent) — batch-composition nondeterminism, not corruption; same result with max_num_seqs=2; prefix caching ON (max_num_seqs=5) also passed the needle checks and showed the same nondeterminism (2 of 5 differ); grid run with prefix caching OFF
- Load time: model healthy 376-468 s after container start over four starts (weights ~86 s, the rest is engine init and compilation) vs ~2-3 min for llama.cpp
- GPU 6 (RTX 5060 Ti) unused by vLLM; production llama.cpp server stopped for the duration of the window
- 5-session generate rows are not like-for-like with llama.cpp: the OpenAI client sends one streaming request per session, so generation is timed from each session's first token to its last; the first session to finish prefill decodes while the other four are still prefilling (chunked prefill shares every step), its decode crawls and the aggregate window spans the remaining prefill — llama.cpp's 5-slot generate was measured with all slots already cached and decoding together
- vLLM is upstream 7d0b4e57a + 7 overlaid Python files, see overlay.diff in this directory — no PP-capable release exists for this model
- prefix caching off for the grid
- KV cache dtype: auto (bf16)
- one streaming request per cell; prefill t/s = prompt_tokens/TTFT, generation t/s from inter-token time (the llama.cpp columns use the server's own timings)

### rented machines

Sections are appended here by `scripts/multigpu/bench/vast/vast-bench.sh land` (one per machine, same
table shape, generated from the run's JSON). None yet.

### machine-02-6x4090 (rented, Vast.ai)

```
multigpu commit:  6a8a5995f69613471b170143803ef1f4615fc5fc  (image cuda-12.9, CUDA 12.9.1; SASS sm_70,sm_86,sm_89,sm_120a,sm_121a; PTX sm_50,sm_61,sm_70,sm_75,sm_80,sm_90)
upstream base:   df03399b885831b2a1603b3abb0d8c156808e363
upstream binary: version: 0.4.0-dev (build 10902, commit df03399b8) built with GNU 13.3.0 for Linux x86_64  from https://github.com/lukolszewski/llama.cpp-multigpu/releases/download/baseline-df03399/llama.cpp-upstream-df03399-bin-ubuntu-cuda-12.9-x64.tar.gz
machine:          8 x NVIDIA GeForce RTX 4090 (191 GB aggregate), GPUs used: 0,1,2,3,4,5 (6 of 8), driver 570.211.01 (CUDA 12.8), CPU Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz, RAM 755.2 GiB; record: benches/multi-gpu/machine-02-6x4090/hardware.md
server:           -c 1310720 --parallel 5 -fa on --cache-type-k/v q8_0 -b 2048 -ub 512 --tensor-split 0.85,1,1,1,1,1 (patched adds --prefill-max-partial 2 and LLAMA_DECODE_PIPELINE=1 LLAMA_SERVER_GROUPS=5 LLAMA_PIPELINE_PARALLEL=1 GGML_CUDA_GRAPHS_FORCE=1 LLAMA_ATTN_ROT_DISABLE=1; speculation off)
grid:             slots 1,5, sizes 5000,50000,150000,200000,250000; upstream slots 1, sizes 5000,50000,150000,200000,250000; same protocol and scripts as machine-01
measured:         2026-10-06; upstream df03399b8 (same cuda-12.9 recipe, baseline prerelease)
raw data:         benches/multi-gpu/machine-02-6x4090/2026-10-06-grid-df03399-vs-6a8a599/
```

| Workload | Context | Upstream llama.cpp | llama.cpp-multigpu | Improvement |
| --- | --- | --- | --- | --- |
| 1 slot - prefill | 5k | 1964 | 2954 | 1.50x (+990 t/s) |
| 1 slot - generate | 5k | 62.2 | 62.8 | 1.01x (+0.7 t/s) |
| 1 slot - prefill | 50k | 1728 | 7050 | 4.08x (+5322 t/s) |
| 1 slot - generate | 50k | 46.2 | 57.5 | 1.24x (+11.3 t/s) |
| 1 slot - prefill | 150k | 1039 | 8078 | 7.78x (+7040 t/s) |
| 1 slot - generate | 150k | 29.3 | 51.1 | 1.74x (+21.8 t/s) |
| 1 slot - prefill | 200k | 866 | 7601 | 8.77x (+6735 t/s) |
| 1 slot - generate | 200k | 25.1 | 50.1 | 2.00x (+25.0 t/s) |
| 1 slot - prefill | 250k | 744 | 7403 | 9.95x (+6659 t/s) |
| 1 slot - generate | 250k | 21.0 | 48.7 | 2.32x (+27.7 t/s) |
| 5 slots - concurrent - prefill | 5k | not run | 6596 (2781/slot) | n/a |
| 5 slots - concurrent - generate | 5k | not run | 245.6 (60.2/slot) | n/a |
| 5 slots - concurrent - prefill | 50k | not run | 8805 (6351/slot) | n/a |
| 5 slots - concurrent - generate | 50k | not run | 208.8 (55.1/slot) | n/a |
| 5 slots - concurrent - prefill | 150k | not run | 7932 (7139/slot) | n/a |
| 5 slots - concurrent - generate | 150k | not run | 136.1 (38.9/slot) | n/a |
| 5 slots - concurrent - prefill | 200k | not run | 7606 (6985/slot) | n/a |
| 5 slots - concurrent - generate | 200k | not run | 86.5 (26.0/slot) | n/a |
| 5 slots - concurrent - prefill | 250k | not run | 7241 (6836/slot) | n/a |
| 5 slots - concurrent - generate | 250k | not run | 101.1 (30.8/slot) | n/a |

---

## Mixed workload behavior

Not part of the headline table, and not to be inferred from it.

When prompt prefill happens while another slot is generating, the two interfere: the prefill's large
batch competes with decode's latency-critical small batch, and on this hardware they also contend for
the same constrained PCIe links that the scheduler uses to move activations between GPUs. The current
result is less-than-ideal behaviour for that mixture, which is exactly why the fork is presented as
**multi-session single-user** tooling rather than a general multi-tenant inference server.

Consequences to state whenever results are published:

- all-prefill and all-generate numbers are best-case slices of the workload;
- a user whose requests arrive continuously and expect consistent latency should not read the headline
  table as their expectation;
- the admission-control patches (`--prefill-max-partial`, `--prefill-max-long`) are the current
  mitigation, i.e. they bound interference by making some slots wait, which trades throughput for
  predictability. Their effect is itself a measurement to be published, **TBD**.

Planned but **not implemented** — deliberately no suite, no scripts and no rows until they exist:

- one generating slot + one prefilling slot;
- several generating slots + one prefilling slot;
- continuous arrival mix.

---

## Reproducing

```sh
# 1. build or download BOTH revisions. Record the two commits.
#    upstream:   master @ <sha>          (stock build, no downstream switches)
#    multigpu:   multigpu @ <sha>

# 2. capture the machine record on the benchmark box (needs sudo for PCIe registers)
scripts/multigpu/env-snapshot.sh benches/multi-gpu/machine-01-7950x-6x3090

# 3. single-slot suite (prefill + generate at 5k/50k/150k/200k/250k)
scripts/multigpu/bench/run-single-slot.sh --model <first-shard.gguf> --out results.json

# 4. five-slot suite against a running server (-np 5), all-prefill and all-generate
scripts/multigpu/bench/run-5-slot.py --server http://127.0.0.1:8080 --slots 5 --size 50000 --phase pp

# 5. paste the resulting table rows here, plus the raw files into benches/multi-gpu/<machine>/
```

The scripts in `scripts/multigpu/bench/` implement the protocol above; the 2026-10-05 grid was produced
with the copies committed next to its raw data (`readme_grid.py`, `readme_table.py`, `bench_server.sh`,
`readme_grid_chain.log`). Report problems, do not quietly adjust the protocol afterwards. Each run must
print both commit hashes into its output file so a stray JSON blob can still be attributed. The README
header chart is generated from the raw JSON by `scripts/multigpu/bench/plot-grid.py`.

For comparisons against upstream, run the same script twice with different `--bin` values, on the same
machine, same model file, same session, ideally back-to-back to keep thermal state comparable. Record
GPU temperatures/power caps if they plausibly affect the numbers — on air-cooled 3090s they do.
