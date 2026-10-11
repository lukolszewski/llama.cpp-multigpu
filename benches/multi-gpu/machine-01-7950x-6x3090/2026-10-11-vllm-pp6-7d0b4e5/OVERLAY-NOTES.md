# T1 overlay — vLLM 7d0b4e57a (vllm/vllm-openai:cu129-nightly, digest d9af02a8…, 2026-10-10)

Done by the supervisor directly (2026-10-11 01:05, extended 01:40 and 01:50): six files, no worker needed.

## What the nightly already has (alanoo81 patches that are obsolete here)
| alanoo81 patch (flashnext-v620-pp3, Sept 2026) | status in 7d0b4e57a |
|---|---|
| 0001 remove PP RuntimeError in nvidia+amd `model_state.py` | **still needed** → this overlay |
| 0001 gpu_worker.py: drop PP from `unsupported`, conditional `_setup_ple_offload` | obsolete: no `_setup_ple_offload`, no PP check in `v1/worker/gpu_worker.py` |
| 0002 `PleOffloadLayer` import fix | obsolete: `vllm/v1/ple_offload/` does not exist; PLE host offload is now `EngramConfig(cpu_offload=True)` (`vllm/config/engram.py`), auto-enabled for `Qwen4ExpFor*` models with `ple_layer_ids` (`config/vllm.py:_resolve_and_verify_engram_config`), CLI `--engram-config '{"cpu_offload": true}'` |
| 0003/0004 skip final HC mixer weights on non-last ranks (amd/model.py) | upstreamed in nvidia/model.py L597–607 (`ignore_unexpected_prefixes=["hyper_connection_mixer."]` when `self.hyper_connection_mixer is None`) |
| 0005 PLE offload worker pops `VLLM_PP_LAYER_PARTITION` | obsolete: no offload worker process |
| 0006 remove `len(get_pp_group().ranks) > 1` guard in gpu_model_runner.py | obsolete: the only such line (L537) is `broadcast_pp_output` for external_launcher, not a guard |

## This overlay
`vllm/models/qwen4_exp/nvidia/model_state.py`, `vllm/models/qwen4_exp/amd/model_state.py`: replace the unconditional
`RuntimeError("N-gram PLE embedding currently requires pipeline_parallel_size=1 …")` with a check that the first entry of
`VLLM_PP_LAYER_PARTITION` keeps the PLE layer (`ple_layer_ids=[2]` → layer index 1) on stage 0, which is the only rank that
receives raw `input_ids` (nvidia/model.py `forward` L478–519 only starts the PLE prefetch for layers in `[start_layer, end_layer)`).
`orig/` = files extracted from the image; `overlay.diff` = `diff -ruN orig overlay`.

Mount: `-v $PWD/overlay/<rel>:/usr/local/lib/python3.12/dist-packages/<rel>:ro` per file (site-packages path is **dist-packages**).
Verified 01:05: py_compile OK; import inside the image with mounts → both modules report overlay/no-guard.

## Second PP guard (added 01:40, found by T4 window 1)
`vllm/model_executor/models/config.py` (`Qwen4ExpForConditionalGenerationConfig.verify_and_update_config`): the
`NotImplementedError("… requires pipeline_parallel_size=1 …")` is replaced by a check that `VLLM_PP_LAYER_PARTITION` is set
(the first-entry rule is enforced by the model_state overlay above).

## GPTQ fixes (added 01:50, found by T4 window 1 — NOT PP-specific, the nightly cannot load this GPTQ checkpoint at PP=1 either)
The checkpoint (gptqmodel 7.3.5) quantizes Linear layers only: q/k/v/o_proj, MLP, expert weights. BF16 stay: PLE n-gram
table shards, `indexer.index_qk_proj`, linear_attn in_proj/out_proj, norms.
1. `vllm/models/qwen4_exp/common/ngram_embedding.py` — `Qwen4ExpPLEEmbeddingMethod.from_quant_config` raised
   `NotImplementedError("… does not support quantization config AutoGPTQConfig")`; GPTQ-family configs now return the
   unquantized method (table shards are BF16).
2. `vllm/models/qwen4_exp/nvidia/qsa.py` — `fuse_indexer_projection` (merges `indexer.index_qk_proj` into `qkv_proj` as a 4th
   shard) is off under GPTQ, exactly as under LoRA; the BF16 indexer projection stays a separate unquantized Linear.
3. `vllm/models/qwen4_exp/nvidia/model.py` — both model classes use the 3-shard `qkv_proj` packed mapping under GPTQ (the
   LoRA branch) and push it into `quant_config.packed_modules_mapping`, because `configure_quant_config` copies the
   class-level 4-shard mapping before `__init__`; the 4-shard mapping made `gptq_utils.is_layer_gptq_quantized` raise
   `ValueError: Detected some but not all shards of …qkv_proj are quantized`.
GPTQ detection is by class name (`AutoGPTQConfig`, `GPTQConfig`, `GPTQMarlinConfig`) to avoid import cycles.

## Image note (not an overlay)
The nightly's torchcodec 0.17.0 links libnvrtc.so.13 (absent in the cu129 image) and `vllm.multimodal.media.audio` only
catches ImportError/RuntimeError, so `vllm serve` dies at import with OSError. `image/Dockerfile` = nightly pinned by digest
+ `pip uninstall -y torchcodec` → local image `vllm-pp6:7d0b4e57a` (eed859ad25fd). Upstream per-commit tag also exists:
`vllm/vllm-openai:cu129-nightly-7d0b4e57aac4c4323225b0eee9ae960ce75ffffe`.

## KV-cache allocation under PP (added 02:25, 7th file)

`overlay/vllm/v1/core/kv_cache_utils.py` — `get_kv_cache_config_from_groups`. Window 2 (6-file
overlay) loaded all weights on 6 ranks but Workers PP1–PP5 died in `vllm/v1/worker/utils.py
allocate_kv_cache` with `StopIteration`: a `KVCacheTensor.layers[0]` was not in any of that rank's
`kv_cache_groups[*].layer_names`.

Cause: `get_kv_cache_configs` projects the global groups onto each worker
(`_project_kv_cache_groups_to_worker`). For a hybrid model the full-attention layers are split into
several `UniformTypeKVCacheSpecs` groups; with PP some of those groups have no layer on a given rank.
The projection keeps such a group (empty `layer_names`) but, because `worker_layer_names` is empty,
does not rebuild its spec — the group still carries the *global* `UniformTypeKVCacheSpecs` listing
every rank's layers. `get_kv_cache_config_from_groups` then iterates
`group_spec.kv_cache_specs.items()` (all layers) instead of `group.layer_names`, producing tensors
for layers that live on other ranks.

Fix: in that loop, only emit layers present in `group.layer_names`. All sizing code
(`_get_kv_cache_bytes_per_block`, `validate_kv_cache_layout`) already iterates `group.layer_names`,
so num_blocks/strides are unchanged; only the bogus tensors disappear. Rank 0 was unaffected in
window 2 because every group had at least one layer there.
