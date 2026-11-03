# P11 — fp8 (e4m3) KV cache for Qwen3.8-Flash-Next QSA on Ampere (RTX 3090): software e4m3 in the Triton kernels

Written 2026-10-11 12:59. Sub-plan of `PLAN-PLE-SSD-Q8KV.md` task P11; self-contained. State of record:
`RUN.md` (`## Next action`, Tasks rows P11*, Facts). Production llama.cpp on :8009 must stay up the whole
time; the only GPU this plan uses is **GPU 6 (RTX 5060 Ti, 16 GB, sm_120)**, for development/tests only
(user 2026-10-11: fine for development, not for long-term serving).

## 0. Goal / done

**Goal.** `vllm serve … --kv-cache-dtype fp8` runs on the six 3090s with the fused QSA path, producing the
same cache bytes and the same attention output as the native e4m3 kernels on Ada/Hopper/Blackwell.

**Done when**
1. Unit test: the two conversion helpers are byte-exact against PyTorch's own e4m3 conversion over all
   256 codes (load) and over ≥ 1 M fp32 values incl. ties, saturation, subnormals, ±0, ±inf, NaN (store).
2. End-to-end on GPU 6: `qsa_sparse_paged_attention` with the emulated path (uint8 cache) gives bitwise the
   same output as the native path (e4m3 cache) for the same bytes; `qsa_prepare`'s main K/V write gives the
   same cache bytes emulated vs native (best effort — see P11e).
3. Overlay regenerated (`work/T1-patches/overlay` + the new file), `python3 -m py_compile` clean, commits on
   fork branch `ampere-e4m3-kv` (from `pp-nvidia-qwen`).
4. The proof on the 3090s is **window W1 of the parent plan** (or a dedicated 30-min window `W-Q8KV-2` =
   re-run of `work/W-Q8KV/steps.sh` unchanged): `--kv-cache-dtype fp8` reaches `/health`, log line
   "GPU KV cache size: ≥ 1,310,720 tokens", 5 × 250k concurrent Running, needle PASS 50k/200k.
   Not run inside this sub-plan unless the user schedules it.

## 1. Facts (verified 2026-10-11)

- Failure: `ValueError("type fp8e4nv not supported in this architecture. The supported fp8 dtypes are
  ('fp8e4b15', 'fp8e5')")` — Triton 3.7.1 `triton/language/core.py:579`; `triton/backends/nvidia/compiler.py:121`
  `supported_fp8_dtypes = ("fp8e5", "fp8e4b15")` for sm < 89. Any `.to(tl.float8e4nv)` or cast FROM a
  `fp8e4nv` value fails at compile time on the 3090. Full trace: `work/W-Q8KV/vllm-server.log` 300–360.
- Files (image `vllm-pp6:7d0b4e57a` = upstream 7d0b4e57a; fork clone `/home/luk/dev/vllm`, branch
  `pp-nvidia-qwen`; `ops/*` and `indexer_qsa.py` are unchanged vs upstream, `nvidia/qsa.py` carries our 7-line
  PP KV-projection patch and is already in the overlay). All under `vllm/models/qwen4_exp/nvidia/`:
  - `ops/qsa_prepare.py` — `_to_dst_dtype(x, dst, scale)` (84–90): `x.to(bf16)`; `if out_ty == tl.float8e4nv:
    x = x.to(f32) / scale`; `return x.to(out_ty)`. Used by `_store_rotated` (94–101) and the main K/V store
    (228–230): K via `_store_rotated(dst, y, o1, o2, main_k_scale)`, V via `tl.store(dst + MAIN_D + dims,
    _to_dst_dtype(v, dst, main_v_scale))`. `dst` is derived from `main_cache_ptr` = host `main_kv_cache`
    (634); the host function `qsa_prepare(...)` (504) takes `main_kv_cache` whose dtype decides `out_ty`.
    Q output (`main_q_out_ptr`, bf16) also goes through `_store_rotated` with scale `None` → untouched.
  - `indexer_qsa.py` 308–309: `if attn.kv_cache_dtype in ("fp8","fp8_e4m3"): main_kv_cache =
    main_kv_cache.view(torch.float8_e4m3fn)` before calling `qsa_prepare` (310).
  - `ops/qsa.py` — `_qsa_sparse_paged_gqa_splitk_kernel` (31–235), constexpr `IS_FP8` (72); loads `keys`
    (135–143, `other=0.0`) and `values` (144–151) from `k_cache_ptr`/`v_cache_ptr`; `if IS_FP8: keys =
    keys.to(query.dtype)` (153–156); `if IS_FP8: values = values.to(tl.float16)` (168–172) then
    `tl.dot(probabilities.to(values.dtype), values, acc=…)`. Host `qsa_sparse_paged_attention` (594):
    `is_fp8 = k_cache.dtype == torch.float8_e4m3fn` (639), asserts `k_scale/v_scale`, folds them into
    `softmax_scale`/`output_scale`; passes `IS_FP8=is_fp8` (734). `warmup_qsa_sparse_paged_attention` (761):
    `is_fp8 = kv_cache.dtype == torch.uint8`, `cache_dtype = torch.float8_e4m3fn if is_fp8 else …` (772–773)
    → `TritonWarmupTensor(cache_dtype, …)` for k/v (795–803), `IS_FP8=is_fp8` (877). `_is_sm120()/_is_sm90()`
    (20–29) use `current_platform.get_device_capability()` with `lru_cache` — same pattern for our gate.
  - `nvidia/qsa.py` (module) 297–312: `key_cache, value_cache = kv_cache.transpose(1, 2).split(head_size)`;
    for fp8: `.view(torch.float8_e4m3fn)` both + `k_scale = layer._k_scale_float` (= 1.0, no scales in the
    checkpoint); then the dtype check `key_cache.dtype not in (bf16, float8_e4m3fn)` → NotImplementedError.
    Cache storage is uint8 (462–464).
  - `ops/qsa_indexer.py` 371/405/566 — only when `indexer_kv_dtype=fp8`; we keep bf16 → out of scope.
- Semantics (probe on GPU 6, `scratchpad/probe_e4m3.py`): torch CPU, torch CUDA and Triton native
  `.to(tl.float8e4nv)` all agree: **saturating** — 448, 464, 470, 480, 500, 1e9, +inf → `0x7E` (448);
  −500 → `0xFE`; 2⁻⁹ → `0x01`; 2⁻¹⁰ → `0x00` (tie → even 0); 1.5·2⁻⁹ → `0x02`; 2.5·2⁻⁹ → `0x02` (RNE).
  Decode: `0x7F`/`0xFF` = NaN; `0x7E` = 448; `0x08` = 2⁻⁶ (min normal); `0x01` = 2⁻⁹. So **torch CPU
  is a byte-exact oracle** and the 5060 Ti additionally runs the native kernels for end-to-end comparison.
- e4m3fn layout: `s eeee mmm`, bias 7; normal value `(1 + m/8)·2^(e−7)` for e ≥ 1; subnormal `m·2⁻⁹`;
  max 448 (`0x7E`); `e=15,m=7` is NaN; no inf.
- Triton facts used: dtype comparisons inside `@triton.jit` are compile-time (`if out_ty == tl.float8e4nv`
  already in the file; `x.dtype == tl.uint8` works the same way); `x.to(tl.uint32, bitcast=True)`
  reinterprets; integer `>>` on unsigned is logical; narrowing `.to(tl.uint8)` truncates; `tl.where`
  selects. A `tl.load` from a uint8 pointer with `other=0.0` is fine (0).
- Host has no torch. Everything that imports torch/triton runs in the image on GPU 6:
  `docker run --rm --gpus '"device=6"' -v <overlay mounts> -v <test dir>:/t:ro --entrypoint python3
  vllm-pp6:7d0b4e57a /t/test.py` (note: `--entrypoint python3 -` with a heredoc gets NO stdin — pass a file).
  pi workers have no docker/GPU (worker contract) → **the supervisor runs every GPU test** and bounces
  results back; workers write Python against copies of the sources in `work/P11/src/`.
- Overlay: `work/T3/run-vllm-pp6.sh` and the launcher mount every `*.py` under the overlay at
  `$SITE/<rel>:ro`. A NEW file (`ops/fp8_ampere.py`) is mounted the same way (docker creates the mount
  point) — VERIFY once with `docker run … ls $SITE/vllm/models/qwen4_exp/nvidia/ops/` (P11s step 6).

## 2. Design

New module `vllm/models/qwen4_exp/nvidia/ops/fp8_ampere.py` (the only new file; both kernels import it):
```python
import os
from functools import lru_cache
import torch
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

@lru_cache(maxsize=1)
def emulate_e4m3() -> bool:
    """True when Triton cannot cast e4m3 on this GPU (sm < 89) or VLLM_QSA_EMULATE_E4M3=1 forces it."""
    v = os.environ.get("VLLM_QSA_EMULATE_E4M3")
    if v is not None:
        return v == "1"
    cap = current_platform.get_device_capability()
    return cap is not None and cap.to_int() < 89

@triton.jit
def f32_to_e4m3_bits(x):
    """fp32 -> e4m3fn code (uint8), round-to-nearest-even, saturating to +-448, NaN -> 0x7F|sign."""
    b = x.to(tl.float32).to(tl.uint32, bitcast=True)
    sign = b & 0x80000000
    a = b ^ sign                                  # |x| bits
    is_nan = a > 0x7F800000
    a = tl.minimum(a, 0x43E00000)                 # clamp |x| to 448.0 (then RNE can never reach 0x7F)
    # subnormal (< 2^-6): add 2^14 so fp32's own RNE lands on multiples of 2^-9, read them back
    sub = (a.to(tl.float32, bitcast=True) + 16384.0).to(tl.uint32, bitcast=True) - 0x46800000
    # normal: rebias 127->7 (subtract 120<<23), RNE on the 20 dropped mantissa bits, keep 8 bits
    odd = (a >> 20) & 1
    nrm = (a - 0x3C000000 + 0x7FFFF + odd) >> 20
    r = tl.where(a < 0x3C800000, sub, nrm)
    r = tl.where(is_nan, 0x7F, r)
    return (r | (sign >> 24)).to(tl.uint8)

@triton.jit
def e4m3_bits_to_f16(b):
    """e4m3fn code (uint8) -> fp16 value, exact (incl. subnormals; 0x7F/0xFF -> NaN)."""
    c = b.to(tl.uint16)
    s = (c & 0x80) << 8
    e = (c >> 3) & 0xF
    m = c & 0x7
    nrm = (s | ((e + 8) << 10) | (m << 7)).to(tl.float16, bitcast=True)
    sub = (m.to(tl.float32) * 0.001953125).to(tl.float16)
    sub = tl.where(s != 0, -sub, sub)
    r = tl.where(e == 0, sub, nrm)
    return tl.where((e == 15) & (m == 7), float("nan"), r)
```
(Exact constants: 0x43E00000 = 448.0f; 0x46800000 = 16384.0f; 0x3C800000 = 2⁻⁶; 0x3C000000 = 120<<23.
The test decides whether `tl.where(…, float("nan"), r)` must be written as `r / 0 * 0` or via a NaN
constant tensor — Triton quirks; document in NOTES.)

Integration (store side, `ops/qsa_prepare.py`): `_to_dst_dtype` gets a third branch
```python
    if out_ty == tl.float8e4nv:
        x = x.to(tl.float32) / scale
    elif out_ty == tl.uint8:                       # Ampere: e4m3 emulated, cache passed as uint8
        return f32_to_e4m3_bits(x.to(tl.float32) / scale)
    return x.to(out_ty)
```
No new constexpr: the pointer dtype IS the switch. Host side (`indexer_qsa.py` 308–309): view as
`float8_e4m3fn` only when `not emulate_e4m3()`; under emulation leave the uint8 view.

Integration (load side, `ops/qsa.py`): in the kernel
```python
        if IS_FP8:
            if keys.dtype == tl.uint8:
                keys = e4m3_bits_to_f16(keys).to(query.dtype)
            else:
                keys = keys.to(query.dtype)
        ...
        if IS_FP8:
            if values.dtype == tl.uint8:
                values = e4m3_bits_to_f16(values)
            else:
                values = values.to(tl.float16)
```
Host: `is_fp8 = k_cache.dtype in (torch.float8_e4m3fn, torch.uint8)` (639, keep the bf16 assert in the else);
warmup `cache_dtype = (torch.uint8 if emulate_e4m3() else torch.float8_e4m3fn) if is_fp8 else key_cache.dtype`.
Module `nvidia/qsa.py` 298–312: do the `.view(torch.float8_e4m3fn)` only when `not emulate_e4m3()`;
accept `torch.uint8` in the dtype check when emulating. Scales unchanged (folded on the host).

Cost: load = ~8 integer ops + 2 selects per element in a bandwidth-bound kernel; store = ~12 ops per
element once per token per layer. Measured in W1 (fp8 vs bf16 decode at 250k).

## 3. Tasks (≤ 5 workers in parallel + supervisor; production up throughout)

| # | Task | Who | Depends | Acceptance |
|---|---|---|---|---|
| P11s | Scaffold (sup): branch `ampere-e4m3-kv` in `/home/luk/dev/vllm` from `pp-nvidia-qwen`; `work/P11/src/` = copies of the 4 source files + `nvidia/qsa.py` from the overlay; write `ops/fp8_ampere.py` per §2 into the branch AND `work/P11/src/`; task files `tasks/P11[a-e].md`; launch workers | sup | — | files exist, `py_compile` clean |
| P11a | Helper unit test `work/P11a/test_helpers.py`: tiny `@triton.jit` wrappers calling `f32_to_e4m3_bits` / `e4m3_bits_to_f16` elementwise on 1-D tensors; cases: all 256 codes (load; NaN==NaN), store on 2 M fp32 = log-uniform |x| in [2⁻¹⁴, 2¹⁰] both signs + every exact e4m3 value + every midpoint between adjacent e4m3 values (ties, both signs) + {±0, ±inf, NaN, ±448, ±464, ±470, ±480, ±2⁻⁹, ±2⁻¹⁰, ±1.5·2⁻⁹}; oracle = `x.to(torch.float8_e4m3fn).view(torch.uint8)` on CPU; byte-exact; prints counts and the first 10 mismatches; exit 1 on any mismatch. Also a round trip: `e4m3_bits_to_f16(f32_to_e4m3_bits(x))` == torch's `x.to(e4m3).to(f16)`. | pi | P11s (reads `src/ops/fp8_ampere.py`) | sup runs it on GPU 6: `ALL PASS` |
| P11b | Store-side integration in copies: `work/P11b/src/ops/qsa_prepare.py` (+ import, `_to_dst_dtype` branch), `work/P11b/src/indexer_qsa.py` (gate the view), `work/P11b/src/qsa.py` (module: gate the views + dtype check). Produce `work/P11b/P11b.diff` (`diff -u` vs `work/P11/src`), minimal, no reformatting. | pi | P11s | `py_compile` clean; diff ≤ 40 lines; sup review |
| P11c | Load-side integration in copies: `work/P11c/src/ops/qsa.py` (import; kernel branches; host `is_fp8`; warmup dtype). `work/P11c/P11c.diff`. | pi | P11s | `py_compile` clean; diff ≤ 40 lines; sup review |
| P11d | E2E attention test `work/P11d/test_attention_e2e.py` (runs in the image on GPU 6 with the merged overlay mounted): build a random uint8 K/V cache `[blocks=64, page=16, kv_heads=2, 2*head_dim]` (head_dim 128; values = random bytes excluding NaN codes 0x7F/0xFF), packed `logical_indices` (int32 `[rows, TOPK+1]`, trailing count column), `block_table`, `token_to_req`, bf16 `q` and `output_gate`; call `qsa_sparse_paged_attention` twice on the same bytes: (A) native: caches `.view(torch.float8_e4m3fn)`, (B) emulated: `VLLM_QSA_EMULATE_E4M3=1`, caches as uint8; both with `k_scale=v_scale=1.0`, `use_prefill_config` False and True, rows ∈ {1, 5, 64}, TOPK ∈ {64, 2048}; assert bitwise equality of outputs (`torch.equal`) and print max abs diff; exit 1 on any difference. Must read `work/P11/src/ops/qsa.py` for the exact argument contract (packed count column, strides, asserts). | pi | P11s | sup runs on GPU 6: `ALL PASS` |
| P11e | **Best effort** E2E prepare test `work/P11e/test_prepare_e2e.py`: call `qsa_prepare` with minimal valid indexer inputs (read the kernel's asserts/shapes in `work/P11/src/ops/qsa_prepare.py` and the call site in `work/P11/src/indexer_qsa.py` 295–340) and a main K/V cache in both modes; assert identical main cache bytes + identical `main_q_out/main_gate_out`. If the indexer inputs cannot be constructed in ≤ 60 turns, fall back to testing `_store_rotated`/`_to_dst_dtype` through a small wrapper kernel (uint8 dst vs e4m3 dst, random `y/o1/o2`, scale 1.0 and 0.5) and say so in RESULT.md. | pi | P11s | sup runs on GPU 6: `ALL PASS` (either variant) |
| P11m | Merge (sup): apply P11b/P11c diffs onto the branch, run P11a/P11d/P11e on GPU 6 against the branch files (overlay mounts), fix small things in place, commit (`git commit` on `ampere-e4m3-kv`, no Claude trailers), regenerate `work/T1-patches/overlay` (`git diff 7d0b4e57a --name-only -- vllm/` → copy), VERIFY the new file mounts, `RUN.md` Fact "P11". | sup | P11a–e | all three tests PASS on GPU 6 with the final overlay |
| W1 / W-Q8KV-2 | GPU window on the 3090s (needs the user's go): `KV_DTYPE=fp8 NSEQ=5 MAXLEN=262144` via `work/W-Q8KV/steps.sh` unchanged → KV tokens, 5×250k Running, needle, parsers. | sup/sub | P11m | §0 item 4 |

Waves: **A** = P11s (≈20 min) → **B** = P11a, P11b, P11c, P11d, P11e in parallel (5 workers, 1–2 h) →
**C** = P11m (sup, GPU 6 only) → **D** = the window, when scheduled. Bounce rule: a failing test or a
review finding goes back as `## Round 2` in the worker's TASK.md with the verbatim output (fresh worker,
max 2 bounces); the supervisor may fix ≤ 10-line issues directly.

Worker launch (from `vllm-pp6/`, production healthy): the `§1.8` recipe of `PLAN-PLE-SSD-Q8KV.md`
(`pi -p --mode json --permission-mode auto --no-approve --tools read,grep,find,ls,bash,edit,write -e
/home/luk/dev/pi-context-budget … --append-system-prompt worker-contract.md`), pid in `.runs/P11x.pid`,
recorded in `RUN.md`; kill only by pid. Poll every 15 min (cron) — `RESULT.md` present = done.
Workers may read `/home/luk/dev/vllm` (read-only) and `work/P11/src/`; they edit only inside their own
`work/P11x/`. No docker, no GPU, no network, no production port.

Supervisor GPU-6 test command (one container per test, overlay = branch files):
```bash
cd /home/luk/dev/airun/vllm-pp6; SITE=/usr/local/lib/python3.12/dist-packages; M=()
for f in $(cd /home/luk/dev/vllm && git diff 7d0b4e57a --name-only -- vllm/ ; echo vllm/models/qwen4_exp/nvidia/ops/fp8_ampere.py); do M+=(-v "/home/luk/dev/vllm/$f:$SITE/$f:ro"); done
timeout 900 docker run --rm --gpus '"device=6"' "${M[@]}" -v "$PWD/work/P11d:/t:ro" -e VLLM_QSA_EMULATE_E4M3=0 --entrypoint python3 vllm-pp6:7d0b4e57a /t/test_attention_e2e.py
```
(the tests flip `VLLM_QSA_EMULATE_E4M3` themselves per sub-case where needed — because `emulate_e4m3()`
is `lru_cache`d, a test that needs both modes in one process must call `emulate_e4m3.cache_clear()` after
setting the env var, or pass the uint8/e4m3 views explicitly to the ops functions, which is what P11d does.)

## 4. Risks / open points
- Triton NaN constant in `tl.where`: may need `tl.full([], float("nan"), tl.float16)`; P11a finds out.
- `keys.dtype == tl.uint8` inside a jit function: if Triton rejects the comparison form, fall back to a
  constexpr `EMULATE_E4M3` passed from the host (host knows `k_cache.dtype`). Same for `_to_dst_dtype`.
- `tl.dot(probabilities.to(values.dtype), values)` with fp16 values stays as is (emulation returns fp16).
- `keys` is loaded with `other=0.0` — for uint8 `other=0` must be an int: use `other=0` when the pointer
  is uint8 (Triton may coerce; P11d catches it).
- The sm_86 compile of the emulated kernels is first exercised in the window (GPU 6 is sm_120); the logic is
  arch-independent, but register pressure of the splitk kernel with the extra ops is unknown until then.
- Performance is not measured here; W1/W2 of the parent plan do.

## 5. Not promised
Indexer fp8 (`ops/qsa_indexer.py`), fp8 for other backends, calibrated kv scales, upstreaming.
