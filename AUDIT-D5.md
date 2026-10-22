# D5 - PR #85 native-quant flash attention on Turing (sm_75) with our shape

Shape: 2x RTX 2080 Ti (cc 750, 64 KiB smem per block opt-in, 96 KiB per SM), D=256, 12 Q heads over 2 KV
heads (GQA 6), q8_0 K and V, mask on, no ALiBi, no softcap, KV up to 262144, n_q in {1, 2, 3, 256}.
Tree: upstream a37cc4610 + PR #85 = 63353d9e5. Everything below was read or run in this tree.

## Conclusions

1. **Take the route at every one of our query batch sizes.** `ggml_cuda_fattn_native_supported()` returns true,
   a kernel is compiled for the tile, and the shared-memory plan fits Turing's 64 KiB with 30 KiB to spare.
   With `--flash-attn-native-quants` the native kernel is consulted before VEC and before MMA_F16, so it wins.

   | n_q | tile on Turing | smem | vs smpbo 65536 | route without the flag |
   |-----|----------------|------|----------------|------------------------|
   | 1 (decode)     | 1x8  | 32912 B | fits | VEC, reads each K/V row once per Q head |
   | 2 (2 slots)    | 2x8  | 33792 B | fits | MMA_F16, K/V cast to f16 first |
   | 3 (MTP verify) | 4x8  | 33344 B | fits | MMA_F16 |
   | 256 (prefill)  | 16x2 | 35072 B | fits | MMA_F16 |

2. **Nothing to add to the build.** `q8_0-q8_0` is in the default `GGML_CUDA_FA_QUANTS` and `q8_0` is in the
   DEFAULT tier, so a plain `75-real` build already compiles all 9 q8_0 D=256 configs x 2 KQ variants. On
   Turing only 1x8, 2x8, 4x8 and 16x2 are reachable; 8x8, 4x2, 32x2, 16x8, 64x2 are dead weight there (the two
   128-column ones need 67584 B and could never fit a 2080 Ti, but `wide_tile` already keeps them off Turing).
3. **Half KQ accumulation: keep it as the production default, A/B it once.** A GeForce + sm_75 build selects the
   half variant. Only 32 values of D are summed in half; those partials are added in float, and the max
   subtraction, the row sums, the exp, the rescaling and the stream-k fixup are all float, so a 262144-long
   softmax row never touches the half path. The fp16 exposure that a long row does hit is the VKQ accumulator,
   which the F16-casting path has always had. The gate is an NMSE of 1e-4 against an F16 cache and the cases
   pass on the sm_75 code path. The PR measured the accuracy of the half variant on sm_86 and sm_89 and says
   sm_75 was not measured, so run one `-DGGML_FATTN_QUANT_KQ_FLOAT` A/B on the 2080 Ti before signing off.
4. **No silent fallback for this shape.** Turing loses the cp.async staging (synchronous loaders) and the
   128-column tiles, both by design; the four guards that would drop us back to VEC/MMA_F16 are the flag,
   softcap, `gqa_opt_applies`, and the smem/alignment checks - none of them trips on this shape (item 5).
5. **Test gap found and filled.** The shipped cases cover GQA 6 at D=256 only with 4 KV heads, kv <= 4096, and
   never 2 KV heads with a 256-token ubatch at q8_0. Added 6 cases (item 6); both gates pass, and the new ones
   assert through the backend route counter that the native kernel really launched at n_q = 1, 2, 3, 256.
6. **Free win for the 262k context**: the f16 copy of the visible window is no longer allocated at all
   (`fattn.cu:941-969`, `ggml-cuda.cu:973`). At D=256 with 2 KV heads that is 2 KiB per token, 512 MiB for a
   262144-token window, gone from the compute buffer.

## Evidence

### 1 What compiles, and what Turing can reach

- Default list: `ggml/CMakeLists.txt:208` (`"q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16"`); `q8_0` is DEFAULT tier in
  `fattn-mma-quant-types.h:10`, so it needs no extra CMake switch. Only EXTRA types are gated per pair
  (`fattn-mma-quant-types.h:22-24`, `ggml/src/ggml-cuda/CMakeLists.txt:129-136`).
- Declared shapes: `fattn-mma-quant-decl.cuh:24-40` (7 per head size, plus 64x2 and 16x8 for D=128/256). The
  generated instances are one file per (type, ncols1, ncols2), e.g.
  `template-instances/fattn-mma-quant-instance-q8_0-ncols1_16-ncols2_2.cu`; `DECL_FATTN_MMA_QUANT_CASE`
  (`fattn-mma-f16.cuh:2500-2504`) instantiates the case, which builds both KQ variants.
- `python3 scripts/fattn-native-inventory.py build-75-real/bin/libggml-cuda.so` runs offline and passes:
  `OK: 58 declared cases, all compiled, nothing else`. The q8_0 D=256 rows (2 symbols each = float + half KQ):
  1x8, 2x8, 4x2, 4x8, 8x8, 16x2, 16x8, 32x2, 64x2.
- Reachable on Turing (`wide_tile = ampere_mma_available(cc) = false`, `fattn.cu:660`): `fattn.cu:693-703` picks
  ncols2=8 with ncols1 in {1, 2, 4} for n_q <= 16 at GQA 6, and ncols1=16, ncols2=2 above that. `4x8` is
  `32/ncols2`, the row `mma_quant_shape` ends on for Turing (`fattn.cu:435-460`); 8x8, 16x8, 32x2 and 64x2 need
  `wide_tile`, 4x2 needs `heads8 = false` with n_q <= 4, which GQA 6 never gives.

### 2 Route and shared memory for cc=750

`ggml_cuda_fattn_native_supported()` (`fattn.cu:632-706`) for D=256, GQA 6, mask, no softcap, q8_0/q8_0:
`turing_mma_available(750)` true, softcap 0, `gqa_opt_applies` true (mask, no ALiBi, `n_kv % 256 == 0` because
the KV view length is padded to `max(n_pad, 256)` - `llama-kv-cache.cpp:1290-1313`, `FATTN_KQ_STRIDE` is 256 -
`fattn-common.cuh:9`, strides `% 16`), not sparse, pair compiled, `K->ne[0] == V->ne[0] == 256`, aligned.
`gqa_ratio = 6` is not `% 8` and not odd, so `heads8 = n_q <= 16` (`fattn.cu:683-685`), giving the tiles in the
table above. Alignment: `copy_bytes(q8_0, 256)` is 16 and the KV row is 272 B (`ggml_row_size(q8_0, 256)`), so
`aligned()` holds for data, nb[1], nb[2] and nb[3] of the cache views (`llama-kv-cache.cpp:1313`, `:1336`).

The smem numbers are not hand-derived: they are the tree's own planner, `ggml_cuda_flash_attn_ext_mma_f16_nbytes_shared`
(`fattn-mma-f16.cuh:2325-2372`), compiled for sm_75 out of tree and called with cc=750, warp_size=32:

```
smem_per_sm(750) = 65536 bytes
cp_async_available(750) = 0, ampere_mma_available(750) = 0, turing_mma_available(750) = 1
ncols1x2   ncols      nthreads  nbatch_fa         K2    combine smem bytes  <=64K
1x8        8              128         64        128        128      32912    yes  nstages_quant=0
2x8        16              128         64        128        128     33792    yes  nstages_quant=0
4x8        32              128         64        128         64     33344    yes  nstages_quant=0
16x2       32              128         64        128         64     35072    yes  nstages_quant=0
8x8        64              128         64        128         64     33920    yes  nstages_quant=0
16x8       128             256         32        128        128     67584     NO  nstages_quant=0
32x2       64              128         64        128         64     37376    yes  nstages_quant=0
64x2       128             256         32        128        128     67584     NO  nstages_quant=0
q8_0 D=256: block_bytes=34 row=272 pitch=272 copy_bytes=16
```

Why it is that size on Turing: `cp_async_available(750)` is false (`common.cuh:300-302`, `:373-375`), so
`nstages` is 0 (`fattn-mma-f16.cuh:370-371`) and `nstages_quant` is 0 with it (`:400-403`). No staging buffer,
no shadow tiles (`:457-461` needs nstages >= 2), no chunk buffers. What remains is one 64 x 128 half2 tile
(32768 B) plus the mask tile (`ncols1*(nbatch_fa/2+4)*4`) or, at ncols=16, the combine buffer
(4 warps x 16 cols x 132 x 4 = 33792 B), which is the larger term for 2x8. `ggml_cuda_fattn_native_applies`
compares that against `info.smpbo` = `sharedMemPerBlockOptin` (`fattn.cu:714-731`, `ggml-cuda.cu:381`), 65536 B
on cc 7.5, so the gate passes with 30464 B of headroom at the worst case. Two blocks per SM (the configured
occupancy, `fattn-mma-f16.cuh:96-99`) still fit the 96 KiB per SM.

The 128-column rows are the only ones over the limit and they are unreachable on Turing twice over: `wide_tile`
never selects them, and for a Turing device `ggml_cuda_highest_compiled_arch` never reaches 800 even in a build
that also targets sm_80 (`common.cuh:160-183`).

### 3 Selector order

`ggml_cuda_get_best_fattn_kernel` (`fattn.cu:749`) enters the tensor-core branch at `fattn.cu:847`
(`turing_mma_available(cc) && Q->ne[0] != 40 && != 72`) and asks `ggml_cuda_fattn_native_applies` first
(`fattn.cu:849-856`), returning `BEST_FATTN_KERNEL_MMA_NATIVE` before any VEC or MMA_F16 test. Without the flag
the same shape falls through to `BEST_FATTN_KERNEL_VEC` at `fattn.cu:873-875` for n_q = 1 (quantized K/V, cc
below Ada, `Q->ne[1] == 1`) and to `BEST_FATTN_KERNEL_MMA_F16` at `fattn.cu:883` for n_q = 3 and 256.
`ggml_cuda_flash_attn_ext` then dispatches on the tile shape the selector stored (`fattn.cu:983-999`), and
`ggml_cuda_flash_attn_ext_get_alloc_size` (`fattn.cu:941-969`) asks the same function, so the f16 K/V window is
not reserved for a native launch.

### 4 Half KQ accumulation on sm_75

- Host selection: `ggml_cuda_fattn_quant_kq_half()` (`fattn-mma-f16.cuh:2374-2383`) = device name contains
  "GeForce" (`ggml-cuda.cu:393-394`; a 2080 Ti is "NVIDIA GeForce RTX 2080 Ti", and it is not on the
  tensor-core-less list at `ggml-cuda.cu:395-402`) and the compiled arch is 750/860/890/1200
  (`fattn-mma-quant.cuh:64-68`). The kernel pointer is then taken with `kq_half = true`
  (`fattn-mma-f16.cuh:2449-2455`).
- What is actually in half: one `tile<16, 8, half2>` per 32 values of D, two m16n8k16 MMAs
  (`fattn-mma-f16.cuh:834-874`). Each partial is converted with `__half22float2` and added into the float
  `KQ_C`, which on Turing is `tile<16, 16, float>` / `tile<16, 8, float>` (`:1337-1352`). So for D=256 there are
  8 independent half runs of 32 products, summed in float.
- What is not in half, and what a 262k row actually runs through: the running max, the row sums, the exp and the
  per-tile rescale (`KQ_max`, `KQ_rowsum`, `KQ_max_scale`, `fattn-mma-f16.cuh:975-1180`), the stream-k fixup,
  which combines the partial numerators and denominators in float (`fattn-common.cuh:963-977`), and the softmax
  logits after the mask add. The one accumulator that is half for the whole row is VKQ
  (`T_C_VKQ = tile<16, 8, half2>`, `:1345`), which is upstream behaviour on Turing and identical for the
  F16-casting path.
- Test tolerance: `test_flash_attn_ext::err` requires `nmse(native, f16) <= 1e-4` between the two halves of the
  same graph, and 1.0 (a hard fail) beyond that, with the CPU-vs-backend NMSE capped at 5e-4
  (`tests/test-backend-ops.cpp:8645-8660`).
- Verdict: the accuracy argument in the docs (`docs/quantized-native-flash-attention.md:82`) is that a 32-value
  half run is far below the fp16 range and small against the q8_0 cache error, and that the 256-value variant
  they rejected is a different thing. That reasoning holds for sm_75; only the measurement is missing, so keep
  the default and A/B once.

### 5 Turing-specific guards, and what each falls back to

| Guard | Effect on sm_75 | Fallback |
|---|---|---|
| `cp_async_available(750)` false (`common.cuh:300-302`) | `nstages = 0` -> synchronous loader `flash_attn_ext_quant_load_tile` (`fattn-mma-quant.cuh:84-146`); the staged loaders are `#ifdef CP_ASYNC_AVAILABLE` (`:222-305`) | stays native, slower loaders |
| `nstages = 0` | no shadow tiles, no mask-in-registers, no chunked mode (`fattn-mma-f16.cuh:400-403`, `:457-461`) | stays native |
| `wide_tile = ampere_mma_available(cc)` false (`fattn.cu:660`) | tiles capped at 32 columns, no 128-column block-per-SM kernel | stays native |
| `(DKQ/2) % nbatch_K2 != 0` at D=512 with < 32 columns | `NO_DEVICE_CODE` (`fattn-mma-f16.cuh:812-815`); this is why supported() forces 4x8 at D=512 on Turing (`fattn.cu:665-670`). At D=256 `nbatch_K2 = 128 = D/2` for all four tiles | D=512 only, not us |
| flag off (`fattn.cu:710-712`) | no native | VEC at n_q=1, else MMA_F16 |
| softcap, no mask, ALiBi, `n_kv % 256`, strides % 16, `n_kv_max > 0`, mixed K/V types, unaligned rows, smem > smpbo (`fattn.cu:632-731`) | declines | VEC / MMA_F16 (cast), never a wrong kernel |
| `mma_f32_acc_half_rate` false (non-GeForce name) | float KQ variant | correct either way |

Decode on our shape fills the GPU: `launch_fattn` runs with `stream_k = true` (`fattn-mma-f16.cuh:2491-2492`),
and for n_q = 1 with 2 KV heads `ntiles_dst = 2`, so `should_use_stream_k` sees 1% efficiency against
`max_blocks = occupancy(2) x nsm` and splits the KV range over the whole device, with the float fixup pass
(`fattn-common.cuh:1141-1190`). At n_q = 256 the 16x2 tile reads each K/V row 3 times per KV head (6 heads / 2
per tile) against once for the F16 path's 4x8, at 8.5 bits per value instead of a cast plus a 16-bit read.

### 6 Test coverage added (`tests/test-backend-ops.cpp`)

The shipped suite covers GQA 6 at D=256 with `nh=4` KV heads (kv 512..4096, nb 1..65), 2-KV-head cases only at
GQA 12/16, only with `n_kv_max = 512/2048` (sparse) or f16, and never a 256-token ubatch at q8_0. Added:

- `:12192-12198`, NATIVE_QUANT_EQUIVALENCE, `hsk=hsv=256, nh=2, nr23={6,1}` (12 Q heads over 2 KV heads), q8_0:
  kv=4096 at nb in {1, 2, 3, 256}, plus kv=8192 nb=3 and kv=16384 nb=1. Each asserts NMSE <= 1e-4 against an f16
  copy of the same cache and asserts the native route through `ggml_backend_cuda_fattn_native_count`.
- `:12361-12364`, FLASH_ATTN_EXT, same geometry, kv=4096 nb=1 and kv=8192 nb=3, `expected_native=1`, dense
  (`n_kv_max=0`), checked against the CPU reference.

### 7 Gates

- `/home/luk/dev/airun/llama-build-kamserw2/pm/tools/cmake-docker.sh . 75-real ggml-cuda` -> `OK (build-75-real)`,
  `ls -l build-75-real/bin/libggml-cuda.so` -> `libggml-cuda.so.0.27.0`, 157508760 bytes (150 MiB).
- `scripts/fattn-native-inventory.py build-75-real/bin/libggml-cuda.so` -> `OK: 58 declared cases, all compiled,
  nothing else` (see item 1).
- `cmake-docker.sh . 75-virtual test-backend-ops` -> `OK (build-75-virtual)`.
- `tbo-ti.sh . 75-virtual -o NATIVE_QUANT_EQUIVALENCE` -> `111/111 tests passed`, `Backend CUDA0: OK`
  (CPU backend skipped by the harness in non-grad mode). All 6 new cases listed OK, e.g.
  `NATIVE_QUANT_EQUIVALENCE(hsk=256,hsv=256,nh=2,nr23=[6,1],kv=4096,nb=256,...,type_K=q8_0,...expected_native=1): OK`.
- `tbo-ti.sh . 75-virtual -o FLASH_ATTN_EXT` -> see RESULT.md for the pasted tail.

### 8 Checklist for the on-hardware run (2x 2080 Ti)

1. Confirm the log line `fa_native_quants = true` and that the build is `75-real` (a `75-virtual` build runs on
   the 2080 Ti but through PTX JIT, which is fine for results and slow to start).
2. `test-backend-ops -o NATIVE_QUANT_EQUIVALENCE` on the real card: expect the same 111/111. If a case aborts in
   `cudaFuncSetAttribute`/launch with a shared memory error, the smem plan above is wrong for the real `smpbo`.
3. `llama-bench -fa 1 -fanq 0,1 -m qwen3.8-27B-...-q8_0 -p 1 -n 1` and `-p 4096 -n 1` at 262k ctx: the native
   route must win at pp and at tg; if tg regresses at n_q=1, that is the 1x8 tile vs VEC and worth a look.
4. Peak VRAM should drop by the f16 window (KiB/token/KV-head x 262144) at long context.
5. One A/B of `-DGGML_FATTN_QUANT_KQ_FLOAT` (per docs line 82 sm_75 is unmeasured): compare perplexity/KLD on a
   long sample before deciding.
