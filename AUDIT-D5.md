# D5 - PR #85 native-quant flash attention on Turing (sm_75) with our shape

Shape: 2x RTX 2080 Ti (cc 750, 64 KiB smem per block opt-in, 96 KiB per SM), D=256, 12 Q heads over 2 KV heads
(GQA 6), q8_0 K and V, mask on, no ALiBi, no softcap, KV up to 262144, n_q in {1, 2, 3, 256}.
Tree: a37cc4610 + PR #85 = 63353d9e5. Everything below was read or run in this tree.

## Conclusions

1. **Take the route at every one of our query batch sizes.** `ggml_cuda_fattn_native_supported()` returns true,
   the tile is compiled, and the shared-memory plan fits Turing's 64 KiB with 30 KiB to spare. With
   `--flash-attn-native-quants` the native kernel is asked about before VEC and before MMA_F16, so it wins.

   | n_q | tile on Turing | smem | vs smpbo 65536 | route without the flag |
   |-----|----------------|------|----------------|------------------------|
   | 1 (decode)     | 1x8  | 32912 B | fits | VEC, reads every K/V row once per Q head |
   | 2 (2 slots)    | 2x8  | 33792 B | fits | MMA_F16, K/V cast to f16 first |
   | 3 (MTP verify) | 4x8  | 33344 B | fits | MMA_F16 |
   | 256 (prefill)  | 16x2 | 35072 B | fits | MMA_F16 |

2. **Nothing to add to the build.** `q8_0-q8_0` is in the default `GGML_CUDA_FA_QUANTS`, `q8_0` is a DEFAULT-tier
   type, so a plain `75-real` build has all 9 q8_0 D=256 configs x 2 KQ variants. On Turing only 1x8, 2x8, 4x8
   and 16x2 are reachable; 4x2, 8x8, 32x2, 16x8, 64x2 are dead weight (the two 128-column ones would need
   67584 B and could never fit a 2080 Ti anyway, but `wide_tile` keeps them off Turing regardless).
3. **Half KQ accumulation: keep the default, A/B it once.** A GeForce + sm_75 build takes the half variant. Only
   32 values of D are summed in half and those partials are added in float; the max subtraction, the row sums,
   the exp, the rescale and the stream-k fixup are float, so a 262144-long row never touches the half path. The
   fp16 accumulator a long row does stress is VKQ, which the F16-casting path has always had. The gate is NMSE
   <= 1e-4 against an f16 cache and it passes on the sm_75 path. The PR measured this on sm_86/sm_89 only and
   says sm_75 was not measured, so run one `-DGGML_FATTN_QUANT_KQ_FLOAT` A/B on the 2080 Ti before signing off.
4. **No silent fallback for this shape.** Turing loses the cp.async staging (synchronous loaders) and the
   128-column tiles, both by design, and nothing else applies to us (item 5).
5. **Test gap found and filled.** Shipped cases cover GQA 6 at D=256 only with 4 KV heads at kv <= 4096; 2 KV
   heads appear only at GQA 12/16 and only f16 or sparse. Added 6 cases; both gates pass, and they assert through
   the backend route counter that the native kernel really launched at n_q = 1, 2, 3 and 256.
6. **Free win at 262k**: the f16 copy of the visible window is not allocated at all any more
   (`fattn.cu:941-969`, `ggml-cuda.cu:973`, `fattn-common.cuh:53-80`). At D=256 with 2 KV heads that is 2 KiB per
   token, 512 MiB for a 262144-token window, off the compute buffer.

## Evidence

### 1 What compiles, what Turing can reach

- Default pairs: `ggml/CMakeLists.txt:208` = `"q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16"`; q8_0 is tier DEFAULT
  (`fattn-mma-quant-types.h:6-11`) so no CMake change is needed - only EXTRA pairs are gated
  (`fattn-mma-quant-types.h:22-24`, `ggml/src/ggml-cuda/CMakeLists.txt:129-136`).
- Shapes declared: `fattn-mma-quant-decl.cuh:24-40` (7 per head size, plus 64x2 and 16x8 at D=128/256), one
  generated file per pair, e.g. `template-instances/fattn-mma-quant-instance-q8_0-ncols1_16-ncols2_2.cu`;
  `DECL_FATTN_MMA_QUANT_CASE` (`fattn-mma-f16.cuh:2500-2504`) builds both KQ variants per shape.
- `scripts/fattn-native-inventory.py` runs offline against the built library and passes (item 7): 58 cases, q8_0
  D=256 rows 1x8, 2x8, 4x2, 4x8, 8x8, 16x2, 16x8, 32x2, 64x2, 2 symbols each.
- `wide_tile = ampere_mma_available(750) = false` (`fattn.cu:660`), so `fattn.cu:693-703` yields ncols2=8 with
  ncols1 in {1, 2, 4} for n_q <= 16 at GQA 6 and 16x2 above. 8x8/16x8/32x2/64x2 need `wide_tile`; 4x2 needs
  `heads8 = false` with n_q <= 4, which GQA 6 never gives. Any other width hits
  `GGML_ASSERT(ncols1 == 32/ncols2)` (`fattn.cu:459`), so a selector/instance mismatch aborts instead of
  silently falling back.

### 2 Route and shared memory at cc=750

`ggml_cuda_fattn_native_supported()` (`fattn.cu:632-706`) for our shape: Turing MMA yes, softcap 0,
`gqa_opt_applies` yes (mask, no ALiBi, `n_kv % 256 == 0` because the KV view length is padded to
`max(n_pad, 256)` (`llama-kv-cache.cpp:1290-1313`) and `FATTN_KQ_STRIDE` is 256 (`fattn-common.cuh:9`), strides
% 16), sparse no, pair compiled, `K->ne[0] == V->ne[0] == 256`, aligned. `gqa_ratio = 6` is neither a multiple of
8 nor odd above 4, so `heads8 = (n_q <= 16)` (`fattn.cu:681-687`) -> the tiles in the table. Alignment:
`copy_bytes(q8_0, 256) = 16` and a cache row is 272 B, so `aligned()` holds for the data pointer and nb[1..3] of
`get_k`/`get_v` (`llama-kv-cache.cpp:1313`, `:1336`).

The smem figures are the tree's own planner, `ggml_cuda_flash_attn_ext_mma_f16_nbytes_shared`
(`fattn-mma-f16.cuh:2325-2372`), compiled for sm_75 out of tree (scratch file in `build-75-real/audit/`, not
committed) and called with cc=750, warp_size=32:

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

`cp_async_available(750)` is false (`common.cuh:300-302`, `:373-375`) so `nstages` is 0
(`fattn-mma-f16.cuh:370-371`) and `nstages_quant` with it (`:400-403`): no raw staging buffers, no shadow tiles
(`:457-461` wants nstages >= 2), no chunk mode. What is left is one 64 x 128 half2 tile (32768 B) plus the mask
tile (`ncols1*(nbatch_fa/2+4)*4`), or the combine buffer where it is bigger - 4 warps x 16 cols x 132 x 4 =
33792 B, which is what sets 2x8. `ggml_cuda_fattn_native_applies` compares that against `info.smpbo` =
`sharedMemPerBlockOptin` (`fattn.cu:714-731`, `ggml-cuda.cu:381`), 65536 B at cc 7.5: passes, 30464 B of
headroom at the worst case, and two blocks per SM (the configured occupancy, `fattn-mma-f16.cuh:96-99`) still fit
the 96 KiB per SM. The 128-column rows are the only ones over the limit, and a Turing device can never select
them: `ggml_cuda_highest_compiled_arch` for arch=750 never returns more than 750, even in a build that also
targets sm_80 (`common.cuh:160-183`).

### 3 Selector order

`ggml_cuda_get_best_fattn_kernel` (`fattn.cu:749`) enters the tensor-core branch at `fattn.cu:847` and asks
`ggml_cuda_fattn_native_applies` before anything else, returning `BEST_FATTN_KERNEL_MMA_NATIVE`
(`fattn.cu:851-856`). Without the flag the same shape falls through to `BEST_FATTN_KERNEL_VEC` for n_q = 1
(quantized K/V below Ada, `fattn.cu:872-875`) and to `BEST_FATTN_KERNEL_MMA_F16` for n_q = 3 and 256
(`fattn.cu:883`). The dispatcher launches the shape the selector stored (`fattn.cu:983-999`);
`ggml_cuda_flash_attn_ext_get_alloc_size` asks the same function, so no f16 window is reserved for a native
launch (`fattn.cu:941-969`).

### 4 Half KQ accumulation

- Selected by `ggml_cuda_fattn_quant_kq_half()` (`fattn-mma-f16.cuh:2374-2383`): name contains "GeForce"
  (`ggml-cuda.cu:393-394`; a 2080 Ti is not on the no-tensor-core list at `ggml-cuda.cu:395-402`) and the
  compiled arch is one of 750/860/890/1200 (`fattn-mma-quant.cuh:64-68`); the `kq_half = true` kernel pointer is
  then taken (`fattn-mma-f16.cuh:2449-2455`).
- In half: one `tile<16, 8, half2>` per 32 values of D (two m16n8k16 MMAs), converted with `__half22float2` into
  the float `KQ_C` (`fattn-mma-f16.cuh:834-874`), which on Turing is `tile<16, 16, float>` / `tile<16, 8, float>`
  (`:1337-1352`). At D=256 that is 8 half runs of 32 products added in float.
- In float, i.e. everything a 262k row actually runs through: running max, row sums, exp, per-tile rescale
  (`fattn-mma-f16.cuh:975-1180`), and the stream-k fixup that merges the partial numerators and denominators
  (`fattn-common.cuh:963-977`). The half accumulator that spans the whole row is VKQ
  (`T_C_VKQ = tile<16, 8, half2>`, `:1345`) - upstream on Turing, identical for the F16-casting path.
- Tolerance in the test: `nmse(native, f16) <= 1e-4` between the two halves of one graph, anything worse is a
  hard fail, and the backend-vs-CPU NMSE stays capped at 5e-4 (`tests/test-backend-ops.cpp:8644-8652`).
- So: the docs argument (`docs/quantized-native-flash-attention.md:82`) that a 32-value half run is negligible
  against the q8_0 cache error is sound for sm_75 too; only the measurement is missing. Keep half, A/B once.

### 5 Turing guards and where each one lands

| Guard | On sm_75 | Where it goes |
|---|---|---|
| `cp_async_available(750)` false (`common.cuh:300-302`) | `nstages = 0` -> synchronous loader `flash_attn_ext_quant_load_tile` (`fattn-mma-quant.cuh:84-146`); staged loaders are `#ifdef CP_ASYNC_AVAILABLE` (`:222-305`) | stays native, slower loaders |
| `nstages = 0` | no shadow tiles, no mask-in-registers, no chunked mode (`fattn-mma-f16.cuh:400-403`, `:457-461`) | stays native |
| `wide_tile` false (`fattn.cu:660`) | 32-column cap, no block-of-8-warps kernel | stays native |
| `(DKQ/2) % nbatch_K2 != 0` at D=512 with < 32 columns | `NO_DEVICE_CODE` (`fattn-mma-f16.cuh:812-815`); supported() forces 4x8 at D=512 on Turing for exactly this reason (`fattn.cu:665-670`). At D=256, `nbatch_K2 = 128 = D/2` for all four tiles | D=512 only, not us |
| flag off (`fattn.cu:710-712`) | route declined | VEC at n_q=1, MMA_F16 otherwise |
| softcap, no mask, ALiBi, `n_kv % 256`, stride % 16, `n_kv_max > 0`, mixed K/V types, unaligned rows, smem > smpbo (`fattn.cu:632-731`) | route declined | VEC / MMA_F16 (cast). None of these trips on our shape |
| non-GeForce device name | float KQ variant | same results |

Decode still fills the card: `launch_fattn` runs with `stream_k = true` (`fattn-mma-f16.cuh:2491-2492`); at n_q=1
with 2 KV heads `ntiles_dst = 2`, `should_use_stream_k` sees ~1% efficiency against `occupancy(2) x nsm` and
splits the KV range over the device with the float fixup (`fattn-common.cuh:1141-1190`). At n_q=256 the 16x2 tile
reads each K/V row 3 times per KV head (6 heads over 2 per tile) against once for the F16 path's 4x8, but at 8.5
bits per value instead of a cast plus a 16-bit read.

### 6 Tests added (`tests/test-backend-ops.cpp`)

Shipped coverage at D=256/GQA 6 is `nh=4` KV heads, kv <= 4096, nb <= 65; with 2 KV heads only GQA 12/16, and
then f16 or with `n_kv_max = 512/2048` (sparse, which declines the native route by design). Added:

- `:12192-12198` NATIVE_QUANT_EQUIVALENCE, `hsk=hsv=256, nh=2, nr23={6,1}` (12 Q heads over 2 KV heads), q8_0:
  kv=4096 at nb in {1, 2, 3, 256}, kv=8192 at nb=3, kv=16384 at nb=1. Asserts NMSE <= 1e-4 vs an f16 copy of the
  same cache and that the route counter moved (`fattn.cu:161-165`).
- `:12361-12364` FLASH_ATTN_EXT, same geometry, kv=4096 nb=1 and kv=8192 nb=3, dense (`n_kv_max=0`),
  `expected_native=1`, checked against the CPU reference.

Note for whoever reads the case names next: `nh` in this test is the number of **KV** heads, so our shape is
`nh=2, nr23={6,1}`; `nh=12, nr23={6,1}` would be 72 Q heads (same GQA ratio, same kernels, more compute).

### 7 Gates (run here, output in RESULT.md)

- `cmake-docker.sh . 75-real ggml-cuda` -> OK; `build-75-real/bin/libggml-cuda.so.0.27.0` = 157508760 B (150 MiB).
- `scripts/fattn-native-inventory.py build-75-real/bin/libggml-cuda.so` -> `OK: 58 declared cases, all compiled,
  nothing else`.
- `cmake-docker.sh . 75-virtual test-backend-ops` -> OK.
- `tbo-ti.sh . 75-virtual -o NATIVE_QUANT_EQUIVALENCE` -> `111/111 tests passed`, `Backend CUDA0: OK`.
- `tbo-ti.sh . 75-virtual -o FLASH_ATTN_EXT` -> `4006/4006 tests passed`, `Backend CUDA0: OK`.
  (CPU is skipped by the harness in non-grad mode, `tests/test-backend-ops.cpp:13771`.)

### 8 For the on-hardware run

1. Check `fa_native_quants = true` in the log and that the image really is a 75-real build.
2. `test-backend-ops -o NATIVE_QUANT_EQUIVALENCE` on the 2080 Ti: expect 111/111. A shared-memory failure in
   `cudaFuncSetAttribute` or the launch would mean the plan in item 2 is wrong for the real `smpbo`.
3. `llama-bench -fa 1 -fanq 0,1` at -p 1/-n 1 and -p 4096/-n 1 with 262k ctx. If tg regresses at n_q=1, the
   comparison is 1x8 vs VEC; if pp regresses, it is 16x2 (3 KV passes) vs 4x8.
4. Peak VRAM should drop by roughly 512 MiB per FA node window at 262k (item 6).
5. One A/B with `-DGGML_FATTN_QUANT_KQ_FLOAT` (`-DCMAKE_CUDA_FLAGS=-DGGML_FATTN_QUANT_KQ_FLOAT`): compare KLD or
   perplexity on a long sample, since sm_75 is the one architecture in the half-KQ list with no measurement.
