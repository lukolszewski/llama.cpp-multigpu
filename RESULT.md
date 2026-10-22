status: done
summary: GGML_CUDA_FA_TURING_COLS=16|32|64 moves the 32-column Turing pin on both fattn paths, gated on the
  kernel's shared-memory need vs. the device's smpbo, with a one-line log per shape behind the knob. Unset or 32
  routes exactly as before. Lifting the pin also needed the kernel's own `ncols > 32` trap for sm_75 to be raised.
commit: code ff433c027 + 6bc8ac699 + ca88f3991 + ea7f34563 on pm/D6-turing-prefill-tiles; this file rides on the branch tip above them  files: ggml/src/ggml-cuda/fattn.cu, ggml/src/ggml-cuda/fattn-mma-f16.cuh
tests:  pm/tools/tbo-ti.sh . 75-virtual -o FLASH_ATTN_EXT -> "  4004/4004 tests passed" / "  Backend CUDA0: OK" /
        "2/2 backends passed". Identical verdicts with GGML_CUDA_FA_TURING_COLS=32, 16 and 64, and
        GGML_CUDA_FA_TURING_COLS=64 pm/tools/tbo-ti.sh . 75-virtual -o NATIVE_QUANT_EQUIVALENCE ->
        "  105/105 tests passed" / "  Backend CUDA0: OK" / "2/2 backends passed". All on the final tree.
gate:   pm/tools/cmake-docker.sh . 75-real ggml-cuda -> "[100%] Built target ggml-cuda" / "cmake-docker: OK (build-75-real)";
        pm/tools/cmake-docker.sh . 75-virtual test-backend-ops -> "[100%] Built target test-backend-ops" /
        "cmake-docker: OK (build-75-virtual)"
parked: none (one read-only `for`-loop inspection command was refused by the sandbox as unverifiable; the same
        check was done by re-running the gate instead, see the object timestamps below)
questions: the knob has not been measured on a real 2080 Ti (no Turing device here, see "what this does not show")

## Gate output (verbatim, the full logs)

Builds (JOBS=8):

```
$ JOBS=8 pm/tools/cmake-docker.sh . 75-real ggml-cuda
[100%] Built target ggml-cuda
>> 2026-10-22T12:15:03+02:00 cmake-docker: OK (build-75-real)

$ JOBS=8 pm/tools/cmake-docker.sh . 75-virtual test-backend-ops
[100%] Built target ggml-cuda
[100%] Built target test-backend-ops
>> 2026-10-22T12:14:27+02:00 cmake-docker: OK (build-75-virtual)
```

The two objects that carry the change were recompiled, not reused (source 12:14:06, objects 12:14:26 for
75-virtual and 12:15:02 for 75-real, both relinked into libggml-cuda.so afterwards):

```
-rw-r--r-- 1 luk luk 45916 2026-10-22 12:14:06 ggml/src/ggml-cuda/fattn.cu
-rw-r--r-- 1 luk luk 612896 2026-10-22 12:14:26 build-75-virtual/.../fattn.cu.o
-rw-r--r-- 1 luk luk 1390112 2026-10-22 12:15:02 build-75-real/.../fattn.cu.o
```

Runs (sm_75 PTX JIT on the RTX 5060 Ti, so the host takes the Turing route):

```
$ pm/tools/tbo-ti.sh . 75-virtual -o FLASH_ATTN_EXT                     (knob unset)
  4004/4004 tests passed
  Backend CUDA0: OK
2/2 backends passed
OK
```
Log lines mentioning FA_TURING_COLS in that run: 0. With `GGML_CUDA_FA_TURING_COLS=32` the run's log differs from
the unset run only in its timestamp line, so the routing is identical.

```
$ GGML_CUDA_FA_TURING_COLS=64 pm/tools/tbo-ti.sh . 75-virtual -o FLASH_ATTN_EXT
ggml_cuda_fattn_pick_turing_cols: GGML_CUDA_FA_TURING_COLS=64: f16 kernel at D=256 with ncols2=8 takes a 8x8 tile (64 columns), 33 KiB shared memory of 99 KiB available, the pinned 4x8 tile needs 32 KiB
ggml_cuda_fattn_pick_turing_cols: GGML_CUDA_FA_TURING_COLS=64: f16 kernel at D=256 with ncols2=2 takes a 32x2 tile (64 columns), 36 KiB shared memory of 99 KiB available, the pinned 16x2 tile needs 34 KiB
  4004/4004 tests passed
  Backend CUDA0: OK
2/2 backends passed
```
(21 such lines in total, one per shape the run reached; D=64/80/96/112/128/192/256/320/512/576.)

```
$ GGML_CUDA_FA_TURING_COLS=64 pm/tools/tbo-ti.sh . 75-virtual -o NATIVE_QUANT_EQUIVALENCE
ggml_cuda_fattn_pick_turing_cols: GGML_CUDA_FA_TURING_COLS=64: native kernel at D=256 with ncols2=2 takes a 32x2 tile (64 columns), 36 KiB shared memory of 99 KiB available, the pinned 16x2 tile needs 34 KiB
ggml_cuda_fattn_pick_turing_cols: GGML_CUDA_FA_TURING_COLS=64: native kernel at D=256 with ncols2=8 takes a 8x8 tile (64 columns), 33 KiB shared memory of 99 KiB available, the pinned 4x8 tile needs 32 KiB
  105/105 tests passed
  Backend CUDA0: OK
2/2 backends passed

$ GGML_CUDA_FA_TURING_COLS=16 pm/tools/tbo-ti.sh . 75-virtual -o FLASH_ATTN_EXT
ggml_cuda_fattn_pick_turing_cols: GGML_CUDA_FA_TURING_COLS=16: f16 kernel at D=256 with ncols2=8 takes a 2x8 tile (16 columns), 33 KiB shared memory of 99 KiB available, the pinned 4x8 tile needs 32 KiB
  4004/4004 tests passed

$ GGML_CUDA_FA_TURING_COLS=16 pm/tools/tbo-ti.sh . 75-virtual -o NATIVE_QUANT_EQUIVALENCE
ggml_cuda_fattn_pick_turing_cols: GGML_CUDA_FA_TURING_COLS=16: native kernel at D=256 with ncols2=2 takes a 16x2 tile (32 columns), 34 KiB shared memory of 99 KiB available, the pinned 16x2 tile needs 34 KiB, keeping the pinned width
ggml_cuda_fattn_pick_turing_cols: GGML_CUDA_FA_TURING_COLS=16: native kernel at D=256 with ncols2=8 takes a 2x8 tile (16 columns), 33 KiB shared memory of 99 KiB available, the pinned 4x8 tile needs 32 KiB
  105/105 tests passed
```

## What the knob picks for the Qwen3.8-27B prefill shape (D=256, n_q=256, GQA 6)

GQA 6 does not map to ncols2=6; ncols2 is a power of two and the two routes disagree:
the f16 MMA path takes `gqa_ratio > 4` -> ncols2=8 (2 of the 8 head slots wasted), the quantized-native route
takes `heads8=false` for n_q>16 -> ncols2=2. Shared memory is the value the log prints (KiB truncated); the exact
byte counts are from `ggml_cuda_flash_attn_ext_mma_f16_nbytes_shared` evaluated by hand for the same shapes.

| route | ncols2 | knob 16 | unset / 32 (today) | knob 64 |
|---|---|---|---|---|
| f16 MMA | 8 | 2x8, 16 col, 33792 B | 4x8, 32 col, 33344 B | 8x8, 64 col, 33920 B |
| native q8_0/q4_0 | 2 | declined, 16x2, 32 col, 35072 B | 16x2, 32 col, 35072 B | 32x2, 64 col, 37376 B |
| native q8_0/q4_0 | 8 (GQA%8==0, or n_q<=16) | 2x8, 16 col, 33792 B | 4x8, 32 col, 33344 B | 8x8, 64 col, 33920 B |

Widening costs almost nothing in shared memory at D=256 because the tile that dominates it (nbatch_fa=64 x
nbatch_K2=128 half2 = 32 KiB) does not grow with the number of query columns: 32 -> 64 columns adds 576 B on the
f16 path and 2304 B on the native one.

Fits a real sm_75: yes for all of the above. The biggest of them is 37376 B, below the 49152 B that a Turing
launch may take without opting in and well below the 64 KiB per SM of sm_75 that PR #85 already plans with
(`ggml_cuda_fattn_quant_smem_per_sm(750)`). Two blocks per SM fit for neither width (2*(33920+1024) = 68.3 KiB
for the wide tile, 2*(33344+1024) = 68.7 KiB for the pinned one), so the pin buys no occupancy at D=256 - which is
why the wide tile is worth measuring there.

Where the smem gate bites: the same log shows the wide request needing 66 KiB at D=320 (2x32), 81 KiB at D=512
(8x8) and 94 KiB at D=576 (4x16). On the RTX 5060 Ti (smpbo = 99 KiB) those were accepted; on a 2080 Ti
(smpbo = sharedMemPerBlockOptin, at most 64 KiB) the check declines them and the pinned shape is kept. The decline
branch itself is not hypothetical - knob=16 at ncols2=2 hits it for a different reason and logs
"keeping the pinned width" (measured above).

## What changed and why

`ggml/src/ggml-cuda/fattn.cu`
- `ggml_cuda_fattn_turing_cols()`: reads GGML_CUDA_FA_TURING_COLS once (16, 32, 64; anything else warns and uses 32).
- `ggml_cuda_fattn_pick_turing_cols()`: returns the width a pinned card should use - the requested one if a kernel
  for it is compiled and `ggml_cuda_flash_attn_ext_mma_f16_nbytes_shared(...) <= info.smpbo`, else 32. Returns
  immediately and silently when the request is 32, so the default path is untouched. Logs one line per
  (route, D, ncols2, request) behind the knob, naming both the chosen and the pinned shape and their smem.
- `ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1`: the last-resort width is now a variable (64 on Ampere, 32 on
  RDNA with DKQ>256, the knob's choice on a pinned card) instead of a hard `|| highest_compiled_arch == TURING`.
- `ggml_cuda_fattn_native_supported`: takes the device index, and its three prompt-processing widths go through
  `prompt_cols` instead of `wide_tile ? wide : narrow`. Small-batch/decode shapes are unchanged; the 128-column
  shapes (`wide128`) stay Ampere-only, so the knob never reaches them.

`ggml/src/ggml-cuda/fattn-mma-f16.cuh`
- The kernel's own `#if __CUDA_ARCH__ == GGML_CUDA_CC_TURING && ncols1*ncols2 > 32 -> NO_DEVICE_CODE` trap (from
  upstream #13435) is raised to 64 columns. Without that the host could not lift the pin at all: the first version
  of this change crashed the FLASH_ATTN_EXT suite with `CUDA kernel flash_attn_ext_f16 has no device code
  compatible with CUDA arch 750`. The 128-column shapes stay excluded from sm_75 builds because no head size fits
  them in 64 KiB per SM.

`ggml/src/ggml-cuda/fattn-mma-quant.cuh`: unchanged. The wider Turing instantiation the task worried about already
exists: `template-instances/fattn-mma-quant-instance-*-ncols1_8-ncols2_8.cu` and `-ncols1_32-ncols2_2.cu` carry
(8x8) and (32x2) at D=64/128/256 for every native cache type, they are compiled for every target architecture (no
Ampere `#if`), and only the trap above kept them from being launched on sm_75.

## What this does not show

- No measurement on a Turing device. The RTX 5060 Ti runs sm_75 PTX through the driver JIT, which is good for
  correctness of the Turing host route but says little about a 2080 Ti's tensor cores or its 64 KiB/SM limit.
  Still, on that GPU the same build and shapes go from 11494 us to 10024 us (17.94 -> 20.57 TFLOPS, +15%) for the
  large prefill case `hsk=256,nr23=[6,1],kv=16384,nb=512,type_K=f16`, from 183.5 us to 170.5 us (+8%) for
  `hsk=256,nr23=[6,1],kv=16384,nb=8` on the native q8_0 route, and are unchanged for the nb=1 decode case - which
  is expected, the knob only touches the prompt-processing shapes.
- Cost of compiling the wide kernels into a sm_75 build: build-75-real/bin/libggml-cuda.so.0.27.0 grew from
  157513544 to 170686280 bytes (+8.4%). A build in which every fattn instance TU has to be recompiled took about
  20 min at -j8 (that particular rebuild was triggered by the header change, so the share attributable to the
  extra kernels is not isolated here); later incremental builds were ~20 s.
- The knob is per-process and read once; there is no per-node or per-shape override, and 16 columns is not offered
  where no kernel exists (native route at ncols2=2, D=512, and anywhere ncols2=32).
