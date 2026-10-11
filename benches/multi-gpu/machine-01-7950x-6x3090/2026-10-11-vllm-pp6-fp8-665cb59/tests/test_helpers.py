# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P11a -- byte-exact unit test for the two e4m3 software helpers.

Subject under test: ``vllm/models/qwen4_exp/nvidia/ops/fp8_ampere.py``
(``f32_to_e4m3_bits``, ``e4m3_bits_to_f16``, ``emulate_e4m3``).

Run inside the vLLM image on a GPU (no pytest, torch + triton only):

    python3 /t/test_helpers.py

The helpers are ``@triton.jit`` functions that work elementwise on Triton
tensors, so they are driven from the three tiny 1-D wrapper kernels below
(BLOCK = 1024, masked loads; uint8 loaded with ``other=0``, float with
``other=0.0``).  Every oracle is PyTorch's own e4m3fn conversion computed on
the CPU (verified byte-exact vs CUDA + native Triton in PLAN-P11 section 1)
and then moved to CUDA.

Cases
  1 LOAD         all 256 codes -> fp16, + NaN set, + -0.0 sign
  2 STORE        ~2.7M fp32 -> uint8 (exact values, RNE ties, ties +-1 ulp,
                 explicit edge list, 2.7M random) byte-exact
  3 ROUND TRIP   f16(e4m3(x)) == x.to(float8_e4m3fn).to(float16) on all of 2
  4 ENV          emulate_e4m3() + VLLM_QSA_EMULATE_E4M3 override + cache_clear

Exit 0 with the single last line ``ALL PASS``, else ``FAIL: <n> mismatches``
and exit 1.
"""

import os
import sys

import torch
import triton
import triton.language as tl

from vllm.models.qwen4_exp.nvidia.ops.fp8_ampere import (
    emulate_e4m3,
    e4m3_bits_to_f16,
    f32_to_e4m3_bits,
)

BLOCK = 1024
SEED = 20261011
MAX_SHOW = 10  # first N mismatches printed (over all groups)


def fail(failures, msg, n=1):
    """Record a failing check; n = number of mismatching elements it found."""
    failures.append((max(1, int(n)), msg))


# --------------------------------------------------------------------------
# wrapper kernels (the only Triton code in this file)
# --------------------------------------------------------------------------
@triton.jit
def _k_store(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """fp32 -> e4m3fn code (uint8), elementwise."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    tl.store(out_ptr + offs, f32_to_e4m3_bits(x), mask=m)


@triton.jit
def _k_load(b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """e4m3fn code (uint8) -> fp16, elementwise."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    b = tl.load(b_ptr + offs, mask=m, other=0)
    tl.store(out_ptr + offs, e4m3_bits_to_f16(b), mask=m)


@triton.jit
def _k_roundtrip(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """fp32 -> e4m3fn code -> fp16, elementwise (both helpers chained)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    tl.store(out_ptr + offs, e4m3_bits_to_f16(f32_to_e4m3_bits(x)), mask=m)


def _launch(kernel, in_t, out_t):
    n = in_t.numel()
    kernel[(triton.cdiv(n, BLOCK),)](in_t, out_t, n, BLOCK=BLOCK)
    return out_t


def run_load(codes_u8):
    """uint8 codes -> fp16 on the same (CUDA) device."""
    out = torch.empty_like(codes_u8, dtype=torch.float16)
    return _launch(_k_load, codes_u8, out)


def run_store(x_f32):
    """fp32 -> uint8 e4m3fn code."""
    out = torch.empty_like(x_f32, dtype=torch.uint8)
    return _launch(_k_store, x_f32, out)


def run_roundtrip(x_f32):
    """fp32 -> fp16 through both helpers."""
    out = torch.empty_like(x_f32, dtype=torch.float16)
    return _launch(_k_roundtrip, x_f32, out)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def fp32_bits(x_f32_cpu):
    """fp32 values as unsigned 32-bit ints (Python ints list-safe via int64)."""
    return x_f32_cpu.view(torch.int32).to(torch.int64) & 0xFFFFFFFF


def rand_fp32_bit_patterns(n):
    """n arbitrary fp32 bit patterns (NaN / inf / denormal / normal)."""
    try:
        b = torch.randint(-(2**31), 2**31, (n,), dtype=torch.int32)
    except Exception:  # older torch: randint(int32) range checks
        b = torch.randint(0, 2**32, (n,), dtype=torch.int64).to(torch.int32)
    return b.view(torch.float32)


def build_store_groups():
    """Return [(name, fp32 CPU tensor)] in the order they are concatenated."""
    codes = torch.arange(256, dtype=torch.uint8)
    # the 254 non-NaN codes (0x7F and 0xFF are the two NaN codes)
    finite_codes = torch.cat([codes[:0x7F], codes[0x80:0xFF]])
    vals = finite_codes.view(torch.float8_e4m3fn).to(torch.float32)
    groups = [("a_exact_values", vals)]

    # midpoints between adjacent finite values (RNE ties); -0.0/0.0 collapse in
    # unique(), so the sorted list runs ... -2^-9, 0.0, 2^-9 ... and both signs
    # of every tie show up.  The mean of two adjacent e4m3 values is exact in
    # fp32 (exponent difference <= 1).
    uniq = torch.unique(vals)
    mid = (uniq[:-1] + uniq[1:]) * 0.5
    groups.append(("b_midpoints_ties", mid))

    # the same ties +-1 fp32 ulp: just above / just below the tie
    pinf = torch.full_like(mid, float("inf"))
    ninf = torch.full_like(mid, float("-inf"))
    groups.append(("c_tie_plus_ulp", torch.nextafter(mid, pinf)))
    groups.append(("c_tie_minus_ulp", torch.nextafter(mid, ninf)))

    # explicit edge list (spec 2d), both signs
    mags = [
        0.0,
        float("inf"),
        448.0,
        449.0,
        464.0,
        470.0,
        480.0,
        500.0,
        1e9,
        2.0**-9,
        2.0**-10,
        1.5 * 2.0**-9,
        2.5 * 2.0**-9,
        0.5 * 2.0**-9,
        2.0**-6,
        0.9 * 2.0**-6,
        3.0,
        0.1,
        1e-30,
        1e30,
        3.4e38,
    ]
    expl = []
    for v in mags:
        expl.append(v)
        expl.append(-v)  # -0.0 for v == 0.0
    expl.append(float("nan"))
    expl.append(-float("nan"))
    groups.append(("d_explicit_edges", torch.tensor(expl, dtype=torch.float32)))

    # random (spec 2e)
    torch.manual_seed(SEED)
    sign = 1.0 - 2.0 * (torch.rand(2_000_000) < 0.5).to(torch.float32)
    mag = torch.exp2(torch.rand(2_000_000) * 24.0 - 14.0)  # 2**U(-14, 10)
    groups.append(("e_random_loguniform", sign * mag))
    groups.append(("e_random_randn50", torch.randn(500_000) * 50.0))
    groups.append(("e_random_bitpatterns", rand_fp32_bit_patterns(200_000)))
    return groups


def show_store_mismatches(name, xs, bits, got, ref, idx, shown):
    for i in idx[: max(0, MAX_SHOW - shown)].tolist():
        print(
            "    %s: x=%r bits=0x%08x got=0x%02x ref=0x%02x"
            % (name, float(xs[i]), int(bits[i]), int(got[i]), int(ref[i]))
        )
    return min(shown + len(idx), MAX_SHOW)


# --------------------------------------------------------------------------
# case 1: LOAD (all 256 codes)
# --------------------------------------------------------------------------
def case_load(failures):
    print("\n== case 1: LOAD  all 256 e4m3fn codes -> fp16 ==")
    codes = torch.arange(256, dtype=torch.uint8)
    ref16 = codes.view(torch.float8_e4m3fn).to(torch.float16)  # CPU oracle
    got16 = run_load(codes.cuda())
    ref16 = ref16.cuda()

    same = (got16 == ref16) | (torch.isnan(got16) & torch.isnan(ref16))
    bad = torch.nonzero(~same).flatten()
    print("  values: n=256 mismatches=%d" % bad.numel())
    for i in bad[:MAX_SHOW].tolist():
        print(
            "    code=0x%02x got=%r ref=%r"
            % (i, float(got16[i]), float(ref16[i]))
        )
    if bad.numel():
        fail(failures, "load: %d code(s) decode differently from torch" % bad.numel(), bad.numel())

    # exactly 0x7F and 0xFF are NaN (both our output and the oracle)
    for what, v in (("got", got16), ("ref", ref16)):
        nan_codes = torch.nonzero(torch.isnan(v)).flatten().tolist()
        ok = nan_codes == [0x7F, 0xFF]
        print("  NaN codes (%s): %s" % (what, [hex(c) for c in nan_codes]))
        if not ok:
            fail(failures, "load: NaN codes (%s) = %s, expected ['0x7f', '0xff']"
                 % (what, [hex(c) for c in nan_codes]))

    # 0x80 is -0.0: value 0.0 with the fp16 sign bit set
    z = got16[0x80]
    zneg = bool((got16.view(torch.int16) < 0)[0x80].item())
    zref_neg = bool((ref16.view(torch.int16) < 0)[0x80].item())
    print(
        "  0x80 -> %r signbit_set=%s (ref signbit_set=%s)"
        % (float(z), zneg, zref_neg)
    )
    if not (float(z) == 0.0 and zneg and zref_neg):
        fail(failures, "load: 0x80 is not -0.0 (got %r signbit=%s)" % (float(z), zneg))

    # sign bit of the fp16 output follows the code's sign bit everywhere
    want_neg = codes >= 0x80
    got_neg = got16.view(torch.int16) < 0
    nsign = int((want_neg.cuda() != got_neg).sum().item())
    print("  sign-bit mismatches: %d" % nsign)
    if nsign:
        fail(failures, "load: %d code(s) with wrong sign bit in the fp16 result" % nsign, nsign)


# --------------------------------------------------------------------------
# case 2 + 3: STORE and ROUND TRIP on the same x
# --------------------------------------------------------------------------
def case_store_and_roundtrip(failures):
    print("\n== case 2: STORE  fp32 -> uint8 (e4m3fn), byte-exact ==")
    groups = build_store_groups()
    x_cpu = torch.cat([g for _, g in groups]).to(torch.float32).contiguous()
    n = x_cpu.numel()
    bits = fp32_bits(x_cpu)
    n_nan = int(torch.isnan(x_cpu).sum().item())
    n_inf = int(torch.isinf(x_cpu).sum().item())
    print(
        "  x: n=%d (NaN=%d, inf=%d) seed=%d, oracle on CPU then moved to CUDA"
        % (n, n_nan, n_inf, SEED)
    )

    x_gpu = x_cpu.cuda()
    got = run_store(x_gpu)  # uint8, CUDA
    ref = x_cpu.to(torch.float8_e4m3fn).view(torch.uint8).cuda()

    got_i = got.to(torch.int32)
    ref_i = ref.to(torch.int32)
    is_nan = torch.isnan(x_cpu.cuda())

    # byte-exact, except: NaN inputs only have to land on *a* NaN code
    # (exponent == 15 and mantissa == 7 -> (code & 0x7F) == 0x7F), sign ignored
    ok_nan = ((got_i & 0x7F) == 0x7F) & ((ref_i & 0x7F) == 0x7F)
    ok = torch.where(is_nan, ok_nan, got_i == ref_i)

    total = 0
    shown = 0
    off = 0
    for name, g in groups:
        gn = g.numel()
        sl = slice(off, off + gn)
        bad = torch.nonzero(~ok[sl]).flatten()
        total += bad.numel()
        print("  %-22s n=%8d mismatches=%d" % (name, gn, bad.numel()))
        if bad.numel() and shown < MAX_SHOW:
            shown = show_store_mismatches(
                name, x_cpu[sl], bits[sl], got[sl].cpu(), ref[sl].cpu(), bad, shown
            )
        off += gn
    if total:
        fail(failures, "store: %d/%d values differ from torch" % (total, n), total)

    print("\n== case 3: ROUND TRIP  f16(e4m3(x)) == x.to(e4m3).to(f16) ==")
    got_rt = run_roundtrip(x_gpu)  # fp16, CUDA
    ref_rt = x_cpu.to(torch.float8_e4m3fn).to(torch.float16).cuda()
    same = (got_rt == ref_rt) | (torch.isnan(got_rt) & torch.isnan(ref_rt))
    badrt = torch.nonzero(~same).flatten()
    print("  round trip: n=%d mismatches=%d" % (n, badrt.numel()))
    for i in badrt[:MAX_SHOW].tolist():
        print(
            "    x=%r got=%r ref=%r" % (float(x_cpu[i]), float(got_rt[i]), float(ref_rt[i]))
        )
    if badrt.numel():
        fail(failures, "round trip: %d/%d values differ from torch" % (badrt.numel(), n),
             badrt.numel())

    # informational only (a real difference would be a bug report, not a FAIL:
    # the spec asks for value equality + NaN==NaN)
    zr = ref_rt == 0
    sign_diff = int(((got_rt.view(torch.int16) < 0) != (ref_rt.view(torch.int16) < 0))[zr].sum().item())
    print("  INFO: zero-sign disagreements (not counted): %d" % sign_diff)
    print("  INFO: store mismatches total = %d, round trip = %d" % (total, badrt.numel()))


# --------------------------------------------------------------------------
# case 4: emulate_e4m3() + env override
# --------------------------------------------------------------------------
def case_env(failures):
    print("\n== case 4: emulate_e4m3() / VLLM_QSA_EMULATE_E4M3 ==")
    cap = torch.cuda.get_device_capability(0)
    orig = os.environ.get("VLLM_QSA_EMULATE_E4M3")
    print("  torch.cuda.get_device_capability() = %s" % (cap,))
    print("  emulate_e4m3() = %r (env=%r)" % (emulate_e4m3(), orig))
    try:
        os.environ["VLLM_QSA_EMULATE_E4M3"] = "1"
        emulate_e4m3.cache_clear()
        v_one = emulate_e4m3()
        os.environ["VLLM_QSA_EMULATE_E4M3"] = "0"
        emulate_e4m3.cache_clear()
        v_zero = emulate_e4m3()
        del os.environ["VLLM_QSA_EMULATE_E4M3"]
        emulate_e4m3.cache_clear()
        v_def = emulate_e4m3()
    finally:
        if orig is not None:
            os.environ["VLLM_QSA_EMULATE_E4M3"] = orig
        emulate_e4m3.cache_clear()
    want_def = cap < (8, 9)
    print('  env "1" -> %r (want True)' % (v_one,))
    print('  env "0" -> %r (want False)' % (v_zero,))
    print("  no env  -> %r (want capability < (8, 9) = %r)" % (v_def, want_def))
    if v_one is not True:
        fail(failures, 'env="1": emulate_e4m3() returned %r, want True' % (v_one,))
    if v_zero is not False:
        fail(failures, 'env="0": emulate_e4m3() returned %r, want False' % (v_zero,))
    if bool(v_def) is not bool(want_def):
        fail(failures, "no env: emulate_e4m3() returned %r, want %r (cap=%s)"
             % (v_def, want_def, cap))


# --------------------------------------------------------------------------
def main():
    print(
        "P11a e4m3 helper test -- torch %s cuda %s triton %s"
        % (torch.__version__, torch.version.cuda, getattr(triton, "__version__", "?"))
    )
    if not torch.cuda.is_available():
        print("FAIL: no CUDA device")
        return 1
    print("device: %s capability %s" % (torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)))

    failures = []
    case_load(failures)
    case_store_and_roundtrip(failures)
    case_env(failures)

    print()
    for _, msg in failures:
        print("MISMATCH: %s" % msg)
    n = sum(c for c, _ in failures)
    if n:
        print("FAIL: %d mismatches" % n)
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
