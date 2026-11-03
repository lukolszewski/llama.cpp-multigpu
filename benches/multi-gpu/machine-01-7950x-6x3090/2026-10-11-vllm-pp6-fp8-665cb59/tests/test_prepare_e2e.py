#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""P11e -- end-to-end test of the e4m3 STORE path in ``qsa_prepare``.

Run inside the vLLM image on a GPU that has *native* e4m3 (sm >= 89, GPU 6),
with the patched ``ops/qsa_prepare.py`` (task P11b) and ``ops/fp8_ampere.py``
mounted over the site-packages copies:

    python3 /t/test_prepare_e2e.py            # variant A + variant B
    python3 /t/test_prepare_e2e.py --only=a   # variant A only
    python3 /t/test_prepare_e2e.py --only=b   # variant B only

Variant A (delivered, preferred): call the real host function
``vllm.models.qwen4_exp.nvidia.ops.qsa_prepare.qsa_prepare`` with minimal but
valid indexer + main-attention inputs (NOTES.md derives every argument from the
kernel's loads/stores and the fused call site) and a main K/V cache allocated
twice with identical initial bytes -- once reinterpreted as
``torch.float8_e4m3fn`` (native path, as production does at
indexer_qsa.py:307-309) and once kept as ``uint8`` (software-emulated path,
task P11b, which routes through ``ops/fp8_ampere.f32_to_e4m3_bits``) -- and
assert both runs leave *byte-identical* caches plus byte-identical
``main_q_out`` / ``main_gate_out`` / indexer outputs.  ``main_k_scale`` and
``main_v_scale`` are exercised with 1.0 and 0.5.

Variant B (extra, same file): a tiny wrapper kernel of our own that calls the
module's ``_store_rotated`` / ``_to_dst_dtype`` on random bf16 ``y, o1, o2, v``
and stores into a ``float8_e4m3fn`` destination and a ``uint8`` destination;
asserts the two agree and that both equal PyTorch's own e4m3 conversion
(PLAN-P11 §1: torch CPU, torch CUDA and Triton's native cast agree -- RNE,
saturating to +-448, NaN -> 0x7F).

Prints ``ALL PASS`` / exit 0 or ``FAIL`` / exit 1 (full traceback for anything
that raises).  Cost: two Triton compiles of the fused prepare kernel (one per
destination dtype) plus two trivial ones -- all cases share one constexpr set on
purpose, so this stays well under the 2 minute budget.
"""

from __future__ import annotations

import argparse
import inspect
import time
import traceback

import torch

from vllm.triton_utils import tl, triton

import vllm.models.qwen4_exp.nvidia.ops.qsa_prepare as qp
from vllm.models.qwen4_exp.nvidia.ops.qsa_prepare import (
    _store_rotated,
    _to_dst_dtype,
    qsa_prepare,
)

E4M3 = torch.float8_e4m3fn
DEV = "cuda"
SEED = 0x5EED

# ---------------------------------------------------------------------------
# Shapes -- mirrors the fused call site (src/indexer_qsa.py:298-341) and the
# asserts inside qsa_prepare() (src/ops/qsa_prepare.py:539-574).  NOTES.md has
# the per-argument derivation.
# ---------------------------------------------------------------------------
# Indexer (QSA) side.  The fused prepare path requires index head_dim 128,
# index_kv_heads == 1 and a power-of-two compress_ratio > 1
# (src/indexer_qsa.py:66-89 _supports_fused_pre_indexer).
IDX_N_HEADS = 8  # config.indexer_n_heads
IDX_D = 128  # config.indexer_head_dim
COMPRESS_RATIO = 4  # config.indexer_compress_ratio
ROTARY_DIM = 64  # rotary_dim: cos_sin_cache rows are [cos(32) | sin(32)]
MAX_POS = 512  # cos_sin_cache rows
EPS = 1e-6  # GemmaRMSNorm variance epsilon
# Main attention.  main_qkv packs [q0 gate0 q1 gate1 ... K(HK) V(HK)];
# num_main_q_heads is derived from its width (qsa_prepare.py:542 + 571).
MAIN_HQ = 8
MAIN_HK = 2
MAIN_D = 128
MAIN_PAGE = 16
MAIN_BLOCKS = 8
QKV_W = 2 * (MAIN_HQ + MAIN_HK) * MAIN_D
# QSA side caches: [blocks, rows, 1, width] (H == 1; the kernel ignores the
# head axis and only uses stride(0)/stride(1), qsa_prepare.py:617-621).
STATE_BLOCKS = 4
STATE_SIZE = COMPRESS_RATIO  # raw-K ring rows per block -> STATE_SIZE constexpr
STATE_W = IDX_D  # rope_pos_offset=None -> no packed int64 positions
COMP_BLOCKS = 4
COMP_PAGE = 4  # -> COMP_PAGE_SIZE constexpr
MAX_TOKENS = 16
SCALES = (1.0, 0.5)

# Offsets inside one cache row / one main_qkv row (element units).
K_OFF = 0  # cache row = [K (MAIN_D) | V (MAIN_D)]
V_OFF = MAIN_D
QGATE_W = 2 * MAIN_HQ * MAIN_D  # main_qkv: interleaved Q and gate
KV_BLOCK_W = MAIN_HK * MAIN_D  # main_qkv: width of the K block (= V block)
CACHE_ROW = (MAIN_BLOCKS, MAIN_HK, MAIN_PAGE, 2 * MAIN_D)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def u8(t: torch.Tensor) -> torch.Tensor:
    """Flat uint8 byte view of a tensor (no float/NaN comparison semantics)."""
    return t.detach().contiguous().view(torch.uint8).reshape(-1).clone()


def check(label: str, ok: bool, detail: str, fails: list[str]) -> None:
    if ok:
        print(f"  PASS  {label}")
        return
    print(f"  FAIL  {label}")
    if detail:
        for line in detail.splitlines():
            print(f"        {line}")
    fails.append(label)


def byte_diff(a: torch.Tensor, b: torch.Tensor, cache_layout: bool = False) -> str:
    """Human-readable description of the differences between two byte views."""
    a, b = u8(a), u8(b)
    if a.shape != b.shape:
        return f"shape {tuple(a.shape)} != {tuple(b.shape)}"
    diff = a != b
    n = int(diff.sum())
    if n == 0:
        return ""
    idx = int(diff.nonzero()[0])
    where = f"flat byte {idx}"
    if cache_layout:
        per_blk = MAIN_PAGE * MAIN_HK * 2 * MAIN_D
        blk, rem = divmod(idx, per_blk)
        page, rem = divmod(rem, MAIN_HK * 2 * MAIN_D)
        head, col = divmod(rem, 2 * MAIN_D)
        where = (
            f"block={blk} row_in_page={page} kv_head={head} col={col} "
            f"({'K' if col < MAIN_D else 'V'})"
        )
    tail = f" (of {a.numel()} bytes)" if n > 1 else ""
    return (
        f"{n} byte(s) differ{tail}; first at {where}: "
        f"native=0x{int(a[idx]):02X} emulated=0x{int(b[idx]):02X}"
    )


def e4m3_oracle(x: torch.Tensor, scale: float, on_cpu: bool = False) -> torch.Tensor:
    """PyTorch's own quantization of an already-BF16-rounded value:
    ``bf16 -> float32 -> /scale -> float8_e4m3fn`` as raw bytes."""
    t = x.detach()
    if on_cpu:
        t = t.cpu()
    return (t.float() / scale).to(E4M3).view(torch.uint8)


# ---------------------------------------------------------------------------
# variant A: inputs for qsa_prepare()
# ---------------------------------------------------------------------------
def build_cos_sin() -> torch.Tensor:
    """[MAX_POS, ROTARY_DIM] bf16; row = [cos(32 angles) | sin(32 angles)].

    The kernel reads cos at ``pos * (D // 2) + rot`` and sin at ``+ D // 4``
    with D = indexer head_dim (qsa_prepare.py:211-212 for the main path,
    285-290 via _norm_rope), and the host asserts
    ``cos_sin_cache.shape[-1] * 2 == head_dim`` (qsa_prepare.py:548).

    bf16 because that is what production registers: ``get_rope(...)`` is called
    without ``dtype`` (qsa.py:426-430) so it defaults to the model dtype, and
    ``RotaryEmbedding.__init__`` does ``cache.to(dtype)``
    (rotary_embedding/base.py:59-63).  It then matches the indexer Q /
    compressed-cache dtypes, so no implicit narrowing store is involved.
    """
    half = ROTARY_DIM // 2
    inv = 1.0 / (
        10000.0 ** (torch.arange(0, half, dtype=torch.float32, device=DEV) / half)
    )
    ang = torch.arange(MAX_POS, dtype=torch.float32, device=DEV)[:, None] * inv[None, :]
    return torch.cat([ang.cos(), ang.sin()], dim=-1).to(torch.bfloat16).contiguous()


def build_inputs(seed: int) -> dict:
    """Allocate every tensor qsa_prepare() reads or writes.

    The buffers the kernel writes (q_out, state, comp, init_cache) keep their
    pristine content in this dict; each run receives a ``clone()`` so both runs
    provably start from identical bytes.
    """
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)

    def randn(*shape, scale: float = 1.0) -> torch.Tensor:
        x = torch.randn(*shape, generator=g, device=DEV, dtype=torch.float32)
        return (x * scale).to(torch.bfloat16)

    # --- indexer projections, bf16 = model dtype (indexer_qsa.py:258-263) ----
    q = randn(MAX_TOKENS, IDX_N_HEADS * IDX_D)  # projected_q -> arg "q"
    k = randn(MAX_TOKENS, IDX_D)  # raw_keys (index_kv_heads == 1) -> arg "k"
    q_norm_w = randn(IDX_D, scale=0.1) + 1.0  # q_layernorm.weight
    k_norm_w = randn(IDX_D, scale=0.1) + 1.0  # k_layernorm.weight
    # --- output buffer of the Q work item (indexer_dtype = bf16) ------------
    q_out = randn(MAX_TOKENS, IDX_N_HEADS, IDX_D)
    # --- QSA raw-K ring (QSAKeyStateCache.kv_cache, rope positions off) -----
    state = randn(STATE_BLOCKS, STATE_SIZE, 1, STATE_W)
    # --- QSA compressed-K cache (QSACompressedKeyCache.kv_cache) ------------
    comp = randn(COMP_BLOCKS, COMP_PAGE, 1, IDX_D)
    # --- main attention fused QKV (bf16) ------------------------------------
    main_qkv = randn(MAX_TOKENS, QKV_W)
    # V is copied straight into the cache (qsa_prepare.py:229-230), so give the
    # V columns a log-uniform magnitude spread across the whole e4m3 range
    # (subnormals .. saturation) plus exact specials on a few rows.
    v = main_qkv[:, QGATE_W + KV_BLOCK_W :].float()
    spread = torch.exp2(
        torch.linspace(-14.0, 8.0, MAIN_D, device=DEV).repeat(MAIN_HK)
    )
    v = v * spread[None, :]
    specials = torch.tensor(
        [
            0.0,
            -0.0,
            448.0,  # largest finite e4m3 -> 0x7E
            456.0,  # above 448 -> saturates to 0x7E
            2.0**-9,  # largest e4m3 subnormal -> 0x01
            2.0**-10,  # tie -> round-even -> +-0
            1.5 * 2.0**-9,  # -> 0x02
            2.5 * 2.0**-9,  # tie -> round-even -> 0x02
            2.0**-6,  # smallest e4m3 normal -> 0x08
            0.017578124,  # just below 1.5*2**-6
            1.0,
            -1.0,
            100.0,
            -100.0,
            float("inf"),  # satfinite -> 0x7E
            float("-inf"),  # -> 0xFE
        ],
        device=DEV,
    )
    n_sp = min(int(specials.numel()), MAX_TOKENS)
    v[:n_sp, 0] = specials[:n_sp]
    v[:n_sp, MAIN_D] = -specials[:n_sp]
    main_qkv[:, QGATE_W + KV_BLOCK_W :] = v.to(torch.bfloat16)
    main_q_norm_w = randn(MAIN_D, scale=0.1) + 1.0
    main_k_norm_w = randn(MAIN_D, scale=0.1) + 1.0
    # --- main KV cache storage: vLLM allocates it as uint8 (qsa.py:462-464) --
    init_cache = torch.randint(
        0,
        256,
        CACHE_ROW,
        dtype=torch.uint8,
        device=DEV,
        generator=g,
    )
    return dict(
        q=q,
        k=k,
        cos_sin=build_cos_sin(),
        q_norm_w=q_norm_w,
        k_norm_w=k_norm_w,
        q_out=q_out,
        state=state,
        state_table=torch.zeros(1, 2, dtype=torch.int32, device=DEV),
        comp=comp,
        main_qkv=main_qkv,
        main_q_norm_w=main_q_norm_w,
        main_k_norm_w=main_k_norm_w,
        init_cache=init_cache,
    )


class Case:
    """One scenario: num_tokens + logical-position offset + its metadata.

    Only runtime values (shapes of the *sliced* tensors, slot maps, scales)
    vary between scenarios; every Triton constexpr stays identical so the whole
    section needs exactly one compiled kernel per destination dtype.
    """

    def __init__(self, name: str, num_tokens: int, logical_start: int):
        self.name = name
        self.nt = num_tokens
        self.logical_start = logical_start
        pos = torch.arange(
            logical_start, logical_start + num_tokens, dtype=torch.int64, device=DEV
        )
        self.positions = pos  # 1D rope positions (non-mrope model)
        self.logical_positions = pos.clone()  # same values, QSA logical space
        self.query_start_loc = torch.tensor(
            [0, num_tokens], dtype=torch.int32, device=DEV
        )
        # k_work_metadata rows (request, work_in_request) exactly as the QSA
        # metadata builder emits them (common/qsa_cache.py:337-381):
        # max(num_groups, 1) items per nonempty request; item 0 also commits the
        # raw-K suffix (qsa_prepare.py:299-309, 448-480).
        chunk_end = logical_start + num_tokens - 1
        num_groups = (
            (chunk_end + 1) // COMPRESS_RATIO - logical_start // COMPRESS_RATIO
        )
        work = max(num_groups, 1)
        self.num_work = work
        self.k_work = torch.stack(
            [
                torch.zeros(work, dtype=torch.int32, device=DEV),
                torch.arange(work, dtype=torch.int32, device=DEV),
            ],
            dim=1,
        ).contiguous()
        # Main cache slots: distinct (no CTA-vs-CTA race), inside the cache, and
        # one padding slot (-1 -> the kernel skips the write, line 220-221).
        slots = (
            torch.arange(num_tokens, dtype=torch.int64, device=DEV) * 5 + 1
        ) % (MAIN_BLOCKS * MAIN_PAGE)
        if num_tokens >= 2:
            slots[0] = -1
        self.main_slots = slots
        # Raw-K ring slots for the suffix commit: distinct, inside the ring.
        self.state_slots = (
            torch.arange(num_tokens, dtype=torch.int64, device=DEV)
            % (STATE_BLOCKS * STATE_SIZE)
        )
        # Compressed-K slots: valid + distinct for every token, so whichever
        # token turns out to be a group boundary has an in-bounds destination.
        self.comp_slots = (
            torch.arange(num_tokens, dtype=torch.int64, device=DEV)
            % (COMP_BLOCKS * COMP_PAGE)
        )

    def __str__(self) -> str:
        end = self.logical_start + self.nt - 1
        return (
            f"{self.name}(nt={self.nt}, logical={self.logical_start}..{end}, "
            f"work={self.num_work})"
        )

    def write_mask(self) -> torch.Tensor:
        """Bool map of the cache bytes this case must overwrite."""
        mask = torch.zeros(CACHE_ROW, dtype=torch.bool, device=DEV)
        for slot in self.main_slots[self.main_slots >= 0].tolist():
            mask[slot // MAIN_PAGE, :, slot % MAIN_PAGE, :] = True
        return mask

    def scatter(self, base: torch.Tensor, per_token: torch.Tensor,
                offset: int) -> torch.Tensor:
        """Copy of ``base`` with per_token [nt, HK, X] written at ``offset`` of
        every mapped cache row (padding slots skipped, like the kernel does).
        """
        out = base.clone()
        for i, slot in enumerate(self.main_slots.tolist()):
            if slot < 0:
                continue
            out[slot // MAIN_PAGE, :, slot % MAIN_PAGE, offset : offset + per_token.shape[-1]] = (
                per_token[i]
            )
        return out

    def expected_v(self, base: torch.Tensor, main_qkv: torch.Tensor,
                   scale: float) -> torch.Tensor:
        """``base`` with the V halves replaced by PyTorch's own e4m3 of the V
        columns.  The V store is a pure copy + scale + quantize
        (qsa_prepare.py:229-230), so this is an exact absolute oracle.
        """
        src = (
            main_qkv[: self.nt, QGATE_W + KV_BLOCK_W :]
            .reshape(self.nt, MAIN_HK, MAIN_D)
        )
        return self.scatter(base, e4m3_oracle(src, scale), V_OFF)

    def reference_k(self, inp: dict, scale: float) -> torch.Tensor:
        """Recompute the K bytes in torch the way the kernel does
        (qsa_prepare.py:203-228) -- informational only.

        The kernel's fp32 reduction/rsqrt order and FMA contraction are not
        reproducible op-for-op in torch, so a byte or two may legitimately
        differ when a value sits exactly on an e4m3 rounding boundary; that is
        why this is printed as an agreement rate instead of asserted.
        """
        outs = []
        w = inp["main_k_norm_w"].float() + 1.0
        cos_sin = inp["cos_sin"]
        half = MAIN_D // 4
        for token in range(self.nt):
            # kernel line 191: src = row + (2*MAIN_HQ + kv_head) * MAIN_D
            base = 2 * MAIN_HQ * MAIN_D
            rows = []
            for kv_head in range(MAIN_HK):
                off = base + kv_head * MAIN_D
                x = inp["main_qkv"][token, off : off + MAIN_D].float()
                inv = torch.rsqrt(x.pow(2).sum() / MAIN_D + EPS)
                y = x * inv * w
                x1 = (x[:half] * inv * w[:half]).to(torch.bfloat16).float()
                x2 = (x[half : 2 * half] * inv * w[half : 2 * half]).to(torch.bfloat16).float()
                pos = int(self.positions[token])
                # the kernel loads cos/sin then casts them to fp32 (lines 211-212)
                cos = cos_sin[pos, :half].float()
                sin = cos_sin[pos, half : 2 * half].float()
                o1 = x1 * cos - x2 * sin
                o2 = x2 * cos + x1 * sin
                parts = []
                for t in (o1, o2, y[2 * half :]):
                    parts.append(e4m3_oracle(t.to(torch.bfloat16), scale))
                rows.append(torch.cat(parts))
            outs.append(torch.stack(rows))
        ref = torch.stack(outs)
        return self.scatter(inp["init_cache"], ref, K_OFF)


def run_case(inp: dict, case: Case, raw_cache: torch.Tensor, native: bool,
             scale: float, q_out: torch.Tensor, state: torch.Tensor,
             comp: torch.Tensor) -> dict:
    """One qsa_prepare() call.

    ``raw_cache`` is the uint8 [blocks, HK, page, 2*MAIN_D] storage; ``native``
    reinterprets it as float8_e4m3fn exactly like the production call site
    (indexer_qsa.py:307-309), while ``native=False`` hands the kernel the uint8
    view, which is what the P11b integration leaves in place on sm < 89.
    """
    kv = raw_cache.view(E4M3).transpose(1, 2) if native else raw_cache.transpose(1, 2)
    main_q_out, main_gate_out = qsa_prepare(
        inp["q"][: case.nt],  # q                projected_q        (in)
        inp["k"][: case.nt],  # k                raw_keys           (in)
        case.positions,  # positions
        inp["cos_sin"],  # cos_sin_cache
        inp["q_norm_w"],  # q_norm_weight
        inp["k_norm_w"],  # k_norm_weight
        EPS,  # eps
        q_out[: case.nt],  # q_out              indexer Q buffer   (out)
        state,  # state_cache      raw-K ring    (in+out)
        case.state_slots,  # state_slots
        inp["state_table"],  # state_block_table
        case.query_start_loc,  # query_start_loc
        case.logical_positions,  # logical_positions
        comp,  # compressed_cache                (out)
        case.comp_slots,  # compressed_slots
        case.k_work,  # k_work_metadata
        compress_ratio=COMPRESS_RATIO,
        mrope_section=None,  # 1D-rope model -> IS_2D_POSITIONS=False
        rope_pos_offset=None,  # ring stores keys only -> CACHE_HAS_ROPE_POS=False
        main_qkv=inp["main_qkv"][: case.nt],
        main_q_norm_weight=inp["main_q_norm_w"],
        main_k_norm_weight=inp["main_k_norm_w"],
        main_eps=EPS,
        main_kv_cache=kv,  # <-- the tensor under test (fp8 vs uint8 view)
        main_slot_mapping=case.main_slots,
        main_k_scale=scale,
        main_v_scale=scale,
    )
    return dict(
        main_q_out=main_q_out,
        main_gate_out=main_gate_out,
        q_out=q_out,
        state=state,
        comp=comp,
        cache=raw_cache,
    )


def report_codes(tag: str, raw: torch.Tensor, mask: torch.Tensor) -> None:
    """Prove the store produced a rich set of e4m3 codes (not a vacuous match)."""
    v_mask = mask.clone()
    v_mask[..., :V_OFF] = False
    v_bytes = raw[v_mask]
    codes = int(torch.unique(v_bytes).numel())
    sat = int(((v_bytes == 0x7E) | (v_bytes == 0xFE)).sum())
    zero = int(((v_bytes == 0x00) | (v_bytes == 0x80)).sum())
    nan = int(((v_bytes == 0x7F) | (v_bytes == 0xFF)).sum())
    print(
        f"        {tag}: {v_bytes.numel()} V bytes written, {codes} distinct "
        f"e4m3 codes, {sat} saturated(448), {zero} zero, {nan} NaN-code"
    )


def section_a() -> list[str]:
    fails: list[str] = []
    print("== variant A: qsa_prepare()  native float8_e4m3fn vs uint8 ==")
    inp = build_inputs(SEED)
    main_qkv = inp["main_qkv"]
    cases = [
        Case("A-decode1", 1, 0),  # no completed group: raw-K suffix commit only
        Case("A-cross3", 3, 1),  # first group reads the raw-K ring
        Case("A-full16", 16, 0),  # 4 completed groups, chunk aligned
        Case("A-cross16", 16, 2),  # 4 groups, the first one from the ring
    ]
    for case in cases:
        # (0) the native path must be reproducible, otherwise "identical" is
        # meaningless; also proves the two runs really collide with the cache.
        raw_1 = inp["init_cache"].clone()
        raw_2 = inp["init_cache"].clone()
        run_case(inp, case, raw_1, True, 1.0, inp["q_out"].clone(),
                 inp["state"].clone(), inp["comp"].clone())
        run_case(inp, case, raw_2, True, 1.0, inp["q_out"].clone(),
                 inp["state"].clone(), inp["comp"].clone())
        check(
            f"case={case}: native run is reproducible",
            bool(torch.equal(raw_1, raw_2)),
            byte_diff(raw_1, raw_2, cache_layout=True),
            fails,
        )
        # Oracle sanity: torch's own CUDA and CPU e4m3 casts must agree on our
        # data (PLAN-P11 §1 says they do); if they do not, the oracle checks
        # below are not trustworthy and the note says so.
        v_src = (
            main_qkv[: case.nt, QGATE_W + KV_BLOCK_W :]
            .reshape(case.nt, MAIN_HK, MAIN_D)
        )
        oracle_ok = bool(
            torch.equal(
                e4m3_oracle(v_src, SCALES[0]).cpu(),
                e4m3_oracle(v_src, SCALES[0], on_cpu=True).cpu(),
            )
        )
        if not oracle_ok:
            print(f"  NOTE  case={case}: torch CUDA and CPU e4m3 casts disagree; "
                  "oracle comparisons below may be invalid")

        for scale in SCALES:
            raw_n = inp["init_cache"].clone()
            raw_e = inp["init_cache"].clone()
            out_n = run_case(inp, case, raw_n, True, scale, inp["q_out"].clone(),
                             inp["state"].clone(), inp["comp"].clone())
            out_e = run_case(inp, case, raw_e, False, scale, inp["q_out"].clone(),
                             inp["state"].clone(), inp["comp"].clone())
            tag = f"case={case} scale={scale}"
            # (1) PRIMARY: the main K/V cache must be byte-identical.
            check(
                f"{tag}: main KV cache bytes identical (native vs emulated)",
                bool(torch.equal(raw_n, raw_e)),
                byte_diff(raw_n, raw_e, cache_layout=True),
                fails,
            )
            # (2) every other output too.
            for key in ("main_q_out", "main_gate_out", "q_out", "state", "comp"):
                a, b = out_n[key], out_e[key]
                check(
                    f"{tag}: {key} bytes identical",
                    bool(torch.equal(u8(a), u8(b))),
                    f"{tuple(a.shape)} -> {byte_diff(a, b)}",
                    fails,
                )
            # (3) the store wrote the mapped slots and nothing else (guards
            #     against a vacuous pass where both paths write nothing).
            mask = case.write_mask()
            init = inp["init_cache"]
            total = int(mask.sum())
            for name, raw in (("native", raw_n), ("emulated", raw_e)):
                changed = raw != init
                stray = int((changed & ~mask).sum())
                kept = int((~changed & mask).sum())
                check(
                    f"{tag}: {name} touched only mapped slots ({total} bytes due)",
                    stray == 0,
                    f"{stray} bytes changed outside every mapped slot",
                    fails,
                )
                check(
                    f"{tag}: {name} really rewrote the mapped slots",
                    kept * 4 < total,
                    f"only {total - kept}/{total} mapped bytes changed -- "
                    "vacuous test (slot map or store path broken?)",
                    fails,
                )
            # (4) absolute oracle for the V copy path.
            exp = case.expected_v(init, main_qkv, scale)
            for name, raw in (("native", raw_n), ("emulated", raw_e)):
                detail = byte_diff(raw[..., V_OFF:], exp[..., V_OFF:])
                if detail and not oracle_ok:
                    detail += "\nNOTE: torch's own CUDA/CPU e4m3 casts disagree "
                    detail += "on this data, so the oracle itself is suspect."
                check(
                    f"{tag}: {name} V bytes == torch's e4m3(bf16/scale)",
                    not detail,
                    detail,
                    fails,
                )
            # (5) informational: independent torch recomputation of the K bytes
            #     (RMSNorm + RoPE + quantize).  Not asserted: Triton's reduction
            #     order / FMA contraction cannot be reproduced op-for-op.
            ref_k = case.reference_k(inp, scale)
            for name, raw in (("native", raw_n), ("emulated", raw_e)):
                a, b = raw[..., K_OFF:V_OFF], ref_k[..., K_OFF:V_OFF]
                same = int((u8(a) == u8(b)).sum())
                print(
                    f"        {tag} {name} K bytes matching a torch "
                    f"reimplementation: {same}/{a.numel()}"
                )
            report_codes(f"{tag} native", raw_n, mask)
    return fails


# ---------------------------------------------------------------------------
# variant B: direct _store_rotated / _to_dst_dtype wrapper kernel
# ---------------------------------------------------------------------------
@triton.jit
def _store_path_kernel(y_ptr, o1_ptr, o2_ptr, v_ptr, dst_ptr, scale,
                       D: tl.constexpr, Q: tl.constexpr):
    """One row per program, exactly what the main-K branch of the fused kernel
    does (qsa_prepare.py:228-230): [o1 | o2 | y[2Q:] | v] -> dst."""
    row = tl.program_id(0).to(tl.int64)
    dims = tl.arange(0, D)
    rot = tl.arange(0, Q)
    y = tl.load(y_ptr + row * D + dims)
    o1 = tl.load(o1_ptr + row * Q + rot)
    o2 = tl.load(o2_ptr + row * Q + rot)
    v = tl.load(v_ptr + row * D + dims)
    dst = dst_ptr + row * 2 * D
    _store_rotated(dst, y, o1, o2, scale)
    tl.store(dst + D + dims, _to_dst_dtype(v, dst, scale))


def section_b() -> list[str]:
    fails: list[str] = []
    print("== variant B: _store_rotated/_to_dst_dtype wrapper, e4m3 dst vs uint8 dst ==")
    rows, D, Q = 64, MAIN_D, MAIN_D // 4
    g = torch.Generator(device=DEV)
    g.manual_seed(SEED + 1)

    def spread(*shape) -> torch.Tensor:
        mag = torch.exp2(
            torch.rand(*shape, generator=g, device=DEV, dtype=torch.float32) * 26.0 - 15.0
        )
        sgn = torch.where(
            torch.rand(*shape, generator=g, device=DEV) < 0.5, -1.0, 1.0
        )
        return (mag * sgn).to(torch.bfloat16)

    y, o1, o2, v = spread(rows, D), spread(rows, Q), spread(rows, Q), spread(rows, D)
    specials = [
        float("nan"),
        float("inf"),
        float("-inf"),
        448.0,
        456.0,
        2.0**-10,
        2.5 * 2.0**-9,
        0.0,
    ]
    for t in (y, o1, o2, v):  # exact specials in the first rows
        for r, val in enumerate(specials):
            t[r, 0] = val
    dst_n = torch.zeros(rows, 2 * D, device=DEV, dtype=E4M3)
    dst_e = torch.zeros(rows, 2 * D, device=DEV, dtype=torch.uint8)
    for scale in SCALES:
        dst_n.view(torch.uint8).zero_()  # zero_() on an fp8 tensor is not relied on
        dst_e.zero_()
        _store_path_kernel[(rows,)](y, o1, o2, v, dst_n, scale, D=D, Q=Q, num_warps=1)
        _store_path_kernel[(rows,)](y, o1, o2, v, dst_e, scale, D=D, Q=Q, num_warps=1)
        check(
            f"scale={scale}: e4m3 destination == uint8 destination (bytes)",
            bool(torch.equal(u8(dst_n), u8(dst_e))),
            byte_diff(dst_n, dst_e),
            fails,
        )
        want = torch.cat(
            [
                e4m3_oracle(o1, scale),
                e4m3_oracle(o2, scale),
                e4m3_oracle(y[:, 2 * Q :], scale),
                e4m3_oracle(v, scale),
            ],
            dim=1,
        )
        for name, dst in (("e4m3", dst_n), ("uint8", dst_e)):
            detail = byte_diff(dst, want)
            check(
                f"scale={scale}: {name} destination == torch's e4m3(bf16/scale)",
                not detail,
                detail,
                fails,
            )
        print(
            f"        scale={scale}: {dst_n.numel()} bytes, "
            f"{int(torch.unique(u8(dst_n)).numel())} distinct e4m3 codes"
        )
    return fails


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="P11e e4m3 store path test")
    ap.add_argument("--only", choices=("a", "b", "ab"), default="ab")
    args = ap.parse_args()

    have_gpu = torch.cuda.is_available()
    print(
        f"torch {torch.__version__}  cuda={have_gpu}  "
        f"device={torch.cuda.get_device_name(0) if have_gpu else '-'}  "
        f"capability={torch.cuda.get_device_capability(0) if have_gpu else '-'}"
    )
    print(f"triton {getattr(triton, '__version__', '?')}")
    print(f"qsa_prepare module file: {getattr(qp, '__file__', '?')}")
    try:
        mod_src = inspect.getsource(qp)
    except Exception:  # pragma: no cover - frozen installs
        mod_src = ""
    imports_helper = "fp8_ampere" in mod_src or "f32_to_e4m3_bits" in mod_src
    emu_branch = "tl.uint8" in mod_src
    print(
        f"qsa_prepare.py carries the P11b emulated-store branch: {emu_branch} "
        f"(imports the helper: {imports_helper})"
    )
    if not emu_branch:
        print(
            "  NOTE: without that branch a uint8 destination is written with a\n"
            "        plain bf16->uint8 conversion, so failure below is EXPECTED.\n"
            "        Mount the patched ops/qsa_prepare.py (task P11b) + ops/fp8_ampere.py."
        )
    if not have_gpu:
        print("FAIL: no CUDA device")
        return 1

    sections = []
    if args.only in ("a", "ab"):
        sections.append(("A", section_a))
    if args.only in ("b", "ab"):
        sections.append(("B", section_b))

    results: dict[str, list[str]] = {}
    for name, fn in sections:
        t0 = time.time()
        try:
            results[name] = fn()
        except Exception:
            traceback.print_exc()
            results[name] = [f"variant {name} raised an exception"]
        print(
            f"-- variant {name}: {len(results[name])} failure(s), "
            f"{time.time() - t0:.1f}s"
        )

    print()
    if any(results.values()):
        print("failed checks:")
        for name, items in results.items():
            for item in items:
                print(f"  [{name}] {item}")
        print("FAIL")
        return 1
    print(f"ALL PASS  (variants {'+'.join(n for n, _ in sections)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
