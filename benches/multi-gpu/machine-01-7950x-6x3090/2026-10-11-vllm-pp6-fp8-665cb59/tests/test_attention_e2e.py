#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P11d: native e4m3 vs emulated (uint8) QSA sparse paged attention, bitwise.

Run inside the vLLM image on GPU 6 (sm_120, native e4m3) with the P11c-patched
`vllm/models/qwen4_exp/nvidia/ops/qsa.py` + `ops/fp8_ampere.py` mounted:

    docker run --rm --gpus '"device=6"' <overlay mounts> \
        -v $PWD/work/P11d:/t:ro --entrypoint python3 vllm-pp6:7d0b4e57a \
        /t/test_attention_e2e.py

Target: `vllm.models.qwen4_exp.nvidia.ops.qsa.qsa_sparse_paged_attention`.
For every case the SAME e4m3 bytes are fed to the op twice:

  (A) caches viewed as torch.float8_e4m3fn -> native e4m3 kernel (IS_FP8=True)
  (B) caches passed as torch.uint8         -> software e4m3 kernel (P11c)
  (C) caches dequantised to bf16 on the host, no scales (informational only)

and the bf16 outputs of (A) and (B) must be BITWISE equal (`torch.equal`).
Any exception (e.g. the host dtype check rejecting uint8, a Triton compile
error from `other=0.0` on a uint8 pointer) prints the full traceback and exits
1 - that is the signal the supervisor needs.

`python3 test_attention_e2e.py --check-data` runs the input-construction logic
and the kernel's index-resolution arithmetic (ops/qsa.py 109-133) in pure
Python, without torch, on a CPU-only host.
"""

from __future__ import annotations

import os
import random
import sys
import time
import traceback

torch = None  # set by main(); keeps module import (and --check-data) torch-free

# --------------------------------------------------------------------------- #
# Shapes, taken from the model / the op's asserts (see NOTES.md for the lines)  #
# --------------------------------------------------------------------------- #
HEAD_DIM = 128          # qsa_sparse_paged_attention: q.shape[2] == k_cache.shape[3]
NUM_KV_HEADS = 2        # k_cache.shape[2]
NUM_Q_HEADS = 16        # q.shape[1]; a multiple of NUM_KV_HEADS -> GROUP_SIZE 8
PAGE_SIZE = 16          # k_cache.shape[1] -> kernel PAGE_SIZE
NUM_REQS = 3            # block_table.shape[0] -> kernel num_requests
BLOCKS_PER_REQ = 128    # block_table.shape[1] -> kernel PAGE_TABLE_WIDTH
NUM_BLOCKS = NUM_REQS * BLOCKS_PER_REQ  # 384 physical pages, all filled
Q_STD = 0.02            # keeps the softmax spread over many tokens
ROWS = (1, 5, 64)
TOPKS = (64, 2048)
SCALES = ((1.0, 1.0), (0.5, 0.5))
BIG_ROWS = 1100         # base_programs = 1100*2 > 2048 -> use_prefill_config matters
NAN_CODES = (0x7F, 0xFF)


def log(msg: str) -> None:
    print(msg, flush=True)


def fail(msg: str) -> None:
    log(f"FAIL {msg}")
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Pure-python case data (no torch): also exercised by --check-data              #
# --------------------------------------------------------------------------- #
def build_block_table() -> list[list[int]]:
    """[NUM_REQS, BLOCKS_PER_REQ] int32: logical page -> physical page.

    Each request owns a disjoint block of BLOCKS_PER_REQ pages, shuffled, so the
    indirection is actually exercised; every value is < NUM_BLOCKS (the kernel
    rejects physical_page >= k_cache.shape[0], ops/qsa.py:132). Rows are
    int32 and contiguous (stride(1) == 1), asserted in main().
    """
    rng = random.Random(1234)
    table = []
    for i in range(NUM_REQS):
        pages = list(range(i * BLOCKS_PER_REQ, (i + 1) * BLOCKS_PER_REQ))
        rng.shuffle(pages)
        table.append(pages)
    return table


def token_to_req_for(rows: int) -> list[int]:
    """One request id per query row.

    Row 3 is a deliberately invalid (-1) request that still carries a non-zero
    count: every tile it reads must be masked out by the kernel's `request >= 0`
    test (ops/qsa.py:118-123) and the row must come out as exact zeros.
    """
    if rows == 0:
        return []
    if rows <= 3:
        return [r % NUM_REQS for r in range(rows)]
    out = [r % NUM_REQS for r in range(rows)]
    out[3] = -1
    return out


def counts_for(rows: int, topk: int) -> list[int]:
    """Per-row valid-entry count (the trailing column of the packed buffer).

    Cycle: full, 1, 3, half, 0 -> a couple of tiny counts, empty rows and rows
    at the maximum, per the task.
    """
    pattern = [topk, 1, 3, max(1, topk // 2), 0]
    if rows == 1:
        return [topk]
    return [pattern[r % len(pattern)] for r in range(rows)]


def logical_indices_for(
    rows: int, topk: int, token_to_req: list[int], counts: list[int], rng: random.Random
) -> list[list[int]]:
    """Packed selection buffer [rows, TOPK + 1] (int32 semantics).

    Column TOPK is the row's valid count; columns [0, count) are logical token
    indices inside the request's own page range; the remaining selection
    columns are -1 so the kernel's `logical_token >= 0` test masks them
    (ops/qsa.py:111-123) - the tile loop can run past the count inside the
    last BLOCK_N tile.
    """
    space = BLOCKS_PER_REQ * PAGE_SIZE  # logical tokens addressable per request
    data = []
    for r in range(rows):
        count = counts[r]
        assert count <= topk, f"count {count} exceeds TOPK {topk}"
        assert count <= space, f"count {count} exceeds the request's {space} tokens"
        row = rng.sample(range(space), count)
        row = row + [-1] * (topk - count) + [count]
        assert len(row) == topk + 1
        data.append(row)
    return data


def case_list() -> list[tuple[int, int, bool, tuple[float, float]]]:
    cases: list[tuple[int, int, bool, tuple[float, float]]] = []
    for topk in TOPKS:
        for rows in ROWS:
            for prefill in (False, True):
                for scales in SCALES:
                    cases.append((rows, topk, prefill, scales))
    # base_programs > 2048 is the only region where use_prefill_config selects a
    # different config (ops/qsa.py:489/524/572), so cover it once per flag.
    for prefill in (False, True):
        cases.append((BIG_ROWS, 64, prefill, (1.0, 1.0)))
    cases.append((0, 64, False, (1.0, 1.0)))  # empty batch: op must early-return
    return cases


def check_data() -> int:
    """Torch-free simulation of the kernel's index arithmetic (ops/qsa.py 84-133)."""
    table = build_block_table()
    page_table_width = len(table[0])
    total = 0
    for rows, topk, _prefill, _scales in case_list():
        counts = counts_for(rows, topk)
        t2r = token_to_req_for(rows)
        idx = logical_indices_for(rows, topk, t2r, counts, random.Random(0xABCD + topk))
        assert len(idx) == rows and len(t2r) == rows
        for r in range(rows):
            request = t2r[r]
            safe_request = min(max(request, 0), len(table) - 1)
            count = idx[r][topk]
            if not 0 <= count <= topk:
                print(f"FAIL row {r}: count column {count} not in [0, TOPK={topk}]")
                return 1
            for c in range(topk):
                token = idx[r][c]
                if c >= count:
                    if request >= 0 and token >= 0:
                        print(f"FAIL row {r} col {c}: unmasked garbage past count")
                        return 1
                    continue
                if token < 0:
                    print(f"FAIL row {r} col {c}: valid slot holds {token}")
                    return 1
                logical_page = token // PAGE_SIZE
                if logical_page >= page_table_width:
                    print(f"FAIL row {r} col {c}: page {logical_page} >= width")
                    return 1
                physical = table[safe_request][logical_page]
                if not 0 <= physical < NUM_BLOCKS:
                    print(f"FAIL row {r} col {c}: physical page {physical} OOB")
                    return 1
            total += count
        # every request's rows must resolve through its own table row
        print(
            f"  data ok rows={rows:<5} topk={topk:<5} counts={counts[:5]} "
            f"token_to_req={t2r[:5]} selected={sum(counts)}"
        )
    print(f"DATA OK ({total} selected index slots resolve in-bounds)")
    return 0


# --------------------------------------------------------------------------- #
# GPU side                                                                      #
# --------------------------------------------------------------------------- #
def build_raw_cache(seed: int) -> "torch.Tensor":
    """One random uint8 KV cache in the production paged layout.

    The QSA module hands the op `kv_cache.transpose(1, 2).split(head_size, -1)`
    (src/qsa.py:296) where the layer cache is the 4-D [B, H, N, C] view of
    vllm/v1/kv_cache_interface.py:319-335, i.e. here [blocks, kv_heads, page,
    2*head_dim] with K in [..., :128] and V in [..., 128:]. Keeping that layout
    gives exactly the strides production uses (token stride 2*head_dim, head
    stride page*2*head_dim) instead of a contiguous toy cache.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    raw = torch.randint(
        0,
        256,
        (NUM_BLOCKS, NUM_KV_HEADS, PAGE_SIZE, 2 * HEAD_DIM),
        dtype=torch.uint8,
        generator=gen,
    )
    # e4m3fn NaN codes would make the reference and the bitwise compare useless.
    raw[raw == NAN_CODES[0]] = 0x7E  # -> +448
    raw[raw == NAN_CODES[1]] = 0xFE  # -> -448
    assert not bool(((raw == NAN_CODES[0]) | (raw == NAN_CODES[1])).any())
    return raw


def kv_views(raw: "torch.Tensor", dtype: torch.dtype):
    """(key, value) as [blocks, page, kv_heads, head_dim] views of one buffer."""
    buf = raw if dtype == torch.uint8 else raw.view(torch.float8_e4m3fn)
    key, value = buf.transpose(1, 2).split(HEAD_DIM, dim=-1)
    return key, value


def make_qa_inputs(rows: int, seed: int):
    gen = torch.Generator()
    gen.manual_seed(seed)
    q = (torch.randn(rows, NUM_Q_HEADS, HEAD_DIM, generator=gen) * Q_STD).to(
        torch.bfloat16
    )
    gate = torch.randn(rows, NUM_Q_HEADS, HEAD_DIM, generator=gen).to(torch.bfloat16)
    return q, gate


def stats(out: "torch.Tensor"):
    if out.numel() == 0:
        return 0.0, 0.0
    return (
        out.isnan().float().mean().item(),
        out.float().abs().max().item(),
    )


def call_op(fn, what: str):
    try:
        out = fn()
        torch.cuda.synchronize()
        return out
    except Exception:
        log(f"FAIL {what} raised:")
        traceback.print_exc()
        sys.exit(1)


def main() -> int:
    global torch
    t_start = time.time()
    import torch as _torch  # noqa: PLC0415

    torch = _torch
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        fail("no CUDA device (this test must run on GPU 6 inside the image)")
    device = torch.device("cuda")
    cap = torch.cuda.get_device_capability(0)
    log(
        f"torch {torch.__version__} device={torch.cuda.get_device_name(0)} "
        f"capability={cap[0]}.{cap[1]}"
    )
    log(f"env VLLM_QSA_EMULATE_E4M3={os.environ.get('VLLM_QSA_EMULATE_E4M3')!r}")
    try:
        from vllm.models.qwen4_exp.nvidia.ops.fp8_ampere import emulate_e4m3

        log(f"ops.fp8_ampere.emulate_e4m3() = {emulate_e4m3()} (dtype drives the op)")
    except Exception:
        log("WARNING: ops/fp8_ampere.py is not importable - overlay not mounted?")
        traceback.print_exc()
    try:
        from vllm.models.qwen4_exp.nvidia.ops.qsa import _select_config
    except Exception:
        _select_config = None

    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention

    raw = build_raw_cache(seed=0).to(device)
    assert raw.is_contiguous()
    log(
        f"cache bytes: raw{tuple(raw.shape)} uint8 sum={int(raw.to(torch.int64).sum())} "
        f"({NUM_BLOCKS} pages x {PAGE_SIZE} tokens x {NUM_KV_HEADS} heads x "
        f"{2 * HEAD_DIM} = K|V packed)"
    )
    table_cpu = build_block_table()
    table = torch.tensor(table_cpu, dtype=torch.int32).to(device)
    assert table.stride(1) == 1 and table.dtype == torch.int32

    # Reference (C) caches: the same bytes dequantised on the host.
    k_bf16_full = raw.view(torch.float8_e4m3fn).to(torch.bfloat16)
    kC, vC = k_bf16_full.transpose(1, 2).split(HEAD_DIM, dim=-1)
    assert kC.stride(3) == 1 and vC.stride(3) == 1

    checked = 0
    launches = 0
    for rows, topk, prefill, (k_scale, v_scale) in case_list():
        counts = counts_for(rows, topk)
        t2r_list = token_to_req_for(rows)
        idx_cpu = logical_indices_for(
            rows, topk, t2r_list, counts, random.Random(0xABCD + topk)
        )
        indices = torch.tensor(idx_cpu, dtype=torch.int32).view(rows, topk + 1)
        token_to_req = torch.tensor(t2r_list, dtype=torch.int32)
        assert indices.stride(1) == 1 and token_to_req.stride(0) == 1
        indices, token_to_req = indices.to(device), token_to_req.to(device)
        q, gate = make_qa_inputs(rows, seed=1000 + rows)
        q, gate = q.to(device), gate.to(device)
        cfg_fp8 = (
            _select_config(rows, NUM_KV_HEADS, prefill, topk, True)
            if _select_config is not None
            else None
        )
        cfg_bf16 = (
            _select_config(rows, NUM_KV_HEADS, prefill, topk, False)
            if _select_config is not None
            else None
        )
        tag = (
            f"rows={rows:<5} topk={topk:<5} prefill={str(prefill):<5} "
            f"scales={k_scale}/{v_scale}"
        )

        # (A) native e4m3 caches
        kA, vA = kv_views(raw, torch.float8_e4m3fn)
        outA = torch.full_like(q, float("nan"))
        outA = call_op(
            lambda: qsa_sparse_paged_attention(
                q,
                kA,
                vA,
                indices,
                table,
                token_to_req,
                prefill,
                outA,
                k_scale=k_scale,
                v_scale=v_scale,
                output_gate=gate,
            ),
            f"{tag}: native (float8_e4m3fn) call",
        )
        launches += 1

        # (B) emulated: the very same bytes as uint8 (the P11c path). A/B read
        # one and the same storage - only its element dtype differs.
        kB, vB = kv_views(raw, torch.uint8)
        assert kB.shape == kA.shape and kB.stride() == kA.stride()
        assert kB.dtype == torch.uint8 and kA.dtype == torch.float8_e4m3fn
        outB = torch.full_like(q, float("nan"))
        outB = call_op(
            lambda: qsa_sparse_paged_attention(
                q,
                kB,
                vB,
                indices,
                table,
                token_to_req,
                prefill,
                outB,
                k_scale=k_scale,
                v_scale=v_scale,
                output_gate=gate,
            ),
            f"{tag}: emulated (uint8) call",
        )
        launches += 1

        # (C) bf16 reference, host-dequantised, no scales (informational)
        outC = call_op(
            lambda: qsa_sparse_paged_attention(
                q,
                kC,
                vC,
                indices,
                table,
                token_to_req,
                prefill,
                None,
                k_scale=None,
                v_scale=None,
                output_gate=gate,
            ),
            f"{tag}: bf16 reference call",
        )
        launches += 1

        nanA, maxA = stats(outA)
        nanB, maxB = stats(outB)
        nanC, maxC = stats(outC)
        if rows == 0:
            log(
                f"{tag} cfg={cfg_fp8} empty-batch ok "
                f"(shape {tuple(outA.shape)}, {time.time() - t_start:.1f}s)"
            )
            continue
        if maxA == 0.0 or maxB == 0.0:
            fail(f"{tag}: output all zeros - the case tests nothing (maxA={maxA})")
        if nanA != 0.0 or nanB != 0.0:
            fail(f"{tag}: NaN outputs native={nanA:.4f} emulated={nanB:.4f}")
        if nanC != 0.0:
            fail(f"{tag}: NaN outputs in the bf16 reference ({nanC:.4f})")

        diff = (outA.float() - outB.float()).abs()
        same = torch.equal(outA, outB)
        if not same:
            bad = (diff > 0).nonzero()
            log(
                f"{tag} cfg_fp8={cfg_fp8} cfg_bf16={cfg_bf16} A==B False "
                f"maxdiff={diff.max().item():.6e} ndiff={int((diff > 0).sum().item())}"
                f"/{outA.numel()} nanA={nanA:.4f} nanB={nanB:.4f}"
            )
            for pos in bad[:8].tolist():
                r, h, d = pos
                log(
                    f"    row={r} head={h} dim={d} A={outA[r, h, d].item():.6e} "
                    f"B={outB[r, h, d].item():.6e} count={counts[r]} "
                    f"req={t2r_list[r]}"
                )
            fail(f"{tag}: native and emulated outputs differ")

        ref = (outA.float() - outC.float()).abs()
        denom = max(outA.float().abs().max().item(), 1e-30)
        log(
            f"{tag} cfg_fp8={cfg_fp8} cfg_bf16={cfg_bf16} A==B True "
            f"max|A-B|={diff.max().item():.1e} nanA={nanA:.3f} nanB={nanB:.3f} "
            f"max|A-C|={ref.max().item():.3e} ({ref.max().item() / denom:.2%} rel) "
            f"[C cfg={cfg_bf16}]"
        )
        checked += 1

    # One case with op-allocated output (out=None) to cover that branch too.
    rows, topk = 5, 64
    counts = counts_for(rows, topk)
    t2r_list = token_to_req_for(rows)
    indices = torch.tensor(
        logical_indices_for(rows, topk, t2r_list, counts, random.Random(7)),
        dtype=torch.int32,
    ).view(rows, topk + 1)
    q, gate = make_qa_inputs(rows, seed=1000 + rows)
    q, gate = q.to(device), gate.to(device)
    outA = call_op(
        lambda: qsa_sparse_paged_attention(
            q,
            *kv_views(raw, torch.float8_e4m3fn),
            indices.to(device),
            table,
            torch.tensor(t2r_list, dtype=torch.int32, device=device),
            False,
            None,
            k_scale=1.0,
            v_scale=1.0,
            output_gate=gate,
        ),
        "out=None native call",
    )
    outB = call_op(
        lambda: qsa_sparse_paged_attention(
            q,
            *kv_views(raw, torch.uint8),
            indices.to(device),
            table,
            torch.tensor(t2r_list, dtype=torch.int32, device=device),
            False,
            None,
            k_scale=1.0,
            v_scale=1.0,
            output_gate=gate,
        ),
        "out=None emulated call",
    )
    if not torch.equal(outA, outB):
        fail(
            f"out=None case differs maxdiff="
            f"{(outA.float() - outB.float()).abs().max().item():.6e}"
        )
    launches += 2
    log("out=None (op-allocated output) case: A==B True")
    log(
        f"ALL PASS  ({checked} compared cases + 1 empty batch + 1 out=None case, "
        f"{launches} kernel calls, {time.time() - t_start:.1f}s)"
    )
    return 0


if __name__ == "__main__":
    if "--check-data" in sys.argv:
        sys.exit(check_data())
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        fail("unexpected error (see traceback above)")
