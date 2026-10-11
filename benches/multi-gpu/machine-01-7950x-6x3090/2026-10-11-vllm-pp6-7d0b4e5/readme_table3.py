#!/usr/bin/env python3
# readme_table3.py upstream.json multigpu.json vllm.json [--prompt-n] -> markdown table, 3 data columns + 2 ratio columns
# Derived from the fork's scripts/multigpu/bench/readme_table.py (28 lines). find/cell/imp are kept EXACTLY as there
# (same "not run" / "failed" / "n/a" rules, same number formats), so the first two data columns and the
# "multigpu vs upstream" ratio come out byte-identical to the two-column tool; this file only adds a third data
# column and a second ratio column.
# Any of the three file args may be "-" (= that side was not measured): its cells read "not run", ratios "n/a".
# vLLM grid JSONs are produced by a different client (readme_grid_openai.py, key "client" at top level) but use the
# same row keys (slots,size_tokens,pp_agg,pp_slot_mean,tg_agg,tg_slot_mean,pp_error,tg_error), so no special-casing.
# Importable: section_vllm.py imports render()/prompts_line()/load().
import json
import sys

LABELS = ("upstream", "multigpu", "vLLM")
HEADER = ("| Workload | Context | Upstream llama.cpp | llama.cpp-multigpu | vLLM-PP (patched) "
          "| multigpu vs upstream | multigpu vs vLLM |")
SEP = "| --- | --- | --- | --- | --- | --- | --- |"
DEFAULT_SLOTS = [1, 5]
DEFAULT_SIZES = [5000, 50000, 150000, 200000, 250000]


def find(d, n, s):
    return next((r for r in d["rows"] if r["slots"] == n and r["size_tokens"] == s), None)


def cell(r, key, n):
    if r is None: return "not run"
    if key == "pp":
        if "pp_error" in r: return "failed"
        return f"{r['pp_slot_mean']:.0f}" if n == 1 else f"{r['pp_agg']:.0f} ({r['pp_slot_mean']:.0f}/slot)"
    if "pp_error" in r or "tg_error" in r: return "failed"
    return f"{r['tg_slot_mean']:.1f}" if n == 1 else f"{r['tg_agg']:.1f} ({r['tg_slot_mean']:.1f}/slot)"


def imp(ru, rm, key):
    if ru is None or rm is None or "pp_error" in ru or "pp_error" in rm or (key=="tg" and ("tg_error" in ru or "tg_error" in rm)): return "n/a"
    k = ("pp_slot_mean" if key=="pp" else "tg_slot_mean") if ru["slots"] == 1 else ("pp_agg" if key=="pp" else "tg_agg")
    a = ru[k]; b = rm[k]
    return f"{b/a:.2f}x ({b-a:+.0f} t/s)" if key=="pp" else f"{b/a:.2f}x ({b-a:+.1f} t/s)"


def load(path):
    """Read one grid JSON; "-" means not measured (empty grid)."""
    return {"rows": []} if path == "-" else json.load(open(path))


def slot_size_union(datas):
    """SLOTS/SIZES are the union over all provided files, as in the original."""
    slots = sorted({r["slots"] for d in datas for r in d.get("rows", [])}) or DEFAULT_SLOTS
    sizes = sorted({r["size_tokens"] for d in datas for r in d.get("rows", [])}) or DEFAULT_SIZES
    return slots, sizes


def render(datas):
    """datas = [upstream, multigpu, vLLM] loaded grids -> markdown table text (no trailing newline)."""
    slots, sizes = slot_size_union(datas)
    out = [HEADER, SEP]
    for n in slots:
        name = "1 slot" if n == 1 else f"{n} slots - concurrent"
        for s in sizes:
            ru, rm, rv = (find(d, n, s) for d in datas)
            k = f"{s // 1000}k"
            for key, wl in (("pp", "prefill"), ("tg", "generate")):
                c = [cell(r, key, n) for r in (ru, rm, rv)]
                out.append(f"| {name} - {wl} | {k} | {c[0]} | {c[1]} | {c[2]} "
                           f"| {imp(ru, rm, key)} | {imp(rv, rm, key)} |")
    return "\n".join(out)


def prompts_line(datas, labels=LABELS):
    """One line listing the actual prompt_n[0] per size for each file that has them (lowest slots row wins).
    Returns None when no file carries prompt_n at all; a size without a value reads "-" so the columns stay aligned."""
    _, sizes = slot_size_union(datas)
    parts = []
    for label, d in zip(labels, datas):
        rows = sorted(d.get("rows", []), key=lambda r: r.get("slots", 0))
        if not any(r.get("prompt_n") for r in rows):
            continue
        vals = []
        for s in sizes:
            r = next((r for r in rows if r.get("size_tokens") == s and r.get("prompt_n")), None)
            vals.append(str(r["prompt_n"][0]) if r else "-")
        parts.append(f"{label} {'/'.join(vals)}")
    return f"prompts: {'; '.join(parts)}" if parts else None


def main(argv):
    flags = [a for a in argv if a.startswith("--")]
    paths = [a for a in argv if not a.startswith("--")]
    unknown = [f for f in flags if f != "--prompt-n"]
    if len(paths) != 3 or unknown:
        print(f"usage: {sys.argv[0]} upstream.json multigpu.json vllm.json [--prompt-n]  ('-' = not measured)"
              + (f"  unknown: {' '.join(unknown)}" if unknown else ""), file=sys.stderr)
        return 2
    datas = [load(p) for p in paths]
    print(render(datas))
    if "--prompt-n" in flags:
        line = prompts_line(datas)
        if line:
            print()
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
