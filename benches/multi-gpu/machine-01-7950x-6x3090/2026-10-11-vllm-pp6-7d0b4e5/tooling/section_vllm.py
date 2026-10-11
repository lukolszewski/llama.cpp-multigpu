#!/usr/bin/env python3
# section_vllm.py RUNDIR [--fork PATH] [--upstream PATH] [--multigpu PATH] [--prompt-n] [--raw-data PATH]
# Prints the markdown text of a new "### machine-01 - vLLM pipeline-parallel column" section for
# docs/multigpu/benchmarks.md, modelled on the machine-02 section of that file: fenced record block (from
# RUNDIR/config.json + RUNDIR/COMMITS.txt verbatim), then the three-column grid table (readme_table3 logic),
# then a short Caveats bullet list built from config.notes plus the always-on caveats.
# Missing config keys print TBD; nothing here crashes on a partial config. Style follows the record blocks that
# scripts/multigpu/bench/vast/vast-bench.sh `land` writes. Stdlib only.
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import readme_table3

TBD = "TBD"
TITLE = "### machine-01 - vLLM pipeline-parallel column"
DEFAULT_FORK = "/home/luk/.local/share/pitmp/pi-20261004-184507.KdxQ5T/llama.cpp"
RUN_SUBPATH = "benches/multi-gpu/machine-01-7950x-6x3090/2026-10-05-grid-df03399b8-vs-134489582"
UPSTREAM_DEFAULT = f"{RUN_SUBPATH}/grid-upstream.json"
MULTIGPU_DEFAULT = f"{RUN_SUBPATH}/grid-multigpu.json"


def s(cfg, key):
    """Scalar field as text; missing / None / empty -> TBD."""
    v = cfg.get(key)
    if v is None or v == "" or v == [] or v == {}:
        return TBD
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def lst(cfg, key, sep=", "):
    v = cfg.get(key)
    if not isinstance(v, list) or not v:
        return TBD
    return sep.join(str(x) for x in v)


def env_str(cfg):
    v = cfg.get("env")
    if not isinstance(v, dict) or not v:
        return TBD
    return " ".join(f"{k}={v[k]}" for k in sorted(v))


def unq(cmd):
    """Undo printf %q escaping (\\" \\, \\{ \\} \\space) so the command reads as typed."""
    for a, b in (('\\"', '"'), ("\\,", ","), ("\\{", "{"), ("\\}", "}"), ("\\ ", " "), ("\\:", ":")):
        cmd = cmd.replace(a, b)
    return cmd


def checkpoint_str(cfg):
    cp = cfg.get("checkpoint")
    if not isinstance(cp, dict) or not cp:
        return TBD
    files = cp.get("files")
    if not isinstance(files, list) or not files:
        return f"{cp.get('repo', TBD)} (files: {TBD})"
    try:
        total = sum(int(f[1]) for f in files)
    except (TypeError, ValueError, IndexError):
        return f"{cp.get('repo', TBD)} ({len(files)} files, total size {TBD})"
    return f"{cp.get('repo', TBD)} ({len(files)} files, {total / 1e9:.0f} GB = {total / 2**30:.0f} GiB; per-file sizes in config.json)"

def field(label, value):
    return f"{(label + ':'):<18}{value}"


def fence(rundir, cfg, raw_data):
    n_patches = len(cfg["patches"]) if isinstance(cfg.get("patches"), list) and cfg["patches"] else TBD
    lines = [
        field("vllm", f"{s(cfg, 'vllm_version')} (commit {s(cfg, 'vllm_commit')})"),
        field("image", f"{s(cfg, 'image')}  digest {s(cfg, 'image_digest')}"),
        field("run", unq(s(cfg, 'docker_run'))),
        field("pipeline", f"part={s(cfg, 'part')}; max_model_len={s(cfg, 'max_model_len')}; "
                          f"max_num_seqs={s(cfg, 'max_num_seqs')}; kv_cache_dtype={s(cfg, 'kv_cache_dtype')}; "
                          f"prefix_caching={s(cfg, 'prefix_caching')}"),
        field("patches", lst(cfg, 'patches')),
        field("overlay files", lst(cfg, "overlay_files")),
        field("env", env_str(cfg)),
        field("checkpoint", checkpoint_str(cfg)),
        field("measured", s(cfg, "measured")),
        field("gpus", s(cfg, "gpus")),
        field("raw data", f"{raw_data.rstrip('/')}/"),
    ]
    commits_path = os.path.join(rundir, "COMMITS.txt")
    try:
        commits = open(commits_path).read().rstrip("\n")
    except OSError:
        commits = TBD
    lines.append(commits if commits.strip() else TBD)
    return "```\n" + "\n".join(lines) + "\n```"


def caveats(cfg):
    notes = cfg.get("notes")
    notes = [str(n) for n in notes] if isinstance(notes, list) and notes else (
        [] if notes is None or notes == "" else [str(notes)])
    always = [
        f"vLLM is upstream {cfg.get('vllm_commit', TBD)} + {len(cfg.get('overlay_files') or [])} overlaid Python files, see {', '.join(str(p).split(' (')[0] for p in cfg['patches'])} — no PP-capable release exists for this model"
        if isinstance(cfg.get("patches"), list) and cfg["patches"] else f"vLLM is upstream + {TBD} patch files ({TBD})",
        f"prefix caching {'off' if not cfg.get('prefix_caching') else 'on'} for the grid",
        f"KV cache dtype: {s(cfg, 'kv_cache_dtype')}",
        "one streaming request per cell; prefill t/s = prompt_tokens/TTFT, generation t/s from inter-token time "
        "(the llama.cpp columns use the server's own timings)",
    ]
    return "Caveats:\n\n" + "\n".join(f"- {c}" for c in notes + always)


def load_grid(path):
    """readme_table3.load(), but a missing/broken grid degrades to "not measured" (cells "not run") with a
    warning instead of a traceback - the generator must always produce a section."""
    try:
        return readme_table3.load(path)
    except (OSError, ValueError) as e:
        print(f"warning: {path} unreadable ({e}); that column reads not run", file=sys.stderr)
        return {"rows": []}


def build(rundir, cfg, upstream, multigpu, raw_data, prompt_n=False):
    """The section text for a parsed config dict (kept separate from section() so it can be exercised without
    a config.json on disk - writing sample/config.json is blocked by the permission gate, see RESULT.md)."""
    vllm_path = os.path.join(rundir, "grid-vllm.json")
    datas = [load_grid(upstream), load_grid(multigpu), load_grid(vllm_path)]
    out = [TITLE, "", fence(rundir, cfg, raw_data), "", readme_table3.render(datas)]
    if prompt_n:
        line = readme_table3.prompts_line(datas)
        if line:
            out += ["", line]
    out += ["", caveats(cfg), ""]
    return "\n".join(out)


def section(rundir, upstream, multigpu, raw_data, prompt_n=False):
    cfg_path = os.path.join(rundir, "config.json")
    try:
        cfg = json.load(open(cfg_path))
        if not isinstance(cfg, dict):
            cfg = {}
    except (OSError, ValueError) as e:
        print(f"warning: {cfg_path} unreadable ({e}); all fields TBD", file=sys.stderr)
        cfg = {}
    return build(rundir, cfg, upstream, multigpu, raw_data, prompt_n)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate the machine-01 vLLM column section for docs/multigpu/benchmarks.md")
    ap.add_argument("rundir", help="run directory holding grid-vllm.json, config.json, COMMITS.txt")
    ap.add_argument("--fork", default=DEFAULT_FORK, help="fork clone root the default upstream/multigpu paths sit in")
    ap.add_argument("--upstream", default=None, help=f"default <fork>/{UPSTREAM_DEFAULT}")
    ap.add_argument("--multigpu", default=None, help=f"default <fork>/{MULTIGPU_DEFAULT}")
    ap.add_argument("--raw-data", default=None, help="path to print in the 'raw data:' field (default: RUNDIR)")
    ap.add_argument("--prompt-n", action="store_true", help="also list actual prompt_n[0] per size below the table")
    args = ap.parse_args(argv)
    upstream = args.upstream or os.path.join(args.fork, UPSTREAM_DEFAULT)
    multigpu = args.multigpu or os.path.join(args.fork, MULTIGPU_DEFAULT)
    print(section(args.rundir, upstream, multigpu, args.raw_data or args.rundir, args.prompt_n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
