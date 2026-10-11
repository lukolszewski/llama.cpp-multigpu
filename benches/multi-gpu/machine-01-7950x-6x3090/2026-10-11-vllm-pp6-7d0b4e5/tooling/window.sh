#!/usr/bin/env bash
# window.sh LABEL CMD...  — run CMD inside a GPU window: stop production llama.cpp, run CMD under a
# timeout, then ALWAYS stop vLLM and restore production (verified on /health), and append a row to
# RUN.md's "## Windows (production down)" table.
#
# Knobs: WINDOW_MAX=2h (timeout for CMD)  PROD_HEALTH_MAX=600 (s to wait for production /health)
#        VLLM_PORT=8010  WHO=sup  DRY=1 (echo every external call; health polls succeed immediately;
#        RUN.md not modified — the row is printed instead)
# Production is ONLY stopped/started through its launcher (stop = docker rm -f; PERSIST=1 = detached).
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PROD_SH=/home/luk/dev/airun/Qwen3.8-Flash-Next/run-qwen38-6x3090-260k.sh
RUNMD=$ROOT/RUN.md
DRY=${DRY:-0}
WHO=${WHO:-sup}
VLLM_PORT=${VLLM_PORT:-8010}
WINDOW_MAX=${WINDOW_MAX:-2h}
PROD_HEALTH_MAX=${PROD_HEALTH_MAX:-600}

LABEL=${1:-}; shift || true
[[ -n "$LABEL" && $# -gt 0 ]] || { echo "usage: window.sh LABEL CMD..." >&2; exit 64; }

mkdir -p "$ROOT/logs"
TS=$(date +%Y%m%d_%H%M%S)
WLOG=$ROOT/logs/window-$LABEL-$TS.log
TMP=$(mktemp -d)
say() { local m="$(date -Is) $*"; echo "$m"; echo "$m" >> "$WLOG"; }
run() { if [[ "$DRY" == 1 ]]; then say "[dry] $*"; else "$@"; fi; }
prod_health() { [[ "$DRY" == 1 ]] && return 0; curl -sf -m 5 http://127.0.0.1:8009/health >/dev/null 2>&1; }
vllm_health() { [[ "$DRY" == 1 ]] && return 0; curl -sf -m 5 "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; }
gpu_mem_max() {  # highest memory.used (MiB) over GPUs 0-5
  if [[ "$DRY" == 1 ]]; then echo 0; return; fi
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1,2,3,4,5 2>/dev/null | sort -n | tail -1
}

STOP_TS=-; UP_TS=-; DOWN_TS=-; HEALTHY_TS=-; CODE=-; DONE=0
cleanup() {
  local rc=$?
  (( DONE )) && exit "$rc"
  DONE=1
  [[ -n "${POLL_PID:-}" ]] && kill "$POLL_PID" 2>/dev/null
  [[ -f "$TMP/up" ]] && UP_TS=$(cat "$TMP/up")
  say "VLLM STOP"
  DRY=$DRY run "$HERE/run-vllm-pp6.sh" stop >> "$WLOG" 2>&1 || true
  DOWN_TS=$(date -Is)
  say "PROD START"
  if [[ "$DRY" == 1 ]]; then say "[dry] PERSIST=1 $PROD_SH"; else PERSIST=1 "$PROD_SH" >> "$WLOG" 2>&1 || say "WARNING: production launcher exited non-zero"; fi
  local waited=0
  while ! prod_health; do
    sleep 10; waited=$((waited + 10))
    if (( waited >= PROD_HEALTH_MAX )); then break; fi
  done
  if prod_health; then
    HEALTHY_TS=$(date -Is); say "PRODUCTION RESTORED $HEALTHY_TS"
  else
    HEALTHY_TS="NOT RESTORED"
    for _ in 1 2 3; do say "!!!!!! PRODUCTION NOT RESTORED $(date -Is) — run: PERSIST=1 $PROD_SH !!!!!!"; done
    CODE=99
  fi
  local row="| $LABEL | $(date +%F) | $STOP_TS | $UP_TS | $DOWN_TS | $HEALTHY_TS | $WHO | exit=$CODE | |"
  if [[ "$DRY" == 1 ]]; then
    say "[dry] RUN.md row: $row"
  else
    python3 - "$RUNMD" "$row" <<'EOF'
import sys
p, row = sys.argv[1], sys.argv[2]
lines = open(p).read().split("\n")
try:
    i = next(k for k, l in enumerate(lines) if l.startswith("## Windows"))
except StopIteration:
    lines += ["", "## Windows (production down)", "| # | date | stop prod | vllm up | vllm down | prod healthy | who | outcome | cost |"]
    i = len(lines) - 2
j = i + 1
while j < len(lines) and lines[j].startswith("|"):
    j += 1
lines.insert(j, row)
open(p, "w").write("\n".join(lines))
EOF
    say "RUN.md row appended: $row"
  fi
  say "WINDOW $LABEL END exit=$CODE"
  rm -rf "$TMP"
  [[ "$CODE" == "-" ]] && CODE=1
  exit "$CODE"
}
trap cleanup EXIT INT TERM

say "WINDOW $LABEL START (log $WLOG, cmd: $*)"
say "PROD STOP"
if [[ "$DRY" == 1 ]]; then say "[dry] $PROD_SH stop"; else "$PROD_SH" stop >> "$WLOG" 2>&1 || true; fi
waited=0
while :; do
  m=$(gpu_mem_max); m=${m:-99999}
  (( m < 500 )) && break
  sleep 5; waited=$((waited + 5))
  if (( waited >= 180 )); then say "WARNING: GPUs not free after 180 s (max used ${m} MiB), continuing"; break; fi
done
STOP_TS=$(date -Is); say "PROD STOPPED $STOP_TS"

# background poll for first vLLM /health success
( while ! vllm_health; do sleep 15; done; date -Is > "$TMP/up" ) &
POLL_PID=$!

say "RUN (timeout $WINDOW_MAX): $*"
if [[ "$DRY" == 1 ]]; then "$@"; CODE=$?; else timeout "$WINDOW_MAX" "$@"; CODE=$?; fi
say "CMD EXIT $CODE"
exit "$CODE"
