#!/usr/bin/env bash
# T7 steps.sh — the grid, runs INSIDE the T7 window. Config from T6 via env: PREFIX_CACHE (default 0), NSEQ (5).
# 1) start vLLM, /health ≤20 min (abort → exit 1); 2) nvidia-smi dmon sampler; 3) 1-session grid all sizes;
# 4) abort rule: if 1-session 250k pp_agg < upstream/3 (262.7/3 = 87.6 t/s) run 5-session 5k,50k only;
# 5) 5-session grid; 6) merge → grid-vllm.json; 7) stop.
set -uo pipefail
ROOT=/home/luk/dev/airun/vllm-pp6; cd "$ROOT"
T7=work/T7; LOGMD=$T7/log.md
PREFIX_CACHE=${PREFIX_CACHE:-0}; NSEQ=${NSEQ:-5}
SIZES=${SIZES:-5000,50000,150000,200000,250000}
UPSTREAM_250K_PP=${UPSTREAM_250K_PP:-262.7}
ts() { date +%T; }; log() { echo "$(ts) $*" | tee -a "$LOGMD"; }
RUNSH="$ROOT/work/T3/run-vllm-pp6.sh"

log "steps.sh START (T7 grid) PREFIX_CACHE=$PREFIX_CACHE NSEQ=$NSEQ SIZES=$SIZES"; free -g | head -2 | tee -a "$LOGMD"
PART=8,8,8,8,8,8 MAXLEN=262144 NSEQ=$NSEQ PREFIX_CACHE=$PREFIX_CACHE LOG="$ROOT/$T7/vllm-server.log" "$RUNSH" 2>&1 | tee -a "$LOGMD"
t0=$(date +%s); ok=0
for i in $(seq 1 80); do curl -sf -m 5 localhost:8010/health >/dev/null 2>&1 && { ok=1; break; }; sleep 15; done
secs=$(( $(date +%s) - t0 ))
if [[ $ok != 1 ]]; then
  log "ABORT: not healthy after ${secs}s"; grep -iE "traceback|error|oom" "$T7/vllm-server.log" | tail -200 > "$T7/abort.txt" || true
  "$RUNSH" stop 2>&1 | tee -a "$LOGMD"; exit 1
fi
log "healthy after ${secs}s"; echo "$secs" > "$T7/load-seconds.txt"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee "$T7/smi-idle.txt"
docker stats --no-stream | tee "$T7/docker-stats-idle.txt"; free -g | head -2 | tee "$T7/free-idle.txt"
curl -sf localhost:8010/v1/models > "$T7/models.json"

nohup nvidia-smi dmon -s um -d 30 > "$T7/gpu-samples.log" 2>&1 & DMON=$!; echo $DMON > "$T7/dmon.pid"

log "grid phase 1: 1 session, sizes $SIZES"
python3 work/T2/readme_grid_openai.py http://localhost:8010 vllm-pp6 "$T7/grid-vllm-1.json" 1 "$SIZES" 2>&1 | tee "$T7/grid-1-stdout.log"
log "phase 1 exit=${PIPESTATUS[0]}"
PP250=$(python3 -c "import json,sys; d=json.load(open('$T7/grid-vllm-1.json')); r=[x for x in d['rows'] if x.get('slots')==1 and x.get('size_tokens')==250000]; print(r[0].get('pp_agg',0) if r else 0)" 2>/dev/null || echo 0)
log "1-session 250k pp_agg=$PP250 (abort threshold ${UPSTREAM_250K_PP}/3)"
SIZES5=$SIZES
if python3 -c "import sys; sys.exit(0 if float('$PP250') < $UPSTREAM_250K_PP/3 else 1)"; then
  log "ABORT RULE: 1-session 250k is >3x slower than upstream llama.cpp → 5-session rows limited to 5k,50k"; SIZES5=5000,50000
  echo "abort-rule: 5-session deep rows skipped (pp_agg 250k = $PP250)" > "$T7/abort-rule.txt"
fi
log "grid phase 2: 5 sessions, sizes $SIZES5"
python3 work/T2/readme_grid_openai.py http://localhost:8010 vllm-pp6 "$T7/grid-vllm-5.json" 5 "$SIZES5" 2>&1 | tee "$T7/grid-5-stdout.log"
log "phase 2 exit=${PIPESTATUS[0]}"

python3 - "$T7" <<'PY'
import json,sys,os
t=sys.argv[1]; a=json.load(open(f"{t}/grid-vllm-1.json")); p5=f"{t}/grid-vllm-5.json"
if os.path.exists(p5): a["rows"]+=json.load(open(p5))["rows"]
json.dump(a,open(f"{t}/grid-vllm.json","w"),indent=1); print("merged rows:",len(a["rows"]))
PY
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee "$T7/smi-after-grid.txt"
docker stats --no-stream | tee "$T7/docker-stats-after-grid.txt"; free -g | head -2 | tee "$T7/free-after-grid.txt"
kill "$DMON" 2>/dev/null || true
log "error-line count vllm-server.log: $(grep -ciE 'error|traceback' "$T7/vllm-server.log")"
log "stopping vLLM"; "$RUNSH" stop 2>&1 | tee -a "$LOGMD"
log "steps.sh END"
