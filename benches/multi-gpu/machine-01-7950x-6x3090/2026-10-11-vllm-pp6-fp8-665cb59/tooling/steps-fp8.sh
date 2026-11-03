#!/usr/bin/env bash
# W-Q8KV steps.sh — fp8 KV smoke test, runs INSIDE the window (work/T3/window.sh W-Q8KV bash work/W-Q8KV/steps.sh).
# Goal: prove 5 users x 262k fit with --kv-cache-dtype fp8: KV tokens >= 1,310,720; 5 concurrent 250k requests all
# Running (0 Waiting); needle PASS 50k/200k; 1-session 5k/50k/250k numbers for a decode/prefill comparison with bf16.
# Also checks the chat parsers (reasoning + tool call) needed for the :8009 launcher.
set -uo pipefail
ROOT=/home/luk/dev/airun/vllm-pp6; cd "$ROOT"
W=work/W-Q8KV; LOGMD=$W/log.md; SRV=$W/vllm-server.log
ts() { date +%T; }; log() { echo "$(ts) $*" | tee -a "$LOGMD"; }
RUNSH="$ROOT/work/T3/run-vllm-pp6.sh"
U=http://localhost:8010

log "steps.sh START (W-Q8KV) KV_DTYPE=fp8 NSEQ=5 MAXLEN=262144 PREFIX_CACHE=${PREFIX_CACHE:-1}"; free -g | head -2 | tee -a "$LOGMD"
PART=8,8,8,8,8,8 MAXLEN=262144 NSEQ=5 KV_DTYPE=fp8 PREFIX_CACHE=${PREFIX_CACHE:-1} LOG="$ROOT/$SRV" \
  EXTRA="--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml" "$RUNSH" 2>&1 | tee -a "$LOGMD"
t0=$(date +%s); ok=0
for i in $(seq 1 80); do
  curl -sf -m 5 $U/health >/dev/null 2>&1 && { ok=1; break; }
  docker ps --format '{{.Names}}' | grep -q '^vllm-pp6$' || { log "container died"; break; }
  sleep 15
done
secs=$(( $(date +%s) - t0 ))
if [[ $ok != 1 ]]; then
  log "ABORT: not healthy after ${secs}s"; grep -iE "traceback|error|oom|kv.cache|fp8" "$SRV" | tail -200 > "$W/abort.txt" || true
  "$RUNSH" stop 2>&1 | tee -a "$LOGMD"; exit 1
fi
log "healthy after ${secs}s"; echo "$secs" > "$W/load-seconds.txt"
grep -iE "KV cache size|kv_cache_dtype|Maximum concurrency|fp8|e4m3" "$SRV" | head -20 | tee "$W/kv-lines.txt" | tee -a "$LOGMD"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee "$W/smi-idle.txt"
free -g | head -2 | tee "$W/free-idle.txt"; curl -sf $U/v1/models > "$W/models.json"

log "needle 50k/200k (+concurrency equality, expected FAIL on vLLM)"
python3 work/T2/openai_check.py $U 2>&1 | tee "$W/check.log" | grep -E "PASS|FAIL" | tee -a "$LOGMD"

log "5 concurrent 250k requests (readme_grid_openai 5 sessions)"
( for i in $(seq 1 60); do sleep 10; grep -E "Running: [0-9]+ reqs, Waiting: [0-9]+" "$SRV" | tail -1; done ) > "$W/running-samples.txt" &
SAMP=$!
python3 work/T2/readme_grid_openai.py $U vllm-fp8 "$W/grid-5x250k.json" 5 250000 2>&1 | tee "$W/grid-5-stdout.log" | tail -3 | tee -a "$LOGMD"
kill $SAMP 2>/dev/null
log "Running/Waiting samples (max Running, max Waiting):"
grep -oE "Running: [0-9]+ reqs, Waiting: [0-9]+" "$W/running-samples.txt" | sort | uniq -c | tee -a "$LOGMD"
grep -oE "Running: [0-9]+ reqs, Waiting: [0-9]+ reqs, GPU KV cache usage: [0-9.]+%" "$SRV" | sort -t: -k4 -n | tail -3 | tee -a "$LOGMD"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee "$W/smi-after-5x.txt"; free -g | head -2 | tee "$W/free-after-5x.txt"

log "1-session grid 5k,50k,250k"
python3 work/T2/readme_grid_openai.py $U vllm-fp8 "$W/grid-1.json" 1 5000,50000,250000 2>&1 | tee "$W/grid-1-stdout.log" | grep -E "^\[|GRID" | tee -a "$LOGMD"

log "parser checks"
curl -s -m 120 $U/v1/chat/completions -H 'content-type: application/json' -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"What is the weather in Paris right now? Use the tool."}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"max_tokens":400,"temperature":0}' > "$W/tool-call.json"
python3 -c "import json;d=json.load(open('$W/tool-call.json'));m=d['choices'][0]['message'];print('TOOL_CALLS:',json.dumps(m.get('tool_calls'))[:300]);print('CONTENT:',repr((m.get('content') or '')[:200]));print('REASONING:',repr((m.get('reasoning_content') or m.get('reasoning') or '')[:120]))" 2>&1 | tee -a "$LOGMD"
curl -s -m 120 $U/v1/chat/completions -H 'content-type: application/json' -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"How many r are in strawberry? Answer briefly."}],"max_tokens":400,"temperature":0,"chat_template_kwargs":{"enable_thinking":true}}' > "$W/think.json"
python3 -c "import json;d=json.load(open('$W/think.json'));m=d['choices'][0]['message'];print('THINK content:',repr((m.get('content') or '')[:200]));print('THINK reasoning:',repr((m.get('reasoning_content') or m.get('reasoning') or '')[:200]))" 2>&1 | tee -a "$LOGMD"
curl -s -m 120 $U/v1/chat/completions -H 'content-type: application/json' -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Say hello in one word."}],"max_tokens":50,"temperature":0}' > "$W/plain.json"
python3 -c "import json;d=json.load(open('$W/plain.json'));m=d['choices'][0]['message'];print('PLAIN content:',repr((m.get('content') or '')[:200]));print('PLAIN reasoning:',repr((m.get('reasoning_content') or m.get('reasoning') or '')[:120]))" 2>&1 | tee -a "$LOGMD"
python3 -c "import json;print('GEN CONFIG:',json.load(open('/home/luk/dev/ai/cache/qwen38-flash-next-gptq4/generation_config.json')))" | tee -a "$LOGMD"

log "error-line count: $(grep -ciE 'traceback|error' "$SRV")"
log "stopping vLLM"; "$RUNSH" stop 2>&1 | tee -a "$LOGMD"
log "steps.sh END"
