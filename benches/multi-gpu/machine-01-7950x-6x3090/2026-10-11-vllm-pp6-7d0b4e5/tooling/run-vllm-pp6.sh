#!/usr/bin/env bash
# run-vllm-pp6.sh [stop] — start (or stop) the vLLM OpenAI server container with pipeline
# parallelism over GPUs 0-5, with patched Python files bind-mounted over the image's dist-packages.
#
# Knobs (env, with defaults):
#   IMG=vllm-pp6:7d0b4e57a         image (local derivative of vllm/vllm-openai:cu129-nightly@7d0b4e57a, torchcodec removed)
#   CONTAINER=vllm-pp6  PORT=8010  DEVICES=0,1,2,3,4,5
#   OVERLAY=.../work/T1-patches/overlay   every *.py under it is mounted at $SITE/<relative path>:ro
#   SITE=/usr/local/lib/python3.12/dist-packages
#   MODEL_DIR=/home/luk/dev/ai/cache/qwen38-flash-next-gptq4   mounted at /model (ro)
#   PART=8,8,8,8,8,8   VLLM_PP_LAYER_PARTITION; must sum to 48, first entry >= 2 (PLE layer index 1 on stage 0)
#   PP=6 TP=1 MAXLEN=262144 NSEQ=5 GPU_UTIL=0.92
#   KV_DTYPE=            empty = flag omitted, else --kv-cache-dtype $KV_DTYPE
#   PREFIX_CACHE=0       0 → --no-enable-prefix-caching, 1 → nothing
#   ENGRAM='{"cpu_offload": true}'   --engram-config (PLE n-gram table host offload)
#   EXTRA_ENV="K=V K2=V2"  extra -e entries;  EXTRA="..."  extra server args (word-split)
#   LOG=.../logs/vllm-<ts>.log   docker logs -f target
#   DRY=1                print everything, run nothing
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)            # vllm-pp6/

IMG=${IMG:-vllm-pp6:7d0b4e57a}
CONTAINER=${CONTAINER:-vllm-pp6}
PORT=${PORT:-8010}
DEVICES=${DEVICES:-0,1,2,3,4,5}
OVERLAY=${OVERLAY:-$ROOT/work/T1-patches/overlay}
SITE=${SITE:-/usr/local/lib/python3.12/dist-packages}
MODEL_DIR=${MODEL_DIR:-/home/luk/dev/ai/cache/qwen38-flash-next-gptq4}
PART=${PART:-8,8,8,8,8,8}
PP=${PP:-6}
TP=${TP:-1}
MAXLEN=${MAXLEN:-262144}
NSEQ=${NSEQ:-5}
GPU_UTIL=${GPU_UTIL:-0.92}
KV_DTYPE=${KV_DTYPE:-}
PREFIX_CACHE=${PREFIX_CACHE:-0}
ENGRAM=${ENGRAM:-'{"cpu_offload": true}'}
EXTRA_ENV=${EXTRA_ENV:-}
EXTRA=${EXTRA:-}
DRY=${DRY:-0}
LOG=${LOG:-$ROOT/logs/vllm-$(date +%Y%m%d_%H%M%S).log}

run() { if [[ "$DRY" == 1 ]]; then echo "[dry] $*"; else "$@"; fi; }

if [[ "${1:-}" == "stop" || "${1:-}" == "down" ]]; then
  run docker rm -f "$CONTAINER" || true
  echo ">> stopped $CONTAINER"
  exit 0
fi

# --- PART validation ------------------------------------------------------------------------
IFS=, read -r -a parts <<< "$PART"
sum=0
for p in "${parts[@]}"; do
  [[ "$p" =~ ^[0-9]+$ ]] || { echo "PART: non-numeric entry '$p'" >&2; exit 2; }
  sum=$((sum + p))
done
if (( sum != 48 )); then echo "PART=$PART sums to $sum, expected 48" >&2; exit 2; fi
if (( ${#parts[@]} != PP )); then echo "PART has ${#parts[@]} entries, PP=$PP" >&2; exit 2; fi
if (( parts[0] < 2 )); then echo "PART first entry ${parts[0]} < 2: PLE layer (index 1) must be on stage 0" >&2; exit 2; fi

# --- overlay mounts -------------------------------------------------------------------------
mounts=()
if [[ -d "$OVERLAY" ]]; then
  while IFS= read -r f; do
    rel=${f#"$OVERLAY"/}
    mounts+=(-v "$f:$SITE/$rel:ro")
  done < <(find "$OVERLAY" -type f -name '*.py' | sort)
  (( ${#mounts[@]} )) || echo "WARNING: overlay dir $OVERLAY contains no *.py files" >&2
else
  echo "WARNING: overlay dir $OVERLAY does not exist; running unpatched image" >&2
fi

# --- env ------------------------------------------------------------------------------------
envs=(-e CUDA_DEVICE_ORDER=PCI_BUS_ID -e HF_HUB_OFFLINE=1 -e "VLLM_PP_LAYER_PARTITION=$PART")
for kv in $EXTRA_ENV; do envs+=(-e "$kv"); done

# --- server args (image ENTRYPOINT is `vllm serve`, so the model path is positional) ---------
args=(/model --served-model-name qwen3.8-flash-next --dtype bfloat16
      --pipeline-parallel-size "$PP" --tensor-parallel-size "$TP"
      --distributed-executor-backend mp
      --max-model-len "$MAXLEN" --max-num-seqs "$NSEQ" --gpu-memory-utilization "$GPU_UTIL"
      --engram-config "$ENGRAM"
      --port "$PORT" --host 0.0.0.0 --trust-remote-code)
[[ "$PREFIX_CACHE" == 1 ]] || args+=(--no-enable-prefix-caching)
[[ -n "$KV_DTYPE" ]] && args+=(--kv-cache-dtype "$KV_DTYPE")
if [[ -n "$EXTRA" ]]; then read -r -a extra_arr <<< "$EXTRA"; args+=("${extra_arr[@]}"); fi

cmd=(docker run -d --name "$CONTAINER" --gpus "\"device=$DEVICES\"" --ipc=host --shm-size 16g
     -p "$PORT:$PORT" "${envs[@]}" -v "$MODEL_DIR:/model:ro" "${mounts[@]}" "$IMG" "${args[@]}")

echo ">> log: $LOG"
printf '%q ' "${cmd[@]}"; echo

if [[ "$DRY" == 1 ]]; then echo "[dry] not running"; exit 0; fi

mkdir -p "$(dirname "$LOG")"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
"${cmd[@]}"
{ printf '# %s\n# ' "$(date -Is)"; printf '%q ' "${cmd[@]}"; echo; } >> "$LOG"
nohup docker logs -f "$CONTAINER" >> "$LOG" 2>&1 </dev/null &
echo ">> started $CONTAINER on port $PORT (logs → $LOG)"
