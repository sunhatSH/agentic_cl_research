#!/usr/bin/env bash
# train — Unified CL training entry point.
#
# Replaces the 7 scattered wrapper scripts (train_4gpu.sh / train_8gpu.sh /
# train_16gpu.sh / train_32gpu.sh / train_64gpu.sh / train.sh / start_train.sh).
# Choose a topology by name, or let it auto-detect from the GPUs on the machine.
#
# Usage:
#   bash scripts/train <TOPOLOGY> [--buckets] [--smoke] [--config CFG] [...]
#   bash scripts/train               # auto-detect GPUs, default config
#
# Topologies:
#   4gpu    1 node  × 4  GPU    dev box / smoke
#   8gpu    1 node  × 8  GPU    single-node 8 GPU
#   16gpu   2 nodes × 8  GPU    two nodes (H800)
#   32gpu   4 nodes × 8  GPU    half rack
#   64gpu   8 nodes × 8  GPU    full rack (production)
#
# Each topology sets sensible defaults for train_batch / ppo_mini / rollout_TP /
# ulysses_SP / GPU mem util. All defaults can be overridden via CLI.
#
# Flags are forwarded to scripts/_train_impl.sh.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"
IMPL="$DIR/_train_impl.sh"

# ── topology presets ─────────────────────────────────────────────────────────
# key ->  nnodes gpus/node rollout-tp ulysses-sp  train-batch ppo-mini  gpu-mem-util  default-config
declare -A TP=()
TP[4gpu]="   1     4           2          1           32         32         0.30          configs/run/b1_9b.yaml"
TP[8gpu]="   1     8           2          1          256         32         0.75          configs/run/b1_8b.yaml"
TP[16gpu]="  2     8           2          2           64         64         0.30          configs/run/b1_9b_16gpu.yaml"
TP[32gpu]="  4     8           4          4          512         64         0.75          configs/run/b1.yaml"
TP[64gpu]="  8     8           4          4         1024         64         0.75          configs/run/b1.yaml"

# ── auto-detect ──────────────────────────────────────────────────────────────
_autodetect() {
  local gpus
  gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l) || gpus=0
  if [ "$gpus" -le 4 ]; then echo "4gpu"
  elif [ "$gpus" -le 8 ]; then echo "8gpu"
  else echo "16gpu"  # multi-node: caller should specify explicitly
  fi
}

# ── parse ────────────────────────────────────────────────────────────────────
TOPOLOGY="${1:-}"  # may be empty → auto-detect
if [ -n "$TOPOLOGY" ] && [ "${TP[$TOPOLOGY]+x}" ]; then
  shift
else
  TOPOLOGY="$(_autodetect)"
fi

read -r NNODES GPUS_PER_NODE ROLLOUT_TP ULYSSES_SP TRAIN_BATCH PPO_MINI GPU_MEM_UTIL DEFAULT_CONFIG <<< "${TP[$TOPOLOGY]}"
_GOT_CONFIG=0
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --config) DEFAULT_CONFIG="$2"; _GOT_CONFIG=1; shift 2;;
    --nnodes) NNODES="$2"; shift 2;;
    --gpus-per-node) GPUS_PER_NODE="$2"; shift 2;;
    --rollout-tp) ROLLOUT_TP="$2"; shift 2;;
    --ulysses-sp) ULYSSES_SP="$2"; shift 2;;
    --train-batch) TRAIN_BATCH="$2"; shift 2;;
    --ppo-mini) PPO_MINI="$2"; shift 2;;
    --gpu-mem-util) GPU_MEM_UTIL="$2"; shift 2;;
    *) EXTRA+=("$1"); shift;;
  esac
done

# Resolve config path. If user passed --config, forward it to impl.
CONFIG_REL="${DEFAULT_CONFIG#configs/}"
if [ "$CONFIG_REL" != "$DEFAULT_CONFIG" ]; then
  CONFIG="$ROOT/$DEFAULT_CONFIG"        # relative → absolute
else
  CONFIG="$DEFAULT_CONFIG"              # already absolute or user-specified
fi
if [ "$_GOT_CONFIG" -eq 1 ]; then
  EXTRA=("--config" "$CONFIG" "${EXTRA[@]}")   # forward user's --config
elif [ -n "$DEFAULT_CONFIG" ]; then
  EXTRA=("--config" "$CONFIG" "${EXTRA[@]}")   # default config
fi

echo "[train] topology=$TOPOLOGY  nnodes=$NNODES  gpus/node=$GPUS_PER_NODE  tp=$ROLLOUT_TP  sp=$ULYSSES_SP"
echo "[train] batch: train=$TRAIN_BATCH  ppo_mini=$PPO_MINI  gpu_mem_util=$GPU_MEM_UTIL"
echo "[train] config=$CONFIG  extra=${EXTRA[*]:-none}"

exec bash "$IMPL" \
  --nnodes "$NNODES" --gpus-per-node "$GPUS_PER_NODE" \
  --rollout-tp "$ROLLOUT_TP" --ulysses-sp "$ULYSSES_SP" \
  --train-batch "$TRAIN_BATCH" --ppo-mini "$PPO_MINI" \
  --gpu-mem-util "$GPU_MEM_UTIL" \
  "${EXTRA[@]}"
