#!/usr/bin/env bash
# train — Unified training entry point.
#
# Usage:
#   bash scripts/train.sh 16gpu --config configs/run/b1_9b_16gpu.yaml   # single experiment
#   bash scripts/train.sh 64gpu --phase 1                                # all experiments in phase 1
#   bash scripts/train.sh 64gpu --phase 2 --only k2                     # single experiment in phase
#   bash scripts/train.sh 64gpu --all                                    # all phases
#
# Topology presets (4gpu/8gpu/16gpu/32gpu/64gpu) set sensible defaults for
# nnodes, gpus/node, train_batch, ppo_mini, TP, SP, and gpu_mem_util.
# All can be overridden: --nnodes 2 --gpus-per-node 8 --train-batch 64 ...
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"
IMPL="$DIR/_train_impl.sh"

# ── experiment registry ──────────────────────────────────────────────────
declare -A PHASE_EXPS=()
PHASE_EXPS[1]="b1"
PHASE_EXPS[2]="k1 k2 k3 k4 k5 k2-r"
PHASE_EXPS[3]="r0-10k r0-25k r3 r4 r4-w r5 r6 r4-k"
PHASE_EXPS[4]="c1 c2 c3 c4"
PHASE_EXPS[5]="s1 s2"

# ── topology presets ─────────────────────────────────────────────────────
# key -> nnodes gpus/node rollout-tp ulysses-sp train-batch ppo-mini gpu-mem-util default-config
declare -A TP=()
TP[4gpu]="   1     4           2          1           32         32         0.30          configs/run/b1_9b.yaml"
TP[8gpu]="   1     8           2          1          256         32         0.75          configs/run/b1_8b.yaml"
TP[16gpu]="  2     8           2          2           64         64         0.30          configs/run/b1_9b_16gpu.yaml"
TP[32gpu]="  4     8           4          4          512         64         0.75          configs/run/b1.yaml"
TP[64gpu]="  8     8           4          4         1024         64         0.75          configs/run/b1.yaml"

# ── auto-detect GPU count ────────────────────────────────────────────────
_autodetect() {
  local gpus
  gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l) || gpus=0
  if   [ "$gpus" -le 4 ]; then echo "4gpu"
  elif [ "$gpus" -le 8 ]; then echo "8gpu"
  else echo "16gpu"
  fi
}

# ── parse CLI ────────────────────────────────────────────────────────────
TOPOLOGY="${1:-}"
if [ -z "$TOPOLOGY" ] || [ -z "${TP[$TOPOLOGY]+x}" ]; then
  TOPOLOGY="$(_autodetect)"
else
  shift
fi

read -r NNODES GPUS_PER_NODE ROLLOUT_TP ULYSSES_SP TRAIN_BATCH PPO_MINI GPU_MEM_UTIL DEFAULT_CONFIG <<< "${TP[$TOPOLOGY]}"

PHASE=""
ONLY=""
ALL=0
CONFIG=""
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --config)        CONFIG="$2"; shift 2;;
    --phase)         PHASE="$2"; shift 2;;
    --only)          ONLY="$2"; shift 2;;
    --all)           ALL=1; shift;;
    --nnodes)        NNODES="$2"; shift 2;;
    --gpus-per-node) GPUS_PER_NODE="$2"; shift 2;;
    --rollout-tp)    ROLLOUT_TP="$2"; shift 2;;
    --ulysses-sp)    ULYSSES_SP="$2"; shift 2;;
    --train-batch)   TRAIN_BATCH="$2"; shift 2;;
    --ppo-mini)      PPO_MINI="$2"; shift 2;;
    --gpu-mem-util)  GPU_MEM_UTIL="$2"; shift 2;;
    *)               EXTRA+=("$1"); shift;;
  esac
done

# ── helper: run a single experiment ──────────────────────────────────────
_run_one() {
  local cfg="$1"; shift
  local cfg_abs="$cfg"
  [[ "$cfg" == /* ]] || cfg_abs="$ROOT/$cfg"

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "[train] topology=$TOPOLOGY  nnodes=$NNODES×$GPUS_PER_NODE  tp=$ROLLOUT_TP  sp=$ULYSSES_SP"
  echo "[train] batch: train=$TRAIN_BATCH  ppo_mini=$PPO_MINI  gpu_mem_util=$GPU_MEM_UTIL"
  echo "[train] config=$cfg_abs"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  exec bash "$IMPL" \
    --nnodes "$NNODES" --gpus-per-node "$GPUS_PER_NODE" \
    --rollout-tp "$ROLLOUT_TP" --ulysses-sp "$ULYSSES_SP" \
    --train-batch "$TRAIN_BATCH" --ppo-mini "$PPO_MINI" \
    --gpu-mem-util "$GPU_MEM_UTIL" \
    --config "$cfg_abs" \
    "${EXTRA[@]}" "$@"
}

_run_impl_bg() {
  bash "$IMPL" \
    --nnodes "$NNODES" --gpus-per-node "$GPUS_PER_NODE" \
    --rollout-tp "$ROLLOUT_TP" --ulysses-sp "$ULYSSES_SP" \
    --train-batch "$TRAIN_BATCH" --ppo-mini "$PPO_MINI" \
    --gpu-mem-util "$GPU_MEM_UTIL" \
    "$@"
}

# ── dispatch ─────────────────────────────────────────────────────────────

# Single experiment (explicit --config)
if [ -n "$CONFIG" ]; then
  _run_one "$CONFIG" "$@"
fi

# Batch: --phase N [--only exp] or --all
if [ -n "$PHASE" ] || [ "$ALL" -eq 1 ]; then
  for p in "${!PHASE_EXPS[@]}"; do
    if [ "$ALL" -ne 1 ] && [ "$p" != "$PHASE" ]; then continue; fi
    cfgdir="phase${p}"
    for exp in ${PHASE_EXPS[$p]}; do
      if [ -n "$ONLY" ] && [ "$exp" != "$ONLY" ]; then continue; fi
      cfg="configs/$cfgdir/${exp}.yaml"
      if [ ! -f "$ROOT/$cfg" ]; then
        echo "[train] WARN: 配置不存在: $cfg，跳过" >&2; continue
      fi
      echo "===[train] Phase $p: $exp  ($cfg)========================================="
      _run_impl_bg --config "$ROOT/$cfg" || echo "[train] ✗ Phase $p $exp 失败，继续" >&2
    done
  done
fi
