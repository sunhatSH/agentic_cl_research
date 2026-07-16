#!/usr/bin/env bash
# B1 bucket-sequential training (8B): catastrophic forgetting baseline.
# Trains 9 buckets in order. Each bucket starts from the previous bucket's ckpt.
# No KL, no replay — measures forgetting caused by sequential learning.
set -euo pipefail

cleanup() {
    local exit_code=$?
    echo "[cleanup] training exited with code=$exit_code, killing GPU processes..."
    # Kill the python training process and all children (Ray workers, LightLLM)
    if [ -n "${_TRAIN_PID:-}" ] && kill -0 "$_TRAIN_PID" 2>/dev/null; then
        kill -- -$(ps -o pgid= -p "$_TRAIN_PID" 2>/dev/null | tr -d ' ') 2>/dev/null || true
    fi
    # Fallback: kill any leftover python processes using GPU
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do
        kill -9 "$pid" 2>/dev/null || true
    done
    echo "[cleanup] done"
}
trap cleanup EXIT

PROJECT_DIR=/mnt/afs_toolcall/sunhao4/agentic_cl_research
LIGHTLLM_DIR=/mnt/afs_toolcall/sunhao4/Documents/LightLLM
VERL_DIR=/mnt/afs_toolcall/sunhao4/Documents/verl
CONDA_ENV=/mnt/afs_toolcall/sunhao4/miniconda3/envs/py310_base

export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH="$LIGHTLLM_DIR:$VERL_DIR:$PROJECT_DIR:$PROJECT_DIR/docker/qwen36-lightllm/flash_attn_shim"
export PYTHON=$CONDA_ENV/bin/python

_NV_LIB_ROOT="$CONDA_ENV/lib/python3.10/site-packages/nvidia"
if [ -d "$_NV_LIB_ROOT" ]; then
  for _d in "$_NV_LIB_ROOT"/*/lib; do
    [ -d "$_d" ] && LD_LIBRARY_PATH="$_d:${LD_LIBRARY_PATH:-}"
  done
  export LD_LIBRARY_PATH
fi

export HF_DATASETS_CACHE=/tmp/hf_datasets_cache
export HF_HOME=/tmp/hf_home
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MODELING_BACKEND=hf
export VERL_LOGGER="[console,swanlab]"

source "$PROJECT_DIR/scripts/load_training_env.sh"

BASE_MODEL=/mnt/afs_toolcall/sunhao4/models/Qwen3-8B
TEMPLATE="$PROJECT_DIR/configs/run/b1_8b.yaml"
DATASETS_DIR="$PROJECT_DIR/datasets"
CKPT_BASE="$PROJECT_DIR/ckpts/qwen3_8b_b1"
LOG_DIR="$PROJECT_DIR/logs/experiments/run/qwen3_8b_b1"

# 9 buckets, steps proportional to data size (total 50)
BUCKETS=(
  "workflow:7"
  "ops:18"
  "qa:1"
  "finance:1"
  "office:4"
  "communication:2"
  "safety:1"
  "coding:9"
  "research:7"
)

mkdir -p "$CKPT_BASE" "$LOG_DIR"

PREV_CKPT="$BASE_MODEL"

for entry in "${BUCKETS[@]}"; do
  BUCKET="${entry%%:*}"
  STEPS="${entry##*:}"

  echo "============================================"
  echo "[b1_bucket] bucket=$BUCKET steps=$STEPS model=$PREV_CKPT"
  echo "============================================"

  BUCKET_CKPT="$CKPT_BASE/$BUCKET"
  mkdir -p "$BUCKET_CKPT" "$LOG_DIR/$BUCKET/rollout" "$LOG_DIR/$BUCKET/val"

  python -m trainer.cl_main --config "$TEMPLATE" \
    "actor_rollout_ref.model.path=$PREV_CKPT" \
    "actor_rollout_ref.ref.path=$BASE_MODEL" \
    "data.train_files=$DATASETS_DIR/train_$BUCKET.parquet" \
    "trainer.total_training_steps=$STEPS" \
    "trainer.save_freq=$STEPS" \
    "trainer.test_freq=$STEPS" \
    "trainer.default_local_dir=$BUCKET_CKPT" \
    "trainer.experiment_name=qwen3_8b_b1_$BUCKET" \
    2>&1 | tee "$LOG_DIR/$BUCKET/train.log" &
  _TRAIN_PID=$!
  wait $_TRAIN_PID || true

  # Find latest checkpoint saved by verl
  LATEST=$(ls -dt "$BUCKET_CKPT"/global_step_* 2>/dev/null | head -1)
  if [ -n "$LATEST" ]; then
    PREV_CKPT="$LATEST/actor"
    echo "[b1_bucket] checkpoint: $PREV_CKPT"
  else
    echo "[b1_bucket] WARNING: no checkpoint found, continuing with previous model"
  fi
done

echo "[b1_bucket] all 9 buckets done"
