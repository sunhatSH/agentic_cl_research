#!/usr/bin/env bash
# Launch B1 9B training on single-node 8 GPU (Qwen3.5-9B).
# Sets up: py310_base conda env + PYTHONPATH (LightLLM/verl/project) + .env creds.
set -euo pipefail

cleanup() {
    local exit_code=$?
    echo "[cleanup] training exited with code=$exit_code, killing GPU processes..."
    if [ -n "${_TRAIN_PID:-}" ] && kill -0 "$_TRAIN_PID" 2>/dev/null; then
        kill -- -$(ps -o pgid= -p "$_TRAIN_PID" 2>/dev/null | tr -d ' ') 2>/dev/null || true
    fi
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do
        kill -9 "$pid" 2>/dev/null || true
    done
    echo "[cleanup] done"
}
trap cleanup EXIT

PROJECT_DIR=/mnt/afs_toolcall/sunhao4/workspace/agentic_cl_research
LIGHTLLM_DIR=/mnt/afs_toolcall/sunhao4/workspace/LightLLM
VERL_DIR=/mnt/afs_toolcall/sunhao4/workspace/verl
CONDA_ENV=/opt/conda

# --- Python env ---
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH="$LIGHTLLM_DIR:$VERL_DIR:$PROJECT_DIR"
export PYTHON=$CONDA_ENV/bin/python

# --- 训练前环境自检（缺失/版本不符依赖自动补装，不过关直接中止）----------
PY="$PYTHON" bash "$PROJECT_DIR/scripts/check_train_env.sh" || {
  echo "[run_b1_8b] 环境自检失败，中止"; exit 5; }

# --- CUDA runtime libs ---
_NV_LIB_ROOT="$CONDA_ENV/lib/python3.11/site-packages/nvidia"
if [ -d "$_NV_LIB_ROOT" ]; then
  for _d in "$_NV_LIB_ROOT"/*/lib; do
    [ -d "$_d" ] && LD_LIBRARY_PATH="$_d:${LD_LIBRARY_PATH:-}"
  done
  export LD_LIBRARY_PATH
fi

# --- HF cache to local disk (AFS doesn't support flock) ---
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache
export HF_HOME=/tmp/hf_home
export HF_HUB_OFFLINE=1              # 模型在本地，不走 hub 校验

# --- Training credentials (.env: SWANLAB_API_KEY + SUFY_API_KEY + TOKENHUB_API_KEY) ---
source "$PROJECT_DIR/scripts/load_training_env.sh"

# --- Reward judge: sufy claude-4.8-opus (resolved from configs/agents.yaml) ---
# REWARD_API_BASE/MODEL not needed — model_reward.py reads agents.yaml reward section.
export MODELING_BACKEND=hf

# --- Checkpoint + log dirs ---
export CKPT_DIR="$PROJECT_DIR/ckpts/qwen35_9b_b1"
export ROLLOUT_DATA_DIR="$PROJECT_DIR/logs/experiments/run/qwen35_9b_b1/rollout"
export VAL_DATA_DIR="$PROJECT_DIR/logs/experiments/run/qwen35_9b_b1/val"
mkdir -p "$CKPT_DIR" "$ROLLOUT_DATA_DIR" "$VAL_DATA_DIR"

# --- Logger: console + swanlab (so we can watch loss live) ---
export VERL_LOGGER="[console,swanlab]"

cd "$PROJECT_DIR"
echo "[run_b1_8b] python: $(python --version)"
echo "[run_b1_8b] model: /mnt/afs_toolcall/sunhao4/models/Qwen3.5-9B"
echo "[run_b1_8b] config: configs/run/b1_8b.yaml"
echo "[run_b1_8b] gpus: $(python -c 'import torch;print(torch.cuda.device_count())')"
echo "[run_b1_8b] swanlab key: ${SWANLAB_API_KEY:+present}"
echo "[run_b1_8b] starting training..."

python -m trainer.cl_main --config configs/run/b1_8b.yaml 2>&1 | tee "$PROJECT_DIR/logs/experiments/run/qwen35_9b_b1/train.log" &
_TRAIN_PID=$!
wait $_TRAIN_PID || true
