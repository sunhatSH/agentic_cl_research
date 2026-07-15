#!/usr/bin/env bash
# Launch B1 8B training on single-node 8 GPU (Qwen3-8B).
# Sets up: py310_base conda env + PYTHONPATH (LightLLM/verl/project) + .env creds.
set -euo pipefail

PROJECT_DIR=/mnt/afs_toolcall/sunhao4/agentic_cl_research
LIGHTLLM_DIR=/mnt/afs_toolcall/sunhao4/Documents/LightLLM
VERL_DIR=/mnt/afs_toolcall/sunhao4/Documents/verl
CONDA_ENV=/mnt/afs_toolcall/sunhao4/miniconda3/envs/py310_base

# --- Python env (py310_base has torch/ray/vllm/verl-deps installed) ---
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH="$LIGHTLLM_DIR:$VERL_DIR:$PROJECT_DIR"
export PYTHON=$CONDA_ENV/bin/python

# --- HF cache to local disk (AFS doesn't support flock) ---
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache
export HF_HOME=/tmp/hf_home
export TOKENIZERS_PARALLELISM=false

# --- Training credentials (.env: SWANLAB_API_KEY + SUFY_API_KEY + TOKENHUB_API_KEY) ---
source "$PROJECT_DIR/scripts/load_training_env.sh"

# --- Reward judge: sufy claude-4.8-opus (resolved from configs/agents.yaml) ---
# REWARD_API_BASE/MODEL not needed — model_reward.py reads agents.yaml reward section.
export MODELING_BACKEND=hf

# --- Checkpoint + log dirs ---
export CKPT_DIR="$PROJECT_DIR/ckpts/qwen3_8b_b1"
export ROLLOUT_DATA_DIR="$PROJECT_DIR/logs/experiments/run/qwen3_8b_b1/rollout"
export VAL_DATA_DIR="$PROJECT_DIR/logs/experiments/run/qwen3_8b_b1/val"
mkdir -p "$CKPT_DIR" "$ROLLOUT_DATA_DIR" "$VAL_DATA_DIR"

# --- Logger: console + swanlab (so we can watch loss live) ---
export VERL_LOGGER="[console,swanlab]"

cd "$PROJECT_DIR"
echo "[run_b1_8b] python: $(python --version)"
echo "[run_b1_8b] model: /mnt/afs_toolcall/sunhao4/models/Qwen3-8B"
echo "[run_b1_8b] config: configs/run/b1_8b.yaml"
echo "[run_b1_8b] gpus: $(python -c 'import torch;print(torch.cuda.device_count())')"
echo "[run_b1_8b] swanlab key: ${SWANLAB_API_KEY:+present}"
echo "[run_b1_8b] starting training..."

python -m trainer.cl_main --config configs/run/b1_8b.yaml 2>&1 | tee "$PROJECT_DIR/logs/experiments/run/qwen3_8b_b1/train.log"
