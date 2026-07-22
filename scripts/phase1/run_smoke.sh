#!/usr/bin/env bash
# Phase 1 / B1 — 4-GPU 1-step smoke 启动器。
#
# 验证 FSDP + lightllm rollout 完整循环跑通 1 个 step。reward 用本地 mock judge
# （固定满分，仅验链路）。需要 4 张空闲 H800。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# 训练环境 = /opt/conda（torch 2.9 + cu129 + ray/vllm/verl 全栈依赖，4 卡 H800）。
VENV="${VENV:-/opt/conda}"
PY="$VENV/bin/python"
CONFIG="$ROOT_DIR/configs/phase1/smoke_1step.yaml"

# verl / LightLLM 走 AFS 源码（含 lightllm-agent 增量），与 dev_env.sh 同源同序。
# smoke rollout 用 lightllm（Qwen3.6 架构 vllm 不支持），故 LightLLM + verl 源码都必须在 path。
VERL_DIR="${VERL_DIR:-/mnt/afs_toolcall/sunhao4/workspace/verl}"
LIGHTLLM_DIR="${LIGHTLLM_DIR:-/mnt/afs_toolcall/sunhao4/workspace/LightLLM}"
export PYTHONPATH="$LIGHTLLM_DIR:$VERL_DIR:$ROOT_DIR:${PYTHONPATH:-}"

# --- 训练前环境自检（缺失/版本不符的依赖自动补装，不过关直接中止）----------
PY="$PY" bash "$ROOT_DIR/scripts/check_train_env.sh" || {
  echo "[smoke] 环境自检失败，中止（见上方 [env] FAIL）"; exit 5; }

# --- mock judge (固定满分 reward) ------------------------------------------
export REWARD_API_BASE="${REWARD_API_BASE:-http://127.0.0.1:8100/v1}"
export REWARD_MODEL="${REWARD_MODEL:-mock-judge}"
export TOKENHUB_API_KEY="${TOKENHUB_API_KEY:-sk-local}"

# 若 mock judge 没在跑就拉起来（后台）
if ! curl -sS -m 3 "$REWARD_API_BASE/models" >/dev/null 2>&1; then
  echo "[smoke] starting mock judge on $REWARD_API_BASE"
  "$PY" "$ROOT_DIR/scripts/mock_judge.py" --port 8100 > /tmp/mock_judge.log 2>&1 &
  for i in $(seq 1 10); do
    curl -sS -m 2 "$REWARD_API_BASE/models" >/dev/null 2>&1 && break; sleep 1
  done
fi

# AFS (quarkfs FUSE) 不支持 fcntl.flock → datasets 5.x filelock 报 FileNotFoundError。
# 将 HF 和 verl 的缓存目录都指向 /tmp（本地 ext4）。
export HF_DATASETS_CACHE="/tmp/hf_datasets_cache"
export HF_HOME="/tmp/hf_home"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# 注意：不要设 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True —— 它与 lightllm 的
# enable_torch_memory_saver 互斥（torch_memory_saver 不支持 expandable_segments，会让
# LightLLMHttpServer 启动即崩）。显存靠 rollout.gpu_memory_utilization=0.30 + memory_saver 控制。
# transformers 由 /opt/conda 提供（5.12，已认 Qwen3.6 qwen3_5 架构）——不再降级安装。
# Qwen3.6 GDN：训练内 vllm rollout 也要绕 FlashInfer 死锁（配置已带 additional_config，
# 这里再兜底一个 env，万一 engine_kwargs 没透传到 vllm）
export VLLM_GDN_PREFILL_BACKEND="${VLLM_GDN_PREFILL_BACKEND:-triton}"

cd "$ROOT_DIR"
echo "[smoke] config=$CONFIG gpus=$CUDA_VISIBLE_DEVICES reward=$REWARD_API_BASE"
exec "$PY" -m trainer.cl_main --config "$CONFIG" "$@"
