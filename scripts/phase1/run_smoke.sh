#!/usr/bin/env bash
# Phase 1 / B1 — 4-GPU 1-step smoke 启动器。
#
# 验证 FSDP + vllm rollout 完整循环跑通 1 个 step。reward 用本地 mock judge
# （固定满分，仅验链路）。需要 4 张空闲 H800。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV="${VENV:-/mnt/afs_toolcall/sunhao4/envs/vllm019_venv}"
PY="$VENV/bin/python"
CONFIG="$ROOT_DIR/configs/phase1/smoke_1step.yaml"

# --- mock judge (固定满分 reward) ------------------------------------------
export JUDGE_API_BASE="${JUDGE_API_BASE:-http://127.0.0.1:8100/v1}"
export JUDGE_MODEL="${JUDGE_MODEL:-mock-judge}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-sk-local}"

# 若 mock judge 没在跑就拉起来（后台）
if ! curl -sS -m 3 "$JUDGE_API_BASE/models" >/dev/null 2>&1; then
  echo "[smoke] starting mock judge on $JUDGE_API_BASE"
  "$PY" "$ROOT_DIR/scripts/mock_judge.py" --port 8100 > /tmp/mock_judge.log 2>&1 &
  for i in $(seq 1 10); do
    curl -sS -m 2 "$JUDGE_API_BASE/models" >/dev/null 2>&1 && break; sleep 1
  done
fi

# AFS (quarkfs FUSE) 不支持 fcntl.flock → datasets 5.x filelock 报 FileNotFoundError。
# 将 HF 和 verl 的缓存目录都指向 /tmp（本地 ext4）。
export HF_DATASETS_CACHE="/tmp/hf_datasets_cache"
export HF_HOME="/tmp/hf_home"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# Qwen3.6 GDN：训练内 vllm rollout 也要绕 FlashInfer 死锁（配置已带 additional_config，
# 这里再兜底一个 env，万一 engine_kwargs 没透传到 vllm）
export VLLM_GDN_PREFILL_BACKEND="${VLLM_GDN_PREFILL_BACKEND:-triton}"

cd "$ROOT_DIR"
echo "[smoke] config=$CONFIG gpus=$CUDA_VISIBLE_DEVICES judge=$JUDGE_API_BASE"
exec "$PY" -m trainer.cl_main --config "$CONFIG" "$@"
