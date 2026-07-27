#!/usr/bin/env bash
# Launch a FROZEN local reward judge as an OpenAI-compatible vLLM server.
#
# The reward judge grades every rollout trajectory (trainer/model_reward.py). It
# must be FROZEN for the whole training run (reproducible reward) and >= the
# policy in capability (anti reward-hacking). The judge model is NOT hardcoded:
# pass a model path / HF id via $REWARD_MODEL_PATH or as the first argument.
#
# Usage:
#   REWARD_MODEL_PATH=/mnt/afs/models/qwen2.5-32b-instruct bash scripts/serve/serve_reward_model.sh
#   bash scripts/serve/serve_reward_model.sh /mnt/afs/models/qwen2.5-32b-instruct
#
# Then point the trainer at it:
#   export REWARD_API_BASE=http://127.0.0.1:${JUDGE_PORT:-8100}/v1
#   export REWARD_MODEL=${JUDGE_SERVED_NAME:-reward-judge}
#   export REWARD_API_KEY=sk-local
#
# Sizing guidance (doc/sandbox/Sandbox_Agent架构.md): 32B is the default sweet spot for a
# 27B policy; validate with ClawEval human-rubric agreement before trusting it.

set -euo pipefail

MODEL_PATH="${REWARD_MODEL_PATH:-${1:-}}"
PORT="${JUDGE_PORT:-8100}"
TP="${JUDGE_TP:-2}"                       # tensor-parallel size = #GPUs for the judge
GPUS="${JUDGE_GPUS:-0,1}"                 # which cards (carve from the inference pool)
MAX_LEN="${JUDGE_MAX_LEN:-32768}"         # long agentic transcripts need long context
GPU_MEM_UTIL="${JUDGE_GPU_MEM_UTIL:-0.90}"
SERVED_NAME="${JUDGE_SERVED_NAME:-reward-judge}"
DTYPE="${JUDGE_DTYPE:-bfloat16}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "[serve_reward_model] ERROR: no model. Set REWARD_MODEL_PATH or pass a path arg."
  echo "  e.g. REWARD_MODEL_PATH=/mnt/afs/models/qwen2.5-32b-instruct bash $0"
  exit 2
fi

echo "[serve_reward_model] model=${MODEL_PATH}"
echo "[serve_reward_model] serving '${SERVED_NAME}' on :${PORT} | GPUs=${GPUS} TP=${TP} max_len=${MAX_LEN}"

CUDA_VISIBLE_DEVICES="${GPUS}" python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --dtype "${DTYPE}" \
  --disable-log-requests

# After startup, sanity check:
#   curl -s http://127.0.0.1:${PORT}/v1/models | python -m json.tool
