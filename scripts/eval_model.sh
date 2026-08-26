#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 评测脚本 1：对单个模型跑 ClawEval 195 题 + 算 pass@N / 均分 / 按桶
#
# 用法：
#   bash scripts/eval_model.sh <model_path> [output_name] [num_runs]
#
# 参数：
#   model_path   — HF 格式模型路径（基座或 merge 后的 ckpt）
#   output_name  — 输出目录名（默认 = model_path 的 basename）
#   num_runs     — Pass^N 的 N（默认 3）
#
# 示例：
#   # 评基座
#   bash scripts/eval_model.sh /mnt/afs_toolcall/sunhao4/models/Qwen3.5-9B base
#   # 评某个 checkpoint（先 merge）
#   bash scripts/eval_model.sh /tmp/merged_hf_cl2r_cl_200 cl2r_cl_step200
#
# 输出：eval/results/<output_name>/
#   per_task.json  — 每题四维打分（cl_eval 输出）
#   scores.json    — 聚合：pass@N / 均分 / 按桶 / 与 baseline 对比（如有）
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY="/opt/conda/bin/python"
VERL_DIR="/mnt/afs_toolcall/sunhao4/dependencies/verl"
LIGHTLLM_DIR="/mnt/afs_toolcall/sunhao4/workspace/LightLLM"
export PYTHONPATH="$LIGHTLLM_DIR:$VERL_DIR:$ROOT_DIR/src:$ROOT_DIR:${PYTHONPATH:-}"

MODEL_PATH="${1:?用法: bash scripts/eval_model.sh <model_path> [output_name] [num_runs]}"
OUT_NAME="${2:-$(basename "$MODEL_PATH")}"
NUM_RUNS="${3:-3}"
NNODES="${NNODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

OUT_DIR="$ROOT_DIR/eval/results/$OUT_NAME"
mkdir -p "$OUT_DIR"

echo "=== 评测: $MODEL_PATH → $OUT_DIR (Pass^$NUM_RUNS) ==="

# 凭证
source "$ROOT_DIR/scripts/env/load_tencent_env.sh" 2>/dev/null || true
source "$ROOT_DIR/scripts/env/load_training_env.sh" 2>/dev/null || true

# 起 Ray
RAY_CPUS="${CL_RAY_NUM_CPUS:-$(nproc 2>/dev/null || echo 128)}"
ray start --head --disable-usage-stats --num-cpus "$RAY_CPUS"

# 跑 cl_eval（复用训练 RemoteAgentLoopManager）
"$PY" -m trainer.cl_eval \
  --config "$ROOT_DIR/configs/run/r0-25k_9b_16gpu.yaml" \
  --tasks-file "$ROOT_DIR/src/eval/claweval_manifest.json" \
  --output "$OUT_DIR/per_task.json" \
  --num-runs "$NUM_RUNS" \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  cl.buffer.enabled=false \
  trainer.nnodes="$NNODES" trainer.n_gpus_per_node="$GPUS_PER_NODE"

ray stop --force >/dev/null 2>&1 || true

# 聚合结果
echo "=== 聚合评测结果 ==="
BASELINE_FILE="$ROOT_DIR/eval/results/base/per_task.json"
if [ -f "$BASELINE_FILE" ] && [ "$OUT_NAME" != "base" ]; then
  "$PY" -m eval.aggregate --results "$OUT_DIR/per_task.json" --baseline "$BASELINE_FILE" --output "$OUT_DIR/scores.json"
else
  "$PY" -m eval.aggregate --results "$OUT_DIR/per_task.json" --output "$OUT_DIR/scores.json"
fi

echo "✅ 评测完成: $OUT_DIR/scores.json"
