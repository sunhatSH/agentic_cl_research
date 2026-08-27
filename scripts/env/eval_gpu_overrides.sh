#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 评测 GPU 规模 → OmegaConf override 组（16gpu 默认 / 4gpu）。source 我后得到：
#   · NNODES / GPUS_PER_NODE（若调用方没显式设，按 GPU_MODE 预设）
#   · EVAL_GPU_OVERRIDES 数组：传给 `python -m trainer.cl_eval` 的 key=value 组
#
# 机制层与 scripts/train_4gpu.sh 的 4 卡参数逐字对齐（只是评测不训练）：
#   16gpu: 2节点×8卡, SP=4→DP=4, rollout_TP=2, gateway/workers=8, gpu_mem=0.65
#   4gpu : 单机 4 卡,  SP=4→DP=1, rollout_TP=2, gateway/workers=4, gpu_mem=0.75
#     （SP=4 → DP=TOTAL_GPUS/SP=4/4=1；评测无 train_batch%DP 整除约束，DP=1 恒安全）
#
# 用法（在 eval_model.sh / eval_all.sh 里）：
#   source "$ROOT_DIR/scripts/env/eval_gpu_overrides.sh"
#   "$PY" -m trainer.cl_eval ... "${EVAL_GPU_OVERRIDES[@]}"
#
# 覆盖优先级：显式 NNODES=/GPUS_PER_NODE= > GPU_MODE 预设 > 默认(16gpu)。
# ─────────────────────────────────────────────────────────────────────────────

GPU_MODE="${GPU_MODE:-16gpu}"

case "$GPU_MODE" in
  4gpu)
    NNODES="${NNODES:-1}"
    GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
    EVAL_GPU_OVERRIDES=(
      actor_rollout_ref.rollout.tensor_model_parallel_size=2
      actor_rollout_ref.actor.ulysses_sequence_parallel_size=4
      actor_rollout_ref.rollout.gpu_memory_utilization=0.75
      actor_rollout_ref.rollout.custom.remote_agent.gateway_count=4
      actor_rollout_ref.rollout.custom.remote_agent.num_session_workers=4
    )
    ;;
  16gpu)
    NNODES="${NNODES:-2}"
    GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
    EVAL_GPU_OVERRIDES=()   # 用 config 里的 16 卡默认值，不额外 override
    ;;
  *)
    echo "[eval_gpu] ERROR: 未知 GPU_MODE='$GPU_MODE'（支持 16gpu / 4gpu）" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac

echo "[eval_gpu] GPU_MODE=$GPU_MODE → NNODES=$NNODES GPUS_PER_NODE=$GPUS_PER_NODE overrides=${#EVAL_GPU_OVERRIDES[@]}项"
