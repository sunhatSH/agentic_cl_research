#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 评测脚本 2：第一部分训练后自动评测（附到训练脚本后，训练完成自动执行）
#
# 逻辑：
#   1. 训练完成（run_cl.sh train 跑完 200 步）后自动触发
#   2. merge 最后一个 checkpoint（FSDP→HF）
#   3. 跑 ClawEval 195 题 × Pass^5
#   4. 聚合 pass@5 / 均分 / 按桶 / 与 base 对比
#   5. 边训边评（stage2 模式：每桶训完立即评）
#
# 用法（自动模式，附在 run_cl.sh stage1 后）：
#   bash scripts/run_cl.sh stage1 cl    # 训完自动评
#   bash scripts/eval_after_train.sh <exp> <step> [method]  # 手动指定
#
# 边训边评（stage2 模式）：
#   bash scripts/run_cl.sh stage2 cl    # 每桶训完立即评
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY="/opt/conda/bin/python"
VERL_DIR="/mnt/afs_toolcall/sunhao4/dependencies/verl"
LIGHTLLM_DIR="/mnt/afs_toolcall/sunhao4/workspace/LightLLM"
export PYTHONPATH="$LIGHTLLM_DIR:$VERL_DIR:$ROOT_DIR/src:$ROOT_DIR:${PYTHONPATH:-}"

# ★ 外部 patch 模块清单（与训练同一份）：本脚本有独立的 cl_eval 路径（_eval_ckpt），
# 评测 rollout 走 AgentSessionWorker，其 create_hook 要认 FQN hook
# trainer.observer_hook.ObserverDiffHook。不 source 则 worker 每 session ValueError:
# Unknown post-run hook → 全 abort（2026-08-26 评测全灭根因）。
source "$ROOT_DIR/scripts/env/verl_external_modules.sh"

NNODES="${NNODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NUM_RUNS="${EVAL_NUM_RUNS:-3}"
BASE_MODEL="/mnt/afs_toolcall/sunhao4/models/Qwen3.5-9B"
RESULTS_DIR="$ROOT_DIR/eval/results"

# ── 评测单个 checkpoint ──
_eval_ckpt() {
  local ckpt_actor="$1" model_type="$2"
  local out_dir="$RESULTS_DIR/$model_type"
  local per_task="$out_dir/per_task.json"

  if [ -f "$per_task" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[$model_type] 已有结果，跳过"
    return
  fi

  echo ">>> 评测: $model_type (ckpt=$ckpt_actor)"

  # merge FSDP→HF（结果放 AFS，跨节点 worker 要加载，/tmp 不共享）
  local merged="$ROOT_DIR/eval/.tmp/merged_${model_type}"
  mkdir -p "$ROOT_DIR/eval/.tmp"
  "$PY" -m verl.model_merger merge --backend fsdp --local_dir "$ckpt_actor" --target_dir "$merged"

  mkdir -p "$out_dir"
  local ray_cpus="${CL_RAY_NUM_CPUS:-$(nproc 2>/dev/null || echo 128)}"
  ray start --head --disable-usage-stats --num-cpus "$ray_cpus"

  "$PY" -m trainer.cl_eval \
    --config "$ROOT_DIR/configs/run/r0-25k_9b_16gpu.yaml" \
    --tasks-file "$ROOT_DIR/src/eval/claweval_manifest.json" \
    --output "$per_task" \
    --num-runs "$NUM_RUNS" \
    actor_rollout_ref.model.path="$merged" \
    cl.buffer.enabled=false \
    trainer.nnodes="$NNODES" trainer.n_gpus_per_node="$GPUS_PER_NODE"

  ray stop --force >/dev/null 2>&1 || true

  # 聚合（与 base 对比）
  local baseline_flag=""
  if [ -f "$RESULTS_DIR/base/per_task.json" ]; then
    baseline_flag="--baseline $RESULTS_DIR/base/per_task.json"
  fi
  "$PY" -m eval.aggregate --results "$per_task" $baseline_flag --output "$out_dir/scores.json"
  echo "    ✅ $model_type: $(cat "$out_dir/scores.json" | "$PY" -c 'import json,sys; s=json.load(sys.stdin)["summary"]; print(f"pass@{0}={1:.3f} mean={2:.3f}".format("N",s["pass_at_n"],s["mean_reward"]))')"
}

# ── 评 base（如果还没评过）──
_ensure_base() {
  if [ ! -f "$RESULTS_DIR/base/per_task.json" ] || [ "${FORCE:-0}" = "1" ]; then
    echo ">>> 评 base（参照）"
    bash "$ROOT_DIR/scripts/eval_model.sh" "$BASE_MODEL" base "$NUM_RUNS"
  fi
}

# ── 主逻辑 ──
case "${1:-}" in
  # 手动：eval_after_train.sh <exp> <step> [method]
  *)
    EXP="${1:?用法: bash eval_after_train.sh <exp> <step>}"
    STEP="${2:?需要 step}"
    METHOD="${3:-}"

    CKPT="$ROOT_DIR/ckpts/$EXP/global_step_$STEP/actor"
    if [ ! -d "$CKPT" ]; then
      echo "❌ checkpoint 不存在: $CKPT"
      exit 1
    fi

    _ensure_base
    MODEL_TYPE="${EXP#cl2r_}_step${STEP}"
    _eval_ckpt "$CKPT" "$MODEL_TYPE"
    ;;

  # 边训边评模式（stage2 调用）
  stage2)
    METHOD="${2:?需要 method}"
    EXP="cl2r_exp2_${METHOD}"
    BUCKETS="office:50 ops:50 workflow:50 qa:4 finance:13 safety:20 communication:5"
    CUM=0
    _ensure_base
    for entry in $BUCKETS; do
      B="${entry%%:*}"; STEPS="${entry##*:}"
      CUM=$((CUM + STEPS))
      CKPT="$ROOT_DIR/ckpts/$EXP/global_step_$CUM/actor"
      if [ -d "$CKPT" ]; then
        _eval_ckpt "$CKPT" "${METHOD}_exp2_step${CUM}"
      else
        echo "⚠️ $EXP step $CUM 还没训完，跳过评测"
      fi
    done
    ;;
esac

echo "✅ 评测完成"
