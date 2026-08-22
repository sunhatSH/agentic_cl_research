#!/usr/bin/env bash
# CL 分阶段实验总入口：一个脚本、四个动作（train / eval / stage1 / stage2）。
#
#   train  — 训练一段（续训靠 auto-resume）
#   eval   — 评测（评 base 或评某个 checkpoint）
#   stage1 — 第一部分：coding→research 续训 200+200 + 评 2 checkpoint
#   stage2 — 第二部分：7 桶少量边训边评
#
# 用法:
#   bash scripts/run_cl.sh train <method> <exp> <total_steps> [override...]
#   bash scripts/run_cl.sh eval  base
#   bash scripts/run_cl.sh eval  <ckpt_actor> <exp> <step>
#   bash scripts/run_cl.sh stage1 [method]     # method ∈ {baseline, kl, clear, cl}
#   bash scripts/run_cl.sh stage2 [method]
#
# env:
#   NNODES / GPUS_PER_NODE   评测 GPU 配置（默认 2×8；debug 机设 NNODES=1 GPUS_PER_NODE=4）
#   CL_RAY_NUM_CPUS          Ray num-cpus（默认 nproc）
#
# 训练固定走 train.sh 16gpu（nnodes=2 gpus=8）；评测用 NNODES/GPUS_PER_NODE（主动传，不自动检测）。
# 详见 scripts/README_experiments.md。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="configs/exp1/cl2r_base.yaml"   # 相对 ROOT，train.sh / train_4gpu.sh 内部拼 ROOT
PY="/opt/conda/bin/python"
VERL_DIR="/mnt/afs_toolcall/sunhao4/dependencies/verl"
LIGHTLLM_DIR="/mnt/afs_toolcall/sunhao4/workspace/LightLLM"
export PYTHONPATH="$LIGHTLLM_DIR:$VERL_DIR:$ROOT_DIR/src:$ROOT_DIR:${PYTHONPATH:-}"
NNODES="${NNODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

# 4 方法（cl2r_base.yaml 默认 = CLEAR 单桶）
declare -A CL_OVERRIDE=(
  [baseline]="cl.buffer.enabled=false cl.lambda_replay=0.0 actor_rollout_ref.actor.use_kl_loss=false actor_rollout_ref.actor.kl_loss_coef=0.0"
  [kl]="cl.buffer.enabled=false cl.lambda_replay=0.0 actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=0.05 actor_rollout_ref.actor.kl_loss_type=low_var_kl"
  [clear]=""
  [cl]="cl.buffer.enabled=true cl.buffer.num_buckets=9 cl.buffer.eviction_type=fifo cl.buffer.bucket_strategy=distance cl.weighting.scheme=W2"
)

_usage() { grep '^#' "$0" | head -14; exit 1; }

_load_env() {
  source "$ROOT_DIR/scripts/env/load_tencent_env.sh"
  source "$ROOT_DIR/scripts/env/load_training_env.sh"
}

# ── train：训练一段（续训靠 train.sh/train_4gpu.sh auto-resume）──
_do_train() {  # <method> <exp> <total_steps> [override...]
  local m="$1" exp="$2" total="$3"; shift 3
  # clear 的 override 是空串（CLEAR baseline 无额外覆盖），故用 -? 而非 :?——
  # :? 对空值也报错，会误杀合法的 method=clear。先校验 key 存在，再取值（可空）。
  [ -v "CL_OVERRIDE[$m]" ] || { echo "未知 method=$m（baseline/clear/cl）" >&2; exit 1; }
  local ov="${CL_OVERRIDE[$m]}"
  local extra=($ov trainer.total_training_steps="$total" trainer.experiment_name="$exp" "$@")
  if [ "$NNODES" = "1" ] && [ "$GPUS_PER_NODE" = "4" ]; then
    # 4 卡 debug：走 train_4gpu.sh（内部固定 nnodes=1 gpus=4, SP=4）
    EXP_NAME="$exp" bash "$ROOT_DIR/scripts/train_4gpu.sh" "$CONFIG" "${extra[@]}"
  else
    # 16 卡：走 train.sh 16gpu
    EXP_NAME="$exp" bash "$ROOT_DIR/scripts/train.sh" 16gpu --config "$CONFIG" "${extra[@]}"
  fi
}

# ── eval：评一个 model.path（起 Ray + cl_eval + 停 Ray）──
_run_eval() {  # <model_path> <out_dir>
  local model_path="$1" out_dir="$2"
  mkdir -p "$out_dir"
  local ray_cpus="${CL_RAY_NUM_CPUS:-$(nproc 2>/dev/null || echo 128)}"
  ray start --head --disable-usage-stats --num-cpus "$ray_cpus"
  "$PY" -m trainer.cl_eval \
    --config "$ROOT_DIR/configs/run/r0-25k_9b_16gpu.yaml" \
    --tasks-file "$ROOT_DIR/eval/claweval_manifest.json" \
    --output "$out_dir/per_task.json" \
    --num-runs 3 \
    actor_rollout_ref.model.path="$model_path" \
    cl.buffer.enabled=false \
    trainer.nnodes="$NNODES" trainer.n_gpus_per_node="$GPUS_PER_NODE"
  ray stop --force >/dev/null 2>&1 || true
}

_do_eval() {  # base | <ckpt_actor> <exp> <step>
  if [ "${1:-}" = "base" ]; then
    _run_eval "/mnt/afs_toolcall/sunhao4/models/Qwen3.5-9B" "$ROOT_DIR/eval/results/base"
  else
    local ckpt="$1" exp="$2" step="$3"
    local merged="/tmp/merged_hf_${exp}_${step}"
    "$PY" -m verl.model_merger merge --backend fsdp --local_dir "$ckpt" --target_dir "$merged"
    _run_eval "$merged" "$ROOT_DIR/eval/results/${exp}_step${step}"
  fi
}

# ── stage1：第一部分 coding→research 续训 200+200 + 评 2 checkpoint ──
_do_stage1() {  # <method>
  local m="$1" exp="cl2r_${m}"
  _do_train "$m" "$exp" 200
  _do_train "$m" "$exp" 400
  _do_eval "$ROOT_DIR/ckpts/$exp/global_step_200/actor" "$exp" 200
  _do_eval "$ROOT_DIR/ckpts/$exp/global_step_400/actor" "$exp" 400
}

# ── stage2：第二部分 7 桶少量边训边评 ──
_do_stage2() {  # <method>
  local m="$1" exp="cl2r_exp2_${m}"
  local buckets="office:50 ops:50 workflow:50 qa:4 finance:13 safety:20 communication:5"
  local cum=0
  for entry in $buckets; do
    local b="${entry%%:*}" steps="${entry##*:}"
    cum=$((cum + steps))
    _do_train "$m" "$exp" "$cum" data.train_files="$ROOT_DIR/datasets/train_exp2.parquet" trainer.save_freq=1
    _do_eval "$ROOT_DIR/ckpts/$exp/global_step_$cum/actor" "$exp" "$cum"
  done
}

# ── 分发 ──
CMD="${1:-}"; [ -z "$CMD" ] && _usage
_load_env
case "$CMD" in
  train)  [ $# -ge 4 ] || _usage; _do_train "$2" "$3" "$4" "${@:5}" ;;
  eval)   _do_eval "${2:-}" "${3:-}" "${4:-}" ;;
  stage1) _do_stage1 "${2:-all}" ;;
  stage2) _do_stage2 "${2:-all}" ;;
  *)      _usage ;;
esac
