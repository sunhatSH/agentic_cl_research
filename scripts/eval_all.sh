#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 批量评测脚本：依次评 base + 所有 ckpt，记录 pass@3 / pass^3 / 均分 / 按桶
#
# 评测对象：
#   1. Qwen3.5-9B 基座（base，参照）
#   2. ckpts/*/global_step_*/actor（所有训练 checkpoint）
#
# 多机（16卡=2机×8）：与训练一致，靠平台注入 RANK/MASTER_ADDR。
#   - RANK=0（master）：ray start --head → 循环评测 → ray stop
#   - RANK>0（worker）：ray start --address $MASTER_ADDR:6379 --block（常驻提供 GPU）
#   单机（NNODES=1）：RANK 默认 0，直接 head。
#
# 输出：eval/results/<model_type>/
#   per_task.json  — 每题四维打分
#   scores.json    — 聚合：pass@3 / pass^3 / 均分 / 按桶 / 与 base 对比
#
# 用法（16 卡，平台自动在 2 机各跑一份，注入 RANK）：
#   bash scripts/eval_all.sh            # Pass^3
# 单机 8 卡：
#   NNODES=1 bash scripts/eval_all.sh 3
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY="/opt/conda/bin/python"
VERL_DIR="/mnt/afs_toolcall/sunhao4/dependencies/verl"
LIGHTLLM_DIR="/mnt/afs_toolcall/sunhao4/workspace/LightLLM"
export PYTHONPATH="$LIGHTLLM_DIR:$VERL_DIR:$ROOT_DIR/src:$ROOT_DIR:${PYTHONPATH:-}"

# ★ 外部 patch 模块清单（与训练同一份）：评测 rollout 走的也是 AgentSessionWorker，
# 其 create_hook 要认 FQN hook trainer.observer_hook.ObserverDiffHook（agent_loop_config.yaml
# 挂的）。不 source 则 worker 不 import observer_hook_register → 每 session ValueError:
# Unknown post-run hook → 全 abort（2026-08-26 评测全灭根因）。verl_runner.run_cl_eval 会把
# VERL_USE_EXTERNAL_MODULES 经 runtime_env 透传给所有 Ray worker。
source "$ROOT_DIR/scripts/env/verl_external_modules.sh"

NUM_RUNS="${1:-3}"   # Pass^N，默认 3（pass@3 + pass^3）
BASE_MODEL="/mnt/afs_toolcall/sunhao4/models/Qwen3.5-9B"
CKPTS_DIR="$ROOT_DIR/ckpts"
RESULTS_DIR="$ROOT_DIR/eval/results"

# 多机环境变量（与训练一致：平台注入 SENSECORE_PYTORCH_* → 映射 RANK/MASTER_ADDR）
source "$ROOT_DIR/scripts/_sensecore_env.sh"
NNODES="${NNODES:-1}"
GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

echo "============================================"
echo "  批量评测 (Pass^$NUM_RUNS, ${NNODES}x${GPUS_PER_NODE}GPU, rank=$RANK)"
echo "============================================"

# 凭证
source "$ROOT_DIR/scripts/env/load_tencent_env.sh" 2>/dev/null || true
source "$ROOT_DIR/scripts/env/load_training_env.sh" 2>/dev/null || true

# ── 日志重定向到 AFS（照搬 _train_impl.sh：当前写 eval.log，历史 archive）──
_EVAL_LOGDIR="$ROOT_DIR/logs/experiments/cl2r_eval"
mkdir -p "$_EVAL_LOGDIR"
if [ "$RANK" = "0" ]; then _rank_sfx=""; else _rank_sfx=".rank$RANK"; fi
_EVAL_LOGFILE="$_EVAL_LOGDIR/eval$_rank_sfx.log"
if [ -f "$_EVAL_LOGFILE" ]; then
  mkdir -p "$_EVAL_LOGDIR/archive"
  mv "$_EVAL_LOGFILE" "$_EVAL_LOGDIR/archive/eval$_rank_sfx-$(date -u +%Y%m%dT%H%M%SZ).log" 2>/dev/null || true
fi
exec > >(tee "$_EVAL_LOGFILE") 2>&1
echo "[eval] === $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) rank=$RANK pid=$$ log=$_EVAL_LOGFILE ==="
echo "[eval] RANK=$RANK MASTER_ADDR=$MASTER_ADDR NNODES=$NNODES WORLD_SIZE=${WORLD_SIZE:-$NNODES}"

_RAY_NUM_CPUS="${CL_RAY_NUM_CPUS-$(nproc 2>/dev/null || echo '')}"
_RAY_NUM_CPUS_ARG=""
[ -n "$_RAY_NUM_CPUS" ] && _RAY_NUM_CPUS_ARG="--num-cpus $_RAY_NUM_CPUS"

# ── 多机 Ray 启动（起一次，常驻；跑完全部模型后 stop）──
if [ "$RANK" = "0" ]; then
  # 清上次残留 shm（同训练，防 lightllm KV cache 残留）
  ipcs -m 2>/dev/null | awk '$6 == 0 {print $2}' | xargs -r ipcrm -m 2>/dev/null || true
  ray start --head --disable-usage-stats ${_RAY_NUM_CPUS_ARG} || { echo "[eval] FATAL: ray start --head 失败" >&2; exit 1; }
else
  echo "[eval] worker: ray start --address $MASTER_ADDR:6379 --block"
  ray start --address "$MASTER_ADDR:6379" ${_RAY_NUM_CPUS_ARG} --block
  exit 0  # worker 常驻到 master stop，之后退出
fi

# ── 评测函数（Ray 已起，不再每模型 start/stop）──
eval_model() {
  local model_path="$1" model_type="$2"
  local out_dir="$RESULTS_DIR/$model_type"
  local per_task="$out_dir/per_task.json"

  if [ -f "$per_task" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[$model_type] 已有结果，跳过（FORCE=1 重评）"
    return
  fi

  echo ""
  echo ">>> 评测: $model_type  (model=$model_path)"

  mkdir -p "$out_dir"

  # merge（FSDP ckpt → HF；base 直接跳过）
  local eval_path="$model_path"
  if [[ "$model_path" != *models* ]] && [ -d "$model_path" ]; then
    # ★ merge 结果必须放 AFS（跨节点 worker 要加载这个 HF 模型，/tmp 不共享）
    local merged="$ROOT_DIR/eval/.tmp/merged_${model_type}"
    mkdir -p "$ROOT_DIR/eval/.tmp"
    echo "    merging FSDP→HF → $merged ..."
    "$PY" -m verl.model_merger merge --backend fsdp --local_dir "$model_path" --target_dir "$merged"
    eval_path="$merged"
  fi

  # 跑 cl_eval（连上已存在的多机 Ray）
  "$PY" -m trainer.cl_eval \
    --config "$ROOT_DIR/configs/run/r0-25k_9b_16gpu.yaml" \
    --tasks-file "$ROOT_DIR/src/eval/claweval_manifest.json" \
    --output "$per_task" \
    --num-runs "$NUM_RUNS" \
    actor_rollout_ref.model.path="$eval_path" \
    cl.buffer.enabled=false \
    trainer.nnodes="$NNODES" trainer.n_gpus_per_node="$GPUS_PER_NODE"

  # 聚合（与 base 对比，除非自己就是 base）
  local baseline_flag=""
  if [ "$model_type" != "base" ] && [ -f "$RESULTS_DIR/base/per_task.json" ]; then
    baseline_flag="--baseline $RESULTS_DIR/base/per_task.json"
  fi
  "$PY" -m eval.aggregate --results "$per_task" $baseline_flag --output "$out_dir/scores.json"

  echo "    ✅ $model_type 完成"
}

# ── 1. 评 base（参照）──
eval_model "$BASE_MODEL" "base"

# ── 2. 评所有 ckpt ──
for exp_dir in "$CKPTS_DIR"/*/; do
  exp_name=$(basename "$exp_dir")
  for step_dir in "$exp_dir"global_step_*/; do
    [ -d "$step_dir" ] || continue
    step=$(basename "$step_dir" | sed 's/global_step_//')
    actor_dir="$step_dir/actor"
    [ -d "$actor_dir" ] || continue
    model_type="${exp_name#cl2r_}_step${step}"
    eval_model "$actor_dir" "$model_type"
  done
done

# ── 3. 收尾：stop Ray + 汇总 ──
ray stop --force >/dev/null 2>&1 || true

echo ""
echo "============================================"
echo "  评测汇总"
echo "============================================"
"$PY" - <<'PYEOF'
import json
from pathlib import Path

results_dir = Path("eval/results")
rows = []
for d in sorted(results_dir.iterdir()):
    scores = d / "scores.json"
    if not scores.exists(): continue
    s = json.loads(scores.read_text())
    summary = s.get("summary", {})
    rows.append({
        "model": d.name,
        "pass@3": summary.get("pass_at_3", "-"),
        "pass^3": summary.get("pass_caret_3", "-"),
        "mean": summary.get("mean_reward", "-"),
        "n": summary.get("n_tasks", "-"),
    })

print(f"\n{'model':<25s} {'pass@3':>8s} {'pass^3':>8s} {'mean':>8s} {'n':>4s}")
print("-" * 58)
for r in rows:
    def f(v): return f"{v:.3f}" if isinstance(v, float) else str(v)
    print(f"{r['model']:<25s} {f(r['pass@3']):>8s} {f(r['pass^3']):>8s} {f(r['mean']):>8s} {r['n']:>4}")

# 按桶对比
print("\n按桶 pass@3 对比：")
model_bucket = {}
for d in sorted(results_dir.iterdir()):
    scores = d / "scores.json"
    if not scores.exists(): continue
    s = json.loads(scores.read_text())
    model_bucket[d.name] = s.get("per_bucket", {})

if len(model_bucket) > 1:
    buckets = sorted({b for pb in model_bucket.values() for b in pb})
    header = f"{'bucket':<15s}" + "".join(f"{m[:12]:>13s}" for m in sorted(model_bucket))
    print(header)
    for b in buckets:
        row = f"{b:<15s}"
        for m in sorted(model_bucket):
            v = model_bucket[m].get(b, {}).get("pass_at_3", "-")
            row += f"{v:>13.3f}" if isinstance(v, float) else f"{'-':>13s}"
        print(row)
PYEOF

echo ""
echo "✅ 全部评测完成。结果在 eval/results/<model_type>/scores.json"

