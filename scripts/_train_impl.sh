#!/usr/bin/env bash
# _train_impl.sh — internal: called by ``scripts/train <TOPOLOGY>``.
# 禁止直接调用。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── 默认值 ──────────────────────────────────────────────────────────────
CONFIG="$ROOT_DIR/configs/run/b1_8b.yaml"
NNODES=1
GPUS_PER_NODE=8
ROLLOUT_TP=""
ULYSSES_SP=1
TRAIN_BATCH=256
PPO_MINI=32
GPU_MEM_UTIL=0.75
CUDA_DEVICES=""
EXP_NAME=""
VENV="/opt/conda"
VERL_DIR="/mnt/afs_toolcall/sunhao4/workspace/verl"
LIGHTLLM_DIR="/mnt/afs_toolcall/sunhao4/workspace/LightLLM"
SMOKE=0
BUCKETS=0
DRY_RUN=0

# ── 解析参数 ────────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
  case "$1" in
    --config)        CONFIG="$2"; shift 2;;
    --nnodes)        NNODES="$2"; shift 2;;
    --gpus-per-node) GPUS_PER_NODE="$2"; shift 2;;
    --rollout-tp)    ROLLOUT_TP="$2"; shift 2;;
    --ulysses-sp)    ULYSSES_SP="$2"; shift 2;;
    --train-batch)   TRAIN_BATCH="$2"; shift 2;;
    --ppo-mini)      PPO_MINI="$2"; shift 2;;
    --gpu-mem-util)  GPU_MEM_UTIL="$2"; shift 2;;
    --cuda-devices)  CUDA_DEVICES="$2"; shift 2;;
    --exp-name)      EXP_NAME="$2"; shift 2;;
    --venv)          VENV="$2"; shift 2;;
    --verl-dir)      VERL_DIR="$2"; shift 2;;
    --lightllm-dir)  LIGHTLLM_DIR="$2"; shift 2;;
    --smoke)         SMOKE=1; shift;;
    --buckets)       BUCKETS=1; shift;;
    --dry-run)       DRY_RUN=1; shift;;
    *) echo "[train_cl] 未知参数: $1" >&2; exit 2;;
  esac
done

[ -z "$ROLLOUT_TP" ] && ROLLOUT_TP="$GPUS_PER_NODE"
[ -z "$CUDA_DEVICES" ] && CUDA_DEVICES="$(seq -s, 0 $((GPUS_PER_NODE-1)))"
PY="$VENV/bin/python"
TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))
DP=$((TOTAL_GPUS / ULYSSES_SP))

# ── 并行约束检查 ────────────────────────────────────────────────────────
echo "[train_cl] ${NNODES}节点×${GPUS_PER_NODE}卡=${TOTAL_GPUS}GPU  SP=${ULYSSES_SP} DP=${DP}  rollout_TP=${ROLLOUT_TP}"
echo "[train_cl] train_batch=${TRAIN_BATCH}  ppo_mini=${PPO_MINI}  config=${CONFIG}"

_fail=0
[ $((TOTAL_GPUS % ULYSSES_SP)) -ne 0 ] && { echo "[train_cl] ERROR: ${TOTAL_GPUS}GPU % SP=${ULYSSES_SP} != 0" >&2; _fail=1; }
[ "$DP" -gt 0 ] && [ $((TRAIN_BATCH % DP)) -ne 0 ] && { echo "[train_cl] ERROR: train_batch=${TRAIN_BATCH} % DP=${DP} != 0" >&2; _fail=1; }
[ "$DP" -gt 0 ] && [ $((PPO_MINI % DP)) -ne 0 ] && { echo "[train_cl] ERROR: ppo_mini=${PPO_MINI} % DP=${DP} != 0" >&2; _fail=1; }
[ "$_fail" -ne 0 ] && exit 3
echo "[train_cl] ✓ 并行约束检查通过"

# ── 环境变量 ────────────────────────────────────────────────────────────
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$LIGHTLLM_DIR:$VERL_DIR:$ROOT_DIR:${PYTHONPATH:-}"
export PYTHON="$PY"
export NNODES N_GPUS_PER_NODE="$GPUS_PER_NODE" ROLLOUT_TP_SIZE="$ROLLOUT_TP"
export ULYSSES_SP_SIZE="$ULYSSES_SP" TRAIN_BATCH_SIZE="$TRAIN_BATCH" PPO_MINI_BATCH_SIZE="$PPO_MINI"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export ROLLOUT_GPU_MEM_UTIL="$GPU_MEM_UTIL"
export HF_DATASETS_CACHE="/tmp/hf_datasets_cache" HF_HOME="/tmp/hf_home"
export VLLM_GDN_PREFILL_BACKEND="${VLLM_GDN_PREFILL_BACKEND:-triton}"

_NV="$VENV/lib/python3.11/site-packages/nvidia"
if [ -d "$_NV" ]; then
  for _d in "$_NV"/*/lib; do [ -d "$_d" ] && LD_LIBRARY_PATH="$_d:${LD_LIBRARY_PATH:-}"; done
  export LD_LIBRARY_PATH
fi

# ── dry-run ─────────────────────────────────────────────────────────────
if [ "$DRY_RUN" = "1" ]; then
  echo "[train_cl] --dry-run"
  env | grep -E "NNODES|N_GPUS|ROLLOUT_TP|ULYSSES|TRAIN_BATCH|PPO_MINI|CUDA_VISIBLE|GPU_MEM" | sort
  exit 0
fi

# ── 环境依赖自检 ────────────────────────────────────────────────────────
PY="$PY" bash "$ROOT_DIR/scripts/env/check_train_env.sh" || {
  echo "[train_cl] 环境依赖安装失败，中止" >&2; exit 5; }

# ── judge 凭证 ──────────────────────────────────────────────────────────
if [ "$SMOKE" = "1" ]; then
  export REWARD_API_BASE="${REWARD_API_BASE:-http://127.0.0.1:8100/v1}"
  export REWARD_MODEL="${REWARD_MODEL:-mock-judge}"
  export TOKENHUB_API_KEY="${TOKENHUB_API_KEY:-sk-local}"
  curl -sS -m 3 "$REWARD_API_BASE/models" >/dev/null 2>&1 || {
    echo "[train_cl] 起 mock judge → $REWARD_API_BASE"
    "$PY" "$ROOT_DIR/scripts/serve/mock_judge.py" --port 8100 > /tmp/mock_judge.log 2>&1 &
    for _ in $(seq 1 10); do curl -sS -m 2 "$REWARD_API_BASE/models" >/dev/null 2>&1 && break; sleep 1; done
  }
else
  [ -f "$ROOT_DIR/scripts/env/load_training_env.sh" ] && { set -a; source "$ROOT_DIR/scripts/env/load_training_env.sh"; set +a; }
  [ -f "$ROOT_DIR/scripts/env/load_tencent_env.sh" ] && { set -a; source "$ROOT_DIR/scripts/env/load_tencent_env.sh"; set +a; }
  export MODELING_BACKEND="${MODELING_BACKEND:-hf}"
fi

cd "$ROOT_DIR"

# ══════════════════════════════════════════════════════════════════════════
# 多机：先 source SenseCore env 取 RANK/WORLD_SIZE，再做日志重定向
# ══════════════════════════════════════════════════════════════════════════
if [ "$NNODES" -gt 1 ]; then
  source "$SCRIPT_DIR/_sensecore_env.sh"
fi

# ── helper 函数 ──────────────────────────────────────────────────────────
_exp_name() {
  [ -n "$EXP_NAME" ] && { echo "$EXP_NAME"; return; }
  local n
  n=$(grep -E "^[[:space:]]*experiment_name:" "$CONFIG" 2>/dev/null | head -1 | sed -E 's/.*experiment_name:[[:space:]]*//;s/[[:space:]"'"'"']*//g') || true
  echo "${n:-$(basename "$CONFIG" .yaml)}"
}

# ── 日志重定向到 AFS（所有 shell + Python 输出都落盘）─────────────────
# 默认：覆写本 rank 的日志（每次启动从干净日志开始，避免历次叠加到同一文件）。
# 多机各 rank 写各自的文件（rank0=train.log，rank>0=train.rank<N>.log），
#   避免两节点 tee 同一文件互相截断/交错。
# TRAIN_LOG_KEEP=1：不覆写——改写到带时间戳的新文件 train[.rankN]-<UTC>.log，
#   保留上一次日志（不追加，故不会越滚越大）。
_exp=$(_exp_name)
_LOGDIR="$ROOT_DIR/logs/experiments/$_exp"
mkdir -p "$_LOGDIR"
_rank="${RANK:-0}"
if [ "$_rank" = "0" ]; then _rank_sfx=""; else _rank_sfx=".rank$_rank"; fi
if [ "${TRAIN_LOG_KEEP:-0}" = "1" ]; then
  _LOGFILE="$_LOGDIR/train$_rank_sfx-$(date -u +%Y%m%dT%H%M%SZ).log"
else
  _LOGFILE="$_LOGDIR/train$_rank_sfx.log"
fi
# tee 不带 -a → 覆写；换文件模式下文件本就是新的。stderr 合并到同一流。
exec > >(tee "$_LOGFILE") 2>&1
echo "[train_cl] === $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) rank=${RANK:-0}/${WORLD_SIZE:-${NNODES}} pid=$$ log=$_LOGFILE ==="
echo "[train_cl] RANK=${RANK:-0} MASTER_ADDR=${MASTER_ADDR:-N/A} NNODES=$NNODES WORLD_SIZE=${WORLD_SIZE:-$NNODES}"

_run_single() {
  local ckpt="$ROOT_DIR/ckpts/$_exp"
  mkdir -p "$ckpt" "$_LOGDIR/rollout" "$_LOGDIR/val"
  export CKPT_DIR="$ckpt" ROLLOUT_DATA_DIR="$_LOGDIR/rollout" VAL_DATA_DIR="$_LOGDIR/val"

  # auto-resume：检测最新 checkpoint，中断后续训
  local latest latest_step
  latest=$(ls -dt "$ckpt"/global_step_* 2>/dev/null | head -1) || true
  if [ -n "$latest" ]; then
    latest_step=$(basename "$latest" | grep -oP '\d+')
    echo "[train_cl] 检测到 checkpoint step=$latest_step → 自动续训"
    set -- "--resume-from" "$latest" "$@"
  fi

  echo "[train_cl] 启动 $_exp → $_LOGDIR"
  "$PY" -m trainer.cl_main --config "$CONFIG" "$@"
}

_run_buckets() {
  local base_model exp ckpt_base data_dir prev _resume latest
  base_model=$(grep -E "^[[:space:]]*path:" "$CONFIG" 2>/dev/null | head -1 | sed -E 's/.*path:[[:space:]]*//;s/[[:space:]"'"'"']*//g') || true
  exp=$(_exp_name)
  ckpt_base="$ROOT_DIR/ckpts/$exp"
  data_dir="$ROOT_DIR/datasets/baseline_9b"
  prev="$base_model"
  _resume=""

  local buckets=(office research coding ops safety workflow finance communication qa)
  for b in "${buckets[@]}"; do
    local blogdir="$ROOT_DIR/logs/experiments/$exp/$b"
    mkdir -p "$ckpt_base" "$blogdir/rollout" "$blogdir/val"
    export CKPT_DIR="$ckpt_base" ROLLOUT_DATA_DIR="$blogdir/rollout" VAL_DATA_DIR="$blogdir/val"
    echo "[train_cl] 桶 $b  model=$prev  resume=${_resume:-无}"
    "$PY" -m trainer.cl_main --config "$CONFIG" \
      "actor_rollout_ref.model.path=$prev" "actor_rollout_ref.ref.path=$base_model" \
      "data.train_files=$data_dir/train_$b.parquet" \
      "trainer.default_local_dir=$ckpt_base" "trainer.resume_from_path=$_resume" \
      2>&1 | tee "$blogdir/train.log"
    latest=$(ls -dt "$ckpt_base"/global_step_* 2>/dev/null | head -1) || true
    if [ -n "$latest" ]; then
      prev="$latest/actor"
      _resume="$latest"
    else
      echo "[train_cl] WARN: 桶 $b 无 ckpt，续用 $prev"
      _resume=""
    fi
  done
  echo "[train_cl] 9 桶全部完成"
}

# ══════════════════════════════════════════════════════════════════════════
# 启动
# ══════════════════════════════════════════════════════════════════════════

if [ "$NNODES" -le 1 ]; then
  if [ "$BUCKETS" = "1" ]; then _run_buckets; else _run_single "$@"; fi
  exit 0
fi

# ── 多机 ────────────────────────────────────────────────────────────────
echo "[train_cl] === 多机模式 rank=${RANK:-0}/${WORLD_SIZE:-?} ==="

# 1. Master 先启动 Ray head
if [ "${RANK:-0}" = "0" ]; then
  echo "[train_cl] master: ray start --head ..."
  ray start --head --disable-usage-stats || { echo "[train_cl] FATAL: ray start --head 失败" >&2; exit 1; }
  ray status
  echo "[train_cl] master: Ray head 就绪 ($(ray status 2>/dev/null | head -3 | tr '\n' ' '))"
fi

# 2. 全节点 barrier 同步
echo "[train_cl] rank=${RANK:-0}: 等待 ${WORLD_SIZE:-?} 节点同步..."
"$PY" << 'PYEOF' || { echo "[train_cl] FATAL: rank=${RANK:-0} 同步失败" >&2; exit 1; }
import os, torch.distributed as dist
addr = os.environ['MASTER_ADDR']
port = int(os.environ['MASTER_PORT'])
rank = int(os.environ['RANK'])
ws   = int(os.environ['WORLD_SIZE'])
print(f'[sync] rank={rank} init tcp://{addr}:{port}', flush=True)
dist.init_process_group('gloo', init_method=f'tcp://{addr}:{port}', rank=rank, world_size=ws)
dist.barrier()
dist.destroy_process_group()
print(f'[sync] rank={rank}: {ws} 节点同步完成', flush=True)
PYEOF
echo "[train_cl] rank=${RANK:-0}: 同步完成"

# 3. 分发：master 训，worker 连 Ray
if [ "${RANK:-0}" = "0" ]; then
  echo "[train_cl] master: 启动训练"
  # 捕获训练退出码。绝不无条件打「训练结束」——否则 cl_main 崩溃(如 rollout
  # IndexError)退出后脚本照样打「成功」+ret 0,平台误判为成功、0 checkpoint 却
  # 显示完成(2026-07-27 10:05 的假成功)。ray stop 始终执行清理,但脚本 exit code
  # 必须等于训练 exit code,让平台看到真实成败。
  _train_rc=0
  if [ "$BUCKETS" = "1" ]; then _run_buckets || _train_rc=$?; else _run_single "$@" || _train_rc=$?; fi
  if [ "$_train_rc" -eq 0 ]; then
    echo "[train_cl] master: 训练正常结束 (rc=0)，ray stop"
  else
    echo "[train_cl] master: !!! 训练失败 rc=$_train_rc（非正常结束，见上方 Traceback）ray stop 清理后以该码退出" >&2
  fi
  ray stop --force
  exit "$_train_rc"
else
  echo "[train_cl] worker: ray start --address $MASTER_ADDR:6379 --block"
  ray start --address "$MASTER_ADDR:6379" --block
fi
