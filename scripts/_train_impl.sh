#!/usr/bin/env bash
# _train_impl.sh — internal: called by ``scripts/train`` (the unified entry point).
# 所有 GPU/机器/集群拓扑都是命令行参数。禁止直接调用；走 ``scripts/train <TOPOLOGY>``。
#
# 并行约束（verl engine_workers.py:258）：
#   DP = 总卡数 / ULYSSES_SP_SIZE ; train_batch % DP == 0 ; ppo_mini % DP == 0
# 启动前整除自检，不满足直接中止（防止换卡数训到一半崩）。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── 默认值 ──────────────────────────────────────────────────────────────
CONFIG="$ROOT_DIR/configs/run/b1_8b.yaml"
NNODES=1
GPUS_PER_NODE=8
ROLLOUT_TP=""            # 空 → 默认 = GPUS_PER_NODE
ULYSSES_SP=1
TRAIN_BATCH=256
PPO_MINI=32
GPU_MEM_UTIL=0.75
CUDA_DEVICES=""         # 空 → 生成 0,1,..,GPUS_PER_NODE-1
EXP_NAME=""             # 空 → 从 config 读 experiment_name
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

# ── 整除自检（4D 并行约束）─────────────────────────────────────────────
echo "[train_cl] 拓扑: ${NNODES}节点 × ${GPUS_PER_NODE}卡 = ${TOTAL_GPUS} | SP=${ULYSSES_SP} DP=${DP} rollout_TP=${ROLLOUT_TP}"
echo "[train_cl] batch: train=${TRAIN_BATCH} ppo_mini=${PPO_MINI} | config=${CONFIG}"
_fail=0
if [ $((TOTAL_GPUS % ULYSSES_SP)) -ne 0 ]; then
  echo "[train_cl] ERROR: 总卡数 $TOTAL_GPUS 不能被 ULYSSES_SP=$ULYSSES_SP 整除" >&2; _fail=1
fi
if [ "$DP" -gt 0 ] && [ $((TRAIN_BATCH % DP)) -ne 0 ]; then
  echo "[train_cl] ERROR: train_batch=$TRAIN_BATCH 不能被 DP=$DP 整除" >&2; _fail=1
fi
if [ "$DP" -gt 0 ] && [ $((PPO_MINI % DP)) -ne 0 ]; then
  echo "[train_cl] ERROR: ppo_mini=$PPO_MINI 不能被 DP=$DP 整除" >&2; _fail=1
fi
[ "$_fail" -ne 0 ] && { echo "[train_cl] 整除自检失败，中止" >&2; exit 3; }
echo "[train_cl] ✓ 整除自检通过"

# ── 环境（PYTHONPATH / PATH / CUDA libs）───────────────────────────────
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$LIGHTLLM_DIR:$VERL_DIR:$ROOT_DIR:${PYTHONPATH:-}"
export PYTHON="$PY"
# nvidia runtime libs（lightllm rollout 子进程加载 libcudart/cudnn/nccl）
_NV="$VENV/lib/python3.11/site-packages/nvidia"
if [ -d "$_NV" ]; then
  for _d in "$_NV"/*/lib; do [ -d "$_d" ] && LD_LIBRARY_PATH="$_d:${LD_LIBRARY_PATH:-}"; done
  export LD_LIBRARY_PATH
fi

# ── 拓扑 env（cluster.yaml 的 ${oc.env:...} 锚点）──────────────────────
export NNODES N_GPUS_PER_NODE="$GPUS_PER_NODE" ROLLOUT_TP_SIZE="$ROLLOUT_TP"
export ULYSSES_SP_SIZE="$ULYSSES_SP" TRAIN_BATCH_SIZE="$TRAIN_BATCH" PPO_MINI_BATCH_SIZE="$PPO_MINI"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export ROLLOUT_GPU_MEM_UTIL="$GPU_MEM_UTIL"
# AFS 不支持 flock → HF/verl cache 指向本地盘
export HF_DATASETS_CACHE="/tmp/hf_datasets_cache" HF_HOME="/tmp/hf_home"
export VLLM_GDN_PREFILL_BACKEND="${VLLM_GDN_PREFILL_BACKEND:-triton}"

if [ "$DRY_RUN" = "1" ]; then
  echo "[train_cl] --dry-run: 打印 env 后退出（不启动训练）"
  env | grep -E "NNODES|N_GPUS|ROLLOUT_TP|ULYSSES|TRAIN_BATCH|PPO_MINI|CUDA_VISIBLE|GPU_MEM" | sort
  exit 0
fi

# ── 环境自检：① 检查 ② 不符就装/改版本 ③ 训练。只有【安装失败】才中止 ──
# （快速失败，不带缺依赖硬跑白费排队）。多机时每个节点各自跑（各机 /opt/conda 独立）。
PY="$PY" bash "$ROOT_DIR/scripts/check_train_env.sh" || {
  echo "[train_cl] 环境依赖安装失败，中止训练（先修好环境再排队）" >&2; exit 5; }

# ── judge：--smoke 用 mock，否则 agents.yaml sufy judge ─────────────────
if [ "$SMOKE" = "1" ]; then
  export REWARD_API_BASE="${REWARD_API_BASE:-http://127.0.0.1:8100/v1}"
  export REWARD_MODEL="${REWARD_MODEL:-mock-judge}"
  export TOKENHUB_API_KEY="${TOKENHUB_API_KEY:-sk-local}"
  if ! curl -sS -m 3 "$REWARD_API_BASE/models" >/dev/null 2>&1; then
    echo "[train_cl] 起 mock judge on $REWARD_API_BASE"
    "$PY" "$ROOT_DIR/scripts/mock_judge.py" --port 8100 > /tmp/mock_judge.log 2>&1 &
    for _ in $(seq 1 10); do curl -sS -m 2 "$REWARD_API_BASE/models" >/dev/null 2>&1 && break; sleep 1; done
  fi
else
  # 真训练：judge 走 agents.yaml sufy；加载训练凭证（.env）
  [ -f "$ROOT_DIR/scripts/load_training_env.sh" ] && { set -a; source "$ROOT_DIR/scripts/load_training_env.sh"; set +a; }
  # rollout 走沙箱 Hermes（agent_loop_manager, backend=e2b）时需要 E2B_API_KEY/E2B_DOMAIN
  # 等沙箱凭证——它们在 docker/sandbox/tencent.env，由 load_tencent_env.sh 导出。
  [ -f "$ROOT_DIR/scripts/load_tencent_env.sh" ] && { set -a; source "$ROOT_DIR/scripts/load_tencent_env.sh"; set +a; }
  export MODELING_BACKEND="${MODELING_BACKEND:-hf}"
fi

cd "$ROOT_DIR"

# ── 单实验 / 桶序 运行函数 ─────────────────────────────────────────────
_exp_name() {  # 从 config 读 experiment_name，fallback basename
  [ -n "$EXP_NAME" ] && { echo "$EXP_NAME"; return; }
  local n; n=$(grep -E "^[[:space:]]*experiment_name:" "$CONFIG" 2>/dev/null | head -1 | sed -E "s/.*experiment_name:[[:space:]]*//;s/[[:space:]\"']*//g")
  echo "${n:-$(basename "$CONFIG" .yaml)}"
}

_run_single() {  # 单实验：直接跑 or ray head（多机）
  local exp; exp=$(_exp_name)
  local ckpt="$ROOT_DIR/ckpts/$exp"
  local logdir="$ROOT_DIR/logs/experiments/$exp"
  mkdir -p "$ckpt" "$logdir/rollout" "$logdir/val"
  export CKPT_DIR="$ckpt" ROLLOUT_DATA_DIR="$logdir/rollout" VAL_DATA_DIR="$logdir/val"
  export VERL_LOGGER="${VERL_LOGGER:-[console,swanlab]}"
  echo "[train_cl] 启动 $exp → $logdir"
  "$PY" -m trainer.cl_main --config "$CONFIG" "$@" 2>&1 | tee "$logdir/train.log"
}

_run_buckets() {  # 9 桶顺序训练：每桶从上一桶 ckpt 续训
  local base_model; base_model=$(grep -E "^[[:space:]]*path:" "$CONFIG" 2>/dev/null | head -1 | sed -E "s/.*path:[[:space:]]*//;s/[[:space:]\"']*//g")
  local exp; exp=$(_exp_name)
  local ckpt_base="$ROOT_DIR/ckpts/$exp"
  # 桶序 = 训练组间序（评测须同序对齐，防遗忘方案）。
  # total_training_steps/save_freq/test_freq 来自 config（不按桶数据量覆盖）。
  local buckets=(office research coding ops safety workflow finance communication qa)
  local data_dir="$ROOT_DIR/datasets/baseline_9b"
  local prev="$base_model"
  local _resume=""                              # 首桶不 resume，后续 resume 上一桶最后一个 ckpt
  for b in "${buckets[@]}"; do
    local logdir="$ROOT_DIR/logs/experiments/$exp/$b"
    mkdir -p "$ckpt_base" "$logdir/rollout" "$logdir/val"
    export CKPT_DIR="$ckpt_base" ROLLOUT_DATA_DIR="$logdir/rollout" VAL_DATA_DIR="$logdir/val"
  echo "[train_cl] 桶 $b model=$prev resume=${_resume}"
  "$PY" -m trainer.cl_main --config "$CONFIG" \
    "actor_rollout_ref.model.path=$prev" "actor_rollout_ref.ref.path=$base_model" \
    "data.train_files=$data_dir/train_$b.parquet" \
    "trainer.default_local_dir=$ckpt_base" "trainer.resume_from_path=$_resume" \
    2>&1 | tee "$logdir/train.log"
    local latest; latest=$(ls -dt "$ckpt_base"/global_step_* 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
      prev="$latest/actor"
      _resume="$latest"                         # 下一桶从这 resume
    else
      echo "[train_cl] WARN: 桶 $b 无 ckpt，续用 $prev"
      _resume=""                                # 没 ckpt 就不 resume
    fi
  done
  echo "[train_cl] 9 桶全部完成"
}

# ── 单机直跑 / 多机 rendezvous ─────────────────────────────────────────
if [ "$NNODES" -le 1 ]; then
  export RAY_ADDRESS="${RAY_ADDRESS:-}"   # 单机由 cl_main 内 ray.init(address=local)
  if [ "$BUCKETS" = "1" ]; then _run_buckets; else _run_single "$@"; fi
else
  # 多机：SenseCore 注入 RANK/MASTER_ADDR/MASTER_PORT（_sensecore_env 兼容映射）
  source "$SCRIPT_DIR/_sensecore_env.sh"
  OMP_NUM_THREADS=1 "$PY" -c "import os,torch.distributed as d; s,r,w=d.rendezvous(f'tcp://{os.environ[\"MASTER_ADDR\"]}:{os.environ[\"MASTER_PORT\"]}'); d.init_process_group('gloo',store=s,rank=r,world_size=w); d.barrier()"
  if [ "${RANK:-0}" = "0" ]; then
    ray start --head --disable-usage-stats && ray status
    if [ "$BUCKETS" = "1" ]; then _run_buckets; else _run_single "$@"; fi
    ray stop --force
  else
    ray start --address "$MASTER_ADDR:6379" --block
  fi
fi
