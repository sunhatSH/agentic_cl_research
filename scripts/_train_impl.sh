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
VERL_DIR="/mnt/afs_toolcall/sunhao4/dependencies/verl"
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
# flash_attn shim(AFS 上,补齐镜像 shim 缺的 flash_attn_interface,转发真 FA3 flash_attn_3)。
# 镜像 shim 只有 bert_padding,导致 transformer_engine.pytorch(recipe_custom→megatron 链
# import)找不到 flash_attn_interface 而崩(2026-07-29 集群定位)。补丁目录含完整 shim
# (bert_padding + flash_attn_interface)。
# 优先级靠后：先探测真 flash_attn(非 shim 且带 flash_attn_interface)——在则【不挂 shim】,
# 让真包(pip 装进 site-packages 的 flash-attn)优先;缺失/残缺才把 shim 前置兜底
# (PYTHONPATH 整体先于 site-packages,故要真包优先只能"不挂 shim",而非调 shim 在 PYTHONPATH 内位置)。
_FA_SHIM="$ROOT_DIR/docker/qwen36-lightllm/flash_attn_shim"
export PYTHONPATH="$LIGHTLLM_DIR:$VERL_DIR:$ROOT_DIR:${PYTHONPATH:-}"
if "$PY" -c "import flash_attn; assert 'flash_attn_shim' not in (getattr(flash_attn,'__file__','') or ''); from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func; from flash_attn.bert_padding import unpad_input" >/dev/null 2>&1; then
  echo "[train_cl] 真 flash_attn 可用(含 interface+bert_padding),不挂 shim"
else
  echo "[train_cl] 真 flash_attn 缺失/残缺,前置 shim 兜底: $_FA_SHIM"
  export PYTHONPATH="$_FA_SHIM:$PYTHONPATH"
fi
export PYTHON="$PY"
export NNODES N_GPUS_PER_NODE="$GPUS_PER_NODE" ROLLOUT_TP_SIZE="$ROLLOUT_TP"
export ULYSSES_SP_SIZE="$ULYSSES_SP" TRAIN_BATCH_SIZE="$TRAIN_BATCH" PPO_MINI_BATCH_SIZE="$PPO_MINI"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export ROLLOUT_GPU_MEM_UTIL="$GPU_MEM_UTIL"
export HF_DATASETS_CACHE="/tmp/hf_datasets_cache" HF_HOME="/tmp/hf_home"
export VLLM_GDN_PREFILL_BACKEND="${VLLM_GDN_PREFILL_BACKEND:-triton}"
# ══════════════════════════════════════════════════════════════════════════════
# recipe_custom 原生 agent_loop 路线 env（迁移自自写 rollout；见 plan swift-juggling-toast /
# debug §30）。所有值 ${VAR:-默认} 形式,可外部覆盖。env 三处来源分工：
#   · 本区块          —— recipe_custom/verl 框架开关(下方,集中在此,不散落)
#   · load_tencent_env —— 沙箱凭证 E2B_*/TENCENT_*(读 docker/sandbox/{tencent,image,runtime}.env)
#   · load_training_env—— 训练凭证 SWANLAB/TOKENHUB(读 .env);judge 端点 REWARD_* 见下方 judge 段
# ══════════════════════════════════════════════════════════════════════════════
# (1) 加载 recipe_custom 注册:lightllm replica / custom_language_model engine / Qwen3.5 GDN
#     monkey_patch(变长packed forward)/ omni reward / agent_loop。这是 Qwen3.5-9B 混合 GDN
#     能用 remove_padding+flash_attn3+长序列(65536) 的前提。
export VERL_USE_EXTERNAL_MODULES="${VERL_USE_EXTERNAL_MODULES:-recipe_custom.bootstrap}"
export MODELING_BACKEND="${MODELING_BACKEND:-hf}"
# (2) agent trace / transfer_queue(照参考脚本 debug_rl_qwen35_9b.sh)
export VERL_AGENT_TRAINABLE_TRACE_TYPES="${VERL_AGENT_TRAINABLE_TRACE_TYPES:-agent,context_compression}"
export VERL_FORCE_TQ_NESTED_READBACK="${VERL_FORCE_TQ_NESTED_READBACK:-1}"
# (3) 缓存 / 日志级别(降噪:lightllm/verl/TQ 的 debug 刷屏)
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
export LIGHTLLM_LOG_LEVEL="${LIGHTLLM_LOG_LEVEL:-info}"
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-WARNING}"
export TQ_LOGGING_LEVEL="${TQ_LOGGING_LEVEL:-WARNING}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-1}"
# (4) e2b 沙箱:不校验 api_key 存在性(腾讯 e2b 兼容端点,E2B_API_KEY/E2B_DOMAIN 由 load_tencent_env
#     从 docker/sandbox/tencent.env export,agent_loop_config.yaml 的 ${oc.env:E2B_*} 取用)
export E2B_VALIDATE_API_KEY="${E2B_VALIDATE_API_KEY:-false}"
# 注:不设 E2B_MAX_KEEPALIVE_CONNECTIONS/E2B_MAX_CONNECTIONS —— 用 e2b SDK 原生默认
# (keepalive=20,复用长连接,短任务省资源/低延迟)。GOAWAY(入口网关单连接~1000 stream
# 后回收)只在长任务触发;本项目采集/训练以短任务为主,不改。长任务需要时再按需调大。
# 注:不要开 PYTORCH_CUDA_ALLOC_CONF=expandable_segments —— 它与 lightllm 的
# torch_memory_saver 互斥(报 "TorchMemorySaver is disabled ... expandable_segments
# not supported"),会导致 lightllm 启动失败、整训练崩(见 debug doc §22)。
# torch_memory_saver(训练时让 lightllm sleep 让出显存)是 util 0.7 不 OOM 的前提,须保留。
# 边界碎片 OOM 改用降 ppo_max_token_len 治理,不动全局分配器。

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
  # judge 端点权威来源 = configs/agents.yaml 的 reward 段(model_reward.py config-first 读它,
  # 用 SUFY_API_KEY);env REWARD_* 仅 fallback。MODELING_BACKEND 已在上方集中区设,此处不重复。
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
# 存放约定：当前一次训练始终写 train.log（rank0）/ train.rank<N>.log（rank>0）；
#   启动时若上一次的同名日志还在，自动移进 archive/ 子目录并带 UTC 时间戳保存，
#   于是 train.log 永远是"最新一次"，历史全部沉到 logs/experiments/<exp>/archive/。
# 多机各 rank 写各自的文件，避免两节点 tee 同一文件互相截断/交错。
_exp=$(_exp_name)
_LOGDIR="$ROOT_DIR/logs/experiments/$_exp"
mkdir -p "$_LOGDIR"
_rank="${RANK:-0}"
if [ "$_rank" = "0" ]; then _rank_sfx=""; else _rank_sfx=".rank$_rank"; fi
_LOGFILE="$_LOGDIR/train$_rank_sfx.log"
# 归档上一次:把已存在的同名日志移到 archive/，文件名带其自身的“归档时刻”UTC 戳。
if [ -f "$_LOGFILE" ]; then
  mkdir -p "$_LOGDIR/archive"
  mv "$_LOGFILE" "$_LOGDIR/archive/train$_rank_sfx-$(date -u +%Y%m%dT%H%M%SZ).log" 2>/dev/null || true
fi
# 一次性把历史遗留的 train.archive-*.log（旧手工归档）也归拢进 archive/。
for _old in "$_LOGDIR"/train.archive-*.log "$_LOGDIR"/train*-20*Z.log; do
  [ -f "$_old" ] || continue
  mkdir -p "$_LOGDIR/archive"; mv "$_old" "$_LOGDIR/archive/" 2>/dev/null || true
done
# tee 不带 -a → 写全新的 train.log（旧的已移走）。stderr 合并到同一流。
exec > >(tee "$_LOGFILE") 2>&1
echo "[train_cl] === $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) rank=${RANK:-0}/${WORLD_SIZE:-${NNODES}} pid=$$ log=$_LOGFILE ==="
echo "[train_cl] RANK=${RANK:-0} MASTER_ADDR=${MASTER_ADDR:-N/A} NNODES=$NNODES WORLD_SIZE=${WORLD_SIZE:-$NNODES}"

_run_single() {
  local ckpt="$ROOT_DIR/ckpts/$_exp"
  mkdir -p "$ckpt" "$_LOGDIR/rollout" "$_LOGDIR/val" "$ROOT_DIR/logs/metrics/$_exp"
  export CKPT_DIR="$ckpt" ROLLOUT_DATA_DIR="$_LOGDIR/rollout" VAL_DATA_DIR="$_LOGDIR/val"
  # verl 原生 FileLogger 落盘路径(logger:[...,file] 时生效)。每 step 实时写 JSONL,
  # 含 reward/advantage/loss 全套聚合 metrics。取代坏掉的 _log_training_metrics。
  export VERL_FILE_LOGGER_PATH="$ROOT_DIR/logs/metrics/$_exp/metrics.jsonl"

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

  # ── 护栏(R3/R2/R5,2026-07-27 审查):--buckets 路径有三个已知坑,先 fail-loud ──
  # R3: per-bucket parquet 曾在 datasets/_archive_multiturn_20260723/,datasets/baseline_9b/
  #     可能是空目录 → create_rl_dataset 打不开首桶就崩在 verl 深处。这里先检查。
  # R2: 桶行数 < train_batch_size 时 verl `assert len(dataloader)>=1` 崩(batch=64,
  #     多数桶行数远小于 64)。无法在 shell 廉价读 parquet 行数,故仅提示。
  # R5: 9 桶共享 $ckpt_base + verl resume_mode=auto 会让第 2 桶起误从上一桶 step 续训
  #     (resume_from_path 是死配置,auto 不读它),遗忘实验语义被毁。按桶训练方案本身
  #     待重构(见 doc/archive/RunLog.md),此处不深修,仅护栏 + 警告。
  if [ ! -d "$data_dir" ] || [ -z "$(ls -A "$data_dir" 2>/dev/null)" ]; then
    echo "[train_cl] FATAL: --buckets 数据目录 $data_dir 不存在或为空。" >&2
    echo "[train_cl]   per-bucket parquet 可能在 datasets/_archive_multiturn_20260723/;" >&2
    echo "[train_cl]   请先把 train_<bucket>.parquet 放进 $data_dir 再跑 --buckets。" >&2
    return 1
  fi
  echo "[train_cl] WARN(R2): 桶行数若 < train_batch_size 会触发 verl 'dataloader empty' 断言;" >&2
  echo "[train_cl] WARN(R5): 9 桶共享 ckpt 目录 + verl resume_mode=auto → 第2桶起可能误续训," >&2
  echo "[train_cl]           跨桶 global_steps 继承会破坏遗忘实验语义。按桶训练方案待重构。" >&2

  local buckets=(office research coding ops safety workflow finance communication qa)
  for b in "${buckets[@]}"; do
    local bfile="$data_dir/train_$b.parquet"
    if [ ! -f "$bfile" ]; then
      echo "[train_cl] FATAL: 桶 $b 的数据文件不存在: $bfile" >&2
      return 1
    fi
    local blogdir="$ROOT_DIR/logs/experiments/$exp/$b"
    mkdir -p "$ckpt_base" "$blogdir/rollout" "$blogdir/val"
    export CKPT_DIR="$ckpt_base" ROLLOUT_DATA_DIR="$blogdir/rollout" VAL_DATA_DIR="$blogdir/val"
    echo "[train_cl] 桶 $b  model=$prev  resume=${_resume:-无}"
    "$PY" -m trainer.cl_main --config "$CONFIG" \
      "actor_rollout_ref.model.path=$prev" "actor_rollout_ref.ref.path=$base_model" \
      "data.train_files=$bfile" \
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
