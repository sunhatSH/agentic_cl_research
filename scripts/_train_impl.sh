#!/usr/bin/env bash
# _train_impl.sh — internal: called by ``scripts/train <TOPOLOGY>``.
# 禁止直接调用。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── 默认值 ──────────────────────────────────────────────────────────────
CONFIG="$ROOT_DIR/configs/run/b1_9b_16gpu.yaml"
NNODES=1
GPUS_PER_NODE=8
ROLLOUT_TP=""
ULYSSES_SP=1
TRAIN_BATCH=256
PPO_MINI=32
GPU_MEM_UTIL=0.75
CUDA_DEVICES=""
EXP_NAME="${EXP_NAME:-}"
VENV="/opt/conda"
VERL_DIR="/mnt/afs_toolcall/sunhao4/dependencies/verl"
LIGHTLLM_DIR="/mnt/afs_toolcall/sunhao4/workspace/LightLLM"
SMOKE=0
DRY_RUN=0
OVERRIDES=()   # 透传给 cl_main 的 hydra override(如 trainer.total_training_steps=1)

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
    --dry-run)       DRY_RUN=1; shift;;
    *) OVERRIDES+=("$1"); shift;;   # 未知裸参数(如 trainer.total_training_steps=1)当 hydra override,透传给 cl_main
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

# ── 内存分配器：LD_PRELOAD jemalloc（防长跑 GatewayActor OOM）─────────────────
# 现象(2026-07-31 qwen35_9b_b1_4gpu)：跑到 step 53 节点 512GB 打满,OOM dump 显示
# 2 个 ray::GatewayActor 各涨到 ~185/178GB(合计 363GB),而 trainer/WorkerDict/lightllm
# 全程稳定。根因不是 session 泄漏(全程仅 1 次 timeout,finalize 正常 pop),而是
# **glibc malloc arena 碎片/保留**：gateway 反复分配/释放海量大 Python list
# (prompt_ids≤131072 / response_ids≤65536 / logprobs / decode 文本),freed chunk 留在
# per-arena free list 不还 OS;本机 128 核 → glibc 默认最多 8×128 arena,碎片被放大。
# trainer 用 PyTorch caching allocator(固定复用)故不涨。verl 官方 best-practice 文档
# (docs/ascend_tutorial/.../dapo|gspo)亦明确:长跑需 LD_PRELOAD jemalloc 否则 Ray 进程
# 内存不回收会 OOM。本机 libjemalloc.so.2 已装但未启用。
# CL_ENABLE_JEMALLOC=0 可关(回退到 glibc arena 收敛参数)。
_JEMALLOC_SO="${CL_JEMALLOC_SO:-}"
if [ -z "$_JEMALLOC_SO" ]; then
  for _cand in \
    /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
    /lib/x86_64-linux-gnu/libjemalloc.so.2 \
    /usr/local/lib/libjemalloc.so.2 \
    /usr/lib64/libjemalloc.so.2; do
    [ -e "$_cand" ] && { _JEMALLOC_SO="$_cand"; break; }
  done
fi
if [ "${CL_ENABLE_JEMALLOC:-1}" = "1" ] && [ -n "$_JEMALLOC_SO" ] && [ -e "$_JEMALLOC_SO" ]; then
  # LD_PRELOAD 前置 jemalloc(保留已有 LD_PRELOAD,如 flash-attn/nvidia 库不冲突)
  export LD_PRELOAD="$_JEMALLOC_SO${LD_PRELOAD:+:$LD_PRELOAD}"
  # background_thread + dirty/muzzy decay:后台线程周期性把空闲页 madvise 还给 OS
  export MALLOC_CONF="${MALLOC_CONF:-background_thread:true,dirty_decay_ms:10000,muzzy_decay_ms:10000,narenas:4}"
  echo "[train_cl] ✓ jemalloc 已启用(LD_PRELOAD=$_JEMALLOC_SO, MALLOC_CONF=$MALLOC_CONF)"
else
  # 降级:无 jemalloc 时收敛 glibc arena —— 限 arena 数 + 主动 trim,减少碎片保留
  export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
  export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-134217728}"
  echo "[train_cl] ⚠ 未启用 jemalloc(CL_ENABLE_JEMALLOC=${CL_ENABLE_JEMALLOC:-1}, so='${_JEMALLOC_SO:-未找到}')" \
       "→ 降级 glibc: MALLOC_ARENA_MAX=$MALLOC_ARENA_MAX MALLOC_TRIM_THRESHOLD_=$MALLOC_TRIM_THRESHOLD_"
fi

export NNODES N_GPUS_PER_NODE="$GPUS_PER_NODE" ROLLOUT_TP_SIZE="$ROLLOUT_TP"
export ULYSSES_SP_SIZE="$ULYSSES_SP" TRAIN_BATCH_SIZE="$TRAIN_BATCH" PPO_MINI_BATCH_SIZE="$PPO_MINI"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export ROLLOUT_GPU_MEM_UTIL="$GPU_MEM_UTIL"
export HF_DATASETS_CACHE="/tmp/hf_datasets_cache" HF_HOME="/tmp/hf_home"
export VLLM_GDN_PREFILL_BACKEND="${VLLM_GDN_PREFILL_BACKEND:-triton}"
# ── NCCL cuMem 关闭（防 16卡 lightllm 起服 hang）──────────────────────────────
# lightllm 开 enable_torch_memory_saver(cuMem VMM 劫持 cudaMalloc) × NCCL 默认
# NCCL_CUMEM_ENABLE=1 冲突 → 部分副本 uvicorn 起不来 → verl 无超时 gather 死等 →
# 16卡 rollout 从没开始就 hang(b1_16gpu 6 次复发;§45)。verl 已给 vllm/sglang 设 =0
# (sgl #6723),lightllm 漏了。关 CUMEM 不关 P2P,TP 带宽保留、保持 2 机。verl_runner.py
# 也会透传进 Ray worker(worker 不继承本 shell env),这里 export 是双保险 + 单机路径。
export NCCL_CUMEM_ENABLE="${CL_NCCL_CUMEM:-0}"
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
# ★ 多模态策略:TEXT_MODEL_ONLY 控制视觉加载和更新(需经 verl_runner passthrough 透传到 LightLLM):
#   0=全多模态(视觉开+不冻结) 1=冻结视觉(视觉开+冻结参数,默认,同事方案) 2=纯文本(视觉关+标准RoPE)
export TEXT_MODEL_ONLY="${TEXT_MODEL_ONLY:-1}"
export VERL_USE_EXTERNAL_MODULES="${VERL_USE_EXTERNAL_MODULES:-recipe_custom.bootstrap}"
# 追加项目侧外部 patch 模块（VERL_USE_EXTERNAL_MODULES 逗号分隔，verl/__init__ 在【每个】
# verl 进程——含 AgentSessionWorker，即真正跑 AsyncSandbox.create / create_hooks 的进程——
# import verl 时消费；经 verl_runner.py 的 runtime_env passthrough 传进 Ray worker）：
#   · rollout.e2b_http1_patch      —— 关 e2b SDK 默认 http2 走 HTTP/1.1，从根上消除腾讯
#       AGS 网关的 GOAWAY（单 HTTP/2 连接 ~1000 stream 后回收）。CL_E2B_DISABLE_HTTP2=0 可关。
#   · trainer.observer_hook_register —— monkey-patch recipe_custom hook factory 认 FQN hook
#       名（如 trainer.observer_hook.ObserverDiffHook），使自定义 hook 无需改 verl 源码即可挂载。
#   · trainer.pause_generation_bounded_patch —— lightllm pause_generation 的无限 while True 改成
#       有界等待超时放行（CL_PAUSE_MAX_WAIT 默认 180s）。lightllm refcount 泄漏(4卡16卡都有)让
#       abort_all 僵尸请求回收不掉，16卡每 step 边界 pause 时无限重试→死锁 hang（§59，step6 卡死）。
#       放行让 16卡泄漏像 4卡一样良性(泄漏但不死)。不改 LightLLM 源码。
#   · trainer.dataproto_tensordict_patch —— v1 引擎 worker 的 DataProto→TensorDict 系统性修复。
#       r0 掺回放行后 infer_batch/train_batch/train_mini_batch(engine_workers.py) 的 data 入参
#       实际收到 DataProto(非函数标注的 TensorDict),函数体用 TensorDict API(data.keys()/data.shape[0]/
#       tu.pop/tu.assign_non_tensor/tu.make_iterator) 每 step 崩(b1 无 replay 是 TensorDict 不崩)。
#       在 tqbridge 层统一把 DataProto→to_tensordict(),一处覆盖全部 @register 分发函数(不再逐点
#       打地鼠);并保留 tu.pop 的 DataProto 兼容作兜底。
# 都必须在 worker 进程生效（patch 目标都在 worker/lightllm 副本），故走 VERL_USE_EXTERNAL_MODULES
# 而非 driver-only import。逐个幂等去重。
for _mod in rollout.e2b_http1_patch trainer.observer_hook_register trainer.pause_generation_bounded_patch trainer.dataproto_tensordict_patch; do
  case ",$VERL_USE_EXTERNAL_MODULES," in
    *,"$_mod",*) : ;;  # 已含,不重复追加
    *) export VERL_USE_EXTERNAL_MODULES="$VERL_USE_EXTERNAL_MODULES,$_mod" ;;
  esac
done
export MODELING_BACKEND="${MODELING_BACKEND:-hf}"
# (2) agent trace / transfer_queue(照参考脚本 debug_rl_qwen35_9b.sh)
export VERL_AGENT_TRAINABLE_TRACE_TYPES="${VERL_AGENT_TRAINABLE_TRACE_TYPES:-agent,context_compression}"
export VERL_FORCE_TQ_NESTED_READBACK="${VERL_FORCE_TQ_NESTED_READBACK:-1}"
# (3) 缓存 / 日志级别(降噪:lightllm/verl/TQ 的 debug 刷屏)
#     CL_DIAG=1 → 调到最详细 + 关 Ray 去重 + NCCL INFO,用于排查起服 hang(§48/§49)。
#     诊断完置 0 恢复降噪。默认按 CL_DIAG 决定。
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
if [ "${CL_DIAG:-0}" = "1" ]; then
  export LIGHTLLM_LOG_LEVEL="${LIGHTLLM_LOG_LEVEL:-debug}"
  export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-DEBUG}"
  export TQ_LOGGING_LEVEL="${TQ_LOGGING_LEVEL:-INFO}"
  export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"          # 关去重:每副本独立打,看清哪个 rank/副本卡
  export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"               # NCCL 集合选路/挂起点(warmup all-reduce)
  export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,COLL,P2P}"
  echo "[train_cl] ★ CL_DIAG=1 诊断模式:LIGHTLLM=debug VERL=DEBUG RAY_DEDUP=0 NCCL_DEBUG=INFO"
else
  export LIGHTLLM_LOG_LEVEL="${LIGHTLLM_LOG_LEVEL:-info}"
  export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-WARNING}"
  export TQ_LOGGING_LEVEL="${TQ_LOGGING_LEVEL:-WARNING}"
  export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-1}"
fi
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
  # 用 TOKENHUB_API_KEY);env REWARD_* 仅 fallback。MODELING_BACKEND 已在上方集中区设,此处不重复。
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

# ── 模型预热到 node-local（防 16卡 lightllm 起服 hang）────────────────────────
# 8 副本并发从 AFS 网盘(/mnt/afs_toolcall,quarkfs fuse)各加载 2 次 tokenizer +
# AutoProcessor(qwen3_5 多模态分支,disable_vision 管不到)→ I/O 争抢拖到分钟级 →
# lightllm startup_event 的同步 set_args 冻结 uvloop → 副本到不了 594 → verl 无超时
# gather 死等 → 整 job hang(§47/§45,4卡副本少扛得住、16卡崩)。预热=启动前把模型
# 目录拷到【本节点本地盘】,path 指过去,消除网盘并发争抢。每个 rank 各拷一份(两节点
# 都要)。CL_PREWARM_MODEL=0 可关;CL_MODEL_LOCAL_ROOT 覆盖本地根(默认 /dev/shm 不够则 /tmp)。
# 输出:设 CL_MODEL_LOCAL=<本地路径>,训练命令用它 override actor_rollout_ref.model.path。
_prewarm_model() {
  CL_MODEL_LOCAL=""
  [ "${CL_PREWARM_MODEL:-1}" = "1" ] || { echo "[train_cl] 预热关闭(CL_PREWARM_MODEL=0),用 AFS 原路径"; return 0; }
  local src; src=$(grep -E "^[[:space:]]*path:" "$CONFIG" 2>/dev/null | head -1 | sed -E 's/.*path:[[:space:]]*//;s/[[:space:]"'"'"']*//g')
  [ -n "$src" ] && [ -d "$src" ] || { echo "[train_cl] WARN: 模型源路径无效($src),跳过预热"; return 0; }
  # 本地根:优先 /dev/shm(tmpfs,最快),空间不足退 /tmp。用模型 basename 作子目录。
  local root="${CL_MODEL_LOCAL_ROOT:-/dev/shm/cl_models}"
  local need_kb; need_kb=$(du -sk "$src" 2>/dev/null | awk '{print $1}')
  local shm_free_kb; shm_free_kb=$(df -k /dev/shm 2>/dev/null | awk 'NR==2{print $4}')
  if [ "${CL_MODEL_LOCAL_ROOT:-}" = "" ] && [ -n "$need_kb" ] && [ -n "$shm_free_kb" ] && [ "$shm_free_kb" -lt "$((need_kb + need_kb/5))" ]; then
    root="/tmp/cl_models"; echo "[train_cl] /dev/shm 空间不足($((shm_free_kb/1024/1024))G < 模型 $((need_kb/1024/1024))G),预热改用 /tmp"
  fi
  local dst="$root/$(basename "$src")"
  mkdir -p "$root"
  # 已存在且大小一致则复用(同节点多次启动免重拷)。
  if [ -d "$dst" ] && [ "$(du -sk "$dst" 2>/dev/null|awk '{print $1}')" = "$need_kb" ]; then
    echo "[train_cl] 预热命中缓存: $dst"; CL_MODEL_LOCAL="$dst"; return 0
  fi
  echo "[train_cl] 预热模型到 node-local: $src → $dst ($((need_kb/1024/1024))G) ..."
  local _t0; _t0=$(date +%s)
  if cp -a "$src/." "$dst/" 2>/dev/null; then
    echo "[train_cl] ✓ 预热完成,耗时 $(( $(date +%s) - _t0 ))s → $dst"
    CL_MODEL_LOCAL="$dst"
  else
    echo "[train_cl] WARN: 预热拷贝失败,回退 AFS 原路径 $src" >&2; CL_MODEL_LOCAL=""
  fi
}

# 把上一 run 的 metrics.jsonl(verl 下次启动会 open("wb") 覆盖它)折叠进永久累积文件
# metrics.all.jsonl。按 step 去重(同 step 后写为准);resume 续训 step 不与旧重叠,故正常即纯
# append,去重只防同一 run 内异常重复。metrics.all.jsonl 永不被覆盖 = 完整历史单一来源。
# 仅在 resume(有 ckpt)时调用。依赖显式 PY(项目铁律,不乱用其他 python),纯标准库无第三方依赖。
_fold_metrics() {
  local mdir="$1"
  local cur="$mdir/metrics.jsonl" all="$mdir/metrics.all.jsonl"
  [ -s "$cur" ] || return 0   # 上一 run 无 metrics(未产出)→ 无需折叠
  "$PY" - "$cur" "$all" <<'PYEOF'
import json, os, sys
cur, allp = sys.argv[1], sys.argv[2]
by_step = {}
for p in (allp, cur):          # 先读已有累积,再用当前 run 覆盖同 step
    if not os.path.exists(p):
        continue
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            step = row.get("step")
            if step is not None:
                by_step[step] = row
tmp = allp + ".tmp"
with open(tmp, "w") as f:
    for step in sorted(by_step):
        f.write(json.dumps(by_step[step], ensure_ascii=False) + "\n")
os.replace(tmp, allp)
print(f"[train_cl] metrics 折叠: 累积 {len(by_step)} step → {os.path.basename(allp)}")
PYEOF
}

_run_single() {
  local ckpt="$ROOT_DIR/ckpts/$_exp"
  local _mdir="$ROOT_DIR/logs/metrics/$_exp"
  # 清理上一轮的 rollout 记录(_persist_winners/_persist_rollout_status 写 rollouts/training/<exp>/),
  # 每次实验启动清掉,防磁盘膨胀。metrics/ckpt 不在此列(有各自的历史折叠/轮转)。
  rm -rf "$ROOT_DIR/rollouts/training/$_exp"
  mkdir -p "$ckpt" "$_LOGDIR/rollout" "$_LOGDIR/val" "$_mdir"
  export CKPT_DIR="$ckpt" ROLLOUT_DATA_DIR="$_LOGDIR/rollout" VAL_DATA_DIR="$_LOGDIR/val"
  export VERL_FILE_LOGGER_PATH="$_mdir/metrics.jsonl"

  # auto-resume:有 ckpt 才续训。默认行为=检测到最新 ckpt 就自动 --resume-from(无需显式传参),
  #   verl resume_path 会 load_checkpoint + dataloader.load_state_dict(data.pt 存的数据游标,
  #   trainer_base.py:769) → 从上次数据位置续采,不重复用已训过的数据。
  #   ★ 无 ckpt = 全新训练:不 resume、metrics 从头重开(不接旧的)。
  local latest latest_step
  latest=$(ls -dt "$ckpt"/global_step_* 2>/dev/null | head -1) || true

  # verl 原生 FileLogger(logger:[...,file] 时生效)每 step 实时写 JSONL(reward/advantage/loss)。
  # ⚠️ verl FileLogger 硬编码 open(path,"wb") → 每次启动覆盖 metrics.jsonl(tracking.py:420),
  #   不改 verl 源码(项目铁律)。故:
  #   · resume(有 ckpt):启动前把上一 run 的 metrics.jsonl 折叠进永久累积 metrics.all.jsonl
  #     (按 step 去重;续训 step 不与旧重叠,合并无损)。all 永不覆盖、跨 run 连续 = 完整历史。
  #   · 全新(无 ckpt):启动前**同样**把上一 run 的 metrics.jsonl 折叠进 metrics.all.jsonl
  #     (不再直接清掉;防止 2026-08-05 的 b1_16gpu step-38 数据丢失事故重现),然后 rm 当前
  #     metrics.jsonl 让 verl 重写。all 只增不减,跨多次全新训练累积全部 step。
  if [ -n "$latest" ]; then
    latest_step=$(basename "$latest" | grep -oP '\d+')
    echo "[train_cl] 检测到 checkpoint step=$latest_step → 自动续训(数据游标随 data.pt 续,不重复)"
    _fold_metrics "$_mdir"
    set -- "--resume-from" "$latest" "$@"
  else
    echo "[train_cl] 无 checkpoint → 全新训练(不 resume;旧 metrics 先折叠进 .all 再重开)"
    _fold_metrics "$_mdir"
    rm -f "$_mdir/metrics.jsonl"
  fi

  echo "[train_cl] 启动 $_exp → $_LOGDIR"
  # 预热命中则把 model.path override 到 node-local(消除 AFS 并发加载 hang,§47)。
  local _model_ovr=()
  [ -n "${CL_MODEL_LOCAL:-}" ] && _model_ovr=("actor_rollout_ref.model.path=$CL_MODEL_LOCAL")
  # TEXT_MODEL_ONLY 控制视觉加载/更新,覆写 engine_kwargs
  case "${TEXT_MODEL_ONLY:-1}" in
    0|1)
      _model_ovr+=("actor_rollout_ref.rollout.engine_kwargs.lightllm.enable_multimodal=true")
      _model_ovr+=("actor_rollout_ref.rollout.engine_kwargs.lightllm.disable_vision=false")
      # ★ 音频必须关:Qwen3.5-9B config 无 audio_config,开音频 server 起 audioserver
      #   init_model 时 `model_cfg["audio_config"]` KeyError 崩(实测 2026-08-04 vision_smoke)。
      #   我们只需视觉(工具产图),不需音频。enable_multimodal=true + disable_audio=true 即可。
      _model_ovr+=("actor_rollout_ref.rollout.engine_kwargs.lightllm.disable_audio=true")
      [ "${TEXT_MODEL_ONLY}" = "1" ] && \
        _model_ovr+=("actor_rollout_ref.model.override_config.freeze_module_pattern=model\\.visual\\.")
      ;;
    2) ;; # 保持 config 基线(enable_multimodal=false)
  esac
  "$PY" -m trainer.cl_main --config "$CONFIG" "${_model_ovr[@]}" "$@"
}

# ══════════════════════════════════════════════════════════════════════════
# 启动
# ══════════════════════════════════════════════════════════════════════════

if [ "$NNODES" -le 1 ]; then
  _prewarm_model   # 单机:预热到本地盘(见 _prewarm_model 注释)
  # 单机也需自起 Ray head——verl run_ppo 写死 ray.init(address='auto'),要求已有集群;
  # 且必须复用与多机同样的 --num-cpus 名额(§55:CFS quota 低估 → server actor 抢不到
  # 名额静默 PENDING → init_hybrid 永等 hang)。复现 16 卡崩溃时环境须一致。
  _RAY_NUM_CPUS="${CL_RAY_NUM_CPUS-$(nproc 2>/dev/null || echo '')}"
  _RAY_NUM_CPUS_ARG=""
  [ -n "$_RAY_NUM_CPUS" ] && _RAY_NUM_CPUS_ARG="--num-cpus $_RAY_NUM_CPUS"
  echo "[train_cl] 单机: Ray num_cpus 名额 = ${_RAY_NUM_CPUS:-(Ray默认探测)}"
  echo "[train_cl] 单机: ray start --head ... ${_RAY_NUM_CPUS_ARG}"
  ray start --head --disable-usage-stats ${_RAY_NUM_CPUS_ARG} || { echo "[train_cl] FATAL: ray start --head 失败" >&2; exit 1; }
  ray status
  # 捕获训练 exit code:ray stop 始终清理,但脚本退出码必须=训练码(同多机路径,防假成功)。
  _train_rc=0
  _run_single ${OVERRIDES[@]+"${OVERRIDES[@]}"} || _train_rc=$?
  if [ "$_train_rc" -eq 0 ]; then
    echo "[train_cl] 单机: 训练正常结束 (rc=0)，ray stop"
  else
    echo "[train_cl] 单机: !!! 训练失败 rc=$_train_rc（见上方 Traceback）ray stop 清理后以该码退出" >&2
  fi
  ray stop --force
  exit "$_train_rc"
fi

# ── 多机 ────────────────────────────────────────────────────────────────
echo "[train_cl] === 多机模式 rank=${RANK:-0}/${WORLD_SIZE:-?} ==="

# 0. 每个节点各自预热模型到本地盘(两节点都要,lightllm 副本在各节点加载)。
#    在 ray start / barrier 之前做,拷贝耗时不占用集群同步窗口。
_prewarm_model

# 0.5 Ray num_cpus 名额（§55 修 16卡起服 hang）：Ray 默认按 cgroup CFS quota(cpu.max)
#     估 num_cpus，容器里常被压到远小于真实可用逻辑核（实测 quota=16 而 nproc=128）。
#     CFS quota 只限"平均算力"(每 period 最多用 quota 核·时)、**不阻止**起多线程/actor
#     (它们分时跑)；但 Ray 按"名额"记账调度：训练 worker 的 placement group 每卡预留
#     1 CPU + 每个 lightllm server actor(num_cpus=1,纯 IO 壳,真算力在共卡 GPU worker)
#     也要 1 名额。quota 太小时,最后 1 个 server actor(replica_rank=7)抢不到名额 → Ray
#     静默 PENDING → 8 副本缺 1 → verl init_hybrid 的 asyncio.gather 永等 → 整 job hang
#     (§55 定案:8 副本只起 7,driver 同机那台差 1 个 free CPU 名额)。
#     用 nproc(affinity 视角"这台机允许用的逻辑核")作 num_cpus 给足名额;server/train 都
#     不吃满 CPU,over-provision 名额安全。CL_RAY_NUM_CPUS 可覆盖(设空串=退回 Ray 默认探测)。
_RAY_NUM_CPUS="${CL_RAY_NUM_CPUS-$(nproc 2>/dev/null || echo '')}"
_RAY_NUM_CPUS_ARG=""
[ -n "$_RAY_NUM_CPUS" ] && _RAY_NUM_CPUS_ARG="--num-cpus $_RAY_NUM_CPUS"
echo "[train_cl] rank=${RANK:-0}: Ray num_cpus 名额 = ${_RAY_NUM_CPUS:-(Ray默认探测)}"

# 1. Master 先启动 Ray head
if [ "${RANK:-0}" = "0" ]; then
  echo "[train_cl] master: ray start --head ... ${_RAY_NUM_CPUS_ARG}"
  ray start --head --disable-usage-stats ${_RAY_NUM_CPUS_ARG} || { echo "[train_cl] FATAL: ray start --head 失败" >&2; exit 1; }
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
  _run_single ${OVERRIDES[@]+"${OVERRIDES[@]}"} || _train_rc=$?
  if [ "$_train_rc" -eq 0 ]; then
    echo "[train_cl] master: 训练正常结束 (rc=0)，ray stop"
  else
    echo "[train_cl] master: !!! 训练失败 rc=$_train_rc（非正常结束，见上方 Traceback）ray stop 清理后以该码退出" >&2
  fi
  ray stop --force
  exit "$_train_rc"
else
  echo "[train_cl] worker: ray start --address $MASTER_ADDR:6379 --block ${_RAY_NUM_CPUS_ARG}"
  ray start --address "$MASTER_ADDR:6379" ${_RAY_NUM_CPUS_ARG} --block
fi
