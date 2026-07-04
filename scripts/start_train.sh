#!/usr/bin/env bash
# 集群训练启动脚本（默认 64 卡 = 8 节点 x 8 卡 = 8 个 rank）。
#
# 仿照集群参考脚本的 ray 多机 + torch distributed rendezvous 骨架，已按本项目改写：
#   - 入口改用本项目 trainer.cl_main（保留 CL Loss + Replay Buffer 注入）。
#     参考脚本直调 verl.trainer.main_ppo_sync 会绕过项目的 CL 注入，故不沿用。
#   - 引擎 / 规模 / rollout 等原本写在命令行的 override，改放进 config：
#     configs/cluster.yaml（共用引擎层）+ configs/run/<exp>.yaml（各实验语义）。
#   - 路径、用户、64 卡、不卸载 CPU、swanlab key 均已替换为本项目 / 本集群。
#
# 本次配置：GRPO + 你的 CL loss（cl_main 注入）、B1 语义（无 KL/无 replay）、
# lightllm rollout + agentic 多轮 e2b 沙箱、reward=项目 model judge、日志 console+swanlab。
# 跑别的实验：传 config，如 bash start_train.sh configs/run/r4.yaml。
# 一键串跑多个 phase 请用 scripts/run_phases.sh。
set -uo pipefail

# 全部走 AFS 共享挂载（集群各节点都能访问）。verl/LightLLM 已拷至自己 AFS 目录
# （含 lightllm_rollout + recipe_custom），与泽寰目录解耦。
PROJECT_DIR=/mnt/afs_toolcall/sunhao4/agentic_cl_research
LIGHTLLM_DIR=/mnt/afs_toolcall/sunhao4/Documents/LightLLM
VERL_DIR=/mnt/afs_toolcall/sunhao4/Documents/verl
# 默认先跑 b1（无 replay，最简单，仍走你的 CL loss）验证 mock 链路；
# 链路 OK 后换 configs/run/r4.yaml 开 7 桶 replay：bash start_train.sh configs/run/r4.yaml
CONFIG="${1:-$PROJECT_DIR/configs/run/b1.yaml}"

# experiment_name 从 config 动态读（与 verl 内部 trainer.experiment_name 一致）。
# fallback 到 basename（与 run_phases.sh 一致）。
EXPERIMENT_NAME=$(grep -E "^[[:space:]]*experiment_name:" "$CONFIG" 2>/dev/null | head -1 | sed -E "s/.*experiment_name:[[:space:]]*//;s/[[:space:]\"']*//g" || true)
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(basename "$CONFIG" .yaml)}"
export LIGHTLLM_LOG_LEVEL=WARNING TQ_LOGGING_LEVEL=WARNING
export MODELING_BACKEND=hf
# lightllm rollout 需要 LightLLM + verl 在 PYTHONPATH 上，再加本项目根（trainer.*）。
export PYTHONPATH=$LIGHTLLM_DIR:$VERL_DIR:$PROJECT_DIR

# ---- runs/ 产物布局（自包含实验目录，见 runs/README.md）----
# 每个实验一个自包含子目录 runs/<phase>/<exp>/：config 快照 + eval + 日志 + buffer，
# 权重不塞进来（太大）——verl 仍把真实 ckpt 写到 runs 外的 ckpts/<exp>/，
# runs/<phase>/<exp>/checkpoints 软链过去。
# phase 从 config 路径推（configs/phaseN/ 或 configs/run/ → 归 runN；显式传 PHASE 覆盖）。
if [ -z "${PHASE:-}" ]; then
    case "$CONFIG" in
        *"/phase"[0-9]*) PHASE=$(echo "$CONFIG" | sed -E 's#.*/(phase[0-9]+)/.*#\1#') ;;
        *"/run/"*)       PHASE="run" ;;
        *)               PHASE="misc" ;;
    esac
fi
export RUN_DIR="$PROJECT_DIR/runs/$PHASE/$EXPERIMENT_NAME"
export CKPT_DIR="$PROJECT_DIR/ckpts/$EXPERIMENT_NAME"   # 真实权重（runs 外）
export RESULT_DIR="$RUN_DIR"                            # 日志写进实验自包含目录
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/eval" "$RUN_DIR/buffer" "$CKPT_DIR"
# checkpoints 软链：runs/<phase>/<exp>/checkpoints -> ../../../ckpts/<exp>
ln -sfn "$CKPT_DIR" "$RUN_DIR/checkpoints"
# config 快照（可追溯该实验用的完整参数）
cp -f "$CONFIG" "$RUN_DIR/config.snapshot.yaml" 2>/dev/null || true

# 训练密钥（gitignore 的 .env）：SWANLAB_API_KEY + REWARD_API_BASE/MODEL/KEY 都从这里来。
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi
# SWANLAB_API_KEY must come from .env — no hardcoded default.
if [ -z "${SWANLAB_API_KEY:-}" ]; then
    echo "[start_train.sh] WARNING: SWANLAB_API_KEY not set (neither .env nor env). W&B logging will fail."
fi

# AFS (quarkfs FUSE) 不支持 fcntl.flock，HF/verl 缓存指向本地盘
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache
export HF_HOME=/tmp/hf_home

# reward = project model judge（trainer/model_reward.py 读 REWARD_API_BASE/MODEL/KEY）。
# 2026-07-01: Reward model 改走 sufy（anthropic/claude-4.8-opus），不再本地 vLLM 部署。
# 值来自 configs/agents.yaml（reward 段）；仅当缺省时才回退 mock。
export REWARD_API_BASE="${REWARD_API_BASE:-http://127.0.0.1:8100/v1}"
export REWARD_MODEL="${REWARD_MODEL:-mock-judge}"
# API key: TOKENHUB_API_KEY 在 .env 中设置，所有 agent 共用。
export TOKENHUB_API_KEY="${TOKENHUB_API_KEY:-sk-local}"

# Observer / Questioner env（agents/base.py 读取；值来自 .env）。
export OBSERVER_API_BASE="${OBSERVER_API_BASE:-}"
export OBSERVER_MODEL="${OBSERVER_MODEL:-}"
# Questioner: 单模型 fallback（USERSIM_ENDPOINTS 优先，见下）
export USERSIM_API_BASE="${USERSIM_API_BASE:-}"
export USERSIM_MODEL="${USERSIM_MODEL:-}"
# Questioner: 多模型轮换（JSON 数组，每 5 次 query 切换模型，抗模式坍缩）
export USERSIM_ENDPOINTS="${USERSIM_ENDPOINTS:-}"
export USERSIM_ROTATE_EVERY="${USERSIM_ROTATE_EVERY:-5}"

# Agentic rollout 走 e2b 腾讯沙箱：加载凭证（docker/sandbox/tencent.env，gitignore，已填）。
if [ -f "$PROJECT_DIR/docker/sandbox/tencent.env" ]; then
    set -a; source "$PROJECT_DIR/docker/sandbox/tencent.env"; set +a
fi

# 集群节点容器里没有我这个账号，按 sunhao4 / uid 10183 现场创建（已存在则忽略）。
useradd -M -d /mnt/afs_toolcall/sunhao4 -u 10183 -s "$(which bash)" sunhao4 2>/dev/null || true

# 运行时依赖（与 docker/qwen36-lightllm/Dockerfile 固化版本对齐；镜像已带则秒过）
python -m pip install -U tensordict
python -m pip install -U accelerate==1.13.0 nvidia-cutlass-dsl>=4.4.2 transformers==5.8.0
python -m pip install e2b==2.24.0 flash-linear-attention==0.4.2 langchain-openai mistral-common==1.11.2 rapidfuzz TransferQueue==0.1.6

# 权重拷到节点本地盘（各节点各自读 /tmp，避开 AFS 带宽抢占）
[ -d /tmp/qwen36 ] || cp -rL /mnt/afs_agents/share_models/Qwen/Qwen3.6-27B /tmp/qwen36
# 对象存储配置：原参考脚本借用的是别人的 aoss.conf；存在才软链，缺了不报错。
# 若你有自己的 aoss.conf，把下面源路径换成你的。
[ -f /mnt/afs_reason/liangjinwei/aoss.conf ] && ln -sf /mnt/afs_reason/liangjinwei/aoss.conf ~/aoss.conf
mkdir -p $RUN_DIR/logs

cd "$PROJECT_DIR"

# inter-node synchronization with torch distributed package
OMP_NUM_THREADS=1 python -c "import os, torch.distributed as d; s,r,w = d.rendezvous(f'tcp://$MASTER_ADDR:$MASTER_PORT'); d.init_process_group('gloo', store=s, rank=r, world_size=w); d.barrier()"

if [ "$RANK" = "0" ]; then
    # 启动前校验：Observer/Questioner/Reward 三端点应配置且模型不同
    python -c "
import os, sys
sys.path.insert(0, '$PROJECT_DIR')
from agents.base import validate_endpoints_distinct
warnings = validate_endpoints_distinct()
for w in warnings:
    print(f'[start_train.sh] WARNING: {w}')
if any('not configured' in w for w in warnings):
    print('[start_train.sh] ABORT: critical endpoint missing. Check .env')
    sys.exit(1)
"
    # 仅当仍在用 mock judge 时才在本地起 mock 服务；用了真实 judge（tokenhub）则跳过。
    if [ "$REWARD_MODEL" = "mock-judge" ]; then
        python "$PROJECT_DIR/scripts/mock_judge.py" --port 8100 > /tmp/mock_judge.log 2>&1 &
    fi
    ray start --head --disable-usage-stats && ray status
    python -m trainer.cl_main --config "$CONFIG" \
        2>&1 | tee $RUN_DIR/logs/train.log
    ray stop --force
else
    ray start --address $MASTER_ADDR:6379 --block
fi
sleep 10s
