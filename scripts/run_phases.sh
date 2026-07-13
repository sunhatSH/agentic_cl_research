#!/usr/bin/env bash
# 集群一键训练：按顺序跑一个或多个实验。
#
# 卡数/机器数【不写死】，由环境变量注入(configs/cluster.yaml 用 ${oc.env:...})：
#   NNODES(默认8) x N_GPUS_PER_NODE(默认8) = 总卡数(默认 64)。
#   32 卡: NNODES=4 bash scripts/run_phases.sh ...   (4x8=32)
#   并行度按需: ROLLOUT_TP_SIZE / ULYSSES_SP_SIZE (默认4)；TRAIN_BATCH_SIZE(默认1024)。
#
# 仿照集群参考脚本的 ray 多机 + torch distributed rendezvous 骨架，适配本项目：
#   - 入口用本项目 trainer.cl_main（保留 CL Loss + Replay Buffer 注入）。
#   - rank0 起 ray head，按给定顺序逐个跑 config；其余 rank 起 worker 后 --block。
#   - 一次提交把多个 phase 串起来跑完，中间不用人工介入。
#
# 用法：
#   bash scripts/run_phases.sh                              # 默认跑 b1 + r4
#   bash scripts/run_phases.sh configs/run/b1.yaml          # 只跑 b1
#   NNODES=4 bash scripts/run_phases.sh configs/run/b1.yaml # 32 卡
#
# 集群按每节点一份调度本脚本，注入 RANK / MASTER_ADDR / MASTER_PORT。
set -uo pipefail

# 全部走 AFS 共享挂载（集群各节点都能访问）。verl/LightLLM 已拷至自己 AFS 目录
# （含 lightllm_rollout + recipe_custom），与外部目录解耦。
PROJECT_DIR=/mnt/afs_toolcall/sunhao4/agentic_cl_research
LIGHTLLM_DIR=/mnt/afs_toolcall/sunhao4/Documents/LightLLM
VERL_DIR=/mnt/afs_toolcall/sunhao4/Documents/verl
SRC_MODEL=/mnt/afs_agents/share_models/Qwen/Qwen3.6-27B

# SenseCore 多机变量映射(RANK/MASTER_ADDR/MASTER_PORT/NNODES 兼容注入)。
source "$(dirname "$0")/_sensecore_env.sh"

# 要跑的实验配置列表（按顺序执行）。不传参则跑默认两个。
CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
    CONFIGS=(
        "$PROJECT_DIR/configs/run/b1.yaml"
        "$PROJECT_DIR/configs/run/r4.yaml"
    )
fi

export LIGHTLLM_LOG_LEVEL=WARNING TQ_LOGGING_LEVEL=WARNING
export MODELING_BACKEND=hf
# lightllm rollout 需要 LightLLM + verl 在 PYTHONPATH 上，再加本项目根（trainer.*）。
export PYTHONPATH=$LIGHTLLM_DIR:$VERL_DIR:$PROJECT_DIR
export RESULT_DIR=$PROJECT_DIR/outputs

# 训练密钥（gitignore 的 .env）：SWANLAB_API_KEY + REWARD_API_BASE/MODEL/KEY。
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi
# SWANLAB_API_KEY must come from .env — no hardcoded default.
if [ -z "${SWANLAB_API_KEY:-}" ]; then
    echo "[run_phases.sh] WARNING: SWANLAB_API_KEY not set (neither .env nor env). W&B logging will fail."
fi

# AFS (quarkfs FUSE) 不支持 fcntl.flock，HF/verl 缓存指向本地盘
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache
export HF_HOME=/tmp/hf_home

# reward = 项目 model judge（值来自上面的 .env；缺失才回退 mock）。
export REWARD_API_BASE="${REWARD_API_BASE:-http://127.0.0.1:8100/v1}"
export REWARD_MODEL="${REWARD_MODEL:-mock-judge}"
# API key: TOKENHUB_API_KEY 在 .env 中设置，所有 agent 共用。
export TOKENHUB_API_KEY="${TOKENHUB_API_KEY:-sk-local}"

# Agentic rollout 走 e2b 腾讯沙箱：加载凭证（docker/sandbox/tencent.env，gitignore，已填）。
if [ -f "$PROJECT_DIR/docker/sandbox/tencent.env" ]; then
    set -a; source "$PROJECT_DIR/docker/sandbox/tencent.env"; set +a
fi

# 集群节点容器里没有我这个账号，按 sunhao4 / uid 10183 现场创建（已存在则忽略）。
useradd -M -d /mnt/afs_toolcall/sunhao4 -u 10183 -s "$(which bash)" sunhao4 2>/dev/null || true

# 运行时依赖（与 docker/qwen36-lightllm/Dockerfile 固化版本对齐；镜像已带则秒过）
python -m pip install -U tensordict
python -m pip install -U "accelerate==1.13.0" "nvidia-cutlass-dsl>=4.4.2" "transformers==5.8.0"
python -m pip install "e2b==2.24.0" "flash-linear-attention==0.4.2" langchain-openai \
    "mistral-common==1.11.2" rapidfuzz "TransferQueue==0.1.6"

# 权重拷到节点本地盘（各节点各自读 /tmp，避开 AFS 带宽抢占）
[ -d /tmp/qwen36 ] || cp -rL "$SRC_MODEL" /tmp/qwen36
# 对象存储配置：原参考脚本借用的是别人的 aoss.conf；存在才软链，缺了不报错。
[ -f /mnt/afs_reason/liangjinwei/aoss.conf ] && ln -sf /mnt/afs_reason/liangjinwei/aoss.conf ~/aoss.conf

cd "$PROJECT_DIR"

# 跨节点先用 torch distributed gloo 做一次 barrier，确保所有节点就绪
OMP_NUM_THREADS=1 python -c "import os, torch.distributed as d; s,r,w = d.rendezvous(f'tcp://{os.environ[\"MASTER_ADDR\"]}:{os.environ[\"MASTER_PORT\"]}'); d.init_process_group('gloo', store=s, rank=r, world_size=w); d.barrier()"

if [ "${RANK}" = "0" ]; then
    # 打印本次实际卡数/并行度(来自环境变量，未设走 cluster.yaml 默认)。
    _NN="${NNODES:-8}"; _NG="${N_GPUS_PER_NODE:-8}"
    echo "[run_phases] GPU 拓扑: NNODES=${_NN} x N_GPUS_PER_NODE=${_NG} = $((_NN*_NG)) 卡 | " \
         "TP=${ROLLOUT_TP_SIZE:-4} SP=${ULYSSES_SP_SIZE:-4} TRAIN_BATCH=${TRAIN_BATCH_SIZE:-1024}"
    # 仅当仍在用 mock judge 时才在本地起 mock 服务；用了真实 judge（tokenhub）则跳过。
    if [ "$REWARD_MODEL" = "mock-judge" ]; then
        python "$PROJECT_DIR/scripts/mock_judge.py" --port 8100 > /tmp/mock_judge.log 2>&1 &
    fi
    ray start --head --disable-usage-stats && ray status

    rc=0
    for cfg in "${CONFIGS[@]}"; do
        # 与 start_train.sh 一致的 runs/ 布局：每实验自包含目录 runs/<phase>/<exp>/，
        # 真实权重在 runs 外 ckpts/<exp>/，runs/<phase>/<exp>/checkpoints 软链过去。
        exp=$(grep -E "^[[:space:]]*experiment_name:" "$cfg" 2>/dev/null | head -1 | sed -E "s/.*experiment_name:[[:space:]]*//;s/[[:space:]\"']*//g" || true)
        exp="${exp:-$(basename "$cfg" .yaml)}"
        case "$cfg" in
            *"/phase"[0-9]*) phase=$(echo "$cfg" | sed -E 's#.*/(phase[0-9]+)/.*#\1#') ;;
            *"/run/"*)       phase="run" ;;
            *)               phase="misc" ;;
        esac
        run_dir="$PROJECT_DIR/runs/$phase/$exp"
        ckpt_dir="$PROJECT_DIR/ckpts/$exp"
        # 训练/eval 日志 + 生成文本 → logs/experiments/<phase>/<exp>/（单一来源）。
        # runs/<phase>/<exp>/ 只留 config 快照 + checkpoints 软链。
        log_dir="$PROJECT_DIR/logs/experiments/$phase/$exp"
        mkdir -p "$run_dir" "$run_dir/eval" "$run_dir/buffer" "$ckpt_dir" \
                 "$log_dir/rollout" "$log_dir/val"
        ln -sfn "$ckpt_dir" "$run_dir/checkpoints"
        cp -f "$cfg" "$run_dir/config.snapshot.yaml" 2>/dev/null || true
        # 全量记录：verl 把 rollout/验证生成文本落到这两个目录（base.yaml 读同名 env）。
        export ROLLOUT_DATA_DIR="$log_dir/rollout"
        export VAL_DATA_DIR="$log_dir/val"
        # console 仅 debug：DEBUG=1 保留 console，否则只 swanlab（正式训练不刷屏）。
        if [ "${DEBUG:-0}" = "1" ]; then
            export VERL_LOGGER="[console,swanlab]"
        else
            export VERL_LOGGER="[swanlab]"
        fi
        echo "================ [run_phases] start: $exp ($cfg) -> logs/experiments/$phase/$exp ================"
        CKPT_DIR="$ckpt_dir" python -m trainer.cl_main --config "$cfg" \
            2>&1 | tee "$log_dir/train.log"
        status=${PIPESTATUS[0]}
        if [ "$status" != "0" ]; then
            echo "[run_phases] FAILED: $exp (exit $status) — 继续后续实验" >&2
            rc=$status
        else
            echo "[run_phases] done: $exp"
            # 训练成功后自动评测(按 9 能力桶记录分数)。RUN_EVAL=0 可跳过。
            if [ "${RUN_EVAL:-1}" = "1" ]; then
                echo "[run_phases] eval: $exp -> $run_dir/eval (按桶记录)"
                bash "$PROJECT_DIR/scripts/eval.sh" "$ckpt_dir" --exp-dir "$run_dir" \
                    2>&1 | tee "$log_dir/eval.log" || \
                    echo "[run_phases] WARN: eval 失败($exp),训练结果仍保留" >&2
            fi
        fi
    done

    ray stop --force
    exit $rc
else
    ray start --address "$MASTER_ADDR:6379" --block
fi
sleep inf
