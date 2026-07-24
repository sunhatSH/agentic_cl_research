#!/usr/bin/env bash
# SenseCore(商汤)平台多机变量映射 —— 被 train.sh（多机分支）source。
#
# SenseCore PyTorch 任务注入 SENSECORE_PYTORCH_* 变量;本项目脚本用裸名
# RANK/MASTER_ADDR/MASTER_PORT/NNODES。这里做兼容映射:
#   优先用已存在的裸名(手动 export / 其他平台) → 回退 SenseCore 注入 → 最后默认。
# 两种平台 + 单机手跑都不崩。参考嘉伟 official_Qwen3_6_27B_*.sh 的写法。
#
# source 本脚本后,RANK/MASTER_ADDR/MASTER_PORT/NNODES/N_GPUS_PER_NODE 均就绪。

export RANK="${RANK:-${SENSECORE_PYTORCH_NODE_RANK:-0}}"
export MASTER_ADDR="${MASTER_ADDR:-${SENSECORE_PYTORCH_MASTER_ADDR:-127.0.0.1}}"
export MASTER_PORT="${MASTER_PORT:-${SENSECORE_PYTORCH_MASTER_PORT:-29503}}"
export NNODES="${NNODES:-${SENSECORE_PYTORCH_NNODES:-1}}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
# WORLD_SIZE(节点数)供 rendezvous 用;SenseCore 也可能直接注入。
export WORLD_SIZE="${WORLD_SIZE:-${SENSECORE_PYTORCH_WORLD_SIZE:-$NNODES}}"

echo "[sensecore_env] RANK=$RANK NNODES=$NNODES MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=$WORLD_SIZE"

# ---- 启动前整除自检(避免 32/64 卡并行度配错训到一半崩)----
# 约束(verl engine_workers.py:258 + train batch 按 DP 切):
#   DP = 总卡数 / ULYSSES_SP_SIZE;  ppo_mini_batch_size % DP == 0;  train_batch_size % DP == 0
# 只在 rank0 检查+打印(其余节点跳过,减少噪声)。任一不满足直接 exit,不进训练。
if [ "${RANK}" = "0" ]; then
    _TOTAL=$((NNODES * N_GPUS_PER_NODE))
    _SP="${ULYSSES_SP_SIZE:-4}"
    _MINI="${PPO_MINI_BATCH_SIZE:-64}"
    _TRAIN="${TRAIN_BATCH_SIZE:-1024}"
    if [ $((_TOTAL % _SP)) -ne 0 ]; then
        echo "[sensecore_env][ERROR] 总卡数 $_TOTAL 不能被 ULYSSES_SP_SIZE=$_SP 整除。" >&2
        echo "  调 ULYSSES_SP_SIZE(如 32卡试 2/4/8)使其整除总卡数。" >&2
        exit 3
    fi
    _DP=$((_TOTAL / _SP))
    if [ $((_MINI % _DP)) -ne 0 ]; then
        echo "[sensecore_env][ERROR] ppo_mini_batch_size=$_MINI 不能被 DP=$_DP (=总卡$_TOTAL/SP$_SP) 整除。" >&2
        exit 3
    fi
    if [ $((_TRAIN % _DP)) -ne 0 ]; then
        echo "[sensecore_env][ERROR] train_batch_size=$_TRAIN 不能被 DP=$_DP 整除。" >&2
        exit 3
    fi
    echo "[sensecore_env] 整除自检 OK: 总卡=$_TOTAL SP=$_SP DP=$_DP | mini_batch=$_MINI train_batch=$_TRAIN"
fi
