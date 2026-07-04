#!/usr/bin/env bash
# SenseCore(商汤)平台多机变量映射 —— 被 run_phases.sh / start_train.sh source。
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
