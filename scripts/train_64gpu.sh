#!/usr/bin/env bash
# 64 卡集群训练 wrapper（8 节点 × 8 卡）—— 正式训练主规格。
# 拓扑：8×8=64 卡，SP=4 → DP=16，rollout TP=4。
# batch: train=1024 ppo_mini=64（1024%16=0, 64%16=0 ✓）。
#
# 集群按每节点一份调度本脚本，SenseCore 注入 RANK/MASTER_ADDR（train_cl.sh 内
# 走多机 rendezvous + ray head/worker）。默认 27B 正式配置。
#
# 用法：
#   bash scripts/train_64gpu.sh --config configs/run/b1.yaml           # B1 遗忘基线
#   bash scripts/train_64gpu.sh --config configs/run/r4.yaml --buckets # R4 桶序 + replay
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$DIR/train_cl.sh" \
  --nnodes 8 --gpus-per-node 8 --rollout-tp 4 --ulysses-sp 4 \
  --train-batch 1024 --ppo-mini 64 --gpu-mem-util 0.75 \
  --config "$DIR/../configs/run/b1.yaml" \
  "$@"
