#!/usr/bin/env bash
# 32 卡集群训练 wrapper（4 节点 × 8 卡）。
# 拓扑：4×8=32 卡，SP=4 → DP=8，rollout TP=4。
# batch: train=512 ppo_mini=64（512%8=0, 64%8=0 ✓）。
#
# 集群按每节点一份调度本脚本，SenseCore 注入 RANK/MASTER_ADDR（train_cl.sh 内
# 走多机 rendezvous + ray head/worker）。默认 27B 正式配置。
#
# 用法：
#   bash scripts/train_32gpu.sh --config configs/run/b1.yaml
#   bash scripts/train_32gpu.sh --config configs/run/r4.yaml --buckets
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$DIR/train_cl.sh" \
  --nnodes 4 --gpus-per-node 8 --rollout-tp 4 --ulysses-sp 4 \
  --train-batch 512 --ppo-mini 64 --gpu-mem-util 0.75 \
  --config "$DIR/../configs/run/b1.yaml" \
  "$@"
