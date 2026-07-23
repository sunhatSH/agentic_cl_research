#!/usr/bin/env bash
# 16 卡训练 wrapper（2 节点 × 8 卡）。
# 拓扑：2×8=16 卡，SP=2 → DP=8，rollout TP=2。
# batch: train=64（64 query×n8=512 轨迹/step），ppo_mini=64（64%8=0 ✓）
#
# 用法：
#   bash scripts/train_16gpu.sh --buckets --config configs/run/b1_9b_16gpu.yaml
#   bash scripts/train_16gpu.sh --config configs/run/b1_9b_16gpu.yaml   # 单次非桶（如果数据混合）
# 额外参数透传给 train_cl.sh。
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$DIR/train_cl.sh" \
  --nnodes 2 --gpus-per-node 8 --rollout-tp 2 --ulysses-sp 2 \
  --train-batch 64 --ppo-mini 64 --gpu-mem-util 0.30 \
  --config "$DIR/../configs/run/b1_9b_16gpu.yaml" \
  "$@"
