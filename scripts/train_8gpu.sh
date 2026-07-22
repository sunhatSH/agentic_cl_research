#!/usr/bin/env bash
# 8 卡单机训练 wrapper。
# 拓扑：1×8=8 卡，SP=1 → DP=8，rollout TP=2。
# batch: train=256 ppo_mini=32（256%8=0, 32%8=0 ✓）。
#
# 用法：
#   bash scripts/train_8gpu.sh                                  # 默认 b1_8b（9B）
#   bash scripts/train_8gpu.sh --buckets                        # 9 桶顺序训练
#   bash scripts/train_8gpu.sh --config configs/phase3/r4.yaml  # 换实验
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$DIR/train_cl.sh" \
  --nnodes 1 --gpus-per-node 8 --rollout-tp 2 --ulysses-sp 1 \
  --train-batch 256 --ppo-mini 32 --gpu-mem-util 0.75 \
  --config "$DIR/../configs/run/b1_8b.yaml" \
  "$@"
