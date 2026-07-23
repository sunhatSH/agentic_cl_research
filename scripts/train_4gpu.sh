#!/usr/bin/env bash
# 4 卡单机训练 wrapper（开发机 / 9B baseline）。
# 拓扑：1×4=4 卡，SP=1 → DP=4，rollout TP=2。
# batch: train=32（32 query×n8=256 轨迹/step）ppo_mini=32 micro=1（config 内设；OOM 无退路降 max_response）。
#
# 用法：
#   bash scripts/train_4gpu.sh --smoke --config configs/phase1/smoke_1step.yaml
#   bash scripts/train_4gpu.sh --buckets --config configs/run/b1_9b.yaml   # 9B baseline 按桶顺序训
# 额外参数透传给 train_cl.sh（可覆盖 --config / --train-batch 等）。
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$DIR/train_cl.sh" \
  --nnodes 1 --gpus-per-node 4 --rollout-tp 2 --ulysses-sp 1 \
  --train-batch 32 --ppo-mini 32 --gpu-mem-util 0.30 \
  --config "$DIR/../configs/run/b1_9b.yaml" \
  "$@"
