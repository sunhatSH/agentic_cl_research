#!/usr/bin/env bash
# 4 卡单机训练 wrapper（开发机 / smoke）。
# 拓扑：1×4=4 卡，SP=1 → DP=4，rollout TP=2（推理占 2 卡，训练用全部）。
# batch: train=8 ppo_mini=8（8%4=0 ✓）。
#
# 用法：
#   bash scripts/train_4gpu.sh --smoke --config configs/phase1/smoke_1step.yaml
#   bash scripts/train_4gpu.sh --config configs/run/b1_8b.yaml   # 4 卡跑 9B
# 额外参数透传给 train_cl.sh（可覆盖 --config / --train-batch 等）。
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$DIR/train_cl.sh" \
  --nnodes 1 --gpus-per-node 4 --rollout-tp 2 --ulysses-sp 1 \
  --train-batch 8 --ppo-mini 8 --gpu-mem-util 0.20 \
  --config "$DIR/../configs/run/b1_8b.yaml" \
  "$@"
