#!/usr/bin/env bash
# 单实验通用核心：训练 + 按桶评测。被 scripts/experiments/<exp>.sh 调用。
#
# 用法(一般不直接调,通过 <exp>.sh):
#   bash scripts/experiments/_run_one.sh <config.yaml> [extra train args]
#
# 做两件事：
#   1. 训练该 config（通过 _train_impl.sh）
#   2. 训练成功后按 9 能力桶评测(RUN_EVAL=0 可跳过)
#
# 卡数解耦：从 env 读 NNODES/N_GPUS_PER_NODE/ROLLOUT_TP_SIZE/ULYSSES_SP_SIZE/
# TRAIN_BATCH_SIZE/PPO_MINI_BATCH_SIZE，转成 _train_impl.sh 的命令行参数。集群按每
# 节点调度时这些 env 由平台/wrapper 注入；直接调也可 export 覆盖。
set -uo pipefail

CFG="${1:?用法: $0 <config.yaml> [extra args]}"
shift || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 从 env 取拓扑（缺省 = _train_impl.sh 的默认），转成命令行参数。
exec bash "$ROOT_DIR/scripts/_train_impl.sh" --config "$CFG" \
  --nnodes "${NNODES:-1}" --gpus-per-node "${N_GPUS_PER_NODE:-8}" \
  --rollout-tp "${ROLLOUT_TP_SIZE:-4}" --ulysses-sp "${ULYSSES_SP_SIZE:-1}" \
  --train-batch "${TRAIN_BATCH_SIZE:-256}" --ppo-mini "${PPO_MINI_BATCH_SIZE:-32}" \
  "$@"
