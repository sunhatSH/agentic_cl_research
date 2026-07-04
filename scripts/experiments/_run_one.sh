#!/usr/bin/env bash
# 单实验通用核心：训练 + 按桶评测。被 scripts/experiments/<exp>.sh 调用。
#
# 用法(一般不直接调,通过 <exp>.sh):
#   bash scripts/experiments/_run_one.sh <config.yaml> [extra train args]
#
# 做两件事(复用 run_phases.sh 的多机骨架):
#   1. 训练该 config
#   2. 训练成功后按 9 能力桶评测(RUN_EVAL=0 可跳过)
#
# 卡数解耦(同 run_phases): NNODES/N_GPUS_PER_NODE/ROLLOUT_TP_SIZE/... 环境变量覆盖。
set -uo pipefail

CFG="${1:?用法: $0 <config.yaml> [extra args]}"
shift || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 复用 run_phases.sh 作为多机训练+评测入口(它已含 ray/rendezvous/自动评测)。
# run_phases 接受 config 列表,传单个即单实验。
exec bash "$ROOT_DIR/scripts/run_phases.sh" "$CFG" "$@"
