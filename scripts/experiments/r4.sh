#!/usr/bin/env bash
# 实验 R4 = 完整 CL 方案(9 桶 buffer + 抗遗忘 priority，W0 token 均匀)。
# 训练 + 按 9 能力桶评测。
#   启动: bash scripts/experiments/r4.sh
#         NNODES=4 bash scripts/experiments/r4.sh   # 32 卡
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
exec bash "$SCRIPT_DIR/_run_one.sh" "$ROOT_DIR/configs/run/r4.yaml" "$@"
