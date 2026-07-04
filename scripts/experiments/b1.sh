#!/usr/bin/env bash
# 实验 B1 = baseline(纯 27B RL，不加 CL 算法：无 replay/无 KL/无 buffer，仅 entropy=0.001)。
# 训练 + 按 9 能力桶评测。
#   启动: bash scripts/experiments/b1.sh            # 默认 64 卡
#         NNODES=4 bash scripts/experiments/b1.sh   # 32 卡
#         RUN_EVAL=0 bash scripts/experiments/b1.sh # 只训不评
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
exec bash "$SCRIPT_DIR/_run_one.sh" "$ROOT_DIR/configs/run/b1.yaml" "$@"
