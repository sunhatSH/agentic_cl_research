#!/usr/bin/env bash
# Phase 1：启动纯 RL 遗忘基线 (B1)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== Phase 1: B1 (纯 RL 遗忘基线) ==="
bash "$ROOT_DIR/scripts/train.sh" "$ROOT_DIR/configs/phase1/b1.yaml" "$@"
