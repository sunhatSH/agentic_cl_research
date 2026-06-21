#!/usr/bin/env bash
# Phase 4：KL × Replay 组合验证 (C1-C4)
# 可选参数 --only <exp> 仅运行指定实验
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_DIR="$ROOT_DIR/configs/phase4"

EXPERIMENTS=(c1 c2 c3 c4)

ONLY=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only) ONLY="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

for exp in "${EXPERIMENTS[@]}"; do
    if [[ -n "$ONLY" && "$exp" != "$ONLY" ]]; then
        continue
    fi
    echo "=== Phase 4: $exp ==="
    bash "$ROOT_DIR/scripts/train.sh" "$CONFIG_DIR/${exp}.yaml" "${EXTRA_ARGS[@]}"
done
