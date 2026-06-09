#!/usr/bin/env bash
# Phase 3：Replay 单独验证 (R0, R3-R6, R4-w, R4-K)
# 可选参数 --only <exp> 仅运行指定实验
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_DIR="$ROOT_DIR/configs/phase3"

EXPERIMENTS=(r0 r3 r4 r5 r4-w r6 r4-k)

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
    echo "=== Phase 3: $exp ==="
    bash "$ROOT_DIR/scripts/train.sh" "$CONFIG_DIR/${exp}.yaml" "${EXTRA_ARGS[@]}"
done
