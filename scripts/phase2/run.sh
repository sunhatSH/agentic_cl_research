#!/usr/bin/env bash
# Phase 2：KL 单独验证 (K1-K5, K2-R)
# 可选参数 --only <exp> 仅运行指定实验，如 --only k2
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_DIR="$ROOT_DIR/configs/phase2"

EXPERIMENTS=(k1 k2 k3 k4 k5 k2-r)

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
    echo "=== Phase 2: $exp ==="
    bash "$ROOT_DIR/scripts/train.sh" "$CONFIG_DIR/${exp}.yaml" "${EXTRA_ARGS[@]}"
done
