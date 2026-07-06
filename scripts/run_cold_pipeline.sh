#!/usr/bin/env bash
# Cold-start pipeline: classify (32 workers) → collect (128 concurrent).
# Stage 2 only runs if Stage 1 succeeds.
set -e
cd "$(git rev-parse --show-toplevel)"
source scripts/load_tencent_env.sh

echo "========== STAGE 1: 打标 (32 并发) =========="
.venv/bin/python scripts/run_cold_start.py --no-collect --classify-workers 32

echo ""
echo "========== STAGE 2: 采集 (128 并发) =========="
.venv/bin/python scripts/run_cold_start.py \
    --no-classify --num-queries 3878 --max-concurrent 128

echo ""
echo "========== ALL DONE =========="
