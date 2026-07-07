#!/usr/bin/env bash
# Cold-start pipeline: classify+persona (Stage 1) → multi-turn collect (Stage 2).
# Stage 2 only runs if Stage 1 succeeds. Both stages resumable/incremental.
set -e
cd "$(git rev-parse --show-toplevel)"
source scripts/load_tencent_env.sh

echo "========== STAGE 1: LLM 打桶 + 人设 (32 并发) =========="
.venv/bin/python scripts/run_cold_start.py --generate --no-collect --classify-workers 32

echo ""
echo "========== STAGE 2: 多轮采集 (32 并发) =========="
.venv/bin/python scripts/run_cold_start.py \
    --num-queries 2849 --max-concurrent 32 \
    --max-turns 3 --hermes-max-turns 30 --slot-timeout 900

echo ""
echo "========== ALL DONE =========="
echo "  queries      → datasets/queries.jsonl"
echo "  trajectories → rollouts/cold_start/grpo_hermes.jsonl"
