#!/usr/bin/env bash
# Cold-start pipeline: 打标(classify) → 多轮采集(collect).
# Stage 2 only runs if Stage 1 succeeds. Both stages resumable/incremental.
set -e
cd "$(git rev-parse --show-toplevel)"
source scripts/load_tencent_env.sh

echo "========== STAGE 1: LLM 打桶 (32 并发) =========="
# 静态映射 6 种 task_family + sufy 分类其余; 写 datasets/queries.jsonl
.venv/bin/python scripts/run_cold_start.py --no-collect --classify-workers 32

echo ""
echo "========== STAGE 2: 多轮采集 (32 并发, actor+observer+questioner, 无 reward/winner) =========="
# incremental: 跳过已成功, 只补缺失+失败; 原子落盘不覆盖好数据
.venv/bin/python scripts/run_cold_start.py \
    --no-classify --no-generate \
    --num-queries 3878 --max-concurrent 32 \
    --collect-mode incremental \
    --max-turns 3 --hermes-max-turns 30 --slot-timeout 900

echo ""
echo "========== ALL DONE =========="
echo "  queries      → datasets/queries.jsonl"
echo "  trajectories → rollouts/cold_start/grpo_hermes.jsonl"
echo "  下一步: python scripts/trajectory_to_parquet.py --input rollouts/cold_start/grpo_hermes.jsonl --out-dir datasets"
