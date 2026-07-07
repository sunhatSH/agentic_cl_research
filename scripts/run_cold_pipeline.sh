#!/usr/bin/env bash
# Full cold-start pipeline: taskspec → queries → trajectories → parquet → buffer.
# Each stage is incremental — safe to resume if interrupted.
#
# Stages (skip anything already done by commenting out the line):
#   S1  --generate            (re)generate queries.jsonl (classify + persona)
#   S1b filter_unrunnable     LLM 剔除沙箱不可跑任务
#   S2  (default)             multi-turn collection (actor+observer+questioner)
#   S3  trajectory_to_parquet trajectories → train.parquet + val.parquet
#   S4  warmup_buffer         trajectories → buffer.sqlite (训练前预填)
#
# Usage:
#   # Everything from scratch
#   bash scripts/run_cold_pipeline.sh --full
#
#   # Just collect (queries already ready)
#   bash scripts/run_cold_pipeline.sh
#
#   # Collect + parquet + warmup
#   bash scripts/run_cold_pipeline.sh --collect --parquet --warmup

set -e
cd "$(git rev-parse --show-toplevel)"

DO_GENERATE=false
DO_FILTER=false
DO_COLLECT=true
DO_PARQUET=false
DO_WARMUP=false

for arg in "$@"; do
  case "$arg" in
    --full) DO_GENERATE=true; DO_FILTER=true; DO_COLLECT=true; DO_PARQUET=true; DO_WARMUP=true ;;
    --generate) DO_GENERATE=true ;;
    --filter) DO_FILTER=true ;;
    --collect) DO_COLLECT=true ;;
    --parquet) DO_PARQUET=true ;;
    --warmup) DO_WARMUP=true ;;
    --all-collect) DO_GENERATE=true; DO_FILTER=true; DO_COLLECT=true; DO_PARQUET=true; DO_WARMUP=true ;;
    *) echo "unknown: $arg"; exit 2 ;;
  esac
done

QUERIES=datasets/queries.jsonl
TRAJ=rollouts/cold_start/grpo_hermes.jsonl
PARQUET_DIR=datasets
BUFFER=buffer_dumps/warmup.sqlite

echo "============================================"
echo " Cold-Start Pipeline"
echo "  generate=$DO_GENERATE  filter=$DO_FILTER  collect=$DO_COLLECT"
echo "  parquet=$DO_PARQUET  warmup=$DO_WARMUP"
echo "============================================"
echo ""

# ── S1: queries generation ──────────────────────────────────────────────
if $DO_GENERATE; then
  echo "========== S1: 打桶 + 人设 =========="
  .venv/bin/python scripts/run_cold_start.py --generate --no-collect --classify-workers 32
  wc -l "$QUERIES"
fi

# ── S1b: filter unrunnable ──────────────────────────────────────────────
if $DO_FILTER; then
  echo ""
  echo "========== S1b: 剔除沙箱不可跑任务 =========="
  .venv/bin/python scripts/filter_unrunnable.py --workers 32
  wc -l "$QUERIES"
fi

# ── S2: collection ──────────────────────────────────────────────────────
if $DO_COLLECT; then
  echo ""
  N=$(wc -l < "$QUERIES")
  echo "========== S2: 多轮采集 ($N queries, 32 并发) =========="
  source scripts/load_tencent_env.sh
  .venv/bin/python scripts/run_cold_start.py \
      --num-queries "$N" --max-concurrent 32 \
      --max-turns 20 --hermes-max-turns 30 --slot-timeout 900
  wc -l "$TRAJ" 2>/dev/null || echo "(collection may still be running)"
fi

# ── S3: parquet ─────────────────────────────────────────────────────────
if $DO_PARQUET; then
  echo ""
  echo "========== S3: 轨迹 → parquet =========="
  .venv/bin/python scripts/trajectory_to_parquet.py \
      --input "$TRAJ" --out-dir "$PARQUET_DIR"
  ls -lh "$PARQUET_DIR"/train.parquet "$PARQUET_DIR"/val.parquet
fi

# ── S4: buffer warmup ───────────────────────────────────────────────────
if $DO_WARMUP; then
  echo ""
  echo "========== S4: warmup buffer =========="
  .venv/bin/python scripts/warmup_buffer.py \
      --in-dir rollouts/cold_start --out "$BUFFER" --total-capacity 25000
  ls -lh "$BUFFER"
fi

echo ""
echo "========== PIPELINE DONE =========="
echo "  queries      → $QUERIES"
echo "  trajectories → $TRAJ"
echo "  parquet      → $PARQUET_DIR/{train,val}.parquet"
echo "  buffer       → $BUFFER"
