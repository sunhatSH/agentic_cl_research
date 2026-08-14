#!/usr/bin/env bash
# Full cold-start pipeline + data cleaning.
# Each stage is incremental — safe to resume if interrupted.
#
# Stages:
#   S1  --generate            (re)generate queries.jsonl (classify + persona + runnability)
#   S2  --collect             multi-turn collection (actor+observer+questioner)
#   S2b                       garble-character filter (drops bad trajectories)
#   S3  --parquet             trajectories → train.parquet + val.parquet
#   S4  --warmup              trajectories → buffer.sqlite (训练前预填)
#
#   --all-collect             S1 + S1b + S2 + S3 + S4
#   --full                    same as --all-collect

set -e
cd "$(git rev-parse --show-toplevel)"

DO_GENERATE=false
DO_FILTER=false
DO_COLLECT=false
DO_PARQUET=false
DO_WARMUP=false

for arg in "$@"; do
  case "$arg" in
    --full|--all-collect) DO_GENERATE=true; DO_FILTER=true; DO_COLLECT=true; DO_PARQUET=true; DO_WARMUP=true ;;
    --generate) DO_GENERATE=true ;;
    --filter) DO_FILTER=true ;;
    --collect) DO_COLLECT=true ;;
    --parquet) DO_PARQUET=true ;;
    --warmup) DO_WARMUP=true ;;
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

# ── S1: queries generation (classify + persona + runnability) ────────
if $DO_GENERATE; then
  echo "========== S1: 打桶 + 人设 + 可跑性 =========="
  .venv/bin/python scripts/collect/run_cold_start.py --generate --no-collect --classify-workers 32
  wc -l "$QUERIES"
fi

# ── S2: collection (incl. ZW strip + garble filter) ─────────────────────
if $DO_COLLECT; then
  echo ""
  N=$(wc -l < "$QUERIES")
  echo "========== S2: 多轮采集 ($N queries, 32 并发) =========="
  source scripts/env/load_tencent_env.sh
  .venv/bin/python scripts/collect/run_cold_start.py \
      --num-queries "$N" --max-concurrent 32 \
      --max-turns 20 --hermes-max-turns 30 --slot-timeout 900

  # S2b: ZW strip + garbled filter (C++)
  CLEANED="${TRAJ%.jsonl}_clean.jsonl"
  echo ""
  echo "========== S2b: ZW-清洗 + 脏字符过滤 (C++) =========="
  bin/strip_zw < "$TRAJ" | bin/filter_garbled --garble-threshold 0.05 > "$CLEANED" 2> "${TRAJ%.jsonl}_dropped.log"
  mv "$CLEANED" "$TRAJ"
  grep -c "^# total" "${TRAJ%.jsonl}_dropped.log" || true
  wc -l "$TRAJ"
fi

# ── S3: parquet ─────────────────────────────────────────────────────────
if $DO_PARQUET; then
  echo ""
  echo "========== S3: 轨迹 → parquet =========="
  .venv/bin/python scripts/data/trajectory_to_parquet.py \
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
echo ""
echo "  dropped log  → ${TRAJ%.jsonl}.dropped.jsonl"
