#!/usr/bin/env bash
# Timestamped, NON-overwriting smoke collection for the observer / multi-turn loop.
#
# Every run writes to a UNIQUE tag dir  <model>_<type>_<UTC second>  so prior runs are
# never clobbered (the requirement: 模型_任务类型_采集开始时间, 精确到秒). Output:
#   <ROLLOUTS>/smoke/trajectory/<TAG>/grpo_hermes.jsonl
#   <ROLLOUTS>/smoke/debug/observer_report/<TAG>/observer_reports.jsonl
#   logs/smoke/<TAG>.log
#
# Usage:
#   scripts/collect/collect_smoke.sh [N_QUERIES] [ACTOR_MODEL] [TASK_TYPE] [EXTRA run_cold_start args...]
# Examples:
#   scripts/collect/collect_smoke.sh 16                       # 16 queries, gpt-5, type=smoke
#   scripts/collect/collect_smoke.sh 24 openai/gpt-5 iter4     # tag = gpt5_iter4_<ts>
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

N="${1:-16}"
ACTOR_MODEL="${2:-openai/gpt-5}"
TASK_TYPE="${3:-smoke}"
shift $(( $# < 3 ? $# : 3 )) || true   # remaining args pass through to run_cold_start

PY=.venv/bin/python
QUERIES=datasets/queries_w3.jsonl
TASKSPECS=datasources/taskspecs_w3

# model short tag: openai/gpt-5 -> gpt5 ; qwen/qwen3.6-27b -> qwen27b
_short() {
  case "$1" in
    */gpt-5|*gpt-5*)        echo gpt5 ;;
    */qwen3.6-27b|*qwen*)   echo qwen27b ;;
    *) echo "$1" | tr '/:.' '___' ;;
  esac
}
MODEL_SHORT="$(_short "$ACTOR_MODEL")"
# Timestamp precise to the second (UTC, compact ISO): 20260710T111530Z
TS="$(date -u +%Y%m%dT%H%M%SZ)"
TAG="${MODEL_SHORT}_${TASK_TYPE}_${TS}"

mkdir -p logs/smoke
LOG="logs/smoke/${TAG}.log"

source scripts/env/load_tencent_env.sh

echo "[collect_smoke] tag=$TAG  queries=$N  model=$ACTOR_MODEL  log=$LOG"
echo "[collect_smoke] traj -> smoke/trajectory/$TAG/grpo_hermes.jsonl"

# --collect-mode overwrite is safe: TAG dir is unique per run, so nothing pre-exists.
"$PY" scripts/collect/run_cold_start.py \
    --num-queries "$N" --max-concurrent 32 \
    --actor-model "$ACTOR_MODEL" --model-tag "$TAG" \
    --smoke --collect-mode overwrite \
    --taskspecs-dir "$TASKSPECS" --queries "$QUERIES" \
    --max-turns 20 --hermes-max-turns 90 --slot-timeout 900 \
    "$@" 2>&1 | tee "$LOG"

echo "[collect_smoke] DONE tag=$TAG"
