#!/usr/bin/env bash
# W3 完整冷采集 pipeline (GPT-5 + Qwen3.6-27B 两份)
# 数据源: data/taskspecs_w3  |  queries: datasets/queries_w3.jsonl
# 只跑到 S3 parquet (这台机器不做 S4 warmup / 训练)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

TASKSPECS=data/taskspecs_w3
QUERIES=datasets/queries_w3.jsonl
PY=.venv/bin/python
SRC=/mnt/afs_toolcall/wujian1/Projects/seed2traj/taskspecs_w3_full

log() { echo -e "\n========== $* =========="; }

# ── S0: 复制数据(幂等, 已存在则跳过) ─────────────────────────────────
log "S0: 复制数据 → $TASKSPECS"
mkdir -p "$TASKSPECS"
if [ "$(ls "$TASKSPECS" | wc -l)" -lt "$(ls "$SRC" | wc -l)" ]; then
  cp -rn "$SRC"/. "$TASKSPECS"/
fi
echo "taskspecs: $(ls "$TASKSPECS" | wc -l) entries"

# ── S1: 打标 + 人设 + 可跑性 (合并单次 LLM, 32并发) ──────────────────
log "S1: 打标+人设+可跑性 (32并发)"
$PY scripts/run_cold_start.py --generate --no-collect \
    --classify-workers 32 \
    --taskspecs-dir "$TASKSPECS" --queries "$QUERIES"
N=$(wc -l < "$QUERIES")
echo "queries: $N rows"

# ── 采集凭证 ────────────────────────────────────────────────────────
source scripts/load_tencent_env.sh

ROLLOUTS=/mnt/afs_toolcall/sunhao4/agentic_cl_rollouts

collect_model() {  # $1=model  $2=model-tag  $3=parquet-dir
  local MODEL="$1" TAG="$2" PQ="$3"
  local TRAJ="$ROLLOUTS/real/trajectory/$TAG/grpo_hermes.jsonl"
  log "S2: 冷采集 [$MODEL] $N queries, 32并发 → real/trajectory/$TAG/"
  $PY scripts/run_cold_start.py \
      --num-queries "$N" --max-concurrent 32 \
      --actor-model "$MODEL" --model-tag "$TAG" \
      --taskspecs-dir "$TASKSPECS" --queries "$QUERIES" \
      --max-turns 20 --hermes-max-turns 30 --slot-timeout 900

  log "S2b: 后清洗 [$MODEL]"
  bin/strip_zw < "$TRAJ" \
    | bin/filter_garbled --garble-threshold 0.05 > "$TRAJ.clean"
  mv "$TRAJ.clean" "$TRAJ"

  log "S3: parquet [$MODEL] → $PQ"
  $PY scripts/trajectory_to_parquet.py --input "$TRAJ" --out-dir "$PQ"
}

# ── S2/S2b/S3: GPT-5 then Qwen ──────────────────────────────────────
collect_model "openai/gpt-5"     gpt5     datasets/w3_gpt5
collect_model "qwen/qwen3.6-27b" qwen27b  datasets/w3_qwen27b

log "ALL DONE"
echo "queries : $QUERIES"
echo "GPT5    : $ROLLOUTS/real/trajectory/gpt5/grpo_hermes.jsonl → datasets/w3_gpt5/{train,val}.parquet"
echo "Qwen    : $ROLLOUTS/real/trajectory/qwen27b/grpo_hermes.jsonl → datasets/w3_qwen27b/{train,val}.parquet"
