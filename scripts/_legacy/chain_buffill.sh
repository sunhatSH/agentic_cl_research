#!/usr/bin/env bash
# 冷启动 pipeline 全链路：采集 → 借任务 → 补采 → 填 buffer
# 每一步 idempotent（incremental），可安全重跑
set -euo pipefail
cd "$(dirname "$0")/../.."

COMING_SOON="${COMING_SOON:-}"  # 设非空可预览不真正执行

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# ── Step 1: 采集 buffer 任务 ──
log "=== Step 1: buffill 采集 (incremental, 64 并发) ==="
source scripts/env/load_tencent_env.sh >/dev/null 2>&1
if [ -n "$COMING_SOON" ]; then
  log "COMING_SOON 模式，跳过采集"
else
  .venv/bin/python scripts/collect/run_cold_start.py \
    --num-queries 3706 --max-concurrent 64 \
    --actor-model openai/gpt-5 --model-tag gpt5 \
    --actor-impl hermes_structured \
    --taskspecs-dir datasources/taskspecs_w3 \
    --queries datasets/queries_buffer.jsonl \
    --max-turns 20 --hermes-max-turns 90 --slot-timeout 900 \
    --collect-mode incremental \
    2>&1 | tee -a logs/collect/buffill_chain.log
fi

# ── Step 2: filter + borrow + 移除借走任务 ──
log "=== Step 2: filter_and_borrow ==="
bash scripts/pipeline/filter_and_borrow.sh 2>&1 | tee -a logs/collect/buffill_chain.log

# ── Step 3: 补采借来的任务（如果有 borrow.jsonl）──
BORROWED="datasets/queries_borrowed.jsonl"
if [ -f "$BORROWED" ] && [ -s "$BORROWED" ]; then
  N=$(wc -l < "$BORROWED")
  log "=== Step 3: 补采 $N 条借来的任务 ==="
  if [ -z "$COMING_SOON" ]; then
    .venv/bin/python scripts/collect/run_cold_start.py \
      --num-queries "$N" --max-concurrent 64 \
      --actor-model openai/gpt-5 --model-tag gpt5 \
      --actor-impl hermes_structured \
      --taskspecs-dir datasources/taskspecs_w3 \
      --queries "$BORROWED" \
      --max-turns 20 --hermes-max-turns 90 --slot-timeout 900 \
      --collect-mode incremental \
      2>&1 | tee -a logs/collect/buffill_chain.log
  fi
else
  log "=== Step 3: 无借来任务，跳过补采 ==="
fi

# ── Step 4: warmup buffer ──
log "=== Step 4: warmup_buffer ==="
if [ -z "$COMING_SOON" ]; then
  .venv/bin/python scripts/warmup_buffer.py \
    --buffer-jsonl "$AGENTIC_CL_ROLLOUTS/real/trajectory/gpt5/grpo_hermes.jsonl" \
    --output "buffer_dumps/warmup.sqlite" \
    2>&1 | tee -a logs/collect/buffill_chain.log
fi

log "=== DONE: 全链路完成 ==="
echo "DONE" >> logs/collect/buffill_chain.log