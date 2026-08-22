#!/usr/bin/env bash
# Background collection: hermes_structured + Claude model, 64 concurrency
# Fill bucket floors for replay buffer warmup.
# Run in tmux: tmux new -s collect && bash scripts/collect/collect_floor.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"

# Load credentials
set -a
source "$ROOT/docker/sandbox/tencent.env"
source "$ROOT/docker/sandbox/runtime.env"
source "$ROOT/.env"
set +a

cd "$ROOT"
export PYTHONPATH="./src:.:${PYTHONPATH:-}"

OUT_DIR="${OUT_DIR:-/mnt/afs_toolcall/sunhao4/agentic_cl_rollouts/real/trajectory/claude_struct}"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT_DIR="$OUT_DIR/$TIMESTAMP"
mkdir -p "$OUT_DIR"

echo "=== Hermes Structured Collection ==="
echo "Model: claude-sonnet-4-6 (via tokenhub -> Anthropic)"
echo "Concurrency: 64"
echo "Queries: $ROOT/datasets/queries.jsonl"
echo "Output: $OUT_DIR"
echo "Started at: $(date)"
echo ""

.venv/bin/python scripts/collect/sandbox_grpo_collect.py \
  --actor hermes \
  --mode collect \
  --backend e2b \
  --template node-python-hermes-26-7-1 \
  --queries "$ROOT/datasets/queries.jsonl" \
  --actor-model "claude-sonnet-4-6" \
  --num-queries 0 \
  --max-turns 8 \
  --hermes-max-turns 30 \
  --max-concurrent 64 \
  --slot-timeout 300 \
  --out-dir "$OUT_DIR" \
  2>&1 | tee "$OUT_DIR/collect.log"

echo ""
echo "=== Done at $(date) ==="
echo "Output: $OUT_DIR"
