#!/usr/bin/env bash
# Run ClawEval over a checkpoint,按桶产出总账+每桶细则。
#
# Usage:
#   # 过程评测（写进实验自包含目录 runs/<phase>/<exp>/eval/）：
#   bash scripts/serve/eval.sh ckpts/r4 --exp-dir runs/phase3/r4 --baseline-ckpt ckpts/baseline
#   # 旧式（写 eval/results/<run_id>/）：
#   bash scripts/serve/eval.sh ckpts/r4-step-100
#
# 产出（见 runs/README.md）：
#   scores.json          总账：每桶 reward/pass_rate/n_tasks/三维均分 + vs baseline 两端对比
#   per_bucket/<桶>.jsonl 细则：桶内逐题 {task_id, passed_all, safety, completion, robustness, reward}

set -euo pipefail

CKPT="${1:?Usage: $0 <ckpt-path> [--exp-dir runs/<phase>/<exp>] [--baseline-ckpt <path>] ...}"
shift

python -m eval.run_eval --ckpt "$CKPT" "$@"
