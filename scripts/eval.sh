#!/usr/bin/env bash
# Run ClawEval over a checkpoint.
#
# Usage:
#   bash scripts/eval.sh ckpts/r4-step-100

set -euo pipefail

CKPT="${1:?Usage: $0 <ckpt-path>}"
shift

python -m eval.run_eval --ckpt "$CKPT" "$@"
