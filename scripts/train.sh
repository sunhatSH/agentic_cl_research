#!/usr/bin/env bash
# Launch a single CL training run.
#
# Usage:
#   bash scripts/train.sh configs/b1.yaml
#   bash scripts/train.sh configs/r4.yaml --resume-from ckpts/r4-step-50

set -euo pipefail

CONFIG="${1:?Usage: $0 <config.yaml> [extra args]}"
shift

python -m trainer.cl_main --config "$CONFIG" "$@"
