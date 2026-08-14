#!/usr/bin/env bash
# Load local training credentials (gitignored .env only).
#
# Usage:
#   source scripts/env/load_training_env.sh
#   bash scripts/train.sh configs/run/b1_9b_16gpu.yaml

_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
_ENV="${_ROOT}/.env"

if [[ -f "$_ENV" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$_ENV"
  set +a
  echo "[load_training_env] loaded .env"
else
  echo "[load_training_env] skip missing .env"
fi

if [[ -z "${SWANLAB_API_KEY:-}" ]]; then
  echo "[load_training_env] WARN: SWANLAB_API_KEY empty — edit .env or export manually"
else
  echo "[load_training_env] SWANLAB_API_KEY present"
fi
