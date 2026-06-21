#!/usr/bin/env bash
# Sandbox ops entrypoint — add subcommands here.
#
# Usage:
#   bash docker/sandbox/ops/ops.sh query
#   bash docker/sandbox/ops/ops.sh query --tool agentic-cl-code-interpreter
set -euo pipefail

OPS_DIR="$(cd "$(dirname "$0")" && pwd)"
CMD="${1:-}"

if [[ -z "$CMD" ]]; then
  echo "Usage: bash docker/sandbox/ops/ops.sh <command> [args...]"
  echo ""
  echo "Commands:"
  echo "  query   查询沙箱 Tool / Instance（默认）"
  echo ""
  echo "Examples:"
  echo "  bash docker/sandbox/ops/ops.sh query"
  echo "  bash docker/sandbox/ops/ops.sh query --json"
  exit 1
fi

shift || true

case "$CMD" in
  query | status | ps)
    exec bash "${OPS_DIR}/query.sh" "$@"
    ;;
  *)
    echo "[ops] unknown command: $CMD" >&2
    exit 1
    ;;
esac
