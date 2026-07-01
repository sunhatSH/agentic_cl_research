#!/usr/bin/env bash
# Simple upstream healthcheck used by the on-call runbook.
set -euo pipefail
URL="${1:-http://127.0.0.1:9000/healthz}"
if curl -fsS --max-time 5 "${URL}" >/dev/null; then
  echo "OK ${URL}"
else
  echo "FAIL ${URL}" >&2
  exit 1
fi
