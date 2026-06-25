#!/usr/bin/env bash
# Push the custom sandbox image to Tencent CCR.
#
# Usage:
#   bash scripts/push_sandbox_image.sh
#
# Requires docker login first, e.g.:
#   docker login tcr-rl.tencentcloudcr.com

set -euo pipefail

ROOT=/
ENV_FILE=/docker/sandbox/image.env

if [[ -f  ]]; then
  set -a
  # shellcheck source=/dev/null
  source 
  set +a
fi

: tcr-rl.tencentcloudcr.com  # 企业版 TCR
: REPLACE_WITH_YOUR_NAMESPACE
: agentic-cl-sandbox
: v1

FULL_TAG=tcr-rl.tencentcloudcr.com/REPLACE_WITH_YOUR_NAMESPACE/agentic-cl-sandbox:v1

if ! command -v docker >/dev/null 2>&1; then
  echo [push_sandbox_image]
