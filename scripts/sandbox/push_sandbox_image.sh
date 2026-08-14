#!/usr/bin/env bash
# Push the custom sandbox image to Tencent TCR (enterprise).
#
# Usage:
#   bash scripts/sandbox/push_sandbox_image.sh
#
# Requires docker login first, e.g.:
#   docker login tcr-rl.tencentcloudcr.com

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="${ROOT}/docker/sandbox/image.env"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +a
fi

: "${CCR_REGISTRY:=tcr-rl.tencentcloudcr.com}"  # 企业版 TCR
: "${CCR_NAMESPACE:=REPLACE_WITH_YOUR_NAMESPACE}"
: "${IMAGE_NAME:=agentic-cl-sandbox}"
: "${IMAGE_TAG:=v1}"

FULL_TAG="${CCR_REGISTRY}/${CCR_NAMESPACE}/${IMAGE_NAME}:${IMAGE_TAG}"

if ! command -v docker >/dev/null 2>&1; then
  echo "[push_sandbox_image] ERROR: docker not found."
  exit 1
fi

echo "[push_sandbox_image] pushing ${FULL_TAG}"
docker push "${FULL_TAG}"
echo "[push_sandbox_image] OK"
echo "[push_sandbox_image] update configs/sandbox_tool.json CustomConfiguration.Image to: ${FULL_TAG}"
