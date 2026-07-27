#!/usr/bin/env bash
# Build the Agent Runtime custom sandbox image (linux/amd64).
#
# Prerequisites (when account is ready):
#   1. docker installed
#   2. docker login to Tencent CCR (see doc/sandbox/Sandbox_腾讯云操作手册.md §4.4)
#   3. docker pull tcr-rl.tencentcloudcr.com/agentos-cl-namespace/sandbox-code:latest
#
# Usage:
#   cp docker/sandbox/image.env.example docker/sandbox/image.env
#   # edit image.env
#   bash scripts/sandbox/build_sandbox_image.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# Prefer load_tencent_env.sh (image.env + tencent.env); fallback image.env only
if [[ -f "${ROOT}/scripts/env/load_tencent_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/scripts/env/load_tencent_env.sh" 2>/dev/null || true
fi
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
: "${SANDBOX_BASE_IMAGE:=tcr-rl.tencentcloudcr.com/agentos-cl-namespace/sandbox-code:latest}"  # 企业版 TCR

FULL_TAG="${CCR_REGISTRY}/${CCR_NAMESPACE}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "[build_sandbox_image] validating Dockerfile constraints..."
bash "${ROOT}/scripts/sandbox/validate_sandbox_dockerfile.sh"

if ! command -v docker >/dev/null 2>&1; then
  echo "[build_sandbox_image] ERROR: docker not found. Install Docker or run on a machine with docker."
  exit 1
fi

echo "[build_sandbox_image] building ${FULL_TAG} (platform=linux/amd64)"
echo "[build_sandbox_image] base image: ${SANDBOX_BASE_IMAGE}"

docker build \
  -f "${ROOT}/docker/sandbox/Dockerfile" \
  -t "${FULL_TAG}" \
  --platform=linux/amd64 \
  --build-arg "SANDBOX_BASE_IMAGE=${SANDBOX_BASE_IMAGE}" \
  "${ROOT}/docker/sandbox"

echo "[build_sandbox_image] OK -> ${FULL_TAG}"
echo "[build_sandbox_image] next: bash scripts/sandbox/push_sandbox_image.sh"
