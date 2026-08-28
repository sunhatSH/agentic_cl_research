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
#
#   # 从头构建（不用任何层缓存 + 强制拉新 base）——改了 Dockerfile/插件后想确保生效时用：
#   BUILD_NO_CACHE=1 bash scripts/sandbox/build_sandbox_image.sh
#   BUILD_NO_CACHE=1 BUILD_PULL=0 bash scripts/sandbox/build_sandbox_image.sh   # 不拉新 base

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

# 从头构建开关：
#   BUILD_NO_CACHE=1 → --no-cache（不用任何层缓存，每条 RUN/COPY 重跑：clone hermes /
#     pip install / COPY 插件全部重来。改了 Dockerfile/插件后想确保生效时用）。
#   BUILD_PULL（默认随 NO_CACHE=1）→ --pull（强制重新拉 base，避免用本地旧 sandbox-code）。
_BUILD_ARGS=()
if [[ "${BUILD_NO_CACHE:-0}" == "1" ]]; then
  _BUILD_ARGS+=(--no-cache)
  echo "[build_sandbox_image] NO-CACHE 从头构建（每层重跑）"
fi
# NO_CACHE=1 时默认也 --pull；可用 BUILD_PULL=0 关掉。独立 BUILD_PULL=1 也生效。
if [[ "${BUILD_PULL:-${BUILD_NO_CACHE:-0}}" == "1" ]]; then
  _BUILD_ARGS+=(--pull)
  echo "[build_sandbox_image] --pull 强制拉新 base 镜像"
fi

docker build \
  "${_BUILD_ARGS[@]}" \
  -f "${ROOT}/docker/sandbox/Dockerfile" \
  -t "${FULL_TAG}" \
  --platform=linux/amd64 \
  --build-arg "SANDBOX_BASE_IMAGE=${SANDBOX_BASE_IMAGE}" \
  "${ROOT}/docker/sandbox"

echo "[build_sandbox_image] OK -> ${FULL_TAG}"
echo "[build_sandbox_image] next: bash scripts/sandbox/push_sandbox_image.sh"
