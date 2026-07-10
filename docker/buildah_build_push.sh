#!/usr/bin/env bash
# 用 buildah 构建并推送镜像（无需 Docker daemon，可在开发机直接跑）
#
# 前置（只需做一次）：
#   1. 安装 buildah:  sudo apt-get install -y buildah
#   2. 登录 registry（从 Mac 的 ~/.docker/config.json 获取密码）：
#        buildah login --username <UIN> --password <TOKEN> <REGISTRY>
#
# 用法：
#   # 沙箱镜像（AGS）
#   bash docker/buildah_build_push.sh sandbox
#
#   # qwen36 镜像
#   bash docker/buildah_build_push.sh qwen36-lightllm
#
# 环境变量：
#   TAG        镜像 tag（默认 v1）
#   PUSH=0     只 build 不 push
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- 选择镜像 ----
IMAGE_DIR="${1:-}"
if [[ -z "$IMAGE_DIR" ]]; then
  echo "Usage: $0 <image_dir>" >&2
  echo "  Available:" >&2
  for d in "$SCRIPT_DIR"/*/; do
    name=$(basename "$d")
    [[ -f "$d/Dockerfile" ]] && echo "    $name" >&2
  done
  exit 1
fi

DOCKERFILE="${SCRIPT_DIR}/${IMAGE_DIR}/Dockerfile"
CONTEXT="${SCRIPT_DIR}/${IMAGE_DIR}"
ENV_FILE="${SCRIPT_DIR}/${IMAGE_DIR}/image.env"
: "${TAG:=v1}"
PUSH="${PUSH:-1}"

if [[ ! -f "$DOCKERFILE" ]]; then
  echo "ERROR: $DOCKERFILE not found" >&2
  exit 1
fi

# ---- 从 image.env 读配置 ----
IMAGE_NAME="${IMAGE_DIR}"
CCR_REGISTRY=""
CCR_NAMESPACE=""
if [[ -f "$ENV_FILE" ]]; then
  set -a; source "$ENV_FILE"; set +a
fi
: "${CCR_REGISTRY:=tcr-rl.tencentcloudcr.com}"
: "${CCR_NAMESPACE:=agentos-cl-namespace}"
: "${IMAGE_NAME:=${IMAGE_DIR}}"

LOCAL_REF="${IMAGE_NAME}:${TAG}"
REMOTE_REF="${CCR_REGISTRY}/${CCR_NAMESPACE}/${IMAGE_NAME}:${TAG}"

command -v buildah >/dev/null 2>&1 || { echo "ERROR: 需要 buildah: sudo apt-get install -y buildah"; exit 1; }

echo "============================================"
echo "  BUILDAH BUILD & PUSH"
echo "  image:    ${LOCAL_REF}"
echo "  remote:   ${REMOTE_REF}"
echo "  dockerfile: ${DOCKERFILE}"
echo "============================================"

buildah build-using-dockerfile \
  --tag "${LOCAL_REF}" \
  -f "$DOCKERFILE" \
  "$CONTEXT"
echo "[buildah] OK -> ${LOCAL_REF}"

if [[ "$PUSH" == "1" ]]; then
  buildah tag "${LOCAL_REF}" "${REMOTE_REF}"
  buildah push "${REMOTE_REF}"
  echo "[buildah] PUSHED -> ${REMOTE_REF}"
else
  echo "[buildah] PUSH=0, skip push"
fi
echo "[buildah] DONE"
