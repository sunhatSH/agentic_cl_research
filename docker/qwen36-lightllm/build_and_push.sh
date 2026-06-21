#!/usr/bin/env bash
# 构建并推送 Qwen3.6-lightllm 训练镜像到商汤天津 registry。
#
# 必须在【带完整 Docker 权限的裸机/VM】上跑，不能在受限 K8s Pod 里 build
# （Pod 缺 CAP_SYS_ADMIN，BuildKit bind mount / legacy unshare 都会失败）。
#
# 用法：
#   NAMESPACE=ccr-devsfttj bash docker/qwen36-lightllm/build_and_push.sh
# 可选 env：
#   IMAGE   镜像名（默认 qwen36-lightllm）
#   TAG     版本（默认 1.0）
#   USERNAME registry 登录名（默认 devsfttj-sunhao4）
#   PUSH=0  只 build 不 push（默认 1=push）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

REGISTRY="registry.cn-tj-01.sensecore.cn"
IMAGE="${IMAGE:-qwen36-lightllm}"
TAG="${TAG:-1.0}"
USERNAME="${USERNAME:-devsfttj-sunhao4}"
NAMESPACE="${NAMESPACE:?必须设置 NAMESPACE=ccr-devsfttj}"
PUSH="${PUSH:-1}"

LOCAL_REF="${IMAGE}:${TAG}"
REMOTE_REF="${REGISTRY}/${NAMESPACE}/${IMAGE}:${TAG}"

command -v docker >/dev/null 2>&1 || { echo "ERROR: 本机无 docker"; exit 1; }
docker info >/dev/null 2>&1 || { echo "ERROR: docker daemon 未运行"; exit 1; }

echo "[build] $LOCAL_REF  (context=$ROOT_DIR)"
docker build -t "$LOCAL_REF" -f "$SCRIPT_DIR/Dockerfile" "$ROOT_DIR"

if [[ "$PUSH" == "1" ]]; then
  echo "[login] $REGISTRY as $USERNAME"
  docker login "$REGISTRY" --username "$USERNAME"
  echo "[tag]  $LOCAL_REF -> $REMOTE_REF"
  docker tag "$LOCAL_REF" "$REMOTE_REF"
  echo "[push] $REMOTE_REF"
  docker push "$REMOTE_REF"
  echo "[done] pushed: $REMOTE_REF"
else
  echo "[done] built only (PUSH=0): $LOCAL_REF"
fi
