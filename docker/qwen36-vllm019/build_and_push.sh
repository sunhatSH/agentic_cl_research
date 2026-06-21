#!/usr/bin/env bash
# 构建并推送 Qwen3.6-vllm0.19 镜像到商汤天津 registry。
#
# 在【带 docker daemon 的机器】上跑（本训练机无 docker）。Dockerfile 用相对路径，
# 故需在仓库根目录执行，或让脚本自己 cd 到根目录（已处理）。
#
# 用法：
#   NAMESPACE=<你的命名空间> bash docker/qwen36-vllm019/build_and_push.sh
# 可选 env：
#   IMAGE   镜像名（默认 qwen36-vllm019）
#   TAG     版本（默认 1.0）
#   USERNAME registry 登录名（默认 devsfttj-sunhao4）
#   PUSH=0  只 build 不 push（默认 1=push）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

REGISTRY="registry.cn-tj-01.sensecore.cn"
IMAGE="${IMAGE:-qwen36-vllm019}"
TAG="${TAG:-1.0}"
USERNAME="${USERNAME:-devsfttj-sunhao4}"
NAMESPACE="${NAMESPACE:?必须设置 NAMESPACE=<你的命名空间>}"
PUSH="${PUSH:-1}"

LOCAL_REF="${IMAGE}:${TAG}"
REMOTE_REF="${REGISTRY}/${NAMESPACE}/${IMAGE}:${TAG}"

command -v docker >/dev/null 2>&1 || { echo "ERROR: 本机无 docker，请在带 docker 的机器上跑"; exit 1; }

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
