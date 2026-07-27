#!/usr/bin/env bash
# 本机开发环境：把 PYTHONPATH 指向 AFS 共享盘的 verl / LightLLM 源码 + 项目根，
# 与 train.sh 的 PYTHONPATH 同源、同序。
#
# 用法：
#   source scripts/env/dev_env.sh
#
# 为何需要：
#   - verl 走 AFS 源码（workspace/verl，带 lightllm-agent 增量 lightllm_rollout / agent_loop /
#     fully_async_policy，pip 装的原版 verl==0.8.0 没有这些）。
#   - 代码里 import verl 都是 lazy（函数内 import），本机单测不触发，所以本机
#     不装 ray 也能跑单测（47+ passed）。
#   - source 本脚本后，任何【装了 ray/torch 等 verl 依赖】的环境（集群训练机、
#     或本机补装依赖后）import verl 会命中这份 AFS 源码，与集群完全一致。
#
# 镜像（qwen36-lightllm）已含 ray / torch / tensordict / hydra 等 verl 全栈依赖，
# 故在镜像容器里 source 本脚本即可 import verl。

set -euo pipefail

_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
_VERL_DIR="/mnt/afs_toolcall/sunhao4/workspace/verl"
_LIGHTLLM_DIR="/mnt/afs_toolcall/sunhao4/workspace/LightLLM"

# 校验源码在位（AFS 没挂载时给出明确提示，而不是静默设个坏路径）
if [[ ! -d "$_VERL_DIR/verl" ]]; then
  echo "[dev_env] WARN: $_VERL_DIR/verl 不存在 —— AFS 未挂载？仍设 PYTHONPATH，但 import verl 会失败" >&2
fi
if [[ ! -d "$_LIGHTLLM_DIR" ]]; then
  echo "[dev_env] WARN: $_LIGHTLLM_DIR 不存在 —— LightLLM 源码缺失（lightllm rollout 需要）" >&2
fi

# 顺序与 train.sh 一致：LightLLM : verl : 项目根
export PYTHONPATH="$_LIGHTLLM_DIR:$_VERL_DIR:$_ROOT:${PYTHONPATH:-}"

echo "[dev_env] PYTHONPATH set:"
echo "  LightLLM : $_LIGHTLLM_DIR"
echo "  verl     : $_VERL_DIR"
echo "  project  : $_ROOT"
echo "[dev_env] 提示：本机 cold env 缺 ray 等，import verl 仍会报 ray；"
echo "           集群/镜像容器（含 ray）source 本脚本即可 import verl 命中 AFS 源码。"
