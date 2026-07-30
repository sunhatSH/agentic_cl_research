#!/usr/bin/env bash
# train_manual_2node.sh — 手动组 Ray 的 2 节点(16卡)训练启动器.
#
# 用途：裸机集群(无 SenseCore PyTorch 编排,无 SENSECORE_PYTORCH_* 注入)手动组多机。
# master(RANK=0) 起 ray head + 训练；worker(RANK=1) 连 head。两台【各跑一次】本脚本,
# 靠 RANK 区分。复用 scripts/_train_impl.sh 的多机分支(rank0 起 head→gloo barrier→
# master 训/worker --block),本脚本只负责把裸机缺的多机 env 手动 export 好再调 _train_impl。
# 【不改】train.sh / _train_impl.sh(16卡最终脚本)。
#
# 用法（两台各跑一次，先 master 后 worker，30s 内先后启动即可）：
#   # master 节点(内网 IP 10.120.5.165):
#   RANK=0 bash scripts/train_manual_2node.sh
#   # worker 节点:
#   RANK=1 bash scripts/train_manual_2node.sh
#
# 可覆盖：MASTER_ADDR / MASTER_PORT / CONFIG。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── 手动多机 env（裸机无平台注入，全部在此固定/可覆盖）──────────────────────
export RANK="${RANK:?必须指定 RANK：master=0 / worker=1}"
export MASTER_ADDR="${MASTER_ADDR:-10.120.5.165}"   # master 内网 IP(hostname -i)
export MASTER_PORT="${MASTER_PORT:-29503}"
export NNODES="${NNODES:-2}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
export WORLD_SIZE="${WORLD_SIZE:-$NNODES}"           # 节点数(gloo rendezvous 用)

# ── 跳过 check_train_env 的 pip 安装（定制 /opt/conda 已就位，手动 import verl 已验证 OK）──
# 2026-07-29 集群踩坑：check_train_env 按 PINNED 列表 pip install 会覆盖定制环境里
# 和 flash-attn 配套的版本 → flash_attn.flash_attn_interface 失效 → transformer_engine
# .pytorch import 崩 → 训练起步即挂。环境本就配套能跑,不该重装,故直接跳过安装步骤。
export SKIP_ENV_INSTALL="${SKIP_ENV_INSTALL:-1}"

# ── ray 绑到 10.120.5.x 内网(裸机双网卡:10.123.104.x + 10.120.5.x,worker 连的是后者)──
# _train_impl 多机分支的 `ray start --head`/`--address` 不显式传 IP;ray 自动探测多网卡时
# 可能选错网段 → worker 连不上 head。ray 2.x 认 RAY_OVERRIDE_NODE_IP_ADDRESS 强制 node IP。
# master 绑自己(MASTER_ADDR);worker 绑自己的 10.120.5.x(用 hostname -i 取,同网段)。
if [ "$RANK" = "0" ]; then
  export RAY_OVERRIDE_NODE_IP_ADDRESS="$MASTER_ADDR"
else
  # worker：取本机 10.120.5.x 网段 IP(与 master 同网段,保证 ray 通信走对网卡)
  _wip="$(hostname -i 2>/dev/null | tr ' ' '\n' | grep -E '^10\.120\.5\.' | head -1)"
  [ -n "$_wip" ] && export RAY_OVERRIDE_NODE_IP_ADDRESS="$_wip"
fi

# ── 依赖路径（AFS 共用，与开发机同）──────────────────────────────────────
VENV="${VENV:-/opt/conda}"
VERL_DIR="/mnt/afs_toolcall/sunhao4/dependencies/verl"
LIGHTLLM_DIR="/mnt/afs_toolcall/sunhao4/workspace/LightLLM"
CONFIG="${CONFIG:-$ROOT_DIR/configs/run/b1_9b_16gpu.yaml}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[manual_2node] RANK=$RANK  16GPU(${NNODES}节点×${N_GPUS_PER_NODE}卡)"
echo "[manual_2node] MASTER=$MASTER_ADDR:$MASTER_PORT  WORLD_SIZE=$WORLD_SIZE"
echo "[manual_2node] RAY_NODE_IP=${RAY_OVERRIDE_NODE_IP_ADDRESS:-auto}"
echo "[manual_2node] CONFIG=$CONFIG"
echo "[manual_2node] 角色：$([ "$RANK" = 0 ] && echo 'MASTER(起 ray head + 训练)' || echo 'WORKER(连 head 并 block)')"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 调 _train_impl.sh(NNODES>1 触发多机分支)。SP/batch/长度以 b1_9b_16gpu.yaml 为准 ──
exec bash "$SCRIPT_DIR/_train_impl.sh" \
  --config "$CONFIG" \
  --nnodes "$NNODES" \
  --gpus-per-node "$N_GPUS_PER_NODE" \
  --rollout-tp 2 \
  --ulysses-sp 4 \
  --train-batch 32 \
  --ppo-mini 32 \
  --gpu-mem-util 0.75 \
  --verl-dir "$VERL_DIR" \
  --lightllm-dir "$LIGHTLLM_DIR" \
  --venv "$VENV"
