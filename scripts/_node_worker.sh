#!/usr/bin/env bash
# Per-node worker for 8-node cold rollout collection (called by launch_8node.sh via ssh).
# On THIS node: start local vllm (Qwen3.6-27B, TP=8) -> wait ready ->
# run collect_rollout.py for THIS node's shard (--actor local --backend e2b).
#
# Args (env): NODE_RANK (required), NUM_NODES (default 8)
set -euo pipefail
ROOT_DIR=/mnt/afs_toolcall/sunhao4/agentic_cl_research
cd "$ROOT_DIR"

NODE_RANK="${NODE_RANK:?NODE_RANK required}"
NUM_NODES="${NUM_NODES:-8}"
MODEL_PATH="${MODEL_PATH:-/mnt/afs_toolcall/sunhao4/models/Qwen3.6-27B}"
RAW="${RAW:-/mnt/afs_toolcall/sunhao4/datasets/juxiaolong_prefix/source/_stage_prefix_pass.jsonl}"
LIMIT="${LIMIT:-10000}"
CONCURRENCY="${CONCURRENCY:-8}"
PORT="${PORT:-8000}"; ACTOR_TP="${ACTOR_TP:-8}"; SERVED="${SERVED:-cold-actor}"
MAX_LEN="${MAX_LEN:-32768}"
# Only ds32_env's vllm 0.16rc supports the qwen3_5 (Qwen3.6) architecture.
PY="${PY:-/mnt/afs_code/ds32_env/bin/python}"
OUT_BASE="${OUT_BASE:-$ROOT_DIR/data/mock/rollouts}"  # mock = 可行性验证数据，与正式数据隔离
LOGD="$ROOT_DIR/logs/cold"
mkdir -p "$LOGD" "$OUT_BASE/local"

# --- creds: E2B (real sandbox) + remote observer/questioner ---------------
[[ -z "${E2B_API_KEY:-}" && -f "$ROOT_DIR/docker/sandbox/tencent.env" ]] && { set -a; source "$ROOT_DIR/docker/sandbox/tencent.env"; set +a; }
APODEX_ENV="${APODEX_ENV:-/mnt/afs_toolcall/sunhao4/apodex_research/configs/env_deepseek_v4_pro.env}"
REMOTE_BASE="${REMOTE_BASE:-https://openai.sufy.com/v1}"
REMOTE_KEY="${REMOTE_KEY:-${SUFY_API_KEY:-$(grep -h OPENAI_API_KEY "$APODEX_ENV" 2>/dev/null | head -1 | cut -d= -f2)}}"
export OBSERVER_API_BASE="$REMOTE_BASE" OBSERVER_MODEL="${OBSERVER_MODEL_ID:-openai/gpt-5-mini}"     OBSERVER_API_KEY="$REMOTE_KEY"
export USERSIM_API_BASE="$REMOTE_BASE"  USERSIM_MODEL="${QUESTIONER_MODEL_ID:-anthropic/claude-sonnet-5}" USERSIM_API_KEY="$REMOTE_KEY"

# --- python env (ds32_env vllm0.16 is self-contained; only add repo to path) ---
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
# vllm 子进程即时编译 kernel 时调命令行 `ninja` 等工具，走 PATH；ds32_env/bin 必须在 PATH，
# 否则报 FileNotFoundError: 'ninja' → worker 崩 → vllm 连锁 shutdown（8 机空跑根因之一）。
export PATH="$(dirname "$PY"):$PATH"
# 缓存/锁目录必须放本地盘：HOME 在 quarkfs(FUSE)，vllm/triton 编译用 filelock(fcntl.flock)
# 在 FUSE 上不可靠 → 锁文件 FileNotFoundError → worker 崩（8 机崩溃真根因）。指到本地 /tmp。
export VLLM_CACHE_ROOT="/tmp/vcache_r${NODE_RANK}/vllm"
export TRITON_CACHE_DIR="/tmp/vcache_r${NODE_RANK}/triton"
export TORCHINDUCTOR_CACHE_DIR="/tmp/vcache_r${NODE_RANK}/inductor"
export XDG_CACHE_HOME="/tmp/vcache_r${NODE_RANK}/xdg"
# ★真根因：flashinfer JIT 编译缓存默认在 HOME(quarkfs)，其 FileLock(fcntl.flock) 在 FUSE 上
# 崩 → worker 崩 → vllm shutdown。FLASHINFER_WORKSPACE_BASE 指本地盘 /tmp 才是对症修复。
export FLASHINFER_WORKSPACE_BASE="/tmp/vcache_r${NODE_RANK}/flashinfer"
export FLASHINFER_CUBIN_DIR="/tmp/vcache_r${NODE_RANK}/flashinfer_cubin"
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME" "$FLASHINFER_WORKSPACE_BASE" "$FLASHINFER_CUBIN_DIR"

echo "[node $NODE_RANK] $(hostname) starting vllm ($MODEL_PATH TP=$ACTOR_TP) ..."
"$PY" -m vllm.entrypoints.openai.api_server --model "$MODEL_PATH" \
  --served-model-name "$SERVED" --port "$PORT" --tensor-parallel-size "$ACTOR_TP" \
  --max-model-len "$MAX_LEN" --dtype bfloat16 --gpu-memory-utilization 0.90 \
  > "$LOGD/vllm_r${NODE_RANK}.log" 2>&1 &
VLLM_PID=$!
cleanup() { kill "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "[node $NODE_RANK] waiting vllm ready ..."
for i in $(seq 1 600); do
  curl -sS -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo "[node $NODE_RANK] vllm ready (${i}s)"; break; }
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "[node $NODE_RANK] vllm DIED:"; tail -20 "$LOGD/vllm_r${NODE_RANK}.log"; exit 3; }
  sleep 2
done

echo "[node $NODE_RANK] collecting shard $NODE_RANK/$NUM_NODES (e2b real sandbox) ..."
"$PY" "$ROOT_DIR/scripts/collect_rollout.py" \
  --raw "$RAW" --actor local --actor-base "http://127.0.0.1:$PORT/v1" --actor-model "$SERVED" --actor-key sk-local \
  --limit "$LIMIT" --backend e2b --concurrency "$CONCURRENCY" \
  --num-nodes "$NUM_NODES" --node-rank "$NODE_RANK" \
  --out-dir "$OUT_BASE/local" 2>&1 | tee "$LOGD/rollout_r${NODE_RANK}.log"
echo "[node $NODE_RANK] DONE"
