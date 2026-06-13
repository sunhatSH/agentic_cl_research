#!/usr/bin/env bash
# Multi-turn user-sim rollout collection launcher (observer + questioner, NO reward).
#
# Two actors, data kept SEPARATE:
#   local  : local vllm serving Qwen3.6-27B            -> $OUT_BASE/local/
#   remote : remote GPT (tokenhub) gpt-5               -> $OUT_BASE/remote/
# Agents (REQUIRED, remote, 3 DIFFERENT models):
#   actor(remote)= gpt-5      observer = gpt-4.1-mini   questioner = claude-sonnet-4-6
#
# Remote API (key + base) is sourced from apodex_research (the only place a real
# tokenhub key lives). If the remote endpoint is unreachable -> ABORT (this stage
# needs observer+questioner; cannot proceed). E2B creds from docker/sandbox/tencent.env.
#
# Usage:
#   bash scripts/collect_rollout.sh            # runs both actors
#   ACTORS="remote" bash scripts/collect_rollout.sh   # only remote
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- config (override via env) -------------------------------------------
MODEL_PATH="${MODEL_PATH:-/mnt/afs_toolcall/sunhao4/models/Qwen3.6-27B}"
RAW="${RAW:-/mnt/afs_toolcall/sunhao4/datasets/juxiaolong_prefix/source/_stage_prefix_pass.jsonl}"
LIMIT="${LIMIT:-10000}"
CONCURRENCY="${CONCURRENCY:-8}"
BACKEND="${BACKEND:-e2b}"
ACTORS="${ACTORS:-local remote}"
OUT_BASE="${OUT_BASE:-$ROOT_DIR/data/mock/rollouts}"  # mock = 可行性验证数据，与正式数据隔离

# remote API source (apodex). REMOTE_KEY/REMOTE_BASE override allowed.
APODEX_ENV="${APODEX_ENV:-/mnt/afs_toolcall/sunhao4/apodex_research/configs/env_deepseek_v4_pro.env}"
REMOTE_BASE="${REMOTE_BASE:-https://tokenhub.sensetime.com/v1}"
REMOTE_KEY="${REMOTE_KEY:-$(grep -h OPENAI_API_KEY "$APODEX_ENV" 2>/dev/null | head -1 | cut -d= -f2)}"

# three DIFFERENT models
REMOTE_ACTOR_MODEL="${REMOTE_ACTOR_MODEL:-gpt-5}"
OBSERVER_MODEL_ID="${OBSERVER_MODEL_ID:-gpt-4.1-mini}"
QUESTIONER_MODEL_ID="${QUESTIONER_MODEL_ID:-claude-sonnet-4-6}"

# local vllm
PORT="${PORT:-8000}"; ACTOR_TP="${ACTOR_TP:-8}"; SERVED="${SERVED:-cold-actor}"
MAX_LEN="${MAX_LEN:-32768}"
# python: /opt/conda has torch2.9.1+tf5.2(qwen3_5); vllm0.13 overlaid on shared FS.
PY="${PY:-/opt/conda/bin/python}"
VLLM_OVERLAY="${VLLM_OVERLAY:-/mnt/afs_toolcall/sunhao4/envs/vllm013}"

mkdir -p "$ROOT_DIR/logs/cold" "$OUT_BASE"

# ---- E2B creds ------------------------------------------------------------
if [[ -z "${E2B_API_KEY:-}" && -f "$ROOT_DIR/docker/sandbox/tencent.env" ]]; then
  set -a; source "$ROOT_DIR/docker/sandbox/tencent.env"; set +a
fi

# ---- REQUIRED remote check (abort if unreachable) -------------------------
[[ -n "$REMOTE_KEY" ]] || { echo "[rollout.sh] FATAL: no remote API key (looked in $APODEX_ENV). ABORT."; exit 5; }
echo "[rollout.sh] checking 3 remote models on $REMOTE_BASE ..."
for M in "$REMOTE_ACTOR_MODEL" "$OBSERVER_MODEL_ID" "$QUESTIONER_MODEL_ID"; do
  code=$(curl -sS -m 40 -o /tmp/_chk.$$ -w '%{http_code}' "$REMOTE_BASE/chat/completions" \
    -H "Authorization: Bearer $REMOTE_KEY" -H "Content-Type: application/json" \
    -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":2000}" 2>/dev/null || echo 000)
  if [[ "$code" != "200" ]]; then
    echo "[rollout.sh] FATAL: remote model '$M' not reachable (HTTP $code). observer/questioner unusable. ABORT."
    cat /tmp/_chk.$$ 2>/dev/null | head -3; rm -f /tmp/_chk.$$; exit 5
  fi
  echo "[rollout.sh]   $M OK"
done
rm -f /tmp/_chk.$$

# export agent env (agents.base reads these)
export OBSERVER_API_BASE="$REMOTE_BASE"  OBSERVER_MODEL="$OBSERVER_MODEL_ID"   OBSERVER_API_KEY="$REMOTE_KEY"
export USERSIM_API_BASE="$REMOTE_BASE"   USERSIM_MODEL="$QUESTIONER_MODEL_ID"  USERSIM_API_KEY="$REMOTE_KEY"

# ---- python env (vllm overlay) -------------------------------------------
export PYTHONPATH="$VLLM_OVERLAY:$ROOT_DIR:${PYTHONPATH:-}"
NVLIBS=$(ls -d /opt/conda/lib/python3.10/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="$NVLIBS${LD_LIBRARY_PATH:-}"

run_actor() {  # $1 = local|remote
  local actor="$1" base model key outdir
  if [[ "$actor" == "local" ]]; then base="http://127.0.0.1:$PORT/v1"; model="$SERVED"; key="sk-local"
  else base="$REMOTE_BASE"; model="$REMOTE_ACTOR_MODEL"; key="$REMOTE_KEY"; fi
  outdir="$OUT_BASE/$actor"
  echo "[rollout.sh] === actor=$actor model=$model -> $outdir ==="
  "$PY" "$ROOT_DIR/scripts/collect_rollout.py" \
    --raw "$RAW" --actor "$actor" --actor-base "$base" --actor-model "$model" --actor-key "$key" \
    --limit "$LIMIT" --backend "$BACKEND" --concurrency "$CONCURRENCY" \
    --out-dir "$outdir" 2>&1 | tee "$ROOT_DIR/logs/cold/rollout_$actor.log"
}

# ---- local vllm (only if 'local' in ACTORS) ------------------------------
VLLM_PID=""
cleanup() { [[ -n "$VLLM_PID" ]] && { echo "[rollout.sh] stop vllm $VLLM_PID"; kill "$VLLM_PID" 2>/dev/null || true; }; }
trap cleanup EXIT INT TERM

if [[ " $ACTORS " == *" local "* ]]; then
  echo "[rollout.sh] starting local vllm ($MODEL_PATH, TP=$ACTOR_TP) ..."
  "$PY" -m vllm.entrypoints.openai.api_server --model "$MODEL_PATH" \
    --served-model-name "$SERVED" --port "$PORT" --tensor-parallel-size "$ACTOR_TP" \
    --max-model-len "$MAX_LEN" --dtype bfloat16 --gpu-memory-utilization 0.90 \
    > "$ROOT_DIR/logs/cold/vllm.log" 2>&1 &
  VLLM_PID=$!
  echo "[rollout.sh] vllm pid $VLLM_PID; waiting for ready ..."
  for i in $(seq 1 600); do
    curl -sS -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo "[rollout.sh] vllm ready (${i}s)"; break; }
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "[rollout.sh] vllm died:"; tail -30 "$ROOT_DIR/logs/cold/vllm.log"; exit 3; }
    sleep 2
  done
fi

for a in $ACTORS; do run_actor "$a"; done
echo "[rollout.sh] all done. data under $OUT_BASE/{local,remote}/"
