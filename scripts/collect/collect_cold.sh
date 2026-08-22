#!/usr/bin/env bash
# Cold-start data collection launcher (平台「启动命令」直接填这个脚本).
#
# One shot, self-contained:
#   1. launch a LOCAL vllm OpenAI-compatible server loading Qwen3.6-27B (background)
#   2. poll /v1/models until ready
#   3. run scripts/collect/collect_cold.py over the raw seeds (single-turn, NO scoring)
#   4. kill vllm on exit
#
# The actor is the LOCAL 27B (vllm loads the weights on disk), NOT a remote API.
# Cold data needs no observer/questioner/judge -- just seed -> rollout -> sandbox.
#
# Logs (all under logs/cold/ on the shared FS):
#   vllm.log     -- vllm server stdout/stderr (model load, OOM, inference errors)
#   collect.log  -- collector progress ([cold] lines), also shown on platform stdout
#   buffer.sqlite -- the collected 9-bucket replay buffer (the product)
#
# Usage (cluster GPU node) -- E2B creds auto-loaded from docker/sandbox/tencent.env:
#   bash scripts/collect/collect_cold.sh
# (or override creds yourself: E2B_API_KEY=... E2B_DOMAIN=... bash scripts/collect/collect_cold.sh)
#
# Override via env:
#   MODEL_PATH   actor weights        (default: local Qwen3.6-27B)
#   QUERIES      queries JSONL path   (output of prepare_queries / data-filter)
#   LIMIT        #sessions            (default 10000)
#   GROUP_SIZE   slots per query      (default 8)
#   ACTOR_TP     tensor-parallel size (default 8 = single 8-GPU node)
#   PORT         vllm port            (default 8000)
#   VLLM_PY      python with vllm     (default /opt/conda/bin/python)
#   COLLECT_PY   python for collector (default same as VLLM_PY)
#   TENCENT_ENV  creds file to source (default docker/sandbox/tencent.env)
#   OUT          buffer dump path
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Auto-load Tencent / E2B credentials (E2B_API_KEY / E2B_DOMAIN + TENCENTCLOUD_*)
# from the repo's sandbox env file unless already set in the environment.
# Override the path with TENCENT_ENV=, or just export the vars yourself to skip.
TENCENT_ENV="${TENCENT_ENV:-$ROOT_DIR/docker/sandbox/tencent.env}"
if [[ -z "${E2B_API_KEY:-}" && -f "$TENCENT_ENV" ]]; then
  echo "[cold.sh] sourcing credentials from $TENCENT_ENV"
  set -a; source "$TENCENT_ENV"; set +a
fi

MODEL_PATH="${MODEL_PATH:-/mnt/afs_toolcall/sunhao4/models/Qwen3.6-27B}"
QUERIES="${QUERIES:-/mnt/afs_toolcall/sunhao4/datasets/juxiaolong_prefix/queries.jsonl}"
LIMIT="${LIMIT:-10000}"
GROUP_SIZE="${GROUP_SIZE:-8}"
BACKEND="${BACKEND:-e2b}"
ACTOR_TP="${ACTOR_TP:-8}"
PORT="${PORT:-8000}"
SERVED_NAME="${SERVED_NAME:-cold-actor}"
MAX_LEN="${MAX_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
DTYPE="${DTYPE:-bfloat16}"
TOKENIZER="${TOKENIZER:-$MODEL_PATH}"   # tokenizer for token ID recovery from vllm logprobs
# Pick the python that has vllm. Honor an explicit VLLM_PY; otherwise probe
# common locations + PATH and pick the first whose `import vllm` succeeds, so
# this works across images (modelscope py311, /opt/conda, venvs, ...).
if [[ -z "${VLLM_PY:-}" ]]; then
  for cand in /opt/conda/bin/python python python3 \
              /usr/bin/python3 /usr/local/bin/python; do
    p="$(command -v "$cand" 2>/dev/null)" || continue
    if "$p" -c "import vllm" >/dev/null 2>&1; then VLLM_PY="$p"; break; fi
  done
fi
VLLM_PY="${VLLM_PY:-python}"   # last-resort fallback
COLLECT_PY="${COLLECT_PY:-$VLLM_PY}"
OUT="${OUT:-$ROOT_DIR/datasources/mock/buffer_dumps/cold_buffer.sqlite}"  # mock 产物，与正式隔离

# transformers must recognize Qwen3.6-27B's `qwen3_5` arch. Old image transformers
# don't -> we install a new one into a SHARED-FS dir (persists across Pods) and
# prepend it to PYTHONPATH. Override TF_LIBS path or TF_SPEC version if needed.
TF_LIBS="${TF_LIBS:-/mnt/afs_toolcall/sunhao4/pylibs/cold-py311}"
TF_SPEC="${TF_SPEC:-transformers>=5.2}"
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

echo "[cold.sh] python=$VLLM_PY"
"$VLLM_PY" -c "import vllm; print('[cold.sh] vllm', vllm.__version__)" 2>/dev/null \
  || echo "[cold.sh] WARN: '$VLLM_PY' has no importable vllm -- server will likely fail"
echo "[cold.sh] model=$MODEL_PATH tp=$ACTOR_TP port=$PORT backend=$BACKEND limit=$LIMIT"
echo "[cold.sh] queries=$QUERIES out=$OUT"

# --- sanity ----------------------------------------------------------------
[[ -d "$MODEL_PATH" ]] || { echo "[cold.sh] ERROR: MODEL_PATH not found: $MODEL_PATH"; exit 2; }
[[ -f "$QUERIES" ]]     || { echo "[cold.sh] ERROR: QUERIES not found: $QUERIES"; exit 2; }
if [[ "$BACKEND" == "e2b" ]]; then
  [[ -n "${E2B_API_KEY:-}" && -n "${E2B_DOMAIN:-}" ]] || {
    echo "[cold.sh] ERROR: e2b backend needs E2B_API_KEY + E2B_DOMAIN in env"; exit 2; }
fi

# --- ensure transformers knows qwen3_5 (shared-FS install, persists) -------
# Prepend the shared lib dir so a previously-installed new transformers is used.
export PYTHONPATH="$TF_LIBS:${PYTHONPATH:-}"
qwen35_ok() { "$VLLM_PY" -c \
  "from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as M; import sys; sys.exit(0 if 'qwen3_5' in M else 1)" \
  >/dev/null 2>&1; }

if qwen35_ok; then
  echo "[cold.sh] transformers already recognizes qwen3_5"
else
  echo "[cold.sh] transformers too old; installing '$TF_SPEC' -> $TF_LIBS"
  mkdir -p "$TF_LIBS"
  "$VLLM_PY" -m pip install --no-cache-dir --target "$TF_LIBS" -i "$PIP_INDEX" -U "$TF_SPEC" \
    >> "$ROOT_DIR/logs/cold/pip.log" 2>&1 || {
      echo "[cold.sh] ERROR: pip install transformers failed; tail logs/cold/pip.log:"; tail -20 "$ROOT_DIR/logs/cold/pip.log"; exit 4; }
  qwen35_ok || {
    echo "[cold.sh] ERROR: transformers still doesn't recognize qwen3_5 after install"; exit 4; }
  echo "[cold.sh] transformers installed; qwen3_5 OK"
fi
"$VLLM_PY" -c "import transformers; print('[cold.sh] transformers', transformers.__version__)" 2>/dev/null || true

mkdir -p "$(dirname "$OUT")" "$ROOT_DIR/logs/cold"
VLLM_LOG="$ROOT_DIR/logs/cold/vllm.log"

# --- 1. launch local vllm server (background) ------------------------------
echo "[cold.sh] starting vllm server -> $VLLM_LOG"
"$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --tensor-parallel-size "$ACTOR_TP" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --dtype "$DTYPE" \
  --disable-log-requests \
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

cleanup() {
  echo "[cold.sh] stopping vllm (pid $VLLM_PID)"
  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --- 2. wait until ready ---------------------------------------------------
echo "[cold.sh] waiting for vllm on :$PORT ..."
for i in $(seq 1 600); do          # up to ~20 min for 27B load
  if curl -sS -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "[cold.sh] vllm ready after ${i}s"
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[cold.sh] ERROR: vllm died during startup; tail of $VLLM_LOG:"; tail -30 "$VLLM_LOG"; exit 3
  fi
  sleep 2
done
curl -sS -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 || {
  echo "[cold.sh] ERROR: vllm not ready in time; tail of $VLLM_LOG:"; tail -30 "$VLLM_LOG"; exit 3; }

# --- 3. run collector ------------------------------------------------------
COLLECT_LOG="$ROOT_DIR/logs/cold/collect.log"
echo "[cold.sh] collecting ... (progress also tee'd to $COLLECT_LOG)"
cd "$ROOT_DIR"
"$COLLECT_PY" scripts/collect/collect_cold.py \
  --queries "$QUERIES" \
  --limit "$LIMIT" \
  --group-size "$GROUP_SIZE" \
  --backend "$BACKEND" \
  --actor-base "http://127.0.0.1:$PORT/v1" \
  --actor-model "$SERVED_NAME" \
  --tokenizer "$TOKENIZER" \
  --out "$OUT" 2>&1 | tee "$COLLECT_LOG"
COLLECT_RC=${PIPESTATUS[0]}   # collector's exit code, not tee's

echo "[cold.sh] done -> $OUT (collector rc=$COLLECT_RC)"
# trap cleanup kills vllm on exit
exit "$COLLECT_RC"
