#!/usr/bin/env bash
# 必然带图的 lightllm 推理测试 —— 验证 Qwen3.5 收到图片请求是否崩(M-RoPE start_idx=None)。
#
# 用途:不用跑整个训练,直接起一个 lightllm server + 发一条带图请求,第一个请求就带图,
#       未修 → 必崩 `RuntimeError: Could not infer dtype of NoneType`(推理进程死);
#       已修 → 正常返回或图片被安全处理。
#
# 用法(在有 GPU 的训练环境):
#   bash scripts/test_image_inference.sh              # 默认 enable_multimodal=false(复现纯文本收到图的崩溃)
#   ENABLE_MM=1 bash scripts/test_image_inference.sh  # enable_multimodal=true(开视觉,对比)
#
# 前置:模型在 $MODEL(默认 /dev/shm/cl_models/Qwen3.5-9B 或 AFS 原路径),单卡够跑 TP1。

set -uo pipefail
ROOT="/mnt/afs_toolcall/sunhao4/workspace/agentic_cl_research"
LIGHTLLM="/mnt/afs_toolcall/sunhao4/workspace/LightLLM"
PY="${PY:-/mnt/afs_toolcall/sunhao4/miniconda3/bin/python3}"
MODEL="${MODEL:-/mnt/afs_toolcall/sunhao4/models/Qwen3.5-9B}"
PORT="${PORT:-9911}"
IMG="${IMG:-$LIGHTLLM/test/test_api/test.jpg}"
ENABLE_MM="${ENABLE_MM:-0}"      # 0=纯文本(复现崩) 1=开视觉
TEXT_MODEL_ONLY="${TEXT_MODEL_ONLY:-}"  # 传给 lightllm 的纯文本模型开关(测修复用)

export PYTHONPATH="$LIGHTLLM:$ROOT:${PYTHONPATH:-}"
[ -n "$TEXT_MODEL_ONLY" ] && export TEXT_MODEL_ONLY

_mm_flag=""
[ "$ENABLE_MM" = "1" ] && _mm_flag="--enable_multimodal" || _mm_flag="--disable_vision --disable_audio"

echo "[test] 起 lightllm server: model=$MODEL port=$PORT enable_mm=$ENABLE_MM TEXT_MODEL_ONLY=${TEXT_MODEL_ONLY:-unset}"
"$PY" -m lightllm.server.api_server \
  --model_dir "$MODEL" --port "$PORT" --tp 1 \
  --trust_remote_code $_mm_flag \
  --mem_fraction 0.6 --max_total_token_num 40000 \
  > /tmp/lightllm_imgtest.log 2>&1 &
_SRV_PID=$!
echo "[test] server pid=$_SRV_PID, 等起服(最多 240s)..."

# 等 server ready
_ready=0
for i in $(seq 1 48); do
  sleep 5
  if grep -q 'server start up ok' /tmp/lightllm_imgtest.log 2>/dev/null; then _ready=1; break; fi
  if ! kill -0 "$_SRV_PID" 2>/dev/null; then echo "[test] server 进程已死,看日志:"; tail -30 /tmp/lightllm_imgtest.log; exit 1; fi
done
[ "$_ready" = "1" ] || { echo "[test] 起服超时"; tail -30 /tmp/lightllm_imgtest.log; kill "$_SRV_PID" 2>/dev/null; exit 1; }
echo "[test] server ready ✓"

# 发一条【必然带图】的请求
echo "[test] 发带图请求(prompt 含 <|vision_start|><|image_pad|><|vision_end|> + base64 图)..."
"$PY" - "$IMG" "$PORT" <<'PYEOF'
import base64, json, sys, requests
img, port = sys.argv[1], sys.argv[2]
b64 = base64.b64encode(open(img,"rb").read()).decode()
prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
          "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n"
          "<|im_start|>assistant\n")
data = {"inputs": prompt,
        "parameters": {"temperature": 0.0, "do_sample": False, "max_new_tokens": 64},
        "multimodal_params": {"images": [{"type": "base64", "data": b64}]}}
try:
    r = requests.post(f"http://localhost:{port}/generate", json=data, timeout=120)
    print(f"[test] HTTP {r.status_code}")
    print(f"[test] 响应: {r.text[:500]}")
    if r.status_code == 200:
        print("[test] ✅ 带图请求成功返回(未崩)")
    else:
        print("[test] ⚠️ 非200,看是否被安全拒绝 vs 崩溃")
except Exception as e:
    print(f"[test] ❌ 请求异常: {type(e).__name__}: {e}")
PYEOF

echo "[test] 检查 server 日志有无 M-RoPE 崩溃:"
grep -c 'Could not infer dtype of NoneType' /tmp/lightllm_imgtest.log 2>/dev/null | sed 's/^/  M-RoPE崩溃次数: /'
grep -q 'Could not infer dtype' /tmp/lightllm_imgtest.log && echo "  ❌ 复现崩溃(未修)" || echo "  ✓ 无 M-RoPE 崩溃"

echo "[test] 清理 server..."
kill "$_SRV_PID" 2>/dev/null
echo "[test] 完整日志: /tmp/lightllm_imgtest.log"
