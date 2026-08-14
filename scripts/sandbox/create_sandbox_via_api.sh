#!/usr/bin/env bash
# Create Agent Runtime sandbox Tool + API key + optional test instance via tccli.
#
# Requires API keys (NOT login password). Create in CAM console:
#   https://console.cloud.tencent.com/cam/capi
#
# Usage:
#   cp docker/sandbox/tencent.env.example docker/sandbox/tencent.env
#   # fill TENCENTCLOUD_SECRET_ID / TENCENTCLOUD_SECRET_KEY only in tencent.env
#   set -a && source docker/sandbox/tencent.env && set +a
#   bash scripts/sandbox/create_sandbox_via_api.sh
#
# Modes:
#   builtin (default) — code-interpreter Tool, no custom image
#   custom            — configs/sandbox_tool.json (needs CCR image + RoleArn)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODE="${1:-builtin}"

# Auto-load gitignored env files if present
if [[ -f "${ROOT}/scripts/env/load_tencent_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/scripts/env/load_tencent_env.sh" 2>/dev/null || true
fi

REGION="${TENCENTCLOUD_REGION:-ap-beijing}"

if [[ -z "${TENCENTCLOUD_SECRET_ID:-}" || -z "${TENCENTCLOUD_SECRET_KEY:-}" ]]; then
  echo "[create_sandbox] ERROR: fill docker/sandbox/tencent.env then:"
  echo "  source scripts/env/load_tencent_env.sh"
  echo "  API keys: https://console.cloud.tencent.com/cam/capi"
  exit 1
fi

if ! command -v tccli >/dev/null 2>&1; then
  echo "[create_sandbox] installing tccli..."
  pip install -q tccli
fi

echo "[create_sandbox] region=${REGION} mode=${MODE}"

# --- List existing tools ---
echo "[create_sandbox] existing tools:"
tccli ags DescribeSandboxToolList --region "$REGION" 2>&1 | head -20 || true

# --- Create Tool ---
if [[ "$MODE" == "builtin" ]]; then
  CFG="${ROOT}/configs/sandbox_tool_builtin.json"
  TOOL_NAME="agentic-cl-code-interpreter"
  echo "[create_sandbox] creating built-in code-interpreter Tool from ${CFG}"
  CREATE_OUT=$(tccli ags CreateSandboxTool \
    --region "$REGION" \
    --cli-input-json "file://${CFG}")
elif [[ "$MODE" == "custom" ]]; then
  CFG="${ROOT}/configs/sandbox_tool.json"
  TOOL_NAME="agentic-cl-sandbox"
  if grep -q REPLACE "$CFG"; then
    echo "[create_sandbox] ERROR: edit ${CFG} (RoleArn, Image) before custom mode"
    exit 1
  fi
  echo "[create_sandbox] creating custom Tool from ${CFG}"
  CREATE_OUT=$(tccli ags CreateSandboxTool \
    --region "$REGION" \
    --cli-input-json "file://${CFG}")
else
  echo "usage: $0 [builtin|custom]"
  exit 1
fi

echo "$CREATE_OUT"
TOOL_ID=$(python3 -c "import json,sys; print(json.load(sys.stdin)['ToolId'])" <<<"$CREATE_OUT")
echo "[create_sandbox] ToolId=${TOOL_ID} ToolName=${TOOL_NAME}"

# --- API key for E2B SDK ---
echo "[create_sandbox] creating API key (for E2B_SDK)..."
KEY_OUT=$(tccli ags CreateAPIKey --region "$REGION" --Name "agentic-cl-$(date +%Y%m%d)")
echo "$KEY_OUT"
echo ""
echo "=== SAVE THESE (shown once) ==="
python3 -c "import json,sys; d=json.load(sys.stdin); print('E2B_API_KEY=', d.get('APIKey','')); print('KeyId=', d.get('KeyId',''))" <<<"$KEY_OUT"
echo "E2B_DOMAIN=ap-${REGION#ap-}.tencentags.com  # verify with admin"
echo ""

# --- Start one instance (smoke) ---
echo "[create_sandbox] starting test instance (timeout 10m)..."
INST_OUT=$(tccli ags StartSandboxInstance \
  --region "$REGION" \
  --ToolId "$TOOL_ID" \
  --Timeout "10m")
echo "$INST_OUT"

echo "[create_sandbox] done. Test locally (needs network to tencentags):"
echo "  export E2B_API_KEY=<from above>"
echo "  export E2B_DOMAIN=ap-<region>.tencentags.com"
echo "  pip install e2b-code-interpreter"
echo "  python scripts/collect/sandbox_smoke.py --backend e2b  # template=${TOOL_NAME}"
