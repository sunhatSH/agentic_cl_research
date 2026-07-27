#!/usr/bin/env bash
# Create Agent Runtime sandbox TOOL from configs/sandbox_tool.json (after image push).
#
# Option A — agr CLI (if installed):
#   agr tool create --request @configs/sandbox_tool.json -o json
#
# Option B — copy JSON to console: Agent Runtime -> 沙箱工具 -> 新建
#
# This script prints the checklist and validates the JSON template.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CFG="${ROOT}/configs/sandbox_tool.json"

echo "=== Agent Runtime sandbox tool checklist ==="
echo "1. Image pushed: see docker/sandbox/image.env FULL_TAG"
echo "2. configs/sandbox_tool.json filled (RoleArn, Image URL)"
echo "3. CAM role created with CCR/TCR pull permissions"
echo "4. PassRole policy granted to your user"
echo ""
echo "Template: ${CFG}"
echo ""

if grep -q 'REPLACE' "$CFG"; then
  echo "WARN: template still has REPLACE_* placeholders — edit before creating tool."
fi

if command -v agr >/dev/null 2>&1; then
  echo "agr found. To create:"
  echo "  agr tool create --request @${CFG} -o json"
else
  echo "agr CLI not installed. Use console or API (see doc/sandbox/Sandbox_腾讯云操作手册.md)."
fi

# Smoke-test E2B SDK access after tool is created:
echo ""
echo "After tool creation, SDK access:"
echo "  export E2B_DOMAIN=ap-<region>.tencentags.com   # ask admin"
echo "  export E2B_API_KEY=<your-key>"
echo "  python scripts/collect/sandbox_smoke.py --backend e2b"
echo "  # template name = ToolName in sandbox_tool.json (agentic-cl-sandbox)"
