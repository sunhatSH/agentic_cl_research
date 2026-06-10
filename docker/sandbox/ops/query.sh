#!/usr/bin/env bash
# One-click sandbox status query (Tools + Instances summary).
#
# Usage:
#   bash docker/sandbox/ops/query.sh
#   bash docker/sandbox/ops/query.sh --tool agentic-cl-code-interpreter
#   bash docker/sandbox/ops/query.sh --json
set -euo pipefail

OPS_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=docker/sandbox/ops/_common.sh
source "${OPS_DIR}/_common.sh"

TOOL_FILTER=""
JSON_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tool)
      TOOL_FILTER="${2:-}"
      shift 2
      ;;
    --json)
      JSON_ONLY=1
      shift
      ;;
    -h | --help)
      sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "[ops] unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

ops_require_credentials
ops_require_tccli
REGION="$(ops_region)"

python3 - "$REGION" "$TOOL_FILTER" "$JSON_ONLY" <<'PY'
import json
import subprocess
import sys
from collections import Counter

region, tool_filter, json_only = sys.argv[1], sys.argv[2], int(sys.argv[3])


def tccli(*args):
    cmd = ["tccli", "ags", *args, "--region", region]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr or r.stdout, file=sys.stderr)
        sys.exit(r.returncode)
    return json.loads(r.stdout)


tools = tccli("DescribeSandboxToolList")
instances = tccli("DescribeSandboxInstanceList")

tool_set = tools.get("SandboxToolSet") or []
inst_set = instances.get("InstanceSet") or []

if tool_filter:
    inst_set = [i for i in inst_set if i.get("ToolName") == tool_filter]

out = {
    "region": region,
    "tools": tool_set,
    "instances": inst_set,
    "summary": {
        "tool_count": len(tool_set),
        "instance_count": len(inst_set),
        "by_status": dict(Counter(i.get("Status", "?") for i in inst_set)),
        "by_tool": dict(Counter(i.get("ToolName", "?") for i in inst_set)),
    },
}

if json_only:
    print(json.dumps(out, ensure_ascii=False, indent=2))
    sys.exit(0)

print(f"=== Sandbox 查询 | 地域 {region} ===")
if tool_filter:
    print(f"过滤 ToolName: {tool_filter}")
print()

print(f"[沙箱工具] 共 {len(tool_set)} 个")
for t in sorted(tool_set, key=lambda x: x.get("ToolName", "")):
    print(
        f"  {t.get('ToolName', '?'):32} "
        f"type={t.get('ToolType', '?'):16} "
        f"status={t.get('Status', '?')} "
        f"id={t.get('ToolId', '?')}"
    )

print()
summary = out["summary"]
print(f"[沙箱实例] 共 {summary['instance_count']} 个")
if summary["by_status"]:
    print("  按 Status:", ", ".join(f"{k}={v}" for k, v in sorted(summary["by_status"].items())))
if summary["by_tool"]:
    print("  按 ToolName:")
    for name, cnt in sorted(summary["by_tool"].items(), key=lambda x: (-x[1], x[0])):
        mark = "  <-- CL" if name in ("agentic-cl-code-interpreter", "agentic-cl-sandbox") else ""
        print(f"    {cnt:4}  {name}{mark}")

running = [i for i in inst_set if i.get("Status") == "RUNNING"]
if running:
    print()
    print(f"[RUNNING] {len(running)} 个")
    for i in running[:20]:
        print(
            f"  {i.get('InstanceId', '?')[:40]:40} "
            f"{i.get('ToolName', '?'):28} "
            f"expires={i.get('ExpiresAt', '?')}"
        )
    if len(running) > 20:
        print(f"  ... 还有 {len(running) - 20} 个 RUNNING")
else:
    print()
    print("[RUNNING] 0 个")

cl_tools = {"agentic-cl-code-interpreter", "agentic-cl-sandbox"}
cl_inst = [i for i in inst_set if i.get("ToolName") in cl_tools]
print()
print(f"[本项目 CL Tool] 实例 {len(cl_inst)} 个 (RUNNING {sum(1 for i in cl_inst if i.get('Status')=='RUNNING')})")

print()
print("控制台: https://console.cloud.tencent.com/agent-runtime (地域选北京)")
print("JSON:   bash docker/sandbox/ops/query.sh --json")
PY
