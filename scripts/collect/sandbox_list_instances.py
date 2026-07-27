#!/usr/bin/env python3
"""查询腾讯云 Agent Runtime 沙箱实例（E2B 兼容 API）。

用法:
    source docker/sandbox/tencent.env   # 加载 E2B_API_KEY / E2B_DOMAIN
    python3 scripts/collect/sandbox_list_instances.py              # 列全部
    python3 scripts/collect/sandbox_list_instances.py --tool agentic-cl-sandbox   # 按 ToolName 过滤
    python3 scripts/collect/sandbox_list_instances.py --running     # 只看 running
    python3 scripts/collect/sandbox_list_instances.py --id <sandboxID>  # 查单个

数据源: GET https://api.<E2B_DOMAIN>/sandboxes  (E2B 兼容 API)
字段 cpuCount/memoryMB/diskSizeMB 是 E2B API 返回的规格值，可能不反映
Tool CustomConfiguration.Resources 配置的真实分配（以官方 SDK 或后台为准）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import httpx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tool", help="按 ToolName (alias) 过滤")
    ap.add_argument("--tool-id", help="按 ToolId (templateID, sdt-...) 过滤")
    ap.add_argument("--running", action="store_true", help="只看 state=running")
    ap.add_argument("--id", help="查单个 sandboxID")
    ap.add_argument("--json", action="store_true", help="原始 JSON 输出")
    args = ap.parse_args()

    api_key = os.environ.get("E2B_API_KEY")
    domain = os.environ.get("E2B_DOMAIN", "ap-beijing.tencentags.com")
    if not api_key:
        print("✗ 未设置 E2B_API_KEY，先 source docker/sandbox/tencent.env", file=sys.stderr)
        return 1

    api_url = f"https://api.{domain}"
    headers = {"X-API-KEY": api_key}

    # 查单个 / 列表
    if args.id:
        url = f"{api_url}/sandboxes/{args.id}"
        r = httpx.get(url, headers=headers, timeout=30)
        if r.status_code != 200:
            print(f"✗ GET {url} -> {r.status_code}: {r.text[:200]}", file=sys.stderr)
            return 1
        insts = [r.json()]
    else:
        url = f"{api_url}/sandboxes"
        r = httpx.get(url, headers=headers, timeout=30)
        if r.status_code != 200:
            print(f"✗ GET {url} -> {r.status_code}: {r.text[:200]}", file=sys.stderr)
            return 1
        insts = r.json()

    # 过滤
    if args.tool:
        insts = [s for s in insts if s.get("alias") == args.tool]
    if args.tool_id:
        insts = [s for s in insts if s.get("templateID") == args.tool_id]
    if args.running:
        insts = [s for s in insts if s.get("state") == "running"]

    if args.json:
        print(json.dumps(insts, ensure_ascii=False, indent=2))
        return 0

    print(f"实例数: {len(insts)}" + (" (全部)" if not any([args.tool, args.tool_id, args.running]) else ""))
    print()
    for s in insts:
        print(f"  sandboxID : {s.get('sandboxID')}")
        print(f"    templateID : {s.get('templateID')}  alias={s.get('alias')}")
        print(f"    state      : {s.get('state')}")
        print(f"    规格(E2B API): cpu={s.get('cpuCount')} mem={s.get('memoryMB')}MB disk={s.get('diskSizeMB')}MB")
        print(f"    startedAt  : {s.get('startedAt','?')[:19]}  endAt={s.get('endAt','?')[:19]}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
