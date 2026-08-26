#!/usr/bin/env python3
"""从 sweCoding 三个 jsonl 解析 query(issue) + 文件(repo),按当前格式存到 taskspecs_w3。

Id 用 SWE_<6位序号>(跨三个 jsonl 全局递增)。

文件从轨迹重建(file_editor view/create + terminal cat/sed),存到 taskspecs_w3/SWE_xxx/files/。
query = <issue_description> 原文(改 /repo 代码的 SWE 任务)。

用法:
  python3 scripts/data/rebuild_swecoding.py --limit 3        # 冒烟
  python3 scripts/data/rebuild_swecoding.py                 # 全量三个 jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

import yaml

ROOT = Path("/mnt/afs_toolcall/sunhao4/workspace/agentic_cl_research")
OUT = ROOT / "datasources" / "taskspecs_w3"

SRC_LIST = [
    "/mnt/afs_toolcall/zhengnairong/code/data_mllm_agent/agent/sweCoding/sweCoding_opus5-reward1-cot-synth-pipelinev3_claude5opus_hermes-openhands-sdk1391_linux_20260824/sweCoding_opus5-reward1-cot-synth-pipelinev3_claude5opus_hermes-openhands-sdk1391_linux_20260824.jsonl",
    "/mnt/afs_toolcall/zhengnairong/code/data_mllm_agent/agent/sweCoding/sweCoding_opus5-reward1-cot-2batch_claude5opus_hermes-openhands-sdk1391_linux_20260814/sweCoding_opus5-reward1-cot-2batch_claude5opus_hermes-openhands-sdk1391_linux_20260814.jsonl",
    "/mnt/afs_toolcall/zhengnairong/code/data_mllm_agent/agent/sweCoding/sweCoding_rebench-v2-reward1_claude5opus_hermes_linux_20260807/sweCoding_rebench-v2-reward1_claude5opus_hermes_linux_20260807.jsonl",
]


def extract_issue(user_content: str) -> str:
    m = re.search(r"<issue_description>(.*?)</issue_description>", user_content, re.DOTALL)
    return m.group(1).strip() if m else user_content.strip()


def strip_line_numbers(text: str) -> str:
    lines = []
    for ln in text.splitlines():
        m = re.match(r"^\s*\d+\|(.*)$", ln)
        lines.append(m.group(1) if m else ln)
    return "\n".join(lines)


def rebuild_files(msgs: list) -> tuple[OrderedDict, OrderedDict]:
    """重建文件树。返回 (初始文件, 新建文件)。"""
    calls = {}
    for m in msgs:
        if m["role"] == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc["function"]
                if fn["name"] == "file_editor":
                    args = fn.get("arguments", {})
                    if isinstance(args, dict):
                        calls[tc["id"]] = (args.get("command", ""), args.get("path", ""), args)

    results = {}
    for m in msgs:
        if m["role"] == "tool":
            results[m.get("tool_call_id", "")] = m.get("content", "")

    initial = OrderedDict()
    created = OrderedDict()

    for cid, (cmd, path, args) in calls.items():
        if not path or not path.startswith("/repo/"):
            continue
        rel = path[len("/repo/"):]
        if cmd == "create":
            created[rel] = args.get("file_text", "")
        elif cmd in ("view", "open"):
            res = results.get(cid, "")
            if res and "Error" not in res and not res.startswith("File "):
                initial.setdefault(rel, strip_line_numbers(res))

    # terminal cat/sed 的文件内容（相对路径）
    for m in msgs:
        if m["role"] == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc["function"]
                if fn["name"] != "terminal":
                    continue
                args = fn.get("arguments", {})
                if not isinstance(args, dict):
                    continue
                cmd_str = args.get("command", "")
                res = results.get(tc["id"], "")
                if not res:
                    continue
                # 提取命令里的文件路径
                paths = re.findall(r"/repo/[\w./\-]+", cmd_str)
                if not paths:
                    paths = re.findall(r"(?:cat|sed\s+-n\s+[\d,]+p|head\s+-\d+)\s+([\w./\-]+\.\w+)", cmd_str)
                if len(paths) == 1:
                    p = paths[0]
                    rel = p[len("/repo/"):] if p.startswith("/repo/") else p
                    initial.setdefault(rel, res)
                elif len(paths) > 1:
                    parts = re.split(r"={3,}|-{5,}", res)
                    parts = [p.strip() for p in parts if p.strip()]
                    if len(parts) == len(paths):
                        for p, content in zip(paths, parts):
                            rel = p[len("/repo/"):] if p.startswith("/repo/") else p
                            initial.setdefault(rel, content)

    return initial, created


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="每个 jsonl 只跑前 N 条(冒烟)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    global_idx = 0
    n_tasks = 0
    n_files = 0

    for src in SRC_LIST:
        src_name = Path(src).stem.split("_")[0]  # sweCoding
        print(f"\n=== {Path(src).name} ===", flush=True)
        with open(src) as f:
            for line in f:
                if args.limit and n_tasks >= args.limit:
                    break
                d = json.loads(line)
                msgs = d["messages"]

                # query
                query = ""
                for m in msgs:
                    if m["role"] == "user":
                        query = extract_issue(m["content"])
                        break
                if not query:
                    continue

                # 文件
                initial, created = rebuild_files(msgs)
                if not initial and not created:
                    continue  # 无文件可重建,跳过

                global_idx += 1
                rid = f"SWE_{global_idx:06d}"
                d_out = OUT / rid / "files"
                d_out.mkdir(parents=True, exist_ok=True)
                for p, content in initial.items():
                    fp = d_out / p
                    try:
                        fp.parent.mkdir(parents=True, exist_ok=True)
                        fp.write_text(content, encoding="utf-8")
                        n_files += 1
                    except (FileExistsError, NotADirectoryError, OSError):
                        pass  # 路径冲突(文件 vs 目录同名),跳过
                for p, content in created.items():
                    fp = d_out / p
                    try:
                        fp.parent.mkdir(parents=True, exist_ok=True)
                        fp.write_text(content, encoding="utf-8")
                        n_files += 1
                    except (FileExistsError, NotADirectoryError, OSError):
                        pass

                ts = {
                    "task_id": rid,
                    "seed_query": query,
                    "task_family": "sweCoding",
                    "source": src_name,
                }
                (OUT / rid / "taskspec.yaml").write_text(
                    yaml.safe_dump(ts, allow_unicode=True, sort_keys=False), encoding="utf-8")
                n_tasks += 1

                if n_tasks % 200 == 0:
                    print(f"  {n_tasks} 任务, {n_files} 文件", flush=True)

    print(f"\n✅ 完成: {n_tasks} 任务(SWE_000001 ~ SWE_{global_idx:06d}), {n_files} 文件")
    print(f"   输出: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
