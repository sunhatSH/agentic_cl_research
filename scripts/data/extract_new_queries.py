#!/usr/bin/env python3
"""从新轨迹 jsonl 提取 queries，适配 label_capability.py 打标。

Input: /mnt/afs_toolcall/tongronglei/workspace/subagent_delivery/1_release_2607*/**.jsonl
Output: datasets/queries_new.jsonl（每行 {record_id, seed_query, tools, source_file}）

只需要 record_id + 第一条 user query + tools 列表，约 200B/条。
"""
import json, os, sys
from pathlib import Path

DIRS = [
    "/mnt/afs_toolcall/tongronglei/workspace/subagent_delivery/1_release_260731",
    "/mnt/afs_toolcall/tongronglei/workspace/subagent_delivery/1_release_260805",
]
OUT = Path(__file__).resolve().parent.parent.parent / "datasets" / "queries_new.jsonl"

def extract_query(rec):
    msgs = rec.get("messages", [])
    if isinstance(msgs, str):
        try: msgs = json.loads(msgs)
        except: return None
    for m in msgs:
        if m.get("role") == "user":
            c = (m.get("content") or "").strip()
            if c: return c
    return None

def extract_tools(rec):
    tools = rec.get("tools", [])
    if isinstance(tools, list):
        return [t.get("function", {}).get("name", "?") for t in tools if isinstance(t, dict)]
    return []

total = 0
with open(OUT, "w", encoding="utf-8") as fout:
    for d in DIRS:
        if not os.path.isdir(d): continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".jsonl"): continue
            fp = os.path.join(d, fn)
            with open(fp) as fh:
                for line in fh:
                    line = line.strip()
                    if not line: continue
                    try: rec = json.loads(line)
                    except: continue
                    q = extract_query(rec)
                    if not q: continue
                    mid = rec.get("metadata") or {}
                    rid = mid.get("session_id") or mid.get("request_id") or f"unk_{total}"
                    fout.write(json.dumps({
                        "record_id": rid,
                        "seed_query": q[:2000],
                        "tools": extract_tools(rec),
                        "source_file": fn,
                    }, ensure_ascii=False) + "\n")
                    total += 1

print(f"[extract] {total} queries → {OUT} ({OUT.stat().st_size/1024/1024:.1f} MB)")
