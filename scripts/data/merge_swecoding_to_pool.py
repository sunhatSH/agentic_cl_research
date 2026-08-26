#!/usr/bin/env python3
"""把 sweCoding 任务(SWE_xxx)并入 9 桶数据源 all_tasks_labeled.jsonl。

sweCoding 任务元数据:
  - D_id = SWE_xxx
  - user_prompt = issue(seed_query)
  - bucket / difficulty = swecoding_labeled.jsonl 打标结果
  - inputs_dir = taskspecs_w3/SWE_xxx/files
  - label_source = swecoding

用法: python3 scripts/data/merge_swecoding_to_pool.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
TASKSPECS = ROOT / "datasources" / "taskspecs_w3"
SWE_LABELED = ROOT / "datasources" / "labeled" / "swecoding_labeled.jsonl"
POOL = ROOT / "datasources" / "labeled" / "all_tasks_labeled.jsonl"


def main() -> int:
    # 读 swecoding 桶+难度
    swe = {}
    if SWE_LABELED.is_file():
        for line in open(SWE_LABELED, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                swe[d["id"]] = (d.get("bucket", ""), d.get("difficulty"))

    print(f"swecoding_labeled: {len(swe)} 任务", flush=True)

    # 读现有 pool,去重(避免重复 append)
    existing = set()
    if POOL.is_file():
        for line in open(POOL, encoding="utf-8"):
            if line.strip():
                existing.add(json.loads(line)["D_id"])
    print(f"现有 pool: {len(existing)} 任务", flush=True)

    # 生成 swecoding 行
    rows = []
    for rid, (bk, dv) in sorted(swe.items()):
        if rid in existing:
            continue
        if not bk or dv is None:
            continue  # 打标失败跳过
        f = TASKSPECS / rid / "taskspec.yaml"
        if not f.is_file():
            continue
        ts = yaml.safe_load(f.read_text(encoding="utf-8"))
        query = (ts.get("seed_query") or "").strip()
        files_dir = TASKSPECS / rid / "files"
        has_inputs = files_dir.is_dir() and bool(list(files_dir.iterdir()))
        rows.append({
            "D_id": rid,
            "domain": "SWE",
            "user_prompt": query,
            "has_inputs": has_inputs,
            "inputs_dir": str(files_dir) if has_inputs else "",
            "bucket": bk,
            "difficulty": dv,
            "label_source": "swecoding",
        })

    # append
    with open(POOL, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 统计
    total = len(existing) + len(rows)
    bk = Counter()
    for line in open(POOL, encoding="utf-8"):
        if line.strip():
            d = json.loads(line)
            if d.get("bucket"):
                bk[d["bucket"]] += 1
    print(f"\n✅ 新增 {len(rows)} swecoding 任务, pool 总 {total}", flush=True)
    print(f"桶分布: {dict(bk)}", flush=True)
    print(f"coding: {bk['coding']}, research: {bk['research']}", flush=True)

    # coding/research d4-7 有文件
    def has_files(d):
        return d.get("has_inputs", False)
    cr_d47 = Counter()
    for line in open(POOL, encoding="utf-8"):
        if line.strip():
            d = json.loads(line)
            if d.get("bucket") in ("coding", "research") and d.get("difficulty") is not None and 4 <= d["difficulty"] <= 7 and has_files(d):
                cr_d47[d["bucket"]] += 1
    print(f"coding/research d4-7 有文件: {dict(cr_d47)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
