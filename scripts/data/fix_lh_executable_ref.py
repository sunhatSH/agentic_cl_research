#!/usr/bin/env python3
"""修复 177 个 LH 任务 query 里的悬空 executable 引用(F14 后续)。

longhorizonCoding 格式标配一句"运行预编译 executable 观察行为",但该 executable
不存在(源码已内嵌 query)、files/ 也没有。AI 核验(verify_nofiles_legit)标出 4-5 个
最明显的。根治:把 query 里指向 test*_executable 的"执行方法"改写成"参照上方内嵌
Python 源码",删掉"不要用 python / 直接跑 executable"的误导句。

改写规则(只动 query 文本,不改任务语义/GT):
  - "run the pre-compiled executable: `.../test*_executable --arg value`" 描述行 →
    "refer to the embedded Python source above for expected behavior"
  - "Do not use the `python` command; run the executable directly. The executable
    already includes all necessary dependencies…" → 删除(误导:让 agent 跑不存在的文件)
  - "behaves identically to `.../test*_executable --arg val`" → "…to the Python source above"

同步改 train_cl.parquet + jsonl。
用法: python3 scripts/data/fix_lh_executable_ref.py [--apply]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "datasets" / "train_cl.parquet"
JSONL = ROOT / "datasets" / "train_cl.jsonl"


def fix_query(q: str) -> tuple[str, int]:
    """改写悬空 executable 引用。返回 (新 query, 改动次数)。"""
    n = 0
    orig = q
    # 1. "run the pre-compiled executable: `...test*_executable --arg value`" 整句
    q2 = re.sub(
        r"You can observe the behavior of the Python code by running the pre-compiled executable:\s*`[^`]*_executable[^`]*`",
        "You can observe the expected behavior directly from the embedded Python source code shown above",
        q)
    # 2. "Do not use python; run the executable directly. The executable already includes…" 误导句删除
    q2 = re.sub(
        r"[-*\s]*\**Important\**:\s*Do not use the `python` command; run the executable directly\.\s*"
        r"The executable already includes all necessary dependencies,?\s*so there'?s no need to worry about environment issues\.",
        "", q2)
    # 3. "behaves identically to `...test*_executable --arg val`" → 指向源码
    q2 = re.sub(
        r"behaves identically to `[^`]*_executable[^`]*`",
        "behaves identically to the Python source shown above",
        q2)
    # 4. 兜底:任何残留的 `/home/user/workspace/xxx_executable ...` 反引号引用 → 去可执行、指源码
    q2 = re.sub(
        r"`[^`]*_executable[^`]*`",
        "the embedded Python source above",
        q2)
    if q2 != orig:
        n = 1
    return q2, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    t = pq.read_table(str(PARQUET))
    rows = t.to_pylist()
    changed = 0
    still_ref = 0
    for r in rows:
        gid = r["extra_info"].get("gen_task_id", "")
        if not gid.startswith("LH_"):
            continue
        q = (r["extra_info"].get("queries") or [""])[0]
        nq, n = fix_query(q)
        if n:
            changed += 1
            # 同步三处:prompt content / queries[0]
            r["extra_info"]["queries"] = [nq]
            if r.get("prompt") and r["prompt"][0].get("role") == "user":
                r["prompt"][0]["content"] = nq
        if "_executable" in nq:
            still_ref += 1
    print(f"LH 改写: {changed} 行", flush=True)
    print(f"改写后仍残留 _executable 引用: {still_ref}", flush=True)

    # 抽查
    for r in rows:
        if r["extra_info"].get("gen_task_id") == "LH_000032":
            q = (r["extra_info"].get("queries") or [""])[0]
            i = q.lower().find("execution method")
            print(f"\n=== LH_000032 改写后执行方法段 ===\n{q[i-20:i+350] if i>=0 else '(未找到)'}", flush=True)
            break

    if not args.apply:
        print("\n加 --apply 落地", flush=True)
        return 0

    new_t = pa.Table.from_pylist(rows, schema=t.schema)
    tmp = PARQUET.with_suffix(".parquet.tmp")
    pq.write_table(new_t, tmp)
    tmp.replace(PARQUET)
    with open(JSONL.with_suffix(".jsonl.tmp"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    JSONL.with_suffix(".jsonl.tmp").replace(JSONL)
    print(f"\n✅ 落地 {PARQUET.name} + {JSONL.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
