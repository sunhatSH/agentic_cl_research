#!/usr/bin/env python3
"""真实统计 office/research/ops 三桶【全部】满足要求的数据量(不 stop-at-target)。

对每桶【全部 d4-7】记录逐条真实检查:
  - 路径归一化后无残留坏路径(Windows盘符/未归一化外部绝对路径)
  - query 点名文件在 taskspecs_w3/<D_id>/files/ 真实存在(产出型引用不算缺)
  - answer_key.json 有 checks 或 rubric
并发跑(AFS stat IO-bound)。ops 按口径【允许无GT】,单独报"含GT"与"允许无GT"两个数。

输出真实通过数 + d4-6/d7 拆分 + 剔除原因分布。
用法: python3 scripts/data/count_three_buckets_strict.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "pipeline"))
sys.path.insert(0, str(ROOT / "scripts" / "data"))
from filter_three_buckets import check_files, check_path_match, load_gt  # noqa: E402

LABELED = ROOT / "datasources" / "labeled" / "all_tasks_labeled.jsonl"


def load_bucket(bucket: str) -> list[tuple[str, str, int]]:
    recs = []
    for l in open(LABELED, encoding="utf-8"):
        if not l.strip():
            continue
        d = json.loads(l)
        if d.get("bucket") != bucket:
            continue
        did = d.get("D_id") or d.get("record_id") or ""
        if not did.startswith("D"):
            continue
        dv = d.get("difficulty")
        try:
            dv = int(dv)
        except Exception:  # noqa: BLE001
            continue
        if 4 <= dv <= 7:
            recs.append((did, d.get("user_prompt") or "", dv))
    return recs


def check_one(rec):
    did, q, dv = rec
    q_norm, bad_path = check_path_match(did, q)
    if bad_path:
        return (did, dv, "bad_path")
    ok_f, _ = check_files(did, q_norm)
    if not ok_f:
        return (did, dv, "missing_files")
    ok_gt, _ = load_gt(did)
    if not ok_gt:
        return (did, dv, "no_gt")
    return (did, dv, "pass")


def main() -> int:
    for bucket in ("office", "research", "ops"):
        recs = load_bucket(bucket)
        print(f"\n=== {bucket}: d4-7 母池 {len(recs)} 条,全量检查中… ===", flush=True)
        results = []
        with ThreadPoolExecutor(max_workers=48) as ex:
            for i, r in enumerate(ex.map(check_one, recs), 1):
                results.append(r)
                if i % 2000 == 0:
                    print(f"  ...{i}/{len(recs)}", flush=True)
        reason = Counter(r[2] for r in results)
        # pass 的难度拆分
        passed = [r for r in results if r[2] == "pass"]
        pd = Counter(r[1] for r in passed)
        d46 = sum(v for k, v in pd.items() if 4 <= k <= 6)
        d7 = pd.get(7, 0)
        # ops 允许无GT:pass + no_gt(但要文件/路径OK)
        pass_nogt = sum(1 for r in results if r[2] in ("pass", "no_gt"))
        print(f"  剔除原因: {dict(reason)}", flush=True)
        print(f"  【严格】全过(路径+文件+GT): {reason['pass']}  (d4-6={d46}, d7={d7})", flush=True)
        if bucket == "ops":
            print(f"  【允许无GT】(路径+文件OK,GT可缺): {pass_nogt}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
