#!/usr/bin/env python3
"""合并 coding 桶最终候选:可完成 SWE + D类 + LH + generalClaw。

前置:SWE 体检(probe_swe_completability.py)跑完,产出 swe_completability.jsonl
(每条含 task_done/correctness/score)。

流程:
  1. 筛【可完成 SWE】:task_done==1 且 correctness>=阈值(默认0.5),自带 rubric GT。
  2. D类 coding:d4-7 + 过路径/文件/GT 检查(严格),复用 filter 逻辑。
  3. LH:全量池 d4-7(自带 GT、自包含)。
  4. generalClaw coding:d4-7(GT 待造,标记出来)。
  5. 难度优先 d4-6→d7,合并去重,输出统一格式 filtered_coding.jsonl。
  6. 报告:总量、各来源、难度分布、距 6400 缺口、待造 GT 的条数。

用法:
  python3 scripts/data/merge_coding_final.py                 # 汇总+出清单
  python3 scripts/data/merge_coding_final.py --score-threshold 0.6
  python3 scripts/data/merge_coding_final.py --target 6400 --include-gc-nogt
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "pipeline"))
sys.path.insert(0, str(ROOT / "scripts" / "data"))
from filter_three_buckets import check_files, check_path_match, load_gt  # noqa: E402

TS = ROOT / "datasources" / "taskspecs_w3"
LABELED = ROOT / "datasources" / "labeled" / "all_tasks_labeled.jsonl"
PARQUET = ROOT / "datasets" / "train_cl.parquet"
SWE_PROBE = ROOT / "datasources" / "labeled" / "swe_completability.jsonl"
LH_DIFF = ROOT / "datasources" / "labeled" / "lh_difficulty.jsonl"
GC_CODING = ROOT / "datasources" / "labeled" / "claworiented_coding_candidates.jsonl"
OUT = ROOT / "datasources" / "labeled" / "filtered_coding.jsonl"


def rec(did, difficulty, query, gt, src, need_gt=False):
    return {"src": src, "did": did, "bucket": "coding", "difficulty": difficulty,
            "query": query, "gt": gt, "need_gt": need_gt}


def load_completable_swe(threshold: float) -> tuple[list[dict], dict]:
    """筛 task_done==1 且 correctness>=阈值 的 SWE。query+难度取自 parquet。"""
    if not SWE_PROBE.is_file():
        return [], {"probe": "缺 swe_completability.jsonl,SWE 体检未跑"}
    verdict = {}
    for l in open(SWE_PROBE, encoding="utf-8"):
        if not l.strip():
            continue
        d = json.loads(l)
        if d.get("task_done") is None:  # judge_failed / error → 不算
            continue
        verdict[d["swe_id"]] = (int(d["task_done"]) == 1 and float(d.get("correctness", 0)) >= threshold,
                                d.get("score"))
    # parquet 取 SWE query + 难度
    rows = pq.read_table(str(PARQUET)).to_pylist()
    swe_meta = {}
    for r in rows:
        gid = r["extra_info"].get("gen_task_id", "")
        if gid.startswith("SWE_"):
            swe_meta[gid] = (int(r["extra_info"].get("difficulty", 0) or 0),
                             (r["extra_info"].get("queries") or [""])[0])
    out = []
    for sid, (ok, _score) in verdict.items():
        if not ok:
            continue
        dv, q = swe_meta.get(sid, (0, ""))
        if not (4 <= dv <= 7):
            continue  # 只要 d4-7
        # SWE 自带 rubric GT
        ok_gt, why = load_gt(sid)
        out.append(rec(sid, dv, q, why if ok_gt else "swe_rubric", "swe_completable"))
    stat = {"probed": len(verdict),
            "completable(task_done+corr>=%.1f)" % threshold: sum(1 for v in verdict.values() if v[0]),
            "selected_d4-7": len(out)}
    return out, stat


def load_dpool_coding() -> list[dict]:
    """D类 coding d4-7,严格过路径/文件/GT。"""
    recs = []
    for l in open(LABELED, encoding="utf-8"):
        if not l.strip():
            continue
        d = json.loads(l)
        if d.get("bucket") != "coding":
            continue
        did = d.get("D_id") or ""
        if not did.startswith("D"):
            continue
        dv = d.get("difficulty")
        try:
            dv = int(dv)
        except Exception:  # noqa: BLE001
            continue
        if 4 <= dv <= 7:
            recs.append((did, d.get("user_prompt") or "", dv))

    def chk(r):
        did, q, dv = r
        qn, bad = check_path_match(did, q)
        if bad:
            return None
        if not check_files(did, qn)[0]:
            return None
        ok, why = load_gt(did)
        if not ok:
            return None
        return rec(did, dv, qn, why, "dpool")
    out = []
    with ThreadPoolExecutor(max_workers=48) as ex:
        for r in ex.map(chk, recs):
            if r:
                out.append(r)
    return out


def load_lh() -> list[dict]:
    """LH 全量池 d4-7(自带 GT、自包含)。query 取自 parquet(已入的)或 taskspec。"""
    diff = {}
    for l in open(LH_DIFF, encoding="utf-8"):
        d = json.loads(l)
        diff[d["rid"]] = d["difficulty"]
    # parquet 里已有的 LH query
    rows = pq.read_table(str(PARQUET)).to_pylist()
    lh_q = {r["extra_info"]["gen_task_id"]: (r["extra_info"].get("queries") or [""])[0]
            for r in rows if r["extra_info"].get("gen_task_id", "").startswith("LH_")}
    out = []
    for rid, dv in diff.items():
        if not (4 <= dv <= 7):
            continue
        q = lh_q.get(rid, "")
        ok_gt, why = load_gt(rid)
        out.append(rec(rid, dv, q, why if ok_gt else "lh_checks+rubric", "lh"))
    return out


def load_gc(include_nogt: bool) -> list[dict]:
    """generalClaw coding d4-7。GT 待造(need_gt=True)。"""
    out = []
    for l in open(GC_CODING, encoding="utf-8"):
        d = json.loads(l)
        dv = d.get("difficulty")
        if not dv or not (4 <= dv <= 7):
            continue
        # GC 无 taskspecs 落盘,GT 待造
        out.append(rec(d.get("task_id", ""), dv, d.get("query", ""), "TODO_make_gt", "generalClaw", need_gt=True))
    return out if include_nogt else out  # 始终返回,由主流程决定是否用


def select(pool: list[dict], target: int) -> tuple[list[dict], dict]:
    """难度 d4-6 优先→d7;来源优先级:自带GT的(swe/dpool/lh) 先于 待造GT的(gc)。"""
    # 去重(按 did)
    seen = set()
    uniq = []
    for r in pool:
        if r["did"] and r["did"] not in seen:
            seen.add(r["did"])
            uniq.append(r)
    # 排序键:难度层(d4-6=0,d7=1) → GT层(有=0,待造=1) → did
    uniq.sort(key=lambda r: (0 if 4 <= r["difficulty"] <= 6 else 1, 1 if r["need_gt"] else 0, r["did"]))
    sel = uniq[:target]
    used = {"d4-6": sum(1 for r in sel if 4 <= r["difficulty"] <= 6),
            "d7": sum(1 for r in sel if r["difficulty"] == 7),
            "need_gt": sum(1 for r in sel if r["need_gt"])}
    return sel, used


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-threshold", type=float, default=0.5)
    ap.add_argument("--target", type=int, default=6400)
    ap.add_argument("--include-gc-nogt", action="store_true", help="纳入 generalClaw 待造GT 的 coding")
    args = ap.parse_args()

    print("=== 加载各来源 ===", flush=True)
    swe, swe_stat = load_completable_swe(args.score_threshold)
    print(f"  可完成 SWE: {len(swe)}  {swe_stat}", flush=True)
    dpool = load_dpool_coding()
    print(f"  D类 coding(严格): {len(dpool)}", flush=True)
    lh = load_lh()
    print(f"  LH(d4-7): {len(lh)}", flush=True)
    gc = load_gc(args.include_gc_nogt)
    print(f"  generalClaw coding(待造GT): {len(gc)}  {'纳入' if args.include_gc_nogt else '不纳入(加 --include-gc-nogt 启用)'}", flush=True)

    pool = swe + dpool + lh + (gc if args.include_gc_nogt else [])
    sel, used = select(pool, args.target)

    # batch32 对齐
    n32 = (len(sel) // 32) * 32
    sel_aligned = sel[:n32]

    with open(OUT, "w", encoding="utf-8") as f:
        for r in sel_aligned:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    src_c = Counter(r["src"] for r in sel_aligned)
    diff_c = Counter(r["difficulty"] for r in sel_aligned)
    print(f"\n=== coding 最终清单 ===", flush=True)
    print(f"  候选池合计: {len(pool)} → 去重排序取 {len(sel)} → batch32 对齐 {len(sel_aligned)}", flush=True)
    print(f"  距 {args.target} 缺口: {args.target - len(sel_aligned)}", flush=True)
    print(f"  难度: d4-6={used['d4-6']}, d7={used['d7']}", flush=True)
    print(f"  按来源: {dict(src_c)}", flush=True)
    print(f"  待造 GT(generalClaw): {used['need_gt']}", flush=True)
    print(f"  → {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
