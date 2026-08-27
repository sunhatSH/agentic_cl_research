#!/usr/bin/env python3
"""构建 3200×2 数据集:coding + office,各 step 难度比例≈全桶比例。

用户口径(2026-08-27):
  - coding 3200:d4-6 优先,不足补 d7(其余不要)。来源:D类(过路径/文件/GT) + LH(自带GT自包含)
    + 可完成SWE(task_done=1&correctness>=0.5,自带rubric GT)。generalClaw 待造GT,默认不纳入。
  - office 3200:按 coding 的 d4-6/d7 比例,从 office d4-7(过检查)里选。
  - 每桶内用 _balanced_order:每个 batch(32)难度配比≈全桶配比(F6 最大余数法)。
  - batch32 对齐,record_id=D_id/SWE_/LH_,单 user message,GT 在 taskspecs answer_key。

用法:
  python3 scripts/data/build_3200x2.py               # 预览(不写)
  python3 scripts/data/build_3200x2.py --apply        # 落地 train_cl.parquet + jsonl
  python3 scripts/data/build_3200x2.py --include-gc-nogt --apply
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "pipeline"))
sys.path.insert(0, str(ROOT / "scripts" / "data"))
from filter_three_buckets import check_files, check_path_match, load_gt  # noqa: E402
from path_normalize import normalize_paths  # noqa: E402

TS = ROOT / "datasources" / "taskspecs_w3"
LABELED = ROOT / "datasources" / "labeled" / "all_tasks_labeled.jsonl"
PARQUET = ROOT / "datasets" / "train_cl.parquet"
JSONL = ROOT / "datasets" / "train_cl.jsonl"
SWE_PROBE = ROOT / "datasources" / "labeled" / "swe_completability.jsonl"
LH_DIFF = ROOT / "datasources" / "labeled" / "lh_difficulty.jsonl"
GC_CODING = ROOT / "datasources" / "labeled" / "claworiented_coding_candidates.jsonl"

TARGET = 3200
BATCH = 32


def _balanced_order(items: list[dict], seed: int = 42) -> list[dict]:
    """按难度分层、层内 shuffle、最大余数法交错,使每 batch(32)难度配比≈全桶配比(F6同口径)。"""
    rng = random.Random(seed)
    groups: dict[int, list[dict]] = {}
    for it in items:
        groups.setdefault(it["difficulty"], []).append(it)
    for d in groups:
        rng.shuffle(groups[d])
    total = len(items)
    diffs = sorted(groups)
    remaining = {d: len(groups[d]) for d in diffs}
    idx = {d: 0 for d in diffs}
    acc = {d: 0.0 for d in diffs}
    out: list[dict] = []
    for _ in range(total):
        best, best_acc = None, -1.0
        for d in diffs:
            if remaining[d] <= 0:
                continue
            acc[d] += len(groups[d]) / total
            if acc[d] > best_acc:
                best, best_acc = d, acc[d]
        acc[best] -= 1.0
        out.append(groups[best][idx[best]])
        idx[best] += 1
        remaining[best] -= 1
    return out


def _mk(did, difficulty, query):
    return {"did": did, "difficulty": int(difficulty), "query": query}


# ── coding 来源 ──
def coding_dpool() -> list[dict]:
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
        if bad or not check_files(did, qn)[0] or not load_gt(did)[0]:
            return None
        return _mk(did, dv, qn)
    out = []
    with ThreadPoolExecutor(max_workers=48) as ex:
        for r in ex.map(chk, recs):
            if r:
                out.append(r)
    return out


def coding_lh() -> list[dict]:
    diff = {json.loads(l)["rid"]: json.loads(l)["difficulty"] for l in open(LH_DIFF, encoding="utf-8")}
    rows = pq.read_table(str(PARQUET)).to_pylist()
    lh_q = {r["extra_info"]["gen_task_id"]: (r["extra_info"].get("queries") or [""])[0]
            for r in rows if r["extra_info"].get("gen_task_id", "").startswith("LH_")}
    out = []
    for rid, dv in diff.items():
        if 4 <= dv <= 7 and rid in lh_q:
            out.append(_mk(rid, dv, lh_q[rid]))
    return out


def coding_swe(threshold: float) -> list[dict]:
    if not SWE_PROBE.is_file():
        return []
    rows = pq.read_table(str(PARQUET)).to_pylist()
    meta = {r["extra_info"]["gen_task_id"]: (int(r["extra_info"].get("difficulty", 0) or 0),
                                             (r["extra_info"].get("queries") or [""])[0])
            for r in rows if r["extra_info"].get("gen_task_id", "").startswith("SWE_")}
    out = []
    for l in open(SWE_PROBE, encoding="utf-8"):
        if not l.strip():
            continue
        d = json.loads(l)
        if d.get("task_done") is None:
            continue
        if int(d["task_done"]) == 1 and float(d.get("correctness", 0)) >= threshold:
            dv, q = meta.get(d["swe_id"], (0, ""))
            if 4 <= dv <= 7:
                out.append(_mk(d["swe_id"], dv, q))
    return out


def coding_gc() -> list[dict]:
    out = []
    for l in open(GC_CODING, encoding="utf-8"):
        d = json.loads(l)
        dv = d.get("difficulty")
        if dv and 4 <= dv <= 7:
            out.append(_mk(d.get("task_id", ""), dv, normalize_paths(d.get("query", ""))))
    return out


def office_pool() -> list[dict]:
    """office d4-7 过路径/文件/GT。"""
    recs = []
    for l in open(LABELED, encoding="utf-8"):
        if not l.strip():
            continue
        d = json.loads(l)
        if d.get("bucket") != "office":
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
        if bad or not check_files(did, qn)[0] or not load_gt(did)[0]:
            return None
        return _mk(did, dv, qn)
    out = []
    with ThreadPoolExecutor(max_workers=48) as ex:
        for r in ex.map(chk, recs):
            if r:
                out.append(r)
    return out


def pick_coding(pool: list[dict], target: int) -> tuple[list[dict], dict]:
    """d4-6 优先→d7(去重 by did)。"""
    seen = set()
    d46, d7 = [], []
    for r in pool:
        if not r["did"] or r["did"] in seen:
            continue
        seen.add(r["did"])
        (d46 if 4 <= r["difficulty"] <= 6 else d7).append(r)
    sel = d46[:target]
    if len(sel) < target:
        sel += d7[: target - len(sel)]
    used = {"d4-6": sum(1 for r in sel if r["difficulty"] <= 6), "d7": sum(1 for r in sel if r["difficulty"] == 7)}
    return sel, used


def pick_office_by_ratio(pool: list[dict], target: int, n_d46: int, n_d7: int) -> tuple[list[dict], dict]:
    """按 coding 的 d4-6:d7 比例从 office 选 target 个。"""
    ratio_d46 = n_d46 / (n_d46 + n_d7)
    want_d46 = round(target * ratio_d46)
    want_d7 = target - want_d46
    seen = set()
    d46 = [r for r in pool if 4 <= r["difficulty"] <= 6 and (r["did"] not in seen and not seen.add(r["did"]))]
    seen = set()
    d7 = [r for r in pool if r["difficulty"] == 7 and (r["did"] not in seen and not seen.add(r["did"]))]
    # 数量不足则用另一档补
    sel_d46 = d46[:want_d46]
    sel_d7 = d7[:want_d7]
    # 补齐
    if len(sel_d46) < want_d46:
        sel_d7 += d7[want_d7: want_d7 + (want_d46 - len(sel_d46))]
    if len(sel_d7) < want_d7:
        sel_d46 += d46[want_d46: want_d46 + (want_d7 - len(sel_d7))]
    sel = (sel_d46 + sel_d7)[:target]
    used = {"want_d4-6": want_d46, "want_d7": want_d7,
            "got_d4-6": sum(1 for r in sel if r["difficulty"] <= 6),
            "got_d7": sum(1 for r in sel if r["difficulty"] == 7)}
    return sel, used


def to_row(r: dict, bucket: str) -> dict:
    q = r["query"]
    return {
        "prompt": [{"role": "user", "content": q}],
        "data_source": "agentic_cl",
        "reward_model": {"ground_truth": "", "style": "rule",
                         "reward_fn": {"_function_name": "trainer.model_reward_omni.compute_score"}},
        "bucket": bucket,
        "extra_info": {
            "record_id": r["did"], "bucket": bucket, "queries": [q], "persona": "",
            "available_tools": [], "missing_info_slots": [], "safety_constraints": [],
            "difficulty": str(r["difficulty"]), "gen_task_id": r["did"], "ws_dir": "",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--score-threshold", type=float, default=0.5)
    ap.add_argument("--include-gc-nogt", action="store_true")
    args = ap.parse_args()

    print("=== 加载 coding 来源 ===", flush=True)
    dp = coding_dpool()
    lh = coding_lh()
    swe = coding_swe(args.score_threshold)
    gc = coding_gc() if args.include_gc_nogt else []
    print(f"  D类{len(dp)} + LH{len(lh)} + 可完成SWE{len(swe)} + GC{len(gc)}", flush=True)
    coding_pool = dp + lh + swe + gc
    coding_sel, cu = pick_coding(coding_pool, TARGET)
    if len(coding_sel) < TARGET:
        print(f"⚠️ coding 只凑到 {len(coding_sel)}/{TARGET}(d4-6={cu['d4-6']} d7={cu['d7']})", flush=True)
    print(f"  coding 选中 {len(coding_sel)}: d4-6={cu['d4-6']} d7={cu['d7']} "
          f"(比例 {cu['d4-6']/max(len(coding_sel),1)*100:.0f}%/{cu['d7']/max(len(coding_sel),1)*100:.0f}%)", flush=True)

    print("\n=== 加载 office 来源 ===", flush=True)
    off_pool = office_pool()
    print(f"  office 过检查 {len(off_pool)}", flush=True)
    off_sel, ou = pick_office_by_ratio(off_pool, TARGET, cu["d4-6"], cu["d7"])
    print(f"  office 选中 {len(off_sel)}: 目标 d4-6={ou['want_d4-6']}/d7={ou['want_d7']} → "
          f"实得 d4-6={ou['got_d4-6']}/d7={ou['got_d7']}", flush=True)

    # batch32 对齐(两桶都截到 TARGET,已是 3200=100*32)
    coding_sel = coding_sel[: (len(coding_sel) // BATCH) * BATCH]
    off_sel = off_sel[: (len(off_sel) // BATCH) * BATCH]

    # _balanced_order 每桶
    coding_ord = _balanced_order(coding_sel)
    off_ord = _balanced_order(off_sel, seed=43)

    # 验证每 step 难度比例
    def step_check(items, name):
        steps = [items[i:i + BATCH] for i in range(0, len(items), BATCH)]
        d46_ratios = [sum(1 for r in s if r["difficulty"] <= 6) / len(s) for s in steps]
        print(f"  {name} {len(steps)} steps, 每step d4-6占比: min={min(d46_ratios):.2f} "
              f"max={max(d46_ratios):.2f} (全桶={sum(1 for r in items if r['difficulty']<=6)/len(items):.2f})", flush=True)
    print("\n=== 每 step 难度均衡验证 ===", flush=True)
    step_check(coding_ord, "coding")
    step_check(off_ord, "office")

    rows = [to_row(r, "coding") for r in coding_ord] + [to_row(r, "office") for r in off_ord]
    print(f"\n总行数 {len(rows)} (coding {len(coding_ord)} + office {len(off_ord)}), batch32 对齐={len(rows)%32==0}", flush=True)

    if not args.apply:
        print("\n加 --apply 落地", flush=True)
        return 0

    t = pq.read_table(str(PARQUET))
    new_t = pa.Table.from_pylist(rows, schema=t.schema)
    tmp = PARQUET.with_suffix(".parquet.tmp")
    pq.write_table(new_t, tmp)
    tmp.replace(PARQUET)
    with open(JSONL.with_suffix(".jsonl.tmp"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    JSONL.with_suffix(".jsonl.tmp").replace(JSONL)
    print(f"✅ 落地 {PARQUET.name} + {JSONL.name} ({len(rows)} rows)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
