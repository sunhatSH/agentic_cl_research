#!/usr/bin/env python3
"""三桶(office/research/ops)过滤 → 各 6400 候选池。

用户口径(2026-08-27):
  1. 难度优先 d4-6;若 d4-6 不够 6400,补 d7。
  2. 检查路径与 query 匹配(能自动修的先归一化,真不匹配才剔)。
  3. 检查文件缺失:query 里点名的文件必须在 taskspecs_w3/<D_id>/files/ 真存在。
  4. 检查 GT:answer_key.json 有 checks 或 rubric。

桶差异:
  - office/research:D 类母池,taskspecs 全落盘 → 走完整四项检查。
  - ops:D 母池仅 771,主力 6850 来自 generalClaworiented SFT 轨迹(无 taskspecs/
    GT/files)。按用户口径【ops 允许无 GT】(后续补)→ ops 只做难度筛+路径归一化,
    跳过文件/GT 检查,generalClaw 轨迹也纳入。

输出:datasources/labeled/filtered_<bucket>.jsonl(每桶最多 6400,带来源与检查结果)。
用法:python3 scripts/data/filter_three_buckets.py [--target 6400]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "pipeline"))
from path_normalize import normalize_paths  # noqa: E402

TS = ROOT / "datasources" / "taskspecs_w3"
LABELED = ROOT / "datasources" / "labeled" / "all_tasks_labeled.jsonl"
GC_CACHE = ROOT / "datasources" / "labeled" / "claworiented_bucket_cache.jsonl"
GC_DIR = "/mnt/afs_toolcall/zhengnairong/code/data_mllm_agent/agent/generalClaworiented"
OUT_DIR = ROOT / "datasources" / "labeled"

# query 里引用文件名的正则(复用 replace_swe_with_longhorizon.find_bad_swe 的思路)
_FILE_RE = re.compile(
    r"[\w\-]+\.(?:csv|xlsx|xls|json|jsonl|sqlite|txt|pdf|docx|tsv|parquet|db|md|py|ts|js|"
    r"go|rs|java|cpp|c|rb|sh|yml|yaml|toml|ini|cfg|xml|html|css|sql|png|jpg|jpeg)\b",
    re.IGNORECASE,
)
# 产出型上下文关键词(query 提到某文件是"写/产出",不算缺失)
_OUT_KW = ["write", "output", "save", "create", "generate", "produce", "deliver",
           "写到", "输出", "保存", "生成", "创建", "导出", "写入", "产出", "汇总", "报告", "清单", "结果", "建议"]
# 产出型文件名模式(文件名本身就表明是产物,如 analysis_summary.json / *_report.xlsx)
_OUT_NAME_RE = re.compile(
    r"(summary|report|result|analysis|recommend|output|dashboard|_out\b|export|"
    r"汇总|报告|结果|建议|清单|方案)", re.IGNORECASE)


def load_gt(did: str) -> tuple[bool, str]:
    """answer_key.json 有 checks 或 rubric → (True, 类型)。"""
    ak = TS / did / "answer_key.json"
    if not ak.is_file():
        return False, "no_ak"
    try:
        d = json.loads(ak.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False, "bad_ak"
    checks = d.get("checks")
    rubric = d.get("rubric")
    n_checks = len(checks) if isinstance(checks, (list, dict)) else 0
    n_rubric = len(rubric) if isinstance(rubric, list) else (1 if rubric else 0)
    if n_checks or n_rubric:
        return True, f"checks{n_checks}/rubric{n_rubric}"
    return False, "empty_gt"


def check_files(did: str, query: str) -> tuple[bool, str]:
    """query 点名的文件是否都在 taskspecs_w3/<did>/files/ 存在(能修则修,不误杀)。

    2026-08-27 收紧误杀(仅影响判定,不改任务本身):
      - 产出型不算缺:上下文关键词(_OUT_KW) 或【文件名本身表明是产物】(_OUT_NAME_RE,
        如 analysis_summary.json / *_report.xlsx)——这些是 agent 要写的产出,不该在 files/。
      - 正则粘连修复:query 里 "以及automated_quality_checks.xlsx" 把中文粘进文件名,
        剥中文前缀后再比对;并用后缀匹配(实际文件名是 ref 的后缀 or 反之)兜层级/粘连差异。
    只有【真缺】(basename 任何层级找不到、且非产出型)才判失败。
    """
    fdir = TS / did / "files"
    actual = set()
    if fdir.is_dir():
        for p in fdir.rglob("*"):
            if p.is_file():
                actual.add(p.name)
    refs = {r for r in _FILE_RE.findall(query) if len(r) > 3 and not any(c in r for c in "*?")}
    missing = set()
    for m in refs:
        stripped = re.sub(r"^[一-鿿]+", "", m)  # 剥中文前缀(正则粘连)
        if m in actual or stripped in actual:
            continue
        if any(a.endswith(m) or m.endswith(a) for a in actual):  # 后缀匹配(层级/粘连)
            continue
        if _OUT_NAME_RE.search(m):  # 文件名本身是产物
            continue
        idx = query.find(m)
        ctx = query[max(0, idx - 50):idx + len(m) + 15].lower() if idx >= 0 else ""
        if any(k in ctx for k in _OUT_KW):  # 上下文表明是产出
            continue
        missing.add(m)
    if missing:
        return False, f"missing:{sorted(missing)[:3]}"
    return True, f"refs{len(refs)}"


def check_path_match(did: str, query_raw: str) -> tuple[str, bool]:
    """路径归一化(能修先修);返回 (归一化后 query, 是否含残留坏路径)。"""
    q = normalize_paths(query_raw)
    # 残留 Windows 盘符 / 未归一化的外部绝对路径 = 真坏
    bad = bool(re.search(r"[A-Za-z]:\\|/home/user/home|\\\\", q))
    return q, bad


def filter_dpool_bucket(bucket: str, target: int) -> list[dict]:
    """office/research/ops 的 D 母池部分:完整四项检查(ops 跳过 GT/文件)。

    AFS 上 files/ 检查慢(rglob ~1s/目录),故【难度优先排序后逐条检查,够 target 就停】,
    不全量扫。顺序:先 d4-6(打乱保多样),再 d7,再其余。
    """
    skip_gt_files = bucket == "ops"
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
        recs.append({"did": did, "query": d.get("user_prompt") or "", "difficulty": dv})

    # 只保留 d4-7(用户口径:d4-6 优先→加d7;d8+ 与 d1-3 不要)
    recs = [r for r in recs if 4 <= r["difficulty"] <= 7]
    # 难度分层排序:d4-6 → d7(层内按 did 稳定序)
    recs.sort(key=lambda r: (0 if 4 <= r["difficulty"] <= 6 else 1, r["did"]))

    passed = []
    stat = Counter()
    checked = 0
    for r in recs:
        if len(passed) >= target:
            break  # 够了就停,省 AFS IO
        did, q, dv = r["did"], r["query"], r["difficulty"]
        checked += 1
        q_norm, bad_path = check_path_match(did, q)
        if bad_path:
            stat["bad_path"] += 1
            continue
        if not skip_gt_files:
            ok_f, why_f = check_files(did, q_norm)
            if not ok_f:
                stat["missing_files"] += 1
                continue
            ok_gt, why_gt = load_gt(did)
            if not ok_gt:
                stat["no_gt"] += 1
                continue
        else:
            why_gt = "skipped(ops)"
        passed.append({"src": "dpool", "did": did, "bucket": bucket, "difficulty": dv,
                       "query": q_norm, "gt": why_gt if not skip_gt_files else ""})
        stat["passed"] += 1
    print(f"  [{bucket}/dpool] 母池{len(recs)} 检查{checked} → 通过{stat['passed']} "
          f"(坏路径{stat['bad_path']} 缺文件{stat['missing_files']} 无GT{stat['no_gt']})", flush=True)
    return passed


def load_gc_bucket(bucket: str) -> list[dict]:
    """ops 的 generalClaw 部分(无 GT/文件,只做难度需回源判)——本轮先只标记,不判难度。

    generalClaw 记录难度未打(coding 才打了)。ops 若要用需另跑难度。这里先纳入池、
    标 difficulty=None,由后续难度筛决定;本轮 target 优先用 dpool 能过检查的填。
    """
    keys = []
    for l in open(GC_CACHE, encoding="utf-8"):
        d = json.loads(l)
        if d.get("bucket") == bucket:
            keys.append((d["src"], d["row"]))
    return keys


def select_by_difficulty(passed: list[dict], target: int) -> tuple[list[dict], dict]:
    """难度优先 d4-6,不够加 d7。【只到 d7 为止】——d8+ 不补(用户口径:d4-6 优先→加d7)。

    不足 target 就如实返回不足,不用 d8+/d1-3 凑数。
    """
    d46 = [r for r in passed if 4 <= r["difficulty"] <= 6]
    d7 = [r for r in passed if r["difficulty"] == 7]
    sel = list(d46)
    used = {"d4-6": len(sel), "d7": 0}
    if len(sel) < target:
        add = d7[: target - len(sel)]
        sel += add
        used["d7"] = len(add)
    return sel[:target], used


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=6400)
    args = ap.parse_args()

    summary = {}
    for bucket in ("office", "research", "ops"):
        print(f"\n=== {bucket} ===", flush=True)
        passed = filter_dpool_bucket(bucket, args.target)
        sel, used = select_by_difficulty(passed, args.target)
        # 输出
        out = OUT_DIR / f"filtered_{bucket}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for r in sel:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        gc_avail = len(load_gc_bucket(bucket)) if bucket == "ops" else 0
        summary[bucket] = {"dpool_passed": len(passed), "selected": len(sel),
                           "diff_used": used, "gc_available": gc_avail}
        print(f"  选中 {len(sel)}/{args.target}  难度构成 {used}", flush=True)
        if bucket == "ops":
            print(f"  (ops generalClaw 池另有 {gc_avail} 条无GT轨迹,本轮未纳入难度筛;"
                  f"dpool 能过检查的仅 {len(passed)})", flush=True)
        print(f"  → {out}", flush=True)

    print("\n=== 汇总 ===", flush=True)
    for b, s in summary.items():
        enough = "✅" if s["selected"] >= args.target else "❌ 差 %d" % (args.target - s["selected"])
        print(f"  {b}: 选中 {s['selected']}/{args.target} {enough}  难度 {s['diff_used']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
