#!/usr/bin/env python3
"""合并 L2(结构) + L3(语义) 质检结果，判废轨迹并导出重采集清单。

废轨迹 = 满足任一：
  1. L2 结构不合格且不可清洗（toolcall 参数不符 schema / 未定义工具 / 其他结构错）——
     已在清洗阶段修掉可修的（特殊 token / 缺 reasoning_content），剩下的是不可修结构废。
  2. L3 任意轮命中问题信号（rounds_checker.<round>.signals.<sig>.hit == True）。
  3. L3 completion_score 过低（quality_difficulty.completion.completion_score < 阈值）。

输入：
  --cleaned  清洗后的 cold_start_1429.cleaned.jsonl（行序 = warmup sqlite 顺序）
  --l3-dir   L3 输出目录（含 05_verification_grounding.jsonl 累积全轮结果）
  --l2-bad   L2 不合格行号 json（_l2_invalid_after_clean.json）
输出：
  --out      重采集清单 jsonl：{line, query, bucket, waste_from:[l2/l3信号], reasons}

query = 该轨迹首个 user 消息（重采集直接用它）。行序与 cleaned 文件对齐。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
_CLEANED = ROOT / "datasets" / "cold_start" / "cold_start_1429.cleaned.jsonl"
_L3_DIR = ROOT / "logs" / "l3_coldstart" / "cold_start_1429.cleaned"
_L2_BAD = ROOT / "datasets" / "cold_start" / "_l2_invalid_after_clean.json"
_OUT = ROOT / "datasets" / "cold_start" / "waste_requery.jsonl"

# completion_score 低于此判废（quality_difficulty 轮）。0.5 = 明显没完成。
_COMPLETION_MIN = 0.5


def _first_user_query(messages: list) -> str:
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, list):
                c = " ".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)
            if str(c).strip():
                return str(c).strip()
    return ""


def _l3_waste_reasons(rounds_checker: dict) -> list[str]:
    """从 rounds_checker 提取判废原因（命中信号 + 低 completion）。"""
    reasons = []
    for rk, rv in (rounds_checker or {}).items():
        if not isinstance(rv, dict):
            continue
        # 任意轮的 signals.<sig>.hit
        for sname, sv in (rv.get("signals") or {}).items():
            if isinstance(sv, dict) and sv.get("hit"):
                reasons.append(f"l3:{rk}.{sname}")
        # quality_difficulty 的 completion_score
        if rk == "quality_difficulty":
            comp = rv.get("completion")
            if isinstance(comp, dict):
                score = comp.get("completion_score")
                if isinstance(score, (int, float)) and score < _COMPLETION_MIN:
                    reasons.append(f"l3:low_completion({score})")
    return reasons


def main() -> None:
    global _COMPLETION_MIN

    ap = argparse.ArgumentParser()
    ap.add_argument("--cleaned", default=str(_CLEANED))
    ap.add_argument("--l3-dir", default=str(_L3_DIR))
    ap.add_argument("--l2-bad", default=str(_L2_BAD))
    ap.add_argument("--out", default=str(_OUT))
    ap.add_argument("--completion-min", type=float, default=_COMPLETION_MIN)
    args = ap.parse_args()

    _COMPLETION_MIN = args.completion_min

    # L2 不合格行号
    l2_bad = set()
    if Path(args.l2_bad).is_file():
        l2_bad = set(json.loads(Path(args.l2_bad).read_text()))
    print(f"L2 不可修废轨迹: {len(l2_bad)} 行")

    # L3 结果：按 record_id（row-00000014 = 行号）索引 waste reasons
    #   L3 的 record_id 是 row-{行号:08d}（05_verification_grounding.jsonl 含累积全轮）。
    l3_reasons: dict[int, list[str]] = {}
    l3_file = None
    for cand in ["05_verification_grounding.jsonl", "04_quality_difficulty.jsonl"]:
        p = Path(args.l3_dir) / cand
        if p.is_file():
            l3_file = p
            break
    l3_n = 0
    if l3_file:
        for line in open(l3_file, encoding="utf-8", errors="replace"):
            if not line.strip():
                continue
            l3_n += 1
            d = json.loads(line)
            rc = d.get("rounds_checker") or {}
            rid = rc.get("record_id", "")
            # row-00000014 → 14
            try:
                line_no = int(str(rid).replace("row-", ""))
            except ValueError:
                continue
            rs = _l3_waste_reasons(rc)
            if rs:
                l3_reasons[line_no] = rs
        print(f"L3 已判 {l3_n} 条，其中命中废信号 {len(l3_reasons)} 条")
    else:
        print(f"⚠️ L3 结果未找到（{args.l3_dir}），只用 L2 判废")

    # 合并：逐行读 cleaned，任一命中即废
    waste = []
    reason_counter: Counter = Counter()
    bucket_counter: Counter = Counter()
    with open(args.cleaned, encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh):
            if not line.strip():
                continue
            froms = []
            reasons = []
            if line_no in l2_bad:
                froms.append("l2")
                reasons.append("l2_structural_unfixable")
            if line_no in l3_reasons:
                froms.append("l3")
                reasons.extend(l3_reasons[line_no])
            if not froms:
                continue
            d = json.loads(line)
            md = d.get("metadata") or {}
            q = _first_user_query(d.get("messages") or [])
            for r in reasons:
                reason_counter[r.split("(")[0]] += 1
            bucket_counter[md.get("bucket")] += 1
            waste.append(
                {
                    "line": line_no,
                    "query": q,
                    "bucket": md.get("bucket"),
                    "query_index": md.get("query_index"),
                    "waste_from": froms,
                    "reasons": reasons,
                }
            )

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        for w in waste:
            f.write(json.dumps(w, ensure_ascii=False) + "\n")

    no_q = sum(1 for w in waste if not w["query"])
    print(f"\n合并废轨迹: {len(waste)} 条")
    print(f"  仅 L2: {sum(1 for w in waste if w['waste_from']==['l2'])}")
    print(f"  仅 L3: {sum(1 for w in waste if w['waste_from']==['l3'])}")
    print(f"  L2+L3: {sum(1 for w in waste if set(w['waste_from'])=={'l2','l3'})}")
    print(f"  无法提取 query（不能重采集）: {no_q}")
    print(f"  废因分布: {dict(reason_counter.most_common(15))}")
    print(f"  分桶: {dict(bucket_counter)}")
    print(f"→ 重采集清单: {out_path}")


if __name__ == "__main__":
    main()
