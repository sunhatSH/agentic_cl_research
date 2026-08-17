#!/usr/bin/env python3
"""从冷启动采集文件判废轨迹，导出待重采集的 query 清单。

废轨迹定义（用户口径 2026-08-17）：
  1. 无效：工具调用全部失败（tool 返回全含 error / 无成功 output），或最终 assistant
     用放弃性措辞收尾（"无法完成/失败/做不到"）且无有效产出。
  2. 被截断：末尾 assistant 带 dangling tool_call（无对应 tool 返回），或末尾 assistant
     content 空且无 tool_calls（被 length 截断）。
  3. 没跑完：末尾停在 tool 返回（无 assistant 收尾）。

输入：datasets/cold_start/cold_start_1429.jsonl（1429 行，warmup sqlite 的采集源，1:1 对应）
输出：datasets/cold_start/waste_requery.jsonl —— 每行 {line, query, bucket, reasons, query_index}
      query = 该轨迹首个 user 消息（重采集直接用它，不需外部 query 文件）。

⚠️ tool 的 success 字段不可靠（实测 success=True 但 content 是 {"error": ...}），
   故按 content 判定工具成败，不看 success 标志。

用法：python scripts/collect/export_waste_requery.py [--in <jsonl>] [--out <jsonl>]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
_IN = ROOT / "datasets" / "cold_start" / "cold_start_1429.jsonl"
_OUT = ROOT / "datasets" / "cold_start" / "waste_requery.jsonl"

# 放弃性措辞（最终 assistant 说做不到/失败/超时）。
_GIVE_UP = re.compile(
    r"无法完成|无法继续|无法执行|做不到|抱歉[，,].{0,20}(无法|不能|失败)|"
    r"failed to|cannot complete|i (cannot|can'?t|am unable)|unable to (complete|proceed)|超时|timeout|放弃",
    re.I,
)


def _tool_failed(m: dict) -> bool:
    """tool 消息是否失败（按 content 判，不信 success 字段）。"""
    c = str(m.get("content", ""))
    cl = c.lower()
    # 明确成功标志
    if '"status": "success"' in cl or '"error": null' in cl or '"exit_code": 0' in cl:
        return False
    # 明确错误标志
    if '"error"' in cl and '"error": null' not in cl:
        return True
    if '"status": "error"' in cl or '"status": "failed"' in cl:
        return True
    # 兜底：开头就报 error/traceback 且没 output
    if ("error" in cl[:120] or "traceback" in cl[:120]) and '"output"' not in cl:
        return True
    return False


def classify_waste(messages: list[dict]) -> list[str]:
    """返回该轨迹命中的废因标签列表（空 = 不是废轨迹）。"""
    msgs = [m for m in messages if isinstance(m, dict)]
    if not msgs:
        return ["empty"]
    tools = [m for m in msgs if m.get("role") == "tool"]
    last = msgs[-1]
    reasons: list[str] = []

    # 2. 被截断
    if last.get("role") == "assistant":
        if last.get("tool_calls"):
            reasons.append("truncated_dangling_toolcall")
        elif not str(last.get("content", "")).strip():
            reasons.append("truncated_empty_assistant")
    # 3. 没跑完：末尾停在 tool
    if last.get("role") == "tool":
        reasons.append("unfinished_ends_on_tool")
    # 1a. 无效：有工具调用但全失败
    if tools and all(_tool_failed(m) for m in tools):
        reasons.append("invalid_all_tools_failed")
    # 1b. 无效：最终 assistant 放弃性措辞
    final_asst = next(
        (m for m in reversed(msgs) if m.get("role") == "assistant" and str(m.get("content", "")).strip()),
        None,
    )
    if final_asst and _GIVE_UP.search(str(final_asst.get("content", ""))):
        reasons.append("invalid_giveup")
    return reasons


def first_user_query(messages: list[dict]) -> str:
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, list):
                c = " ".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)
            if str(c).strip():
                return str(c).strip()
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(_IN))
    ap.add_argument("--out", default=str(_OUT))
    args = ap.parse_args()

    waste = []
    total = 0
    reason_counter: Counter = Counter()
    bucket_counter: Counter = Counter()
    no_query = 0
    with open(args.inp, encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh):
            if not line.strip():
                continue
            total += 1
            d = json.loads(line)
            msgs = d.get("messages") or []
            md = d.get("metadata") or {}
            reasons = classify_waste(msgs)
            if not reasons:
                continue
            q = first_user_query(msgs)
            if not q:
                no_query += 1
            for r in reasons:
                reason_counter[r] += 1
            bucket_counter[md.get("bucket")] += 1
            waste.append(
                {
                    "line": line_no,
                    "query": q,
                    "bucket": md.get("bucket"),
                    "query_index": md.get("query_index"),
                    "reasons": reasons,
                }
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for w in waste:
            f.write(json.dumps(w, ensure_ascii=False) + "\n")

    print(f"总轨迹 {total} 条，废轨迹 {len(waste)} 条 ({100*len(waste)/total:.1f}%)")
    print(f"废因分布（一条可命中多类）: {dict(reason_counter)}")
    print(f"分桶: {dict(bucket_counter)}")
    print(f"无法提取 query 的废轨迹: {no_query}（这些无法重采集，需人工看）")
    print(f"→ 清单落盘: {out_path}")


if __name__ == "__main__":
    main()
