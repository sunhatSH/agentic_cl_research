#!/usr/bin/env python3
"""Convert raw agentic JSONL -> verl rl_dataset parquet (Gap B).

Input : datasets/_stage_prefix_pass.jsonl
        one line = {record_id, record:{messages:[...], meta:{...}, tools:[...]}, ...}
        each record is ONE multi-turn agentic session (system + many user/assistant/tool turns).
Output: datasets/train.parquet + datasets/val.parquet (verl rl_dataset columns)

verl row schema (with data.return_raw_chat=true):
    prompt        list[{role, content}]  -- chat to roll out FROM (system + first user query)
    data_source   str                    -- "agentic_cl" (compute_score arg 1)
    reward_model  {ground_truth: str}    -- empty; rule reward uses checkers/robustness (Gap A)
    extra_info    {record_id, source, bucket, queries, num_user_turns, checkers, ...}

The pure helpers (pick_prompt / extract_queries / bucket_hint / extract_checkers /
split_assignment) are unit-tested off-disk; ``main`` only does IO + stats.

Notes:
  - bucket here is a best-effort keyword HINT for offline stats. The authoritative
    bucket is emitted by the model at rollout time (trainer/domain_tagging) or
    backfilled by trajectory_adapter; unresolved -> "unknown" (buffer skips it, B12).
  - checker extraction is conservative (Plan B.3): only emit when the final answer
    has a clearly-bounded token; otherwise leave empty and report coverage. Do NOT
    block training on hand-labeling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# Best-effort keyword -> bucket hints (offline stats only; not authoritative).
_BUCKET_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("finance", ("股", "财报", "财务", "营收", "stock", "revenue", "invoice", "账", "交易", "预算")),
    ("ops", ("docker", "服务器", "部署", "命令", "脚本", "日志", "config", "ssh", "运维", "kubectl")),
    ("office", ("excel", "表格", "ppt", "word", "文档", "会议", "邮件", "csv", "spreadsheet")),
    ("communication", ("翻译", "润色", "撰写", "草拟", "回复邮件", "translate", "rewrite", "draft",
                       "澄清", "追问", "继续", "follow up", "clarify")),
    ("qa", ("分析", "总结", "检索", "research", "analyze", "summari", "解释")),
    ("workflow", ("流程", "步骤", "计划", "编排", "workflow", "pipeline", "automate")),
    ("safety", ("安全", "合规", "漏洞", "威胁", "审计", "security", "compliance", "safety")),
    ("coding", ("代码", "编程", "debug", "review", "重构", "refactor", "function", "class")),
    ("research", ("调研", "报告", "综合", "简报", "摘要", "synthesis", "report")),
]

# A "bounded answer" token worth turning into a regex checker: a number (incl.
# thousands/decimal/percent) or a back-quoted code/path token.
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?%?")
_BACKTICK_RE = re.compile(r"`([^`\n]{2,60})`")


def _record_messages(rec: Mapping[str, Any]) -> list[dict[str, Any]]:
    inner = rec.get("record", rec)
    msgs = inner.get("messages") if isinstance(inner, Mapping) else None
    return list(msgs) if isinstance(msgs, list) else []


def pick_prompt(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Leading chat to roll out from: system message(s) + the first user turn."""
    prompt: list[dict[str, Any]] = []
    seen_user = False
    for m in messages:
        role = m.get("role")
        if role == "system" and not seen_user:
            prompt.append({"role": "system", "content": str(m.get("content", ""))})
        elif role == "user":
            prompt.append({"role": "user", "content": str(m.get("content", ""))})
            seen_user = True
            break
    return prompt


def extract_queries(messages: Sequence[Mapping[str, Any]]) -> list[str]:
    """All user turns in order = the session's queries (one 'queries' batch)."""
    return [str(m.get("content", "")) for m in messages if m.get("role") == "user"]


def _final_answer(messages: Sequence[Mapping[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "assistant":
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                return c
    return ""


def bucket_hint(messages: Sequence[Mapping[str, Any]], meta: Mapping[str, Any] | None = None) -> str | None:
    """Best-effort keyword bucket hint, or None (-> 'unknown')."""
    text_parts: list[str] = []
    for m in messages:
        if m.get("role") in ("user", "assistant"):
            c = m.get("content")
            if isinstance(c, str):
                text_parts.append(c)
    blob = "\n".join(text_parts).lower()
    if not blob:
        return None
    scores: Counter[str] = Counter()
    for bucket, kws in _BUCKET_KEYWORDS:
        for kw in kws:
            if kw.lower() in blob:
                scores[bucket] += 1
    if not scores:
        return None
    return scores.most_common(1)[0][0]


def extract_checkers(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Conservative offline checker extraction from the final answer.

    Emits at most one regex checker keyed on a clearly-bounded token (number or
    back-quoted path/code) so reward has a verifiable target. Returns [] when no
    safe target is found (caller reports coverage; robustness still applies).
    """
    final = _final_answer(messages)
    if not final:
        return []
    tail = final[-400:]  # answers usually conclude at the end
    m = _BACKTICK_RE.search(tail)
    if m:
        token = m.group(1)
        return [{"type": "regex", "target": "final_answer", "pattern": re.escape(token), "weight": 1.0}]
    nums = _NUMBER_RE.findall(tail)
    nums = [n for n in nums if len(n.strip(",")) >= 2]
    if nums:
        token = nums[-1]
        return [{"type": "regex", "target": "final_answer", "pattern": re.escape(token), "weight": 1.0}]
    return []


def split_assignment(record_id: str, val_fraction: float = 0.02) -> str:
    """Stable train/val split by hashing record_id (reproducible)."""
    h = hashlib.sha256(record_id.encode("utf-8")).hexdigest()
    bucket = (int(h[:8], 16) % 10000) / 10000.0
    return "val" if bucket < val_fraction else "train"


def record_to_row(rec: Mapping[str, Any]) -> dict[str, Any] | None:
    """Build one verl rl_dataset row from a raw record, or None to skip."""
    messages = _record_messages(rec)
    if not messages:
        return None
    prompt = pick_prompt(messages)
    if not any(p["role"] == "user" for p in prompt):
        return None  # no query to roll out from
    record_id = str(rec.get("record_id", ""))
    inner = rec.get("record", rec)
    meta = inner.get("meta", {}) if isinstance(inner, Mapping) else {}
    queries = extract_queries(messages)
    bucket = bucket_hint(messages, meta) or "unknown"
    checkers = extract_checkers(messages)
    return {
        "prompt": prompt,
        "data_source": "agentic_cl",
        "reward_model": {"ground_truth": "", "style": "rule"},
        "extra_info": {
            "record_id": record_id,
            "source": str(meta.get("source", "")) if isinstance(meta, Mapping) else "",
            "bucket": bucket,
            "queries": queries,
            "num_user_turns": len(queries),
            "checkers": checkers,
        },
    }


def convert(input_path: Path, out_dir: Path, val_fraction: float, limit: int | None) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = {"train": [], "val": []}
    stats = {
        "total": 0,
        "skipped": 0,
        "buckets": Counter(),
        "with_checkers": 0,
        "user_turns": 0,
    }
    with input_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                stats["skipped"] += 1
                continue
            row = record_to_row(rec)
            if row is None:
                stats["skipped"] += 1
                continue
            info = row["extra_info"]
            stats["buckets"][info["bucket"]] += 1
            stats["user_turns"] += info["num_user_turns"]
            if info["checkers"]:
                stats["with_checkers"] += 1
            split = split_assignment(info["record_id"], val_fraction)
            rows[split].append(row)

    out_dir.mkdir(parents=True, exist_ok=True)

    def _write(split: str) -> Path:
        data = rows[split]
        # Store complex fields as JSON strings? No -- verl reads dict/list cells.
        # pyarrow infers struct/list types from python objects.
        table = pa.Table.from_pylist(
            [
                {
                    "prompt": r["prompt"],
                    "data_source": r["data_source"],
                    "reward_model": r["reward_model"],
                    "extra_info": {
                        **r["extra_info"],
                        # pyarrow needs homogeneous types; serialize nested checkers/queries
                        "checkers": json.dumps(r["extra_info"]["checkers"], ensure_ascii=False),
                        "queries": json.dumps(r["extra_info"]["queries"], ensure_ascii=False),
                    },
                }
                for r in data
            ]
        )
        path = out_dir / f"{split}.parquet"
        pq.write_table(table, path)
        return path

    train_path = _write("train")
    val_path = _write("val")
    kept = len(rows["train"]) + len(rows["val"])
    stats["kept"] = kept
    stats["train"] = len(rows["train"])
    stats["val"] = len(rows["val"])
    stats["checker_coverage"] = (stats["with_checkers"] / kept) if kept else 0.0
    stats["avg_user_turns"] = (stats["user_turns"] / kept) if kept else 0.0
    stats["train_path"] = str(train_path)
    stats["val_path"] = str(val_path)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="datasets/_stage_prefix_pass.jsonl")
    ap.add_argument("--out-dir", default="datasets")
    ap.add_argument("--val-fraction", type=float, default=0.02)
    ap.add_argument("--limit", type=int, default=None, help="cap records (debug)")
    args = ap.parse_args()

    stats = convert(Path(args.input), Path(args.out_dir), args.val_fraction, args.limit)
    print("=== convert_dataset stats ===")
    print(f"total read   : {stats['total']}")
    print(f"kept         : {stats['kept']}  (train {stats['train']} / val {stats['val']})")
    print(f"skipped      : {stats['skipped']}")
    print(f"avg user turns/record : {stats['avg_user_turns']:.1f}")
    print(f"checker coverage      : {stats['checker_coverage']:.1%}  ({stats['with_checkers']} records)")
    print("bucket distribution (hint):")
    for b, c in stats["buckets"].most_common():
        print(f"  {b:14s} {c}")
    print(f"-> {stats['train_path']}")
    print(f"-> {stats['val_path']}")


if __name__ == "__main__":
    main()
