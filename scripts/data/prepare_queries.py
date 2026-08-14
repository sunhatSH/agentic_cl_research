"""Stage ②: extract user queries per session from the raw dataset.

Input  (datasets/_stage_prefix_pass.jsonl): one JSON object per line = one
session. The conversation lives in ``record.messages``; user turns are the
``role == "user"`` messages.

Output (one JSON object per line = one session)::

    {"record_id": "record-...", "queries": ["q1", "q2", ...]}

These ``queries`` later drive sandbox rollout (doc/SandboxRollout.md): each
query is replayed as a simulated user request to collect GRPO trajectories.

Decisions (see chat 2026-06-10, configurable via flags):
  - ``<summary>`` compaction blocks are NOT queries -> dropped (--keep-summary
    to keep them). Only turns whose (lstripped) content starts with "<summary"
    are dropped; a normal turn that merely *embeds* a summary is kept.
  - Query text is kept RAW (system-reminders / "Sender (untrusted metadata)"
    envelopes left intact); cleaning is deferred to the rollout stage.
  - No bucket/category label is assigned here (the raw data has none);
    ``record_id`` is retained as the join key for a later bucket-tagging pass.

Usage::

    python scripts/data/prepare_queries.py \
        --input datasets/_stage_prefix_pass.jsonl \
        --output datasets/queries.jsonl
    python scripts/data/prepare_queries.py --limit 50 --output datasets/queries.sample.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path


def extract_user_text(content) -> str:
    """Flatten a message ``content`` (str | content-parts list | other) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict):
                t = p.get("text") or p.get("content")
                if isinstance(t, str):
                    parts.append(t)
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts)
    return "" if content is None else str(content)


def is_summary_block(text: str) -> bool:
    """True for compacted-history turns (content starts with a ``<summary`` tag)."""
    return text.lstrip().startswith("<summary")


def session_queries(record: dict, keep_summary: bool = False) -> list[str]:
    """All user queries in one session, in conversation order.

    Drops empty turns and (unless ``keep_summary``) ``<summary>`` compaction
    blocks.
    """
    out: list[str] = []
    for msg in record.get("messages", []) or []:
        if msg.get("role") != "user":
            continue
        text = extract_user_text(msg.get("content"))
        if not text.strip():
            continue
        if not keep_summary and is_summary_block(text):
            continue
        out.append(text)
    return out


def iter_sessions(path: str | Path) -> Iterator[dict]:
    """Stream one parsed session object per line (memory-safe for large files)."""
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] skipping malformed line {lineno}: {e}", file=sys.stderr)


def build_query_records(
    input_path: str | Path,
    keep_summary: bool = False,
    skip_empty: bool = False,
    limit: int | None = None,
) -> Iterator[dict]:
    """Yield ``{"record_id", "queries"}`` per session (the output line objects)."""
    n = 0
    for obj in iter_sessions(input_path):
        record = obj.get("record", {}) or {}
        queries = session_queries(record, keep_summary=keep_summary)
        if skip_empty and not queries:
            continue
        yield {"record_id": obj.get("record_id"), "queries": queries}
        n += 1
        if limit is not None and n >= limit:
            return


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="datasets/_stage_prefix_pass.jsonl")
    ap.add_argument("--output", default="datasets/queries.jsonl")
    ap.add_argument("--keep-summary", action="store_true", help="keep <summary> compaction blocks")
    ap.add_argument("--skip-empty", action="store_true", help="drop sessions with 0 queries")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N sessions")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sessions = 0
    total_q = 0
    empty = 0
    with out_path.open("w", encoding="utf-8") as out:
        for rec in build_query_records(
            args.input,
            keep_summary=args.keep_summary,
            skip_empty=args.skip_empty,
            limit=args.limit,
        ):
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            sessions += 1
            total_q += len(rec["queries"])
            if not rec["queries"]:
                empty += 1

    avg = total_q / sessions if sessions else 0.0
    print(
        f"[prepare_queries] sessions={sessions} queries={total_q} "
        f"avg/session={avg:.1f} empty_sessions={empty} -> {out_path}"
    )


if __name__ == "__main__":
    main()
