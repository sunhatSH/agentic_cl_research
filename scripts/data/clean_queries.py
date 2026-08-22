"""Batch-clean a queries JSONL file.

Input:  one JSON line per session, {"record_id": "...", "queries": ["q1", "q2", ...]}
Output: same format, with each query stripped of zero-width chars and sessions
        with garbled queries dropped.

Usage:
    python scripts/data/clean_queries.py \
        --input datasets/queries.jsonl \
        --output datasets/queries_clean.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasources.cleaning import clean_query


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch-clean a queries JSONL file.")
    ap.add_argument("--input", required=True, help="input queries JSONL")
    ap.add_argument("--output", required=True, help="output cleaned queries JSONL")
    ap.add_argument("--log-every", type=int, default=5000)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = kept = 0
    queries_dropped = 0
    with open(args.input, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            queries = obj.get("queries") or []
            cleaned = []
            for q in queries:
                cq = clean_query(q)
                if cq is not None:
                    cleaned.append(cq)
                else:
                    queries_dropped += 1
            if not cleaned:
                continue
            obj["queries"] = cleaned
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1
            if kept % args.log_every == 0:
                print(
                    f"[clean_queries] {kept}/{total} kept, " f"{queries_dropped} queries dropped", flush=True
                )

    print(
        f"[clean_queries] DONE: {kept}/{total} sessions kept, "
        f"{queries_dropped} queries dropped -> {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
