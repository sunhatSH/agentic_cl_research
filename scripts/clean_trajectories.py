#!/usr/bin/env python3
"""Post-collection garbled-character filter for trajectory JSONL.

Reads a trajectory JSONL, runs ``data.cleaning.clean_messages()`` per line,
and drops trajectories whose aggregate garble ratio exceeds the threshold.
Produces a cleaned JSONL and a sidecar drop log.

The C++ ``strip_zw`` binary handles zero-width stripping (run separately
before or after this step).  This script handles GARBLED character detection,
which requires per-codepoint Unicode classification that is easier to express
in Python.

Usage:
    .venv/bin/python scripts/clean_trajectories.py \
        --input rollouts/cold_start/grpo_hermes.jsonl \
        --output rollouts/cold_start/grpo_hermes_clean.jsonl \
        --garble-threshold 0.05
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from data.cleaning import clean_messages


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="trajectory JSONL to clean")
    ap.add_argument("--output", required=True, help="cleaned output JSONL")
    ap.add_argument("--garble-threshold", type=float, default=0.05,
                    help="max allowed garble ratio per trajectory (default 0.05)")
    ap.add_argument("--drop-log", default=None,
                    help="sidecar JSONL recording dropped trajectories (default <output>.dropped.jsonl)")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    drop_log = Path(args.drop_log) if args.drop_log else out.with_suffix(".dropped.jsonl")

    total = 0
    kept = 0
    dropped = 0

    with open(inp, encoding="utf-8") as fh_in, \
         open(out, "w", encoding="utf-8") as fh_out, \
         open(drop_log, "w", encoding="utf-8") as fh_drop:

        for line in fh_in:
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                traj = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                fh_drop.write(json.dumps({"query_index": -1, "reason": "json_parse_error"}, ensure_ascii=False) + "\n")
                continue

            result = clean_messages(
                traj.get("messages", []),
                garble_threshold=args.garble_threshold,
                single_msg_threshold=0.20,
                single_msg_policy="drop_all",
            )

            traj["messages"] = result.messages
            traj["_cleaned"] = {
                "total_chars": result.total_chars,
                "garble_chars": result.garble_chars,
                "dropped_by_garbled": result.dropped,
            }

            if result.dropped:
                dropped += 1
                fh_drop.write(json.dumps({
                    "query_index": traj.get("query_index"),
                    "bucket": traj.get("bucket"),
                    "reason": result.drop_reason,
                    "garble_chars": result.garble_chars,
                    "total_chars": result.total_chars,
                }, ensure_ascii=False) + "\n")
                continue

            fh_out.write(json.dumps(traj, ensure_ascii=False) + "\n")
            kept += 1

    print(f"{'='*50}")
    print(f"total={total}  kept={kept}  dropped={dropped} ({dropped/max(1,total)*100:.1f}%)")
    print(f"  cleaned → {out}")
    print(f"  dropped → {drop_log}")


if __name__ == "__main__":
    main()
