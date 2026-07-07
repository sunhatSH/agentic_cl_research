#!/usr/bin/env python3
"""Retry failed cold-start trajectories in place.

Reads rollouts/cold_start/grpo_hermes.jsonl, finds rows with a non-empty
``error`` field, reruns ONLY those queries (with a larger timeout), and merges
the successful reruns back — replacing the failed rows by query_index. Rows that
still fail keep their latest error. Successful rows are never touched.

Usage:
    source scripts/load_tencent_env.sh
    .venv/bin/python scripts/retry_failed.py \
        --max-concurrent 64 --slot-timeout 1200 --hermes-max-turns 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

_QUERIES = _REPO / "datasets" / "queries.jsonl"
_TRAJ = _REPO / "rollouts" / "cold_start" / "grpo_hermes.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-concurrent", type=int, default=64)
    ap.add_argument("--slot-timeout", type=int, default=1200, help="per-sandbox timeout (s), larger than first pass")
    ap.add_argument("--hermes-max-turns", type=int, default=50)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--actor-model", default="openai/gpt-5")
    args = ap.parse_args()

    from scripts.sandbox_grpo_collect import _run_one_collect_query

    # Ensure SUFY_API_KEY (for any downstream use); harmless if already set.
    if not os.environ.get("E2B_API_KEY"):
        print("[retry] WARNING: E2B_API_KEY not set — source scripts/load_tencent_env.sh", file=sys.stderr)

    # 1. Load current trajectories, keyed by query_index.
    rows: dict[int, dict] = {}
    with open(_TRAJ, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            rows[t["query_index"]] = t

    failed_qis = [qi for qi, t in rows.items() if t.get("error")]
    print(f"[retry] total={len(rows)}  failed={len(failed_qis)}")
    if not failed_qis:
        print("[retry] nothing to retry.")
        return

    # 2. Load queries (query_index = line number in queries.jsonl).
    queries: list[dict] = []
    with open(_QUERIES, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qs = obj.get("queries") or []
            queries.append({
                "query": qs[0] if qs else "",
                "record_id": obj.get("record_id", ""),
                "bucket": obj.get("bucket", ""),
            })

    # 3. Rerun failed tasks.
    t0 = time.time()
    fixed = 0
    still = 0

    def _retry(qi: int) -> tuple[int, object]:
        task = queries[qi]
        traj = _run_one_collect_query(
            task, qi,
            actor="hermes", actor_model=args.actor_model, actor_base="",
            max_turns=args.max_turns, hermes_max_turns=args.hermes_max_turns,
            slot_timeout=args.slot_timeout, backend="e2b", template="agentic-cl-sandbox",
        )
        return qi, traj

    with ThreadPoolExecutor(max_workers=args.max_concurrent) as ex:
        futures = {ex.submit(_retry, qi): qi for qi in failed_qis}
        with tqdm(total=len(failed_qis), desc="Retrying failed", unit="traj", smoothing=0.01) as pbar:
            for fut in as_completed(futures):
                qi, traj = fut.result()
                if traj and not traj.error:
                    rows[qi] = json.loads(traj.to_jsonl())  # replace with success
                    fixed += 1
                else:
                    rows[qi] = json.loads(traj.to_jsonl())  # keep latest error
                    still += 1
                pbar.set_postfix(fixed=fixed, still=still, refresh=False)
                pbar.update(1)

    # 4. Rewrite the full file (sorted by query_index for stable output).
    with open(_TRAJ, "w", encoding="utf-8") as f:
        for qi in sorted(rows):
            f.write(json.dumps(rows[qi], ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

    print(f"\n[retry] done in {time.time()-t0:.0f}s  fixed={fixed}  still_failed={still}")
    print(f"[retry] → {_TRAJ}")


if __name__ == "__main__":
    main()
