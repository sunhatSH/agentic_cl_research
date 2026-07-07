#!/usr/bin/env python3
"""Cold-start pipeline: classify → collect, with real-time progress.

Assumes taskspecs are already at data/taskspecs/.  Runs two stages in sequence,
each with a live tqdm progress bar.

Usage:
    source scripts/load_tencent_env.sh
    /tmp/sandbox_venv/bin/python scripts/run_cold_start.py \
        --num-queries 3819 --max-concurrent 128
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

_TASKSPECS_DIR = _REPO / "data" / "taskspecs"
_QUERIES_PATH = _REPO / "datasets" / "queries.jsonl"
_OUT_DIR = _REPO / "rollouts" / "cold_start"

# ── task_family → bucket static mapping (6 types, zero LLM cost) ────────

_STATIC_MAP: dict[str, str] = {
    "file_organize": "ops",
    "file_move_rename": "ops",
    "data_merge": "ops",
    "table_process": "office",
    "risky_op": "safety",
    "process_design": "workflow",
}


# ── Stage 1: queries generation + LLM classify ──────────────────────────


def _ensure_sufy_key() -> None:
    """Ensure SUFY_API_KEY is set for classify.py (reads AGENT_MODEL_KEY or runtime.env)."""
    import os

    if os.environ.get("SUFY_API_KEY", "").strip():
        return
    key = os.environ.get("AGENT_MODEL_KEY", "").strip()
    if key:
        os.environ["SUFY_API_KEY"] = key
        return
    env_file = _REPO / "docker" / "sandbox" / "runtime.env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == "AGENT_MODEL_KEY" and v.strip().strip('"').strip("'"):
                os.environ["SUFY_API_KEY"] = v.strip().strip('"').strip("'")
                return


def _load_one_taskspec(subdir: Path) -> dict[str, Any] | None:
    import yaml

    ts_path = subdir / "taskspec.yaml"
    if not ts_path.is_file():
        return None
    try:
        ts = yaml.safe_load(ts_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(ts, dict):
        return None
    seed = ts.get("seed_query", "")
    if not isinstance(seed, str) or not seed.strip():
        return None
    follow_ups = []
    profile = ts.get("user_profile") or {}
    fu = profile.get("follow_ups") if isinstance(profile, dict) else None
    if isinstance(fu, list):
        follow_ups = [q.strip() for q in fu if isinstance(q, str) and q.strip()]
    return {
        "record_id": ts.get("task_id") or subdir.name,
        "seed_query": seed.strip(),
        "follow_ups": follow_ups,
        "task_family": ts.get("task_family", "unknown"),
    }


# ── Stage 1: queries generation + LLM classify ──────────────────────────


def _classify_one(rec: dict, client: Any) -> dict:
    """Classify one seed_query: static mapping (free) > LLM."""
    tf = rec["task_family"]
    static = _STATIC_MAP.get(tf, "")
    if static:
        rec["bucket"] = static
        rec["classify_source"] = "static"
        return rec
    from data_pipeline.classify import classify_query

    verdict = classify_query(rec["seed_query"], client)
    rec["bucket"] = verdict.get("bucket", "unknown")
    rec["classify_source"] = "llm"
    return rec


def stage_queries(*, classify: bool, classify_workers: int = 16) -> int:
    """Generate queries.jsonl.  Returns number of rows written."""

    # 1a — load taskspecs
    subdirs = sorted(d for d in _TASKSPECS_DIR.iterdir() if d.is_dir())
    records = []
    for d in tqdm(subdirs, desc="Loading taskspecs", unit="file"):
        rec = _load_one_taskspec(d)
        if rec:
            records.append(rec)
    print(f"  loaded {len(records)} taskspecs with seed_query")

    # 1b — classify
    if classify:
        _ensure_sufy_key()
        from data_pipeline.classify import make_default_client

        client = make_default_client()
        print(f"  LLM classifier: {client.model} ({classify_workers} workers)")

        static_recs = []
        llm_recs = []
        for rec in records:
            if _STATIC_MAP.get(rec["task_family"], ""):
                _classify_one(rec, client)
                static_recs.append(rec)
            else:
                llm_recs.append(rec)

        classified = list(static_recs)
        static_count = len(static_recs)
        llm_ok = llm_unknown = 0

        with ThreadPoolExecutor(max_workers=classify_workers) as ex:
            futures = {ex.submit(_classify_one, rec, client): rec for rec in llm_recs}
            with tqdm(total=len(llm_recs), desc="LLM classifying", unit="q", smoothing=0.01) as pbar:
                for fut in as_completed(futures):
                    rec = fut.result()
                    classified.append(rec)
                    if rec.get("bucket") != "unknown":
                        llm_ok += 1
                    else:
                        llm_unknown += 1
                    pbar.set_postfix(ok=llm_ok, unk=llm_unknown, refresh=False)
                    pbar.update(1)

        records = classified
        print(f"  classify done: static={static_count}  LLM_ok={llm_ok}  LLM_unknown={llm_unknown}")

    # 1c — write queries.jsonl
    _QUERIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_QUERIES_PATH, "w", encoding="utf-8") as fh:
        for rec in records:
            row = {
                "record_id": rec["record_id"],
                "queries": [rec["seed_query"]] + rec.get("follow_ups", []),
                "bucket": rec.get("bucket", ""),
                "sub_bucket": rec.get("sub_bucket"),
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    c = Counter(r.get("bucket", "unknown") for r in records)
    print("  bucket distribution:")
    for b, n in c.most_common():
        bar = "█" * (n * 50 // max(c.values()))
        print(f"    {b:15s} {n:5d}  {bar}")

    print(f"  → {_QUERIES_PATH} ({len(records)} rows)")
    return len(records)


# ── Stage 2: sandbox collection ──────────────────────────────────────────


def stage_collect(*, num_queries: int, max_concurrent: int, actor_model: str,
                   max_turns: int, hermes_max_turns: int, slot_timeout: int) -> int:
    """Run parallel sandbox collection — 1 slot per query, collect mode."""

    from scripts.sandbox_grpo_collect import _load_queries, _run_one_collect_query

    tasks = _load_queries(str(_QUERIES_PATH), num_queries)
    total = len(tasks)
    print(f"\n  collection: {total} queries, {max_concurrent} concurrent sandboxes")

    ok = err = 0
    t0 = time.time()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = _OUT_DIR / "grpo_hermes.jsonl"
    fh = open(out_file, "w", encoding="utf-8")  # incremental write, flush every N rows

    with ThreadPoolExecutor(max_workers=max_concurrent) as ex:
        futures = {
            ex.submit(
                _run_one_collect_query,
                t, i,
                actor="hermes", actor_model=actor_model, actor_base="",
                max_turns=max_turns, hermes_max_turns=hermes_max_turns,
                slot_timeout=slot_timeout, backend="e2b", template="agentic-cl-sandbox",
            ): i
            for i, t in enumerate(tasks)
        }
        with tqdm(total=total, desc="Sandbox collecting", unit="traj", smoothing=0.01) as pbar:
            for fut in as_completed(futures):
                traj = fut.result()
                if traj and traj.error:
                    err += 1
                else:
                    ok += 1
                fh.write(traj.to_jsonl() + "\n")
                fh.flush()          # flush every trajectory — never lose data on crash
                os.fsync(fh.fileno())
                pbar.set_postfix(ok=ok, err=err, refresh=False)
                pbar.update(1)

    fh.close()
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.0f}s  ok={ok}  err={err}  ({total/max(1,elapsed):.1f} traj/s)")

    manifest = {
        "actor": "hermes", "mode": "collect", "max_concurrent": max_concurrent,
        "num_queries": total, "ok": ok, "errors": err,
        "elapsed_s": elapsed, "out_file": str(out_file),
    }
    (_OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  → {out_file}")

    return ok + err


# ── CLI ───────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-classify", action="store_true")
    ap.add_argument("--classify-workers", type=int, default=16)
    ap.add_argument("--no-collect", action="store_true")
    ap.add_argument("--num-queries", type=int, default=100)
    ap.add_argument("--max-concurrent", type=int, default=128)
    ap.add_argument("--actor-model", default="openai/gpt-5")
    ap.add_argument("--max-turns", type=int, default=8, help="session turn cap")
    ap.add_argument("--hermes-max-turns", type=int, default=50, help="hermes ReAct limit")
    ap.add_argument("--slot-timeout", type=int, default=600)
    ap.add_argument("--no-generate", action="store_true",
                    help="skip queries.jsonl regeneration (use existing file as-is)")
    args = ap.parse_args()

    t0 = time.time()

    # Only (re)generate queries.jsonl when explicitly asked. Regenerating with
    # --no-classify would OVERWRITE an already-classified file with empty buckets.
    if not args.no_generate:
        stage_queries(classify=not args.no_classify, classify_workers=args.classify_workers)
    else:
        n = sum(1 for _ in open(_QUERIES_PATH)) if _QUERIES_PATH.exists() else 0
        print(f"[skip] using existing queries.jsonl ({n} rows)")

    if not args.no_collect:
        stage_collect(
            num_queries=args.num_queries,
            max_concurrent=args.max_concurrent,
            actor_model=args.actor_model,
            max_turns=args.max_turns,
            hermes_max_turns=args.hermes_max_turns,
            slot_timeout=args.slot_timeout,
        )

    print(f"\n{'='*60}")
    print(f"ALL DONE in {time.time()-t0:.0f}s")
    print(f"  queries      → {_QUERIES_PATH}")
    print(f"  trajectories → {_OUT_DIR}/grpo_hermes.jsonl")


if __name__ == "__main__":
    main()
