#!/usr/bin/env python3
"""Cold-start trajectory collection orchestrator.

Decoupled from the low-level collection engine (sandbox_grpo_collect.py):
this script ONLY handles data preparation + launch.  The engine handles
sandbox lifecycle, hermes execution, and trajectory serialisation.

Pipeline:
  1. rsync taskspecs from --source → --target (default: data/taskspecs/)
  2. Generate queries.jsonl via taskspec_to_queries.py
  3. Run parallel sandbox collection via sandbox_grpo_collect.py

Usage:
    # Full cold-start (copy + generate + collect):
    source scripts/load_tencent_env.sh
    python scripts/cold_start_collect.py \\
        --source /mnt/afs_toolcall/wujian1/Projects/seed2traj/taskspecs_w3_full \\
        --max-concurrent 128 --num-queries 100 --limit 200

    # Skip copy (data already local):
    python scripts/cold_start_collect.py --no-copy --max-concurrent 128

    # Dry-run (copy + gen queries only, no collection):
    python scripts/cold_start_collect.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

DEFAULT_SOURCE = "/mnt/afs_toolcall/wujian1/Projects/seed2traj/taskspecs_w3_full"
DEFAULT_TARGET = str(_REPO / "data" / "taskspecs")
DEFAULT_QUERIES = str(_REPO / "datasets" / "queries.jsonl")


def _run(cmd: list[str], **kw) -> int:
    """Run a subprocess, streaming output.  Returns exit code."""
    print(f"[cold] + {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=False, **kw).returncode


def step_copy(source: str, target: str) -> bool:
    """Copy taskspecs from source → target, preserving full directory structure.

    Uses Python's ``shutil.copytree`` with dirs_exist_ok=True so the target
    is an exact mirror (stale taskspecs removed first, then source merged in).
    Returns True on success.
    """
    import os

    src = Path(source)
    tgt = Path(target)
    if not src.is_dir():
        print(f"[cold] ERROR: source not found: {src}", file=sys.stderr)
        return False

    t0 = time.time()

    # Remove stale target so copytree produces a clean mirror.
    if tgt.exists():
        shutil.rmtree(tgt)

    def _ignore(dirpath: str, names: list[str]) -> set[str]:
        return {n for n in names if n in (".DS_Store", "Thumbs.db") or n.startswith("~$")}

    shutil.copytree(src, tgt, symlinks=False, ignore=_ignore, dirs_exist_ok=True)

    n_dirs = sum(1 for _ in tgt.rglob("taskspec.yaml"))
    size = sum(f.stat().st_size for f in tgt.rglob("*") if f.is_file())
    print(f"[cold] copied {n_dirs} taskspecs ({size/1024/1024:.0f} MB) in {time.time()-t0:.0f}s → {tgt}")
    return True


def step_generate_queries(taskspecs_dir: str, queries_path: str, limit: int | None = None, *, classify: bool = False) -> bool:
    """Run taskspec_to_queries.py, optionally with --classify for LLM bucket labeling."""
    cmd = [sys.executable, str(_REPO / "scripts" / "taskspec_to_queries.py"),
           "--taskspecs", taskspecs_dir, "--output", queries_path]
    if limit:
        cmd += ["--limit", str(limit)]
    if classify:
        cmd.append("--classify")
    return _run(cmd) == 0


def step_collect(
    queries_path: str,
    *,
    num_queries: int,
    max_concurrent: int,
    actor_model: str = "openai/gpt-5",
    max_turns: int = 8,
    hermes_max_turns: int = 50,
    slot_timeout: int = 600,
    out_dir: str = "rollouts/cold_start",
    backend: str = "e2b",
    template: str = "agentic-cl-sandbox",
) -> bool:
    """Run sandbox_grpo_collect.py in parallel collect mode."""
    cmd = [
        sys.executable,
        str(_REPO / "scripts" / "sandbox_grpo_collect.py"),
        "--actor", "hermes",
        "--mode", "collect",
        "--queries", queries_path,
        "--slots", "1",
        "--num-queries", str(num_queries),
        "--backend", backend,
        "--template", template,
        "--actor-model", actor_model,
        "--max-turns", str(max_turns),
        "--hermes-max-turns", str(hermes_max_turns),
        "--slot-timeout", str(slot_timeout),
        "--max-concurrent", str(max_concurrent),
        "--out-dir", out_dir,
    ]
    return _run(cmd) == 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Data
    ap.add_argument("--source", default=DEFAULT_SOURCE,
                    help="source taskspecs directory (wujian1's)")
    ap.add_argument("--target", default=DEFAULT_TARGET,
                    help="local taskspecs directory")
    ap.add_argument("--queries", default=DEFAULT_QUERIES,
                    help="output queries JSONL path")
    ap.add_argument("--limit", type=int, default=None,
                    help="limit queries generation to first N taskspecs")
    # Modes
    ap.add_argument("--no-copy", action="store_true",
                    help="skip rsync (data already local)")
    ap.add_argument("--no-generate", action="store_true",
                    help="skip queries generation (queries.jsonl already exists)")
    ap.add_argument("--classify", action="store_true",
                    help="LLM bucket labeling at queries generation")
    ap.add_argument("--dry-run", action="store_true",
                    help="copy + generate queries only, skip collection")
    # Collection
    ap.add_argument("--num-queries", type=int, default=100,
                    help="number of queries to collect")
    ap.add_argument("--max-concurrent", type=int, default=128,
                    help="parallel sandboxes")
    ap.add_argument("--actor-model", default="openai/gpt-5",
                    help="sufy model for hermes")
    ap.add_argument("--max-turns", type=int, default=8,
                    help="session turn cap (questioner rounds)")
    ap.add_argument("--hermes-max-turns", type=int, default=50,
                    help="hermes internal ReAct loop cap")
    ap.add_argument("--slot-timeout", type=int, default=600,
                    help="per-sandbox timeout (s)")
    ap.add_argument("--out-dir", default="rollouts/cold_start")
    ap.add_argument("--backend", default="e2b", choices=["e2b", "local"])
    ap.add_argument("--template", default="agentic-cl-sandbox")
    args = ap.parse_args()

    t0 = time.time()

    # Step 1: copy taskspecs
    if not args.no_copy:
        if not step_copy(args.source, args.target):
            sys.exit(1)
    else:
        print("[cold] skipping copy (--no-copy)", flush=True)

    # Step 2: generate queries
    if not args.no_generate:
        if not step_generate_queries(args.target, args.queries, args.limit, classify=args.classify):
            sys.exit(1)
    else:
        print("[cold] skipping queries gen (--no-generate)", flush=True)

    # Step 3: collect
    if args.dry_run:
        print("[cold] DRY RUN — stopping before collection.", flush=True)
        print(f"[cold] queries → {args.queries}", flush=True)
        print(f"[cold] next: source scripts/load_tencent_env.sh && python scripts/cold_start_collect.py --no-copy --no-generate", flush=True)
    else:
        if not step_collect(
            args.queries,
            num_queries=args.num_queries,
            max_concurrent=args.max_concurrent,
            actor_model=args.actor_model,
            max_turns=args.max_turns,
            slot_timeout=args.slot_timeout,
            out_dir=args.out_dir,
            backend=args.backend,
            template=args.template,
        ):
            sys.exit(1)

    print(f"\n[cold] ALL DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
