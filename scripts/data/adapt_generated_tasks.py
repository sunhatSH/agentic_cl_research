#!/usr/bin/env python3
"""Adapter: generated_tasks (task.json + inputs/) -> taskspec format (taskspec.yaml + files/).

The subagent task set (tongronglei's generated_tasks, copied to
datasources/generated_tasks_hermes/) uses a DIFFERENT layout than our collection pipeline
(run_cold_start / sandbox_grpo_collect) consumes:

  source  : <D>/<task_id>/{task.json, answer_key.json, inputs/, usage.json}
  target  : <record_id>/{taskspec.yaml, files/, answer_key.json}

This converts them so they can go through OUR own collection (hermes actor +
observer + questioner) — we only borrow the TASKS, not their generation.

Mapping (fields decided in session):
  task.json.user_prompt   -> taskspec.seed_query   (verbatim; NOT LLM-rewritten)
  task.json.task_id       -> taskspec.task_id / record_id
  task.json.domain        -> taskspec.task_family  (D1..D13 provenance; bucket is
                                                    re-derived by the pipeline's LLM)
  task.json.follow_ups    -> taskspec.user_profile.follow_ups  (pipeline supports)
  inputs/*                -> files/*                (workspace uploaded to sandbox)
  answer_key.json         -> answer_key.json        (kept beside taskspec for QC;
                                                     NOT in files/, not uploaded)

Usage:
  # small subset (default: 20 per domain) to validate the adapter:
  python scripts/data/adapt_generated_tasks.py
  # custom:
  python scripts/data/adapt_generated_tasks.py --per-domain 20 \
      --src datasources/generated_tasks_hermes --out datasources/taskspecs_hermes
  python scripts/data/adapt_generated_tasks.py --all      # every task (~48k)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

_JUNK = {".DS_Store", "Thumbs.db"}


def _clean(text: str) -> str:
    return str(text or "").strip()


def adapt_one(task_dir: Path, out_root: Path) -> str | None:
    """Convert one <task_id>/ dir. Returns record_id on success, None on skip."""
    tj = task_dir / "task.json"
    if not tj.is_file():
        return None
    try:
        task = json.loads(tj.read_text(encoding="utf-8"))
    except Exception:
        return None

    seed = _clean(task.get("user_prompt"))
    if not seed:
        return None  # no usable query -> skip (pipeline requires non-empty seed_query)

    record_id = _clean(task.get("task_id")) or task_dir.name
    follow_ups = [
        _clean(q) for q in (task.get("follow_ups") or []) if _clean(q)
    ]
    up = task.get("user_profile") or {}

    spec = {
        "task_id": record_id,
        "seed_query": seed,
        "task_family": _clean(task.get("domain")) or "unknown",
        "source": "generated_tasks_hermes",  # provenance
        "user_profile": {
            "name": _clean(up.get("name")) if isinstance(up, dict) else "",
            "role": _clean(up.get("role")) if isinstance(up, dict) else "",
            "follow_ups": follow_ups,
        },
        "deliverables": list(task.get("deliverables") or []),
    }

    dst = out_root / record_id
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "taskspec.yaml").write_text(
        yaml.safe_dump(spec, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    # inputs/ -> files/  (workspace uploaded to the sandbox, structure preserved)
    src_inputs = task_dir / "inputs"
    n_files = 0
    if src_inputs.is_dir():
        files_dst = dst / "files"
        for fp in src_inputs.rglob("*"):
            if not fp.is_file() or fp.name in _JUNK or fp.name.startswith("~$"):
                continue
            rel = fp.relative_to(src_inputs)
            tgt = files_dst / rel
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fp, tgt)
            n_files += 1

    # answer_key.json -> beside taskspec (QC material, NOT uploaded to sandbox)
    ak = task_dir / "answer_key.json"
    if ak.is_file():
        shutil.copy2(ak, dst / "answer_key.json")

    return record_id


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="datasources/generated_tasks_hermes")
    ap.add_argument("--out", default="datasources/taskspecs_hermes")
    ap.add_argument("--per-domain", type=int, default=20,
                    help="tasks per D* domain (subset validation). Ignored with --all.")
    ap.add_argument("--all", action="store_true", help="convert every task (~48k)")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    domains = sorted(d for d in src.iterdir() if d.is_dir() and d.name.startswith("D"))
    total = 0
    per_dom: dict[str, int] = {}
    for dom in domains:
        tasks = sorted(t for t in dom.iterdir() if t.is_dir())
        if not args.all:
            tasks = tasks[: args.per_domain]
        cnt = 0
        for t in tasks:
            if adapt_one(t, out):
                cnt += 1
        per_dom[dom.name] = cnt
        total += cnt
        print(f"  {dom.name}: {cnt} tasks", flush=True)
    print(f"\n[adapt] {total} taskspecs -> {out}  (per_domain={'ALL' if args.all else args.per_domain})")
    print(f"[adapt] by domain: {json.dumps(per_dom)}")


if __name__ == "__main__":
    main()
