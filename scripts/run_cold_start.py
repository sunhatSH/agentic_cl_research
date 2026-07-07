#!/usr/bin/env python3
"""Cold-start pipeline: collect multi-turn trajectories from sandbox hermes.

Default: use existing datasets/queries.jsonl (with bucket + persona) and
run incremental multi-turn collection (actor + observer + questioner, no
reward/winner). Queries generation is a one-time step via --generate.

Usage:
    # One-time: generate queries (classify bucket + assign persona)
    source scripts/load_tencent_env.sh
    .venv/bin/python scripts/run_cold_start.py --generate --no-collect --classify-workers 32

    # Collect (default: incremental, multi-turn, 32 concurrent)
    .venv/bin/python scripts/run_cold_start.py --num-queries 2849 --max-concurrent 32

    # Full pipeline in one shot
    bash scripts/run_cold_pipeline.sh
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

_PERSONA_CACHE: list[dict] | None = None


def _load_persona_catalog() -> list[dict]:
    """Load persona summary (name, profession, focus, tone) from agents/personas.json."""
    global _PERSONA_CACHE
    if _PERSONA_CACHE is not None:
        return _PERSONA_CACHE
    catalog: list[dict] = []
    for p in json.loads((_REPO / "agents" / "personas.json").read_text(encoding="utf-8"))["personas"]:
        catalog.append({
            "name": p["name"],
            "profession": p["profession"],
            "focus": p.get("observation_focus", ""),
            "tone": p.get("tone", "neutral"),
        })
    _PERSONA_CACHE = catalog
    return catalog


def _pick_persona(query: str, catalog: list[dict], client: Any) -> str:
    """LLM picks the best-matching persona; returns 'random' as fallback."""
    options_text = "\n".join(
        f'- {p["name"]} ({p["profession"]}, focus={p["focus"]}, tone={p["tone"]})'
        for p in catalog
    )
    msgs = [
        {"role": "system", "content": (
            "你是一个任务-人设匹配器。给定一条给 AI agent 的用户任务，从候选人设列表里"
            "选最匹配的那个。匹配原则：人设的职业和关注点应该最能'审阅'这个任务的产出"
            "（比如财务任务选会计/银行家，安全任务选律师/合规，工程任务选运维/架构师，"
            "写作任务选编辑/撰稿人）。只输出人设的 name 字段，不要其他文字。"
        )},
        {"role": "user", "content": f"任务：{query}\n\n候选人设：\n{options_text}\n\n输出人设 name："},
    ]
    try:
        raw = client.chat(msgs, max_tokens=60)
        name = raw.strip().strip('"').strip("'")
        valid = {p["name"] for p in catalog}
        if name in valid:
            return name
        for vn in valid:
            if vn in raw:
                return vn
        return "random"
    except Exception:  # noqa: BLE001
        return "random"


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

        # 1b2 — persona assignment (parallel, same workers)
        catalog = _load_persona_catalog()
        persona_fixed = 0
        persona_random = 0

        def _assign_persona(rec: dict) -> dict:
            nonlocal persona_fixed, persona_random
            name = _pick_persona(rec["seed_query"], catalog, client)
            rec["persona_name"] = name
            if name != "random":
                persona_fixed += 1
            else:
                persona_random += 1
            return rec

        with ThreadPoolExecutor(max_workers=classify_workers) as ex:
            futures = {ex.submit(_assign_persona, rec): rec for rec in records}
            with tqdm(total=len(records), desc="Picking personas", unit="q", smoothing=0.01) as pbar:
                for fut in as_completed(futures):
                    fut.result()
                    pbar.update(1)
        print(f"  persona: assigned={persona_fixed}  fallback_random={persona_random}")

    # 1c — write queries.jsonl
    _QUERIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_QUERIES_PATH, "w", encoding="utf-8") as fh:
        for rec in records:
            row = {
                "record_id": rec["record_id"],
                "queries": [rec["seed_query"]] + rec.get("follow_ups", []),
                "bucket": rec.get("bucket", ""),
                "sub_bucket": rec.get("sub_bucket"),
                "persona_name": rec.get("persona_name", "random"),
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


def _load_existing(out_file: Path) -> dict[int, dict]:
    """Load existing trajectories keyed by query_index (empty if no file)."""
    rows: dict[int, dict] = {}
    if not out_file.exists():
        return rows
    with open(out_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                continue
            qi = t.get("query_index")
            if qi is not None:
                rows[qi] = t
    return rows


def stage_collect(*, num_queries: int, max_concurrent: int, actor_model: str,
                   max_turns: int, hermes_max_turns: int, slot_timeout: int,
                   mode: str = "overwrite", multi_turn: bool = True,
                   out_dir: Path | None = None) -> int:
    """Run parallel sandbox collection — 1 sandbox per query.

    Multi-turn (default): actor(hermes) + observer + questioner drive up to
    K=randint(1,max_turns) turns per query. NO reward / NO winner (that's the
    training stage). Set multi_turn=False for single-turn smoke.

    Modes (which query_index to (re)run; success rows are NEVER re-run except
    in overwrite):
      - overwrite   : run ALL queries, replace the whole file.
      - incremental : run queries that are MISSING or FAILED; keep existing
                      successes untouched. Safe to resume an interrupted run.
      - retry       : run ONLY existing FAILED rows; keep everything else.

    In every mode the file is written by merging on query_index: a successful
    new result replaces whatever was there; existing successes are preserved.
    """
    from scripts.sandbox_grpo_collect import _load_queries, _run_one_collect_query

    # Observer + Questioner (session agents). Created once, shared across threads
    # (each call is stateless per session; LLM clients are thread-safe HTTP).
    observer = questioner = None
    if multi_turn:
        _ensure_sufy_key()          # observer/questioner resolve SUFY_API_KEY from agents.yaml
        from agents.observer import Observer
        from agents.questioner import Questioner
        observer = Observer()       # use_llm defaults True; falls back to deterministic on error
        questioner = Questioner()
        print("  multi-turn: observer + questioner enabled (no reward/winner)")
    else:
        print("  single-turn: seed query only (smoke)")

    tasks = _load_queries(str(_QUERIES_PATH), num_queries)
    total = len(tasks)

    out_root = out_dir or _OUT_DIR
    out_root.mkdir(parents=True, exist_ok=True)
    out_file = out_root / "grpo_hermes.jsonl"

    # Existing state (for incremental / retry merge).
    existing = {} if mode == "overwrite" else _load_existing(out_file)

    # Decide which query indices to run this pass.
    if mode == "overwrite":
        to_run = list(range(total))
    elif mode == "incremental":
        to_run = [i for i in range(total)
                  if i not in existing or existing[i].get("error")]
    elif mode == "retry":
        to_run = [i for i in range(total)
                  if i in existing and existing[i].get("error")]
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    print(f"\n  mode={mode}  total={total}  existing={len(existing)}  to_run={len(to_run)}")
    print(f"  {max_concurrent} concurrent sandboxes")

    if not to_run:
        print("  nothing to run.")
        return len(existing)

    # merged holds the final state; start from existing (successes preserved).
    merged: dict[int, dict] = dict(existing)
    ok = err = 0
    t0 = time.time()

    def _flush() -> None:
        """Rewrite the whole file from merged (atomic-ish: temp + rename)."""
        tmp = out_file.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for qi in sorted(merged):
                f.write(json.dumps(merged[qi], ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(out_file)

    with ThreadPoolExecutor(max_workers=max_concurrent) as ex:
        futures = {
            ex.submit(
                _run_one_collect_query,
                tasks[i], i,
                actor="hermes", actor_model=actor_model, actor_base="",
                max_turns=max_turns, hermes_max_turns=hermes_max_turns,
                slot_timeout=slot_timeout, backend="e2b", template="agentic-cl-sandbox",
                observer=observer, questioner=questioner, rng_seed=i,
            ): i
            for i in to_run
        }
        with tqdm(total=len(to_run), desc=f"Collecting [{mode}]", unit="traj", smoothing=0.01) as pbar:
            done = 0
            for fut in as_completed(futures):
                traj = fut.result()
                row = json.loads(traj.to_jsonl())
                if traj and traj.error:
                    err += 1
                    # In incremental/retry: only overwrite if there was no prior
                    # success (a failed rerun must not clobber an old success —
                    # but to_run already excludes successes, so this is safe).
                    merged[traj.query_index] = row
                else:
                    ok += 1
                    merged[traj.query_index] = row
                done += 1
                # Flush cadence: every completion for small runs (smoke / retry),
                # every 25 for large runs (full rewrite is O(n), keep it bounded).
                flush_every = 1 if len(to_run) <= 20 else 10
                if done % flush_every == 0:
                    _flush()
                pbar.set_postfix(ok=ok, err=err, refresh=False)
                pbar.update(1)

    _flush()
    elapsed = time.time() - t0
    total_ok = sum(1 for t in merged.values() if not t.get("error"))
    total_err = sum(1 for t in merged.values() if t.get("error"))
    print(f"  done in {elapsed:.0f}s  this_pass(ok={ok} err={err})  "
          f"file_total(ok={total_ok} err={total_err})  ({len(to_run)/max(1,elapsed):.2f} traj/s)")

    manifest = {
        "actor": "hermes", "mode": mode, "max_concurrent": max_concurrent,
        "num_queries": total, "ran_this_pass": len(to_run),
        "file_ok": total_ok, "file_err": total_err,
        "elapsed_s": elapsed, "out_file": str(out_file),
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  → {out_file}")

    return len(merged)


# ── CLI ───────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # ── Stage 1: queries generation (only when --generate) ──
    ap.add_argument("--generate", action="store_true",
                    help="(re)generate queries.jsonl from taskspecs (classify + persona)")
    ap.add_argument("--classify-workers", type=int, default=32)
    # ── Stage 2: collect ──
    ap.add_argument("--no-collect", action="store_true", help="skip collection")
    ap.add_argument("--num-queries", type=int, default=100)
    ap.add_argument("--max-concurrent", type=int, default=32)
    ap.add_argument("--actor-model", default="openai/gpt-5")
    ap.add_argument("--max-turns", type=int, default=3, help="session turn cap")
    ap.add_argument("--hermes-max-turns", type=int, default=30, help="hermes ReAct limit")
    ap.add_argument("--slot-timeout", type=int, default=900, help="per-sandbox timeout (s)")
    ap.add_argument("--collect-mode", choices=["overwrite", "incremental", "retry"],
                    default="incremental")
    ap.add_argument("--single-turn", action="store_true",
                    help="seed query only, no observer/questioner")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default rollouts/cold_start)")
    args = ap.parse_args()

    t0 = time.time()

    if args.generate:
        stage_queries(classify=True, classify_workers=args.classify_workers)
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
            mode=args.collect_mode,
            multi_turn=not args.single_turn,
            out_dir=Path(args.out_dir) if args.out_dir else None,
        )

    print(f"\n{'='*60}")
    print(f"ALL DONE in {time.time()-t0:.0f}s")
    print(f"  queries      → {_QUERIES_PATH}")
    print(f"  trajectories → {_OUT_DIR}/grpo_hermes.jsonl")


if __name__ == "__main__":
    main()
