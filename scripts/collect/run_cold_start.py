#!/usr/bin/env python3
"""Cold-start pipeline: collect multi-turn trajectories from sandbox hermes.

Default: use existing datasets/queries.jsonl (with bucket + persona) and
run incremental multi-turn collection (actor + observer + questioner, no
reward/winner). Queries generation is a one-time step via --generate.

Usage:
    # One-time: generate queries (classify bucket + assign persona)
    source scripts/env/load_tencent_env.sh
    .venv/bin/python scripts/collect/run_cold_start.py --generate --no-collect --classify-workers 32

    # Collect (default: incremental, multi-turn, 32 concurrent)
    .venv/bin/python scripts/collect/run_cold_start.py --num-queries 2849 --max-concurrent 32

    # Full pipeline in one shot
    bash scripts/pipeline/run_cold_pipeline.sh
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

_TASKSPECS_DIR = _REPO / "datasources" / "taskspecs"
_QUERIES_PATH = _REPO / "datasets" / "queries.jsonl"
# Rollout data lives OUTSIDE the repo (gitignored by location), organized as:
#   <ROOT>/{real,smoke}/trajectory/<model>/grpo_hermes.jsonl + manifest.json
#   <ROOT>/{real,smoke}/debug/observer_report/<model>/observer_reports.jsonl
_ROLLOUTS_ROOT = Path("/mnt/afs_toolcall/sunhao4/agentic_cl_rollouts")
_OUT_DIR = _ROLLOUTS_ROOT / "real"   # default collection target (real/, not smoke/)


def _model_tag(actor_model: str) -> str:
    """Short folder tag inferred from the actor model id (gpt5 / qwen27b / ...)."""
    m = actor_model.lower()
    if "gpt-5" in m or "gpt5" in m:
        return "gpt5"
    if "qwen3.6-27b" in m or "qwen27b" in m:
        return "qwen27b"
    if "qwen" in m:
        return "qwen"
    # fallback: last path segment, sanitized
    tail = actor_model.rsplit("/", 1)[-1]
    return "".join(c if c.isalnum() else "-" for c in tail).strip("-") or "model"


# ── task_family → bucket static mapping (6 types, zero LLM cost) ────────

# ── Stage 1: queries generation + LLM classify ──────────────────────────


def _ensure_tokenhub_key() -> None:
    """Ensure TOKENHUB_API_KEY is set for classify.py (reads AGENT_MODEL_KEY or runtime.env)."""
    import os

    if os.environ.get("TOKENHUB_API_KEY", "").strip():
        return
    key = os.environ.get("AGENT_MODEL_KEY", "").strip()
    if key:
        os.environ["TOKENHUB_API_KEY"] = key
        return
    env_file = _REPO / "docker" / "sandbox" / "runtime.env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == "AGENT_MODEL_KEY" and v.strip().strip('"').strip("'"):
                os.environ["TOKENHUB_API_KEY"] = v.strip().strip('"').strip("'")
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
    # Front cleaning: strip ZW + drop if any garbled char remains (threshold=0)
    from datasources.cleaning import analyze_text, strip_zw
    seed = strip_zw(seed)
    a = analyze_text(seed)
    if a.garble_chars > 0:
        return None   # dirty seed → discard entirely
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


def _classify_one(rec: dict, client: Any) -> dict:
    """Classify one seed_query: bucket + persona in a single LLM call."""

    from data_pipeline.classify import build_classify_prompt, parse_classify_output

    # Build prompt with bucket options + persona catalog + output format
    persona_options = "\n".join(
        f'- {p["name"]} ({p["profession"]}, focus={p["focus"]}, tone={p["tone"]})'
        for p in _load_persona_catalog()
    )
    bucket_msgs = build_classify_prompt(rec["seed_query"])
    combined_user = (
        bucket_msgs[1]["content"]
        + "\n\n此外，从以下 42 个人设中选出最适合审阅这个任务的 1 个：\n"
        + persona_options
        + "\n\n最后，判断这个任务在一个**纯 Linux 沙箱容器**（有预置工作区文件、"
        + "能读写文件、能运行代码和命令，但**没有**用户本机磁盘如 Windows E:\\/C:\\ 盘、"
        + "没有真实飞书/微信/钉钉消息通道，没有真实 webhook/公众号/外部 API 凭证）"
        + "里能否产出有意义的结果。\n"
        + "规则：如果任务的核心必须依赖上述沙箱不具备的资源（如必须读取用户本机 "
        + "E:\\盘的具体文件，必须真实发送飞书/微信消息），则 runnable=false；"
        + "如果任务主要是读写沙箱内预置文件、写代码、处理数据、回答问题，则 runnable=true。\n"
        + "\n输出一个 JSON，同时给出 bucket、persona 和 runnable：\n"
        + '{"bucket": "九桶之一", "sub_bucket": null, "rationale": "...", '
        + '"persona_name": "某个人设的 name", "runnable": true/false}'
    )
    msgs = [bucket_msgs[0], {"role": "user", "content": combined_user}]
    try:
        raw = client.chat(msgs, max_tokens=300)
    except Exception:  # noqa: BLE001
        rec["bucket"] = "unknown"; rec["classify_source"] = "llm"; return rec

    # Parse: extract JSON then validate
    import re as _re
    m = _re.search(r"\{.*\}", raw, _re.S)
    if not m:
        rec["bucket"] = "unknown"; rec["classify_source"] = "llm"; return rec
    try:
        obj = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        rec["bucket"] = "unknown"; rec["classify_source"] = "llm"; return rec

    parsed = parse_classify_output(json.dumps(obj))  # reuse existing bucket validator
    rec["bucket"] = parsed.get("bucket", "unknown")
    rec["classify_source"] = "llm"

    # Persona from same response
    pn = obj.get("persona_name", "").strip()
    valid_personas = {p["name"] for p in _load_persona_catalog()}
    rec["persona_name"] = pn if pn in valid_personas else "random"
    # Runnability from same response (default True - keep on parse error)
    rec["runnable"] = obj.get("runnable", True)
    return rec


def stage_queries(*, taskspecs_dir: Path | None = None, queries_path: Path | None = None,
                   classify_workers: int = 32) -> int:
    """Generate queries.jsonl.  Returns number of rows written."""

    ts_dir = taskspecs_dir or _TASKSPECS_DIR
    q_path = queries_path or _QUERIES_PATH

    # 1a — load taskspecs
    subdirs = sorted(d for d in ts_dir.iterdir() if d.is_dir())
    records = []
    for d in tqdm(subdirs, desc="Loading taskspecs", unit="file"):
        rec = _load_one_taskspec(d)
        if rec:
            records.append(rec)
    print(f"  loaded {len(records)} taskspecs with seed_query")

    # 1b — classify bucket + persona + runnability (single LLM call per query)
    _ensure_tokenhub_key()
    from data_pipeline.classify import make_default_client

    client = make_default_client()
    print(f"  LLM classifier: {client.model} ({classify_workers} workers)")

    llm_ok = llm_unknown = 0

    with ThreadPoolExecutor(max_workers=classify_workers) as ex:
        futures = {ex.submit(_classify_one, rec, client): rec for rec in records}
        with tqdm(total=len(records), desc="LLM classifying", unit="q", smoothing=0.01) as pbar:
            for fut in as_completed(futures):
                rec = fut.result()
                if rec.get("bucket") != "unknown":
                    llm_ok += 1
                else:
                    llm_unknown += 1
                pbar.set_postfix(ok=llm_ok, unk=llm_unknown, refresh=False)
                pbar.update(1)

    print(f"  classify done: LLM_ok={llm_ok}  LLM_unknown={llm_unknown}")

    # 1c — write queries.jsonl (skip unrunnable; only seed query, no follow-ups)
    q_path.parent.mkdir(parents=True, exist_ok=True)
    skipped_unrunnable = 0
    with open(q_path, "w", encoding="utf-8") as fh:
        for rec in records:
            if not rec.get("runnable", True):
                skipped_unrunnable += 1
                continue
            row = {
                "record_id": rec["record_id"],
                "queries": [rec["seed_query"]],
                "bucket": rec.get("bucket", ""),
                "sub_bucket": rec.get("sub_bucket"),
                "persona_name": rec.get("persona_name", "random"),
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    kept = [r for r in records if r.get("runnable", True)]
    c = Counter(r.get("bucket", "unknown") for r in kept)
    print("  bucket distribution:")
    _mx = max(c.values()) if c else 1
    for b, n in c.most_common():
        bar = "█" * (n * 50 // _mx)
        print(f"    {b:15s} {n:5d}  {bar}")

    written = len(records) - skipped_unrunnable
    print(f"  → {q_path} ({written} rows, skipped {skipped_unrunnable} unrunnable)")
    return written


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


def stage_collect(*, queries_path: Path, num_queries: int, max_concurrent: int,
                   actor_model: str, max_turns: int, hermes_max_turns: int,
                   slot_timeout: int, mode: str = "overwrite",
                   multi_turn: bool = True, out_dir: Path | None = None,
                   backend: str = "e2b", template_name: str = "agentic-cl-sandbox",
                   workspace_dir: str | None = None,
                   model_tag: str | None = None, smoke: bool = False,
                   actor_impl: str = "hermes_cli") -> int:
    """Run parallel sandbox collection — 1 sandbox per query.

    Multi-turn (default): actor(hermes) + observer + questioner drive up to
    K=randint(1,max_turns) turns per query. NO reward / NO winner (that's the
    training stage). multi_turn=False (--no-usersim) skips observer/questioner
    for smoke/debug (seed only, no follow-up).

    Modes (which query_index to (re)run; success rows are NEVER re-run except
    in overwrite):
      - overwrite   : run ALL queries, replace the whole file.
      - incremental : run queries that are MISSING or FAILED; keep existing
                      successes untouched. Safe to resume an interrupted run.
      - retry       : run ONLY existing FAILED rows; keep everything else.

    Post-cleaning: after all queries complete, the output JSONL is piped through
    the C++ ``strip_zw`` binary to strip zero-width chars. Front cleaning on
    seed_queries happens in _load_one_taskspec.
    """
    from scripts.collect.sandbox_grpo_collect import _load_queries, _run_one_collect_query

    # Observer + Questioner (session agents). Created once, shared across threads
    # (each call is stateless per session; LLM clients are thread-safe HTTP).
    observer = questioner = None
    if multi_turn:
        _ensure_tokenhub_key()          # observer/questioner resolve TOKENHUB_API_KEY from agents.yaml
        from agents.observer import Observer
        from agents.questioner import Questioner
        observer = Observer()       # use_llm defaults True; falls back to deterministic on error
        questioner = Questioner()
        print("  multi-turn: observer + questioner enabled (no reward/winner)")
    else:
        print("  --no-usersim（调试）：跳过 observer/questioner，只跑单个 seed task，无追问")

    tasks = _load_queries(str(queries_path), num_queries)
    total = len(tasks)

    # Output layout: <base>/trajectory/<model>/  +  <base>/debug/observer_report/<model>/
    # where <base> defaults to <ROLLOUTS_ROOT>/{real|smoke}. --out-dir overrides <base>.
    base = out_dir or (_ROLLOUTS_ROOT / ("smoke" if smoke else "real"))
    tag = model_tag or _model_tag(actor_model)
    traj_dir = base / "trajectory" / tag
    debug_dir = base / "debug" / "observer_report" / tag
    traj_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    out_root = traj_dir  # manifest + trajectory colocated
    out_file = traj_dir / "grpo_hermes.jsonl"
    observer_log = debug_dir / "observer_reports.jsonl"  # session_id + turn indexed
    print(f"  output: {out_file}")
    print(f"  observer reports: {observer_log}")


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
    qc_dropped = 0
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
                slot_timeout=slot_timeout, backend=backend, template=template_name,
                observer=observer, questioner=questioner, rng_seed=i,
                workspace_dir=workspace_dir, observer_log=observer_log,
                actor_impl=actor_impl,
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
                elif getattr(traj, "qc_hard", False):
                    # Failed hard QC (tool hallucination / truncation / loop / ...)
                    # -> drop from the buffer-bound output. Recorded as an error row
                    # (with qc_codes) so it's not silently lost and can be inspected.
                    err += 1
                    qc_dropped += 1
                    row["error"] = "qc_hard: " + ",".join(row.get("qc_codes") or [])
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
    # Post-cleaning: pipe through C++ strip_zw to remove zero-width chars
    _strip_bin = _REPO / "bin" / "strip_zw"
    if _strip_bin.is_file():
        import tempfile
        tmp = out_file.with_suffix(".jsonl.tmp")
        with open(out_file, "rb") as src, open(tmp, "wb") as dst:
            subprocess.run([str(_strip_bin)], stdin=src, stdout=dst, check=True)
        tmp.replace(out_file)
        print("  [strip_zw] post-cleaned with C++ binary")

    elapsed = time.time() - t0
    total_ok = sum(1 for t in merged.values() if not t.get("error"))
    total_err = sum(1 for t in merged.values() if t.get("error"))
    print(f"  done in {elapsed:.0f}s  this_pass(ok={ok} err={err})  "
          f"file_total(ok={total_ok} err={total_err})  ({len(to_run)/max(1,elapsed):.2f} traj/s)")
    if qc_dropped:
        print(f"  [qc] dropped {qc_dropped} trajectories on hard QC failure (see qc_codes in rows)")

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
    ap.add_argument("--taskspecs-dir", default=None,
                    help="taskspecs root (default datasources/taskspecs)")
    ap.add_argument("--queries", default=None,
                    help="queries JSONL path (default datasets/queries.jsonl)")
    ap.add_argument("--classify-workers", type=int, default=32)
    # ── Stage 2: collect ──
    ap.add_argument("--no-collect", action="store_true", help="skip collection")
    ap.add_argument("--num-queries", type=int, default=100)
    ap.add_argument("--max-concurrent", type=int, default=32)
    ap.add_argument("--backend", default="e2b", help="sandbox backend (e2b|local)")
    ap.add_argument("--template", default="agentic-cl-sandbox", help="sandbox template name")
    ap.add_argument("--actor-model", default="gpt-5")
    ap.add_argument("--actor-impl", default="hermes_cli",
                    choices=["hermes_cli", "hermes_structured"],
                    help="actor backend: hermes_cli (stdout, default) | "
                         "hermes_structured (in-sandbox structured tool_calls + sub-agent capture)")
    ap.add_argument("--max-turns", type=int, default=20, help="K_max: follow-up upper bound (§3.5 U{1..K_max})")
    ap.add_argument("--hermes-max-turns", type=int, default=90, help="hermes ReAct limit")
    ap.add_argument("--slot-timeout", type=int, default=900, help="per-sandbox timeout (s)")
    ap.add_argument("--collect-mode", choices=["overwrite", "incremental", "retry"],
                    default="incremental")
    # 单轮采集（2026-07-23 全链路单轮化）：正式采集默认单轮——只跑 seed query，
    # 不调 observer/questioner（单轮不需要它们）。多轮追问已 deprecated（训练侧 verl
    # 固定 batch 契约 vs 多轮变长不可调和，见 rollout/simulated_session.py docstring）。
    # --no-usersim 现在是正式单轮的开关（不再是"仅调试"）；--multi-turn 可回退多轮（deprecated）。
    ap.add_argument("--no-usersim", "--single-task", "--single-turn", action="store_true",
                    dest="no_usersim", default=True,
                    help="单轮采集（默认）：只跑 seed query，跳过 observer/questioner。"
                         "全链路单轮化后的正式方案。")
    ap.add_argument("--multi-turn", action="store_true",
                    help="（deprecated）回退多轮追问：actor + observer + questioner。"
                         "多轮已搁置，仅留作未来轮池方案恢复时用。")
    ap.add_argument("--out-dir", default=None,
                    help="override output BASE dir (default <ROLLOUTS_ROOT>/{real|smoke}); "
                         "trajectory/<model>/ and debug/observer_report/<model>/ are created under it")
    ap.add_argument("--model-tag", default=None,
                    help="model folder tag (default inferred from --actor-model: gpt5/qwen27b/...)")
    ap.add_argument("--smoke", action="store_true",
                    help="write under <ROLLOUTS_ROOT>/smoke instead of real/")
    ap.add_argument("--workspace-dir", default=None,
                    help="workspace files root (default datasources/taskspecs)")
    args = ap.parse_args()

    t0 = time.time()

    queries_path = Path(args.queries) if args.queries else _QUERIES_PATH

    if args.generate:
        taskspecs_dir = Path(args.taskspecs_dir) if args.taskspecs_dir else _TASKSPECS_DIR
        stage_queries(taskspecs_dir=taskspecs_dir, queries_path=queries_path,
                      classify_workers=args.classify_workers)
    else:
        n = sum(1 for _ in open(queries_path)) if queries_path.exists() else 0
        print(f"[skip] using existing {queries_path} ({n} rows)")

    if not args.no_collect:
        stage_collect(
            queries_path=queries_path,
            num_queries=args.num_queries,
            max_concurrent=args.max_concurrent,
            actor_model=args.actor_model,
            actor_impl=args.actor_impl,
            backend=args.backend,
            template_name=args.template,
            max_turns=args.max_turns,
            hermes_max_turns=args.hermes_max_turns,
            slot_timeout=args.slot_timeout,
            mode=args.collect_mode,
            multi_turn=args.multi_turn,  # 单轮为默认；--multi-turn 显式回退（deprecated）
            out_dir=Path(args.out_dir) if args.out_dir else None,
            workspace_dir=args.workspace_dir,
            model_tag=args.model_tag,
            smoke=args.smoke,
        )

    print(f"\n{'='*60}")
    print(f"ALL DONE in {time.time()-t0:.0f}s")
    print(f"  queries      → {queries_path}")
    _tag = args.model_tag or _model_tag(args.actor_model)
    _base = Path(args.out_dir) if args.out_dir else (_ROLLOUTS_ROOT / ("smoke" if args.smoke else "real"))
    print(f"  trajectories → {_base}/trajectory/{_tag}/grpo_hermes.jsonl")
    print(f"  observer     → {_base}/debug/observer_report/{_tag}/observer_reports.jsonl")


if __name__ == "__main__":
    main()
