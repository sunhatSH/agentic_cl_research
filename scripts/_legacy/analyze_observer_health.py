#!/usr/bin/env python3
"""Observer / multi-turn health analyzer for a collected rollout JSONL.

Reads a ``grpo_hermes.jsonl`` (as produced by run_cold_start / sandbox_grpo_collect)
and prints a health report focused on the observer + multi-turn conversation loop:

  - observer report coverage: FS diff / system diff / discrepancy / empty rate
  - port-noise level (ephemeral PORT LISTENING flooding the diff)
  - multi-turn depth: turn distribution, ended_by breakdown, single-turn rate
  - "ignored red flag" cases: session ended by <end_session> while the LAST turn's
    observer report carried a real discrepancy (questioner didn't push back)

Usage:
    .venv/bin/python scripts/analysis/analyze_observer_health.py \
        /mnt/afs_toolcall/sunhao4/agentic_cl_rollouts/smoke/trajectory/gpt5/grpo_hermes.jsonl

    # compare two runs (before/after a fix):
    .venv/bin/python scripts/analysis/analyze_observer_health.py OLD.jsonl NEW.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Reuse the SINGLE source of truth for the red-flag verdict (positive-phrase-wins
# over reassuring openers). Falls back to a local copy if agents/ isn't importable
# (e.g. running the script from an unusual cwd) so the tool still works standalone.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from agents.prompts import _is_real_red_flag as _is_real_discrepancy  # noqa: F401
except Exception:  # noqa: BLE001 — standalone fallback
    _NON_DISCREPANCY_MARKERS = (
        "no clear internal contradiction",
        "no explicit discrepanc",
        "no empty or corrupt",
        "no file content was available",
        "no file-content diff",
        "no filesystem changes",
        "no content-based discrepanc",
        "no discrepanc",
        "none detected",
        "no red flag",
    )
    _RED_FLAG_PHRASES = (
        "discrepancy is present",
        "potential red flag",
        "one potential",
        "one internal",
        "internal inconsistenc",
        "conflicting value",
        "mismatch",
        "truncated",
        "empty file",
        "zero-size",
        "corrupt",
        "placeholder",
    )

    def _is_real_discrepancy(disc: str) -> bool:
        d = (disc or "").strip().lower()
        if not d:
            return False
        if any(p in d for p in _RED_FLAG_PHRASES):
            return True
        return not any(m in d for m in _NON_DISCREPANCY_MARKERS)


def _report_has_flag(r: dict[str, Any]) -> bool:
    """Authoritative red-flag verdict, matching observer._finalize_red_flag.

    The observer's stored ``has_red_flag`` boolean is unreliable in both directions
    (iter4: model set True on clean text, False after a boilerplate opener). So the
    TEXT verdict reconciles it: when ``discrepancies`` is non-empty, the text wins;
    only an empty-text report defers to the stored boolean. This keeps the health
    metric correct even on data collected before the runtime reconcile existed.
    """
    disc = r.get("discrepancies", "") or ""
    if disc.strip():
        return _is_real_discrepancy(disc)
    raw = r.get("has_red_flag")
    return raw if isinstance(raw, bool) else False


def _load(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8", errors="replace") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def analyze(path: Path) -> dict[str, Any]:
    rows = _load(path)
    n = len(rows)

    # trajectory-level
    ended = Counter()
    turns: list[int] = []
    traj_err = 0
    buckets = Counter()

    # report-level
    tot_reports = 0
    with_fs = with_port = with_pkg = with_proc = 0
    with_real_disc = empty_diff = 0

    # coupling: ended satisfied despite a real red flag on the last turn
    ended_despite_disc: list[dict] = []

    # productivity: of turns carrying a red flag, how many were FOLLOWED by another
    # turn (questioner pushed back) vs ended the session. High follow rate = the
    # Iter1 banner is doing its job across ALL turns, not just terminal ones.
    flag_turns = flag_turns_followed = 0

    for d in rows:
        ended[d.get("ended_by", "?")] += 1
        turns.append(int(d.get("num_turns", 0) or 0))
        if d.get("error"):
            traj_err += 1
        buckets[d.get("bucket", "?")] += 1

        reps = d.get("observer_reports", []) or []
        n_reps = len(reps)
        for i, r in enumerate(reps):
            tot_reports += 1
            sd = r.get("state_diff", "") or ""
            if _report_has_flag(r):
                with_real_disc += 1
                flag_turns += 1
                # "followed" = there is a later turn in this trajectory (the
                # questioner did not stop the session right after this flag).
                if i < n_reps - 1:
                    flag_turns_followed += 1
            if any(k in sd for k in ("+ ADDED", "~ MODIFIED", "- REMOVED")):
                with_fs += 1
            if "PORT LISTENING" in sd:
                with_port += 1
            if "INSTALLED" in sd:
                with_pkg += 1
            if "+ PROCESS" in sd:
                with_proc += 1
            if "no filesystem or system changes" in sd or not sd.strip():
                empty_diff += 1

        if reps and d.get("ended_by") == "end_session":
            last = reps[-1]
            if _report_has_flag(last):
                ended_despite_disc.append(
                    {
                        "q": d.get("query_index"),
                        "bucket": d.get("bucket"),
                        "turns": d.get("num_turns"),
                        "disc": (last.get("discrepancies") or "").strip()[:160],
                    }
                )

    single_turn = sum(1 for t in turns if t == 1)

    return {
        "path": str(path),
        "n_traj": n,
        "traj_err": traj_err,
        "ended_by": dict(ended),
        "turns": {
            "min": min(turns) if turns else 0,
            "max": max(turns) if turns else 0,
            "mean": round(sum(turns) / n, 2) if n else 0,
            "single_turn": single_turn,
            "single_turn_pct": round(100 * single_turn / n, 1) if n else 0,
            "dist": dict(sorted(Counter(turns).items())),
        },
        "buckets": dict(buckets),
        "reports": {
            "total": tot_reports,
            "with_fs_diff": with_fs,
            "with_port_noise": with_port,
            "port_noise_pct": round(100 * with_port / tot_reports, 1) if tot_reports else 0,
            "with_installed_pkg": with_pkg,
            "with_process": with_proc,
            "with_real_discrepancy": with_real_disc,
            "empty_diff": empty_diff,
        },
        "red_flag_followed": {
            "flag_turns": flag_turns,
            "followed": flag_turns_followed,
            "followed_pct": round(100 * flag_turns_followed / flag_turns, 1) if flag_turns else 0,
        },
        "ended_despite_real_discrepancy": ended_despite_disc,
    }


def _print(rep: dict[str, Any]) -> None:
    print("=" * 70)
    print(f"FILE: {rep['path']}")
    print(f"  trajectories: {rep['n_traj']}  (traj-level error: {rep['traj_err']})")
    t = rep["turns"]
    print(f"  turns: min={t['min']} max={t['max']} mean={t['mean']}  "
          f"single-turn: {t['single_turn']} ({t['single_turn_pct']}%)")
    print(f"  turn dist: {t['dist']}")
    print(f"  ended_by: {rep['ended_by']}")
    print(f"  buckets: {rep['buckets']}")
    r = rep["reports"]
    print(f"  observer reports: {r['total']}")
    print(f"    with FS diff        : {r['with_fs_diff']}")
    print(f"    with real discrepancy: {r['with_real_discrepancy']}")
    print(f"    port noise          : {r['with_port_noise']} ({r['port_noise_pct']}%)")
    print(f"    installed pkg / proc: {r['with_installed_pkg']} / {r['with_process']}")
    print(f"    empty diff          : {r['empty_diff']}")
    rf = rep["red_flag_followed"]
    print(f"  red-flag turns FOLLOWED by another turn: "
          f"{rf['followed']}/{rf['flag_turns']} ({rf['followed_pct']}%)  "
          f"(higher = questioner pushes back on flags)")
    ed = rep["ended_despite_real_discrepancy"]
    print(f"  !! ended satisfied despite real red flag: {len(ed)}")
    for e in ed:
        print(f"     q{e['q']} [{e['bucket']}] turns={e['turns']}: {e['disc']}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="one or more grpo_hermes.jsonl paths")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of text")
    args = ap.parse_args()

    reports = [analyze(Path(p)) for p in args.paths]
    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2))
    else:
        for rep in reports:
            _print(rep)


if __name__ == "__main__":
    main()
