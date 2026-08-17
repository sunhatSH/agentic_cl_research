#!/usr/bin/env python3
"""Post-collection quality check: agent_data_tools (rule) → LLMChecker (LLM).

Replaces the old inline qc_trajectory.audit_trajectory() with the two
external quality-check tools at /mnt/afs_toolcall/sunhao4/workspace/quality-check/:

  Stage A — agent_data_tools validate-openai
    Rule-based: field integrity, suspicious chars, token injection, empty turns.
    Takes meta.json → {annotation: path/to/data.jsonl}.
    Errors logged to <log_dir>/.

  Stage B — LLMChecker
    LLM-based 7-round quality annotation: mechanical → giveup → text_quality →
    scoring → verification → subagent → OCR. Each round reads prior findings.
    Config via YAML; tokenhub endpoints for LLM.

Usage:
    # Full QC pipeline
    python scripts/analysis/qc_trajectories.py --input rollouts/smoke/trajectory/gpt5/grpo_hermes.jsonl

    # Rule-only (skip LLMChecker)
    python scripts/analysis/qc_trajectories.py --input ... --no-llm

    # LLM-only (skip agent_data_tools)
    python scripts/analysis/qc_trajectories.py --input ... --no-rules
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
_QC_ROOT = Path("/mnt/afs_toolcall/sunhao4/workspace/quality-check")
_AGENT_TOOLS_SRC = _QC_ROOT / "agent_data_tools" / "src"
_LLMCHECKER_ROOT = _QC_ROOT / "LLMChecker"

# ── Stage A: agent_data_tools ────────────────────────────────────────────

def _run_agent_data_tools(input_jsonl: Path, log_dir: Path) -> bool:
    """Run agent-data-tools validate-openai on a single JSONL.

    Returns True if validation ran successfully (exit 0); failures are logged to
    <log_dir>/ not thrown.
    """
    # Write a minimal meta.json pointing to the input file.
    meta = {
        "cold_start": {
            "annotation": str(input_jsonl.resolve()),
            "length": _count_lines(input_jsonl),
        }
    }
    meta_path = log_dir / "meta_qc.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    cmd = [
        sys.executable, "-m", "agent_data_tools.cli",
        "validate-openai",
        "-i", str(meta_path),
        "-l", str(log_dir),
        "-w", "4",
    ]
    env = {"PYTHONPATH": str(_AGENT_TOOLS_SRC), **__import__("os").environ}
    print(f"[qc:rules] {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, env=env, cwd=str(_REPO))
    ok = result.returncode == 0
    print(f"[qc:rules] {'PASS' if ok else 'FAIL'} (exit {result.returncode})", flush=True)
    return ok


# ── Stage B: LLMChecker ──────────────────────────────────────────────────

def _write_llmchecker_config(input_jsonl: Path, output_dir: Path) -> Path:
    """Write a LLMChecker YAML config for the given input JSONL."""
    config = {
        "input": str(input_jsonl.resolve()),
        "output_dir": str(output_dir.resolve()),
        "data_format": "openai",
        "defaults": {"rpm": 60, "tpm": 500000},
        "endpoints": [
            {
                "base_url": "https://tokenhub.sensetime.com/v1",
                "model": "gpt-5.4-mini",
                "rpm": 60,
                "tpm": 500000,
                "keys": [{"key": __import__("os").environ.get("TOKENHUB_API_KEY", "")}],
            }
        ],
        "concurrency": 8,
        "request_timeout": 120,
        "max_retries": 1,
        "ramp_start": 2,
        "ramp_step": 2,
        "ramp_interval": 30,
        "id_field": "query_index",
    }
    config_path = output_dir / "llmchecker_config.yaml"
    # Write JSON as a simple dict — YAML is a superset so PyYAML can read both.
    # But llmchecker uses its own YAML parser. Write proper YAML.
    try:
        import yaml
        config_path.write_text(yaml.dump(config, default_flow_style=False), encoding="utf-8")
    except ImportError:
        # Fallback: write as JSON and hope for the best
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return config_path


def _run_llmchecker(input_jsonl: Path, output_dir: Path) -> bool:
    """Run LLMChecker multi-round quality annotation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = _write_llmchecker_config(input_jsonl, output_dir)

    cmd = [
        sys.executable, "-m", "llmchecker",
        "--config", str(config_path),
    ]
    env = {"PYTHONPATH": str(_LLMCHECKER_ROOT), **__import__("os").environ}
    log = output_dir / "llmchecker.log"
    print(f"[qc:llm] {' '.join(cmd)}  (log={log})", flush=True)
    with open(log, "w") as fh:
        result = subprocess.run(cmd, env=env, cwd=str(_REPO), stdout=fh, stderr=subprocess.STDOUT)
    ok = result.returncode == 0
    print(f"[qc:llm] {'PASS' if ok else 'FAIL'} (exit {result.returncode})", flush=True)
    return ok


# ── helpers ───────────────────────────────────────────────────────────────

def _count_lines(path: Path) -> int:
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


# ── main ──────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Post-collection quality check pipeline")
    ap.add_argument("--input", required=True, help="grpo_hermes.jsonl path")
    ap.add_argument("--output-dir", default=None, help="QC output dir (default: alongside input)")
    ap.add_argument("--no-rules", action="store_true", help="skip agent_data_tools")
    ap.add_argument("--no-llm", action="store_true", help="skip LLMChecker")
    args = ap.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(args.output_dir) if args.output_dir else input_path.parent.parent / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)

    rules_ok = llm_ok = True
    if not args.no_rules:
        rules_ok = _run_agent_data_tools(input_path, out_dir)
    if not args.no_llm:
        llm_ok = _run_llmchecker(input_path, out_dir)

    if rules_ok and llm_ok:
        print(f"[qc] ALL PASS — output in {out_dir}")
    else:
        failed = []
        if not rules_ok: failed.append("rules")
        if not llm_ok: failed.append("llm")
        print(f"[qc] FAILED stages: {', '.join(failed)} — see {out_dir}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
