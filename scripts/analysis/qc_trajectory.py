"""Trajectory quality-check — failure-mode QC for hermes trajectories (方案3 P3).

Ported from tongronglei's ``gen_code/v2/qc_checks.py`` failure-mode taxonomy
(scan_messages), adapted to THIS project:
  - runs on a trajectory's structured ``messages`` (roles + tool_calls + tool
    results) and, when present, each sub-agent ``children[i].messages``;
  - only the FAILURE-MODE checks are ported — NOT tongronglei's showcase
    ``scan_structure`` gates (>=3 delegate, >=20 tool rounds), which are his
    multi-agent showcase requirements, not a general quality bar;
  - NO API keys, NO CLI-schedule code copied — pure detection logic.

Failure taxonomy (HARD = should bounce the trajectory):
  A1  hallucinated tool name      (name not in the turn's defined tools)   HARD
  A2  tool-name token jitter      (non [A-Za-z_][\\w.-]* chars)            HARD
  A2b tool-args jitter/injection  (args not JSON obj / illegal / tag leak) HARD
  A3  XML literal leak to content (<tool_call>/<invoke>/<think> in content)HARD
  B2  same tool+args >=4 in a row (blind retry loop, consecutive)          HARD
  D1  both-empty assistant turn   (no content AND no tool_calls)           HARD
  B1  same-path write_file >=3    (full-rewrite-on-error)                  warn
  C1  early bail                  (<=3 asst turns, 0 tool calls)           warn
  C2  blame-the-user after error                                          warn
  C3  self-blame "tool broken"                                            warn

Two entry points:
  audit_trajectory(traj) -> {findings, hard, codes, children_hard}   (one row)
  main(): audit a grpo_hermes.jsonl and print a code histogram + hard-fail rate.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]*$")
_XML_LEAK = (
    "<tool_call>", "</tool_call>", "<function=", "<think>", "</think>",
    "<|tool", "<｜tool", "<invoke name=", "</invoke>", "<parameter name=",
)
_ZERO_WIDTH = ("⁠", "﻿", "­", "�")
_BLAME_USER = (
    "you provided", "your input", "incorrect input", "you gave",
    "please provide a valid", "您提供的", "你提供的", "您输入", "请检查您", "请提供正确",
)
_SELF_BLAME = (
    "tool is broken", "tool seems broken", "tool appears broken", "tool is buggy",
    "tool isn't working", "tool is not working", "工具坏了", "工具有问题", "工具故障",
    "the tool failed me", "broken tool",
)
_ERROR_MARKERS = (
    "cannot access", "no such file", "not found", "error:", "traceback",
    "command timed out", "permission denied", "failed",
)

HARD_CODES = {"A1", "A2", "A2b", "A3", "B2", "D1"}

# B2 loop threshold: min CONSECUTIVE identical (tool, args) calls to flag a blind
# retry loop. 4 (not 3) + consecutive-only avoids killing legitimate long multi-turn
# sessions that re-issue the same read across different turns (2026-07-13 struct smoke:
# total-count B2 wrongly hard-dropped 8/13 long trajectories with 100s of tool calls).
_B2_CONSECUTIVE = 4


def _has_illegal_chars(s: Any) -> bool:
    if not isinstance(s, str):
        return False
    if any(ch in s for ch in _ZERO_WIDTH):
        return True
    for ch in s:
        o = ord(ch)
        if o in (0x09, 0x0A, 0x0D):
            continue
        if o < 0x20 or 0xD800 <= o <= 0xDFFF or 0xE000 <= o <= 0xF8FF or (o & 0xFFFF) in (0xFFFE, 0xFFFF):
            return True
    return False


def _looks_error(content: Any) -> bool:
    if not isinstance(content, str) or not content:
        return False
    low = content.lower()
    return any(k in low for k in _ERROR_MARKERS)


def _args_obj(arguments: Any) -> tuple[dict | None, str]:
    if isinstance(arguments, dict):
        return arguments, json.dumps(arguments, ensure_ascii=False)
    if isinstance(arguments, str):
        try:
            return json.loads(arguments), arguments
        except Exception:
            return None, arguments
    return None, str(arguments)


def _tool_names(messages: list[dict]) -> set[str]:
    """Best-effort set of tool names actually CALLED in this trajectory.

    Our trajectories don't carry a tools-definition schema, so A1 (hallucinated
    tool) can't compare against a defined set — we skip A1 unless a caller passes
    ``defined``. The other checks (A2/A2b/A3/B2/D1/...) need no schema.
    """
    names: set[str] = set()
    for m in messages or []:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                nm = (tc.get("function") or {}).get("name") or ""
                if nm:
                    names.add(nm)
    return names


def scan_messages(messages: list[dict], *, label: str = "parent",
                  defined_tools: set[str] | None = None) -> list[dict]:
    """Failure-mode findings for one message list (parent or a child)."""
    f: list[dict] = []
    defined = defined_tools or set()
    asst = [m for m in (messages or []) if m.get("role") == "assistant"]
    n_tool_calls = 0
    write_paths: Counter = Counter()
    # B2 = blind-retry LOOP: the SAME (tool, args) issued repeatedly with no
    # intervening progress. In a long multi-turn hermes session calling read_file
    # on the same path 3x across different turns is legitimate (re-checking state),
    # so we detect the max CONSECUTIVE run of an identical call, not the total.
    call_seq: list[tuple] = []  # ordered (name, raw) of every tool call

    for i, m in enumerate(messages or []):
        role = m.get("role")
        content = m.get("content") or ""
        if role == "assistant":
            if isinstance(content, str) and any(tok in content for tok in _XML_LEAK):
                f.append({"code": "A3", "where": f"{label}#{i}", "detail": "XML token in content"})
            tcs = m.get("tool_calls") or []
            n_tool_calls += len(tcs)
            if not (content or "").strip() and not tcs:
                f.append({"code": "D1", "where": f"{label}#{i}", "detail": "empty content + no tool_calls"})
            for tc in tcs:
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                if defined and name not in defined:
                    f.append({"code": "A1", "where": f"{label}#{i}", "detail": f"undefined tool '{name}'"})
                if not _NAME_RE.match(name):
                    f.append({"code": "A2", "where": f"{label}#{i}", "detail": f"bad tool name '{name[:40]}'"})
                obj, raw = _args_obj(fn.get("arguments"))
                if obj is None:
                    f.append({"code": "A2b", "where": f"{label}#{i}", "detail": "args not JSON object"})
                if _has_illegal_chars(raw) or any(t in (raw or "") for t in ("<think>", "<tool_call>")):
                    f.append({"code": "A2b", "where": f"{label}#{i}", "detail": "illegal char / tag in args"})
                if name in ("write_file", "write") and isinstance(obj, dict) and obj.get("path"):
                    write_paths[obj["path"]] += 1
                call_seq.append((name, raw))
    # B1 keeps total-count (warn only): many writes to one path is a soft smell.
    for p, c in write_paths.items():
        if c >= 3:
            f.append({"code": "B1", "where": label, "detail": f"write_file x{c} to {p}"})
    # B2 HARD: a CONSECUTIVE run of the identical (tool, args) — a blind-retry loop
    # with no intervening progress. Total-count would wrongly flag legitimate
    # re-checks across a long multi-turn session, so we use max consecutive run.
    max_run, run, run_sig = 1, 1, None
    for sig in call_seq:
        if sig == run_sig:
            run += 1
            if run > max_run:
                max_run, max_sig = run, sig
        else:
            run, run_sig = 1, sig
    if max_run >= _B2_CONSECUTIVE:
        f.append({"code": "B2", "where": label,
                  "detail": f"{max_sig[0]} identical call x{max_run} in a row (loop)"})
    if n_tool_calls == 0 and len(asst) <= 3:
        f.append({"code": "C1", "where": label, "detail": f"{len(asst)} asst turns, 0 tool calls"})
    # C2 / C3: assistant text right after a failed tool result
    msgs = messages or []
    for i, m in enumerate(msgs):
        if m.get("role") != "tool":
            continue
        tc = m.get("content") or ""
        if not ((m.get("success") is False) or _looks_error(tc)):
            continue
        for j in range(i + 1, min(i + 3, len(msgs))):
            if msgs[j].get("role") == "assistant":
                txt = (msgs[j].get("content") or "").lower()
                if any(k in txt for k in _BLAME_USER):
                    f.append({"code": "C2", "where": f"{label}#{j}", "detail": "blames user after tool error"})
                if any(k in txt for k in _SELF_BLAME):
                    f.append({"code": "C3", "where": f"{label}#{j}", "detail": "calls tool broken"})
                break
    return f


def audit_trajectory(traj: dict) -> dict:
    """Audit one collected trajectory (parent messages + any sub-agent children).

    Returns {findings, codes(set), hard(bool), children_hard(int)}. ``hard`` is
    True if the parent OR any child trips a HARD code.
    """
    parent_msgs = traj.get("messages") or []
    findings = scan_messages(parent_msgs, label="parent")
    child_hard = 0
    for ci, child in enumerate(traj.get("children") or []):
        cmsgs = child.get("messages") or []
        cf = scan_messages(cmsgs, label=f"child{ci}")
        findings.extend(cf)
        if {x["code"] for x in cf} & HARD_CODES:
            child_hard += 1
    codes = {x["code"] for x in findings}
    return {
        "findings": findings,
        "codes": sorted(codes),
        "hard": bool(codes & HARD_CODES),
        "children_hard": child_hard,
    }


def _iter_jsonl(path: Path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", help="grpo_hermes.jsonl to audit")
    ap.add_argument("--show", type=int, default=0, help="print first N hard-fail examples")
    args = ap.parse_args()

    rows = list(_iter_jsonl(Path(args.jsonl)))
    n = len(rows)
    code_ct: Counter = Counter()
    hard = 0
    examples: list[dict] = []
    for d in rows:
        res = audit_trajectory(d)
        for c in res["codes"]:
            code_ct[c] += 1
        if res["hard"]:
            hard += 1
            if len(examples) < args.show:
                examples.append({"query_index": d.get("query_index"), "bucket": d.get("bucket"),
                                 "codes": res["codes"]})
    print(f"=== QC failure-mode audit: {n} trajectories ===")
    print(f"HARD-fail : {hard}/{n} ({round(100 * hard / n) if n else 0}%)")
    print("code histogram (trajectories tripping each):")
    for c, ct in code_ct.most_common():
        tag = "HARD" if c in HARD_CODES else "warn"
        print(f"  {c:14s} {ct:4d}  [{tag}]")
    for e in examples:
        print(f"  q{e['query_index']} [{e['bucket']}]: {e['codes']}")


if __name__ == "__main__":
    main()
