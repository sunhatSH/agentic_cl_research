"""Observer agent (no persona, objective) -- doc §3.3 / §7.3 / O6.

DIFF-DRIVEN observation (2026-06-19). The observer's evidence is a DETERMINISTIC
before/after diff of the agent's sandbox workspace (files created / modified /
removed THIS turn, WITH content), computed by code -- not the model's reading of
what the actor *claimed*. The actor trajectory is still passed, but only as
CLAIMS to cross-check against the real diff (it fills ``discrepancies``).

Why diff-driven instead of actor-claim-driven (the design's original §3.3 lead):
  1. Hallucination -- the actor's narrative can assert files/values that do not
     exist; only the sandbox diff is ground truth.
  2. Lost intermediates -- claims state *results* and routinely omit the
     intermediate artifacts (temp files, computed values) that the design most
     wants captured (they are "easily overwritten by later steps"); a workspace
     diff surfaces them regardless of whether the actor mentioned them.
  3. Anti reward-hacking -- if reward grounds on claims, the policy learns to
     *say* it finished without finishing; grounding on the real effect removes
     that attack surface (the whole reason the observer is a separate model).
  4. Omission, not just commission -- a wrong/missing file the actor never
     mentioned is invisible to a claim reader but appears in the diff.
  5. Content-level verification -- "Q3 total = 12345" can be checked against the
     actual file bytes in the diff; a claim reader has no content to compare.
  6. Determinism -- the diff is produced by code (reproducible), so the model
     only narrates/cross-checks; it is not a second layer of model error inside
     the evidence itself.

Performance (2026-06-19, to keep observer cost off the critical path):
  - #1 skip the observer LLM entirely on an empty diff (no FS change this turn).
  - #2 one snapshot per turn: the driver carries the previous turn's post-snapshot
    forward as the next turn's baseline (``observe(..., post=...)``), instead of
    snapshotting both before and after.
  - #3 change-detection by (size, mtime) instead of hashing every file's full
    bytes every snapshot -- the probe only os.stat's + reads a small text head.
  - #4 the prompt carries only the diff (no redundant full file tree), with capped
    content excerpts and a capped number of rendered files.

The probe uses ONLY ``sandbox.run_code`` (the backend-agnostic interface), so it
works identically on the local / Tencent E2B / Alibaba AgentBay backends.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Protocol

from agents.base import ChatClient, resolve_observer_client
from agents.prompts import build_observer_prompt
from agents.schema import ObservationReport

# Rendering caps for the prompt (#4): bound tokens regardless of workspace size.
_MAX_RENDER_FILES = 50
_MAX_RENDER_CHARS = 1200


class ReadOnlySandbox(Protocol):
    """Read-only surface the observer is allowed on the live winner instance."""

    def run_code(self, code: str, language: str = "python") -> Any:
        """Run a read-only probe; returns an object with .stdout / .stderr."""
        ...


# Read-only workspace snapshot probe. Runs in the sandbox via run_code and prints
# a JSON map {path: {size, mtime, text|binary}}. Change-detection uses (size, mtime)
# -- NO full-file hashing (that read every byte of every file on every snapshot);
# we only os.stat each file and read a small text head for evidence.
_SNAPSHOT_PROBE = (
    "import os, json\n"
    "ROOT='.'; MAX_TEXT=2048\n"
    "SKIP={'.git','__pycache__','node_modules','.cache','.ipynb_checkpoints','.venv'}\n"
    "out={}\n"
    "for root, dirs, files in os.walk(ROOT):\n"
    "    if root.count(os.sep) > 5:\n"
    "        dirs[:]=[]; continue\n"
    "    dirs[:]=[d for d in dirs if d not in SKIP]\n"
    "    for fn in files:\n"
    "        p=os.path.join(root, fn)\n"
    "        try:\n"
    "            st=os.stat(p)\n"
    "        except OSError:\n"
    "            continue\n"
    "        rec={'size': st.st_size, 'mtime': round(st.st_mtime, 3)}\n"
    "        try:\n"
    "            with open(p,'rb') as f:\n"
    "                raw=f.read(MAX_TEXT)\n"
    "            try:\n"
    "                rec['text']=raw.decode('utf-8'); rec['truncated']=st.st_size>MAX_TEXT\n"
    "            except UnicodeDecodeError:\n"
    "                rec['binary']=True\n"
    "        except OSError:\n"
    "            pass\n"
    "        out[p]=rec\n"
    "print(json.dumps(out))\n"
)


def _trajectory_text(actor_trajectory: list[dict[str, Any]]) -> str:
    """Flatten the winner message list into the actor's claim text."""
    lines: list[str] = []
    for msg in actor_trajectory:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                (c.get("text", "") if isinstance(c, dict) else str(c)) for c in content
            )
        role = msg.get("role", "?")
        lines.append(f"[{role}] {str(content).strip()}")
    return "\n".join(lines)


def snapshot_workspace(sandbox: ReadOnlySandbox | None) -> dict[str, dict]:
    """Read-only snapshot of the sandbox workspace: ``{path: {size, mtime, text?}}``.

    Returns ``{}`` on no sandbox / probe failure (observation must never crash the
    session). Backend-agnostic: uses only ``run_code``.
    """
    if sandbox is None:
        return {}
    try:
        res = sandbox.run_code(_SNAPSHOT_PROBE)
        out = (getattr(res, "stdout", "") or "").strip()
        data = json.loads(out) if out else {}
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 -- observation must never crash the session
        return {}


def _excerpt(rec: dict) -> dict:
    if rec.get("binary"):
        return {"kind": "binary", "size": rec.get("size", 0)}
    return {
        "kind": "text",
        "size": rec.get("size", 0),
        "content_excerpt": rec.get("text", ""),
        "truncated": bool(rec.get("truncated")),
    }


def _sig(rec: dict) -> tuple:
    """Cheap change signature: (size, mtime). Any write updates mtime."""
    return (rec.get("size"), rec.get("mtime"))


def diff_snapshots(pre: dict[str, dict] | None, post: dict[str, dict] | None) -> dict[str, list]:
    """Deterministic diff of two workspace snapshots: added / modified / removed.

    Change is detected by (size, mtime) -- no content hashing. ``modified`` carries
    the new content excerpt plus the prior one (``before_excerpt``).
    """
    pre = pre or {}
    post = post or {}
    added: list[dict] = []
    modified: list[dict] = []
    removed: list[dict] = []
    for path, rec in post.items():
        if path not in pre:
            added.append({"path": path, **_excerpt(rec)})
        elif _sig(pre[path]) != _sig(rec):
            entry = {"path": path, **_excerpt(rec)}
            entry["before_excerpt"] = pre[path].get("text", "")
            modified.append(entry)
    for path in pre:
        if path not in post:
            removed.append({"path": path})
    return {"added": added, "modified": modified, "removed": removed}


def _diff_is_empty(diff: dict[str, list]) -> bool:
    return not any(diff.get(k) for k in ("added", "modified", "removed"))


def _render_file(prefix: str, f: dict) -> str:
    head = f"{prefix} {f['path']} ({f.get('kind', '?')}, {f.get('size', 0)}B)"
    body = f.get("content_excerpt", "")
    if not body:
        return head
    if len(body) > _MAX_RENDER_CHARS:
        body = body[:_MAX_RENDER_CHARS] + " …[truncated]"
    indented = "\n".join("    " + ln for ln in body.splitlines())
    return f"{head}\n{indented}"


def _format_changes(diff: dict[str, list]) -> str:
    """Render a before/after diff as the observer's ground-truth evidence (capped)."""
    if _diff_is_empty(diff):
        return "mode: before/after diff -- (no filesystem changes detected this turn)"
    out = ["mode: before/after diff (what THIS turn changed in the workspace)"]
    shown = 0
    for f in diff["added"]:
        if shown >= _MAX_RENDER_FILES:
            break
        out.append(_render_file("+ ADDED", f))
        shown += 1
    for f in diff["modified"]:
        if shown >= _MAX_RENDER_FILES:
            break
        out.append(_render_file("~ MODIFIED", f))
        shown += 1
    out += [f"- REMOVED {f['path']}" for f in diff["removed"]]
    total = len(diff["added"]) + len(diff["modified"])
    if total > _MAX_RENDER_FILES:
        out.append(f"… and {total - _MAX_RENDER_FILES} more changed files (truncated)")
    return "\n".join(out)


def _format_state(post: dict[str, dict]) -> str:
    """Render the current workspace (no baseline) as content-level evidence (capped)."""
    if not post:
        return ""
    out = ["mode: current workspace snapshot (no baseline -- content-level evidence)"]
    for p, r in sorted(post.items())[:_MAX_RENDER_FILES]:
        out.append(_render_file("•", {"path": p, **_excerpt(r)}))
    if len(post) > _MAX_RENDER_FILES:
        out.append(f"… and {len(post) - _MAX_RENDER_FILES} more files (truncated)")
    return "\n".join(out)


class Observer:
    """Objective state observer. Persona-free (§3.3), diff-driven evidence."""

    def __init__(self, client: ChatClient | None = None, *, max_tokens: int = 1024):
        self._client = client
        self._max_tokens = max_tokens

    @property
    def client(self) -> ChatClient:
        if self._client is None:
            self._client = resolve_observer_client()
        return self._client

    def snapshot(self, sandbox: ReadOnlySandbox | None) -> dict[str, dict]:
        """Capture the workspace; pass forward as next turn's ``baseline``/``post``."""
        return snapshot_workspace(sandbox)

    def observe(
        self,
        actor_trajectory: Sequence[dict[str, Any]],
        sandbox: ReadOnlySandbox | None = None,
        *,
        baseline: dict[str, dict] | None = None,
        post: dict[str, dict] | None = None,
    ) -> ObservationReport:
        """Produce the objective report from the sandbox DIFF + the actor's claims.

        Args:
            actor_trajectory: winner messages -- used as CLAIMS to cross-check.
            sandbox: live (winner) instance; snapshotted for ``post`` if ``post`` is
                not supplied.
            baseline: pre-turn snapshot. With it we emit a true before/after diff;
                without it we fall back to a current-state snapshot.
            post: the post-turn snapshot. The driver passes this (carried forward as
                the next turn's baseline) so the observer does NOT snapshot twice (#2).
        """
        actor_text = _trajectory_text(list(actor_trajectory))
        if post is None:
            post = snapshot_workspace(sandbox) if sandbox is not None else {}
        file_tree = "\n".join(sorted(post.keys()))

        if baseline is not None:
            diff = diff_snapshots(baseline, post)
            if _diff_is_empty(diff):
                # #1: nothing changed on disk this turn -> skip the observer LLM.
                return ObservationReport(
                    actor_claims=actor_text,
                    file_tree=file_tree,
                    state_diff="mode: before/after diff -- (no filesystem changes this turn)",
                )
            state_diff = _format_changes(diff)
        else:
            state_diff = _format_state(post)

        # #4: prompt carries only the diff (no redundant full file tree).
        messages = build_observer_prompt(actor_trajectory=actor_text, state_diff=state_diff)
        try:
            raw = self.client.chat(messages, max_tokens=self._max_tokens)
        except Exception:  # noqa: BLE001 -- degrade to minimal report, never crash
            return ObservationReport(
                actor_claims=actor_text, file_tree=file_tree, state_diff=state_diff
            )
        return parse_observation_report(
            raw, fallback_claims=actor_text, fallback_tree=file_tree, fallback_diff=state_diff
        )


def parse_observation_report(
    text: str,
    *,
    fallback_claims: str = "",
    fallback_tree: str = "",
    fallback_diff: str = "",
) -> ObservationReport:
    """Robustly parse the observer's JSON into an ObservationReport.

    Falls back to a minimal report (actor text / tree / diff) when the model
    output is not valid JSON. ``state_diff`` is the code-computed evidence and is
    always carried (the model does not emit it), so reward/debug see the diff.
    """
    obj: Any = None
    if text:
        try:
            obj = json.loads(text)
        except (TypeError, ValueError):
            start, end = text.find("{"), text.rfind("}")
            if 0 <= start < end:
                try:
                    obj = json.loads(text[start : end + 1])
                except (TypeError, ValueError):
                    obj = None
    if not isinstance(obj, dict):
        return ObservationReport(
            actor_claims=fallback_claims, file_tree=fallback_tree, state_diff=fallback_diff
        )

    def _as_list(v: Any) -> list[dict]:
        return [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []

    return ObservationReport(
        intermediate=_as_list(obj.get("intermediate")),
        final=_as_list(obj.get("final")),
        actor_claims=str(obj.get("actor_claims") or fallback_claims),
        discrepancies=str(obj.get("discrepancies") or ""),
        file_tree=str(obj.get("file_tree") or fallback_tree),
        state_diff=str(obj.get("state_diff") or fallback_diff),
    )
