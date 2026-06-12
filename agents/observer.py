"""Observer agent (no persona, objective) -- doc §3.3 / §7.3 / O6.

Reads the winner actor's trajectory (its claims) and a read-only view of the
winner sandbox, then emits an objective ``ObservationReport`` that is consumed by
BOTH the Questioner and the Reward judge (§3.3 要点 3).

The observer has read-only sandbox access (§3.3 表). It probes the workspace
driven by the actor's claims; the concrete read-only command set / token budget
is O6 and can be tuned later -- here we collect a file tree plus the actor text
and let the model assemble the structured report.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Protocol

from agents.base import ChatClient, resolve_observer_client
from agents.prompts import build_observer_prompt
from agents.schema import ObservationReport


class ReadOnlySandbox(Protocol):
    """Read-only surface the observer is allowed on the live winner instance."""

    def run_code(self, code: str, language: str = "python") -> Any:
        """Run a read-only probe; returns an object with .stdout / .stderr."""
        ...


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


def _probe_file_tree(sandbox: ReadOnlySandbox | None) -> str:
    """Best-effort read-only workspace tree (fallback evidence). Empty on failure."""
    if sandbox is None:
        return ""
    code = (
        "import os\n"
        "for root, dirs, files in os.walk('.'):\n"
        "    depth = root.count(os.sep)\n"
        "    if depth > 3:\n"
        "        dirs[:] = []\n"
        "        continue\n"
        "    for f in files:\n"
        "        print(os.path.join(root, f))\n"
    )
    try:
        res = sandbox.run_code(code)
        return (getattr(res, "stdout", "") or "").strip()
    except Exception:  # noqa: BLE001 -- observation must never crash the session
        return ""


class Observer:
    """Objective state observer. Persona-free (§3.3)."""

    def __init__(self, client: ChatClient | None = None, *, max_tokens: int = 1024):
        self._client = client
        self._max_tokens = max_tokens

    @property
    def client(self) -> ChatClient:
        if self._client is None:
            self._client = resolve_observer_client()
        return self._client

    def observe(
        self,
        actor_trajectory: Sequence[dict[str, Any]],
        sandbox: ReadOnlySandbox | None = None,
    ) -> ObservationReport:
        """Produce the objective report from the winner trajectory + workspace."""
        actor_text = _trajectory_text(list(actor_trajectory))
        file_tree = _probe_file_tree(sandbox)
        messages = build_observer_prompt(actor_trajectory=actor_text, file_tree=file_tree)
        try:
            raw = self.client.chat(messages, max_tokens=self._max_tokens)
        except Exception:  # noqa: BLE001 -- degrade to minimal report, never crash
            return ObservationReport(actor_claims=actor_text, file_tree=file_tree)
        return parse_observation_report(raw, fallback_claims=actor_text, fallback_tree=file_tree)


def parse_observation_report(
    text: str, *, fallback_claims: str = "", fallback_tree: str = ""
) -> ObservationReport:
    """Robustly parse the observer's JSON into an ObservationReport.

    Falls back to a minimal report (just the actor text / tree) when the model
    output is not valid JSON -- the failure/patience path keys off is_empty().
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
        return ObservationReport(actor_claims=fallback_claims, file_tree=fallback_tree)

    def _as_list(v: Any) -> list[dict]:
        return [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []

    return ObservationReport(
        intermediate=_as_list(obj.get("intermediate")),
        final=_as_list(obj.get("final")),
        actor_claims=str(obj.get("actor_claims") or fallback_claims),
        discrepancies=str(obj.get("discrepancies") or ""),
        file_tree=str(obj.get("file_tree") or fallback_tree),
    )
