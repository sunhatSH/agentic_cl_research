"""Actor facade + factory — decouples "how a turn is run in the sandbox" from the
collection loop (``scripts/sandbox_grpo_collect.py``), mirroring the sandbox-backend
registry in ``rollout/sandbox_client.py``.

Why (2026-07-13): the collection loop hard-coded ``hermes chat -q`` and captured
only stdout TEXT — hermes' internal ReAct (structured tool_calls) was flattened
away, so QC on tool use / structured training were impossible. The facade lets us
swap the actor implementation by NAME without touching the loop:

  - ``hermes_cli``        : current behavior (run ``hermes chat``, wrap stdout as one
                            assistant message). children always empty. Default —
                            keeps existing collection (w3real) bit-for-bit unchanged.
  - ``hermes_structured`` : (P2) run a patched ``run_conversation`` inside the sandbox
                            and return STRUCTURED messages (tool_calls, post-compression
                            = train/infer consistent) + any sub-agent (delegate_task)
                            child trajectories hermes spawned on its own.

Register an out-of-tree actor with ``register_actor(name, builder)`` — NO edit to
the loop. ``make_actor(name)`` only resolves a name -> builder.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

# Path to the sandbox-side capture script (uploaded into the sandbox by the
# structured actor). Kept as a real file (not an inline string) so it is
# lintable / unit-testable on its own.
_CAPTURE_SCRIPT = Path(__file__).with_name("_hermes_capture.py")
_CAPTURE_MARKER = "__CAPTURE__"


@dataclass
class ChildTraj:
    """One sub-agent (delegate_task) child trajectory, captured independently.

    Empty for single-agent turns / the CLI actor. Populated only when hermes
    autonomously delegates and the structured actor harvests each child's own
    ``run_conversation`` messages (its own tool_calls, post-compression).
    """

    task_index: int
    goal: str
    messages: list[dict] = field(default_factory=list)
    system_prompt: str = ""
    base_system_prompt: str = ""
    tools: list[dict] = field(default_factory=list)


@dataclass
class ActorTurn:
    """Result of running ONE turn (one query) of the actor in the sandbox.

    ``messages`` is the parent trajectory AFTER this turn — for the structured
    actor it is the full structured conversation (roles + tool_calls + tool
    results); for the CLI actor it is a single ``{role: assistant, content: stdout}``
    appended to the running history. ``children`` holds any sub-agent trajectories
    hermes spawned this turn (empty unless the actor supports + captured them).
    """

    messages: list[dict] = field(default_factory=list)
    children: list[ChildTraj] = field(default_factory=list)
    ok: bool = True
    error: str = ""
    session_id: str | None = None
    system_prompt: str = ""
    tools: list[dict] = field(default_factory=list)
    api_calls: int = 0
    partial: bool = False


class Actor(Protocol):
    """The one interface the collection loop depends on."""

    def run_turn(
        self,
        sb: Any,
        query: str,
        *,
        conversation_history: Sequence[dict] | None = None,
        model: str,
        base: str,
        max_turns: int,
        timeout: int,
        resume_sid: str | None = None,
        session_id: str | None = None,
    ) -> ActorTurn:
        """Run one query in the sandbox and return the turn's structured result.

        ``conversation_history`` is the prior structured messages (for multi-turn
        continuation); ``resume_sid`` is the CLI-mode hermes session id (legacy
        continuation channel); ``session_id`` is the hermes persistent session id
        for native resume across turns.
        """
        ...


# --------------------------------------------------------------------------- #
# Actor registry — same shape as rollout.sandbox_client.register_backend.      #
# --------------------------------------------------------------------------- #

ActorBuilder = Callable[..., Actor]
_ACTORS: dict[str, ActorBuilder] = {}


def register_actor(name: str, builder: ActorBuilder) -> None:
    """Register (or override) an actor implementation by name."""
    _ACTORS[name] = builder


def make_actor(name: str = "hermes_cli", **kwargs) -> Actor:
    """Resolve a registered actor by name. Unknown -> ValueError listing choices."""
    try:
        builder = _ACTORS[name]
    except KeyError:
        raise ValueError(
            f"unknown actor: {name!r}; registered: {sorted(_ACTORS)}"
        ) from None
    return builder(**kwargs)


# --------------------------------------------------------------------------- #
# Implementation: hermes_cli — wraps the existing `hermes chat -q` stdout path. #
# Behavior is IDENTICAL to the pre-factory code; children is always empty.      #
# --------------------------------------------------------------------------- #


class CliStdoutActor:
    """Run ``hermes chat -q`` and wrap stdout as one assistant message.

    This is the pre-2026-07-13 behavior, unchanged: it delegates to the module-level
    ``_hermes_chat`` in the collection script (passed in as ``chat_fn`` to avoid a
    circular import). Continuation is via hermes ``--resume`` (session_id), NOT via
    ``conversation_history`` — matching the CLI's own memory model.
    """

    def __init__(self, chat_fn: Callable[..., tuple[str, str, bool, str | None]]):
        self._chat_fn = chat_fn

    def run_turn(
        self,
        sb: Any,
        query: str,
        *,
        conversation_history: Sequence[dict] | None = None,
        model: str,
        base: str,  # noqa: ARG002 — CLI reads model/base from sandbox env, not here
        max_turns: int,
        timeout: int,
        resume_sid: str | None = None,
        session_id: str | None = None,  # noqa: ARG002 — CLI uses --resume, not session_id
    ) -> ActorTurn:
        stdout, stderr, ok, sid = self._chat_fn(
            sb, query, model, max_turns, timeout, resume_sid=resume_sid
        )
        # Mirror the loop's prior message construction: user + assistant(stdout).
        msgs: list[dict] = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": stdout},
        ]
        if stderr:
            msgs.append({"role": "system", "content": f"[stderr] {stderr[:300]}"})
        return ActorTurn(
            messages=msgs,
            children=[],
            ok=ok,
            error="" if ok else (stderr[:200] or "hermes produced no output"),
            session_id=sid,
        )


# --------------------------------------------------------------------------- #
# Implementation: hermes_structured (方案3 P2) — run a patched run_conversation  #
# INSIDE the sandbox, return STRUCTURED messages (tool_calls, post-compression   #
# = train/infer consistent) + any delegate_task sub-agent child trajectories.    #
# --------------------------------------------------------------------------- #

_SANDBOX_CAPTURE_PATH = "/tmp/_hermes_capture.py"
_SANDBOX_INPUT_PATH = "/tmp/_hermes_capture_in.json"


class StructuredHermesActor:
    """Capture hermes' STRUCTURED trajectory by running a patched
    ``run_conversation`` inside the sandbox (see rollout/_hermes_capture.py).

    - Uploads the capture script once per sandbox (idempotent).
    - Per turn: writes {query, history} to the sandbox, runs the script, extracts
      the ``__CAPTURE__`` payload from stdout -> ActorTurn(messages, children).
    - Multi-turn continuation is via ``conversation_history`` (structured prior
      messages), NOT hermes ``--resume`` — the CLI resume channel is unused here.
    - ⚠️ Requires in-sandbox validation against the pinned HERMES_VERSION (P2 smoke):
      AIAgent kwargs / run_conversation signature must match. Fails loud (error
      payload) if the sandbox hermes API differs.
    """

    def __init__(self) -> None:
        self._script_src = _CAPTURE_SCRIPT.read_text(encoding="utf-8")
        self._uploaded: set[int] = set()  # id(sb) -> uploaded flag (per-sandbox)

    def _ensure_script(self, sb: Any) -> None:
        if id(sb) in self._uploaded:
            return
        sb._sb.files.write_files(  # type: ignore[union-attr]
            [{"path": _SANDBOX_CAPTURE_PATH, "data": self._script_src}]
        )
        self._uploaded.add(id(sb))

    def run_turn(
        self,
        sb: Any,
        query: str,
        *,
        conversation_history: Sequence[dict] | None = None,
        model: str,  # noqa: ARG002 — sandbox reads model/base/key from its own env/config
        base: str,  # noqa: ARG002
        max_turns: int,
        timeout: int,
        resume_sid: str | None = None,  # noqa: ARG002 — structured uses session_id, not --resume
        session_id: str | None = None,
    ) -> ActorTurn:
        try:
            self._ensure_script(sb)
            spec = {
                "query": query,
                "session_id": session_id,
                "max_iterations": max_turns,
            }
            sb._sb.files.write_files(  # type: ignore[union-attr]
                [{"path": _SANDBOX_INPUT_PATH, "data": json.dumps(spec, ensure_ascii=False)}]
            )
            out = sb._sb.commands.run(  # type: ignore[union-attr]
                f"python3 {_SANDBOX_CAPTURE_PATH} {_SANDBOX_INPUT_PATH}",
                timeout=timeout,
                cwd="/tmp",
            )
            stdout = out.stdout or ""
            payload = _extract_capture(stdout)
            if payload is None:
                return ActorTurn(
                    ok=False,
                    error="capture payload not found; stderr=" + ((out.stderr or "")[:200]),
                )
            children = [
                ChildTraj(
                    task_index=c.get("task_index", -1),
                    goal=c.get("goal", ""),
                    messages=c.get("messages") or [],
                    system_prompt=c.get("system_prompt") or "",
                    base_system_prompt=c.get("base_system_prompt") or "",
                    tools=c.get("tools") or [],
                )
                for c in (payload.get("children") or [])
            ]
            return ActorTurn(
                messages=payload.get("messages") or [],
                children=children,
                ok=bool(payload.get("ok")),
                error=str(payload.get("error") or ""),
                session_id=payload.get("session_id") or session_id,
                system_prompt=payload.get("system_prompt") or "",
                tools=payload.get("tools") or [],
                api_calls=payload.get("api_calls", 0),
                partial=payload.get("partial", False),
            )
        except Exception as exc:  # noqa: BLE001 — isolate slot failures
            return ActorTurn(ok=False, error=f"{type(exc).__name__}: {exc}")


def _extract_capture(stdout: str) -> dict | None:
    """Pull the ``__CAPTURE__<json>`` payload out of (possibly noisy) stdout."""
    idx = stdout.rfind(_CAPTURE_MARKER)
    if idx == -1:
        return None
    raw = stdout[idx + len(_CAPTURE_MARKER):].strip().splitlines()[0]
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


register_actor("hermes_structured", lambda **kw: StructuredHermesActor())
