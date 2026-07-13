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

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


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
    ) -> ActorTurn:
        """Run one query in the sandbox and return the turn's structured result.

        ``conversation_history`` is the prior structured messages (for multi-turn
        continuation); ``resume_sid`` is the CLI-mode hermes session id (legacy
        continuation channel). Implementations use whichever they support.
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
