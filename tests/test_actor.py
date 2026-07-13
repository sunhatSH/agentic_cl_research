"""Tests for the Actor facade + factory (rollout/actor.py).

P1 contract: CliStdoutActor reproduces the pre-factory `hermes chat` stdout
behavior exactly (messages = [user, assistant(stdout), (system[stderr])],
children always empty); the registry resolves actors by name.
"""

from __future__ import annotations

import pytest

from rollout.actor import (
    ActorTurn,
    ChildTraj,
    CliStdoutActor,
    make_actor,
    register_actor,
)


def _ok_chat(sb, query, model, max_turns, timeout, resume_sid=None):
    return ("the answer", "", True, "sess-1")


def _err_chat(sb, query, model, max_turns, timeout, resume_sid=None):
    return ("", "kaboom", False, None)


def test_cli_actor_success_matches_legacy_message_shape():
    a = CliStdoutActor(chat_fn=_ok_chat)
    t = a.run_turn(None, "do X", conversation_history=[], model="m", base="b",
                   max_turns=30, timeout=900)
    assert t.messages == [
        {"role": "user", "content": "do X"},
        {"role": "assistant", "content": "the answer"},
    ]
    assert t.ok is True
    assert t.session_id == "sess-1"
    assert t.children == []  # CLI actor never captures sub-agents
    assert t.error == ""


def test_cli_actor_error_carries_stderr_and_system_message():
    a = CliStdoutActor(chat_fn=_err_chat)
    t = a.run_turn(None, "q", conversation_history=[], model="m", base="b",
                   max_turns=30, timeout=900)
    assert t.ok is False
    assert t.error == "kaboom"
    # stderr surfaces as a system message (legacy loop behavior)
    assert {"role": "system", "content": "[stderr] kaboom"} in t.messages


def test_cli_actor_empty_output_error_fallback():
    def _empty(sb, q, m, mt, to, resume_sid=None):
        return ("", "", False, None)

    t = CliStdoutActor(chat_fn=_empty).run_turn(
        None, "q", conversation_history=[], model="m", base="b", max_turns=30, timeout=900
    )
    assert t.error == "hermes produced no output"


def test_cli_actor_passes_resume_sid_through():
    seen = {}

    def _cap(sb, query, model, max_turns, timeout, resume_sid=None):
        seen["resume_sid"] = resume_sid
        return ("ok", "", True, "sid-2")

    CliStdoutActor(chat_fn=_cap).run_turn(
        None, "q", conversation_history=[], model="m", base="b",
        max_turns=30, timeout=900, resume_sid="prev-sid",
    )
    assert seen["resume_sid"] == "prev-sid"


def test_registry_resolves_by_name():
    register_actor("t_ok", lambda **kw: CliStdoutActor(chat_fn=_ok_chat))
    a = make_actor("t_ok")
    assert isinstance(a, CliStdoutActor)


def test_registry_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown actor"):
        make_actor("no_such_actor_xyz")


def test_child_traj_and_actorturn_defaults():
    ct = ChildTraj(task_index=0, goal="sub goal")
    assert ct.messages == []
    turn = ActorTurn()
    assert turn.messages == [] and turn.children == [] and turn.ok is True
