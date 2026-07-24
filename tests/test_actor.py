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


# --- P2: StructuredHermesActor round-trip (offline, fake sandbox) -------------
# In-sandbox hermes API compat is validated by P2 smoke; here we lock the
# upload/run/extract round-trip + payload parsing.

from rollout.actor import StructuredHermesActor, _extract_capture  # noqa: E402


def test_extract_capture_finds_payload_in_noisy_stdout():
    payload = {"messages": [{"role": "assistant", "content": "x"}], "children": [],
               "ok": True, "error": ""}
    noisy = "hermes banner\nsome logs\n__CAPTURE__" + __import__("json").dumps(payload) + "\n"
    got = _extract_capture(noisy)
    assert got == payload


def test_extract_capture_missing_marker_returns_none():
    assert _extract_capture("no marker here") is None


class _FakeInner:
    def __init__(self, stdout, stderr="", ok=True):
        self._stdout, self._stderr = stdout, stderr
        self.written = []

        class _Files:
            def __init__(self, outer):
                self._outer = outer

            def write_files(self, entries):
                self._outer.written.extend(entries)

        class _Cmds:
            def __init__(self, outer):
                self._outer = outer

            def run(self, cmd, timeout=None, cwd=None):
                class R:
                    pass
                r = R()
                r.stdout, r.stderr, r.exit_code = self._outer._stdout, self._outer._stderr, 0
                return r

        self.files = _Files(self)
        self.commands = _Cmds(self)


class _FakeSb:
    def __init__(self, stdout):
        self._sb = _FakeInner(stdout)


def test_structured_actor_roundtrip_parses_messages_and_children():
    import json as _json
    payload = {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "read_file", "arguments": '{"path":"a"}'}}]},
        ],
        "children": [{"task_index": 0, "goal": "sub", "messages": [{"role": "assistant", "content": "child"}]}],
        "ok": True, "error": "",
    }
    sb = _FakeSb("noise\n__CAPTURE__" + _json.dumps(payload) + "\n")
    a = StructuredHermesActor()
    t = a.run_turn(sb, "q", conversation_history=[], model="m", base="b",
                   max_turns=30, timeout=900)
    assert t.ok is True
    assert t.messages[1]["tool_calls"][0]["function"]["name"] == "read_file"  # structured!
    assert len(t.children) == 1 and t.children[0].goal == "sub"
    assert t.children[0].messages[0]["content"] == "child"
    # script + input were uploaded
    paths = [e["path"] for e in sb._sb.written]
    assert "/tmp/_hermes_capture.py" in paths and "/tmp/_hermes_capture_in.json" in paths



def test_structured_actor_preserves_leading_system_message():
    """Capture script now prepends the hermes system prompt (tool defs + agent
    instructions) as messages[0] on the first turn. The actor must pass it
    through unchanged so the trajectory starts with role:system, matching the
    CC pipeline's _join_system output and pick_prompt's expectation."""
    import json as _json
    payload = {
        "messages": [
            {"role": "system", "content": "# Tools\nYou are a coding agent..."},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ],
        "ephemeral_system_prompt": "# Tools\nYou are a coding agent...",
        "children": [],
        "ok": True, "error": "",
    }
    sb = _FakeSb("__CAPTURE__" + _json.dumps(payload) + "\n")
    t = StructuredHermesActor().run_turn(sb, "q", conversation_history=[],
                                         model="m", base="b", max_turns=30, timeout=900)
    assert t.ok is True
    assert t.messages[0]["role"] == "system"
    assert t.messages[0]["content"].startswith("# Tools")
    assert t.messages[1]["role"] == "user"


def test_structured_actor_no_payload_is_error_not_crash():
    sb = _FakeSb("hermes exploded, no marker")
    t = StructuredHermesActor().run_turn(sb, "q", conversation_history=[], model="m",
                                         base="b", max_turns=30, timeout=900)
    assert t.ok is False and "payload not found" in t.error
