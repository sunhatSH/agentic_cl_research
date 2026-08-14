"""Tests for scripts/analysis/qc_trajectory.py — failure-mode QC on structured trajectories."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "qc_trajectory",
    Path(__file__).resolve().parent.parent / "scripts" / "analysis" / "qc_trajectory.py",
)
_qc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_qc)
HARD_CODES, audit_trajectory, scan_messages = _qc.HARD_CODES, _qc.audit_trajectory, _qc.scan_messages


def _tc(name, args):
    return {"function": {"name": name, "arguments": args}}


def test_clean_structured_trajectory_no_hard():
    msgs = [
        {"role": "user", "content": "read a.txt"},
        {"role": "assistant", "content": "", "tool_calls": [_tc("read_file", '{"path":"a.txt"}')]},
        {"role": "tool", "content": "hello", "success": True},
        {"role": "assistant", "content": "done, it says hello"},
    ]
    f = scan_messages(msgs)
    assert not ({x["code"] for x in f} & HARD_CODES)


def test_d1_empty_turn_is_hard():
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": []},  # both empty
    ]
    codes = {x["code"] for x in scan_messages(msgs)}
    assert "D1" in codes


def test_a2_bad_tool_name_and_a2b_bad_args():
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [
            _tc("读文件", "not json"),  # A2 (non-ascii name) + A2b (args not JSON)
        ]},
    ]
    codes = {x["code"] for x in scan_messages(msgs)}
    assert "A2" in codes and "A2b" in codes


def test_a3_xml_leak_is_hard():
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": 'sure <invoke name="read_file">...'},
    ]
    codes = {x["code"] for x in scan_messages(msgs)}
    assert "A3" in codes


def test_b2_consecutive_loop_is_hard():
    # 4 identical calls in a row = blind retry loop -> HARD.
    call = _tc("run", '{"cmd":"ls"}')
    msgs = [{"role": "user", "content": "q"}]
    for _ in range(4):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [call]})
        msgs.append({"role": "tool", "content": "err", "success": False})
    assert "B2" in {x["code"] for x in scan_messages(msgs)}


def test_b2_not_flagged_for_legit_repeats_across_turns():
    # Same read 3x total but NOT consecutive (different calls in between) -> no B2.
    # This is the false-positive that total-count B2 wrongly killed on long sessions.
    read = _tc("read_file", '{"path":"a.txt"}')
    other = _tc("read_file", '{"path":"b.txt"}')
    msgs = [{"role": "user", "content": "q"}]
    for oc in (read, other, read, other, read):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [oc]})
        msgs.append({"role": "tool", "content": "ok", "success": True})
    assert "B2" not in {x["code"] for x in scan_messages(msgs)}


def test_a1_only_when_defined_tools_given():
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [_tc("ghost_tool", "{}")]}]
    # no defined set -> no A1
    assert "A1" not in {x["code"] for x in scan_messages(msgs)}
    # with a defined set that excludes it -> A1
    codes = {x["code"] for x in scan_messages(msgs, defined_tools={"read_file"})}
    assert "A1" in codes


def test_audit_trajectory_flags_child_hard():
    traj = {
        "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "ok, delegating"}],
        "children": [
            {"task_index": 0, "goal": "sub",
             "messages": [{"role": "assistant", "content": "", "tool_calls": []}]},  # child D1
        ],
    }
    res = audit_trajectory(traj)
    assert res["hard"] is True          # child hard bubbles up
    assert res["children_hard"] == 1


def test_audit_trajectory_clean():
    traj = {"messages": [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [_tc("read_file", '{"path":"a"}')]},
        {"role": "tool", "content": "x", "success": True},
        {"role": "assistant", "content": "the file says x, all good here"},
    ], "children": []}
    res = audit_trajectory(traj)
    assert res["hard"] is False
