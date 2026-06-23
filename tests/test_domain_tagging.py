"""Tests for LLM-emitted task-domain tagging + adapter fallback."""

from __future__ import annotations

from trainer.domain_tagging import (
    DEFAULT_BUCKETS,
    build_domain_instruction,
    parse_domain,
)
from trainer.trajectory_adapter import extract_trajectories_from_batch


class FakeBatch:
    def __init__(self, non_tensor_batch):
        self.non_tensor_batch = non_tensor_batch


def test_parse_domain_basic_and_canonical():
    assert parse_domain("...work done\n<task_domain>Finance</task_domain>") == "Finance"
    # case-insensitive tag + value
    assert parse_domain("<TASK_DOMAIN>finance</TASK_DOMAIN>") == "Finance"


def test_parse_domain_aliases():
    assert parse_domain("<task_domain>Knowledge/Analysis</task_domain>") == "Knowledge"
    assert parse_domain("<task_domain>system operations</task_domain>") == "SysOps"
    assert parse_domain("<task_domain>office qa</task_domain>") == "OfficeQA"


def test_parse_domain_last_tag_wins():
    text = "<task_domain>Workflow</task_domain> ... revised <task_domain>SysOps</task_domain>"
    assert parse_domain(text) == "SysOps"


def test_parse_domain_unknown_or_absent_returns_none():
    assert parse_domain("no tag here") is None
    assert parse_domain("<task_domain>Nonsense</task_domain>") is None
    assert parse_domain("") is None


def test_parse_domain_respects_valid_subset():
    # If the configured buckets exclude Finance, an emitted Finance is rejected.
    assert parse_domain("<task_domain>Finance</task_domain>", valid=["Workflow", "SysOps"]) is None


def test_build_instruction_lists_all_buckets():
    instr = build_domain_instruction()
    for name in DEFAULT_BUCKETS:
        assert name in instr
    assert "task_domain" in instr


def test_adapter_recovers_domain_when_no_bucket_field():
    # No bucket/category field -> recover from the assistant's <task_domain> tag.
    msgs = [
        [
            {"role": "user", "content": "帮我算下利润"},
            {"role": "assistant", "content": "结果是...\n<task_domain>Finance</task_domain>"},
        ]
    ]
    out = extract_trajectories_from_batch(FakeBatch({"messages": msgs}))
    assert len(out) == 1
    _traj, bucket, meta = out[0]
    assert bucket == "Finance"
    assert meta["bucket"] == "Finance"


def test_adapter_explicit_bucket_overrides_tag():
    msgs = [
        [
            {"role": "assistant", "content": "x\n<task_domain>Finance</task_domain>"},
        ]
    ]
    out = extract_trajectories_from_batch(FakeBatch({"messages": msgs, "bucket": ["SysOps"]}))
    assert out[0][1] == "SysOps"  # explicit field wins over the emitted tag


def test_adapter_skips_when_no_bucket_and_no_tag():
    msgs = [[{"role": "assistant", "content": "no tag"}]]
    assert extract_trajectories_from_batch(FakeBatch({"messages": msgs})) == []


def test_adapter_respects_valid_buckets():
    msgs = [[{"role": "assistant", "content": "<task_domain>Finance</task_domain>"}]]
    # Finance not in the configured set -> unresolved -> skipped.
    out = extract_trajectories_from_batch(FakeBatch({"messages": msgs}), valid_buckets=["Workflow", "SysOps"])
    assert out == []
