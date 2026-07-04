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
    assert parse_domain("...work done\n<task_domain>finance</task_domain>") == "finance"
    # case-insensitive tag + value
    assert parse_domain("<TASK_DOMAIN>Finance</TASK_DOMAIN>") == "finance"


def test_parse_domain_aliases():
    assert parse_domain("<task_domain>knowledge</task_domain>") == "qa"
    assert parse_domain("<task_domain>system operations</task_domain>") == "ops"
    assert parse_domain("<task_domain>office qa</task_domain>") == "office"


def test_parse_domain_last_tag_wins():
    text = "<task_domain>workflow</task_domain> ... revised <task_domain>ops</task_domain>"
    assert parse_domain(text) == "ops"


def test_parse_domain_unknown_or_absent_returns_none():
    assert parse_domain("no tag here") is None
    assert parse_domain("<task_domain>Nonsense</task_domain>") is None
    assert parse_domain("") is None


def test_parse_domain_respects_valid_subset():
    # If the configured buckets exclude finance, an emitted finance is rejected.
    assert parse_domain("<task_domain>finance</task_domain>", valid=["workflow", "ops"]) is None


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
            {"role": "assistant", "content": "结果是...\n<task_domain>finance</task_domain>"},
        ]
    ]
    out = extract_trajectories_from_batch(FakeBatch({"messages": msgs}))
    assert len(out) == 1
    _traj, bucket, meta = out[0]
    assert bucket == "finance"
    assert meta["bucket"] == "finance"


def test_adapter_explicit_bucket_overrides_tag():
    msgs = [
        [
            {"role": "assistant", "content": "x\n<task_domain>finance</task_domain>"},
        ]
    ]
    out = extract_trajectories_from_batch(FakeBatch({"messages": msgs, "bucket": ["ops"]}))
    assert out[0][1] == "ops"  # explicit field wins over the emitted tag


def test_adapter_skips_when_no_bucket_and_no_tag():
    msgs = [[{"role": "assistant", "content": "no tag"}]]
    assert extract_trajectories_from_batch(FakeBatch({"messages": msgs})) == []


def test_adapter_respects_valid_buckets():
    msgs = [[{"role": "assistant", "content": "<task_domain>finance</task_domain>"}]]
    # finance not in the configured set -> unresolved -> skipped.
    out = extract_trajectories_from_batch(FakeBatch({"messages": msgs}), valid_buckets=["workflow", "ops"])
    assert out == []
