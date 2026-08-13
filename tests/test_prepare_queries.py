"""Tests for stage ② query extraction (scripts/data/prepare_queries.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "prepare_queries",
    Path(__file__).resolve().parent.parent / "scripts" / "data" / "prepare_queries.py",
)
pq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pq)


def test_extract_user_text_variants():
    assert pq.extract_user_text("hello") == "hello"
    assert pq.extract_user_text([{"type": "text", "text": "a"}, {"text": "b"}]) == "a\nb"
    assert pq.extract_user_text(None) == ""


def test_is_summary_block():
    assert pq.is_summary_block('<summary id="x">...</summary>')
    assert pq.is_summary_block("   \n<summary>")  # leading whitespace tolerated
    # A turn that merely embeds a summary later is NOT a pure summary block.
    assert not pq.is_summary_block("The history was compacted: <summary>...")
    assert not pq.is_summary_block("怎么样了")


def test_session_queries_drops_summary_keeps_order():
    record = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "<summary>old</summary>"},
            {"role": "user", "content": "真实问题1"},
            {"role": "assistant", "content": "ans"},
            {"role": "user", "content": "   "},  # blank -> dropped
            {"role": "user", "content": "真实问题2"},
        ]
    }
    assert pq.session_queries(record) == ["真实问题1", "真实问题2"]
    # keep_summary retains the compaction block.
    assert pq.session_queries(record, keep_summary=True)[0].startswith("<summary")


def test_build_query_records_shape_and_limit(tmp_path):
    import json

    src = tmp_path / "raw.jsonl"
    rows = [
        {"record_id": "r1", "record": {"messages": [{"role": "user", "content": "q1"}]}},
        {"record_id": "r2", "record": {"messages": [{"role": "user", "content": "<summary>x"}]}},
    ]
    src.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    out = list(pq.build_query_records(src))
    assert out == [
        {"record_id": "r1", "queries": ["q1"]},
        {"record_id": "r2", "queries": []},  # only-summary session -> empty queries
    ]
    # skip_empty removes the all-summary session.
    assert list(pq.build_query_records(src, skip_empty=True)) == [{"record_id": "r1", "queries": ["q1"]}]
    # limit caps the number of emitted sessions.
    assert len(list(pq.build_query_records(src, limit=1))) == 1
