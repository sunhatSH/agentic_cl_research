"""Tests for scripts/data/convert_dataset.py (Gap B: jsonl -> verl parquet rows)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "data"))

import convert_dataset as cd  # noqa: E402


def _rec(messages, record_id="rec-1", meta=None):
    return {"record_id": record_id, "record": {"messages": messages, "meta": meta or {}}}


def test_pick_prompt_takes_system_plus_first_user():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    prompt = cd.pick_prompt(msgs)
    assert [p["role"] for p in prompt] == ["system", "user"]
    assert prompt[1]["content"] == "q1"


def test_extract_queries_lists_all_user_turns():
    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "q2"},
        {"role": "tool", "content": "obs"},
        {"role": "user", "content": "q3"},
    ]
    assert cd.extract_queries(msgs) == ["q1", "q2", "q3"]


def test_bucket_hint_keyword_and_unknown():
    fin = [{"role": "user", "content": "帮我分析这只股票的财务营收"}]
    assert cd.bucket_hint(fin) == "finance"
    blank = [{"role": "user", "content": "hello there"}]
    assert cd.bucket_hint(blank) is None


def test_extract_checkers_number_and_backtick():
    num = [{"role": "assistant", "content": "最终结论：总额为 1234.56"}]
    chk = cd.extract_checkers(num)
    assert len(chk) == 1 and chk[0]["type"] == "regex"
    assert chk[0]["pattern"] == "1234\\.56"  # re.escape applied

    bt = [{"role": "assistant", "content": "文件路径是 `/mnt/data/out.csv` 完成"}]
    chk2 = cd.extract_checkers(bt)
    assert chk2 and "/mnt/data/out\\.csv" in chk2[0]["pattern"]

    none = [{"role": "assistant", "content": "好的，我明白了。"}]
    assert cd.extract_checkers(none) == []


def test_split_assignment_stable_and_bounded():
    a = cd.split_assignment("rec-xyz")
    b = cd.split_assignment("rec-xyz")
    assert a == b  # deterministic
    assert a in ("train", "val")
    # val fraction ~2% over many ids
    ids = [f"rec-{i}" for i in range(2000)]
    vals = sum(1 for i in ids if cd.split_assignment(i) == "val")
    assert 0 < vals < 120  # ~40 expected, allow slack


def test_record_to_row_full_and_skip_empty():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "帮我部署 docker 服务并看日志"},
        {"role": "assistant", "content": "完成，端口 8080"},
    ]
    row = cd.record_to_row(_rec(msgs, "rec-A", {"source": "s"}))
    assert row["data_source"] == "agentic_cl"
    assert row["extra_info"]["record_id"] == "rec-A"
    assert row["extra_info"]["bucket"] == "ops"
    assert row["extra_info"]["num_user_turns"] == 1

    assert cd.record_to_row(_rec([], "rec-empty")) is None
    # no user turn -> skip
    assert cd.record_to_row(_rec([{"role": "system", "content": "x"}], "rec-nosys")) is None
