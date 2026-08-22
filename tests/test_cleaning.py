"""Tests for data.cleaning (ZW strip + garble detection + trajectory filter)."""

from datasources.cleaning import (
    analyze_text,
    clean_messages,
    clean_query,
    strip_zw,
)


def test_strip_zw_removes_word_joiner():
    assert strip_zw("hello⁠world") == "helloworld"


def test_strip_zw_removes_bom():
    assert strip_zw("test﻿end") == "testend"


def test_strip_zw_removes_soft_hyphen():
    assert strip_zw("soft­hyphen") == "softhyphen"


def test_strip_zw_preserves_normal():
    assert strip_zw("hello world 123") == "hello world 123"


def test_analyze_normal_text():
    a = analyze_text("hello world")
    assert a.garble_chars == 0
    assert a.total_chars > 0


def test_analyze_garbled_replacement():
    a = analyze_text("abc�def")
    assert a.garble_chars >= 1
    assert "�" in a.garbled_unique


def test_analyze_mojibake():
    a = analyze_text("café")
    assert a.garble_chars >= 1


def test_analyze_rare_cjk():
    a = analyze_text("\U00020000")
    assert a.garble_chars >= 1


def test_clean_query_normal():
    assert clean_query("calculate 23*17") == "calculate 23*17"


def test_clean_query_strips_zw():
    assert clean_query("hello⁠world") == "helloworld"


def test_clean_query_drops_all_garbled():
    assert clean_query("����") is None


def test_clean_query_keeps_mostly_clean():
    q = "这是一条正常的查询" + "�"
    result = clean_query(q)
    assert result is not None


def test_clean_messages_normal():
    msgs = [{"role": "user", "content": "hello"}]
    r = clean_messages(msgs)
    assert not r.dropped
    assert r.messages[0]["content"] == "hello"


def test_clean_messages_strips_zw():
    msgs = [{"role": "user", "content": "hello⁠world"}]
    r = clean_messages(msgs)
    assert not r.dropped
    assert r.messages[0]["content"] == "helloworld"


def test_clean_messages_drops_high_garble_total():
    # Each message has 50% garble — total ratio is 50%, above 5% threshold.
    # Per-message check catches it first (single_msg_threshold=0.20).
    msgs = [
        {"role": "user", "content": "a" * 10 + "�" * 10},
        {"role": "assistant", "content": "b" * 10 + "�" * 10},
    ]
    r = clean_messages(msgs)
    assert r.dropped
    assert r.drop_reason is not None


def test_clean_messages_drops_high_garble_single_msg():
    msgs = [
        {"role": "user", "content": "normal query"},
        {"role": "assistant", "content": "a" * 5 + "�" * 50},
    ]
    r = clean_messages(msgs, single_msg_policy="drop_all")
    assert r.dropped


def test_clean_messages_drop_tail_policy():
    msgs = [
        {"role": "user", "content": "normal query"},
        {"role": "assistant", "content": "good answer"},
        {"role": "user", "content": "a" * 5 + "�" * 50},
    ]
    r = clean_messages(msgs, single_msg_policy="drop_tail")
    assert not r.dropped
    assert len(r.messages) == 2


def test_clean_messages_ignores_non_string_content():
    msgs = [{"role": "user", "content": None}, {"role": "assistant", "content": 42}]
    r = clean_messages(msgs)
    assert not r.dropped
