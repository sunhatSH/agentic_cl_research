"""data_pipeline 三步管道的单元测试（纯逻辑，不触网/不依赖 GPU）。

extract / route 用真实 OpenClaw 事件流结构的内联 fixture 验证；classify 用
mock ChatClient 验证 prompt 构造 + JSON 容错解析（不调真模型）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_pipeline.classify import (
    SUB_BUCKETS,
    build_classify_prompt,
    classify_query,
    parse_classify_output,
)
from data_pipeline.extract import (
    extract_text,
    find_main_session_log,
    first_user_query_text,
    iter_message_events,
)
from data_pipeline.route import rebuild_messages, route_trajectories

# --------------------------------------------------------------------------- #
# Fixtures: 仿 OpenClaw 事件流结构的迷你会话目录                                 #
# --------------------------------------------------------------------------- #


def _evt(typ: str, msg: dict | None = None, **extra) -> dict:
    """构造一个 OpenClaw 日志事件。"""
    e = {"type": typ, "id": extra.get("id", "x"), "timestamp": 0}
    if msg is not None:
        e["message"] = msg
    e.update(extra)
    return e


def _user(text: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _assistant_with_toolcall(call_id: str, name: str, args: dict, text: str = "") -> dict:
    content: list[dict] = []
    if text:
        content.append({"type": "text", "text": text})
    content.append({"type": "toolCall", "id": call_id, "name": name, "arguments": args})
    return {"role": "assistant", "content": content}


def _tool_result(call_id: str, text: str, *, is_error: bool = False, exit_code: int = 0) -> dict:
    return {
        "role": "toolResult",
        "toolCallId": call_id,
        "toolName": "exec",
        "content": [{"type": "text", "text": text}],
        "details": {"status": "completed", "exitCode": exit_code},
        "isError": is_error,
    }


def _assistant_thinking() -> dict:
    """含 thinking part 的 assistant（route 应丢弃 thinking）。"""
    return {
        "role": "assistant",
        "content": [
            {"type": "thinking", "text": "让我想想..."},
            {"type": "text", "text": "已完成"},
        ],
    }


def _make_session(tmp_path: Path, record_id: str, events: list[dict], *, system: str | None = None, tools: list | None = None) -> Path:
    """在 tmp_path 下造一个 OpenClaw 会话目录（含主日志 + trajectory）。"""
    sess_dir = tmp_path / record_id
    sessions_dir = sess_dir / "agent" / "sessions"
    sessions_dir.mkdir(parents=True)
    (sess_dir / "workspace_init").mkdir()

    log_name = f"dyn-tmp-main-test-{record_id}-12345"
    log_path = sessions_dir / f"{log_name}.jsonl"
    with open(log_path, "w", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")

    # trajectory.jsonl（context.compiled 带 systemPrompt + tools）
    traj_path = sessions_dir / f"{log_name}.trajectory.jsonl"
    traj_events = [
        {"type": "session.started", "data": {}},
        {
            "type": "context.compiled",
            "data": {
                "systemPrompt": system or "",
                "tools": tools or [],
                "messages": [],
            },
        },
        {"type": "session.ended", "data": {"status": "ok"}},
    ]
    with open(traj_path, "w", encoding="utf-8") as fh:
        for e in traj_events:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    return sess_dir


@pytest.fixture
def sample_root(tmp_path: Path) -> Path:
    """造两个会话目录：一个单轮、一个多轮带工具调用。"""
    # 会话 1：单轮 user + assistant 文本
    _make_session(
        tmp_path,
        "000001",
        [
            _evt("message", _user("帮我算一下 Q3 营收")),
            _evt("message", _assistant_with_toolcall("c1", "exec", {"command": "echo 100"}, text="计算中")),
            _evt("message", _tool_result("c1", "100")),
            _evt("message", {"role": "assistant", "content": [{"type": "text", "text": "Q3 营收是 100"}]}),
        ],
        system="你是一个财务助手",
        tools=[{"name": "exec", "parameters": {"type": "object", "properties": {}}}],
    )
    # 会话 2：多轮，含 thinking（route 应丢）
    _make_session(
        tmp_path,
        "000002",
        [
            _evt("message", _user("第一轮问题")),
            _evt("message", _assistant_thinking()),
            _evt("message", _user("第二轮追问")),
            _evt("message", {"role": "assistant", "content": [{"type": "text", "text": "回答"}]}),
        ],
        system=None,
    )
    # 一个非数字目录（应被 iter_session_dirs 跳过）
    (tmp_path / "not-a-session").mkdir()
    return tmp_path


# --------------------------------------------------------------------------- #
# extract                                                                      #
# --------------------------------------------------------------------------- #


class TestExtract:
    """底层工具函数测试（1:1 聚合逻辑已留空待重写，见 extract.py）。

    extract_text / find_main_session_log / iter_message_events / first_user_query_text
    是与 query 聚合无关的底层工具，1:n 重写后仍复用，故保留测试。
    """

    def test_extract_text_flattens_parts(self) -> None:
        assert extract_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a\nb"
        assert extract_text("plain") == "plain"
        assert extract_text(None) == ""

    def test_find_main_session_log_skips_trajectory_and_subagent(self, sample_root: Path) -> None:
        # 主日志存在
        log = find_main_session_log(sample_root / "000001")
        assert log is not None
        assert "trajectory" not in log.name
        assert log.name.startswith("dyn-tmp-main")

    def test_first_user_query_text_returns_first_user(self, sample_root: Path) -> None:
        query = first_user_query_text(find_main_session_log(sample_root / "000002"))
        assert query == "第一轮问题"

    def test_iter_message_events_skips_malformed(self, sample_root: Path) -> None:
        log = find_main_session_log(sample_root / "000001")
        # 追加一行坏 JSON
        with open(log, "a", encoding="utf-8") as fh:
            fh.write("not json\n")
        events = list(iter_message_events(log))
        # 坏行被跳过，原 4 个 message 事件仍在
        assert len(events) == 4

    def test_extract_initial_queries_not_implemented(self) -> None:
        """1:n 聚合逻辑待重写，当前抛 NotImplementedError。"""
        import pytest

        with pytest.raises(NotImplementedError):
            from data_pipeline.extract import extract_initial_queries

            extract_initial_queries("/nonexistent")


# --------------------------------------------------------------------------- #
# classify                                                                     #
# --------------------------------------------------------------------------- #


class TestClassify:
    def test_sub_buckets_cover_all_seven(self) -> None:
        from trainer.domain_tagging import DEFAULT_BUCKETS

        assert set(SUB_BUCKETS.keys()) == set(DEFAULT_BUCKETS)
        for subs in SUB_BUCKETS.values():
            assert len(subs) >= 1

    def test_build_prompt_contains_bucket_and_sub_options(self) -> None:
        msgs = build_classify_prompt("算营收")
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        # 九桶名都在
        for b in SUB_BUCKETS:
            assert b in msgs[1]["content"]
        # 子桶也在（如 finance）
        assert "finance" in msgs[1]["content"]

    def test_parse_valid_json(self) -> None:
        out = parse_classify_output('{"bucket": "finance", "sub_bucket": "finance", "rationale": "营收计算"}')
        assert out["bucket"] == "finance"
        assert out["sub_bucket"] == "finance"
        assert out["rationale"] == "营收计算"

    def test_parse_invalid_sub_bucket_keeps_bucket(self) -> None:
        out = parse_classify_output('{"bucket": "finance", "sub_bucket": "不存在的子桶"}')
        assert out["bucket"] == "finance"
        assert out["sub_bucket"] is None

    def test_parse_invalid_bucket_to_unknown(self) -> None:
        out = parse_classify_output('{"bucket": "Magic", "sub_bucket": "x"}')
        assert out["bucket"] == "unknown"
        assert out["sub_bucket"] is None

    def test_parse_non_json_falls_back_to_block_regex(self) -> None:
        out = parse_classify_output('前缀文字 {"bucket": "ops", "sub_bucket": "ops"} 后缀')
        assert out["bucket"] == "ops"
        assert out["sub_bucket"] == "ops"

    def test_parse_truncated_or_empty_is_unknown_not_silent(self) -> None:
        assert parse_classify_output("")["bucket"] == "unknown"
        assert parse_classify_output("半截 JSON 没闭合 {")[ "bucket"] == "unknown"
        assert parse_classify_output("纯文本无 JSON")["bucket"] == "unknown"

    def test_classify_query_with_mock_client(self) -> None:
        class MockClient:
            model = "mock-model"

            def chat(self, messages, *, max_tokens=256):
                # 返回合法分类 JSON
                return '{"bucket": "finance", "sub_bucket": "finance", "rationale": "营收"}'

        out = classify_query("帮我算 Q3 营收", MockClient())
        assert out["bucket"] == "finance"
        assert out["sub_bucket"] == "finance"
        assert out["ok"] is True
        assert out["model"] == "mock-model"

    def test_classify_query_llm_failure_is_unknown_not_crash(self) -> None:
        class FailingClient:
            model = "mock"

            def chat(self, messages, *, max_tokens=256):
                raise ConnectionError("network down")

        out = classify_query("x", FailingClient())
        assert out["bucket"] == "unknown"
        assert out["ok"] is False
        assert "network down" in out["rationale"]


# --------------------------------------------------------------------------- #
# route                                                                        #
# --------------------------------------------------------------------------- #


class TestRoute:
    def test_rebuild_messages_converts_to_openai_format(self, sample_root: Path) -> None:
        messages, tools = rebuild_messages(sample_root / "000001")
        # system + user + assistant(toolCall) + tool + assistant(text)
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "你是一个财务助手"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "帮我算一下 Q3 营收"
        # assistant toolCall → OpenAI tool_calls
        asst = messages[2]
        assert asst["role"] == "assistant"
        assert asst["tool_calls"][0]["function"]["name"] == "exec"
        assert asst["tool_calls"][0]["id"] == "c1"
        # toolResult → role:tool
        assert messages[3]["role"] == "tool"
        assert messages[3]["tool_call_id"] == "c1"
        assert messages[3]["content"] == "100"
        # tools 来自 trajectory
        assert tools and tools[0]["name"] == "exec"

    def test_rebuild_messages_drops_thinking(self, sample_root: Path) -> None:
        messages, _ = rebuild_messages(sample_root / "000002")
        # 无 system（trajectory systemPrompt 为空字符串 → 不 prepend）
        assert messages[0]["role"] == "user"
        # assistant 的 thinking part 被丢，只留 text
        asst = messages[1]
        assert asst["role"] == "assistant"
        assert asst["content"] == "已完成"
        assert "thinking" not in str(asst)

    def test_route_writes_per_bucket_files(self, sample_root: Path, tmp_path: Path) -> None:
        classified = [
            {
                "record_id": "000001",
                "session_id": "dyn-tmp-main-test-000001-12345",
                "first_query": "营收",
                "session_dir": str(sample_root / "000001"),
                "bucket": "finance",
                "sub_bucket": "finance",
                "rationale": "营收",
                "model": "mock",
                "ok": True,
            },
            {
                "record_id": "000002",
                "session_id": "dyn-tmp-main-test-000002-12345",
                "first_query": "对话",
                "session_dir": str(sample_root / "000002"),
                "bucket": "qa",
                "sub_bucket": "what",
                "rationale": "多轮",
                "model": "mock",
                "ok": True,
            },
            {
                "record_id": "000099",
                "session_id": "x",
                "first_query": "x",
                "session_dir": str(sample_root / "000099"),  # 不存在
                "bucket": "workflow",
                "sub_bucket": "workflow",
                "model": "mock",
                "ok": True,
            },
            {
                "record_id": "000100",
                "session_id": "x",
                "first_query": "x",
                "session_dir": str(sample_root / "000100"),
                "bucket": "unknown",  # 应被跳过
                "sub_bucket": None,
                "model": "mock",
                "ok": False,
            },
        ]
        out_root = tmp_path / "buckets"
        stats = route_trajectories(classified, out_root)
        # finance、qa 各 1 条入桶；workflow 因 session_dir 不存在 skipped；unknown 跳过
        assert stats["per_bucket"] == {"finance": 1, "qa": 1}
        assert stats["unknown"] == 1
        assert stats["skipped"] == 1
        assert stats["total"] == 4

        # 验证写入的轨迹文件内容
        traj_path = out_root / "finance" / "000001.jsonl"
        assert traj_path.exists()
        traj = json.loads(traj_path.read_text(encoding="utf-8"))
        assert traj["bucket"] == "finance"
        assert traj["sub_bucket"] == "finance"
        assert traj["record_id"] == "000001"
        assert traj["messages"][0]["role"] == "system"
        assert traj["meta"]["source"] == "sample105_v2"
        assert traj["meta"]["cold_seed"] is True
        # token/logprob/reward 留空占位（本批数据无）
        assert traj["reward"] is None

    def test_route_unknown_bucket_skipped(self, sample_root: Path, tmp_path: Path) -> None:
        classified = [
            {
                "record_id": "000001",
                "session_id": "x",
                "first_query": "x",
                "session_dir": str(sample_root / "000001"),
                "bucket": "unknown",
                "sub_bucket": None,
                "model": "mock",
                "ok": False,
            }
        ]
        stats = route_trajectories(classified, tmp_path / "out")
        assert stats["per_bucket"] == {}
        assert stats["unknown"] == 1


