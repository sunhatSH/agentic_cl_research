"""Step 1 — 从 OpenClaw 采集目录抽取每个主会话的【首 query】。

OpenClaw 会话目录结构（实测，见 ``data_pipeline/__init__.py``）::

    <root>/<NNNNNN>/                 # 6 位序号目录（会话单元）
      agent/sessions/
        dyn-tmp-main-*.jsonl         # 主会话日志（每目录恰 1 个，本步取它）
        dyn-tmp-main-*.trajectory.jsonl   # 遥测（systemPrompt+tools，Step 3 用）
        <UUID>.jsonl                 # subagent 子会话（本步跳过）
      workspace_init/                # 沙箱初始状态种子
      workspace_final/               # 沙箱终态

主会话日志是事件流，``type=="message"`` 的事件携带 ``message.{role, content}``，
``role ∈ {user, assistant, toolResult}``。**首 query** = 文件中第一个
``role=="user"`` 的 message 的文本拼接。

只取主会话（``dyn-tmp-main-*``，排除 ``trajectory`` 与 subagent 纯-UUID 文件）：
subagent 是主会话 ``sessions_spawn`` 派生的子任务，不是独立样本（本次范围外）。

纯函数 + 流式读文件，不联网、不依赖 GPU，可离线单测。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

# str | Path 都接受（type alias，供签名注解）。
AnyPath = "str | Path"

# 主会话日志文件名前缀（OpenClaw 固定），trajectory / subagent 不以此开头或带后缀。
_MAIN_SESSION_GLOB = "dyn-tmp-main-*.jsonl"


def iter_session_dirs(root: AnyPath) -> Iterator[Path]:
    """遍历采集根目录下的每个会话目录（6 位序号子目录）。

    按目录名排序，保证多次运行输出稳定。
    """
    root_path = Path(root)
    if not root_path.is_dir():
        return
    for child in sorted(root_path.iterdir(), key=lambda p: p.name):
        if child.is_dir() and child.name.isdigit():
            yield child


def find_main_session_log(session_dir: AnyPath) -> Path | None:
    """定位一个会话目录里的主会话日志（``dyn-tmp-main-*.jsonl``，排除 trajectory）。

    每目录应有且仅有 1 个；多于 1 个时取字典序首个并告警（数据异常，不应静默）。
    """
    candidates = [
        p
        for p in Path(session_dir).joinpath("agent", "sessions").glob(_MAIN_SESSION_GLOB)
        if "trajectory" not in p.name
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name)
    if len(candidates) > 1:
        # 数据异常：一个目录里多个主会话。取首个但暴露给调用方（返回首个 + 不静默）。
        import sys

        print(
            f"[warn] {session_dir}: {len(candidates)} main-session logs found, "
            f"using {candidates[0].name}",
            file=sys.stderr,
        )
    return candidates[0]


def iter_message_events(log_path: AnyPath) -> Iterator[dict]:
    """流式读主会话日志，按文件顺序产出 ``type=="message"`` 的事件。

    文件追加顺序即事件发生顺序；malformed 行跳过（JSONL 容错）。
    """
    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                import json

                evt = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(evt, dict) and evt.get("type") == "message":
                yield evt


def extract_text(content) -> str:
    """把 message.content（str | content-parts list | None）拍平为纯文本。

    user/assistant 的 content 通常是 ``[{type:text, text}, ...]``；
    str 直接返回；其它降级为空串。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict):
                t = p.get("text") or p.get("content")
                if isinstance(t, str):
                    parts.append(t)
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts)
    return "" if content is None else str(content)


def first_user_query(session_dir: AnyPath) -> tuple[str, str] | None:
    """取一个会话目录主会话的首个 user query。

    返回 ``(session_id, query_text)``；无主会话日志或无 user message 时返回 None。
    """
    log_path = find_main_session_log(session_dir)
    if log_path is None:
        return None
    session_id = log_path.stem  # dyn-tmp-main-...（去 .jsonl）
    for evt in iter_message_events(log_path):
        msg = evt.get("message") or {}
        if msg.get("role") != "user":
            continue
        text = extract_text(msg.get("content"))
        if text.strip():
            return session_id, text
    return None


def extract_first_queries(
    root: AnyPath,
    *,
    limit: int | None = None,
) -> list[dict]:
    """遍历所有会话目录，抽取首 query 记录。

    每条::

        {"record_id": "000001",          # 会话目录序号（稳定 join key）
         "session_id": "dyn-tmp-main-...",# OpenClaw session id
         "first_query": "...",            # 首 user query 文本
         "session_dir": "<root>/000001",  # 原始会话目录绝对路径
         "workspace_init": "<root>/000001/workspace_init"}  # 沙箱种子路径（可能不存在）

    跳过无主会话日志 / 无 user query 的目录（计入返回的 skipped 统计见 CLI）。
    """
    records: list[dict] = []
    for session_dir in iter_session_dirs(root):
        got = first_user_query(session_dir)
        if got is None:
            continue
        session_id, query = got
        records.append(
            {
                "record_id": session_dir.name,
                "session_id": session_id,
                "first_query": query,
                "session_dir": str(session_dir.resolve()),
                "workspace_init": str(session_dir.joinpath("workspace_init").resolve()),
            }
        )
        if limit is not None and len(records) >= limit:
            break
    return records
