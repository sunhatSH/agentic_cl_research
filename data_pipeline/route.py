"""Step 3 — 按已标注的 bucket，把每个主会话的【完整轨迹】重建并路由进对应桶。

输入：``classify_queries`` 的产出（每条带 ``record_id / bucket / sub_bucket /
session_dir``）。对每条：

1. 从 ``session_dir`` 重读主会话日志，把**全部** message 事件重建为 OpenAI
   chat messages（顺序保持）：
     - ``user``       → {role:user, content:str}
     - ``assistant``  → {role:assistant, content:str, tool_calls:[...]}（toolCall
                        part 转 OpenAI tool_calls；text/thinking part 并入 content，
                        thinking 按"只取首 query / 不含推理过程"原则丢弃）
     - ``toolResult`` → {role:tool, tool_call_id, content:str}（role 改名）
2. 从同名 ``*.trajectory.jsonl`` 的 ``context.compiled`` 事件取 systemPrompt +
   tools，prepend system message（主日志无 system role）。
3. 整条轨迹写入 ``data/buckets/<bucket>/<record_id>.jsonl``，含 sub_bucket 字段。

subagent 子会话**不处理**（本次范围外，主会话 ``sessions_spawn`` 的 toolCall
已记录派生关系）。

输出对齐 ``doc/Buffer_冷启动数据需求.md §2.2`` 的轨迹字段（messages / bucket /
record_id），但 token/logprob/reward 等需 π₀ 重跑的字段留空——本批数据只有
文本，是"对话语料"层，不是完整冷启动 trajectory（见数据盘点）。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

# str | Path 都接受（type alias，供签名注解）。
AnyPath = "str | Path"

from data_pipeline.extract import find_main_session_log, iter_message_events  # noqa: E402

# 遇到 toolCall part 时，OpenAI tool_calls 的结构键。
_TOOL_CALL_ID_KEY = "id"
_TOOL_CALL_NAME_KEY = "name"


def _assistant_content_to_openai(content: list) -> tuple[str, list[dict]]:
    """把 assistant message 的 content-parts 转 OpenAI (text_content, tool_calls)。

    OpenAI tool-use 格式：assistant 同时有 ``content``（文本，可空）和
    ``tool_calls``（结构化调用）。OpenClaw 把两者都放在 content-parts 里：
    ``{type:text}`` → content；``{type:toolCall}`` → tool_calls；
    ``{type:thinking}`` 丢弃（只取首 query、不含推理过程）。
    """
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for p in content or []:
        if not isinstance(p, dict):
            continue
        ptype = p.get("type")
        if ptype == "text":
            t = p.get("text")
            if isinstance(t, str):
                text_parts.append(t)
        elif ptype == "toolCall":
            args = p.get("arguments")
            tool_calls.append(
                {
                    "id": p.get(_TOOL_CALL_ID_KEY, ""),
                    "type": "function",
                    "function": {
                        "name": p.get(_TOOL_CALL_NAME_KEY, ""),
                        "arguments": json.dumps(args, ensure_ascii=False) if args is not None else "",
                    },
                }
            )
        # thinking: 丢弃
    return "\n".join(text_parts), tool_calls


def _tool_result_to_openai(msg: dict) -> dict:
    """toolResult message → OpenAI tool role message。

    OpenClaw: {role:'toolResult', toolCallId, toolName, content, details, isError}
    OpenAI  : {role:'tool', tool_call_id, content(str)}
    """
    from data_pipeline.extract import extract_text

    content = msg.get("content")
    text = extract_text(content)
    return {
        "role": "tool",
        "tool_call_id": msg.get("toolCallId", ""),
        "content": text,
    }


def rebuild_messages(session_dir: AnyPath) -> tuple[list[dict], list[dict]]:
    """重建主会话完整轨迹为 OpenAI chat messages + tools。

    返回 ``(messages, tools)``。system message（若有 trajectory 的 systemPrompt）
    prepend 到 messages 头部；tools 来自 trajectory ``context.compiled``。
    无 systemPrompt 时 messages 不含 system（主日志本就无 system role）。
    """
    log_path = find_main_session_log(session_dir)
    if log_path is None:
        return [], []
    messages: list[dict] = []
    for evt in iter_message_events(log_path):
        msg = evt.get("message") or {}
        role = msg.get("role")
        if role == "user":
            from data_pipeline.extract import extract_text

            messages.append({"role": "user", "content": extract_text(msg.get("content"))})
        elif role == "assistant":
            text, tool_calls = _assistant_content_to_openai(msg.get("content"))
            m: dict = {"role": "assistant", "content": text}
            if tool_calls:
                m["tool_calls"] = tool_calls
            messages.append(m)
        elif role == "toolResult":
            messages.append(_tool_result_to_openai(msg))
    # system + tools 来自同名 trajectory
    system_prompt, tools = _load_system_and_tools(log_path)
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    return messages, tools


def _load_system_and_tools(log_path: AnyPath) -> tuple[str | None, list[dict]]:
    """从主会话同名的 ``*.trajectory.jsonl`` 取 systemPrompt + tools。

    取首个 ``context.compiled`` 事件（首轮编译的完整上下文最完整）；systemPrompt
    可能被平台截断（``truncated:true``）——接受截断版（无来源补全）。
    """
    log_path = Path(log_path)
    traj_path = log_path.with_suffix("").with_name(log_path.stem + ".trajectory.jsonl")
    if not traj_path.exists():
        return None, []
    with open(traj_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if evt.get("type") != "context.compiled":
                continue
            data = evt.get("data") or {}
            sp = data.get("systemPrompt")
            system_text: str | None = None
            if isinstance(sp, dict):
                system_text = sp.get("text") or sp.get("content")
            elif isinstance(sp, str):
                system_text = sp
            tools = data.get("tools") if isinstance(data.get("tools"), list) else []
            return (str(system_text) if system_text else None), tools
    return None, []


def _trajectory_record(record: dict, messages: list[dict], tools: list[dict]) -> dict:
    """组装一条进桶的轨迹记录（对齐 Buffer_冷启动数据需求 §2.2 文本字段）。"""
    return {
        "trajectory_id": f"{record['record_id']}-traj0",
        "record_id": record["record_id"],
        "session_id": record.get("session_id", ""),
        "bucket": record["bucket"],
        "sub_bucket": record.get("sub_bucket"),
        "messages": messages,
        "tools": tools,
        "meta": {
            "source": "sample105_v2",
            "policy": "pi0",  # OpenClaw 采集用的是采集期模型，近似 pi0 基座语义
            "cold_seed": True,
            "first_query_only": True,
            "classifier_model": record.get("model", ""),
            "classifier_rationale": record.get("rationale", ""),
        },
        # token/logprob/reward 需 π₀ 重跑，本批数据无，留空占位（见模块 docstring）。
        "response_token_ids": None,
        "logprobs": None,
        "original_logprobs": None,
        "reward": None,
    }


def route_trajectories(
    classified: Iterable[dict],
    out_root: AnyPath,
) -> dict:
    """按 bucket 把完整轨迹分组写入 ``out_root/<bucket>/<record_id>.jsonl``。

    跳过 ``bucket == "unknown"``（不入 buffer，B12 纪律）。每条一个文件，便于
    按桶增量追加。返回统计：``{per_bucket: {bucket: count}, unknown, skipped, total}``。
    """
    out_root = Path(out_root)
    stats: dict = {"per_bucket": {}, "unknown": 0, "skipped": 0, "total": 0}
    for rec in classified:
        stats["total"] += 1
        bucket = rec.get("bucket", "unknown")
        if bucket == "unknown" or bucket not in {
            "Workflow",
            "SysOps",
            "Finance",
            "Knowledge",
            "Communication",
            "OfficeQA",
            "Dialogue",
        }:
            stats["unknown"] += 1
            continue
        session_dir = rec.get("session_dir")
        if not session_dir:
            stats["skipped"] += 1
            continue
        messages, tools = rebuild_messages(session_dir)
        if not messages:
            stats["skipped"] += 1
            continue
        bucket_dir = out_root / bucket
        bucket_dir.mkdir(parents=True, exist_ok=True)
        traj = _trajectory_record(rec, messages, tools)
        out_path = bucket_dir / f"{rec['record_id']}.jsonl"
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(traj, ensure_ascii=False) + "\n")
        stats["per_bucket"][bucket] = stats["per_bucket"].get(bucket, 0) + 1
    return stats
