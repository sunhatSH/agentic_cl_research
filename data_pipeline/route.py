"""Step 3 — 把会话事件流按文件顺序转成 OpenAI chat，路由成训练轨迹。

主轨迹（``route_trajectories``）：按已标注的 bucket，把主会话 message 事件流
按文件顺序重建为 OpenAI chat messages（``user`` / ``assistant(tool_calls)`` /
``toolResult→tool``），system+tools 取自同名 ``*.trajectory.jsonl`` 的
``context.compiled``，整条写入 ``data/buckets/<bucket>/<record_id>.jsonl``。

子轨迹（``route_subagents``）：每个子会话事件流**原样**按文件顺序转成 chat、
独立成一条训练样本，写入 ``data/subagent_trajectories/<child_session_id>.jsonl``。
**主轨迹与子轨迹各自单独训练、互不并入**；格式=原数据，只转 chat、不增减字段、
不改内容、不重排顺序（文件顺序即因果顺序）。主轨迹何时拿子数据看 Agent +
原数据，我们不编排。DAG（``data_pipeline.dag``）仅用于定位存在的子日志。

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


def rebuild_messages_from_log(log_path: AnyPath) -> tuple[list[dict], list[dict]]:
    """从【指定日志文件路径】重建 OpenAI chat messages + tools（主/子会话通用）。

    与 ``rebuild_messages`` 的区别：后者按 session_dir 找主会话日志
    （``dyn-tmp-main-*``）；本函数直接吃一个日志路径，供子会话轨迹重建
    （子会话日志名是纯 UUID，不是 dyn-tmp-main）。
    """
    log_path = Path(log_path)
    if not log_path.exists():
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
    system_prompt, tools = _load_system_and_tools(log_path)
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    return messages, tools


def rebuild_messages(session_dir: AnyPath) -> tuple[list[dict], list[dict]]:
    """重建主会话完整轨迹为 OpenAI chat messages + tools。

    返回 ``(messages, tools)``。system message（若有 trajectory 的 systemPrompt）
    prepend 到 messages 头部；tools 来自 trajectory ``context.compiled``。
    无 systemPrompt 时 messages 不含 system（主日志本就无 system role）。
    """
    log_path = find_main_session_log(session_dir)
    if log_path is None:
        return [], []
    return rebuild_messages_from_log(log_path)


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
            "workflow",
            "ops",
            "qa",
            "finance",
            "office",
            "communication",
            "safety",
            "coding",
            "research",
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


# --------------------------------------------------------------------------- #
# subagent 轨迹：子会话事件流原样转 OpenAI chat，独立成样本                    #
# --------------------------------------------------------------------------- #
#
# 规定（与主轨迹一致）：
#   - 主轨迹与子轨迹**各自单独训练**，互不并入。
#   - **格式 = 原数据格式**，只把每个会话自己的事件流按文件顺序转成 OpenAI chat
#     消息列表（user / assistant(tool_calls) / tool），不增减字段、不改内容、
#     不重排顺序（文件里的顺序即因果顺序）。
#   - 主轨迹何时拿到子会话数据，完全看 Agent 行为 + 原数据怎么记（spawn/yield/
#     read 按原顺序保留），我们不替它编排、不假设出栈点。
#   - 子轨迹就是子会话自己的事件流，转成 chat、独立成一条样本。不加自造的 meta
#     字段、不清洗首条 user、不随父桶——字段全部来自原数据。
#
# DAG（``data_pipeline.dag``）的用途仅限：**找出哪些子会话日志存在、挂在哪个
# 主会话目录下**，从而知道该把哪些子会话转成子轨迹。不用于编排顺序、不打 flag。


def route_subagents(
    dag_nodes: list,
    out_root: AnyPath,
) -> dict:
    """把每个子会话事件流原样转成 OpenAI chat、独立写入训练样本。

    遍历 DAG 节点（主会话），对每个已挂接子会话（``child_session_id`` 非空，
    即子日志存在），用 ``rebuild_messages_from_log`` 把子会话事件流按文件顺序
    转成 chat，原样写出（字段与主轨迹 ``_trajectory_record`` 同构，仅
    ``record_id`` 取子会话 id、``bucket`` 留空待后续分桶）。

    子日志缺失（unlinked）的 spawn 跳过——属原数据缺失，非转换错误。
    返回统计：``{written, skipped, total}``。
    """
    out_root = Path(out_root)
    stats: dict = {"written": 0, "skipped": 0, "total": 0}
    for node in dag_nodes:
        for link in node.spawns:
            if not link.child_session_id or not link.child_session_dir:
                continue  # unlinked（子日志缺失），跳过
            stats["total"] += 1
            sub_log = Path(link.child_session_dir) / f"{link.child_session_id}.jsonl"
            messages, tools = rebuild_messages_from_log(sub_log)
            if not messages:
                stats["skipped"] += 1
                continue
            # 字段原样：子会话 id 作 record_id，bucket 留空（子轨迹分桶待后续）。
            traj = {
                "trajectory_id": link.child_session_id,
                "record_id": link.child_session_id,
                "session_id": link.child_session_id,
                "bucket": None,
                "sub_bucket": None,
                "messages": messages,
                "tools": tools,
                "response_token_ids": None,
                "logprobs": None,
                "original_logprobs": None,
                "reward": None,
            }
            out_path = out_root / f"{link.child_session_id}.jsonl"
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(traj, ensure_ascii=False) + "\n")
            stats["written"] += 1
    return stats
