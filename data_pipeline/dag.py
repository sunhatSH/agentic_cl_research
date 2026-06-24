"""Step 4 — OpenClaw subagent DAG 解析：把子会话挂回主会话 + 检测失败。

OpenClaw 的主/子会话关系**不在主日志里显式串**，靠两个字段 join（实测可靠）：

  - 主会话 ``sessions_spawn`` 的 toolResult 含 ``childSessionKey``，
    形如 ``agent:<父sessionId>:subagent:<childUUID>``
  - 子会话 ``*.trajectory.jsonl`` 首行的 ``sessionKey`` = 同一个
    ``agent:<父sessionId>:subagent:<childUUID>``

注意：子会话**日志文件名 id ≠ childUUID**（是另一个 UUID），但子会话
trajectory 的 ``sessionKey`` 含 childUUID，用 sessionKey 的
``<父sessionId>:subagent:<childUUID>`` 段做 join。

本模块纯函数 + 流式读，不联网、可离线单测。产出 ``SessionNode`` 列表
（主会话 + 各子会话 + 父子边 + 失败标志），供 ``route_subagents`` 把子轨迹
独立成样本、标 ``had_subagent_timeout`` flag（见
``doc/Hermes_Subagent_训练数据方案.md`` §6）。

OpenClaw 失败语义（实测，000037）：子会话 ``session.ended.status == "error"``、
``timedOut/idleTimedOut == true``（如 ``LLM idle timeout``）。主会话的
``sessions_yield`` **不反映子失败**（永远返回 ``yielded``），故失败检测必须
看子会话自己的 trajectory，不能看主会话。
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from data_pipeline.extract import find_main_session_log, iter_message_events, iter_session_dirs

AnyPath = "str | Path"

# sessions_spawn toolResult 里 childSessionKey 的正则。
# 形如 agent:<父sessionId>:subagent:<childUUID>
_CHILDKEY_RE = re.compile(r'"childSessionKey":\s*"([^"]+)"')
# sessionKey 拆出 <父sessionId>:subagent:<childUUID> 的 subagent UUID 段
_SUBAGENT_UUID_RE = re.compile(r":subagent:([0-9a-fA-F-]+)")

_SPAWN_TOOL = "sessions_spawn"


@dataclass
class SpawnLink:
    """一次 sessions_spawn 调用与它派生子会话的链接。

    - ``spawn_toolcall_id``：主轨迹里 ``sessions_spawn`` toolCall 的 id
      （线性化时定位插入点；OpenClaw 主轨迹如实保留，不内联子消息）。
    - ``child_session_id``：子会话日志文件名 id（≠ childUUID）。
    - ``child_uuid``：childSessionKey 里的 subagent UUID（join 用）。
    - ``label`` / ``task``：spawn 时传的 label 与任务文本（子轨迹分桶/审计用）。
    - ``child_session_dir``：子会话所在目录（读子日志用）。
    """

    spawn_toolcall_id: str
    parent_session_id: str
    child_uuid: str
    child_session_id: str | None = None
    child_session_dir: str | None = None
    label: str | None = None
    task: str | None = None
    child_status: str | None = None  # success / error / cleanup / None
    child_timed_out: bool = False
    child_prompt_error: str | None = None


@dataclass
class SessionNode:
    """一个会话（主或子）的 DAG 节点。"""

    session_id: str  # 日志文件名 id（主=dyn-tmp-main-..., 子=UUID）
    session_dir: str
    is_main: bool
    parent_session_id: str | None = None  # 子会话的父（主会话无）
    spawns: list[SpawnLink] = field(default_factory=list)  # 主会话派生的子链接


def _iter_session_logs(session_dir: AnyPath) -> Iterator[Path]:
    """一个会话目录下所有非 trajectory 的 .jsonl（主 + 子）。"""
    sessions_dir = Path(session_dir) / "agent" / "sessions"
    if not sessions_dir.is_dir():
        return
    for p in sorted(sessions_dir.glob("*.jsonl")):
        if "trajectory" in p.name:
            continue
        yield p


def _first_json_object(path: Path) -> dict | None:
    """读 jsonl 首行 JSON（容错）。"""
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    return json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
    except OSError:
        return None
    return None


def _trajectory_path(log_path: Path) -> Path:
    """主/子日志对应的 trajectory.jsonl 路径（同名加 .trajectory）。"""
    return log_path.with_name(log_path.stem + ".trajectory.jsonl")


def _session_key(log_path: Path) -> str | None:
    """从 trajectory.jsonl 首行取 sessionKey（子会话挂回主会话的 join 键）。"""
    traj = _trajectory_path(log_path)
    if not traj.exists():
        return None
    obj = _first_json_object(traj)
    if isinstance(obj, dict):
        return obj.get("sessionKey") or obj.get("session_key")
    return None


def _ended_status(log_path: Path) -> dict:
    """从 trajectory.jsonl 取最后一个 session.ended 事件的 data。

    返回 ``{status, timed_out, idle_timed_out, prompt_error}``（缺省值兜底）。
    子会话失败检测靠这个（主会话 yield 不反映子失败）。
    """
    traj = _trajectory_path(log_path)
    ended: dict | None = None
    if traj.exists():
        with open(traj, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if evt.get("type") == "session.ended":
                    ended = evt.get("data") or {}
    if not isinstance(ended, dict):
        ended = {}
    return {
        "status": ended.get("status"),
        "timed_out": bool(ended.get("timedOut") or ended.get("idleTimedOut")),
        "prompt_error": ended.get("promptError"),
    }


def _extract_spawn_links(log_path: Path, parent_session_id: str) -> list[SpawnLink]:
    """从主会话日志抽 sessions_spawn 调用 + 对应 result 的 childSessionKey。

    配对：assistant 的 ``sessions_spawn`` toolCall（id + label + task）↔ 紧随其
    后的 ``toolResult``（toolCallId 匹配）里的 childSessionKey。child_uuid 从
    childSessionKey 的 ``:subagent:<UUID>`` 段取。
    """
    pending: dict[str, SpawnLink] = {}  # toolCallId -> SpawnLink（待配 result）
    links: list[SpawnLink] = []
    for evt in iter_message_events(log_path):
        msg = evt.get("message") or {}
        role = msg.get("role")
        if role == "assistant":
            for p in msg.get("content") or []:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "toolCall" and p.get("name") == _SPAWN_TOOL:
                    args = p.get("arguments") or {}
                    link = SpawnLink(
                        spawn_toolcall_id=p.get("id", ""),
                        parent_session_id=parent_session_id,
                        child_uuid="",  # 待 result 填
                        label=args.get("label") or args.get("taskName"),
                        task=args.get("task"),
                    )
                    pending[p.get("id", "")] = link
        elif role == "toolResult":
            tcid = msg.get("toolCallId", "")
            if tcid not in pending:
                continue
            txt = ""
            for p in msg.get("content") or []:
                if isinstance(p, dict) and isinstance(p.get("text"), str):
                    txt += p["text"]
            m = _CHILDKEY_RE.search(txt)
            child_key = m.group(1) if m else ""
            m2 = _SUBAGENT_UUID_RE.search(child_key)
            child_uuid = m2.group(1) if m2 else ""
            link = pending.pop(tcid)
            link.child_uuid = child_uuid
            links.append(link)
    return links


def _find_subagent_log_by_uuid(session_dir: AnyPath, child_uuid: str) -> Path | None:
    """在会话目录下，找 sessionKey 含该 child_uuid 的子会话日志。

    子会话文件名 id ≠ child_uuid，必须读每个子 trajectory 的 sessionKey 匹配。
    只在本会话目录搜（子会话日志通常与父同目录）；跨目录的孤儿/缺失子日志
    返回 None（由 dag_summary 计入 unlinked，属数据缺失非 join 错误）。
    """
    if not child_uuid:
        return None
    sessions_dir = Path(session_dir) / "agent" / "sessions"
    if not sessions_dir.is_dir():
        return None
    for p in sorted(sessions_dir.glob("*.jsonl")):
        if "trajectory" in p.name or p.name.startswith("dyn-tmp-main"):
            continue  # 主会话 / trajectory 跳过
        skey = _session_key(p) or ""
        if child_uuid in skey:
            return p
    return None


def build_session_dag(root: AnyPath) -> list[SessionNode]:
    """遍历采集根目录，建主/子会话 DAG。

    每个会话目录产出一个主会话 ``SessionNode``（is_main=True），其 ``spawns``
    列出该主会话派生的所有子链接（含子会话 id/dir/失败标志）。子会话不单独
    成 node（作为主会话 spawn 的 child 信息携带），因为子轨迹独立训练时由
    ``route_subagents`` 直接读子日志重建——DAG 只需提供挂接关系 + 失败标志。
    """
    nodes: list[SessionNode] = []
    for session_dir in iter_session_dirs(root):
        main_log = find_main_session_log(session_dir)
        if main_log is None:
            continue
        main_id = main_log.stem
        node = SessionNode(
            session_id=main_id,
            session_dir=str(Path(session_dir).resolve()),
            is_main=True,
        )
        links = _extract_spawn_links(main_log, main_id)
        for link in links:
            sub_log = _find_subagent_log_by_uuid(session_dir, link.child_uuid)
            if sub_log is not None:
                link.child_session_id = sub_log.stem
                link.child_session_dir = str(sub_log.parent.resolve())
                ended = _ended_status(sub_log)
                link.child_status = ended["status"]
                link.child_timed_out = ended["timed_out"]
                link.child_prompt_error = ended["prompt_error"]
            node.spawns.append(link)
        nodes.append(node)
    return nodes


def dag_summary(nodes: list[SessionNode]) -> dict:
    """DAG 统计：主会话数 / spawn 总数 / 已挂接子会话 / 失败子会话。"""
    total_spawns = sum(len(n.spawns) for n in nodes)
    linked = sum(1 for n in nodes for s in n.spawns if s.child_session_id)
    failed = sum(
        1
        for n in nodes
        for s in n.spawns
        if s.child_session_id and (s.child_status == "error" or s.child_timed_out)
    )
    return {
        "main_sessions": len(nodes),
        "total_spawns": total_spawns,
        "linked_subagents": linked,
        "unlinked_spawns": total_spawns - linked,
        "failed_subagents": failed,
    }
