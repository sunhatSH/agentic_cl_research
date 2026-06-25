"""Step 1 — 抽取【初始 query】（待重写：1 沙箱 ↔ N 会话 ↔ N 首 query）。

⚠️ 本模块当前为留空状态——数据单元逻辑待重写。

**当前正确的数据单元关系**（见 ``doc/沙箱_实例_Queries对应关系_待定.md``）::

    1 个沙箱（workspace_init / Dockerfile）
       ├── 会话_1（首 query_1）
       ├── 会话_2（首 query_2）
       ├── ...
       └── 会话_N（首 query_N）
    每个会话的首 query → 8 实例（GRPO 8 路）→ 共 8N 实例

即：**一个沙箱对应 N 个会话、N 个首 query**。N 个首 query 互不依赖、都从同一
沙箱初始状态起跑。

**为什么留空**：sample105_v2 数据结构是错的（每个会话一个独立 workspace_init，
而非"N 会话共享 1 沙箱"），数据侧后续会改正。在正确数据结构给出前，"按沙箱聚合
N 个会话的首 query"逻辑无法写实——需要知道：

  1. 正确数据怎么标识"哪 N 个会话属于同一个沙箱"（沙箱 id / Dockerfile 标识）。
  2. 每个会话的"首 query"在数据里怎么取（仍是会话首个 user message？）。
  3. 1 沙箱 ↔ N 会话的清单 schema。

这些待开会定。本模块先留空，函数体抛 NotImplementedError，等数据结构定了重写。

旧的 1:1 假设（"每会话取 1 个首 query"）已删除——见 git 历史。

**保留的通用工具函数**（不依赖 1:1，可在重写后复用）：
  - ``iter_session_dirs`` / ``find_main_session_log`` / ``iter_message_events`` /
    ``extract_text``：OpenClaw 会话目录/日志解析的底层工具，与 query 聚合逻辑无关。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

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
    import json

    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(evt, dict) and evt.get("type") == "message":
                yield evt


def extract_text(content) -> str:
    """把 message.content（str | content-parts list | None）拍平为纯文本。"""
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


def first_user_query_text(log_path: AnyPath) -> str | None:
    """从一个会话日志取首个 ``role=user`` 的文本（底层工具，不含 1:1 假设）。

    返回 query 文本（无 session_id）；无 user message 时返回 None。重写后的
    "1 沙箱 ↔ N 会话"聚合可复用此函数逐个会话取首 query。
    """
    for evt in iter_message_events(log_path):
        msg = evt.get("message") or {}
        if msg.get("role") != "user":
            continue
        text = extract_text(msg.get("content"))
        if text.strip():
            return text
    return None


# --------------------------------------------------------------------------- #
# 待重写：1 沙箱 ↔ N 会话 ↔ N 首 query 聚合                                    #
# --------------------------------------------------------------------------- #


def extract_initial_queries(root: AnyPath, *, limit: int | None = None) -> list[dict]:
    """【待重写】按沙箱聚合 N 个会话的首 query（1 沙箱 ↔ N 首 query）。

    正确产出形态（待数据结构定稿）::

        [{"sandbox_id": "<沙箱/Dockerfile 标识>",
          "workspace_init": "<沙箱初始状态路径>",
          "sessions": [
              {"session_id": "...", "record_id": "...", "first_query": "...",
               "session_dir": "..."},
              ...
          ]},
         ...]

    依赖待定项：
      - 正确数据怎么标识"哪 N 个会话属于同一沙箱"（sample105 是错的：每会话一沙箱）。
      - 沙箱清单 schema。

    等开会定稿 + 数据侧改正结构后实现。当前抛 NotImplementedError。
    """
    raise NotImplementedError(
        "extract_initial_queries 待重写：1 沙箱 ↔ N 会话 ↔ N 首 query 聚合逻辑"
        "依赖正确数据结构（sample105 当前是错的）。见 "
        "doc/沙箱_实例_Queries对应关系_待定.md。"
    )
