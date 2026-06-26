"""OpenClaw 采集数据 → 7 桶 Replay Buffer 数据管道。

⚠️ **数据单元逻辑待重写**：原按"1 沙箱 ↔ 1 首 query"（1:1）处理，现要改为
"1 沙箱 ↔ N 会话 ↔ N 首 query"（1:n，N 个初始 query 互不依赖、都从同一沙箱
初始状态起跑，每个 query 8 实例）。sample105_v2 数据结构当前是错的（每会话一
独立 workspace_init，非"N 会话共享 1 沙箱"），数据侧后续改正。
见 ``doc/sandbox/沙箱_实例_Queries对应关系_待定.md``。

sample105_v2 是 OpenClaw 平台采集产物：事件流（``agent/sessions/*.jsonl`` 的
``type=message`` 事件，role ∈ {user, assistant, toolResult}）+ ``*.trajectory.jsonl``
遥测（``context.compiled`` 带 systemPrompt + tools）。

管道三步（``scripts/sample105_pipeline.py`` 串联，也可单步跑）：

1. **extract_initial_queries** — 【待重写，当前 NotImplementedError】按沙箱聚合
   N 个会话的首 query（1 沙箱 ↔ N 首 query）。底层工具（``iter_session_dirs`` /
   ``find_main_session_log`` / ``iter_message_events`` / ``extract_text`` /
   ``first_user_query_text``）已就绪、可复用。
2. **classify** — LLM 给首 query 分 7 桶 + 子桶（category，取自
   ``doc/BucketDesign.md``），JSON 容错解析（截断/非 JSON 降级 unknown）。
3. **route_trajectories** — 按桶把主会话完整轨迹重建为 OpenAI chat 入桶；
   **route_subagents** — 子会话事件流原样转 chat、独立成样本（主子各自单独训练）。

设计约束：
  - 纯函数（extract / route）不联网，可离线单测；classify 用可注入的
    ``ChatClient``（与 ``agents/base.py`` 同一 Protocol），mock 单测。
  - 复用既有约定：bucket 名取自 ``trainer/domain_tagging.DEFAULT_BUCKETS``，
    category 取自 ``BucketDesign.md``（见 ``SUB_BUCKETS``）。
  - 输出落 ``data/``（gitignored 运行时产物），不入库。
"""

