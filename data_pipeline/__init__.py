"""OpenClaw 采集数据 → 7 桶 Replay Buffer 数据管道。

sample105_v2 是 OpenClaw 平台采集产物（见 ``doc/sample105_v2`` 实测结构），
与旧的 ``datasets/_stage_prefix_pass.jsonl``（``{record_id, record:{messages,...}}``
一行一会话）格式不同：OpenClaw 用**事件流**（``agent/sessions/*.jsonl`` 里的
``type=message`` 事件，role ∈ {user, assistant, toolResult}）记录会话，外加
``*.trajectory.jsonl`` 遥测（``context.compiled`` 带 systemPrompt + tools）。

三步管道（``scripts/sample105_pipeline.py`` 串联，也可单步跑）：

1. **extract_first_queries** — 遍历 105 个主会话目录，从主会话日志
   (``dyn-tmp-main-*.jsonl``，排除 trajectory / subagent) 的 message 事件流
   取**第一个** ``role=user`` 的 text，输出
   ``{session_id, record_id, first_query, session_dir, workspace_init}``。
   subagent 子会话（纯 UUID）不取——主会话里 ``sessions_spawn`` 的 toolCall
   已记录派生关系。

2. **classify_bucket** — 把首 query 交给 LLM，判断属于 7 桶哪一桶
   **+ 精确到子桶（category）**。子桶取自 ``doc/BucketDesign.md`` 各桶的
   category 清单（如 SysOps 的 ops/terminal/security、Finance 的
   finance/compliance/procurement）。输出
   ``{record_id, bucket, sub_bucket, rationale, model}``，JSON 容错解析
   （截断/非 JSON 降级为 unknown）。

3. **route_trajectories** — 按已标注的 ``{record_id, bucket}`` 取对应主会话
   **完整轨迹**（重建 OpenAI chat messages，``toolResult→tool``、
   ``toolCall`` part → ``assistant.tool_calls``），按 bucket 分组写入
   ``data/buckets/<bucket>/*.jsonl``。subagent 不处理（本次范围外）。

设计约束：
  - 纯函数（extract / route）不联网，可离线单测；classify 用可注入的
    ``ChatClient``（与 ``agents/base.py`` 同一 Protocol），mock 单测。
  - 复用既有约定：bucket 名取自 ``trainer/domain_tagging.DEFAULT_BUCKETS``，
    category 取自 ``BucketDesign.md``（见 ``SUB_BUCKETS``）。
  - 输出落 ``data/``（gitignored 运行时产物），不入库。
"""
