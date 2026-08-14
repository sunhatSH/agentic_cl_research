# Hermes 子 Agent 训练数据准备方案

> **状态**：方案已定，待落地（OpenClaw 侧另行处理，见末尾）。  
> **日期**：2026-06-24  
> **背景**：Continual Learning over Agentic LLM，主 Agent 调用子 Agent（subagent）完成多步任务。本文记录 **Hermes 框架**下子 Agent 轨迹如何进训练数据。

---

## 0. 核心约定

**Hermes 调用子 Agent 时，主 Agent 等待所有子 Agent 完成再继续（同步阻塞）。**

这条约定是出入栈方案的基石：因为主 Agent 同步等待，所以子 Agent 的执行段在因果上是**主轨迹中一段连续的、不被主消息打断的区间**——可以干净地用入栈/出栈建模，不需要处理真实并发交织。

> Transformer 学的是 causal trace（因果顺序），不是 wall-clock trace（真实并发顺序）。保存因果 DAG，训练时按 DFS replay。

---

## 1. 数据保存：因果 DAG，不保存并发顺序

Runtime 记录每个 Agent / Subagent 节点：

| 字段 | 含义 |
|------|------|
| `node_id` | 节点唯一 id（= 该 agent 的 session id） |
| `parent_id` | 父 agent 的 node_id（spawn 它的那个主 agent） |
| `agent_type` | main / subagent（及角色，如 ResearchAgent / CodingAgent） |
| `input` | 父 agent spawn 时传入的 task 文本 |
| `output` | 子 agent 最终输出（末条 assistant 文本） |
| `children` | 该节点 spawn 的子节点 id 列表 |
| `spawn_toolcall_id` | 父轨迹中发起本次 spawn 的 toolCall id（用于线性化时定位插入点） |

形成 DAG：

```
Main
├── ResearchAgent
│     step1
│     step2
├── CodingAgent
│     step1
│     step2
└── Main continue
```

**只记父子因果关系，不记真实时间交织。**

---

## 2. 训练时线性化：DFS 出入栈

按 DFS 展开（入栈 = 进入子 agent，出栈 = 子 agent 返回）：

```
Main think

<call ResearchAgent>          ← 入栈
research step1
research step2
<return>                       ← 出栈

Main continue

<call CodingAgent>             ← 入栈
coding step1
coding step2
<return>                       ← 出栈

Main finish
```

**而不是**按真实时间戳交织：

```
Main
research step1
Main
coding step1
research step2     ← 真实并发，不保存
...
```

### 2.1 入栈/出栈的表达方式

**不引入特殊 token**（如 `<call>`/`<return>`）。入栈出栈由 OpenAI 的 **tool_call / tool_result 机制本身**体现，subagent 当 toolcall 处理：

- **入栈** = 主轨迹中 `sessions_spawn`（或等价 spawn 工具）的 toolCall
- **出栈** = 该 spawn 对应的 toolResult，**携带子 agent 的最终输出**（Hermes 同步等待，故 result 含子结果）

> 与 OpenClaw 的关键区别：Hermes 同步等待 → spawn 的 toolResult **直接含子 agent 输出**，出栈语义干净。OpenClaw 的 yield 不阻塞、result 不含子结果（见末尾），故 Hermes 走本方案，OpenClaw 另议。

---

## 3. 两条独立训练轨迹

### 3.1 主轨迹（main trajectory）

主 agent 的完整 messages，其中：
- `sessions_spawn` 的 toolCall 保留（=入栈标记）
- 其 toolResult = 子 agent 的最终输出文本（=出栈，Hermes 同步等待保证）
- **不内联展开子 agent 的内部消息**（子 agent 单独训练，见 3.2）
- 主视角看就是一次 tool 调用：调了子 agent、拿到结果、继续

### 3.2 子轨迹（subagent trajectory）—— 单独训练

每个 subagent 的完整 messages 独立成一条训练样本：
- system：子 agent 自己的角色 prompt
- user：父 spawn 传入的 task
- assistant/tool：子 agent 内部的工具调用序列
- 末条 assistant：子 agent 的最终输出（=回填到主轨迹 spawn toolResult 的那段）

子轨迹按它自己的首 query / task 语义分桶（9 桶），独立进 Replay Buffer。**子 agent 的子 agent 递归同理**（DFS）。

### 3.3 为何单独训练

- 子 agent 是独立能力单元（ResearchAgent 的检索能力 ≠ CodingAgent 的编码能力），单独训练让各自能力被 Replay Buffer 覆盖、不被主轨迹稀释。
- 主轨迹学的是"编排 + 何时派子任务 + 如何合并结果"；子轨迹学的是"被派任务后怎么完成"。两者梯度分离，避免主轨迹过长导致子任务段被 U 形块权重压低。

---

## 4. 整理流程（Hermes）

```
原始采集（主会话日志 + 各子会话日志 + DAG 父子关系）
  │
  ├─ Step A. 解析 + 建 DAG
  │    ├─ 主会话 → OpenAI chat（toolResult→tool, toolCall→tool_calls, 丢 thinking）
  │    ├─ 子会话 → 同法重建
  │    └─ DAG 边：parent_id / spawn_toolcall_id / child node_id（spawn result 的 childSessionKey ↔ 子会话 sessionKey join）
  │
  ├─ Step B. 主轨迹线性化
  │    ├─ sessions_spawn toolCall 保留
  │    ├─ 其 toolResult 填子 agent 最终输出（同步等待，结果可得）
  │    └─ 不内联子消息
  │
  └─ Step C. 两条独立轨迹入桶
       ├─ 主轨迹 → 按 bucket 入桶（首 query 分桶）
       └─ 子轨迹 → 各自独立成样本，按自身 task 分桶
```

---

## 5. 待落地项

- [ ] DAG 解析模块（`parent_id` / `spawn_toolcall_id` / `child node_id` join）
- [ ] 主轨迹线性化（spawn toolResult 回填子输出）
- [ ] 子轨迹独立成样本 + 分桶
- [ ] 子 agent 失败/超时轨迹的处理策略（丢弃 / 标低分 flag，待定）
- [ ] 递归子 agent（子任务的子任务）的 DFS 展开

---

## 6. OpenClaw 侧方案（与 Hermes 不同步，独立处理）

OpenClaw 采集数据与 Hermes 的同步假设**不符**，出入栈方案不能直接套用。已定方案如下。

### 6.1 OpenClaw 实测性质（与 Hermes 的区别）

| | Hermes（§1-5） | OpenClaw（实测） |
|---|---|---|
| 主 agent 等待子 agent | **同步阻塞** | **异步**，yield 让出本轮 |
| spawn 的 toolResult | **含子输出**（出栈干净） | `status:accepted`，不含子输出 |
| 阻塞等待 toolcall | spawn 本身阻塞 | `sessions_yield`，但 result 仅 `yielded`、不等子完成、不报子失败 |
| 子结果回传 | tool result 直接 | **无结构化通道**，靠文件系统落盘 + 下一轮 read |
| 子会话失败 | 主会话能感知 | **主会话不感知**，yield 后 success 结束，靠真人下一轮发现文件缺失纠错 |

实测证据（000037）：子会话 `LLM idle timeout (180s)` 崩溃 → `sessions_yield` 照常返回 `yielded`（`isError=False`）→ 主会话 `session.ended=success` → 下一轮真人 user 消息直接点破"子任务产出没落盘"→ 主会话自己 read/exec 检查 + 补做（不再 spawn）。**主会话 yield 后 NO_REPLY 结束，子失败完全靠真人纠错。**

### 6.2 处理方案：主/子轨迹各自单独训练，格式原样只转 chat

**最简规定**（已落地）：

1. **主轨迹与子轨迹各自单独训练**，两条独立训练样本，互不并入。
2. **格式 = 原数据格式**：把每个会话自己的事件流**按文件顺序**转成 OpenAI chat 消息列表（`user` / `assistant(tool_calls)` / `toolResult→tool`），**不增减字段、不改内容、不重排顺序**（文件里的顺序即因果顺序）。
3. **主轨迹何时拿到子会话数据，完全看 Agent 行为 + 原数据怎么记**——spawn/yield/read 按原顺序如实保留，我们不替它编排、不假设出栈点、不把子消息并入主轨迹。
4. **子轨迹**就是子会话自己的事件流，转成 chat、独立成一条样本。不加自造 meta、不清洗首条 user、不随父桶——字段全部来自原数据。
5. **DAG（`data_pipeline.dag`）用途仅限**：找出哪些子会话日志存在、挂在哪个主会话目录下，从而知道该把哪些子会话转成子轨迹。**不用于编排顺序、不打 flag**。

落地点（`data_pipeline/route.py`）：
- `rebuild_messages_from_log(log_path)`：主/子通用，把任一会话日志事件流按文件顺序转成 OpenAI chat + tools。
- `route_trajectories(classified, out_root)`：主轨迹按首 query 分桶入 `data/buckets/<bucket>/`。
- `route_subagents(dag_nodes, out_root)`：子会话事件流原样转 chat、独立写入 `data/subagent_trajectories/`（`record_id`=子会话 id，`bucket` 留空待后续分桶）。
- CLI：`scripts/sample105_pipeline.py route-subagents --root <OpenClaw根> --out-dir <输出>`。

### 6.3 OpenClaw 失败语义（仅记录，不用于编排）

实测：子会话 `session.ended.status==error` / `timedOut` 时，主会话 `sessions_yield` 照常返回 `yielded`（不报子失败）、主会话 `session.ended=success`，靠真人下一轮发现文件缺失纠错。**离线层如实保留这些 toolcall、不做特殊处理**——主轨迹里 yield/read 该在哪在哪；子轨迹就是子会话事件流原样。失败检测/纠错是**训练时**三 Agent（observer 发现 + Questioner 纠错）的职责，不是离线数据层的事。

### 6.4 训练时层（待落地，非本次范围）

- Observer diff 取证：发现"子任务声称产文件但未落盘"写入 `ObservationReport.discrepancies`（字段已有，需确认覆盖 spawn 场景）。
- Questioner 纠错追问分支：读 `discrepancies` 非空时切换（接口不变，复用现有读报告机制）。
- 主 agent 补救方式（补做 vs 重 spawn）由训练数据 + 模型自选，不硬编码。
- 离线开关：是否启用"子任务失败 + 纠错"多轮（默认保留）。

### 6.5 关键设计点

1. **yield 不赋予出栈语义**：OpenClaw 主轨迹 `sessions_yield` 的 result 永远是 `yielded`、不含子结果，"子任务结束"在数据里没有干净对应物。主轨迹如实记录，不强行线性化为出入栈。
2. **子结果经文件回传**：主会话拿子结果是后续 `read` 子会话写的文件——read 文件名 ↔ childSessionKey 无直接对应，离线层不强求配对（主子各自独立训练，主轨迹不需要精确知道哪次 read 对应哪个子会话）。
3. **DAG 父子 join**：spawn result 的 `childSessionKey` ↔ 子会话 trajectory `sessionKey` join，得 `parent_session_id / spawn_toolcall_id / child_session_id`。注意子会话**文件名 id ≠ childSessionKey 里的 subagent UUID**，需用 sessionKey join（已实测可靠）。该 join 仅用于定位存在的子日志，不进训练数据字段。
4. **Questioner 纠错分支不写死补救方式**（训练时层）：主 agent 补做 vs 重 spawn 由训练数据分布 + 模型自主决定。

### 6.6 数据现状说明

> sample105_v2（OpenClaw 采集）已**弃用**（数据结构不符——每会话一独立 workspace_init，非"1 沙箱↔N 会话"）。本节 subagent 处理方案（主子各自单独训练、格式原样、DAG 仅定位子日志）**设计有效**，待新数据结构（`data/taskspecs/` taskspec）产出含子 agent 的轨迹后再落地。taskspec 当前只含单 task 声明 + 初始 `files/`，不含已采集轨迹——主/子轨迹均由 rollout 现场产（见 `doc/ops/sandbox/沙箱_Dockerfile制作方案.md`）。

### 6.6.1 `workspace_init/` 与 `workspace_final/` 的用途边界

每个会话目录还含两份沙箱快照（当前数据管道**两者都没读**——只读 `agent/sessions/*.jsonl` 事件流转 chat）：

| 用途 | 需要 `workspace_init/` | 需要 `workspace_final/` |
|------|------------------------|-------------------------|
| **训练轨迹**（messages + tools，主/子各一条） | ❌ 不需要 | ❌ 不需要 |
| **冷启动沙箱镜像种子**（rollout 复现采集期初始环境） | ✅ 需要 | ❌ 不需要 |
| **Ground-truth 终态对照**（observer diff / reward 判分参照） | ❌ | ⚠️ 可选（当前 reward 走 LLM judge 不靠它） |

**结论**：
- 只做训练数据 → `init`/`final` **都不需要**。
- 做冷启动沙箱镜像（`Buffer_冷启动数据需求.md` 里的"每会话首条 query + 对应沙箱镜像"）→ **只要 `workspace_init/`**，不要 `final`。镜像机制见 `docker/sandbox/fs-seeds/` + `bin/seed_workspace.sh`。
- `workspace_final/` 训练侧用不上（agent 训练时自己跑产生终态）；仅当要做"终态对照评测"时才考虑。

### 6.7 OpenClaw 待落地项（训练时层，非本次范围）

- [ ] observer 取证层：子任务"声称产文件但未落盘"写入 `discrepancies`（部分已有，需确认覆盖 spawn 场景）
- [ ] Questioner 纠错追问分支（读 `discrepancies` 非空时切换）
- [ ] 离线开关：是否启用"子任务失败 + 纠错"多轮（默认保留）
- [ ] 子轨迹按自身 task 二次分桶（当前留空，可后续用 classify 对子会话首 task 打标）
