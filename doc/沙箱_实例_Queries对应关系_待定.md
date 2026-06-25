# 沙箱 ↔ 实例 ↔ Queries 对应关系（待定）

> **状态**：结构未定，待开会讨论。本文件记录当前理解 + 标明代码里需改的 1:1 假设点。  
> **日期**：2026-06-25  
> **背景**：数据管道 `data_pipeline/` 原按"1 沙箱 ↔ 1 首 query"（1:1）处理，现要改为"1 沙箱 ↔ N 个初始 query"（1:n）。subagent 处理方式不动。

---

## 1. 目标关系（待开会确认）

```
1 个沙箱（= 1 个 workspace_init 状态 = 1 个 Dockerfile）
   ├── 初始 query_1  →  8 个实例（GRPO 8 路并行，rollout.n=8）
   ├── 初始 query_2  →  8 个实例
   ├── ...
   └── 初始 query_N  →  8 个实例
   = 共 8N 个实例，全部从同一沙箱初始状态起跑
```

**要点**：
- N 个都是**初始 query**——互不依赖，都从同一个 `workspace_init`（沙箱初始状态 / Dockerfile）起跑。
- 不是多轮里的后续轮次（q2 不接 q1 的 winner，而是与 q1 同起点）。
- 每个初始 query 跑 8 个实例（GRPO 组内 8 路，`base.yaml: rollout.n=8`）。
- 总实例数 = 8N。

**N 个 query 的并发度**（已定）：
- 单个 query 内：8 个实例**并行**（GRPO 组）。
- N 个 query 之间：**不全部并行**——选 **K 个并行**（K 可被 N 整除），K 组同时跑（峰值 8K 个实例），跑完一批再跑下一批。
- **K 数值待定**（开会商量）。
- 例：N=8、K=2 → 每批 2 个 query（16 实例）并行，共 4 批，总实例化 64 个（=8N）。

### 1.1 实例生命周期与会话结束条件

**实例生命周期**：每个初始 query 的 8 个实例，从沙箱初始状态 fork 出来 → 跑该 query → **跑完即销毁**（不跨 query 存活，独立 query 无"下一轮依赖上一轮 winner"，故无 winner-sync）。下一个 query 从沙箱初始状态**重新 fork** 8 个实例。

**winner**（已定）：保留。独立模型下仍选 winner = **优势（advantage）最大的那条轨迹**（等价于 reward 最高，与现有 `select_winner` 一致）。winner 用于固化/记录（如旧 `sandbox_smoke.py` 的"winner 固化"概念），但**不用于跨 query 状态同步**（独立 query 间无依赖，winner 不传给下一 query）。即 winner 只在"单 query 内"有意义——选出来固化、收轨迹算 advantage，跑完该 query 即弃。

> 这与现有 `rollout/session_pool.py` 的"多轮依赖 + winner 跨轮存活"模型不同——独立模型是 per-query 生命周期（winner 不跨 query），现有代码是 per-session 生命周期（winner 贯穿所有 query、sync 给下一轮）。详见 §2.2。

**会话结束条件（当前确定 2 种，其余待定）**：

| # | 结束条件 | 说明 | 状态 |
|---|---------|------|------|
| 1 | **Questioner 说结束** | Questioner 主动输出 `<end_session>`（用户满意语义），即结束当前会话。对应现有 `agents/questioner.py::END_SESSION` 机制。 | ✅ 确定 |
| 2 | **Questioner 提问达到上限次数** | 一个会话内 Questioner 的提问轮数达到上限（**数值待定**）即结束，防无限追问。 | ✅ 确定（数值待定） |
| 3 | （其余条件待定） | 如 scorer_error（打分异常选不出 winner）、沙箱硬失败等——是否纳入、如何处理，待开会定。 | ⏳ 待定 |

> 现有 `rollout/simulated_session.py` 还有 `patience`（winner 连续失败耐心耗尽）、`k_budget`（轮数预算）两种结束——这俩是多轮依赖模型的产物。独立模型下是否保留/如何调整，归入"其余条件待定"。

**未决问题（开会讨论）**：
- [ ] N 个初始 query 的**来源/结构**：是从 sample105 会话数据取，还是另配一组独立 query 清单？sample105 单会话只有 1 个首 user（后续轮次依赖前面，不是"初始"），给不出"N 个互不依赖的初始 query"——需定数据单元怎么定义。
- [ ] **K 数值**（N 个 query 间的并行批大小，K 可被 N 整除）。
- [ ] **Questioner 提问上限次数**的数值。
- [ ] "1 个沙箱 = 1 个 Dockerfile" 与 sample105 的 `workspace_init/`（文件系统初始状态）如何对应：一个 workspace_init 是否就产出一个 Dockerfile？
- [ ] 与现有 `rollout/session_pool.py` 的"多轮依赖(q1→winner→q2)"模型如何区分/共存。

---

## 2. 当前代码的 1:1 假设点（需改成 1:n）

### 2.1 数据管道 `data_pipeline/`（1:1，待改）

| 文件:行 | 当前假设 | 改动方向 |
|---------|---------|---------|
| `extract.py:114 first_user_query()` | 遇首个 `role=user` 即 return，**只取 1 个** | 改成取 N 个初始 query（N 来源待定） |
| `extract.py:133 extract_first_queries()` | 每目录产 1 条 `{record_id, first_query}` | 改成 1 沙箱 ↔ N query |
| `extract.py:160` | `"first_query": query` | 改成 `queries: [...]` |
| `classify.py:185` | 对每条 `first_query` 分桶 | 改成对每个 query 分桶 |
| `route.py:178` | meta `"first_query_only": True` | 撤掉该标记 |
| `route.py: route_trajectories` | 每条 1 主轨迹 | 改成 1 沙箱 ↔ N 轨迹（轨迹切分待定） |
| `scripts/sample105_pipeline.py` | extract/classify/route 全链路 1:1 | 同步改 |

### 2.2 运行时 `rollout/`（已是 1:n，但是"多轮依赖"模型，与目标不同）

| 文件 | 当前模型 | 与目标的差异 |
|------|---------|-------------|
| `rollout/session_pool.py` | 1 会话 → `queries:[q1..qn]`，8 槽，**q1→winner→q2→winner→…**（多轮依赖，q_k 从 q_{k-1} winner 态起） | 目标是 N 个 query **都从同一 workspace_init 起、互不依赖**——不同模型 |
| `rollout/scheduler.py` | 1 step = 16 会话并行 × 8 槽 = 128 实例 | 实例数算法一致（sessions×slots），但会话内 query 关系不同 |
| `configs/base.yaml:29` | `rollout.n: 8`（traj/query，GRPO 组 8 路） | 一致，8 实例/query 不变 |

**关键**：`rollout/session_pool.py` 的 winner-sync 是为"多轮依赖"设计的（q2 接 q1 的环境+对话正史）。目标"N 个初始 query 共享同一初始沙箱、互不依赖"**不需要 winner-sync**——N 个 query 各自独立 8 实例，跑完即弃。开会要定：是改 session_pool 支持"独立初始 query 模式"，还是新写一层。

### 2.3 沙箱镜像 `docker/sandbox/` + `workspace_init`

- `docker/sandbox/fs-seeds/` + `bin/seed_workspace.sh`：把 workspace 物化成 Dockerfile/镜像种子。
- 数据管道里 `workspace_init` 字段当前只记录路径、未实际使用（route 没读 workspace）。
- 目标"1 沙箱 = 1 Dockerfile ↔ N query"需明确：workspace_init 怎么变成可实例化的 Dockerfile、N 个 query 怎么挂到它下面。

---

## 3. 已确认不动

- **subagent 处理方式**（`data_pipeline/dag.py` + `route.py::route_subagents`）：主子各自单独训练、格式原样只转 chat、DAG 仅定位子日志。不在本次改动范围。
- `rollout.n=8`（GRPO 组 8 路）：不变。
- `prepare_queries.py` / `convert_dataset.py`：这两个处理的是旧 `_stage_prefix_pass.jsonl` 格式（已删的数据），与 sample105 无关，暂不动。

---

## 4. 待开会定后落地

1. N 个初始 query 的来源与数据单元定义。
2. 1 沙箱 ↔ N query 的数据格式（会话清单 schema）。
3. `data_pipeline/` 1:1 → 1:n 改造（extract/classify/route）。
4. `rollout/` 是否需要"独立初始 query 模式"（区别于多轮 winner-sync）。
5. workspace_init → Dockerfile 的物化路径。

> 本文件为"数据理解"留档；代码改动等开会结论后再做。
