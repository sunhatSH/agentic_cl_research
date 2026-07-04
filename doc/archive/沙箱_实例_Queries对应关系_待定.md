# 沙箱 ↔ 实例 ↔ Queries 对应关系

> **状态**：核心关系已定，部分数值/细节待开会确认（见 §3 待定项）。  
> **日期**：2026-06-25  
> **背景**：数据管道 `data_pipeline/` 原按"1 沙箱 ↔ 1 首 query"（1:1）处理，现改为"1 沙箱 ↔ N 个初始 query"（1:n）。subagent 处理方式不动。

---

## 1. 概念对应关系（树形）

同一行 = 同概念/等价/一一对应；缩进 = 从属/实例化关系。

```
workspace_init          沙箱初始文件系统状态（一个目录快照）
├─ Dockerfile           workspace_init 物化成的镜像定义  ← 与 workspace_init 一一对应
├─ 沙箱 (sandbox)       = workspace_init = Dockerfile（三者等价，指同一个"初始状态"）
│
├─ 会话 (session)        一个沙箱下挂 N 个会话（每个会话有 1 个首 query）
│  ├─ 会话_1
│  │  └─ 首 query_1     初始 query（互不依赖，都从同一沙箱起跑）
│  ├─ 会话_2
│  │  └─ 首 query_2
│  ├─ ...
│  └─ 会话_N
│     └─ 首 query_N
│
└─ 容器 (instance)       沙箱的运行实例（fork 自 workspace_init，可写拷贝）
   └─ 每个首 query → 8 个容器（GRPO 8 路并行，rollout.n=8）
      ├─ query_1 → 8 容器
      ├─ query_2 → 8 容器
      ├─ ...
      └─ query_N → 8 容器
      总实例化 = 8N（并发峰值 8K，K 见下）
```

**关键对应**：
- **workspace_init ↔ Dockerfile ↔ 沙箱**：三者一一对应/等价，指同一个"初始状态"。N 个 query 共享这一个。
- **沙箱 → 容器**：1 个沙箱 fork 出 8N 个容器（每个 query 8 个，N 个 query）。
- **会话 ↔ 首 query**：1 个会话 = 1 个首 query（会话即"从首 query 开始的一次任务"）。1 沙箱 ↔ N 会话 ↔ N 首 query。
- **容器生命周期**：fork 自沙箱 → 跑 1 个 query → 跑完即销毁（不跨 query 存活，无 winner-sync）。下个 query 重新 fork。

---

## 2. 已确定项

| 项 | 确定 |
|----|------|
| 1 沙箱(workspace_init/Dockerfile) ↔ N 会话 ↔ N 首 query | ✅ |
| 每首 query → 8 容器并行（GRPO，rollout.n=8） | ✅ |
| N query 选 K 个并行（K 可被 N 整除），峰值 8K、跑完一批再下一批 | ✅（K 数值待定） |
| 容器 per-query 生命周期：fork→跑完即销毁，无 winner-sync | ✅ |
| winner 保留：单 query 内选优势最大那条（等价 reward 最高，同 `select_winner`），不跨 query | ✅ |
| workspace_init ↔ Dockerfile 一一对应 | ✅ |
| **subagent**：主 Agent 轨迹按数据本身（含 spawn 调用 + accepted 返回 + 后续 read，**不含子轨迹**）；子 Agent 轨迹单独训练（独立成样本） | ✅ 实测确认 |
| 会话结束①：Questioner 输出 `<end_session>`（用户满意） | ✅ |
| 会话结束②：Questioner 提问达上限次数 | ✅（数值待定） |

### 2.1 subagent 实测结论（数据证据，000069）

主 Agent 轨迹内**没有**子 Agent 内部轨迹，只有调用 + 返回：
```
[13] assistant: toolCall: sessions_spawn          ← 发起子任务
[14] toolResult: {"status":"accepted","childSessionKey":"..."}  ← 只返回"已接收"+子会话id
[17] assistant: toolCall: sessions_yield          ← 让出等待
[18] toolResult: {"status":"yielded",...}
[20] user: "读回 deliverables/..."                ← 下一轮驱动
[21] assistant: read×4                            ← 主 agent 自己读子会话写的文件
```
- 主轨迹 toolResult 全是主会话自己的工具（read/exec/search）+ sessions_spawn/yield，无子会话工具混入。
- 子 Agent 结果经文件系统回传（主 agent 后续 read），不在 spawn 返回值里。
- **所以主子轨迹在数据里天然分离**：主轨迹不含子轨迹、子轨迹是独立日志。`route_trajectories`（主）+ `route_subagents`（子）已对上，无需改。

---

## 3. 待定项（开会确认）

> 已切换数据源为 `data/taskspecs/`（taskspec），workspace↔query 回到 **1:1**（1 task = 1 `files/` = 1 `seed_query`），sample105 已弃。原"1 沙箱↔N 会话"方案作废，下列 N/K 相关项随之作废。

| # | 待定项 | 类别 | 备注 |
|---|--------|------|------|
| 1 | ~~N 个初始 query 来源/结构~~ | ~~数据~~ | **作废**：1:1，每 task 1 个 seed_query（taskspec.yaml 字段直接给）。 |
| 2 | ~~K 数值（N query 并行批）~~ | ~~运行时~~ | **作废**：1 query → 8 实例（GRPO），无 N query 并发问题。 |
| 3 | **Questioner 提问上限次数** | 运行时 | 会话结束条件②的数值。 |
| 4 | **会话结束其余条件** | 运行时 | scorer_error（打分异常）、沙箱硬失败等是否纳入及如何处理。现有 `simulated_session.py` 的 patience/k_budget（多轮产物）是否保留/调整。 |
| 5 | ~~数据单元 schema~~ | ~~数据~~ | **作废**：1:1，taskspec.yaml 即 schema（task_id/seed_query/verifier/user_profile + files/）。 |
| 6 | **judge 校准** | 模型 | 选型已定（`anthropic/claude-4.8-opus` 走 sufy，见 `doc/source/usersim.md`）。真正待定的是一致率校准（`calibrate_judge.py` judge↔人工，需 ClawEval 标注数据 @杨益博）。 |

> 历史待定项已清理：~~workspace_init↔Dockerfile 是否一一对应~~（已定一一对应）、~~与 session_pool 多轮模型如何共存~~（多余，删除）、~~N/K（1:n 作废）~~。**当前仅剩 #3/#4/#6 三项待定。**

---

## 4. 代码改动点（待定项定后落地）

### 4.1 数据管道 `data_pipeline/`（1:1 → 1:n）

| 文件 | 当前（1:1） | 改动方向 |
|------|------------|---------|
| `extract.py::extract_initial_queries` | 已留空抛 NotImplementedError | 按沙箱聚合 N 会话首 query（待数据结构/schema 定） |
| `classify.py` | 对每条 query 分桶 | 对每个 query 分桶（N 条） |
| `route.py::route_trajectories` | 每条 1 主轨迹 | 1 沙箱 ↔ N 主轨迹（每 query 一条） |
| `route.py::route_subagents` | 子轨迹独立成样本 | **不动**（已对） |
| `scripts/sample105_pipeline.py` | extract/classify/route 1:1 | 同步改 1:n |

### 4.2 运行时 `rollout/`

| 文件 | 现状 | 是否需改 |
|------|------|---------|
| `session_pool.py` | 多轮依赖 + winner 跨轮存活 | 独立模型需 per-query 生命周期（fork→跑完即销毁、无 winner-sync）。待定项定后改 |
| `scheduler.py` | 16 会话 × 8 槽 = 128 实例 | N query 选 K 并发的调度逻辑 |
| `base.yaml: rollout.n=8` | GRPO 组 8 路 | 不变 |

### 4.3 沙箱镜像

- `docker/sandbox/fs-seeds/` + `bin/seed_workspace.sh`：workspace_init → Dockerfile 物化（一一对应已确认）。
- 数据管道 `workspace_init` 字段当前只记路径、未实际使用——1:n 后需挂 N query。

---

## 5. 已确认不动

- **subagent 处理**（`dag.py` + `route_subagents`）：主子各自单独训练、格式原样、DAG 仅定位子日志。
- `rollout.n=8`（GRPO 8 路）。
- `prepare_queries.py` / `convert_dataset.py`：处理旧 `_stage_prefix_pass.jsonl`（已删数据），暂不动。

> 代码改动等 §3 待定项定后做；本文件为设计留档。
