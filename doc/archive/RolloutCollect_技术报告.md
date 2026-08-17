# Rollout 采集系统技术报告

> **范围**：本报告记录**冷启动多轮 rollout 采集系统**的设计、实现与验证（@孙豪 搭建）。
> **创建**：2026-06-13。配套设计见 [`UserSim_多轮Query在线生成.md`](UserSim_多轮Query在线生成.md)、运行手册见 [`ColdRollout_采集.md`](../ops/sandbox/ColdRollout_采集.md)、论文表述见 `paper/drafts/Paper_Method_draft_*.md` §4.5。

## 0. 贡献边界（务必明确）

本系统及其产出的 **rollout 数据结构由本工作（@孙豪）定义与产出**。模块分工：

| 模块 | 负责人 | 边界 |
|------|--------|------|
| **query / 种子 + 镜像** | @吴健 | 提供每会话首条 query 作种子，**以及该会话对应的沙箱镜像文件**（即会话的初始环境快照）。**只给 query+镜像，不碰 rollout。** |
| **冷启动 + 后续 rollout 全链路** | **@孙豪（本工作）** | observer/questioner agent、沙箱 rollout、采集脚本、**最终 rollout 数据结构**、入桶与训练衔接 |

即：**query + 镜像是输入（吴健提供），rollout 轨迹及其数据结构是本系统的产出**。下游训练（replay buffer 预热 / verl）消费的是本系统定义的数据结构（§3），而非吴健的 query 原始格式。

## 1. 系统形态（冷启动阶段）

冷启动采集是完整 UserSim 设计（三 agent + GRPO + 奖励）的**简化子集**，针对"先产出可用数据"目标裁剪：

| 维度 | 冷启动（本阶段） | 完整训练态（设计目标） |
|------|------------------|------------------------|
| 组结构 | **1 query = 1 rollout**（单轨迹） | 8 槽 GRPO 组 |
| winner / 正史对齐 | **无**（单轨迹无需选 winner / sync） | winner-sync |
| 奖励模型 | **无**（冷启动不打分） | observation-grounded judge |
| **observer（观察 agent）** | **有** | 有 |
| **questioner（出题人设 agent）** | **有** | 有 |
| 多轮 | 有（observer→questioner 驱动，K~U{1..k_max}） | 有 |

> 与论文 §4.5"三 agent 含 reward"的关系：冷启动是其**去 reward、去 GRPO 组**的子集；论文描述的是完整训练态，本阶段只跑数据产出所需的最小环。

## 2. 双 Actor 架构（数据分开存）

actor（被采集策略）有两套并跑，**数据严格分目录**：

| Actor | 模型 | 后端 | 数据目录 |
|-------|------|------|----------|
| 本地 | Qwen3.6-27B | 本地 verl rollout（lightllm） | `data/mock/rollouts/local/` |
| 远程/沙箱 | **openai/gpt-5** | sufy（OpenAI 兼容，沙箱内 hermes 调） | `data/mock/rollouts/remote/` |

三个 agent 角色用**三个不同模型**（抗 self-preference）：
- 沙箱内 actor = `openai/gpt-5`；observer = `openai/gpt-5-mini`；questioner = `anthropic/claude-sonnet-5`（+ 轮换池）。选型单一信源见 [`模型选型.md`](模型选型.md)。
- 远程 API = sufy（`https://openai.sufy.com/v1`，key `SUFY_API_KEY`）。旧 tokenhub 已弃用（沙箱连不上商汤内网）。
- **硬约束**：observer/questioner 必须用远程模型；远程不可用则**中止并报告**，不降级（无 observer = 无报告 = 多轮无法进行）。

## 3. 最终 Rollout 数据结构（本工作定义）

每条 session 一行 JSONL，结构如下（即下游训练消费的契约）：

```
session = {
  record_id:          str          # 来自种子数据的 join key
  seed_query:         str          # q1，真实回流种子（吴健 query 的首条）
  sandbox_image:      str          # 会话对应的沙箱镜像标识（吴健提供，用于初始化环境）
  persona:            str          # 会话级随机人设（42 选 1）
  num_turns:          int
  ended_by:           str          # k_budget | end_session | agent_error
  generated_queries:  list[str]    # questioner 在线生成的后续 query（本系统产出）
  trajectories:       list[Traj]   # 每轮一条轨迹（单 rollout）
  reports:            list[Report] # 每轮 observer 的客观报告
}
Traj = {
  trajectory_id, messages: list[{role, content}],
  bucket: str|None,                # actor <task_domain> 标签 → 7 桶路由
  response_token_ids: list[int], response_mask: list[int],  # 训练用（native 字段）
  num_turns: int
}
Report(observer) = { intermediate, final, actor_claims, discrepancies, file_tree }
```

**关键点**：`query` 来自吴健（seed）+ 本系统在线生成（generated_queries）；`trajectories`/`reports`/`bucket`/token 级字段**全部由本系统产出**。这套结构对接 `rollout/collect.py::ingest_trajectories` 入 7 桶 buffer，及 verl 训练。

## 4. 实现（本次新增代码）

- `rollout/usersim_collect.py` — `run_usersim_session`：slots=1 单轨迹多轮，observer→questioner，无 winner/reward。
- `scripts/collect_rollout.py` — 入口：抽种子→N 路并发→落 JSONL；`--actor local|remote`，`--node-rank/--num-nodes` 分片。observer/questioner 缺失 `exit 5`。
- `scripts/collect_rollout.sh` / `_node_worker.sh` / `launch_8node.sh` — 单机 / 每机 worker / 8 机下发。
- 运行环境：**唯一能加载 Qwen3.6-27B（`qwen3_5` 架构）的是 `/mnt/afs_code/ds32_env` 的 vllm 0.16rc**（自带 `qwen3_5.py`）；旧 vllm 0.11/0.13 不支持。详见 `ColdRollout_采集.md` §4。

## 5. 验证结果（2026-06-13）

| 验证项 | 结果 |
|--------|------|
| 远程三模型连通 | ✅ sufy（openai/gpt-5 / openai/gpt-5-mini / anthropic/claude-sonnet-5） |
| e2b 真沙箱 | ✅ 起实例→run_code→回收，连通 |
| 本地 27B vllm（ds32_env，TP=8） | ✅ 加载成功，`Resolved architecture: Qwen3_5ForConditionalGeneration` |
| **远程 actor + 真沙箱 + 多轮** | ✅ `limit=3 done=3 failed=0`，observer 探到真实 file_tree，questioner 生成带人设的追问 |
| **本地 27B actor + 真沙箱 + 多轮** | ✅ `limit=2 done=2 failed=0` |

**已知局限（如实记录）**：当前种子来自验证用 queries JSONL（非吴健正式 query），多为闲聊开场，actor 常不触发 `<toolcall>` → `bucket=None`、无沙箱执行动作。这是**种子质量**问题（吴健正式 query 到位后改善），非链路缺陷——沙箱/observer/questioner 经验证均正常工作。

## 6. 数据隔离

正式数据未到位前，所有产物（数据+模型）落 `data/mock/`，与正式 `data/rollouts`、`ckpts/`、`buffer_dumps/` 隔离；**base 模型不隔离**。详见 `ColdRollout_采集.md` §6。
