# 沙盒 Rollout 设计

基于**腾讯云 Agent Runtime**（E2B 协议兼容）的 GRPO trajectory 采集方案。

> **调度与状态同步（16 会话 × 8 槽、会话内 winner 对齐、会话结束回母版）**：见 [`Sandbox_管理调度指南.md`](Sandbox_管理调度指南.md)。

## 1. 问题

GRPO 在同 query 的 M 条 trajectory 内归一化 reward 得到 advantage。若 M 个执行环境的初始状态不一致，方差里混入环境噪声，advantage 不再纯粹反映策略差异。

> 例：query 是"生成季度 PPT"。环境 A 残留一份历史 HTML、B 没有。A 直接转换、B 从头生成——这种差异源于环境而非策略。

**要求**：

- 同 query 的 M 个环境，rollout 起点位级一致
- 不同 query 之间互相独立
- 同 query 跨轮次允许知识演化（前一轮装的依赖、留下的文件继承到下一轮）
- 组内对齐只发生在 rollout 起止两个边界；中间故意发散——发散是 advantage 的方差来源

## 2. 为什么选沙盒（vs Docker）

| 项 | Docker 方案 | 沙盒方案 |
|---|---|---|
| 启动 | 秒级 | 毫秒级 |
| 快照 | `commit + push` ~10s+ | 进程级 ~亚秒 |
| 一对多孵化 | `pull + run × N` | 一个模板派生 N 个 |
| 资源回收 | 手动 | 暂停自动释放 |
| 协议 | Docker API | E2B 兼容，agent 框架可直接接 |

腾讯 Agent Runtime 提供 SDK（Python/Go）+ CLI（`agr`）+ RESTful API + MCP 多种接入；底层支持进程级快照、毫秒级启动、数万实例并发。团队决定用 AGS 替代 Docker。

## 3. 核心机制

### 3.1 关键概念

| 概念 | 定义 |
|---|---|
| **batch** | 一次 rollout 同时处理 B 个 query（默认 8） |
| **group** | 同 query 的 M 个沙盒（CL 主方案 **M=8**，`actor_rollout_ref.rollout.n=8`） |
| **总并发** | B × M |
| **环境链** | 每个 query 维护一份不断演化的环境状态（具体载体见 §4） |

### 3.2 生命周期

```
启动      母版状态 ──派生 M 个──> sandbox_1 ... sandbox_M
                                  （初始位级一致）

Rollout   M 路 agent 各自独立执行；状态自然发散；中途不同步

收尾      rewards → advantages → winner = argmax(advantage)
          winner 状态固化为新母版 → 写入 query 的环境链
          其余 M-1 个销毁
```

下一轮轮到同一 query 时，从新母版再派生 M 个——天然一致，**不需要"把 winner 状态扩散到其他沙盒"这一步**。

### 3.3 优胜者选择

GRPO 组内归一化后取 `argmax(advantage)`，等价于 `argmax(reward)`。

用 advantage 而非 reward 的理由：概念上对齐 GRPO 训练目标——"被固化的沙盒"和"梯度方向偏向的 trajectory"一致。

边界：M 条 reward 完全相同时按 `trajectory_id` 字典序选，确定性可复现。

### 3.4 跨 query 知识演化

每次 rollout 结束写出新母版，agent 装的依赖、留下的文件持续累积到下一轮，更接近真实"长期使用一台机器"的训练场景。

环境链定期回滚到 base（每 N 轮或文件累计过大时），防止环境无限膨胀。N 待实测确定。

## 4. 沙盒 API（与 Docker 对照）

### 4.1 抽象层次

AGS 把 Docker 的 image/container 二元拆成 **Tool / Instance** 两层：

| Docker | AGS |
|---|---|
| Dockerfile / image | **Tool**（模板：镜像、网络、超时、挂载） |
| `docker run` | **Instance**（从 Tool 起的运行体） |

### 4.2 操作对照

| Docker | AGS CLI | AGS SDK（E2B 兼容） |
|---|---|---|
| `docker run` | `agr instance create --tool-id` | `Sandbox.create(template=...)` |
| `docker exec` | `agr instance code run` | `sandbox.run_code(...)` / `sandbox.commands.run(...)` |
| `docker stop`（保留状态） | `agr instance pause` | （pause 仅 CLI/REST 暴露） |
| `docker start`（恢复） | `agr instance resume` | （同上） |
| `docker rm` | `agr instance delete` | `sandbox.kill()` |
| `docker attach` | — | `Sandbox.connect(sandbox_id)` |
| `docker commit` | — | — |
| Tool→Tool 克隆 | `agr tool fork` | — |

### 4.3 关键约束

- **Tool 是不可变的运行模板**。`agr tool fork` 仅 Tool→Tool 复制，**不直接支持"把 Instance 当前状态固化成新 Tool"**——cookbook CLI 层没有 commit 等价物，这是方案的关键 PoC 项（§6）。
- **网络模式**（`SANDBOX` / `PUBLIC` / `VPC`）在 Tool 创建时定，update 无法切换，要换网络必须 fork 新 Tool。CL 训练 query 大概率需要 `PUBLIC`（装包、调外部 API）。
- **SDK vs CLI 分工**：训练框架（Python）日常调用走 SDK；构建 Tool 模板、跨 batch 维护、固化操作走 CLI / RESTful。

## 5. 调用流程

### 5.1 单 query 伪代码

```python
from e2b_code_interpreter import Sandbox

# 环境变量
# E2B_API_KEY=...
# E2B_DOMAIN=ap-beijing.tencentags.com

# 1. 派生 M 个沙盒（同一 Tool 模板）
tool_id = query_tool_map[query.id]
group = [Sandbox.create(template=tool_id, timeout=600) for _ in range(M)]

# 2. M 路并行 rollout，互不通信
trajectories = parallel_map(M, lambda i: run_agent(group[i], query))
# run_agent 内：policy → sandbox.run_code / commands.run / files.write → observation → ...

# 3. 收尾
rewards = [scorer(t) for t in trajectories]
winner = int(argmax(grpo_norm(rewards)))

# 4. 固化优胜者状态为新 Tool（实现见 §6）
new_tool_id = freeze_instance_into_tool(group[winner])
query_tool_map[query.id] = new_tool_id

# 5. 销毁其余
for i, sb in enumerate(group):
    if i != winner:
        sb.kill()
```

`query_tool_map`（query_id → tool_id）随训练 checkpoint 持久化，断点恢复直接读回。

### 5.2 与 trainer / buffer 对接

```
verl rollout step
  ├─ batch = sample B queries
  ├─ for each query (parallel):
  │     ├─ sandboxes = [Sandbox.create(template=query_tool_map[q.id]) × M]
  │     ├─ trajectories = run_agent_each(sandboxes, q)
  │     ├─ advantage = grpo_norm([scorer(t) for t in trajectories])
  │     ├─ new_tool = freeze_instance_into_tool(sandboxes[argmax(advantage)])
  │     ├─ query_tool_map[q.id] = new_tool.id
  │     └─ kill other sandboxes
  └─ B × M 条 trajectory 路由进 7 桶 buffer
```

### 5.3 Trajectory 字段映射

| Buffer 字段 | 来源 |
|---|---|
| `response_token_ids` | agent 生成的 assistant message 拼接后 token |
| `messages` | OpenAI chat 格式 message 列表（带 token_span） |
| `reward` | scorer 计算结果 |
| `original_logprobs` | 该次 rollout 时 policy 的 logprob |
| `pattern_id` | query 模板或 tool 调用序列哈希 |
| `success_rate` | 同 query 组内 `mean(passed_i)` |
| `bucket`（7 桶领域） | **LLM 在处理任务时顺带输出**：rollout 系统 prompt 追加 `trainer.domain_tagging.build_domain_instruction()`，agent 末尾输出 `<task_domain>NAME</task_domain>`，由 `parse_domain()` 解析。无显式 bucket 字段时 `trajectory_adapter` 自动从轨迹文本回收；无法解析则跳过不入桶（B12） |

### 5.4 关键设计取舍

1. **不"resume 回到 winner"**：winner 已是新母版来源，其他销毁即可。下一轮再派生 M 个就天然一致，省 M-1 次 IO。
2. **状态不"统一字段"**：母版是不透明的，复制就行——不比较或合并内容。
3. **母版不可变**：写出来不再改。若固化失败，旧母版保留，本轮 trajectory 仍入 buffer，环境不演化。
4. **B 个 query 完全独立并行**：每个 query 一条自己的环境链，互不影响。

### 5.5 领域 / 入桶粒度 = per-query（契约）

trajectory 与入桶的最小单元是 **query**，不是 session：一个 query（带其会话上下文）跑出一条（组）trajectory，路由进它自己的 7 桶领域。**同一 session 的不同 query 进不同桶是正常的**——session 只是多轮上下文来源，本身不是入桶单元，因此不需要"一会话一领域"假设。

**对 rollout 的硬性要求**：

- 每个 query 单独产出 trajectory；**不要把整段会话合成一条 trajectory**。
- agent 在**该 query 的 response 末尾** emit `<task_domain>NAME</task_domain>`（指令来自 `trainer.domain_tagging.build_domain_instruction()`，注入系统 prompt）。
- 标签由模型在**看到完整上下文后**给出，因此 context-dependent 的追问（如"怎么样了"）也能被正确标成当前进行中任务的领域。
- `parse_domain()` 按条解析；无显式 `bucket` 字段时 `trajectory_adapter` 自动从轨迹文本回收，无法解析则跳过不入桶（B12）。

> `datasets/queries.jsonl` 仍按 session 分行存 `queries:[...]`（保留会话分组以便上下文回放），但**消费单元是其中的单个 query**。

## 6. PoC 待验证

接入 Phase 1 前必须用真账号实测（参考 ags-cookbook 的 SDK / CLI）：

| # | 验证项 | 决策含义 |
|---|---|---|
| 1 | **`freeze_instance_into_tool` 是否可行**：cookbook CLI 没有 Instance→Tool commit 命令，需核实底层 RESTful API 是否提供 | **阻塞项**。可行→走路径 A；不可行→走路径 B（见下） |
| 2 | 并发 `Sandbox.create` 起 M 个 Instance 的端到端 latency | 必须与 verl GPU 前向同量级，否则 GPU 空转 |
| 3 | 单账号 Instance 并发上限与配额 | 决定 B × M 能开多大 |
| 4 | `pause` 写出快照的大小与延迟 | 决定每 query 维护环境链的存储成本是否可持续 |
| 5 | 网络模式选择（CL query 多大比例需要 `PUBLIC`） | 决定 Tool 的默认 `NetworkMode` |

**实现路径二选一**（验证项 1 的结果决定）：

- **路径 A（首选，依赖 commit 能力）**：每个 query 维护一个 **Tool**，每轮 rollout 固化优胜 Instance 为新 Tool。
- **路径 B（兜底，仅靠 pause/resume）**：每个 query 维护一个 **长寿命 master Instance**（始终 pause 着）；rollout 时 resume master 得到工作实例 → 派生 M-1 个并行实例（API 待确认）→ 选 winner → 旧 master delete、winner pause 成新 master。

## 7. 风险与回退

| 风险 | 回退 |
|---|---|
| Instance→Tool 固化 API 不存在 | 走路径 B（pause/resume 长寿命 master） |
| 并发 `Sandbox.create` 性能不达预期 | 退回串行 resume，尾延迟增加 ~M × latency |
| 固化操作失败 | 旧母版保留，trajectory 仍入 buffer，环境不演化（损失：一轮知识累积） |
| 环境链膨胀 | 定期回滚到 base，或只保留最近 K 版 |
| E2B 协议与腾讯实现差异 | 抽 `SandboxClient` adapter 层（~50 行）隔离厂商耦合 |
| 部分 query 依赖外部状态（如真实数据库） | 这类 query 改用专门后端，不强求 100% 覆盖 |

## 参考

- 腾讯云 Agent Runtime 文档：<https://cloud.tencent.com/document/product/1814/129423>
- AGS Cookbook（本机已克隆至 `/root/workspace/ags-cookbook`）：<https://github.com/TencentCloudAgentRuntime/ags-cookbook>
  - `examples/mini-rl/main.py` — RL + 沙盒最小示例
  - `tutorials/sdk/e2b/e2b_base.ipynb` — SDK 入门
  - `skills/ags/SKILL.md` — `agr` CLI 完整命令参考
- E2B 协议：<https://e2b.dev/>
- 项目内：`doc/CL_Update_Sunhao.md`（GRPO advantage）、`doc/BucketDesign.md`（trajectory 入桶）、`doc/VerlIntegration.md`（trainer 对接）
