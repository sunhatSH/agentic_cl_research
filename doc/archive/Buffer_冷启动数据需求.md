# Replay Buffer 冷启动数据需求

> **受众**：数据制造（@吴健）、沙箱 rollout（@郑乃榕）  
> **目的**：在正式 RL 训练开始前，向 Replay Buffer 预灌一批「已跑完全程」的冷数据，避免 buffer 全空时 `L_replay = 0`、空桶 starvation。  
> **依据**：`configs/base.yaml`（`q_min=2000`×7 桶、`total_capacity=25000`）、`doc/source/CL_Design.md`、`doc/ops/sandbox/Sandbox_管理调度指南.md`。

---

## 0. 术语（必读）

| 术语 | 含义 |
|------|------|
| **queries（会话）** | 一条训练样本：`queries: [q1, q2, …]`，会话内顺序执行、共享 winner 同步后的环境与对话正史 |
| **query** | 会话内的一次用户轮次；每个 query 在训练时做 **GRPO 组内 8 路并行**（`rollout.n = 8`） |
| **trajectory（轨迹）** | Buffer 里存的单位：**一个槽位、一个 query** 的完整 rollout 记录（messages + token/logprob 等） |
| **冷数据** | 用 **π₀（初始策略）** 或等价基座在沙箱里跑完的记录；训练 step 0 前批量 `add` 进 buffer，**不参与** 当轮 RL 梯度（仅 replay） |

换算关系：

```
trajectory 数 = Σ (每个已执行 query × 每 query 轨迹条数)
每 query 轨迹条数 = 8（完整 GRPO 冷跑）或 1（经济版，见 §2）
```

---

## 1. 需要多少数据？

### 1.1 硬下限（必须达到）

Buffer 每桶 **hard floor** `q_min = 2000`。7 桶合计：

| 指标 | 数值 |
|------|------|
| **轨迹总数下限** | **14,000** |
| 含义 | 每桶至少 2,000 条 trajectory，采样时不会长期空桶 |

### 1.2 推荐目标（建议做到）

| 档位 | 轨迹总数 | 说明 |
|------|----------|------|
| **推荐** | **20,000** | 各桶约为 soft target 的 ~80%，replay 采样更稳 |
| 充裕 | 25,000 | 与 `total_capacity` 对齐，训练初期即满配 |

`replay_warmup_size`（默认 0）若设为 32，则轨迹数 ≥32 即可满 replay batch；**但桶级配额仍要求每桶 2000**，不能只做总量。

### 1.3 各桶轨迹数量（按 soft target 比例）

比例来自 `bucket_task_counts` + `allocate_quota(25000, 2000, α=0.5)`，与 ClawEval 纯文本 195 任务分布一致。

**下限档：共 14,000 轨迹**（完整 GRPO：每 query 8 条 trajectory）

| 桶 | ClawEval 任务数 | 占比 | 轨迹数（下限） | 约需 query 数 |
|----|-----------------|------|----------------|---------------|
| Workflow | 54 | 17.3% | **2,418** | 302 |
| SysOps | 52 | 17.1% | **2,392** | 299 |
| Dialogue | 38 | 15.8% | **2,208** | 276 |
| Finance | 18 | 13.3% | **1,868** | 233 |
| Communication | 12 | 12.4% | **1,731** | 216 |
| Knowledge | 11 | 12.2% | **1,705** | 213 |
| OfficeQA | 10 | 12.0% | **1,678** | 209 |
| **合计** | **195** | **100%** | **14,000** | **1,748** |

**推荐档：共 20,000 轨迹**（每 query 8 条）

| 桶 | 轨迹数（推荐） | 约需 query 数 |
|----|----------------|---------------|
| Workflow | 3,454 | 431 |
| SysOps | 3,418 | 427 |
| Dialogue | 3,154 | 394 |
| Finance | 2,669 | 333 |
| Communication | 2,473 | 309 |
| Knowledge | 2,436 | 304 |
| OfficeQA | 2,397 | 299 |
| **合计** | **20,000** | **2,497** |

### 1.4 需要制造多少条「会话」（queries 行）？

冷数据不必复刻生产集「平均 ~20 个 user 轮」；可用 **较短会话** 降成本，但 **多 query 会话占比** 不宜过低（见 §3）。

假设：**每 query 8 轨迹**、**平均每会话 8 个 query**：

| 档位 | 轨迹总数 | 约需会话总数 | 各桶约需会话数（同比例） |
|------|----------|--------------|---------------------------|
| 下限 | 14,000 | **~218** | Workflow 38 / SysOps 37 / Dialogue 34 / Finance 29 / Communication 27 / Knowledge 27 / OfficeQA 26 |
| 推荐 | 20,000 | **~312** | Workflow 54 / SysOps 53 / Dialogue 49 / Finance 42 / Communication 39 / Knowledge 38 / OfficeQA 37 |

若平均会话只有 **4 个 query**，会话数约为上表的 **2 倍**。

### 1.5 经济版（仅在下限压力极大时）

| 模式 | 每 query 轨迹数 | 达到 14k 轨迹所需 query 数 | 风险 |
|------|-----------------|----------------------------|------|
| **标准（首选）** | **8** | ~1,748 | 与训练 GRPO 分布一致；priority 中 success_rate / diversity 信号完整 |
| 经济版 | 1 | ~14,000 | 成本低 8 倍；**仅作救火**；需在 meta 标 `cold_seed_single_sample: true` |

**不接受**：无 token/logprob、无 messages 的纯文本摘要——replay 前向需要 `response_token_ids` + `original_logprobs`（或训练侧重算 logprob 的约定版本，需与 @孙豪 对齐）。

---

## 2. 交付格式与必填字段

### 2.1 会话清单（给 rollout 调度）

与 `scripts/convert_dataset.py` / verl `extra_info` 对齐，每行 JSON：

```json
{
  "record_id": "cold_00001",
  "bucket": "SysOps",
  "queries": ["用户第一句", "用户第二句", ...],
  "checkers": [],
  "persona": "finance_analyst",
  "meta": { "cold_seed": true, "policy": "pi0" }
}
```

- `bucket`：**必填**，七桶之一（见上表）；错误桶会导致 replay 配额失衡。
- `queries`：有序列表；长度 = 本会话的 query 数（见 §3 单/多 query）。
- `checkers`：冷数据可留空；打分走 model judge（`trainer/model_reward.py`），与训练一致。

**数据清洗**：queries 文件在进入采集前建议先用 `scripts/clean_queries.py` 清洗（去零宽字符 + 乱码过滤）；`collect_cold.py` / `collect_rollout.py` 默认也会在采集时对 seed query 和轨迹消息做在线清洗（`--no-clean` 可跳过）。采集完成后可用 `scripts/clean_buffer.py` 对 buffer 快照做批量后处理。详见 `data/cleaning.py`。

### 2.2 轨迹记录（进 buffer）

每条 trajectory 一条记录（或嵌套在会话结果里），字段对齐 `rollout/session_pool.Trajectory` + `trainer/trajectory_adapter`：

| 字段 | 必填 | 说明 |
|------|------|------|
| `trajectory_id` | 是 | 全局唯一 |
| `record_id` | 是 | 关联会话 |
| `query_index` | 是 | 会话内第几个 query（0-based） |
| `slot_idx` | 是 | 0–7 |
| `bucket` | 是 | 与清单一致 |
| `messages` | 是 | OpenAI chat（assistant + tool）；**仅本 query 产生的消息**，不含已写入 session 正史的前缀（与训练一致） |
| `reward` | 是 | model judge 分数；无法打分时标 `null` 并整条丢弃（见 §4） |
| `response_token_ids` | 是 | 本 query response 的 token id |
| `logprobs` / `original_logprobs` | 是 | 与 token 对齐 |
| `success` | 建议 | judge 或 checker 二值结果 |
| `meta` | 建议 | `policy=pi0`, `cold_seed=true`, 沙箱 persona、耗时等 |

交付形态二选一（与工程约定即可）：

1. **JSONL**：`cold_trajectories.jsonl`（扁平 trajectory）+ `cold_sessions.jsonl`（会话清单）  
2. **Parquet**：与 `rollout/collect.py` dry-run 输出字段一致，便于 `buffer.add` 脚本直接导入

---

## 3. 单 query vs 多 query 会话

### 3.1 定义

| 类型 | 条件 | 调度行为 |
|------|------|----------|
| **单 query 会话** | `len(queries) == 1` | spawn 8 槽 → 跑 1 个 query → destroy；无跨 query 同步 |
| **多 query 会话** | `len(queries) >= 2` | 每 query：8 路 rollout → winner 同步环境与 **session_history** → 下一 query |

### 3.2 整体占比（会话条数，非轨迹数）

生产用户数据（queries JSONL）多为长会话（平均 user 轮次高）。冷数据可适当缩短单会话 query 数，但 **类型占比** 建议：

| 类型 | 占冷启动 **会话** 比例 | 说明 |
|------|------------------------|------|
| 多 query 会话 | **≥ 60%** | 覆盖 winner-sync、多轮状态、对话正史 |
| 单 query 会话 | **≤ 40%** | 覆盖单轮工具任务、短 QA |

### 3.3 各桶建议（会话级）

| 桶 | 单 query | 多 query | 多 query 会话内建议 query 数 |
|----|----------|----------|------------------------------|
| **Dialogue** | **0%** | **100%** | 4–12（必须测澄清/指代/「刚才那个」） |
| **Workflow** | **≤ 10%** | **≥ 90%** | 4–10（步骤链、编排） |
| **OfficeQA** | 30% | 70% | 3–8（表格/文档多轮追问） |
| **SysOps** | 40% | 60% | 2–6（部署失败重试、日志追问） |
| **Finance** | 50% | 50% | 2–5（算完再问、核对） |
| **Communication** | **60%** | 40% | 2–4（润色/翻译多为短会话） |
| **Knowledge** | **60%** | 40% | 2–5（检索+总结可单轮） |

**硬规则**：

- **Dialogue 桶不得交付单 query 会话**（与桶语义冲突）。
- 多 query 会话内 query 必须 **语义前后关联**（后一轮依赖前轮 winner 的环境或对话），禁止同一任务拆成无关 query 凑数。

### 3.4 与「任务 / query」数量的关系

- ClawEval **195 个任务类型**是能力分桶依据，**不是**冷数据要各跑 195 条的硬性清单。
- 冷数据按 **桶轨迹配额**（§1.3）采即可；同一任务类型可多条会话、不同 persona/种子文件。

---

## 4. Query 跑失败时，后续 query 还执行吗？

> 本节同时约束 **正式训练 rollout** 与 **冷数据制造**。

### 4.1 失败分类

| 类别 | 典型表现 | `reward` |
|------|----------|----------|
| **A. 打分器异常** | judge 超时、API 错误 | `null` |
| **B. 软失败** | 轨迹完整但答错、无 final_answer、低分 | 有数值（通常低） |
| **C. 硬失败** | 8 槽全崩、沙盒不可恢复、磁盘状态损坏、轨迹为空无法拼 prompt | 无有效轨迹或环境不可信 |

### 4.2 正式训练 rollout 策略（与当前 `session_pool.py` 对齐）

| 类别 | 本 query 是否进 buffer | 是否 sync 到 winner | **后续 query 是否继续** |
|------|------------------------|---------------------|-------------------------|
| **A. 打分器异常** | **否**（8 条均不入库或标 discard） | **否**（保持上一边界状态） | **是，继续** |
| **B. 软失败** | **是**（低分轨迹可入库，供 replay / 负样本） | **是**（§3③：全同分/全失败 → **随机** winner） | **是，继续** |
| **C. 硬失败** | 已产出 traj 按质量单条审查；未产出则无 | **否** | **否，终止本 session 剩余 query** |

**理由（多 query 会话）**：

- 后续 query 的 prompt 前缀 = **session_history**（历代 winner 消息），环境 = **上一轮 winner 磁盘状态**。
- A 类：环境未坏，只是没分 → 不 sync 但可继续（与 `pick_winner → None → continue` 一致）。
- B 类：仍有可比较轨迹 → 必须 sync（哪怕随机），否则 8 槽环境分叉，多轮语义崩坏。
- C 类：winner 不存在或环境不可信 → **继续跑只会制造「假多轮」**，应 **abort session**，换一条新会话。

代码参考：`rollout/session_pool.run_session` — `winner is None` 时 `continue` 且不 `sync_to_winner`（对应 A）；`select_winner_with_fallback` 全同分随机（对应 B）。

### 4.3 冷数据制造策略

| 情形 | 要求 |
|------|------|
| A / B | 按上表继续；交付时保留 `query_index` 连续 |
| C | **不要**再跑同会话后续 query；会话标 `session_status: aborted_at_query_k`；可选另起新会话补桶配额 |
| 部分槽崩溃（非 8 槽全灭） | 视为 B：其余槽继续评比；**不要**中止 session |

### 4.4 待 PoC 细项（不阻塞冷数据开工）

`doc/ops/sandbox/Sandbox_管理调度指南.md` §3③ 中「全同分随机 vs 不 sync」在 **B 类全零分** 已定为 **随机 winner + sync**（当前代码）。若 PoC 发现随机 sync 污染严重，可改为「不 sync 但继续」——需改代码并回溯冷数据规则。

---

## 5. 验收清单

数据侧交付前自检：

- [ ] 轨迹总数 ≥ 14,000（推荐 20,000）
- [ ] 每桶轨迹数 ≥ 2,000（对照 §1.3 表）
- [ ] Dialogue 桶无单 query 会话；多 query 会话占比 ≥ 60%
- [ ] 每条 trajectory 含 messages + token + logprob + bucket + query_index
- [ ] `reward is null` 的轨迹不计入配额（或单独报表）
- [ ] 硬失败 session 无「abort 后仍交付后续 query」的脏数据
- [ ] `record_id` / `trajectory_id` 无碰撞

工程侧导入后：

- [ ] `buffer.stats()` 各桶 `size ≥ q_min`
- [ ] `pytest` 中 buffer 相关用例仍通过
- [ ] 可选：`replay_warmup_size: 32` 时 step 0 的 `effective_replay_batch_size` 已满

**数据来源配比（Phase 0，见 [`CL_Design.md`](../source/CL_Design.md)）：**

- [ ] 每条 trajectory 带 `meta.policy` 来源标记（`pi0_27b` = 本地 Qwen3.6-27B on-policy；`gpt5` = sufy `openai/gpt-5` off-policy）——`collect_rollout.py` 已按 `--actor` 自动写入
- [ ] 两来源**分目录**存（`data/rollouts/{local,remote}/`），便于 `warmup_buffer.py --ratio-27b` 按桶内配比混合
- [ ] `warmup_buffer.py` 输出的 `*.manifest.json` 核对：每桶实际 27B/gpt5 条数与目标配比一致（P0-A 1.0 / P0-B 0.0 / P0-C 0.5 / P0-D 0.7 / P0-E 0.3）
- [ ] **红线**：gpt-5 采的轨迹只进 buffer 做 replay，**绝不**拿去 SFT 蒸馏 27B（违背 B2/C1 立论 + 污染 21 实验可比性）

---

## 6. 联系人

| 事项 | 负责人 |
|------|--------|
| 桶语义 / 配额 / buffer 导入 | @孙豪 |
| 冷数据会话选题与清单 | @吴健 |
| 沙箱跑批、winner-sync、失败 abort | @郑乃榕 |
| judge 打分与 `reward=null` 界定 | @孙豪 + 评测 |

**文档版本**：2026-06-11
