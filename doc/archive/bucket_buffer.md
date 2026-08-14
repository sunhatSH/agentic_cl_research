# Replay Buffer 桶结构与冷启动数据需求

> 合并自 `BucketDesign.md`（7桶结构设计）与 `Buffer_冷启动数据需求.md`。
> 目标：纯文本 continual learning，学新任务时不遗忘旧能力。不纳入 multimodal（仅 4 任务，能力结构不同）。

---

## 1. 7 桶结构

按**能力/领域**分桶，不按难度分桶（难度随模型能力提升漂移，非稳定依据；灾难性遗忘更常沿能力类型发生）。每桶保底配额，桶内按 priority 存留，桶间不直接竞争。Priority 不依赖 reward 绝对值（reward 整体上升会使旧轨迹系统性被淘汰，buffer 退化为滑动窗口）。

### 1.1 桶结构树形图

```text
ReplayBuffer
├── Workflow [54]
│   ├── workflow [47]
│   └── productivity [7]
├── SysOps [52]
│   ├── ops [31]
│   ├── operations [6]
│   ├── terminal [5]
│   ├── safety [5]
│   ├── security [2]
│   ├── coding [2]
│   └── file_ops [1]
├── Finance [18]
│   ├── finance [14]
│   ├── compliance [2]
│   └── procurement [2]
├── Knowledge/Analysis [11]
│   ├── research [3]
│   ├── knowledge [2]
│   ├── synthesis [2]
│   ├── comprehension [2]
│   ├── data_analysis [1]
│   └── memory [1]
├── Communication [12]
│   ├── communication [8]
│   ├── content [2]
│   ├── rewriting [1]
│   └── organization [1]
├── OfficeQA [10]
│   └── office_qa [10]
└── Dialogue [38]
    ├── what [26]
    └── user_agent [12]
```

纯文本 buffer 总任务数 = 195（199 纯文本任务剔除 4 个 multimodal）。

### 1.2 各桶描述

| 桶 | 任务数 | 核心能力 |
|---|---:|---|
| **Workflow** | 54 | 多步骤任务组织、流程推进、子任务拆解与执行顺序控制 |
| **SysOps** | 52 | 工具使用、命令与系统操作、约束遵守、安全边界、程序性执行；各 category 共享环境交互与操作型特征，遗忘模式相近 |
| **Finance** | 18 | 结构化业务规则、数值意识、规范判断与合规约束 |
| **Knowledge/Analysis** | 11 | 检索、阅读理解、归纳总结、信息整合与轻量分析推理 |
| **Communication** | 12 | 表达、改写、风格控制、信息组织和面向受众的沟通 |
| **OfficeQA** | 10 | 办公语境中的结构化问答、字段定位、文档细节理解；能力边界清晰，单独成桶不并入 Knowledge（并入会被稀释） |
| **Dialogue** | 38 | 多轮交互中的状态跟踪、上下文保持、角色一致性与对话策略 |

---

## 2. Quota 分配

### 2.1 公式

$$q_i = q_{min} + (C - B \cdot q_{min}) \cdot \frac{n_i^{\alpha}}{\sum_j n_j^{\alpha}}$$

- $C$：总 buffer 容量（轨迹条数）
- $B$：桶数，$B = 7$
- $q_{min}$：每桶 hard floor
- $n_i$：第 $i$ 个桶任务数
- $\alpha$：次线性指数，推荐 $0.5$（平方根分配）

**含义**：$q_{min}$ 保证每桶最低生存空间；第二项将剩余容量按桶规模次线性分配。大桶容量更多但增长慢于任务数增长，兼顾主流与长尾能力。不采用纯比例分配（中小桶长期得不到 replay）；不采用完全均匀分配（大桶覆盖不足）。

### 2.2 工程理解

- **hard floor**：每桶最低必须保留容量，不可跌破
- **soft target**：上式计算的目标容量；桶容量 < soft target → 新轨迹更易接纳；桶容量 > soft target → 桶内淘汰更积极

### 2.3 示例：C = 25,000, q_min = 2,000, α = 0.5

总 hard floor = 7 × 2,000 = 14,000；剩余可分配 = 11,000。

| 桶 | 任务数 | √n 权重 | quota 近似值 |
|---|---:|---:|---:|
| Workflow | 54 | 7.35 | 4,319 |
| SysOps | 52 | 7.21 | 4,272 |
| Dialogue | 38 | 6.16 | 3,938 |
| Finance | 18 | 4.24 | 3,337 |
| Communication | 12 | 3.46 | 3,092 |
| Knowledge/Analysis | 11 | 3.32 | 3,047 |
| OfficeQA | 10 | 3.16 | 2,995 |
| **合计** | **195** | **34.90** | **25,000** |

Workflow/SysOps/Dialogue 大桶配额更高；Finance 居中；Knowledge/Communication/OfficeQA 因 hard floor + 次线性加权获得足够容量。

---

## 3. Priority 设计

训练推进时 reward 整体上升，按 reward 绝对值排序会使新轨迹系统性压制旧轨迹，buffer 变成"最近任务缓存"。Priority 反映的是**轨迹对防止遗忘的重要性**，而非**当时取得多高 reward**。

### 3.1 四信号构成

| 信号 | 权重 | 定义 |
|---|---:|---|
| **Forgetting Risk（遗忘风险）** | 0.5 | 当前模型在该轨迹上是否出现性能回退；旧轨迹 replay 时较历史最好水平明显变差 → 抗遗忘价值高 |
| **Rarity（稀缺性）** | 0.25 | 桶内出现频率较低的模式/模板/任务子类更难被淘汰，防 buffer 被少数热门模板占满 |
| **Difficulty（桶内相对难度）** | 0.25 | 仅桶内相对难度，非全局绝对 reward；刻画同一能力类型内的边界情况、复杂流程、高约束场景 |
| ~~Diversity/Redundancy~~ | v1 禁用 | 桶内已有大量近似轨迹时新轨迹边际价值下降；每 query 仅 2 条轨迹，重复堆积风险高，后续版本启用 |

> v1 priority = forgetting_risk(0.5) + rarity(0.25) + difficulty(0.25)，diversity 禁用。

高 priority 轨迹 = 能代表旧能力 + 当前已退化 + 桶内少见 + 与其他样本不重复 + 覆盖关键边界场景。

---

## 4. 淘汰规则：桶内淘汰

**禁止跨桶挤出**。全局淘汰会使高频任务持续吞噬其他类型容量，分桶沦为"名义上分桶"。

流程：新轨迹按能力映射进入所属桶 → 桶未满直接接纳 → 桶已满仅在该桶内部比较并淘汰低价值轨迹 → 不允许跨桶挤出。

淘汰解决的不是"单步读取量"，而是"长期存储集合的质量"：把 buffer 维持为固定预算下的高密度代表集。

---

## 5. 两级采样

**第一级（采桶）**：混合策略——一部分概率按桶 quota/soft target 比例采样，一部分按桶均匀采样。保证大桶覆盖需求 + 长尾能力持续被 replay。

**第二级（桶内采轨迹）**：按桶内 priority **加权随机**采样，**非 top-k 贪心**。避免桶内只重复使用少数"明星轨迹"，损害覆盖度。

---

## 6. Buffer 组织与检索

**轨迹级元数据**（每条至少维护）：`trajectory_id` / `record_id` / `query_index` / `slot_idx` / `bucket` / `source_task` / `source_category` / `priority` / `insert_step` / `last_replay_step` / `replay_count` / `token_length` / `pattern_id` / `template_id`（如可获得）/ 遗忘相关状态标记（近期性能回退、稀有模式、边界样本）。

**主存储 + 轻量索引**，至少支持：按 ID 访问（精确复查）、按 bucket 访问（分析桶内容）、按 priority 排序访问（查看最该保留/淘汰）、按 pattern/template/task family 访问（检查重复堆积与稀有模式覆盖）。

**精确查询 vs 训练采样分开**：精确查询（deterministic retrieval）给定条件返回轨迹集合，用于调试/诊断/评估；训练采样（stochastic replay sampling）按概率抽样，用于训练。第一版至少支持：查某 bucket 全部轨迹、top-k/bottom-k priority、长时间未 replay 轨迹、某 pattern/task_id 轨迹。

---

## 7. 冷启动数据需求

> **受众**：数据制造（@吴健）、沙箱 rollout（@郑乃榕）
> **目的**：RL 训练开始前向 buffer 预灌「已跑完全程」的冷数据，避免 `L_replay = 0`、空桶 starvation。
> **依据**：`configs/base.yaml`（`q_min=2000`×7、`total_capacity=25000`）。

### 7.1 术语

| 术语 | 含义 |
|------|------|
| **queries（会话）** | 一条训练样本：`queries: [q1, q2, …]`，会话内顺序执行、共享 winner 同步后的环境与对话正史 |
| **query** | 会话内一次用户轮次；训练时做 GRPO 组内 8 路并行（`rollout.n = 8`） |
| **trajectory（轨迹）** | Buffer 存储单位：一个槽位、一个 query 的完整 rollout 记录 |
| **冷数据** | π₀（初始策略）在沙箱跑完的记录；step 0 前批量 `add` 进 buffer，不参与当轮 RL 梯度（仅 replay） |

换算：`trajectory 数 = Σ(每个已执行 query × 每 query 轨迹条数)`；每 query 轨迹数 = 8（标准）或 1（经济版）。

### 7.2 数量需求

**硬下限**：每桶 `q_min = 2000`，7 桶合计 **14,000 轨迹**。

| 档位 | 轨迹总数 | 说明 |
|------|----------|------|
| 下限 | 14,000 | 每桶至少 2,000 |
| **推荐** | **20,000** | 各桶约为 soft target ~80% |
| 充裕 | 25,000 | 与 `total_capacity` 对齐 |

`replay_warmup_size`（默认 0）若设 32，轨迹数 ≥32 即满 replay batch；**但桶级配额仍要求每桶 2000**，不能只做总量。

**下限档各桶轨迹数**（完整 GRPO，每 query 8 条）：

| 桶 | 任务数 | 占比 | 轨迹数 | 约需 query 数 |
|----|---:|---:|---:|---:|
| Workflow | 54 | 17.3% | 2,418 | 302 |
| SysOps | 52 | 17.1% | 2,392 | 299 |
| Dialogue | 38 | 15.8% | 2,208 | 276 |
| Finance | 18 | 13.3% | 1,868 | 233 |
| Communication | 12 | 12.4% | 1,731 | 216 |
| Knowledge | 11 | 12.2% | 1,705 | 213 |
| OfficeQA | 10 | 12.0% | 1,678 | 209 |
| **合计** | **195** | **100%** | **14,000** | **1,748** |

**推荐档各桶轨迹数**（每 query 8 条）：

| 桶 | 轨迹数 | 约需 query 数 |
|----|---:|---:|
| Workflow | 3,454 | 431 |
| SysOps | 3,418 | 427 |
| Dialogue | 3,154 | 394 |
| Finance | 2,669 | 333 |
| Communication | 2,473 | 309 |
| Knowledge | 2,436 | 304 |
| OfficeQA | 2,397 | 299 |
| **合计** | **20,000** | **2,497** |

### 7.3 会话数估算

假设每 query 8 轨迹、平均每会话 8 个 query：

| 档位 | 轨迹总数 | 约需会话总数 |
|------|---:|---:|
| 下限 | 14,000 | ~218 |
| 推荐 | 20,000 | ~312 |

若平均会话仅 4 个 query，会话数约为上表 2 倍。

### 7.4 经济版（仅救火）

| 模式 | 每 query 轨迹数 | 达 14k 所需 query | 风险 |
|------|---:|---:|------|
| **标准（首选）** | 8 | ~1,748 | 与训练 GRPO 分布一致；priority 信号完整 |
| 经济版 | 1 | ~14,000 | 成本低 8 倍；仅作救火；meta 标 `cold_seed_single_sample: true` |

**不接受**：无 token/logprob、无 messages 的纯文本摘要——replay 前向需要 `response_token_ids` + `original_logprobs`。

### 7.5 交付格式

**会话清单**（每行 JSON，对齐 `scripts/convert_dataset.py` / verl `extra_info`）：

```json
{
  "record_id": "cold_00001",
  "bucket": "SysOps",
  "queries": ["用户第一句", "用户第二句"],
  "checkers": [],
  "persona": "finance_analyst",
  "meta": { "cold_seed": true, "policy": "pi0" }
}
```

- `bucket`：**必填**，七桶之一；错误桶导致 replay 配额失衡
- `checkers`：冷数据可留空；打分走 model judge（`trainer/model_reward.py`）
- 数据清洗：采集前用 `scripts/clean_queries.py`；`collect_cold.py` / `collect_rollout.py` 默认在线清洗（`--no-clean` 可跳过）；采集后用 `scripts/clean_buffer.py` 批量后处理

**轨迹记录**（字段对齐 `rollout/session_pool.Trajectory` + `trainer/trajectory_adapter`）：

| 字段 | 必填 | 说明 |
|------|:---:|------|
| `trajectory_id` | 是 | 全局唯一 |
| `record_id` | 是 | 关联会话 |
| `query_index` | 是 | 会话内第几个 query（0-based） |
| `slot_idx` | 是 | 0–7 |
| `bucket` | 是 | 与清单一致 |
| `messages` | 是 | OpenAI chat（assistant + tool）；**仅本 query 产生的消息**，不含 session 正史前缀 |
| `reward` | 是 | model judge 分数；无法打分标 `null` 并整条丢弃 |
| `response_token_ids` | 是 | 本 query response 的 token id |
| `logprobs` / `original_logprobs` | 是 | 与 token 对齐 |
| `success` | 建议 | judge/checker 二值结果 |
| `meta` | 建议 | `policy=pi0`, `cold_seed=true`, persona, 耗时等 |

交付形态二选一：JSONL（`cold_trajectories.jsonl` + `cold_sessions.jsonl`）或 Parquet（与 `rollout/collect.py` dry-run 输出一致）。

### 7.6 单 query vs 多 query 会话

| 类型 | 条件 | 调度行为 |
|------|------|----------|
| 单 query | `len(queries) == 1` | spawn 8 槽 → 跑 1 query → destroy；无跨 query 同步 |
| 多 query | `len(queries) >= 2` | 每 query：8 路 rollout → winner 同步环境与 session_history → 下一 query |

**整体占比**（会话条数）：多 query ≥ 60%，单 query ≤ 40%。

**各桶建议**（会话级）：

| 桶 | 单 query | 多 query | 多 query 内建议 query 数 |
|----|---:|---:|---|
| Dialogue | **0%** | **100%** | 4–12（必须测澄清/指代/「刚才那个」） |
| Workflow | ≤ 10% | ≥ 90% | 4–10（步骤链、编排） |
| OfficeQA | 30% | 70% | 3–8（表格/文档多轮追问） |
| SysOps | 40% | 60% | 2–6（部署失败重试、日志追问） |
| Finance | 50% | 50% | 2–5（算完再问、核对） |
| Communication | 60% | 40% | 2–4（润色/翻译多为短会话） |
| Knowledge | 60% | 40% | 2–5（检索+总结可单轮） |

**硬规则**：
- Dialogue 桶不得交付单 query 会话（与桶语义冲突）
- 多 query 会话内 query 必须**语义前后关联**（后轮依赖前轮 winner 的环境或对话），禁止同任务拆无关 query 凑数
- ClawEval 195 任务类型是分桶依据，**不是**冷数据各跑 195 条的硬性清单；按桶轨迹配额采即可

### 7.7 Query 失败处理

同时约束正式训练 rollout 与冷数据制造。

**失败分类**：

| 类别 | 典型表现 | reward |
|------|----------|--------|
| A. 打分器异常 | judge 超时、API 错误 | `null` |
| B. 软失败 | 轨迹完整但答错/无 final_answer/低分 | 有数值（通常低） |
| C. 硬失败 | 8 槽全崩/沙盒不可恢复/磁盘损坏/轨迹为空 | 无有效轨迹或环境不可信 |

**正式训练 rollout 策略**（对齐 `session_pool.py`）：

| 类别 | 进 buffer | sync winner | 后续 query |
|------|:---:|:---:|:---:|
| A. 打分器异常 | 否（8 条均不入库或标 discard） | 否（保持上一边界状态） | **是，继续** |
| B. 软失败 | 是（低分轨迹可入库，供 replay/负样本） | 是（全同分/全失败 → 随机 winner） | **是，继续** |
| C. 硬失败 | 已产出 traj 按质量单条审查；未产出则无 | 否 | **否，终止本 session 剩余 query** |

理由（多 query 会话）：后续 query 的 prompt 前缀 = session_history（历代 winner 消息），环境 = 上一轮 winner 磁盘状态。A 类环境未坏只是没分 → 不 sync 但可继续；B 类仍有可比较轨迹 → 必须 sync（否则 8 槽环境分叉，多轮语义崩坏）；C 类 winner 不存在或环境不可信 → 继续只会制造"假多轮"，应 abort 换新会话。代码参考：`rollout/session_pool.run_session`。

**冷数据制造策略**：

| 情形 | 要求 |
|------|------|
| A / B | 按上表继续；交付时保留 `query_index` 连续 |
| C | 不再跑同会话后续 query；会话标 `session_status: aborted_at_query_k`；可选另起新会话补桶配额 |
| 部分槽崩溃（非 8 槽全灭） | 视为 B：其余槽继续评比；不中止 session |

**待 PoC**：B 类全零分当前定为随机 winner + sync；若 PoC 发现随机 sync 污染严重，可改为"不 sync 但继续"——需改代码并回溯冷数据规则。

---

## 8. 验收清单

**数据侧交付前自检**：

- [ ] 轨迹总数 ≥ 14,000（推荐 20,000）
- [ ] 每桶轨迹数 ≥ 2,000（对照 §7.2 表）
- [ ] Dialogue 桶无单 query 会话；多 query 会话占比 ≥ 60%
- [ ] 每条 trajectory 含 messages + token + logprob + bucket + query_index
- [ ] `reward is null` 的轨迹不计入配额（或单独报表）
- [ ] 硬失败 session 无"abort 后仍交付后续 query"的脏数据
- [ ] `record_id` / `trajectory_id` 无碰撞

**工程侧导入后**：

- [ ] `buffer.stats()` 各桶 `size ≥ q_min`
- [ ] `pytest` 中 buffer 相关用例仍通过
- [ ] 可选：`replay_warmup_size: 32` 时 step 0 的 `effective_replay_batch_size` 已满

**数据来源配比（Phase 0）**：

- [ ] 每条 trajectory 带 `meta.policy` 来源标记（`pi0_27b` = 本地 Qwen3.6-27B on-policy；`gpt5` = sufy `openai/gpt-5` off-policy）——`collect_rollout.py` 已按 `--actor` 自动写入
- [ ] 两来源**分目录**存（`data/rollouts/{local,remote}/`），便于 `warmup_buffer.py --ratio-27b` 按桶内配比混合
- [ ] `warmup_buffer.py` 输出的 `*.manifest.json` 核对：每桶实际 27B/gpt5 条数与目标配比一致（P0-A 1.0 / P0-B 0.0 / P0-C 0.5 / P0-D 0.7 / P0-E 0.3）
- [ ] **红线**：gpt-5 采的轨迹只进 buffer 做 replay，**绝不**拿去 SFT 蒸馏 27B

---

## 9. 联系人

| 事项 | 负责人 |
|------|--------|
| 桶语义 / 配额 / buffer 导入 | @孙豪 |
| 冷数据会话选题与清单 | @吴健 |
| 沙箱跑批、winner-sync、失败 abort | @郑乃榕 |
| judge 打分与 `reward=null` 界定 | @孙豪 + 评测 |
