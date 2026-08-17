# 参考文献（论文/学位论文单一信源）

> 本文件是论文（`paper/`）与学位论文（`master-thesis/`）参考文献的**唯一信源**。
> 新增引用前先在此处更新筛选，确认无误后再同步到 `paper/refs/references.bib` 与 `master-thesis/latex/ref.bib`。
> 旧 bib 曾因重复 arXiv ID（A1/A2 同 1811.11682、B2/B6 同 2507.05386、B4/C1 同 2504.20073、A6/B5 同 2601.05787）污染引用，已清理。

## 一、外部论文引用（已入 bib）

| key | 论文 | 作者 | 年份 | arXiv | 用途 |
|-----|------|------|------|-------|------|
| A2 | Experience Replay for Continual Learning (CLEAR) | Rolnick et al. | 2019 | 1811.11682 | 经验回放基线、R0 对照 |
| A9 | Prioritized Generative Replay | Wang et al. | 2023 | 2311.11557 | 优先级回放（相关工作引用） |
| A11 | Overcoming catastrophic forgetting (EWC) | Kirkpatrick et al. | 2017 | PNAS | 参数空间正则对照 |
| A12 | Continual Learning Through Synaptic Intelligence (SI) | Zenke et al. | 2017 | 1703.04200 (ICML) | 参数空间正则对照（与 EWC 并列） |
| B2 | RFT Naturally Mitigates Forgetting | Lai et al. | 2026 | 2507.05386 | RFT vs SFT 遗忘对比 |
| B4 | Multi-Turn RL for LLM Agents: Echo Trap | Wang et al. | 2025 | 2504.20073 | 策略多样性坍缩、熵正则动机 |
| B7 | It Takes Two: Your GRPO Is Secretly DPO | Wu et al. | 2025 | 2510.00977 | GRPO 理论联系 |
| B8 | DeepSeekMath (GRPO 原论文) | Shao et al. | 2024 | 2402.03300 | GRPO 提出出处、策略梯度基线 |
| C1 | CGL: Advancing Continual GUI Learning via RFT | Yao et al. | 2025 | 2504.20073 | GUI 持续学习对照 |
| D1 | Gradient Episodic Memory (GEM) | Lopez-Paz & Ranzato | 2017 | 1706.08840 | 灾难性遗忘经典 |
| E1 | HybridFlow (verl 框架) | Sheng et al. | 2024 | 2409.19256 | 零框架改动注入的底层 RL 框架 |

## 二、候选论文（调研背景，未入 bib）

A1/A3/A4/A5/A6/A7/A8/A10、B1/B3/B5/B6、C2/C3（保留在 `doc/source/CL_Design.md` 调研表，论文如需引用先在此处登记用途再入 bib）

## 三、本地技术参考文档（设计信源）

> 旧架构文档已删除（`paper/refs/` 下 Observer/RolloutCollect/vllm/汇报/集群推理采集等 5 篇，含 27B/7桶/两阶段/多轮 rollout 等已废弃思路），不再列在这里。

### 核心算法与设计

| 文件 | 内容 |
|------|------|
| `doc/source/CL_Design.md` | 主设计文档：CL Loss / 9桶 Buffer / 实验路线 / 评测 |
| `doc/source/BucketAlgorithm.md` | 9桶 Replay Buffer 算法规范（单一权威信源） |
| `doc/source/ClawEval_Metadata.md` | ClawEval 评测基准（195 纯文本任务, Pass^3） |
| `doc/source/训练与推理流程.md` | 训练循环 + 推理全链路 |
| `doc/重构指南.md` | 当前算法方向（safety 0/1、桶内 uniform+FIFO、U形暂关） |

### 系统实现与操作

| 文件 | 内容 |
|------|------|
| `doc/ops/reward.md` | 奖励设计（4维聚合、safety 二元门控） |
| `doc/ops/Migration_64GPU.md` | 集群迁移与冷启动步骤 |
| `doc/ops/sandbox/接口使用_Sandbox与三Agent.md` | 沙箱接口 + 三 Agent |
| `doc/ops/sandbox/Sandbox_冒烟指南.md` | 沙箱操作入口 |

### 评测与实验

| 文件 | 内容 |
|------|------|
| `doc/eval/防遗忘评测方案.md` | 按桶分组训练 + 统一评测 + 权重后置 |
| `doc/expr/README.md` | 实验档案：算法迭代 + reward judge 选型 |

### 数据

| 文件 | 内容 |
|------|------|
| `datasets/cold_start/cold_start_1429.jsonl` | 冷启动完整轨迹（1429条, 已归一 reasoning_content） |

## 四、引用规范

- 学位论文用 `\upcite{key}`（上标引用）
- 英文 Paper 用 `\cite{key}`
- bib 两处同步：`paper/refs/references.bib` 与 `master-thesis/latex/ref.bib`
- **禁止**直接编辑 bib 加新条目——先改本文件，再同步
