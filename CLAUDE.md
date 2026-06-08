# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 仓库性质

这是一个**研究设计文档仓库**，不含可执行代码。所有文件为 Markdown 格式的 Continual Learning 设计文档。仓库所有者：@孙豪。

- **无构建/测试/lint 命令** — 仓库内无代码，所有操作均为 Markdown 编辑
- **无 git 仓库** — 文件版本通过手动备份管理，不做 commit/push

## 文档结构与关系

| 文件 | 内容 | 定位 |
|------|------|------|
| `CL_Update_Sunhao.md` | CL 总设计：Loss 公式、Replay Buffer、实验路线、评测指标、参考文献 | **主文档**，其他文档的上下文依赖 |
| `BucketDesign.md` | Replay Buffer 7 桶结构的详细论证（为什么这样分桶、为什么不用难度分桶、quota 推导过程） | CL_Update_Sunhao.md 中 Replay Buffer 部分的完整展开 |
| `BucketDesign_compressed.md` | BucketDesign.md 的精简版，仅保留结论和公式 | 快速查阅版 |
| `ContinualLearning.md` | CL Loop 全流程概述：数据获取、更新策略、工具环境、评测 | 多人协作总览，各模块负责人分工 |
| `ClawEval_Metadata.md` | ClawEval 评测数据集的任务分类、难度分布、工具能力层、模型排名 | 评测基准参考 |

阅读顺序建议：`ContinualLearning.md`（全貌）→ `CL_Update_Sunhao.md`（技术细节）→ `BucketDesign.md`（分桶论证）→ `ClawEval_Metadata.md`（评测数据）。

## 核心设计要点

### CL Loss

$$L_{cl} = \lambda_1 L_{rl} + \lambda_2 L_{kl} + \lambda_3 L_{replay} + \lambda_4 L_{ent}$$

- $L_{reg}$（参数 L2 正则）**弃用**，权重为 0，槽位让给 $L_{ent}$
- $\lambda_4 = 0.001$ **所有 Phase 固定开启**，防 Echo Trap（traj/query=2 下关闭 entropy 会训练崩盘），不参与 ablation
- $L_{kl}$ 用 reverse KL：$D_{KL}(\pi_{new} \| \pi_{ref})$

### Replay Buffer 7 桶结构

桶按**能力/领域**划分，不按难度划分（难度会随模型能力漂移）：

1. Workflow [54] — 多步骤任务组织
2. SysOps [52] — 工具使用与系统操作
3. Dialogue [38] — 多轮交互与状态跟踪
4. Finance [18] — 结构化业务规则
5. Communication [12] — 表达与沟通
6. Knowledge/Analysis [11] — 检索与推理
7. OfficeQA [10] — 办公语境问答（单独成桶，不并入 Knowledge）

**不纳入 multimodal(4)**：模态不同、数据太少、目标不一致。纯文本共 195 任务。

### 关键设计约束

- **桶内淘汰，禁止跨桶挤出** — 保证不同能力不互相侵占
- **Priority 不用 reward 绝对值** — reward 整体上升会系统性淘汰旧轨迹，buffer 退化为滑动窗口
- **Quota 分配**：保底 $q_{min}$ + 平方根加权（$\alpha=0.5$），大桶得更多但不按比例膨胀
- **采样**：两级采样（先采桶→桶内采轨迹），混合 quota 比例 + 均匀

### 实验路线

```
Phase 1 (B1) → Phase 2 (K1-K5, K2-R) ──┐
                                          ├── Phase 4 (C1-C4) → Phase 5 (S1, S2) → Phase 6 (X1-X7)
                    Phase 3 (R0-R6) ─────┘
```

共 20 个独立训练，单实验 ~16 GPU-day (8×H100, ~100 step)。

### $L_{replay}$ 权重公式

$$w_t^{(i)} = \text{normalize}\Big(\text{clip}\big(\text{priority}_i \cdot \gamma^{\text{block}(t)} \cdot \beta_{\text{type}(t)},\; q_5,\; q_{95}\big)\Big)$$

三维度：Priority（trajectory 级）× 块位置衰减（$\gamma$=0.97, $K$=20）× final_answer boost（$\beta_{\text{final}}$=2.0）。实验对比 W0（均权）vs W2（主方案）。

## 术语与缩写

| 术语 | 含义 |
|------|------|
| CL | Continual Learning |
| Echo Trap | 多轮 agent RL 中因策略坍缩导致训练崩溃的现象（参考文献 B4） |
| traj/query | 每 query 的 rollout 轨迹数，当前方案为 2（原 8） |
| CLEAR | Rolnick et al. 2019 的 experience replay 基线方法（参考文献 A2） |
| verl | 使用的 RL 训练框架 |
| GRPO | 使用的 RL 算法 |
| $\pi_0$ | 初始策略（模型初始 checkpoint） |
| $\pi_{t-1}$ | 上一阶段 checkpoint 策略 |
| hard floor | 每桶不可跌破的最低配额 $q_{min}$ |
| soft target | 按公式计算的桶目标配额 |

## 评测基准

使用 **ClawEval**（300 任务，3 split：General 161 / Multimodal 101 / Multi-turn 38）。评分公式：$score = s_{safety} \times (0.8 \cdot s_{completion} + 0.2 \cdot s_{robustness})$，Pass³ 标准（三次独立运行全通过）。当前仅使用纯文本任务（195 个）。

## 分工

| 模块 | 负责人 |
|------|--------|
| CL 更新策略 / 调研 | @孙豪 |
| 用户数据获取 | @吴健 |
| 工具环境（Agent Framework） | @郑乃榕 |
| 评测 | @杨益博 |
