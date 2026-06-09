# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 仓库性质

Continual Learning over Agentic LLM 的训练项目。仓库所有者：@孙豪。

- **设计文档**：全部位于 `doc/`，是项目的需求与设计依据
- **代码骨架**：`replay_buffer/`、`trainer/`、`configs/`、`eval/`、`scripts/`、`tests/`
- **训练框架**：[verl](https://github.com/volcengine/verl)（pip 安装，不 fork，详见 `doc/VerlIntegration.md`）

## 目录结构

```
agentic_cl_research/
├── CLAUDE.md                # 本文件，AI 协作指南
├── README.md                # 项目入口
├── pyproject.toml           # 项目元数据与依赖
├── doc/                     # 设计文档（详见下表）
├── replay_buffer/           # 7 桶 Buffer 实现（与 verl 解耦的纯 Python 模块）
├── trainer/                 # CL Loss 与训练入口（基于 verl，零源码改动）
├── configs/                 # 20 个实验的 yaml
├── eval/                    # ClawEval 评测
├── scripts/                 # 训练 / 评测启动脚本
└── tests/                   # 单元测试
```

## 文档结构与关系

所有设计文档位于 `doc/`：

| 文件 | 内容 | 定位 |
|------|------|------|
| `doc/CL_Update_Sunhao.md` | CL 总设计：Loss 公式、Replay Buffer、实验路线、评测指标、参考文献 | **主文档**，其他文档的上下文依赖 |
| `doc/BucketDesign.md` | Replay Buffer 7 桶结构的详细论证（为什么这样分桶、为什么不用难度分桶、quota 推导过程） | `CL_Update_Sunhao.md` 中 Replay Buffer 部分的完整展开 |
| `doc/BucketDesign_compressed.md` | `BucketDesign.md` 的精简版，仅保留结论和公式 | 快速查阅版 |
| `doc/ContinualLearning.md` | CL Loop 全流程概述：数据获取、更新策略、工具环境、评测 | 多人协作总览，各模块负责人分工 |
| `doc/ClawEval_Metadata.md` | ClawEval 评测数据集的任务分类、难度分布、工具能力层、模型排名 | 评测基准参考 |
| `doc/VerlIntegration.md` | verl 集成指导：是否 fork、Replay Buffer 接入方式、推荐工程结构、风险点 | 实现路径参考 |

阅读顺序建议：`ContinualLearning.md`（全貌）→ `CL_Update_Sunhao.md`（技术细节）→ `BucketDesign.md`（分桶论证）→ `ClawEval_Metadata.md`（评测数据）→ `VerlIntegration.md`（落地工程）。

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

### GPU 部署

**分离 40+24（Deep Research 推荐）**：推理组 40 卡 (5×TP8 vLLM) + 训练组 24 卡 FSDP，流水线化后吞吐 ~10.6 step/hr。Deep Research 中 WebSearch/WebFetch 占交互时间 ~80%，分离模式训练组可利用 CPU 交互空转。比 Colocate 快 3–7%。Colocate 64 为后备（tool exec <2s/turn 时切回）。详见 `doc/CL_Update_Sunhao.md` § GPU 资源分配与训练流水线。

### $L_{replay}$ 权重公式

$$w_t^{(i)} = \text{normalize}\Big(\text{clip}\big(\text{priority}_i \cdot \frac{\gamma^{\text{block}(t)} + \delta^{K_i - \text{block}(t)}}{2},\; q_5,\; q_{95}\big)\Big)$$

两维度：Priority（trajectory 级）× **U 形块权重**（首尾两端高、中间低；起步 $\gamma=\delta=0.88$）。块按**动作块**（`<think>` / `<toolcall>` / `<observation>` / `<final_answer>` 等结构标签）划分，$K_i$ 因 trajectory 而异——具体标签集合与切分规则待数据到位后定，代码 fallback 用等长 $K=20$。Phase 3 对照 W0（均权）vs W2（主方案）。

> **2026-06-08 反转**：原方案为单调块衰减 + final_answer boost；改为 U 形是因为"末端的重要性不止 final_answer 一个 token 段，靠近末端的多个块都重要"，单点 boost 抓不住整段。详见 `doc/CL_Update_Sunhao.md`。

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
