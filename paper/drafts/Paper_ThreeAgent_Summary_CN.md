# 三 Agent 多轮构造 — 论文向摘要

> **用途**：Method §4.5 / 贡献 C5 的**独立可读摘要**，供写稿、答辩、对外讲解。完整散文见 [`Paper_Method_draft_CN.md`](Paper_Method_draft_CN.md) §4.5；技术细节见 [`UserSim_三Agent架构与技术设计.md`](../../doc/UserSim_三Agent架构与技术设计.md)。
> **写作日期**：2026-06-13

---

## 1. 在全文中的位置

本文把 agentic 持续 RL 拆成两半：**先在源头把 GRPO 信号做干净，再巩固不遗忘**。三 agent 流水线负责**第一半中的多轮信号**：

| 污染来源 | 三 agent 如何回应 |
|---------|------------------|
| 静态多轮 follow-up 的**前提漂移**（query 引用上一轮结果，但 rollout 状态与采集时不一致） | 只保留真实首条 query 作种子；后续 query **在看到 winner 实际状态之后**再生成 → 前提由构造保证成立 |
| Reward hacking（actor 口头宣称完成，实际未交付） | 奖励依据观察 agent 收来的**实际效果**，而非轨迹自述 |
| 自对话**模式坍缩**（follow-up 趋同于"请再优化"） | 42 个会话级固定人设 + 压短轮数 + winner 状态逐轮演化 + **Questioner 4 模型轮换**（每 5 次提问切换，跨厂商输出风格异质性） |

对应论文贡献表述（见 [`Paper_论文向总览.md`](Paper_论文向总览.md) C5）：**模拟用户在线构造多轮 query**，根除静态多轮数据的前提漂移，构成弱对抗自适应课程。

---

## 2. 问题：前提漂移（Premise Drift）

多轮 follow-up $q_{k+1}$ 通常依赖上一轮执行后的状态 $e_k$（"这份 PPT 第 3 页数据错了"）。静态数据集在**采集时**写死 $q_{k+1}$，但训练时 $e_k$ 是策略 $\pi_\theta$ 的随机输出，故

$$\Pr\big[\phi(q_{k+1})(e_k)\big] < 1,$$

且随策略变强而**持续漂移**。后果两类：

1. **错误梯度**：对不存在的问题硬编修复 → 被奖励；或如实指出问题不存在 → 被惩罚。
2. **信号稀释**：前提不成立的样本把环境噪声注入 GRPO 组内归一化，污染 advantage。

**关键澄清（审稿预防）**：回流数据里真实的 $q_2,\ldots,q_K$ 同样是针对*原始*会话执行结果写下的；训练 rollout 中策略对 $q_1$ 的状态已变，复用它们 = 重新引入漂移。故**只保留种子 $q_1$**——真实数据的价值在首条 query 的意图分布，不在与某次执行绑定的后续轮次。

---

## 3. 方法概览：Winner 边界上的三 agent 环

嵌入 **winner 同步**之后（8 槽位级一致起点 → 并行 rollout → 选 winner → sync），每个 query 边界执行：

```text
winner 轨迹 a_t^w + winner 沙箱 e_t^w
        │
        ▼
   【观察 agent】→ 客观报告 R_t = Obs(a_t^w, e_t^w)
        │
        ├──────────────┬──────────────┐
        ▼              ▼              │
   【奖励模型】    【出题 agent】       │
   r_t = Reward   q_{t+1} ~ Q(·|p,R_t,H_t) 或 <end_session>
   (R_t, rubric)  (人设 p 会话级固定)
```

- $p$：42 人设之一，**一会话抽一次、全程不变**。
- $H_t$：会话正史 = 历代 **winner** 消息拼接（与调度语义一致）。
- $R_t$ **一份两用**：打分与出题读同一份证据 → "用来给分的事实"与"用来出题的事实"严格一致。

> **图占位**：Fig 5 — 三 agent 在 winner-sync 边界的协作环（见 `paper/assets/README.md`）。

---

## 4. 三个 Agent 分别做什么

### 4.1 观察 agent（Observer）— 客观取证

| 维度 | 内容 |
|------|------|
| **人设** | **无**（中立、不带用户偏好） |
| **输入** | Winner 的 actor 轨迹 + winner 沙箱（**只读**） |
| **职责** | 读 actor **声明**（"已生成 report.xlsx"）→ **定向**收集中间结果与最终交付物 → 产出结构化客观报告 $R_t$ |
| **输出** | `ObservationReport`：中间项 / 最终项 / actor 自述 / **claim-vs-actual 差异** / 工作区文件树 |
| **设计要点** | ① 观察与立场解耦 ② 由 actor 输出驱动（非固定模板快照），捕获易被后续步骤覆盖的**中间产物** ③ 只观察 **1 个 winner**（与会话正史、8 槽同起点一致；观察 8 槽会重新引入前提漂移） ④ **训练时必须独立于 actor**：线上推理可 actor 自观察（无 reward 信号，无 hacking 风险），但训练中 actor 自观察可学到"声称完成但实际未交付"的策略骗取高分（reward hacking）；独立 Observer 作为第三方取证，结构上杜绝此攻击面 |

### 4.2 出题 agent（Questioner）— 人设化 follow-up

| 维度 | 内容 |
|------|------|
| **人设** | **有**：42 固定人设（职业 / 偏好 / 用户画像 / **观察偏好**：整体 vs 细节、形式 vs 内容） |
| **输入** | $R_t$ + $H_t$ + 人设 $p$（**不直接读沙箱**） |
| **职责** | 以人设视角阅读客观报告，模拟真实用户发起下一条 query，或输出 `<end_session>` |
| **输出** | 下一条 user query，或终止会话 |
| **设计要点** | 偏好只发生在出题侧：同一份 $R_t$ 被不同人设**问出不同侧面** → 对抗模式坍缩 |

### 4.3 奖励模型（Reward）— 观察 grounded 打分

| 维度 | 内容 |
|------|------|
| **人设** | 无（冻结 judge，与策略解耦） |
| **输入** | $R_t$ + winner 轨迹 + ClawEval 同构 rubric |
| **职责** | 依报告中的**实际产出与效果**打分，而非 actor 文字声明 |
| **输出** | $r_t$；与评测同构：$score = s_{safety}\times(0.8\cdot s_{completion}+0.2\cdot s_{robustness})$ |
| **设计要点** | `discrepancies` 字段是 anti-hacking 信号；三方后端独立 env（`OBSERVER_*` / `USERSIM_*` / `JUDGE_*`）抗 self-preference |

---

## 5. 耐心机制：把"放弃"做成用户性格（§3.6.5）

当 winner 轮**失败**（无有效产物 / 观察报告为空 / 提前停止）时，是否要求重做由**人设携带的耐心**决定，而非工程硬编码"≤3 次"。

**两轴解耦**（每个人设自带）：

| 符号 | 含义 | 性格轴 |
|------|------|--------|
| $P_0(p)$ | 初始耐心 | **初始容忍度**——愿不愿给机会 |
| $d_0(p)$ | 基础扣减 | **升级速度**——脾气 / 挫败感涨多快 |

第 $k$ 次失败后耐心指数衰减：

$$P_k = P_0(p) - d_0(p)\,(2^{k}-1).$$

**决策**：以概率 $\mathrm{clip}(P_k,0,1)$ 发出"请重做"类 follow-up，否则 `<end_session>`；$P_k<0$ 时必停。

**为何这样设计（论文可写点）**：

1. **天然有界**：扣减翻倍 → 约 $\log_2(P_0/d_0)$ 次失败后耐心穿零；$P_0\approx1, d_0=0.1$ 时平滑复现常见"≤3 次"，但上限是**涌现的**、随人设可变。
2. **人格可分**：高 $P_0$+低 $d_0$ = 始终耐心；高 $P_0$+高 $d_0$ = 先礼后兵——同一"重试次数"无法表达的差异。
3. **递增挫败感**：几何衰减比线性更符合"越失败越烦"的用户模型。
4. **只管失败路径**：成功轮走正常 follow-up 配额 $K$；重做轮**不计入** $K$。

---

## 6. 防模式坍缩的四机制

| 机制 | 粒度 | 作用 |
|------|------|------|
| **42 人设** $p$ | 会话级固定 | 改变"以谁视角问、强调报告哪一面" |
| **轮数 $K$ 压制** | 会话级 $K\sim U\{1..3\}$ | 截断自回归链；出题 agent 亦可提前结束 |
| **Winner 状态演化** | 轮级天然 | 同人设每轮 $R_t$ 不同 → 条件分布持续变化 |
| **Questioner 模型轮换** | 每 5 次提问切换 | 4 个跨厂商模型（anthropic/claude-sonnet-5 / deepseek/deepseek-v4-pro / qwen/qwen3.7-max / moonshotai/kimi-k2.6）从模型层注入输出风格异质性，与人设正交互补：人设决定"问什么"，模型决定"怎么问" |

合起来构成**弱对抗自适应课程**：策略越强，残留缺陷越细，follow-up 越精细，难度自动跟随能力前沿。

---

## 7. 与静态方案的对比（Method 论证表）

| | 静态多轮 | **三 agent 在线构造（本方案）** |
|--|---------|-------------------------------|
| B/C 类 follow-up | 前提随机失效 | 看结果后生成，前提由构造成立 |
| 策略变强后 | "挑错"类 query 前提成立率下降 | 始终对**当前实际输出**挑刺 |
| 数据增益 | 一会话一条静态链 | 一种子可 rollout 出**不同** follow-up 链 |
| 真实分布 | 全真实但前提错位 | **首条全真实**；follow-up 由 42 人设控制 |

---

## 8. 实现状态（写 Results 前占位）

| 组件 | 状态 |
|------|------|
| `agents/{observer,questioner,reward}.py` + 42 人设 + `PatienceTracker` | ✅ 已实现，mock 单测覆盖 |
| Prompts O3/O4/O6 | ✅ 落盘 `agents/prompts.py` |
| `rollout/simulated_session`（8 槽 + winner-sync + 三 agent） | ✅ Algorithm 1 已实现 |
| 接 verl scheduler + 原生 generate | ⏳ 待集群（Gap D） |
| `scripts/collect_cold.*` | ⚠ **不含三 agent**（单轮冷启动 buffer，见 [`Buffer_冷启动数据需求.md`](../../doc/Buffer_冷启动数据需求.md)） |
| `scripts/collect_rollout.*` | ✅ observer + questioner 多轮采集（本阶段无 reward） |

---

## 9. 文档交叉引用

| 需求 | 文档 |
|------|------|
| Method 完整散文 §4.5 | [`Paper_Method_draft_CN.md`](Paper_Method_draft_CN.md) |
| 英文投稿版 | [`Paper_Method_draft_EN.md`](Paper_Method_draft_EN.md) |
| 技术规格 + 接口契约 | [`UserSim_多轮Query在线生成.md`](../../doc/UserSim_多轮Query在线生成.md) |
| 实现模块与 env | [`UserSim_三Agent架构与技术设计.md`](../../doc/UserSim_三Agent架构与技术设计.md) |
| Winner 同步上下文 | [`Sandbox_管理调度指南.md`](../../doc/sandbox/Sandbox_管理调度指南.md) §3 |
