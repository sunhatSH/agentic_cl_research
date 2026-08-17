# 模拟用户在线生成多轮 Query（User-Sim Session Construction）

> **定位**：多轮训练数据的构造方案——真实回流数据只保留会话首条 query 作种子，后续 query 由**三个协作 agent**（观察 / 出题 / 奖励）在 rollout 运行时、观察 winner 状态后在线生成与评分。
> **状态**：设计已定稿；三 agent 代码已落盘 `agents/`（2026-06-12）。**速查**：论文向 [`Paper_ThreeAgent_Summary_CN.md`](../../paper/drafts/Paper_ThreeAgent_Summary_CN.md)；技术汇报向 [`UserSim_三Agent架构与技术设计.md`](UserSim_三Agent架构与技术设计.md)。
> 调度机制见 [`Sandbox_管理调度指南.md`](../ops/sandbox/Sandbox_管理调度指南.md)；reward 见 `trainer/model_reward.py` 与本文 §6。

**写作日期**：2026-06-11（@孙豪 方案定稿）；2026-06-12 升级为观察/出题/奖励三 agent 架构

---

## 1. TL;DR

| 维度 | 决策 |
|------|------|
| 多轮数据来源 | 真实回流数据**只取每会话第一条 query**（保真实分布锚点），其余丢弃 |
| 三 agent 架构 | **观察 agent**（无人设，从 actor 输出判断要收集什么，产出客观状态报告）→ 同时喂给 **出题 agent**（有人设，生成下一 query）和 **奖励模型**（按报告+判分准则给 reward）。**观察 agent 必须独立于 actor**——线上推理时 actor 可自观察（无 reward 信号，无 hacking 风险），但训练中若 actor 自观察则可学到"声称完成但实际未交付"的策略来骗取高分（reward hacking）；独立 Observer 用不同模型做第三方取证，从结构上杜绝此攻击面 |
| 后续 query | 出题 agent 在 **每条 query 的 winner 选出并 sync 之后**，基于观察 agent 对 **winner 单一状态** 的报告在线生成 |
| 观察范围 | **只观察 winner**，不观察 8 个槽（与"会话正史 = winner 轨迹拼接"自洽，省 7 份观察成本） |
| 出题人设 | **42 个固定人设**（职业 / 偏好 / 用户画像 / 观察偏好=整体vs细节、形式vs内容），每会话**随机抽 1 个**，会话内保持一致 |
| 防模式坍缩 | ① 模拟轮数压小（follow-up 1–3 轮，抽样）② 42 人设会话级随机 ③ winner 状态逐轮演化（同人设每轮看到的报告不同）④ **Questioner 多模型轮换**（4 个跨厂商模型每 5 次提问切换，从模型层面注入输出风格异质性） |
| Reward | 模型 judge，走 **外部 OpenAI 兼容 API**（@孙豪 提供，env 注入，见 §6）；**以观察报告为评分证据**，判分 prompt 暂空 |
| 实现进度 | 文档先行；三 agent **暂不写代码** |

---

## 2. 问题定义：静态多轮数据的前提漂移（Premise Drift）

### 2.1 形式化

一条多轮会话 $S = (q_1, q_2, \dots, q_K)$，其中 $q_{k}$（$k \ge 2$）通常**引用** $q_{k-1}$ 执行后的结果（"这个 PPT 的第 3 页数据有问题"）。记：

- $e_k$ = 执行 $q_k$ 后的环境状态（沙箱磁盘 + 产物 + 对话历史）
- $\phi(q_{k+1})$ = $q_{k+1}$ 成立所需的**前提谓词**（如"PPT 存在且第 3 页有数据错误"）

静态数据集把 $q_{k+1}$ 在**数据收集时**写死，但训练 rollout 中 $e_k$ 是策略 $\pi_\theta$ 的随机函数。于是出现**前提漂移**：

$$\Pr\big[\phi(q_{k+1})(e_k) = \text{true}\big] < 1$$

且该概率随 $\pi_\theta$ 演化而漂移（模型变强 → "你做错了 xxx"类前提成立率下降）。前提不成立时有两类训练污染：

1. **错误梯度**：模型对不存在的"问题"硬编一个修复 → 被 judge 奖励 → 训练出幻觉迎合；或如实说"没有这个问题" → 被静态 checker 判失败 → 惩罚诚实行为。
2. **信号稀释**：前提不成立的样本在组内引入与策略无关的 reward 方差，污染 GRPO 的组内归一化（与环境噪声污染 advantage 同构，参见调度指南 §3）。

### 2.2 依赖强度分级（用于论文中刻画问题谱系）

| 类型 | follow-up 依赖 | 前提成立概率 | 静态数据可行性 |
|------|---------------|------------|---------------|
| A | 仅依赖"交付物存在" | 高 | 可行 |
| B | 依赖结果的具体性质/缺陷 | 低且随策略漂移 | **不可行**（本方案动机） |
| C | 依赖中间产物 | 极低 | 不可行 |

本方案对 B/C 类的回答是：**不再让数据假设结果，而是让"用户"看到结果之后再发问**——前提由构造保证成立（grounded by construction）。

---

## 3. 方法：模拟用户在线会话构造

### 3.1 数据侧：只保留种子 query

`datasets/queries.jsonl` 每行（一个会话）只消费 `queries[0]`：

```text
旧:  {queries: [q1, q2, ..., qK]}     全部静态执行（前提漂移）
新:  {seed_query: q1}                  q1 = 真实回流 query（真实分布锚点）
                                       q2..qK 运行时由模拟用户生成
```

- 丢弃发生在**数据加载层**，原始 jsonl 不动（真实 follow-up 保留备查，亦是 O2 归纳 42 人设的原料，见 §10.1）。
- **为什么不直接用回流数据里真实的 $q_2..q_K$**（关键，预防审稿/质疑）：那些真实后续 query 本身就是采集时针对*原始*会话执行结果写下的 follow-up；训练 rollout 中我们的策略对 $q_1$ 产出的状态与原始结果不同，这些真实 follow-up 的前提同样不成立——直接复用 = 重新引入前提漂移（§2）。故只保留种子 $q_1$。
- 设计含义：**真实数据贡献"用户会发起什么任务"的分布；出题 agent 贡献"用户看到结果后会怎么跟进"的分布**。前者难合成（真实意图分布），后者难预录（依赖随机执行结果）——按各自比较优势分工。回流数据的价值在其**首条 query 的真实意图分布**，而非与某次特定执行绑定的后续轮次。

### 3.2 运行时：三 agent 协作的会话流水线

嵌入 `SessionSandboxPool` 会话循环（`rollout/session_pool.py`），插入点 = `sync_to_winner` 之后。每个 query 边界由**观察 → 出题 / 奖励**两步构成：

```text
母版 → 派生 8 槽（位级同起点）；会话开始随机抽 1 个出题人设 p ∈ {42}
  │
  ├─ q1（真实种子）: 8 路并行 rollout → 选 winner → sync 8 槽
  │
  ├─ 【观察 agent】（无人设）                                  ← 新增环节①
  │     读 actor(winner) 输出 → 判断本任务要收集什么
  │     （尤其中间结果 + 最终结果）→ 产出客观状态报告 R_t
  │        │
  │        ├──→ 【奖励模型】 reward(R_t, 轨迹, 判分准则)        ← 新增环节②（评分）
  │        │
  │        └──→ 【出题 agent】（人设 p） q_{t+1}(R_t, H_t)      ← 新增环节③（出题）
  │                 → 下一 query，或 <end_session>
  │
  ├─ q2: 8 路并行（同起点 = winner 状态 + winner 正史）→ winner → sync
  │
  └─ ≤ K_max 轮后 destroy_all（母版不回写，规则 2 不变）
```

形式化。第 $t$ 轮 winner 选定并 sync 后，先由观察 agent 产出报告

$$R_t = \mathrm{Obs}\big(a_t^{w},\; e_t^{w}\big)$$

其中 $a_t^w$ = winner 这条 actor 轨迹的输出（观察对象由它**驱动**——actor 说"已存到 report.xlsx"，观察 agent 就去读 report.xlsx），$e_t^{w}$ = winner 沙箱环境。报告 $R_t$ **一份两用**：

- 评分：$r_t = \mathrm{Reward}\big(R_t,\; a_t^{w},\; \text{rubric}\big)$
- 出题：$q_{t+1} \sim Q\big(\cdot \,\big|\, p,\; R_t,\; H_t\big)$

$p$ = 会话级随机人设（42 选 1），$H_t$ = 会话正史（历代 winner 消息，即现有 `session_history`）。

**前提成立性由构造保证**：$q_{t+1}$ 是在观察 agent 看过 $e_t^w$ 之后生成的，$\Pr[\phi(q_{t+1})(e_t^w)] \approx 1$（残余误差仅来自观察不完整与出题 agent 幻觉，见 §9 风险）。

### 3.3 三个 agent 的职责边界（核心结构）

| | **观察 agent (Observer)** | **出题 agent (Questioner)** | **奖励模型 (Reward)** |
|---|---|---|---|
| 人设 | **无**（客观中立） | **有**（42 选 1，会话级固定） | 无（冻结 judge） |
| 输入 | winner 的 actor 输出 + winner 沙箱 | 观察报告 $R_t$ + 会话正史 $H_t$ + 人设 $p$ | 观察报告 $R_t$ + winner 轨迹 + 判分准则 |
| 职责 | 从 actor 输出**判断本任务要收集什么**，主动收集**中间结果 + 最终结果**，产出结构化客观报告 | 以人设视角（含观察偏好）阅读报告，模拟真实用户发起下一 query 或结束 | 依报告中的**实际产出与详细效果** + rubric 打 reward |
| 输出 | 状态报告 $R_t$（喂给出题 + 奖励两方） | 下一 query 文本 / `<end_session>` | reward 标量（+ 附加指标） |
| 沙箱权限 | **只读**（在存活 winner 上跑只读命令翻产物） | 无（不碰沙箱，只读报告） | 无（只读报告） |

设计要点：

1. **观察与立场解耦**：观察 agent 只负责"客观把证据收全"，不带任何用户偏好；偏好只发生在出题 agent 一侧（人设的观察偏好决定它**强调**报告里的哪部分）。好处：奖励模型与出题 agent 看到的是**同一份中立证据**，评分不被某个用户人设的主观视角污染。
2. **观察由 actor 输出驱动**：不是套固定模板抓快照，而是让观察 agent 读 actor 自己的声明（"我生成了 X / 中间算出了 Y"）去定向收集——这样**中间结果**（易被后续步骤覆盖删除）也能在被丢弃前被捕获，正好接住依赖强度 C 类（§2.2）。
3. **一次观察、两个下游**：报告 $R_t$ 同时供评分与出题，省一次重复观察，且保证"用来给分的事实"和"用来出题的事实"严格一致。
4. **奖励落到实处**：reward 不再只读轨迹文本，而是读观察 agent 收来的**实际效果**（文件真生成了没、内容对不对），抗"嘴上说做完了"的 reward hacking。

### 3.4 为什么只观察 winner（而非 8 个槽）

这不仅是省成本的优化，而是**与会话语义一致性的要求**：

1. **正史一致**：调度指南 §3① 已定义"会话正史 = 历代 winner 轨迹拼接"，输家轨迹不进会话谱系。出题 agent 是会话中的"另一方"，它能看见的世界**只能是 winner 这条世界线**——看输家状态生成的 query 会引用不存在于正史的产物，等于重新引入前提漂移。
2. **同起点不破坏**：q_{t+1} 的 8 槽起点 = sync 后的 winner 状态。出题基于该状态，恰好保证"题目与 8 槽共同起点匹配"，GRPO 组内比较仍然干净。
3. **成本**：观察 1 份而非 8 份；且 7 个输家在 sync 时即销毁，观察它们毫无用处。

### 3.5 防模式坍缩：42 人设 + 轮数压制 + 状态演化

LLM 自我对话的已知失效模式是**模式坍缩**——follow-up 趋同于少数模板腔（"请再优化一下"），熵随轮数衰减。三个机制对抗它：

| 机制 | 粒度 | 内容 | 作用机理 |
|------|------|------|---------|
| **42 人设随机** $p$ | 会话级（会话内固定） | 每个人设 = 职业 + 偏好 + 用户画像 + **观察偏好**（整体 vs 细节、形式 vs 内容）。每会话从 42 个中随机抽 1 个（见 §7 / O2）。观察偏好让同一份客观报告被不同人设**问出不同侧面** | 改变"以谁的视角问、强调结果的哪一面"的先验 |
| **轮数 $K$ 压制** | 会话级 | follow-up 轮数抽样（暂定 1–3 均匀，期望 2；非固定值），出题 agent 亦可提前 `<end_session>` 自然终止 | 截断自回归生成链——链越长越易漂进模板腔；轮数随机化同时避免模型学到"固定第 N 轮结束"的捷径 |
| **winner 状态逐轮演化** | 轮级（天然） | 每轮 winner 状态都被上一条 follow-up 改变，观察报告 $R_t$ 随之不同 | 即使人设固定，每轮报告不同 → 出题条件分布天然变化，不靠重抽视角 |
| **Questioner 多模型轮换** | 轮级（每 5 次提问切换） | Questioner 在 4 个跨厂商模型（anthropic/claude-sonnet-5 / deepseek/deepseek-v4-pro / qwen/qwen3.7-max / moonshotai/kimi-k2.6）之间轮换，每 5 次 `chat()` 调用后切换到下一个 | 从**模型层面**注入输出风格异质性——不同厂商的训练数据、对齐方式、语言风格各异（一个偏正式、一个偏口语、一个爱追问细节、一个爱抓格式问题），轮换后 follow-up 的关注点、语气、角度自然分散 |

> **与旧版差异**：原设计把"观察视角"作为**轮级**单独抽样的维度；现版把观察偏好并入**会话级人设**（点 4 要求），轮内多样性改由 winner 状态演化（机制三）提供。这样观察 agent 保持无人设、客观，视角偏好只存在于出题 agent 一侧。

附加机制（实现期可选）：出题 agent 采样温度调高；对同 batch 生成的 query 做 n-gram / embedding 去重监控（只监控告警，不在线拒绝，避免引入选择偏置，见 §8.1）。

### 3.5.1 Questioner 多模型轮换：设计动机与实现

**问题**：单一 LLM 反复生成 follow-up query，即使 temperature 设为 0.9，输出分布的"中心"仍是同一个——在同一语义空间内抖动，不会跳到另一个关注维度。模型会陷入固定的提问模式（如总问"能加个总结吗"、总挑同一类错误），导致训练数据中 follow-up query 多样性不足，policy 学到的多轮能力单一化。

**方案**：Questioner 使用 `RotatingChatClient`（`agents/base.py`），在 4 个跨厂商模型之间轮换——每 5 次 `chat()` 调用后自动切换到下一个模型，循环往复。轮换池选择不同厂商的理由是**最大化输出风格的异质性**：不同模型的训练数据、对齐方式和语言风格天然不同，这种"模型层面的多样性"是 temperature 调高无法提供的——两者是正交的抗坍缩维度。

**四维抗坍缩的互补关系**：

| 维度 | 粒度 | 抗坍缩机制 | 与其他维度的正交性 |
|------|------|-----------|-------------------|
| 人设随机 $p$ | 会话级 | 改变"以谁的视角问" | 人设改变关注侧面，但不改变模型的表达风格 |
| 轮数 $K$ 压制 | 会话级 | 截断自回归生成链 | 不涉及生成质量 |
| 状态演化 | 轮级 | 每轮输入不同 | 不涉及生成模型 |
| **多模型轮换** | 轮级 | 改变"用什么风格问" | 人设决定问什么，模型决定怎么问——正交互补 |

**为什么是 5 次切换**：不会太频繁（每次换模型破坏上下文连贯感），也不会太稀疏（一个模型连出 5 条已足够形成局部风格、又不至于整 session 被同一模型主导）。

**实现**：`USERSIM_ENDPOINTS` 环境变量（JSON 数组），`USERSIM_ROTATE_EVERY=5`。不设置时回退到单模型（`USERSIM_API_BASE/MODEL/KEY`），完全向后兼容。详见 `doc/source/usersim.md` §2 / `doc/ops/sandbox/接口使用_Sandbox与三Agent.md` §2.4。

### 3.6.5 失败兜底：人设耐心机制（替代固定 ≤3 上限）

当某轮优胜 response **失败 / 提前停止 / 未做完**（无有效产物）时，是否让模型重做由**人设携带的耐心值**决定——把"放弃"做成模拟用户的属性，而非写死的工程上限。

**机制**：每个人设带两个失败相关字段（均人设级、加载时读入、数值留空，见 §7.5）：初始耐心 $P_0(p)$ 与基础扣减 $d_0(p)$。$d_0$ 也由人设携带（而非全局常数），用于解耦两个独立人格轴——$P_0$ = 初始容忍度（愿不愿给机会），$d_0$ = 挫败升级速度（脾气）；无显式 $d_0$ 时回退到全局默认。会话内第 $k$ 次失败（$k=1,2,\dots$）时耐心**指数增长地衰减**：

$$P_k = P_{k-1} - d_0(p)\cdot 2^{\,k-1} \;=\; P_0(p) - d_0(p)\,(2^{k}-1).$$

> 例（$d_0=0.1$）：第 1 次失败扣 0.1，第 2 次扣 0.2，第 3 次扣 0.4 …… 累计扣减 $0.1,0.3,0.7,1.5,\dots$
>
> 两轴示例：高 $P_0$+低 $d_0$=好脾气的耐心用户；高 $P_0$+高 $d_0$=先礼后兵（起初客气、一旦不行迅速失去耐心）；低 $P_0$+低 $d_0$=期待低但不计较。

**决策**（第 $k$ 次失败、算出 $P_k$ 后）：

- $P_k < 0$ → 终止会话（`<end_session>`）；
- $P_k \ge 0$ → 以概率 $p_{\text{重做}}=\mathrm{clip}(P_k,0,1)$ 要求重做，否则终止。

两支可统一写成"以 $\mathrm{clip}(P_k,0,1)$ 概率重做"（$P_k<0$ 时 clip 为 0 即必停）。

**设计性质**：

- **天然自封顶**：扣减翻倍 → 耐心穿过 0 仅需约 $\log_2(P_0/d_0)$ 次失败。取 $d_0=0.1,P_0\approx1$ 时约 ≤3 次，**平滑复现原"≤3 硬上限"，但上限是涌现的、且随人设可变**。
- **人设可分（两维）**：$P_0$ 与 $d_0$ 控制不同性格轴——$P_0$ 调"起步给不给机会"，$d_0$ 调"崩得多快"。二者非完全正交（重试次数 ≈ $\log_2(P_0/d_0)$，单次继续概率看 $P_k$ 绝对值），设值时按"(初始容忍, 升级速度)"语义来想。
- **递增挫败感**：翻倍扣减刻画"越失败越烦"。
- **只管失败路径**：成功轮不消耗耐心，照常走 $K$；重做轮不计入 $K$ 的 follow-up 配额。
- **实现注记**：随机决策用会话级 seeded RNG（与 `session_pool` 一致）保证可复现。

### 3.6 与现有训练链路的耦合（全部在已有轨道内）

| 模块 | 影响 |
|------|------|
| GRPO / winner-sync | **零改动**。三 agent 只在 query 边界工作，8 路同起点性质不变 |
| 会话正史 | 复用 `session_history` 机制；出题 agent 生成的 query 以 user 消息身份进正史 |
| 入桶 | 不变。per-query 入桶 + `<task_domain>` 标签（`SandboxRollout.md` §5.5），生成 query 同样适用，同会话不同桶合法 |
| Reward | q1 可保留 convert 时提取的静态 checker；生成 follow-up **只能走模型 judge**（无预录 ground truth），且以观察报告为证据，见 §6 |
| batch 换算（Gap F） | 每会话 query 数 = $1 + K$，$K$ 为随机变量 → `gen_batch_size` 换算用 $\mathbb{E}[1+K]$（K~U{1..3} 时期望 3 条/会话） |
| 母版 / 会话销毁 | 不变（规则 2：会话结束销毁、不回写母版） |

---

## 4. 与静态方案的对比（论文 Method 部分的论证素材)

| | 静态多轮（baseline） | 前提门控+截断 | **在线模拟用户（本方案）** |
|--|---------------------|--------------|--------------------------|
| B 类 follow-up（挑刺/修改） | 前提随机失效 | 失效时只能丢弃 → B 类覆盖率低 | 看结果后生成，前提由构造成立 |
| 数据复用率 | 整会话绑定 | 截断浪费尾部 | 一条种子可重复 rollout 出**不同**会话（每次 winner 不同 → follow-up 不同），数据增益 |
| 真实分布保真 | 全真实但前提错位 | 同左 | **首轮全真实**；follow-up 分布由 42 人设库控制（保真度风险见 §9） |
| 对抗策略漂移 | 模型变强 → "挑错"类 query 前提成立率持续下降 | 同左（截断率上升） | 出题 agent 始终对观察 agent 报告的**当前策略实际输出**挑刺 → 难度自适应跟随策略演化（curriculum 效应） |

最后一行值得在论文里单独强调：在线出题构成一个**弱对抗的自适应课程**——策略越强，残留缺陷越细微，出题 agent 挑出的刺也越细，follow-up 难度自动跟随能力前沿，无需人工设计难度调度。

---

## 5. 会话构造算法（伪代码，论文 Algorithm 1 底稿）

```text
Algorithm 1: User-Sim Session Rollout（单会话，三 agent）
输入: 种子 q1（真实回流）、母版 M、42 人设库 P、K_max
 1:  p ~ P；K ~ U{1..K_max}                       # 会话级抽样（出题人设 + 轮数）
 2:  slots ← spawn(M, 8)                          # 位级同起点
 3:  H ← []；q ← q1
 4:  for t = 1, 2, ... do
 5:      T ← parallel_rollout(slots, q, H)         # 8 条轨迹
 6:      w ← select_winner_with_fallback(T)        # 先按已有 reward 选 winner（含兜底）
 7:      if w = None: buffer←T; continue/终止       # 沿用现行 scorer-error 兜底
 8:      sync_to_winner(w)；H ← H ∥ T[w].messages
 9:      R_t ← Observer(actor=T[w], sandbox=winner) # 观察 agent：客观报告（中间+最终结果）
10:      r ← Reward(R_t, T[w], rubric)              # 奖励模型：以报告为证据打分（§6）
11:      buffer ← T（全部 8 条，per-query 入桶；winner 用 r）
12:      if t > K: break
13:      q ← Questioner(p, R_t, H)                  # 出题 agent：人设视角生成下一 query
14:      if winner response 失败/停止/未完成:        # 兜底（决策 #10，耐心机制）
15:          P ← P - d0(p) * 2^(fail_cnt)；fail_cnt += 1  # d0 亦人设自带，指数扣减
16:          if P < 0 or rand() > clip(P,0,1): q ← <end_session>  # 否则要求重做
17:      if q = <end_session>: break
18:  destroy_all()                                 # 母版不回写
```

> 说明：第 6 行选 winner 仍用现有 reward（q1 用静态 checker / 已有 judge）；第 9–10 行的"观察→奖励"链是新增的、面向**生成 follow-up**的评分路径。两者关系待实现时统一（见 §6：observer 报告是否回灌 winner 选择）。

与现行 `run_session`（`session_pool.py`）的差异在 9–17 行：新增观察 agent（9）、报告驱动的奖励（10）与出题（13），循环驱动从"遍历静态 queries 列表"变成"出题 agent 决定下一条 / 终止"。

---

## 6. Reward：外部 judge API

**决策（2026-06-11，@孙豪）**：reward judge 使用 @孙豪 提供的**外部 OpenAI 兼容 API**，替代原"本地 vLLM 起冻结 judge"的部署方式。

- **机制不变**：`trainer/model_reward.py` 的 `JudgeClient` 本就按 OpenAI 兼容协议 + env 解析设计，**代码零改动**，只换 env 值：

```bash
export JUDGE_API_BASE=<外部 API base url>      # 待 @孙豪 提供后填入
export JUDGE_MODEL=<模型名>                    # 同上
export JUDGE_API_KEY=<key>                     # 同上
```

- `scripts/serve_reward_model.sh`（本地 vLLM 自部署）降级为**备用路径**（外部 API 不可用 / 限流时的 fallback），不删除。
- **一致率校准照旧**：外部 judge 同样要过 `scripts/calibrate_judge.py` 的人工一致率门（Gap A 决策门不变）。
- 评分对象差异：
  - **q1（真实种子）**：静态 checker（convert 时提取）+ judge 混合，按现行 Gap A 设计；
  - **生成 follow-up**：无预录 ground truth，**纯 judge**，且**以观察 agent 报告 $R_t$ 为评分证据**——judge 输入 = (follow-up query, 会话正史 $H_t$, winner 轨迹 $a_t^w$, **观察报告 $R_t$**, 判分准则)。报告里的"实际产出与详细效果"让 judge 据实判分，而非只看 actor 嘴上声明（抗 reward hacking，对应 §3.3 要点 4）。
- **判分 prompt（rubric）暂空**：评分准则的具体 prompt 模板**先留空**（O4），实现时再定（是否沿用 ClawEval 的 $s_{safety}\times(0.8 s_{completion}+0.2 s_{robustness})$、输入裁剪 token 预算等）。`JudgeClient` 接口不变，只是 prompt 体后填。
- **待实现统一**：winner 选择（Algorithm 1 第 6 行，用已有 reward）与 follow-up 评分（第 10 行，用观察报告）目前是两条路径。是否让观察报告也回灌 winner 选择（即第 6 行也走 observer-grounded reward）留待实现时定。
- ⚠️ **同模型耦合风险**：观察 / 出题 / 奖励三方若复用同一外部 API（同一模型既观察、又出题、又阅卷），存在自我偏好（self-preference）偏置。优先**异模型或异 endpoint**，至少 prompt 角色隔离 + env 分离配置（§7.5）。

> 截至本文写作，外部 API 的具体地址/模型/key 尚未入库（仓库与历史会话中均未找到）。**待 @孙豪 提供后填入运行环境**（env 注入，密钥不进 git，与 `tencent.env` 同等待遇）。

---

## 7. 接口契约（暂不实现，施工时按此对齐）

> 本节是后续补码的施工边界。**当前迭代不写观察 / 出题 / 奖励三 agent 的任何代码。**三方边界见 §3.3。

### 7.1 数据 schema（`datasets/queries.jsonl` 加载层）

```json
{
  "session_id": "...",
  "queries": ["q1 真实", "q2 真实(加载时丢弃)", "..."],
  "max_followups": 3
}
```

加载层只取 `queries[0]`；保留原始字段备查（不删原始数据）。**注意**：出题人设**不来自数据**，而是运行时从 42 人设库随机抽（见 §7.5）——数据侧不再带 `persona` 字段。

### 7.2 观察报告（三 agent 共享的中间数据结构）

观察 agent 的输出，**同时**喂给出题 agent 与奖励模型（§3.3 要点 3）。客观、无人设：

```python
@dataclass
class ObservationReport:
    state_diff: str            # 环境 before/after diff（含内容 + SysOps）= ground truth（observer 唯一判据）
    intermediate: list[dict]   # 中间结果: {desc, source(命令/文件), value_excerpt}（从 diff 提取）
    final: list[dict]          # 最终交付物: {path, kind, content_excerpt}（从 diff 提取）
    discrepancies: str         # 状态内部红旗（空/损坏/自相矛盾；可空）
    actor_trajectory: str      # pass-through：actor 轨迹文本，observer 模型不看，仅给奖励模型判 safety/robustness
    file_tree: str             # winner workspace 文件树（深度截断，兜底）
    has_effect: bool           # 本轮 diff 是否非空；False → 短路 reward、走失败/耐心路径
```

采集时机与 `run_checkers` 相同（sync 后、在存活 winner 实例上跑只读命令）。**实现定稿（2026-06-19，diff-driven）**：观察 agent **不读 actor 轨迹**，而是对 winner 沙箱做确定性 **before/after diff**（文件内容 + 二进制提取 + SysOps），以真实状态增量为唯一判据；轨迹仅由观察**组件** pass-through 给奖励模型（observer **模型**永不看，省 token）。反 hacking 由此变为**结构性**——声称从不进入观察判断与 completion。

### 7.3 观察 agent 协议（Observer，无人设）

```python
class Observer(Protocol):
    def observe(
        self,
        actor_trajectory: list[dict],   # winner 这条 actor 的输出（驱动观察）
        sandbox: "ReadOnlySandbox",      # 只读权限的存活 winner 实例
    ) -> ObservationReport: ...
```

### 7.4 出题 agent 协议（Questioner，有人设）

```python
class Questioner(Protocol):
    def next_query(
        self,
        persona: "Persona",              # 会话级随机人设（42 选 1）
        report: ObservationReport,       # 观察 agent 的报告（不直接碰沙箱）
        session_history: list[dict],     # 正史（历代 winner 消息）
    ) -> str | None: ...                 # None == <end_session>
```

奖励模型沿用 `trainer/model_reward.py` 的 `JudgeClient`，新增入参把 `ObservationReport` 拼进 judge prompt（prompt 体 = O4，留空）。

### 7.5 42 人设库 + agent 后端配置（届时）

- 人设库：`agents/personas.json`（42 条纯数据 + 随机抽样函数，与 verl 解耦，可单测）。每条字段：

```python
@dataclass
class Persona:
    name: str
    profession: str       # 职业
    preference: str       # 偏好（关注点 / 容忍度）
    profile: str          # 用户画像（专业度 / 语气 / 耐心的文字描述）
    observation_focus: str  # 观察偏好: 整体|细节 × 形式|内容（决定它强调报告的哪部分）
    patience: float        # 初始耐心 P0（失败兜底用，见 §3.6.5；数值留空待定）
    patience_decay: float  # 基础扣减 d0（脾气/升级速度；无则回退全局默认；数值留空）
```

  - 具体 42 条内容 = O2（延后；起步可从 `docker/sandbox/fs-seeds/` 3 persona 扩到 42）。**观察 agent 无人设，不进此库。**
- `SessionSandboxPool`：`run_session(queries, agent_fn)` 旁增 `run_simulated_session(seed_query, agent_fn, observer, questioner, reward, persona, ...)`（Algorithm 1 的 9–17 行）；现行静态入口保留（兼容/调试用）。
- agent 后端：三方均复用 `JudgeClient` 同款 OpenAI 兼容 HTTP 封装，**各自独立 env**，便于异模型/异 endpoint 隔离（抗 self-preference，§6）：
  - 观察：`OBSERVER_API_BASE` / `OBSERVER_MODEL`
  - 出题：`USERSIM_API_BASE` / `USERSIM_MODEL`
  - 奖励：`JUDGE_API_BASE` / `JUDGE_MODEL`（现有）

---

## 8. 质量保障：过程监控（不增设消融实验）

> **决策（2026-06-11，@孙豪）**：本方案**不单列消融实验**——现有 20 实验路线已饱和，不再为 query 构造扩实验矩阵。方案有效性靠两条保障：① 训练中持续记录的过程指标（在线发现坍缩/幻觉，成本≈0）；② ClawEval Multi-turn split 终点指标。论文中本方案作为**系统设计贡献**呈现，以过程指标 + 终点指标为证据，不做组件级 ablation。

### 8.1 过程指标（训练中持续记录，接 `replay_metrics` 同款 sidecar 思路）

| 指标 | 定义 | 检测什么 |
|------|------|---------|
| 生成 query 多样性 | distinct-n / 批内 embedding 平均成对距离 | 模式坍缩（早期预警） |
| 前提成立率 | 抽样人工/judge 审计：follow-up 引用的事实在观察报告中存在的比例 | 出题 agent 幻觉（§9 残余风险） |
| 观察报告 vs 实际差异率 | `discrepancies` 非空的比例 | 观察 agent 漏收 / actor 虚报 |
| follow-up 类型分布 | 挑错 / 追加 / 追问解释 的占比随训练步演化 | 42 人设是否实际产生多样性 |
| follow-up 组内 reward std | 生成 query 上 GRPO 组内标准差 | 题目是否仍有区分度（std→0 = 太易/太难） |
| 桶分布漂移 | 生成 query 的 `<task_domain>` 分布 vs 种子分布 | 出题 agent 是否把会话拖向少数领域 |

### 8.2 端到端

ClawEval **Multi-turn split（38 任务）**为主要终点指标（方案直接针对多轮能力）；General split 监控无回退。

---

## 9. 风险与边界（论文 Limitations 底稿）

| 风险 | 说明 | 缓解 |
|------|------|------|
| follow-up 分布失真 | 42 人设分布 ≠ 真实用户 follow-up 分布，训练目标有系统偏移 | 首轮保持全真实锚点；42 人设从真实回流 follow-up 中归纳（构建库时用真实数据，运行时不用） |
| 出题 agent 幻觉 | 观察报告不完整 → 出题 agent 引用不存在的细节，前提漂移以小概率回归 | 观察由 actor 输出驱动定向收集（§3.3 要点 2）；前提成立率 + 差异率审计（§8.1） |
| 三 agent 同模型耦合 | 观察/出题/阅卷同模型自我偏好 | 各自独立 env（§7.5），优先异模型/异 endpoint |
| Expert-iteration 偏置 | winner-sync 已知偏置（调度指南 §3②）：模型学不到"从自己的烂摊子恢复"；三 agent 只看 winner 会强化它 | 与现行方案同等接受；若多轮容错差，改 softmax 抽样 sync 目标（既有预案） |
| response 失败/停止/未完成 | 优胜 response 失败、提前停止或任务未做完（含 q1 全失败、观察报告近乎为空） | **已拍板（决策 #10，耐心机制）**：按人设耐心值 $P$ 决定重做/终止——每次失败 $P$ 指数衰减，$P<0$ 必停，否则以 $\mathrm{clip}(P,0,1)$ 概率重做（要求"请重来"，真实用户行为）。耐心翻倍衰减天然自封顶（约 ≤3 次），防失败循环把会话拖垮。详见 §3.6.5 |
| 多 agent 调用延迟 | 每 query 边界多 观察+出题+奖励 三次 LLM 调用，串行计入会话尾延迟 | 16 会话并行天然摊薄；观察报告截断控制下游输入长度；观察一次两用省一次 |

---

## 10. 决策记录

| # | 决策 | 结论 | 日期 |
|---|------|------|------|
| 1 | 多轮 follow-up 来源 | 真实数据只留 q1，其余在线生成（放弃静态 follow-up 主路径，原始数据留存备查） | 2026-06-11 |
| 2 | 观察范围 | 只观察 winner（正史一致性 + 成本） | 2026-06-11 |
| 3 | 防坍缩 | 轮数压小（1–3 抽样）+ ~~画像库（会话级）+ 视角库（轮级）~~ → **修订见 #11** | 2026-06-11 |
| 4 | Reward | @孙豪 提供的外部 OpenAI 兼容 API；机制走现有 `JudgeClient` env 注入；本地 vLLM 降级为 fallback；**API 地址待提供** | 2026-06-11 |
| 5 | 实现节奏 | 文档先行；三 agent 代码暂缓，按 §7 契约后补 | 2026-06-11 |
| 6 | 观察接口 | ~~固定 digest~~ → **修订见 #12**：升级为**观察 agent**（主动、由 actor 输出驱动收集） | 2026-06-11 |
| 7 | **不增设消融实验** | 20 实验已饱和；有效性靠过程监控（§8.1）+ ClawEval Multi-turn 终点指标 | 2026-06-11 |
| 8 | 出题侧安全 | 不在本方案内建设——已有专门 gate 覆盖（O5 关闭） | 2026-06-11 |
| 9 | 种子筛选 + 单/多轮配比 | 划归**数据侧（@吴健）**，本仓库只消费（O1 移交） | 2026-06-11 |
| 10 | response 失败/停止/未完成兜底 | **泛化 + 双参人设耐心机制（2026-06-12）**：任一轮失败时按人设耐心决定，$P_k=P_0(p)-d_0(p)(2^k-1)$；$P_0(p)$=初始容忍、$d_0(p)$=升级速度，**两者均人设自带**（$d_0$ 无则回退全局默认；数值留空）。$P<0$ 必停，否则以 $\mathrm{clip}(P,0,1)$ 概率重做。翻倍衰减天然自封顶（约 ≤3 次），取代原固定硬上限；重做轮不计入 $K$。详见 §3.6.5 | 2026-06-11；2026-06-12 泛化+双参耐心 |
| 11 | **拆为三 agent 架构** | 观察 agent（无人设，从 actor 输出判断收集什么，产出客观报告）→ 同报告喂给出题 agent（有人设，出下一 query）+ 奖励模型（按报告+rubric 打分）。详见 §3.2/§3.3 | 2026-06-12 |
| 12 | 观察升级为 agent | 由"固定 digest"升级为**主动观察 agent**：读 actor 声明定向收集中间+最终结果（O6 重定义为观察 agent 设计） | 2026-06-12 |
| 13 | 出题人设 = 42 固定随机 | 42 个人设（职业/偏好/画像/观察偏好），每会话随机抽 1 个、会话内固定；观察偏好并入人设（取代原轮级视角抽样）。**观察 agent 无人设** | 2026-06-12 |
| 14 | 奖励以观察报告为证据 | reward = f(观察报告, 轨迹, rubric)；据实际产出/效果判分抗 hacking；**判分 prompt 暂空**（O4） | 2026-06-12 |

### 10.1 待设计清单

| # | 待设计项 | 说明 | 归属 / 状态 |
|---|---------|------|------------|
| O1 | 种子筛选标准 + 单/多轮配比 | "可追问性"过滤规则 + 多轮会话与单轮 query 的训练数据配比 | **数据侧（@吴健）负责**；本仓库只消费产出（schema 见 §7.1），Gap B 转换时对接 |
| O2 | **42 人设库的具体条目** | 42 条 persona 内容（字段 schema 见 §7.5），从真实回流 follow-up 归纳 | **延后**，出题 agent 实现前再做；起步可从 3 个 fs-seed persona 扩到 42 |
| O3 | 出题 agent prompt 模板 | 出题 system prompt：人设 + 观察报告注入、`<end_session>` 触发、防"AI 腔" | **留空**（实现时再定） |
| O4 | 奖励判分 rubric prompt | judge 阅卷 prompt：观察报告 + 轨迹的输入裁剪、评分维度是否沿用 ClawEval 公式 | **留空**（实现时再定） |
| ~~O5~~ | ~~生成 query 的安全过滤~~ | **不需要**：已有专门 gate 覆盖出题侧安全（2026-06-11 @孙豪 确认），本方案不重复建设 | 已关闭 |
| O6 | **观察 agent 设计 + 采集细则** | 观察 system prompt、只读命令集、中间结果识别与捕获、file_tree 深度、token 预算 | **留空**（实现时再定） |
| ~~O7~~ | ~~q1 全失败兜底拍板~~ | **已拍板**：泛化为任一轮失败兜底 + 人设耐心机制（指数衰减 + clip 概率重做），见决策 #10 / §3.6.5 | 已关闭 |

---

## 11. 文档索引

| 文档 | 关系 |
|------|------|
| `sandbox/Sandbox_管理调度指南.md` | winner-sync / 正史 / 兜底规则——本方案的插入骨架 |
| `sandbox/SandboxRollout.md` §5.5 | per-query 入桶契约——生成 query 沿用 |
| `Plan_训练链路补齐.md` Gap A | judge reward 决策门——外部 API 同样适用 |
| `sandbox/Sandbox_Agent架构.md` | persona / fs-seeds——42 人设库的初始来源 |
| `CL_Design.md` | CL Loss / 实验路线——本方案产出的轨迹按原路线入桶训练 |
