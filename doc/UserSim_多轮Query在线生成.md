# 模拟用户在线生成多轮 Query（User-Sim Session Construction）

> **定位**：多轮训练数据的构造方案——真实回流数据只保留会话首条 query 作种子，后续 query 由"模拟用户 agent"在 rollout 运行时、观察 winner 状态后在线生成。
> **状态**：设计已定稿，**模拟器与观察 agent 代码暂不实现**（接口契约见 §7，后续按契约补齐）。
> 调度机制见 [`Sandbox_管理调度指南.md`](Sandbox_管理调度指南.md)；reward 见 `trainer/model_reward.py` 与本文 §6。

**写作日期**：2026-06-11（@孙豪 方案定稿）

---

## 1. TL;DR

| 维度 | 决策 |
|------|------|
| 多轮数据来源 | 真实回流数据**只取每会话第一条 query**（保真实分布锚点），其余丢弃 |
| 后续 query | 模拟用户 agent 在 **每条 query 的 winner 选出并 sync 之后**，观察 **winner 单一状态** 在线生成 |
| 观察范围 | **只观察 winner**，不观察 8 个槽（与"会话正史 = winner 轨迹拼接"自洽，省 7 份观察成本） |
| 防模式坍缩 | ① 模拟轮数压小（follow-up 1–3 轮，抽样）② 用户画像库（会话级抽样）③ 观察视角库（轮级抽样：整体/细节、格式/内容等） |
| Reward | 模型 judge，走 **外部 OpenAI 兼容 API**（@孙豪 提供，env 注入，见 §6） |
| 实现进度 | 文档先行；模拟器 / 观察 agent **暂不写代码** |

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

- 丢弃发生在**数据加载层**，原始 jsonl 不动（真实 follow-up 保留备查，亦是 O2 画像/视角库归纳的原料，见 §10.1）。
- 设计含义：**真实数据贡献"用户会发起什么任务"的分布；模拟器贡献"用户看到结果后会怎么跟进"的分布**。前者难合成（真实意图分布），后者难预录（依赖随机执行结果）——按各自比较优势分工。

### 3.2 运行时：模拟器插在 winner-sync 边界

嵌入 `SessionSandboxPool` 会话循环（`rollout/session_pool.py`），插入点 = `sync_to_winner` 之后：

```text
母版 → 派生 8 槽（位级同起点）
  │
  ├─ q1（真实种子）: 8 路并行 rollout → reward → winner → sync 8 槽
  │
  ├─ 【模拟用户 agent】 U_sim(persona, lens_t, H_t, Obs(winner))   ← 新增环节
  │       → 产出 q2 文本，或 <end_session>
  │
  ├─ q2: 8 路并行（同起点 = winner 状态 + winner 正史）→ winner → sync
  │
  ├─ 【模拟用户 agent】 → q3 或 <end_session>
  │
  └─ ≤ K_max 轮后 destroy_all（母版不回写，规则 2 不变）
```

形式化：第 $t$ 轮 follow-up 由

$$q_{t+1} \sim U_{\text{sim}}\big(\cdot \,\big|\, p,\; \ell_t,\; H_t,\; \mathrm{Obs}(e_t^{w})\big)$$

生成，其中 $p$ = 用户画像（会话级抽样），$\ell_t$ = 观察视角（轮级抽样），$H_t$ = 会话正史（历代 winner 消息，即现有 `session_history`），$e_t^{w}$ = 第 $t$ 轮 **winner** 的环境状态，$\mathrm{Obs}$ = 状态摘要算子。

**前提成立性由构造保证**：$q_{t+1}$ 是在看到 $e_t^w$ 之后生成的，$\Pr[\phi(q_{t+1})(e_t^w)] \approx 1$（残余误差仅来自摘要不完整与模拟器幻觉，见 §9 风险）。

### 3.3 为什么只观察 winner（而非 8 个槽）

这不仅是省成本的优化，而是**与会话语义一致性的要求**：

1. **正史一致**：调度指南 §3① 已定义"会话正史 = 历代 winner 轨迹拼接"，输家轨迹不进会话谱系。模拟用户是会话中的"另一方"，它能看见的世界**只能是 winner 这条世界线**——看输家状态生成的 query 会引用不存在于正史的产物，等于重新引入前提漂移。
2. **同起点不破坏**：q_{t+1} 的 8 槽起点 = sync 后的 winner 状态。模拟器基于该状态出题，恰好保证"题目与 8 槽共同起点匹配"，GRPO 组内比较仍然干净。
3. **成本**：观察 1 份而非 8 份；且 7 个输家在 sync 时即销毁，观察它们毫无用处。

### 3.4 防模式坍缩：三维抽样

LLM 自我对话的已知失效模式是**模式坍缩**——follow-up 趋同于少数模板腔（"请再优化一下"），熵随轮数衰减。三个独立的随机化维度对抗它：

| 维度 | 抽样粒度 | 内容 | 作用机理 |
|------|---------|------|---------|
| **用户画像 $p$** | 会话级（会话内保持一致） | 身份 / 专业度 / 语气 / 耐心 / 表达风格。起步复用 `docker/sandbox/fs-seeds/` 的 3 个 persona（finance_analyst / sysops_engineer / office），与沙箱种子文件系统天然对齐（财务画像挑财务产物的刺），后续扩库 | 改变"问什么"的先验 |
| **观察视角 $\ell_t$** | 轮级（每轮重抽） | 整体 vs 细节 / 格式 vs 内容 / 正确性 vs 偏好 / 追加需求 vs 挑错 vs 追问解释 | 同一 winner 状态在不同视角下产生不同 follow-up，制造条件分布的多样性 |
| **轮数 $K$** | 会话级 | follow-up 轮数抽样（暂定 1–3 均匀，期望 2；非固定值），模拟器亦可提前输出 `<end_session>` 自然终止 | 截断自回归生成链——链越长越容易漂进模板腔；轮数随机化同时避免模型学到"固定在第 N 轮结束"的捷径 |

附加机制（实现期可选）：模拟器采样温度调高；对同 batch 生成的 query 做 n-gram / embedding 去重监控（只监控告警，不在线拒绝，避免引入选择偏置，见 §8.1）。

### 3.5 与现有训练链路的耦合（全部在已有轨道内）

| 模块 | 影响 |
|------|------|
| GRPO / winner-sync | **零改动**。模拟器只是 query 边界上多了一个"出题人"，8 路同起点性质不变 |
| 会话正史 | 复用 `session_history` 机制；模拟器生成的 query 以 user 消息身份进正史 |
| 入桶 | 不变。per-query 入桶 + `<task_domain>` 标签（`SandboxRollout.md` §5.5），生成 query 同样适用，同会话不同桶合法 |
| Reward | q1 可保留 convert 时提取的静态 checker；生成 follow-up **只能走模型 judge**（无预录 ground truth），见 §6 |
| batch 换算（Gap F） | 每会话 query 数 = $1 + K$，$K$ 为随机变量 → `gen_batch_size` 换算用 $\mathbb{E}[1+K]$（K~U{1..3} 时期望 3 条/会话） |
| 母版 / 会话销毁 | 不变（规则 2：会话结束销毁、不回写母版） |

---

## 4. 与静态方案的对比（论文 Method 部分的论证素材)

| | 静态多轮（baseline） | 前提门控+截断 | **在线模拟用户（本方案）** |
|--|---------------------|--------------|--------------------------|
| B 类 follow-up（挑刺/修改） | 前提随机失效 | 失效时只能丢弃 → B 类覆盖率低 | 看结果后生成，前提由构造成立 |
| 数据复用率 | 整会话绑定 | 截断浪费尾部 | 一条种子可重复 rollout 出**不同**会话（每次 winner 不同 → follow-up 不同），数据增益 |
| 真实分布保真 | 全真实但前提错位 | 同左 | **首轮全真实**；follow-up 分布由 persona/lens 库控制（保真度风险见 §9） |
| 对抗策略漂移 | 模型变强 → "挑错"类 query 前提成立率持续下降 | 同左（截断率上升） | 模拟器始终对**当前策略的实际输出**挑刺 → 难度自适应跟随策略演化（curriculum 效应） |

最后一行值得在论文里单独强调：在线模拟器构成一个**弱对抗的自适应课程**——策略越强，残留缺陷越细微，模拟器挑出的刺也越细，follow-up 难度自动跟随能力前沿，无需人工设计难度调度。

---

## 5. 会话构造算法（伪代码，论文 Algorithm 1 底稿）

```text
Algorithm 1: User-Sim Session Rollout（单会话）
输入: 种子 q1（真实回流）、母版 M、画像库 P、视角库 L、K_max
 1:  p ~ P；K ~ U{1..K_max}                    # 会话级抽样
 2:  slots ← spawn(M, 8)                       # 位级同起点
 3:  H ← []；q ← q1
 4:  for t = 1, 2, ... do
 5:      T ← parallel_rollout(slots, q, H)      # 8 条轨迹
 6:      r ← judge(T)（+ t=1 时静态 checker）    # §6
 7:      w ← select_winner_with_fallback(r)     # 含同分/全失败兜底
 8:      buffer ← T（全部 8 条，per-query 入桶）
 9:      if w = None: continue/终止（沿用现行兜底）
10:      sync_to_winner(w)；H ← H ∥ T[w].messages
11:      if t > K: break
12:      ℓ_t ~ L                                # 轮级视角
13:      q ← U_sim(p, ℓ_t, H, Obs(slots.winner)) # 只观察 winner
14:      if winner 全失败/无产物:                 # 兜底（决策 #10）
15:          q ← 随机{抱怨要求重做, <end_session>}
16:          if 累计抱怨次数 > 3: q ← <end_session>   # 防崩溃循环
17:      if q = <end_session>: break
18:  destroy_all()                              # 母版不回写
```

与现行 `run_session`（`session_pool.py`）的差异只有 11–17 行：循环驱动从"遍历静态 queries 列表"变成"模拟器决定下一条 / 终止"。

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
  - **生成 follow-up**：无预录 ground truth，**纯 judge**——rubric 即模拟器发出的 query 本身（"用户这条要求是否被落实"），judge 输入 = (follow-up query, winner 同步后的会话上下文, 本轮轨迹, 沙箱产物摘要)。
- ⚠️ **同模型耦合风险**：若模拟器后续也复用同一个外部 API（同一模型既出题又阅卷），存在自我偏好（self-preference）偏置。实现模拟器时优先考虑出题/阅卷用不同模型，或至少 prompt 角色隔离 + env 分离配置（§7.4）。

> 截至本文写作，外部 API 的具体地址/模型/key 尚未入库（仓库与历史会话中均未找到）。**待 @孙豪 提供后填入运行环境**（env 注入，密钥不进 git，与 `tencent.env` 同等待遇）。

---

## 7. 接口契约（暂不实现，施工时按此对齐）

> 本节是后续补码的施工边界。**当前迭代不写模拟器与观察 agent 的任何代码。**

### 7.1 数据 schema（`datasets/queries.jsonl` 加载层）

```json
{
  "session_id": "...",
  "queries": ["q1 真实", "q2 真实(加载时丢弃)", "..."],
  "persona": "finance_analyst | null（null 则运行时从画像库抽）",
  "max_followups": 3
}
```

加载层只取 `queries[0]`；保留原始字段备查（不删原始数据）。

### 7.2 模拟器协议（Python Protocol，纯接口）

```python
class UserSimulator(Protocol):
    def next_query(
        self,
        persona: str,            # 会话级画像
        lens: str,               # 本轮观察视角
        session_history: list[dict],   # 正史（历代 winner 消息）
        winner_obs: "WinnerObservation",
    ) -> str | None: ...         # None == <end_session>
```

### 7.3 winner 观察摘要（固定 digest，首版不给模拟器工具权限）

```python
@dataclass
class WinnerObservation:
    file_tree: str           # winner 沙箱 workspace 文件树（深度截断）
    artifacts: list[dict]    # 主要产物: {path, kind, content_excerpt(截断)}
    last_response: str       # winner 本轮最终回复
    checker_results: dict    # 本轮 checker 结果（若有）
```

采集时机与 `run_checkers` 相同（sync 后、在存活 winner 实例上跑只读命令）。"给模拟器只读工具权限自己翻沙箱"作为升级路径记录，digest 不够用（前提成立率掉，见 §8）时再启用。

### 7.4 `SessionSandboxPool` 改动点（届时）

- `run_session(queries, agent_fn)` 旁增 `run_simulated_session(seed_query, agent_fn, simulator, ...)`（Algorithm 1 的 11–17 行）；现行静态入口保留（兼容/调试用）。
- 画像库 / 视角库：`rollout/user_sim_profiles.py`（纯数据表 + 抽样函数，与 verl 解耦，可单测）。
- 模拟器后端：复用 `JudgeClient` 同款 OpenAI 兼容 HTTP 封装，模型从 `USERSIM_API_BASE`/`USERSIM_MODEL` env 解析（与 judge 隔离配置，即使初期填同一地址）。

---

## 8. 质量保障：过程监控（不增设消融实验）

> **决策（2026-06-11，@孙豪）**：本方案**不单列消融实验**——现有 20 实验路线已饱和，不再为 query 构造扩实验矩阵。方案有效性靠两条保障：① 训练中持续记录的过程指标（在线发现坍缩/幻觉，成本≈0）；② ClawEval Multi-turn split 终点指标。论文中本方案作为**系统设计贡献**呈现，以过程指标 + 终点指标为证据，不做组件级 ablation。

### 8.1 过程指标（训练中持续记录，接 `replay_metrics` 同款 sidecar 思路）

| 指标 | 定义 | 检测什么 |
|------|------|---------|
| 生成 query 多样性 | distinct-n / 批内 embedding 平均成对距离 | 模式坍缩（早期预警） |
| 前提成立率 | 抽样人工/judge 审计：follow-up 引用的事实在 winner 状态中存在的比例 | 模拟器幻觉（§9 残余风险） |
| follow-up 类型分布 | 挑错 / 追加 / 追问解释 的占比随训练步演化 | 视角抽样是否实际生效 |
| follow-up 组内 reward std | 生成 query 上 GRPO 组内标准差 | 题目是否仍有区分度（std→0 = 太易/太难） |
| 桶分布漂移 | 生成 query 的 `<task_domain>` 分布 vs 种子分布 | 模拟器是否把会话拖向少数领域 |

### 8.2 端到端

ClawEval **Multi-turn split（38 任务）**为主要终点指标（方案直接针对多轮能力）；General split 监控无回退。

---

## 9. 风险与边界（论文 Limitations 底稿）

| 风险 | 说明 | 缓解 |
|------|------|------|
| follow-up 分布失真 | 模拟器分布 ≠ 真实用户 follow-up 分布，训练目标有系统偏移 | 首轮保持全真实锚点；persona/lens 库从真实回流 follow-up 中归纳（构建库时用真实数据，运行时不用） |
| 模拟器幻觉 | digest 不完整 → 模拟器引用不存在的细节，前提漂移以小概率回归 | 前提成立率审计（§8.1）；digest 覆盖主要产物全文摘录 |
| 出题/阅卷耦合 | 同模型自我偏好 | env 隔离配置（§7.4），优先异模型 |
| Expert-iteration 偏置 | winner-sync 已知偏置（调度指南 §3②）：模型学不到"从自己的烂摊子恢复"；模拟器只看 winner 会强化它 | 与现行方案同等接受；若多轮容错差，改 softmax 抽样 sync 目标（既有预案） |
| q1 全失败边界 | winner 也无产物时模拟器看到空状态 | **已拍板（决策 #10）**：随机二选一——顺势抱怨（要求重做，真实用户行为）或 `<end_session>`；**抱怨重做累计 ≤ 3 次/会话**，超限强制结束，防全失败循环把会话拖垮 |
| 模拟器调用延迟 | 每 query 边界多一次 LLM 调用，串行计入会话尾延迟 | 16 会话并行天然摊薄；digest 截断控制输入长度 |

---

## 10. 决策记录

| # | 决策 | 结论 | 日期 |
|---|------|------|------|
| 1 | 多轮 follow-up 来源 | 真实数据只留 q1，其余在线生成（放弃静态 follow-up 主路径，原始数据留存备查） | 2026-06-11 |
| 2 | 观察范围 | 只观察 winner（正史一致性 + 成本） | 2026-06-11 |
| 3 | 防坍缩 | 轮数压小（1–3 抽样）+ 画像库（会话级）+ 视角库（轮级） | 2026-06-11 |
| 4 | Reward | @孙豪 提供的外部 OpenAI 兼容 API；机制走现有 `JudgeClient` env 注入；本地 vLLM 降级为 fallback；**API 地址待提供** | 2026-06-11 |
| 5 | 实现节奏 | 文档先行；模拟器/观察 agent 代码暂缓，按 §7 契约后补 | 2026-06-11 |
| 6 | 观察接口首版 | 固定 digest（非工具自主观察），只读工具观察作升级路径 | 2026-06-11 |
| 7 | **不增设消融实验** | 20 实验已饱和；有效性靠过程监控（§8.1）+ ClawEval Multi-turn 终点指标 | 2026-06-11 |
| 8 | 出题侧安全 | 不在本方案内建设——已有专门 gate 覆盖（O5 关闭） | 2026-06-11 |
| 9 | 种子筛选 + 单/多轮配比 | 划归**数据侧（@吴健）**，本仓库只消费（O1 移交） | 2026-06-11 |
| 10 | q1 全失败兜底 | **随机二选一**：顺势抱怨（要求重做）或 `<end_session>`；抱怨重做**累计 ≤ 3 次/会话**，超限强制结束（防崩溃循环）。注意：抱怨轮**不计入** K 的 follow-up 配额（K 只数正常 follow-up），但计入会话总轮数硬上限 | 2026-06-11 |

### 10.1 待设计清单（2026-06-11 @孙豪 已分流）

| # | 待设计项 | 说明 | 归属 / 状态 |
|---|---------|------|------------|
| O1 | 种子筛选标准 + 单/多轮配比 | "可追问性"过滤规则 + 多轮会话与单轮 query 的训练数据配比 | **数据侧（@吴健）负责**；本仓库只消费产出（schema 见 §7.1），Gap B 转换时对接 |
| O2 | 画像库 / 视角库的具体条目 | persona 条目数、字段 schema、从真实回流 follow-up 归纳的流程 | **延后**，模拟器实现前再做；起步可先用 3 个 fs-seed persona |
| O3 | 模拟器 prompt 模板 | 出题 system prompt：persona/lens 注入、`<end_session>` 触发、防"AI 腔" | **留空**（2026-06-11 @孙豪：暂不设计，实现时再定） |
| O4 | follow-up 评分 rubric（judge 的评分细则：按什么标准判"用户要求被落实"） | judge 阅卷 prompt：输入裁剪 token 预算、评分维度是否沿用 ClawEval 公式 | **留空**（同上） |
| ~~O5~~ | ~~生成 query 的安全过滤~~ | **不需要**：已有专门 gate 覆盖出题侧安全（2026-06-11 @孙豪 确认），本方案不重复建设 | 已关闭 |
| O6 | digest 采集细则（喂给模拟器的 winner 状态摘要的生成规则） | 只读命令集、file_tree 深度、产物类型摘录规则、token 预算 | **留空**（同上） |
| ~~O7~~ | ~~q1 全失败兜底拍板~~ | **已拍板**：随机抱怨/结束 + 抱怨 ≤ 3 次/会话，见决策 #10 | 已关闭 |

---

## 11. 文档索引

| 文档 | 关系 |
|------|------|
| `Sandbox_管理调度指南.md` | winner-sync / 正史 / 兜底规则——本方案的插入骨架 |
| `SandboxRollout.md` §5.5 | per-query 入桶契约——生成 query 沿用 |
| `Plan_训练链路补齐.md` Gap A | judge reward 决策门——外部 API 同样适用 |
| `Sandbox_Agent架构.md` | persona / fs-seeds——画像库的初始来源 |
| `CL_Update_Sunhao.md` | CL Loss / 实验路线——本方案产出的轨迹按原路线入桶训练 |
