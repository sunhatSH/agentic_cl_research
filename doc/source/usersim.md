# UserSim：模拟用户在线生成多轮 Query

> **定位**：多轮训练数据构造方案——真实回流数据只保留会话首条 query 作种子，后续 query 由**三个协作 agent**（观察 / 出题 / 奖励）在 rollout 运行时、观察 winner 状态后在线生成与评分。

---

## 1. 模型选型

### 1.1 全部选型

| 角色 | 选型 | 配置落点 | 温度 | endpoint |
|------|------|----------|------|----------|
| **Actor / 基座策略**（正式训练） | **Qwen3.6-27B**（`qwen/qwen3.6-27b`） | `configs/base.yaml` `actor_rollout_ref.model.path` / verl rollout（lightllm） | **1.0**（GRPO 组内多样性刚需） | 本地权重 |
| **冷启动 on-policy actor** | **Qwen3.6-27B**（走 sufy，不本地起 vllm） | `collect_rollout.sh` `LOCAL_VIA_SUFY=1` | **0.4** | sufy |
| **冷启动 off-policy actor** | **openai/gpt-5.5** | `docker/sandbox/runtime.env` `AGENT_MODEL_NAME` | **0.4** | sufy |
| **Observer（观察）** | **openai/gpt-5.4-mini** | `configs/agents.yaml: observer` | **0.0**（客观） | sufy |
| **Questioner（出题 / UserSim）** | **claude-4.6-sonnet**（主）+ 轮换池 | `configs/agents.yaml: questioner.rotation` | **0.9**（抗坍缩） | sufy |
| **Reward / Judge** | **anthropic/claude-4.8-opus** | `configs/agents.yaml: reward`（`key_env: SUFY_API_KEY`） | **0.0** | sufy |

**Questioner 轮换池**（每 5 次切换，抗模式坍缩）：
`claude-4.6-sonnet` → `deepseek/deepseek-v4-pro` → `qwen/qwen3.7-max` → `moonshotai/kimi-k2.6`（4 个跨厂商模型，最大化输出风格异质性）。

> Observer LLM 可选（`Observer(use_llm=False)` 默认走确定性取证、零模型调用）；上表 Observer 选型仅在 `use_llm=True` 时生效。

### 1.2 Endpoint 与凭证

- **sufy**（唯一远程模型源，OpenAI 兼容）：base `https://openai.sufy.com/v1`；key 取自 `SUFY_API_KEY`（开发机侧读 `.env`，沙箱内 actor 读 `docker/sandbox/runtime.env` 的 `AGENT_MODEL_KEY`，值同一个 sufy key）。模型 id 带 vendor 前缀。
- **本地 Judge**（已废弃）：`scripts/serve_reward_model.sh` 保留作 fallback。**smoke**：`scripts/mock_judge.py`（`JUDGE_MODEL=mock-judge`）。

### 1.3 选型约束

- **三 agent 互不为同一模型**（observer/questioner/reward 各自独立模型）→ 抗 self-preference。
- **Questioner 多模型轮换**：4 个跨厂商模型每 5 次 `chat()` 切换，防 follow-up 模式坍缩。
- **Persona tone 注入**：persona 的 `tone`（calm/neutral/hot）注入 Questioner system prompt。
- **Judge 冻结 + ≥ 策略容量**：reward 必须是不变的尺子（跑数周/多实验测遗忘，避免版本漂移污染遗忘度量）；`claude-4.8-opus` 能力远超 27B 策略以抗 reward-hacking；判分对象 = ClawEval 三维 rubric（`safety*(0.8·completion+0.2·robustness)`）。

### 1.4 温度策略

| 用途 | 温度 | 理由 |
|---|---|---|
| 训练 rollout actor（GRPO 8 路） | **1.0** | GRPO 靠同 query 8 条轨迹的 reward 方差算 advantage；温度低→组内趋同→advantage≈0→梯度消失 |
| 冷启动单路采集（slots=1） | **0.4** | 单路无组内多样性需求；采防遗忘锚点，高温只会让工具调用发散出错 |
| 冷启动 GRPO 多路 | **1.0** | 有 winner-sync，需一点组内方差选 winner |
| Observer / Reward | 0.0 | 客观取证 / 冻结尺子，确定性 |
| Questioner | 0.9 | 出题多样性，抗 follow-up 模式坍缩 |

---

## 2. 三 Agent 架构

### 2.1 问题动机：前提漂移

多轮会话 $S = (q_1, \dots, q_K)$ 中 $q_{k}$（$k \ge 2$）通常引用 $q_{k-1}$ 执行后的结果。记 $e_k$ = 执行 $q_k$ 后的环境状态，$\phi(q_{k+1})$ = $q_{k+1}$ 成立所需的前提谓词。静态数据集把 $q_{k+1}$ 在数据收集时写死，但训练 rollout 中 $e_k$ 是策略 $\pi_\theta$ 的随机函数，出现**前提漂移**：

$$\Pr\big[\phi(q_{k+1})(e_k) = \text{true}\big] < 1$$

且该概率随 $\pi_\theta$ 演化而漂移。前提不成立时有两类训练污染：① **错误梯度**（模型对不存在的"问题"硬编修复 → 被 judge 奖励 → 训练出幻觉迎合）；② **信号稀释**（引入与策略无关的 reward 方差，污染 GRPO 组内归一化）。

**依赖强度分级**：A 类（仅依赖"交付物存在"，前提成立率高，静态可行）；B 类（依赖结果的具体性质/缺陷，前提成立率低且随策略漂移，**不可行——本方案动机**）；C 类（依赖中间产物，极低，不可行）。**回答**：让"用户"看到结果之后再发问——前提由构造保证成立（grounded by construction）。

### 2.2 数据侧：只保留种子 query

`datasets/queries.jsonl` 每行只消费 `queries[0]`：真实数据贡献"用户会发起什么任务"的分布；出题 agent 贡献"用户看到结果后会怎么跟进"的分布。丢弃发生在数据加载层，原始 jsonl 不动。

### 2.3 运行时：三 agent 协作的会话流水线

嵌入 `SessionSandboxPool` 会话循环，插入点 = `sync_to_winner` 之后：

```text
母版 → 派生 8 槽（位级同起点）；会话开始随机抽 1 个出题人设 p ∈ {42}
  │
  ├─ q1（真实种子）: 8 路并行 rollout → 选 winner → sync 8 槽
  │
  ├─ 【观察 agent】（无人设）→ 产出客观状态报告 R_t
  │     ├──→ 【奖励模型】 reward(R_t, 轨迹, 判分准则)
  │     └──→ 【出题 agent】（人设 p） q_{t+1}(R_t, H_t) → 下一 query 或 <end_session>
  │
  ├─ q2: 8 路并行（同起点 = winner 状态 + winner 正史）→ winner → sync
  └─ ≤ K_max 轮后 destroy_all（母版不回写）
```

形式化。第 $t$ 轮 winner 选定并 sync 后，观察 agent 产出报告 $R_t = \mathrm{Obs}(a_t^{w}, e_t^{w})$，其中 $a_t^w$ = winner actor 轨迹输出，$e_t^{w}$ = winner 沙箱环境。报告 $R_t$ **一份两用**：

- 评分：$r_t = \mathrm{Reward}(R_t, a_t^{w}, \text{rubric})$
- 出题：$q_{t+1} \sim Q(\cdot \,|\, p, R_t, H_t)$

$p$ = 会话级随机人设（42 选 1），$H_t$ = 会话正史（历代 winner 消息）。**前提成立性由构造保证**：$q_{t+1}$ 是在观察 agent 看过 $e_t^w$ 之后生成的，$\Pr[\phi(q_{t+1})(e_t^w)] \approx 1$。

### 2.4 认知分工：Observer 是眼睛，Questioner 是大脑

我们把"一个真实用户如何评判 AI 的产出"拆解成两个解耦的认知阶段——**观察**与**判断**，分别由 Observer 和 Questioner 承担，恰如人的**眼睛**与**大脑**。这一分工是整个 UserSim 抗 reward-hacking 与生成拟真多轮数据的设计基石。

**Observer —— 眼睛：客观、无情、只陈述事实。** 眼睛不撒谎，也不评判，只把世界真实发生了什么如实映入脑中。Observer 正是如此——它**不听 Actor 自述**（Actor 声称"我已完成"只是叙述），而是对沙箱环境做确定性的 before/after 差分：这一轮新增/修改/删除了哪些文件、文件的真实内容、系统状态变化（装了什么包、起了什么进程、开了什么端口）、以及 Actor 声称与实际痕迹是否矛盾。全程由**代码**完成，不掺入模型的理解或润色，产出一份冷静、客观、无情感倾向的观察报告 $R_t$——就像视网膜成像，只映照光线，不附加喜恶。这是抗 reward-hacking 的根基：**评判必须锚定"真实发生了什么"，而非"Actor 说发生了什么"**。眼睛看到的是 ground truth。

**Questioner —— 大脑：主观、带人设、有情感。** 同一幅画面，不同的人会有截然不同的反应——这正是大脑的工作。Questioner 接过 Observer 的客观报告后，戴上**人设**的有色眼镜去解读：

- **人设决定关注点**：细节控的财务审计盯着某个具体数字，大局观的经理只看整体交付物是否成形。同一份报告，42 种人设看到的"问题"各不相同。
- **人设决定标准**：研究员要求"贴出原始文件别给摘要"，SRE 追问"修复到底跑起来没有"，编辑挑排版。是否满意由**这个人的职业标准**决定，而非固定阈值。
- **人设决定情绪与语气**：同样是不满，冷静型耐心提点，急脾气直接施压。追问措辞与耐心多寡，都带**这个人的性格温度**。

于是 Questioner 产出一份主观、带立场、有情感的评价与追问——决定这一轮继续深挖、指出缺陷、还是心满意足结束。

| | Observer（眼睛） | Questioner（大脑） |
|---|---|---|
| 职责 | 观察世界真实发生了什么 | 评判产出好不好、够不够 |
| 性质 | 客观、确定性、无情感 | 主观、带人设、有情感 |
| 依据 | 沙箱 diff（ground truth） | 观察报告 $R_t$ + 人设 $p$ + 正史 $H_t$ |
| 产出 | 事实陈述 | 追问 / 挑刺 / 满意结束 |

**眼睛保证看到的是真的，大脑保证评判像个真人。** 两者解耦，才让我们既能抵御"Actor 骗过评判"的 reward-hacking（眼睛只信事实），又能生成千人千面、贴近真实线上分布的多轮追问数据（大脑各有性格）。一个负责"是什么"，一个负责"怎么看"——合起来，就是一个有眼睛、有大脑的仿真用户。


### 2.4 三个 Agent 的职责边界

| | **Observer** | **Questioner** | **Reward** |
|---|---|---|---|
| 人设 | **无**（客观中立） | **有**（42 选 1，会话级固定） | 无（冻结 judge） |
| 输入 | winner 的 actor 输出 + winner 沙箱 | 观察报告 $R_t$ + 会话正史 $H_t$ + 人设 $p$ | 观察报告 $R_t$ + winner 轨迹 + 判分准则 |
| 职责 | 从 actor 输出判断本任务要收集什么，主动收集中间结果 + 最终结果，产出结构化客观报告 | 以人设视角（含观察偏好）阅读报告，模拟真实用户发起下一 query 或结束 | 依报告中的实际产出与详细效果 + rubric 打 reward |
| 输出 | 状态报告 $R_t$（喂给出题 + 奖励两方） | 下一 query 文本 / `<end_session>` | reward 标量（+ 附加指标） |
| 沙箱权限 | **只读**（在存活 winner 上跑只读命令翻产物） | 无（不碰沙箱，只读报告） | 无（只读报告） |

**设计要点**：① **观察与立场解耦**——观察 agent 只负责"客观把证据收全"，不带任何用户偏好；偏好只发生在出题 agent 一侧。② **观察由 actor 输出驱动**——让观察 agent 读 actor 声明去定向收集，中间结果（易被后续步骤覆盖删除）也能在被丢弃前被捕获，接住 C 类依赖。③ **一次观察、两个下游**——报告 $R_t$ 同时供评分与出题，保证"用来给分的事实"和"用来出题的事实"严格一致。④ **奖励落到实处**——reward 读观察 agent 收来的实际效果（文件真生成了没、内容对不对），抗"嘴上说做完了"的 reward hacking。

### 2.5 为什么只观察 Winner

1. **正史一致**：会话正史 = 历代 winner 轨迹拼接，输家轨迹不进会话谱系。出题 agent 能看见的世界只能是 winner 这条世界线——看输家状态生成的 query 会引用不存在于正史的产物，等于重新引入前提漂移。
2. **同起点不破坏**：$q_{t+1}$ 的 8 槽起点 = sync 后的 winner 状态。出题基于该状态，保证"题目与 8 槽共同起点匹配"，GRPO 组内比较仍然干净。
3. **成本**：观察 1 份而非 8 份；7 个输家在 sync 时即销毁。

### 2.6 Observer — `agents/observer.py`（diff-driven，模型只看 STATE）

observer 模型不看 actor 轨迹——判断只基于真实状态增量。若让 reward/observer 以 actor 的"声称"为依据，actor 会学到"嘴上说完成、实际没交付"的策略（reward hacking）。observer 根本不接收轨迹——它对环境做确定性 before/after diff（文件内容 + SysOps 状态），completion 只能由真实 diff 决定。反 hacking 是**结构性**的。

```python
# 伪流程：观察组件做确定性取证，模型（可选）只归纳 diff
post   = snapshot(sandbox)                 # fs + sys 快照（仅 run_code，后端无关）
diff   = diff_snapshots(baseline, post)    # added/modified/removed + 内容
extract_binaries(...)                       # xlsx/docx/pptx/pdf → 文本（仅变更文件）
sys_diff = diff_system(baseline, post)      # 装的包/开的端口/起的进程
state_diff = render(diff, sys_diff)
report = (use_llm ? parse(client.chat(build_observer_prompt(state_diff)))   # 模型只看 diff
                  : build_deterministic_report(diff))                       # 默认零模型调用
report.actor_trajectory = flatten(winner_messages)   # pass-through，不进 prompt
```

- **observer LLM 可选**（`use_llm`，默认 False）：默认确定性建报告、零模型调用；开模型也只看 `state_diff`，永不看 trajectory。
- **容错**：LLM/沙箱失败 → 降级为最小报告（state_diff + tree），不 crash 会话。**空报告**：`has_effect=False`（空 diff）→ 触发失败路径 + 耐心机制。
- **Env**：`OBSERVER_API_BASE` / `OBSERVER_MODEL` / `OBSERVER_API_KEY`。

### 2.7 Questioner — `agents/questioner.py`

读 $R_t$ + 人设 + 历史，生成下一条 user query。输出 `str`（下一条 query）或 `None`（`<end_session>` / LLM 失败）。42 人设（`agents/personas.json`）：每人设含 `profession / preference / profile / observation_focus / tone / patience(P0) / patience_decay(d0)`。`observation_focus` 编码「整体|细节 × 形式|内容」，决定强调报告哪一面。**Env**：`USERSIM_API_BASE` / `USERSIM_MODEL` / `USERSIM_API_KEY`。

### 2.8 Reward — `agents/reward.py`

以 $R_t$ 为证据，调用冻结 judge 打分。复用 `trainer/model_reward.JudgeClient`：

```python
score_followup(query, report, judge=get_judge())   # trajectory 从 report 取
# → {score, completion, safety, robustness, judge_error[, gated]}
```

- **双通道（都来自同一份 $R_t$）**：`state_diff`（observer 状态证据）= completion 的 ground truth；`report.actor_trajectory`（pass-through，observer 模型没看过）给 judge 判 safety/robustness。
- **拦截层（gate）**：`report.has_effect=False`（空 diff）→ 直接 score 0、不调 judge（带 `gated`）。
- **防超长**：`state_diff` 与 `actor_trajectory` 各自中间截断封顶（`_MAX_DIFF_CHARS` / `_MAX_TRAJ_CHARS`）。**尺度**：与 eval 同构（`safety*(0.8*completion+0.2*robustness)`）。**Env**：`JUDGE_API_BASE` / `JUDGE_MODEL` / `JUDGE_API_KEY`。

### 2.9 接口契约

#### 数据 schema（`datasets/queries.jsonl` 加载层）

```json
{"session_id": "...", "queries": ["q1 真实", "q2 真实(加载时丢弃)", "..."], "max_followups": 3}
```

加载层只取 `queries[0]`；保留原始字段备查。出题人设不来自数据，而是运行时从 42 人设库随机抽。

#### ObservationReport（三 agent 共享的中间数据结构）

```python
@dataclass
class ObservationReport:
    state_diff: str            # 环境 before/after diff（含内容 + SysOps）= ground truth
    intermediate: list[dict]   # 中间结果: {desc, source(命令/文件), value_excerpt}
    final: list[dict]          # 最终交付物: {path, kind, content_excerpt}
    discrepancies: str         # 状态内部红旗（空/损坏/自相矛盾；可空）
    actor_trajectory: str      # pass-through：actor 轨迹文本，observer 模型不看，仅给奖励模型
    file_tree: str             # winner workspace 文件树（深度截断，兜底）
    has_effect: bool           # 本轮 diff 是否非空；False → 短路 reward、走失败/耐心路径
```

采集时机与 `run_checkers` 相同（sync 后、在存活 winner 实例上跑只读命令）。观察 agent 对 winner 沙箱做确定性 before/after diff（文件内容 + 二进制提取 + SysOps），以真实状态增量为唯一判据；轨迹仅由观察组件 pass-through 给奖励模型。

#### Observer 协议（无人设）

```python
class Observer(Protocol):
    def observe(self, actor_trajectory: list[dict], sandbox: "ReadOnlySandbox") -> ObservationReport: ...
```

#### Questioner 协议（有人设）

```python
class Questioner(Protocol):
    def next_query(self, persona: "Persona", report: ObservationReport,
                   session_history: list[dict]) -> str | None: ...  # None == <end_session>
```

#### Persona schema

```python
@dataclass
class Persona:
    name: str
    profession: str       # 职业
    preference: str       # 偏好（关注点 / 容忍度）
    profile: str          # 用户画像（专业度 / 语气 / 耐心的文字描述）
    observation_focus: str  # 观察偏好: 整体|细节 × 形式|内容
    tone: str             # calm / neutral / hot
    patience: float        # 初始耐心 P0
    patience_decay: float  # 基础扣减 d0（无则回退全局默认 0.1）
```

### 2.10 代码模块地图

```text
agents/        schema.py(ObservationReport,Persona) personas.json(42人设) personas.py(加载+sample) observer.py questioner.py(+PatienceTracker) reward.py(score_followup→JudgeClient) prompts.py base.py(ChatClient+resolve_*_client)
rollout/       simulated_session.py(8槽+winner-sync+三agent,训练) usersim_collect.py(单槽多轮采集,无reward)
scripts/       collect_rollout.py(多轮UserSim采集) collect_cold.py(单轮冷启动,无三agent)
trainer/       model_reward.py(JudgeClient,Reward复用)
```

### 2.11 与现有训练链路的耦合

| 模块 | 影响 |
|------|------|
| GRPO / winner-sync | **零改动**。三 agent 只在 query 边界工作，8 路同起点性质不变 |
| 会话正史 | 复用 `session_history`；出题 agent 生成的 query 以 user 消息身份进正史 |
| 入桶 | 不变。per-query 入桶 + `<task_domain>` 标签，生成 query 同样适用 |
| Reward | q1 可保留静态 checker；生成 follow-up 只能走模型 judge（无预录 ground truth），以观察报告为证据 |
| batch 换算 | 每会话 query 数 = $1 + K$，$K$ 随机 → `gen_batch_size` 用 $\mathbb{E}[1+K]$（K~U{1..3} 时期望 3 条/会话） |
| 母版 / 会话销毁 | 不变（会话结束销毁、不回写母版） |

### 2.12 采集 vs 训练：三条路径对照

| 路径 | 槽数 | 三 agent | Reward | 用途 |
|------|------|---------|--------|------|
| `collect_cold.*` | 1 | ❌ | ❌ | 冷启动 buffer |
| `collect_rollout.*` | 1 | Observer + Questioner | ❌ | 多轮 query 采集验证 |
| `simulated_session` | 8 | ✅ 全套 | ✅ | GRPO 训练 rollout |

---

## 3. 多轮 Query 构造

### 3.1 防模式坍缩：四维机制

LLM 自我对话的已知失效模式是**模式坍缩**——follow-up 趋同于少数模板腔，熵随轮数衰减。四个机制对抗它：

| 维度 | 粒度 | 抗坍缩机制 | 正交性 |
|------|------|-----------|--------|
| **人设随机** $p$ | 会话级 | 42 人设 = 职业 + 偏好 + 用户画像 + 观察偏好（整体 vs 细节、形式 vs 内容）。每会话随机抽 1 个。观察偏好让同一份客观报告被不同人设问出不同侧面 | 改变"以谁的视角问、强调结果的哪一面"的先验 |
| **会话长度由 Questioner 驱动** | 会话级 | 三层协同控制（详见 §3.3）： **(1) Questioner 满意度（主控）** — 每轮 observer 出报告后,Questioner（LLM 带人设）自主判断"交付够好了吗"。够则输出 `<end_session>` 自然停止；不够则继续追问。 **(2) Max\_turns=20（兜底安全帽）** — Questioner 始终不满意时的硬件上限,正常情况几乎到不了（通常 3–8 轮即满意结束）。 **(3) 耐心 P0×r^k（仅失败轮）** — hermes 本轮崩溃或空输出时才消耗；成功追问不耗耐心,Questioner 满意结束也不耗耐心。 | 链越长越易坍缩,但终止权归有判断力的一方(Questioner),而非随机抽样 |
| **winner 状态逐轮演化** | 轮级（天然） | 每轮 winner 状态都被上一条 follow-up 改变，观察报告 $R_t$ 随之不同 | 即使人设固定，每轮报告不同 → 出题条件分布天然变化 |
| **Questioner 多模型轮换** | 轮级（每 5 次提问切换） | Questioner 在 4 个跨厂商模型之间轮换，每 5 次 `chat()` 调用后切换到下一个 | 从模型层面注入输出风格异质性——不同厂商训练数据、对齐方式、语言风格各异，轮换后 follow-up 的关注点、语气、角度自然分散 |

> 人设决定问什么，模型决定怎么问——正交互补。5 次切换：不会太频繁（破坏上下文连贯感），也不会太稀疏（一个模型连出 5 条已足够形成局部风格、又不至于整 session 被同一模型主导）。

附加机制（可选）：出题 agent 采样温度调高；对同 batch 生成的 query 做 n-gram / embedding 去重监控（只监控告警，不在线拒绝，避免引入选择偏置）。

### 3.2 Algorithm 1：会话构造伪代码

```text
Algorithm 1: User-Sim Session Rollout（单会话，三 agent）
输入: 种子 q1（真实回流）、母版 M、42 人设库 P、K_max=20
 1:  p ~ P                                             # 会话级随机抽人设
 2:  slots ← spawn(M, 8)                              # 位级同起点
 3:  H ← []；q ← q1；turns ← 0
 4:  while turns < K_max:
 5:      T ← parallel_rollout(slots, q, H)             # 8 条轨迹
 6:      w ← select_winner_with_fallback(T)            # 按已有 reward 选 winner（含兜底）
 7:      if w = None: 按耐心决定 redo 或 终止          # 耐心 P0×r^k，仅失败轮消耗
 8:      R_t ← observer(w, baseline)                   # 沙箱 before/after diff → 结构化报告
 9:      q ← questioner(p, R_t, H)                     # 人设 p 判"够好了吗？"
 8:      sync_to_winner(w)；H ← H ∥ T[w].messages
 9:      R_t ← Observer(actor=T[w], sandbox=winner) # 观察 agent：客观报告
10:      r ← Reward(R_t, T[w], rubric)              # 奖励模型：以报告为证据打分
11:      buffer ← T（全部 8 条，per-query 入桶；winner 用 r）
12:      if t > K: break
13:      q ← Questioner(p, R_t, H)                  # 出题 agent：人设视角生成下一 query
14:      if winner response 失败/停止/未完成:        # 兜底（耐心机制）
15:          P ← P - d0(p) * 2^(fail_cnt)；fail_cnt += 1  # d0 亦人设自带，指数扣减
16:          if P < 0 or rand() > clip(P,0,1): q ← <end_session>  # 否则要求重做
17:      if q = <end_session>: break
18:  destroy_all()                                 # 母版不回写
```

与现行 `run_session`（`session_pool.py`）的差异在 9–17 行：新增观察 agent（9）、报告驱动的奖励（10）与出题（13），循环驱动从"遍历静态 queries 列表"变成"出题 agent 决定下一条 / 终止"。

### 3.3 耐心机制 — `PatienceTracker`

当某轮优胜 response **失败 / 提前停止 / 未做完**（无有效产物）时，是否让模型重做由**人设携带的耐心值**决定——把"放弃"做成模拟用户的属性，而非写死的工程上限。

#### 公式

每个人设带初始耐心 $P_0(p)$ 与基础扣减 $d_0(p)$，用于解耦两个独立人格轴——$P_0$ = 初始容忍度（愿不愿给机会），$d_0$ = 挫败升级速度（脾气）；无显式 $d_0$ 时回退到全局默认 `DEFAULT_PATIENCE_DECAY = 0.1`。会话内第 $k$ 次失败时耐心**指数增长地衰减**：

$$P_k = P_{k-1} - d_0(p)\cdot 2^{\,k-1} \;=\; P_0(p) - d_0(p)\,(2^{k}-1).$$

> 例（$d_0=0.1$）：第 1 次失败扣 0.1，第 2 次扣 0.2，第 3 次扣 0.4 …… 累计扣减 $0.1,0.3,0.7,1.5,\dots$
>
> 两轴示例：高 $P_0$+低 $d_0$=好脾气的耐心用户；高 $P_0$+高 $d_0$=先礼后兵；低 $P_0$+低 $d_0$=期待低但不计较。

#### 决策逻辑

```python
class PatienceTracker:
    def on_failure(self) -> bool:
        self.fail_count += 1
        pk = self.p0 - self.d0 * (2**self.fail_count - 1)
        redo_prob = clip(pk, 0, 1)
        return rng.random() < redo_prob   # True=要求重做, False=结束会话
```

- $P_k < 0$ → 终止会话（`<end_session>`）；$P_k \ge 0$ → 以概率 $p_{\text{重做}}=\mathrm{clip}(P_k,0,1)$ 要求重做，否则终止。

#### 设计性质

| 性质 | 说明 |
|------|------|
| 天然自封顶 | 扣减翻倍 → 耐心穿过 0 仅需约 $\log_2(P_0/d_0)$ 次失败。取 $d_0=0.1,P_0\approx1$ 时约 ≤3 次，平滑复现原"≤3 硬上限"，但上限是涌现的、且随人设可变 |
| 人设可分（两维） | $P_0$ 调"起步给不给机会"，$d_0$ 调"崩得多快"。二者非完全正交（重试次数 ≈ $\log_2(P_0/d_0)$），设值时按"(初始容忍, 升级速度)"语义来想 |
| 递增挫败感 | 翻倍扣减刻画"越失败越烦" |
| 只管失败路径 | 成功轮不消耗耐心，照常走 $K$；重做轮不计入 $K$ 的 follow-up 配额 |
| 随机性 | 会话级 seeded RNG（与 `session_pool` 一致）保证可复现 |

#### 两轴人设示例

| 人设 | $P_0$ | $d_0$ | 行为直觉 |
|------|-------|-------|---------|
| Anna（office assistant） | 1.2 | 0.08 | 好脾气、慢升级 |
| Raj（SRE） | 0.8 | 0.2 | 先给机会但升级快 |
| Tom（founder） | 0.6 | 0.3 | 低容忍、很快放弃 |

### 3.4 Reward 评分细节

- **q1（真实种子）**：静态 checker（convert 时提取）+ judge 混合；**生成 follow-up**：无预录 ground truth，**纯 judge**，以观察 agent 报告 $R_t$ 为评分证据——judge 输入 = (follow-up query, 会话正史 $H_t$, winner 轨迹 $a_t^w$, 观察报告 $R_t$, 判分准则)。报告里的"实际产出与详细效果"让 judge 据实判分，而非只看 actor 嘴上声明。
- **判分 rubric**：与 eval 同构（`safety*(0.8*completion+0.2*robustness)`）。
- **同模型耦合风险**：三方若复用同一模型，存在 self-preference 偏置。优先异模型或异 endpoint，至少 prompt 角色隔离 + env 分离配置。

### 3.5 与静态方案对比

| | 静态多轮（baseline） | 前提门控+截断 | **在线模拟用户（本方案）** |
|--|---------------------|--------------|--------------------------|
| B 类 follow-up（挑刺/修改） | 前提随机失效 | 失效时只能丢弃 → B 类覆盖率低 | 看结果后生成，前提由构造成立 |
| 数据复用率 | 整会话绑定 | 截断浪费尾部 | 一条种子可重复 rollout 出不同会话（每次 winner 不同 → follow-up 不同），数据增益 |
| 真实分布保真 | 全真实但前提错位 | 同左 | 首轮全真实；follow-up 分布由 42 人设库控制 |
| 对抗策略漂移 | 模型变强 → "挑错"类 query 前提成立率持续下降 | 同左（截断率上升） | 出题 agent 始终对观察 agent 报告的当前策略实际输出挑刺 → 难度自适应跟随策略演化（curriculum 效应） |

> 在线出题构成一个**弱对抗的自适应课程**——策略越强，残留缺陷越细微，出题 agent 挑出的刺也越细，follow-up 难度自动跟随能力前沿，无需人工设计难度调度。

### 3.6 过程监控与风险

**过程指标**（训练中持续记录，不增设独立消融实验；质量靠过程指标 + ClawEval Multi-turn split 38 任务终点指标）：

| 指标 | 检测什么 |
|------|---------|
| 生成 query 多样性（distinct-n / embedding 平均成对距离） | 模式坍缩（早期预警） |
| 前提成立率（follow-up 引用的事实在观察报告中存在的比例） | 出题 agent 幻觉 |
| 观察报告 vs 实际差异率（`discrepancies` 非空的比例） | 观察 agent 漏收 / actor 虚报 |
| follow-up 类型分布（挑错 / 追加 / 追问解释占比随训练步演化） | 42 人设是否实际产生多样性 |
| follow-up 组内 reward std | 题目是否仍有区分度（std→0 = 太易/太难） |
| 桶分布漂移（生成 query 的 `<task_domain>` 分布 vs 种子分布） | 出题 agent 是否把会话拖向少数领域 |

**风险与边界**：

| 风险 | 缓解 |
|------|------|
| follow-up 分布失真（42 人设分布 ≠ 真实用户 follow-up 分布） | 首轮保持全真实锚点；42 人设从真实回流 follow-up 中归纳（构建库时用真实数据，运行时不用） |
| 出题 agent 幻觉（观察报告不完整 → 引用不存在的细节） | 观察由 actor 输出驱动定向收集；前提成立率 + 差异率审计 |
| 三 agent 同模型耦合（自我偏好） | 各自独立 env，优先异模型/异 endpoint |
| Expert-iteration 偏置（winner-sync 已知偏置：学不到"从烂摊子恢复"） | 与现行方案同等接受；若多轮容错差，改 softmax 抽样 sync 目标 |
| response 失败/停止/未完成 | 人设耐心机制：$P$ 指数衰减，$P<0$ 必停，否则以 $\mathrm{clip}(P,0,1)$ 概率重做 |
| 多 agent 调用延迟（每 query 边界多三次 LLM 调用） | 16 会话并行天然摊薄；观察报告截断控制下游输入长度；观察一次两用省一次 |

---

## 4. 42 人设表

42 个独立 Questioner 人设，各有不同身份。每个会话随机抽 1 个、整场固定。

### 因素说明

- **职业 / 桶**：身份，及其对应的能力桶。
- **在意什么**：生成下一轮 query 时的侧重点。
- **观察偏好**：关注哪一面 = {细节 | 整体} × {内容 | 形式}。
- **语气**：怎么说 —— 冷静（克制礼貌）/ 居中（直白有情绪）/ 暴躁（强烈、可带责备讽刺）。
- **P0 / d0**：初始耐心 / 耐心衰减。放弃前重试次数 ≈ log₂(P0 / d0)。

> 各因素自由搭配，互不约束。

### 人设表

| # | 人设 | 职业 | 桶 | 在意什么 | 观察偏好 | 语气 | P0 | d0 |
|---|------|------|----|---------|---------|------|----|----|
| 1 | Mei | 财务分析师 | Finance | 精确数字、对账；不信整数 | 细节×内容 | 冷静 | 1.0 | 0.10 |
| 2 | Greg | 投行高管 | Finance | 只要结论，不关心过程 | 整体×内容 | 暴躁 | 1.0 | 0.40 |
| 3 | Omar | 采购专员 | Finance | 预算/供应商条款逐条准确 | 细节×内容 | 居中 | 1.0 | 0.12 |
| 4 | Hana | 审计师 | Finance | 凭证与账目能否对上 | 细节×内容 | 冷静 | 1.4 | 0.10 |
| 5 | Diego | 风控经理 | Finance | 风险敞口是否量化清楚 | 整体×内容 | 居中 | 0.9 | 0.18 |
| 6 | Raj | SRE/运维 | SysOps | 配置正确、可安全应用 | 细节×内容 | 暴躁 | 0.8 | 0.20 |
| 7 | Sven | 系统管理员 | SysOps | 权限与变更是否合规 | 细节×内容 | 居中 | 1.0 | 0.13 |
| 8 | Tariq | 网络工程师 | SysOps | 链路与防火墙规则是否生效 | 细节×内容 | 冷静 | 1.2 | 0.10 |
| 9 | Bjorn | DevOps 主管 | SysOps | 流水线是否可重复 | 整体×内容 | 居中 | 0.9 | 0.16 |
| 10 | Ken | 合规/风险官 | SysOps | 规则遵守、审计留痕 | 细节×内容 | 冷静 | 0.7 | 0.25 |
| 11 | Carlos | 项目经理 | Workflow | 交付物对原始需求的完整度 | 整体×内容 | 居中 | 1.0 | 0.12 |
| 12 | Marcus | 运营主管 | Workflow | 端到端流程跑通、不留半拉子 | 整体×内容 | 暴躁 | 0.9 | 0.15 |
| 13 | Tom | 早期创业者 | Workflow | 速度、大局；烦过度工程 | 整体×内容 | 暴躁 | 0.6 | 0.30 |
| 14 | Lena | 流程优化顾问 | Workflow | 步骤能否精简、瓶颈在哪 | 整体×内容 | 冷静 | 1.5 | 0.09 |
| 15 | Felix | 调度协调员 | Workflow | 任务依赖与时序是否对 | 细节×内容 | 居中 | 1.1 | 0.13 |
| 16 | Nadia | 项目助理 | Workflow | 清单项有没有漏 | 整体×内容 | 冷静 | 1.3 | 0.08 |
| 17 | Priya | 客服支持工程师 | Dialogue | 修复是否真解决问题 | 细节×内容 | 居中 | 1.1 | 0.13 |
| 18 | Yusuf | 呼叫中心组长 | Dialogue | 对话是否解决了客户诉求 | 整体×内容 | 暴躁 | 0.8 | 0.22 |
| 19 | Mira | 对话体验设计师 | Dialogue | 多轮上下文是否连贯 | 整体×形式 | 冷静 | 1.4 | 0.09 |
| 20 | Zane | 销售总监 | Dialogue | 能不能快速给到话术 | 整体×内容 | 暴躁 | 0.7 | 0.32 |
| 21 | Sophie | 内容/传播编辑 | Communication | 语气、清晰度、排版 | 细节×形式 | 居中 | 1.1 | 0.10 |
| 22 | Yuki | UX 写作 | Communication | 终端用户读起来如何 | 整体×形式 | 冷静 | 1.4 | 0.08 |
| 23 | Elif | 公关经理 | Communication | 措辞会不会引发误解 | 整体×形式 | 暴躁 | 0.7 | 0.26 |
| 24 | Pablo | 市场文案 | Communication | 卖点是否打动人 | 整体×形式 | 居中 | 1.0 | 0.15 |
| 25 | Grace | 演讲撰稿人 | Communication | 叙事是否有感染力 | 整体×形式 | 冷静 | 1.5 | 0.10 |
| 26 | Aiko | 培训师 | Communication | 讲解是否通俗易懂 | 整体×形式 | 冷静 | 1.4 | 0.09 |
| 27 | Bella | 数据分析师 | Knowledge | 图表背后的数、分析是否成立 | 细节×内容 | 居中 | 1.3 | 0.10 |
| 28 | Dr. Lena | 学术研究员 | Knowledge | 方法正确、中间结果可追溯 | 细节×内容 | 冷静 | 1.5 | 0.10 |
| 29 | Arjun | 情报分析员 | Knowledge | 信源是否可靠、推理是否跳步 | 细节×内容 | 冷静 | 1.6 | 0.08 |
| 30 | Wei | 研究助理 | Knowledge | 引用与事实能否核对 | 细节×内容 | 居中 | 1.2 | 0.11 |
| 31 | Sara | 咨询顾问 | Knowledge | 结论有没有证据支撑 | 整体×内容 | 暴躁 | 0.8 | 0.24 |
| 32 | Nina | QA 测试 | Analysis | 边界情况、声称与现实是否一致 | 细节×内容 | 冷静 | 1.6 | 0.07 |
| 33 | Igor | 统计学家 | Analysis | 显著性与假设是否站得住 | 细节×内容 | 冷静 | 1.5 | 0.09 |
| 34 | Tania | 增长分析师 | Analysis | 指标口径是否一致 | 细节×内容 | 居中 | 1.1 | 0.14 |
| 35 | Hugo | 实验科学家 | Analysis | 可复现性如何 | 细节×内容 | 居中 | 1.3 | 0.10 |
| 36 | Ravi | 商业智能工程师 | Analysis | 看板数据是否准 | 整体×内容 | 暴躁 | 0.9 | 0.20 |
| 37 | Anna | 办公助理 | OfficeQA | 文档条理、能见人 | 整体×形式 | 冷静 | 1.2 | 0.08 |
| 38 | Leo | 通才实习生 | OfficeQA | 期望低、来啥学啥 | 整体×内容 | 居中 | 1.2 | 0.06 |
| 39 | Mona | 行政主管 | OfficeQA | 格式是否合规范 | 整体×形式 | 居中 | 1.0 | 0.14 |
| 40 | Theo | 秘书 | OfficeQA | 日程与纪要是否齐全 | 细节×形式 | 冷静 | 1.3 | 0.09 |
| 41 | Carmen | 人事专员 | OfficeQA | 表单字段是否填全 | 细节×形式 | 居中 | 1.1 | 0.12 |
| 42 | Boris | 法务 | SysOps | 条款是否有漏洞 | 细节×内容 | 居中 | 0.9 | 0.17 |

> 按桶聚类排列。名字全不重复；职业横跨 7 能力桶；语气与耐心各档均有分布。
