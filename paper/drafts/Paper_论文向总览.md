# Continual Learning over Agentic LLMs：论文向总览

> **定位**：把整个 agentic CL research 项目的设计**收敛成一篇论文的骨架**——问题、方法、系统、实验、局限。每节对应论文章节，并指向仓库内的单一信源文档。
> **本文是综述/索引，不是新信源**：公式与超参以 [`CL_Update_Sunhao.md`](../../doc/CL_Update_Sunhao.md) 为准，调度以 [`Sandbox_管理调度指南.md`](../../doc/sandbox/Sandbox_管理调度指南.md) 为准，多轮数据以 [`UserSim_多轮Query在线生成.md`](../../doc/UserSim_多轮Query在线生成.md) 为准。
> **写作日期**：2026-06-12

---

## 0. 一页纸摘要（Abstract 底稿）

我们研究**在 agentic 场景下对大语言模型做持续强化学习（Continual RL）而不发生灾难性遗忘**的问题。线上 agent（工具使用、多轮交互、办公/运维/财务等领域任务）每隔一段时间用新回流数据做一次 RL 更新；朴素地把 GRPO 搬到这个流程会同时遭遇两类失败：**灾难性遗忘**（新任务挤掉旧能力）与 **Echo Trap**（多轮 RL 的策略坍缩）。

本工作提出一套**端到端可训练的持续学习方案**，由四个相互独立又彼此咬合的设计构成：

1. **四项 CL Loss** $L_{cl}=\lambda_1 L_{rl}+\lambda_2 L_{kl}+\lambda_3 L_{replay}+\lambda_4 L_{ent}$，以零源码改动注入 verl/GRPO；
2. **7 桶能力划分的 Replay Buffer**，按"抗遗忘价值"而非 reward 绝对值排优先级，桶内淘汰、禁止跨桶挤出；
3. **U 形块权重的 Reweighted Replay**，按动作块给 replay token 赋权，首尾重、中间轻；
4. **沙箱 rollout + winner-sync 会话调度 + 模拟用户在线生成多轮 query**，从源头保证组内 advantage 不被环境噪声污染、并根除静态多轮数据的"前提漂移"。

我们在 **Qwen3.6-27B** 上以 64 卡（40 推理 + 24 训练，Fully Async）部署，用 **ClawEval**（Pass³）量化遗忘与新任务习得，设计了 **21 个受控消融实验**逐项验证每个组件的边际贡献。

---

## 1. 引言（Introduction）

### 1.1 问题背景

线上 agent 产品的能力来自持续的 RL 后训练：每 ~2 周用新回流数据更新一次模型。这构成一个 **task-agnostic 的持续强化学习**问题——没有清晰的任务边界，新旧能力在同一参数空间里竞争。两个相互独立的失败模式：

- **灾难性遗忘（catastrophic forgetting）**：在新任务上的 RL 更新系统性侵蚀旧领域能力。
- **Echo Trap（B4）**：多轮 agent RL 中策略多样性坍缩 → 组内轨迹趋同 → GRPO advantage 退化 → 训练崩溃。

### 1.2 核心贡献

| # | 贡献 | 对应章节 |
|---|------|---------|
| C1 | 一个把抗遗忘正则与经验回放统一进 GRPO 的 **CL Loss**，零 fork 注入 verl | §3 |
| C2 | 按**能力/领域**而非难度划分、以**抗遗忘价值**排序的 **7 桶 Replay Buffer**；论证为何不能用 reward 绝对值排序 | §4 |
| C3 | **U 形动作块权重**的 reweighted replay：首端（早期高方差决策）与末端（结论生成段）同时受重视 | §5 |
| C4 | **沙箱 winner-sync 会话调度**：组内 8 槽位级一致，保证 advantage 只反映策略差异 | §6 |
| C5 | **模拟用户在线生成多轮 query**（观察/出题/奖励三 agent）：根除静态多轮数据的前提漂移，构成弱对抗自适应课程 | §7 |

> **数据归属边界（实现说明）**：本工作的输入为 **taskspec**（每个 task = 1 份声明 `taskspec.yaml` + 1 份初始文件系统 `files/`，含 `seed_query` / `hidden_goal` / `verifier` 判分 rubric / `user_profile`）。**多轮后续 query 的在线生成、rollout 轨迹采集、以及最终入桶/训练消费的 rollout 数据结构，均为本系统（C4/C5）的产出**——"信号产出"这一半从 taskspec 开始、到结构化轨迹结束，都在本工作范围内。1 个 seed → fork 8 容器跑同一 `seed_query`（GRPO 8 路、位级一致起点）。冷启动采集阶段先跑 C5 的 **observer + questioner 子集（不含奖励模型、不做 GRPO 组，单 query 单 rollout）**；完整训练态再启用奖励与 8 槽。

### 1.3 与已有工作的关系（一句话定位）

RFT 天然比 SFT 抗遗忘（B2/C1），但**仍会遗忘**；CLEAR（A2）证明 task-agnostic 下朴素经验回放即可大幅缓解遗忘，是我们的 baseline（R0）。我们在其上叠加 **能力分桶 + 抗遗忘优先级 + 块级重加权**，并把整套放进**真实多轮 agentic** 场景（含工具、沙箱、用户模拟），这是与既有 continual-RL 文献（多在玩具/GUI 环境，C1–C3）的关键区别。完整文献见 [`CL_Update_Sunhao.md` § 参考文献](../../doc/CL_Update_Sunhao.md#参考文献)（A/B/C/D 四组）。

---

## 2. 问题设定（Problem Setup）

- **基座**：Qwen3.6-27B；**算法**：GRPO（无 critic），每 query rollout $M=8$ 条轨迹，组内归一化得 advantage $A_i=\frac{r_i-\mathrm{mean}(r)}{\mathrm{std}(r)+\epsilon}$。
- **数据**：纯文本 195 任务，跨 7 个能力域；多轮会话（每会话多条顺序 query，共享上下文）。
- **目标**：最大化 **CL Score = New Task Perf − $\alpha$·Old Task Forgetting**（$\alpha=1.0$）。
- **隐含前提（贯穿全文）**：GRPO 组内归一化只在"$M$ 条轨迹差异纯来自策略采样随机性"时才无偏——任何环境噪声 / 数据前提错位都会污染 advantage。§6/§7 即从源头守住这个前提。

---

## 3. 方法一：CL Loss（Method · Objective）

$$L_{cl} = \lambda_1 L_{rl} + \lambda_2 L_{kl} + \lambda_3 L_{replay} + \lambda_4 L_{ent}$$

| 项 | 作用 | 关键决策 |
|----|------|---------|
| $L_{rl}$ | 新任务 GRPO 主目标 | $\lambda_1=1.0$ 固定 |
| $L_{kl}$ | 策略约束，缓解遗忘 | **reverse KL** $D_{KL}(\pi_{new}\|\pi_{ref})$；$\pi_{ref}=\pi_0$ vs $\pi_{t-1}$ 由 Phase 2 实验定 |
| $L_{replay}$ | buffer 监督回放，巩固旧能力 | $\mathbb{E}_{(s,a)\sim Buffer}[-\log\pi_{new}(a\|s)\cdot w]$，权重 $w$ 见 §5 |
| $L_{ent}$ | entropy 正则，防 Echo Trap | **所有 Phase 固定 $\lambda_4=0.001$，不参与 ablation** |

两个易被忽视的设计点：

- **弃用 $L_{reg}$（参数 L2）**：参数距离 ≠ 功能距离，$L_{kl}$ 在输出分布空间约束更精确，原槽位让给 $L_{ent}$。
- **$L_{ent}$ 必须默认开**：GRPO 与 $L_{rl}/L_{replay}$ 都是 mode-seeking、加速 entropy 下降；KL 只保形状不保 entropy。关掉 entropy 会让 B1 baseline 崩盘，测到的就不是"真实遗忘量"而是"崩盘退化"——为保所有 Phase 可比，entropy 全程同值开启。

> 工程注入：`trainer/cl_loss.py::make_cl_loss` 返回 verl 兼容 loss_fn，replay 行用双 mask 与 PPO 行隔离（replay 行 `response_mask=0` 不污染 PPO 分母）。**2026-06-12 已审计 verl 0.8.0 兼容性并修复三处接线 bug**：loss 闭包签名对齐 verl keyword 调用（去 config 位置参）、replay log_probs 先 `no_padding_2_padding` 还原 dense（engine 原生为 NestedTensor）、replay 行构造为 left-right padded rollout-row 以过 packing 断言。纯逻辑单测通过；真 verl 全链路端到端待 GPU 集群。详见 [`VerlIntegration.md`](../../doc/VerlIntegration.md) 与 `doc/RunLog.md`。

---

## 4. 方法二：7 桶 Replay Buffer（Method · Memory）

### 4.1 为什么按能力分桶、按抗遗忘排序

- **按能力/领域分桶，不按难度**：难度随模型能力漂移，难度桶会不断重新分类；能力是稳定的划分轴。7 桶 = Workflow[54] / SysOps[52] / Dialogue[38] / Finance[18] / Communication[12] / Knowledge[11] / OfficeQA[10]（OfficeQA 单独成桶，避免并入 Knowledge 被稀释）。论证见 [`BucketDesign.md`](../../doc/BucketDesign.md)。
- **Priority 不用 reward 绝对值**：训练推进 reward 整体上升，按 reward 排序会系统性淘汰旧轨迹，buffer 退化成滑动窗口、丧失 CL 意义。改用**抗遗忘价值**：

$$priority_i = f(\text{forgetting\_risk}_i,\ \text{rarity}_i,\ \text{diversity}_i,\ \text{within\_bucket\_difficulty}_i)$$

起步融合权重设计为 (0.4, 0.2, 0.2, 0.2)，forgetting_risk 主导。v1 实现中 diversity 信号因缺 embedding pipeline 暂禁用（权重置 0），其余三信号重归一化为 (0.5, 0.25, 0, 0.25)，仍保持 forgetting_risk 主导。

### 4.2 配额、淘汰、采样

- **Quota = 保底 + 平方根加权**：$q_i = q_{min} + (C-Bq_{min})\cdot\frac{n_i^{0.5}}{\sum_j n_j^{0.5}}$。大桶得更多但不线性膨胀，小桶有 hard floor 不被挤空。
- **桶内淘汰、禁止跨桶挤出**：保证不同能力不互相侵占——这是"按能力分桶"主张的执行保障。
- **两级采样**：先采桶（soft target 比例 + 均匀混合 + starvation_boost）→ 桶内按 priority 加权随机（非 top-k，避免只刷明星轨迹）。
- **冷启动**：训练前用预采集的多轮轨迹预填 buffer（`warmup_buffer.py` → `buffer.load(sqlite)` → `_preload_warmup()`），使 $L_{replay}$ 从 step 0 就有旧经验可用。冷启动数据需求见 [`Buffer_冷启动数据需求.md`](../../doc/Buffer_冷启动数据需求.md)。

> 实现：`replay_buffer/`（纯 Python，与 verl 完全解耦，可独立单测）。

---

## 5. 方法三：U 形块权重的 Reweighted Replay（Method · Credit）

replay token 级权重（仅作用于 response token）：

$$w_t^{(i)} = \text{normalize}\Big(\text{clip}\big(\text{priority}_i \cdot \tfrac{\gamma^{\text{block}(t)} + \delta^{K_i - 1 - \text{block}(t)}}{2},\ q_5,\ q_{95}\big)\Big)$$

两维度合成：**A. trajectory 级 priority**（§4.1 四信号）× **B. token 级 U 形块权重**。

- **按动作块切分**：理想按 `<think>/<toolcall>/<observation>/<final_answer>` 等结构标签划分为 $K_i$ 块（每条轨迹 $K_i$ 不同）。实测回流数据无此类 XML 标签，v1 实现改按 **message 边界**（role=assistant/tool 的对话轮）切块；仍无可切结构时回退等长切分（fallback_K=20），过长块二次切分。
- **U 形动机（2026-06-08 反转）**：原方案是单调衰减 + final_answer 单点 boost；反转为 U 形，因为"末端的重要性不止 final_answer 一个 token 段，靠近末端的多个块（结论前总结、关键判断）都重要"，单点 boost 抓不住整段。首块=早期高方差分支决策，末块=结论生成段，中间执行细节相对降权。
- **$\gamma=\delta=1$ 内置等权基线**：所有块权重恒为 1 → 退化为均权（W0），无需单独 scheme。Phase 3 对照 **W0（$\gamma=\delta=1$）vs W2（$\gamma=\delta=0.88$）**，唯一变量是超参值。
- **clip 整体乘积**：极端值由 priority×块权重组合放大，必须对最终 $w_t^{(i)}$ 剪枝（非仅 priority）。

> 完整推导、备选方案表、与 advantage 的量级配合（端点/中点比 ~1.93×，控制在 2–3×）见 [`CL_Update_Sunhao.md` § $L_{replay}$ 权重](../../doc/CL_Update_Sunhao.md#l_replay-权重-w-计算规则)。

---

## 6. 方法四：沙箱 Rollout 与 Winner-Sync 会话调度（Method · Environment）

### 6.1 动作在内、推理在外

推理（27B 前向/采样/logprob）在沙箱外的 GPU 集群（vLLM）；动作（装包、写文件、跑命令）在腾讯云沙箱内（OpenClaw agent harness）。128 个沙箱不可能各带一份 27B，而动作必须落到"那台用户机器"的磁盘上。详见 [`Sandbox_Agent架构.md`](../../doc/sandbox/Sandbox_Agent架构.md)。

### 6.2 16×8 + winner-sync

每 step **16 会话 × 8 槽 = 128 并发**。会话内每条 query：8 槽从同一母版派生（**位级一致**）→ 8 路并行 rollout（中途发散 = advantage 方差来源）→ 选 winner → **8 槽磁盘 + 对话正史都对齐 winner** → 下一条 query。会话结束销毁全部、不回写母版。

- **为什么必须 sync 到 winner**：若不同步，下一条 query 的 8 槽从不同起点出发，组内比较混入历史路径差异，GRPO 的 credit assignment 方向出错、σ 被环境噪声污染且逐 query 累积。
- **关键约束**：会话内**绝不 kill winner**（它是累积状态的唯一活载体，含进程/内存态）——只 kill 输家 7 个、从存活 winner fork 替补。
- **轨迹收集用 verl 原生字段**（`response_mask`/`rollout_log_probs`），不自造 proxy；winner-sync 与 verl 批量 rollout 的冲突用"自管会话编排 + 复用 verl 单步生成"调和（D-2）。

### 6.3 随机用户环境（防过拟合单一文件系统）

按会话随机选 persona，对该组 8 槽施加**同一** persona 种子（确定性物化，保证 8 槽位级一致）。两层职责：调度器注入 persona seed → 实例启动确定性物化。

---

## 7. 方法五：模拟用户在线生成多轮 Query（Method · Data）

### 7.1 前提漂移问题

静态多轮数据把 $q_{k+1}$ 写死，但它通常引用 $q_k$ 的执行结果（"这个 PPT 第 3 页数据有问题"）。rollout 随机 → 前提 $\Pr[\phi(q_{k+1})(e_k)]<1$ 且随策略漂移，导致错误梯度（对不存在的问题硬编修复 / 惩罚诚实）与 GRPO 信号稀释。

### 7.2 三 agent 在线构造

真实回流数据**只保留首条 query 作种子**，后续 query 在 winner-sync 边界由三 agent 协作在线生成（前提由构造保证成立）：

| agent | 人设 | 职责 |
|-------|------|------|
| **观察 agent** | 无（客观） | **state-only，diff-driven**：以沙箱 before/after 内容级 diff（+SysOps）为 ground truth 收集中间+最终结果，产出 $R_t$。观察**模型不接收 actor 轨迹** → 反 hacking 是**结构性**的（声称从不进入观察判断与 completion）。observer LLM 可选（默认确定性、零模型调用） |
| **出题 agent** | 有（42 选 1，会话级随机） | 以人设视角读 $R_t$（状态），模拟真实用户出下一 query |
| **奖励模型** | 无（冻结 judge） | **双通道**：$R_t$ 的 state_diff 判 completion + pass-through 的 actor 轨迹判 safety/robustness；空 diff 短路不调 judge |

报告 $R_t$ **一份两用**（喂出题 + 喂奖励），保证给分与出题的事实一致；reward 的 completion 落到真实 diff 上、抗"嘴上说做完了"的 hacking；轨迹仅作 pass-through 给 reward（观察模型不看）。

- **只观察 winner**：与"会话正史 = winner 轨迹拼接"自洽。
- **防模式坍缩（四维）**：① 42 人设会话级 LLM 预选（根据 seed_query 语义匹配最佳人设，不再随机）② 会话长度完全由 Questioner 控制——满意自动 `<end_session>`、不满意继续追问，不设固定预算 ③ winner 状态逐轮演化 ④ Questioner 多模型轮换（从模型层面注入输出风格异质性）。
- **自适应课程**：出题 agent 始终对当前策略的实际输出挑刺 → 策略越强、刺越细，难度自动跟随能力前沿。

> 完整设计（接口契约、决策记录、风险）见 [`UserSim_多轮Query在线生成.md`](../../doc/UserSim_多轮Query在线生成.md)。
> **实现状态（2026-07-07）**：三 agent 已落盘 `agents/`（observer/questioner/reward + 42 人设库 + 乘法衰减耐心 `PatienceTracker(P0,r)`）。多轮冷采集已跑通：`sandbox_grpo_collect.py` 在腾讯沙箱内驱动 hermes（`--resume` 跨轮续接），observer 看沙箱 diff 产报告，questioner 人设化追问，全场无 reward/winner。冷采集 pipeline（`run_cold_start.py` → `run_cold_pipeline.sh`）覆盖打桶(静态6种+LLM)→人设预选→采集(incremental)→parquet→warmup。1029 条不可跑任务已 LLM 筛选剔除。模型选型见 [`configs/agents.yaml`](../../configs/agents.yaml) + [`doc/模型选型.md`](../../doc/模型选型.md)。待集群：接 verl 原生 rollout、启动 GRPO 8-slot 正式训练。

---

## 8. 实验设计（Experiments）

### 8.1 评测

**ClawEval**（300 任务，3 split：General 161 / Multimodal 101 / Multi-turn 38；当前用纯文本 195）。评分 $score = s_{safety}\times(0.8\cdot s_{completion}+0.2\cdot s_{robustness})$，**Pass³**（三次独立运行全过）。Reward 与评测同构（模型 judge），保证 reward/eval 一致。评测元数据见 [`ClawEval_Metadata.md`](../../doc/ClawEval_Metadata.md)。

### 8.2 21 个受控消融（逐组件验证）

```
Phase 1 (B1)                            纯 RL 遗忘下界
   ├── Phase 2 (K1–K5, K2-R)            KL：λ₂ 扫描 / πref 锚点 / KL×Replay
   └── Phase 3 (R0,R3,R4,R5,R4-w,R6,R4-K) Replay：桶结构 vs CLEAR / priority 类型 / U 形权重
           └── Phase 4 (C1–C4)          KL × Replay 2×2 组合
                   └── Phase 5 (S1,S2)  rollout 规模 1024×8 → 4096×8
                           └── Phase 6 (X*) 按需探索
```

每个对照只动一个变量（如 R3→R4 = 是否用 priority，R4→R5 = 抗遗忘 vs reward priority，R4→R4-w = 是否加 U 形权重），保证边际贡献可归因。**多轮 query 构造方案不单列消融**（实验已饱和）——靠过程监控 + ClawEval Multi-turn 终点指标验证。

### 8.3 过程指标（Echo Trap / 遗忘早期预警）

Output Entropy 曲线（前 100 step 降 >50% 即调大 $\lambda_4$）、Trajectory Diversity（组内 distinct-n / self-BLEU）、KL 趋势、$L_{replay}/L_{rl}$ 比值、各 loss 分量梯度范数、buffer 各桶动态（`trainer/replay_metrics.py` 落 sidecar）。

---

## 9. 系统与基础设施（System）

| 维度 | 方案 |
|------|------|
| 部署 | 64×H800，**分离 40 推理 + 24 训练**（Deep Research tool exec 4–8s/turn 下比 Colocate 快 3–7%）；Colocate 64 为后备 |
| 异步 | **Fully Async Policy**（verl），`staleness_threshold=0.3` 严格控制；可平滑退化为同步 |
| 精度 | **BF16 全栈** + FP32 主权重 + FP32 Adam m/v；FP8 不进主路径（Phase 5 可选 FP8 rollout-only） |
| 训练框架 | verl 0.8.0，pip 安装不 fork，唯一注入点 `actor.set_loss_fn(cl_loss)` |
| 工程解耦 | `replay_buffer/` 不 import verl/Ray，可独立单测 |
| 沙箱厂商无关 | rollout 沙箱走**接口/实现解耦的注册表**（`SandboxClient` 协议 + `register_backend`）：后端 `local`（dev）/ `e2b`（腾讯）/ `aliyun`（Alibaba AgentBay，留空待实现）按名互换、rollout loop 零改动；三 Agent 配 `scripts/agents_harness.py` **离线 harness**（无 GPU 跑通 observer/questioner/reward 回路，利于复现） |

> ⚠️ `CL_Update_Sunhao.md` 中按 70B 估算的显存/耗时数字待按 27B 重算（标记 C2）。落地施工图见 [`Plan_训练链路补齐.md`](../../doc/Plan_训练链路补齐.md)，状态见 [`Progress.md`](../../doc/Progress.md)。

---

## 10. 局限与未决（Limitations）

- **U 形 / priority 融合权重未实证调优**：$\gamma=\delta=0.88$、设计 $\alpha=(0.4,0.2,0.2,0.2)$（v1 实跑 diversity 禁用 → $(0.5,0.25,0,0.25)$）均为起步值，待 Phase 3 数据校准；短轨迹（小 $K_i$）下 U 形可能近乎消失，需更小 $\gamma$。
- **模拟用户分布失真 / 幻觉**：42 人设分布 ≠ 真实 follow-up 分布；观察不全时前提漂移以小概率回归（靠首轮真实锚点 + 前提成立率审计缓解）。
- **观察 grounding 强度 = 取证能力**：反 reward-hacking 的强度上限 = observer 能取到的证据强度。**已落地 diff-driven、observer 模型只看 state**（2026-06-19）：observer 以沙箱 before/after **内容级 diff**（含二进制格式提取 + SysOps 状态）为 ground truth，**不接收 actor 轨迹**（结构性反 hacking——声称从不进入观察判断与 completion）；轨迹仅 pass-through 给 reward 判 safety/robustness。详见 [`Observer_DiffDriven_技术报告.md`](../refs/Observer_DiffDriven_技术报告.md)。**残余局限**：二进制提取的真值核对需库 + 真实文件（本机仅验 fallback）；瞬态/被覆盖的中间产物需 `watch_dir` 事件流（当前只看净变化）；真实 e2b/aliyun 后端连通 + 8 槽 FS/SYS baseline 正确性待集群验证。
- **winner-sync 进程态保真**：平台若只支持磁盘快照，替补槽的进程/内存态可能与 winner 不一致（头号 PoC）。
- **judge 选型未定**：须用 ClawEval 人工 rubric 一致率校准；外部 judge API 地址待提供（reward 走 `JudgeClient` env 注入，代码零改动）。
- **27B 成本数字待重算**；**全栈 64 卡 smoke 未跑**——Loss 链路三处接线 bug 已审计修复且纯逻辑单测通过（约 200 测试函数），但 replay 行真过 verl forward + log_probs 选回 + packing 断言不触发，仍待集群 1-step 全栈验证。`inference/VerlRolloutGenerateFn` 为占位（接 verl 原生 generate 是 Gap D）。

---

## 11. 文档地图（论文各节 → 仓库单一信源）

| 论文章节 | 信源文档 |
|---------|---------|
| 方法·CL Loss / 实验路线 / GPU / 精度 / 文献 | [`CL_Update_Sunhao.md`](../../doc/CL_Update_Sunhao.md)（**主文档**） |
| **模型选型**（actor / observer / questioner / judge） | [`模型选型.md`](../../doc/模型选型.md)（**单一信源**） |
| 方法·7 桶 Buffer 论证 | [`BucketDesign.md`](../../doc/BucketDesign.md)（+ `_compressed` 速查） |
| 方法·环境/调度 | [`Sandbox_管理调度指南.md`](../../doc/sandbox/Sandbox_管理调度指南.md)、[`SandboxRollout.md`](../../doc/SandboxRollout.md)、[`Sandbox_Agent架构.md`](../../doc/sandbox/Sandbox_Agent架构.md) |
| 方法·多轮数据 | [`UserSim_多轮Query在线生成.md`](../../doc/UserSim_多轮Query在线生成.md) |
| **方法·三 agent 论文摘要** | [`Paper_ThreeAgent_Summary_CN.md`](Paper_ThreeAgent_Summary_CN.md) |
| **方法·三 agent 技术设计** | [`UserSim_三Agent架构与技术设计.md`](../../doc/UserSim_三Agent架构与技术设计.md) |
| 实验·评测 | [`ClawEval_Metadata.md`](../../doc/ClawEval_Metadata.md) |
| 系统·落地 | [`VerlIntegration.md`](../../doc/VerlIntegration.md)、[`Plan_训练链路补齐.md`](../../doc/Plan_训练链路补齐.md) |
| 进度 | [`Progress.md`](../../doc/Progress.md) |
| **论文主旨 + Introduction 初稿** | [`Paper_Intro_draft_CN.md`](Paper_Intro_draft_CN.md) / [`Paper_Intro_draft_EN.md`](Paper_Intro_draft_EN.md)（单主旨框架：一个核心信息 + 部件降格为手段） |
| **论文 Method 散文初稿** | [`Paper_Method_draft_CN.md`](Paper_Method_draft_CN.md)（中文）/ [`Paper_Method_draft_EN.md`](Paper_Method_draft_EN.md)（英文投稿用） |

> 阅读顺序：本文（全貌）→ `CL_Update_Sunhao.md`（技术细节）→ `BucketDesign.md`（分桶论证）→ Sandbox 三件套（环境）→ `UserSim_*`（多轮数据）→ `ClawEval_Metadata.md`（评测）。
