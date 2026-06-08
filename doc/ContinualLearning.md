# Continual Learning Loop

**目标：** 在模拟环境中验证整个 Continual Learning loop 是否可行
**原则：** 以模型能力提升为主要目标，不迭代优化 harness
**方式：** 需要继续根据明确的任务来调研

---

## 1. 用户数据如何获得

负责人：@吴健 | 方式：真实环境模拟，模拟 user 使用 agent 任务

**优势与挑战：**
- 模拟数据质量高于真实数据，但多样性不足
- 直接按最优轨迹模拟对话，相当于跳过真实场景回流数据的质量过滤流程

**数据库构建：**
- 需要构建 user 用户画像库 + seed query 库
- 直接从回流数据中提取用户 query 作为 seed query
- 获取一个 user 对应一个具体 workspace，指定 query 时默认持续使用该 user 的 workspace
  - 问题：可能导致同一个 query 不能问多次

**多样性保证：**
- 建立行业任务库，根据回流数据做分析归类

**待解决问题：**
- 操作现有文件的行为很难模拟，能否规避？如何模拟？

---

## 2. 如何更新

负责人：@孙豪 | 方式：正常上线 2 周更新一次，使用 RL 模拟 rollout 128 更新，使用 RL 验证策略

**在 RL verl 架构上修改，支持 Continual Learning。**

**Rollout 策略调整：**
- 正常：rollout 32×8
- 调整为：1024×2 或 4096×2
- 效果：query 多样性提高，单 query rollout 下降，一个 step 更久，一个 step 存一次

# @孙豪 调研思路

```text
新任务 rollout → compute reward / advantage → 计算 L_rl
replay buffer 采样旧数据 → 计算 L_replay
当前策略与参考策略对齐 → 计算 L_kl
最后按权重汇总 loss → update
```

**核心想法：** 在现有 RL 更新流程上加入 replay 和策略约束，使模型在学习新任务时尽量减小对旧任务能力的遗忘。

CL loss：
---
$$ L_{cl} = \lambda_1 L_{rl} + \lambda_2 L_{kl} + \lambda_3 L_{replay} + \lambda_4 L_{ent} $$
---

其中各项含义如下：

- $L_{rl}$：新任务上的原始 RL 损失，是主要优化目标。
- $L_{kl}$：策略约束项，用于限制当前策略相对参考策略的漂移，缓解灾难性遗忘。
- $L_{replay}$：在 replay buffer 上进行监督回放，巩固旧任务行为能力。
- $L_{ent}$：**Entropy 正则项，最大化当前策略的输出分布展开度，防止策略坍缩（Echo Trap）。所有 Phase 默认开启，$\lambda_4 = 0.001 \sim 0.005$**。

> **不使用 $L_{reg}$（参数 L2 正则）**：$L_{reg}$ 在函数空间约束策略行为，而 $L_{kl}$ 直接在输出分布空间约束策略行为，两者目标重叠但 $L_{kl}$ 更精确（参数距离 ≠ 功能距离），因此该项权重为 0，原 $\lambda_4$ 槽位让给 $L_{ent}$ 使用。

$$ L_{kl} = D_{KL}(\pi_{new} || \pi_{ref}) = \sum_{a}\pi_{new}(a|s) \cdot \log\left(\frac{\pi_{new}(a|s)}{\pi_{ref}(a|s)}\right) $$

$$ L_{replay} = \mathbb{E}_{(s,a) \sim Buffer}[-\log \pi_{new}(a|s) \cdot w] $$

$$ L_{ent} = -\mathbb{E}_{s \sim \text{online rollout}}[H(\pi_{new}(\cdot|s))] = \mathbb{E}_{s}\left[\sum_a \pi_{new}(a|s) \log \pi_{new}(a|s)\right] $$

> **$L_{ent}$ 公式直觉**：本质就是"在 entropy 前面加个负号"。
> - $H$（entropy）想被**最大化**（让分布更平均、更不确定，防止坍缩）
> - 但训练框架是 **minimize loss**，所以加负号 → minimize $(-H)$ 等价于 maximize $H$
> - 两个等价写法的关系：$H = -\sum \pi \log \pi$（定义自带负号），所以 $-H = \sum \pi \log \pi$（两个负号抵消，看起来就没负号了）
> - **信息论 entropy 越大 = 分布越平均 = 越不确定 = 多样性越高**（与日常语义"有序/确定"正好相反）

$$ L_{reg} = ||\theta - \theta_{prev}||^2 \quad \text{（弃用，权重 0）} $$

其中，$w$ 表示 replay 样本的权重，详见下方"$L_{replay}$ 权重 $w$ 计算规则"。

> **为什么 $L_{ent}$ 默认开启（直接进 B1 配置，不放 Phase 6 探索）：**
>
> - **traj/query=2 失去方差兜底**：rollout 调整为 1024×2 / 4096×2，若某 query 上 $\pi_{new}$ entropy 塌，两条轨迹大概率走同一路径 → GRPO 的 $A \approx 0$ → 该 query 梯度信号消失。$L_{ent}$ 是唯一直接抬 entropy 的反向力量。
> - **$L_{rl}$/$L_{replay}$/$L_{kl}$ 都不能替代**：前两者 mode-seeking，加速 entropy 下降；KL 只保形状接近 $\pi_{ref}$，不保 entropy 不塌。详见 B4（Echo Trap）。
> - **成本 0**：verl/GRPO 标配 entropy bonus，无额外工程。
>
> **默认 $\lambda_4 = 0.001$**；若 output entropy 在前 100 step 下降 > 50%，调大到 0.005 ~ 0.01。

**Replay buffer 设计：**

**结构：按能力/领域分 7 桶，桶内按抗遗忘 priority 存留，禁止跨桶淘汰。**

```text
ReplayBuffer [195]
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
├── Dialogue [38]
│   ├── what [26]
│   └── user_agent [12]
├── Finance [18]
│   ├── finance [14]
│   ├── compliance [2]
│   └── procurement [2]
├── Communication [12]
│   ├── communication [8]
│   ├── content [2]
│   ├── rewriting [1]
│   └── organization [1]
├── Knowledge/Analysis [11]
│   ├── research [3]
│   ├── knowledge [2]
│   ├── synthesis [2]
│   ├── comprehension [2]
│   ├── data_analysis [1]
│   └── memory [1]
└── OfficeQA [10]
    └── office_qa [10]
```

> OfficeQA 单独成桶：办公语境与一般 knowledge 遗忘模式不同，并入 Knowledge/Analysis 会被稀释。不纳入 multimodal(4)：模态不同、数据太少、目标不一致。纯文本总计 195 任务。

**核心参数：**
- 总容量 $C$：10k–50k 条轨迹
- 桶数 $B = 7$
- 各桶配额 $q_i$：保底 + 次线性加权（见下方 Quota 分配）

**Quota 分配：保底 + 平方根加权**

$$q_i = q_{min} + (C - B \cdot q_{min}) \cdot \frac{n_i^{0.5}}{\sum_j n_j^{0.5}}$$

- $n_i$：第 $i$ 个桶的任务数，$\alpha = 0.5$（平方根，次线性）
- 大桶得更多但不按比例膨胀，小桶有保底不被挤空
- 工程上分两层：**hard floor**（$q_{min}$，不可跌破）+ **soft target**（上式计算值，超则加速淘汰，低则加速接纳）

**25k 示例**（$q_{min}=2000$）：Workflow≈4319, SysOps≈4272, Dialogue≈3938, Finance≈3337, Communication≈3092, Knowledge≈3047, OfficeQA≈2995

**Priority 定义：抗遗忘价值，不使用 reward 绝对值**

$$priority_i = f(forgetting\_risk_i,\; rarity_i,\; diversity_i,\; within\_bucket\_difficulty_i)$$

| 信号 | 含义 |
|------|------|
| **Forgetting Risk** | 当前模型在该轨迹上性能回退程度 |
| **Rarity** | 桶内低频模式/模板，防热门模板占满 |
| **Diversity/Redundancy** | 与桶内已有轨迹的重复度（每 query 仅 2 条轨迹，去重尤其重要） |
| **Within-bucket Difficulty** | 桶内相对难度，覆盖边界/复杂场景 |

> 高 priority = 代表旧能力 + 已出现退化 + 稀有 + 不重复 + 覆盖边界。**Priority 反映的是"这条轨迹对防止遗忘有多重要"，而非"这条轨迹当时取得了多高 reward"。** 随训练推进 reward 整体上升，若按 reward 绝对值排优先级，旧轨迹会系统性被淘汰，buffer 退化为滑动窗口，失去 CL 意义。

**淘汰规则：桶内淘汰，禁止跨桶挤出**
- 桶未满 → 新轨迹直接接纳
- 桶已满 → 只在该桶内淘汰最低 priority 轨迹
- 空桶首次接收 → 直接接纳，给较高初始 priority（开创性样本 boost）
- 不允许新任务跨桶挤出旧任务

**采样规则：两级采样**
1. **采桶**：混合策略——部分按 soft target 比例 + 部分按均匀，兼顾大桶覆盖与长尾能力；长期无新任务的桶给予 starvation_boost
2. **桶内采轨迹**：按 priority 加权随机采样，不贪心选 top-k，避免只重复"明星轨迹"

**冷启动处理：**
- Buffer 全空 → replay_ratio = 0，纯学新任务
- 轨迹积累未达 warmup 阈值 → replay_ratio 线性爬升至目标值
- 空桶 quota 暂不分配给其他桶，等轨迹到来时优先接纳

---

### $L_{replay}$ 权重 $w$ 计算规则

> **公式与超参的单一信源在 [`CL_Update_Sunhao.md` § $L_{replay}$ 权重 $w$ 计算规则](CL_Update_Sunhao.md#l_replay-权重-w-计算规则)**。本文档不再重复，避免双份维护漂移。

简版（截至 2026-06-08）：

$$w_t^{(i)} = \text{normalize}\Big(\text{clip}\big(\text{priority}_i \cdot \big(\gamma^{\text{block}(t)} + \delta^{K_i - \text{block}(t)}\big),\; q_5,\; q_{95}\big)\Big)$$

两维度：Priority（trajectory 级，4 信号融合）× **U 形块权重**（首尾两端高、中间低；起步 $\gamma=\delta=0.88$）。块按**动作块**（`<think>` / `<toolcall>` / `<observation>` / `<final_answer>` 等结构标签）划分，$K_i$ 因 trajectory 而异——具体标签集合与切分规则待数据到位后定，代码 fallback 用等长 $K=20$。Phase 3 对照 W0（均权）vs W2（主方案）。

> **2026-06-08 反转**：原方案为单调块衰减 + final_answer boost；改为 U 形是因为"末端的重要性不止 final_answer 一个 token 段，靠近末端的多个块都重要"，单点 boost 抓不住整段。详细论证、备选方案、与 advantage 的关系见主文档。

### 实验参数组合设计

#### 参数符号约定

| 参数 | 含义 | 默认 / 可调 |
|---|---|---|
| $\lambda_1$ | $L_{rl}$ 权重 | **固定 1.0** |
| $\lambda_2$ | $L_{kl}$ 权重 | 可调 |
| $\lambda_3$ | $L_{replay}$ 权重 | 可调 |
| $\lambda_4$ | $L_{ent}$ 权重 | **所有 Phase 固定 0.001，防 Echo Trap，不参与 ablation** |
| $L_{reg}$ | 参数 L2 正则 | **弃用，权重为 0** |

#### 全局实验路线

```
Phase 1 (B1)          建立纯 RL 遗忘基线
   │
   ├── Phase 2 (K1-K5, K2-R)             KL 单独验证 → top-2 KL 配置
   │
   └── Phase 3 (R0, R3, R4, R5, R4-w, R6, R4-K)   Replay 单独验证 → top-2 Replay 配置
           │
           └── Phase 4 (C1-C4)            KL × Replay 组合验证 → 最优 CL 配置
                   │
                   └── Phase 5 (S1, S2)   Rollout 规模扩展
                           │
                           └── Phase 6 (X1-X7)  按需探索
```

实验总数：**B 系列 1 + K 系列 6 + R 系列 7 + C 系列 4 + S 系列 2 = 20 个独立训练**。Phase 6 X 系列按需触发。

---

#### 实验数量与成本总览

| Phase | 训练数 | 备注 |
|---|---|---|
| Phase 1 | 1 | B1 |
| Phase 2 | 6 | K1-K5, K2-R |
| Phase 3 | 7 | R0, R3, R4, R5, R4-w, R6, R4-K |
| Phase 4 | 4 | C1-C4 |
| Phase 5 | 2 | S1, S2 |
| **核心总计** | **20** | |
| Phase 6 | 0~7 | X1-X7，按需触发 |

**成本估算**（单实验 ~16 GPU-day on 8×H100, ~100 step）：
- 核心 ablation：20 × 16 = **320 GPU-day**
- 8 机并行：**~5 天**；16 机并行：**~2.5 天**

---

#### Phase 1：Baseline

**验证目标**：建立纯 RL 遗忘基线，量化灾难性遗忘程度。

| 编号 | $\lambda_2$ | $\lambda_3$ | $\lambda_4$ | 配置 | 角色 |
|---|---|---|---|---|---|
| B1 | 0 | 0 | 0.001 | $\lambda_1=1.0$，纯 RL + entropy bonus | 遗忘下界 |

> **B1 必须开 $\lambda_4 = 0.001$**：traj/query=2 下关闭 entropy 会让 B1 直接训练崩盘，得到的 FM 不是真实"无 CL 手段"的遗忘量，而是"崩盘后退化"。所有 Phase 用同样的 $\lambda_4$ 保证可比性。

---

#### Phase 2：KL 单独验证

**验证目标**：KL 约束能否减缓遗忘？$\pi_{ref}$ 选什么？$\lambda_2$ 多大？KL 在有 replay 时是否仍有效？

| 编号 | $\pi_{ref}$ | $\lambda_2$ | $\lambda_3$ | 角色 |
|---|---|---|---|---|
| K1 | $\pi_0$（初始） | 0.01 | 0 | 弱 KL + 初始锚定 |
| K2 | $\pi_0$ | 0.05 | 0 | 中 KL + 初始锚定 |
| K3 | $\pi_0$ | 0.10 | 0 | 强 KL + 初始锚定 |
| K4 | $\pi_{t-1}$（上阶段 ckpt） | 0.05 | 0 | 中 KL + 阶段锚定（vs K2） |
| K5 | $\pi_{t-1}$ | 0.10 | 0 | 强 KL + 阶段锚定（vs K3） |
| **K2-R** | $\pi_0$ | 0.05 | **0.5** | **K2 + replay，交互验证（与 R4-K 对偶）** |

**对照轴：**

| 对照 | 实验组 | 变化变量 | 验证 |
|---|---|---|---|
| $\lambda_2$ 扫描 | K1→K2→K3 | 0.01→0.05→0.10 | KL 权重影响 |
| $\pi_{ref}$ 锚点 | K2↔K4，K3↔K5 | $\pi_0$ → $\pi_{t-1}$ | 全局 vs 阶段性 |
| KL × Replay 交互 | K2 → K2-R | $\lambda_3$：0 → 0.5 | KL 在有 replay 时是否冗余 |

**输出**：选出 top-2 KL 配置（K-best1, K-best2）供 Phase 4 组合使用。

---

#### Phase 3：Replay 单独验证

**验证目标**：桶结构和 priority 是否真有价值（vs CLEAR 单 buffer）？Replay 在有 KL 时是否仍有效？

**Buffer 容量约定**：CLEAR baseline (R0) 沿用原论文 **10k** 配置；其他实验统一 **25k**。

| 编号 | $\lambda_2$ | $\lambda_3$ | Buffer | 采样策略 | 角色 |
|---|---|---|---|---|---|
| R0 | 0 | 0.5 | 10k | 单 buffer + reservoir + 均匀 | CLEAR baseline，简单方案下限 |
| R3 | 0 | 0.5 | 25k | 两级采样（quota + 均匀混合） | BucketDesign 基础版（vs R0） |
| R4 | 0 | 0.5 | 25k | 两级采样 + 抗遗忘 priority | BucketDesign 完整版（vs R3） |
| R5 | 0 | 0.5 | 25k | 两级采样 + reward-based priority | priority 类型对照（vs R4） |
| R4-w | 0 | 0.5 | 25k | 同 R4 + W2 方案（priority × U 形块权重 + clip） | Reweighted Replay 主方案，详见 $L_{replay}$ 权重 $w$ 计算规则段 |
| R6 | 0 | 0.8 | 25k | 同 R4 | 高 replay 权重（vs R4） |
| R4-K | 0.05 | 0.5 | 25k | 同 R4 | 交互验证：R4 + KL（与 K2-R 对偶） |

**对照轴：**

| 对照轴 | 实验组 | 变化变量 | 验证目标 |
|---|---|---|---|
| 桶结构 vs CLEAR | R0 → R3 | 单 buffer reservoir → 两级 + 桶配额 | 桶结构在长尾分布下是否补偿稀释 |
| 是否使用 priority | R3 → R4 | 桶内均匀 → priority | 抗遗忘 priority 价值 |
| Priority 类型 | R4 → R5 | 抗遗忘 → reward | BucketDesign 核心主张 |
| Priority 用法 | R4 → R4-w | 仅采样（$w$ 等权）→ W2 方案（priority × U 形块权重 + clip） | Reweighted Replay 综合效果 |
| $\lambda_3$ 权重 | R4 → R6 | 0.5 → 0.8 | replay 权重对新/旧任务平衡 |
| KL × Replay 交互 | R4 → R4-K | $\lambda_2$：0 → 0.05 | Replay 在有 KL 时是否冗余 |

**BucketDesign vs CLEAR 维度对比：**

| 维度 | R0（CLEAR baseline） | R3（基础版） | R4（完整版） |
|---|---|---|---|
| 任务边界 | task-agnostic | 7 桶分类 | 7 桶分类 |
| Buffer 结构 | 单 buffer（10k） | 7 桶独立配额（25k） | 7 桶独立配额（25k） |
| 淘汰规则 | reservoir 随机 | 桶内均匀 | 桶内 priority |
| 采样策略 | 全 buffer 均匀 | 两级（quota + 均匀） | 两级 + priority 加权 |
| 工程复杂度 | 极低 | 中 | 高 |

**输出**：选出 top-2 Replay 配置（R-best1, R-best2）供 Phase 4 组合使用。

---

#### Phase 4：KL × Replay 组合验证

**验证目标**：KL + Replay 组合是否互补？冗余还是协同？最优组合在哪？

| 编号 | $\lambda_2$ | $\lambda_3$ | KL 配置 | Replay 配置 | 角色 |
|---|---|---|---|---|---|
| C1 | K-best1 | best | K-best1 | R-best1 | 强 KL + 强 Replay（网格 1,1） |
| C2 | K-best2 | best | K-best2（低剂量） | R-best1 | 弱 KL + 强 Replay（网格 2,1） |
| C3 | K-best1 | next | K-best1 | R-best2（低剂量） | 强 KL + 弱 Replay（网格 1,2） |
| C4 | K-best2 | next | K-best2（低剂量） | R-best2（低剂量） | 弱 KL + 弱 Replay（网格 2,2） |

> K-best1/K-best2 来自 Phase 2 top-2；R-best1/R-best2 来自 Phase 3 top-2。"低剂量"指扫描区间相对较小的 $\lambda$ 值。"组合 vs 单边"对照直接读 Phase 3 R-best1（即 $\lambda_2=0$ + R-best1 配置）的结果，无需重复跑。

**对照轴：**

| 对照 | 实验组 | 变化变量 | 验证 |
|---|---|---|---|
| 组合 vs 单边 | R-best1 → C1 | 加入 KL | KL 在最优 Replay 之上是否增量 |
| 减弱 KL | C1 → C2 | KL 剂量降低 | 高剂量 KL 是否过度约束 |
| 减弱 Replay | C1 → C3 | Replay 剂量降低 | 高剂量 Replay 是否冗余 |
| 双减弱 | C1 → C4 | 同时降低 | "中庸更优"假说 |

---

#### Phase 5：Rollout 规模扩展

**验证目标**：Phase 4 最优 CL 配置下，扩大 rollout 规模是否进一步提升效果？

| 编号 | Query × Traj | 总轨迹 | CL 配置 | 角色 |
|---|---|---|---|---|
| S1 | 1024 × 2 | 2048 | Phase 4 最优 | 小规模 rollout |
| S2 | 4096 × 2 | 8192 | Phase 4 最优 | 大规模 rollout |

---

#### Phase 6：补充探索项（按需触发）

| 类别 | 探索项 | 启动条件 |
|---|---|---|
| 高优先（有明确触发逻辑） | **X6** Forward KL 替代 reverse KL | Phase 5 entropy 仍不稳定 |
| 高优先（有明确触发逻辑） | **X7a** Adaptive $\lambda_4$（entropy 阈值反馈） | Phase 4 后某些 query entropy 不稳定 |
| 低优先（按需探索） | X1-X4 / X7b：Soft reward 加权、Advantage 温度系数、动态 $\lambda_2$/$\lambda_3$ 调度、桶间亲和度加权 replay、Per-bucket 自适应 $\lambda_4$ | 资源充足或遇到对应问题时启动 |

> **为什么 X6/X7 不进 Phase 1-4**：固定 $\lambda_2$/$\lambda_4$ 是 ablation 可比性的前提；引入动态调度会让 K2-R / R4-K / C 系列对照变量失控。X5（蒸馏）/ 时间衰减等已排除（部署约束或场景不适用）。

---

### 评测指标体系

| 指标 | 定义 | 用途 |
|------|------|------|
| **New Task Performance** | 新任务上的 reward / pass rate | 衡量新任务学习效果 |
| **Old Task Forgetting** | 旧任务评测分数相对上一阶段的下降量 | 衡量灾难性遗忘程度 |
| **CL Score** | New Task Perf − $\alpha$ · Old Task Forgetting | 综合指标，$\alpha$ 可设为 1.0 |
| **KL 趋势** | 各 step 的 $D_{KL}(\pi_{new} \| \pi_{ref})$ 曲线 | 监控策略漂移是否受控 |
| **Replay/Online Loss 比值** | $L_{replay} / L_{rl}$ 的变化趋势 | 判断 replay 与在线学习的平衡性 |
| **Advantage 分布** | 新旧任务 rollout 的 advantage 均值与方差 | 诊断梯度信号是否稳定 |
| **梯度范数** | 各 loss 分量的梯度 L2 norm | 检测某一分量是否主导训练 |
| **Output Entropy** | $H(\pi_{new}(\cdot\|s))$ 在 online rollout 状态上的均值曲线 | **Echo Trap 早期预警；前 100 step 下降 > 50% 即需调大 $\lambda_4$。traj/query=2 场景下为关键监控指标** |
| **Trajectory Diversity** | 单 query 内 2 条 trajectory 的 distinct-n / self-BLEU | **直接量化 Echo Trap；若同 query 两条轨迹趋同 → advantage 退化 → 该 query 失效** |

---

### 待解决问题

1. $\pi_{ref}$ 具体使用哪个参考策略（初始模型 $\pi_0$ vs 上一阶段 $\pi_{t-1}$）→ Phase 2 实验回答
2. replay buffer 的采样单位（整条轨迹 vs token-level segment）与采样比例 → Phase 3 实验回答
3. 抗遗忘 priority 的 4 个信号（forgetting risk / rarity / diversity / within-bucket difficulty）的具体融合公式与权重 → Phase 3 R4/R5 对比实验回答
4. 远距离桶 replay 的梯度冲突处理策略（降权 vs 投影 vs 自适应）→ Phase 6 X4 探索

### 参考文献

#### A. Replay Buffer 优化 & Continual RL

| # | 论文 | 作者 | 年份 | 核心观点 | 链接 |
|---|------|------|------|----------|------|
| A1 | Replay-enhanced Continual Reinforcement Learning | Tiantian Zhang et al. | 2018 | 存储全部历史经验 + 在线/离线混合学习 + 行为克隆，可显著减少灾难性遗忘，无需任务标识信号 | [arxiv](https://arxiv.org/abs/1811.11682) |
| A2 | Experience Replay for Continual Learning (CLEAR) | David Rolnick et al. | 2019 | task-agnostic 设定下，vanilla experience replay + on/off-policy 混合学习（V-trace + behavioral cloning）即可大幅缓解遗忘，追平需要 task ID 的 EWC/P&C；buffer 受限时随机丢弃即可接近无限 buffer 效果 | [arxiv](https://arxiv.org/abs/1811.11682) |
| A3 | Using Curiosity for Even Representation of Tasks in Continual Offline RL | Pankayaraj Pathmanathan et al. | 2023 | 用好奇心做双重功能：(1) 检测任务边界（task-agnostic 时）(2) 作为 buffer 保留优先级，替代 FIFO/reservoir；HRBTS 按任务分区 + HCB 按好奇心优先保留 | [arxiv](https://arxiv.org/abs/2312.03177) |
| A4 | DISTR: Diffusion-based Trajectory Replay for Continual RL | Feng Chen et al. | 2026 | 用扩散模型记忆旧任务的高 reward 轨迹分布，replay 时生成分布而非回放固定样本，比固定 buffer 在 stability-plasticity 上更优 | [arxiv](https://arxiv.org/abs/2603.02951) |
| A5 | ARROW: Augmented Replay for Robust World Models | Abdulaziz Alyahya et al. | 2026 | 双 buffer 设计：短期 buffer（近期经验）+ 长期 buffer（保持任务多样性的智能采样），比同大小 FIFO buffer 更抗遗忘 | [arxiv](https://arxiv.org/abs/2601.22475) |
| A6 | Improvements of Dark Experience Replay and Reservoir Sampling | Taisuke Kobayashi et al. | 2026 | Reservoir sampling 随数据流增长会对新技能新数据产生隐式偏置，存在 consolidation-plasticity 隐性 trade-off；提出改进方案 | [arxiv](https://arxiv.org/abs/2601.05787) |
| A7 | OER: Offline Experience Replay for Continual Offline RL | Sibo Gai et al. | 2023 | ER 是 CORL（连续离线 RL）最适用的算法，用小 buffer 维持多任务性能；提供 baseline 对比 | [arxiv](https://arxiv.org/abs/2302.11510) |
| A8 | Selective Experience Replay Compression Using Coresets | Guangyao Zheng et al. | 2026 | Reward 分布保持的 coreset 压缩实现 10x buffer 压缩，性能无显著下降；但高压缩比下某些任务仍退化 | [arxiv](https://arxiv.org/abs/2603.08561) |
| A9 | Prioritized Generative Replay | Renhao Wang et al. | 2023 | 最优 replay 分布可由正则化 RL 目标推导，TD-error 驱动的占据比率可将离策略数据拉向在线策略最优分布 | [arxiv](https://arxiv.org/abs/2311.11557) |
| A10 | Adaptive Replay Buffer for Offline-to-Online RL | Chihyeon Song et al. | 2025 | 固定数据混合比在 offline-to-online RL 中导致早期性能退化和上限受限；需自适应调整在线/离线数据比例 | [arxiv](https://arxiv.org/abs/2512.10510) |

#### B. LLM Agent 持续学习 & RL 训练

| # | 论文 | 作者 | 年份 | 核心观点 | 链接 |
|---|------|------|------|----------|------|
| B1 | RetroAgent: Retrospective Dual Intrinsic Feedback | Xiaoying Zhang et al. | 2026 | 在线 RL 框架，记忆 buffer 用 SimUtil-UCB 检索策略平衡相关性/历史效用/探索；在 WebShop 上超 GRPO +15.4% | [arxiv](https://arxiv.org/abs/2605.02913) |
| B2 | RFT Naturally Mitigates Forgetting in Continual Post-Training | Song Lai et al. | 2026 | RFT（Reinforcement Fine-Tuning）天然缓解遗忘，性能接近多任务训练；SFT 导致严重遗忘 | [arxiv](https://arxiv.org/abs/2507.05386) |
| B3 | Generate, Filter, Control, Replay (GFCR) Survey | Rohan Surana et al. | 2026 | LLM RL 的 rollout 策略综述，提出 GFCR 分类法；Replay 阶段包括经验回放和自进化课程；提供 rollout 病理诊断索引 | [arxiv](https://arxiv.org/abs/2601.20732) |
| B4 | Multi-Turn RL for LLM Agents: Echo Trap | Zihan Wang et al. | 2025 | 直接把 PPO/GRPO 搬到多轮 agent 场景会因 Echo Trap（重复记忆路径 → 多样性坍缩）导致训练崩溃 | [arxiv](https://arxiv.org/abs/2504.20073) |
| B5 | BEPA: Bi-level Expert-to-Policy Assimilation | Zezhou Wang et al. | 2025 | 每 task 维护一条动态 cache 轨迹，用 on-policy 成功覆盖旧轨迹，比静态 off-policy expert 显著降低分布偏移（JS 散度 0.037 vs 0.168） | [arxiv](https://arxiv.org/abs/2601.05787) |
| B6 | Continual Policy Distillation from Distributed RL Teachers | Yuxuan Li et al. | 2025 | 解耦 continual RL 为分布式单任务 RL + 策略蒸馏，恢复 >85% teacher 性能，任务遗忘 <10% | [arxiv](https://arxiv.org/abs/2507.05386) |

#### C. GUI Agent RL & 持续学习

| # | 论文 | 作者 | 年份 | 核心观点 | 链接 |
|---|------|------|------|----------|------|
| C1 | CGL: Advancing Continual GUI Learning via RFT | Zhenquan Yao et al. | 2025 | GRPO 的马太效应使高概率正优势动作获得最大更新 → 熵衰减 → 保留已学策略结构，天然抗遗忘；SFT FM=-5.73 vs GRPO FM=-0.62 | [arxiv](https://arxiv.org/abs/2504.20073) |
| C2 | Continual GUI Agents | Ziwei Liu et al. | 2026 | GUI 分布随时间变化，现有方法无法维持稳定 grounding；提出持续学习框架应对 GUI 分布偏移 | [arxiv](https://arxiv.org/abs/2603.11395) |
| C3 | GUI Agents with RL: Toward Digital Inhabitants | Junan Hu et al. | 2024 | SFT 无法处理长程信用分配/分布偏移/安全探索，RL 对 GUI 自动化至关重要 | [arxiv](https://arxiv.org/abs/2410.18082) |

#### D. 经典基础方法

| # | 论文 | 作者 | 年份 | 核心观点 | 链接 |
|---|------|------|------|----------|------|
| D1 | Gradient Episodic Memory (GEM) | David Lopez-Paz et al. | 2017 | 在 episodic memory 上使用梯度不等式约束（而非蒸馏/等式约束），可实现正向向后迁移 | [arxiv](https://arxiv.org/abs/1706.08840) |

---
---

## 3. 工具环境（Agent Framework）

负责人：@郑乃榕 | 方式：通用工具环境，不需要多样性

**待办：**
- 需要一套自己的简单 agent loop harness 实现，参考 hermes 等
- 只做必要的工具实现，约 30 个

**待确认：**
- 模型是否需要自己优化代码、优化 harness？
- 选择 openclaw / hermes / 模拟工具？

---

## 4. 评测

负责人：@杨益博 | 方式：全面能力评测在外部，一个 step 做一次评测

---

## 5. 与之前方案的区别

1. **轨迹自生成** — 模型自己产出训练轨迹
2. **大循环** — 每次迭代数据量大
