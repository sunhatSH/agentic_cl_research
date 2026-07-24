# 9 桶 Replay Buffer 算法规范

> **用途**：桶 replay buffer 的单一权威规范。所有实现和设计讨论以本文档为准。
> **代码**：`replay_buffer/`（`bucket.py`, `sampler.py`, `priority.py`, `eviction.py`, `weighting.py`, `store.py`）
> **配置**：`configs/base.yaml` § `cl.buffer`
> **引用本文件的文档**：`doc/source/CL_Design.md`, `doc/source/训练与推理流程.md`
> **最后更新**：2026-07-23（桶级采样改为 distance 距离加权；quota/uniform 归档；CLEAR 基线明确为全池均匀）

---

## 1. 桶定义

按**能力/领域**分桶，不按难度分桶（难度随模型能力提升漂移）。

桶体系（2026-07-04）= ClawEval 官方 category 合并（去多模态），**单层、无子桶**。

| # | 桶名 | task_count | 核心能力 | 来源 category |
|---|------|-----------|---------|-------------|
| 0 | workflow | 56 | 多步骤任务组织 | workflow + productivity + organization |
| 1 | ops | 44 | 工具使用与系统操作 | ops + operations + terminal + file_ops |
| 2 | qa | 36 | 问答检索与阅读理解 | what + knowledge + comprehension + memory |
| 3 | finance | 20 | 结构化业务规则 | finance + procurement |
| 4 | office | 11 | 办公文档与数据处理 | office_qa + data_analysis |
| 5 | communication | 11 | 表达与沟通 | communication + content + rewriting |
| 6 | safety | 9 | 安全合规与风险判断 | safety + security + compliance |
| 7 | coding | 2 | 代码编写与调试 | coding |
| 8 | research | 6 | 多源检索并综合成报告 | research + synthesis |

纯文本 buffer 总任务数 = 195（含 12 条 user_agent 多轮按首轮归桶，去多模态）。

配置：
```yaml
bucket_names: [workflow, ops, qa, finance, office, communication, safety, coding, research]
bucket_task_counts: [56, 44, 36, 20, 11, 11, 9, 2, 6]
```

---

## 2. 配额分配（allocate_quota）

### 2.1 公式

```
soft_target_i = C × n_i^α / Σ_j n_j^α
```

- `C = total_capacity`（总容量，默认 25000 条轨迹）
- `n_i = bucket_task_counts[i]`（第 i 桶任务数）
- `α = 0.5`（平方根加权，次线性压缩大小桶差距）

整数取整余数由最大桶吸收，保证 Σ soft_target = C。

### 2.2 为什么 α=0.5

| α | 含义 | workflow:coding 比 | 问题 |
|----|------|------|------|
| 1.0 | 按比例 | 9.3:1 | 大桶碾压，长尾饿死 |
| 0.5 | 平方根 | 3.1:1 | 大桶温和倾斜，小桶被抬高 |
| 0.0 | 均匀 | 1:1 | 无视任务分布，主流能力吃亏 |

### 2.3 当前配额（C=25000, α=0.5）

| 桶 | 占比 | soft_target |
|------|------|------|
| workflow | 19.7% | 4915 |
| ops | 17.4% | 4354 |
| qa | 15.8% | 3938 |
| finance | 11.7% | 2935 |
| office | 8.7% | 2177 |
| communication | 8.7% | 2177 |
| safety | 7.9% | 1969 |
| coding | 3.7% | 928 |
| research | 6.4% | 1607 |

配置固化值：
```yaml
bucket_caps: [4915, 4354, 3938, 2935, 2177, 2177, 1969, 928, 1607]
```

---

## 3. 每桶硬下限（bucket_floors）

### 3.1 定义

```
floor_i = cap_i / 30
```

每桶独立的淘汰保护线。桶内轨迹数低于 floor 时**不可淘汰**——哪怕整条 bucket priority 最低。

### 3.2 当前值

```yaml
bucket_floors: [163, 145, 131, 97, 72, 72, 65, 30, 53]
```

| 桶 | cap | floor |
|------|------|------|
| workflow | 4915 | 163 |
| ops | 4354 | 145 |
| qa | 3938 | 131 |
| finance | 2935 | 97 |
| office | 2177 | 72 |
| communication | 2177 | 72 |
| safety | 1969 | 65 |
| coding | 928 | 30 |
| research | 1607 | 53 |

### 3.3 设计理由

- 不设统一的 floor 常数（小桶数据少，统一高 floor 不可能凑到；大桶保护过度）
- cap/30 保底：下限随桶规模缩放，合计 ~833 条（占 25k 的 3.3%，不挤占容量）
- floor 以下的轨迹永不被淘汰，保证每种能力都不会在 buffer 里完全消失

---

## 4. 插入与淘汰（add_trajectory）

### 4.1 流程

```
add_trajectory(trajectory, bucket, metadata)
  1. 计算 priority（见 §5）
  2. pioneer_boost（桶接近空时新轨迹 priority += 衰减加成，保护初期样本）
  3. while 桶 size >= soft_target[bucket]:
       if 桶 size <= bucket_floor: break    ← 硬保护
       victim = 桶内最低 priority 轨迹
       删除 victim
       eviction_count[bucket]++
  4. 插入新轨迹
```

### 4.2 关键约束

- **桶内淘汰**：只在自己桶内竞争，**禁止跨桶挤出**
- **floor 保护**：floor 以下永不淘汰
- **soft_target 是稳态上限**：插入时先淘汰再插入，size 不会超过 soft_target

### 4.3 Reservoir 模式（R0 CLEAR 基线）

`eviction_type=reservoir` 时用经典 reservoir sampling：未满直接接纳，满了以 `cap/n_seen` 概率随机替换。这是 Phase 3 R0 的 control 条件。

---

## 5. Priority（4 信号融合）

### 5.1 设计原则

Priority **不用 reward 绝对值**。训练推进时 reward 整体上升，按 reward 排序会使新轨迹系统性压制旧轨迹、buffer 退化为滑动窗口。Priority 反映的是轨迹**对防止遗忘的价值**。

### 5.2 公式

```
priority = 0.5 × forgetting_risk + 0.25 × rarity + 0.0 × diversity + 0.25 × difficulty
```

| 信号 | 权重 | 公式 | 含义 |
|------|------|------|------|
| forgetting_risk | 0.5 | clip(avg(orig_logprob − curr_logprob), 0, 1) | 模型对该动作的概率下降越大 → 遗忘风险越高 |
| rarity | 0.25 | 1/(1 + ln(1 + bucket 内同 pattern 数)) | 稀有 pattern 更值得保留 |
| diversity | 0.0 | 1 − max_cos_sim(emb, peers) | v1 禁用（无 trajectory embedding 管线） |
| difficulty | 0.25 | 1 − success_rate | 非平凡样本更值得回放 |

### 5.3 变体

| 变体 | 用途 |
|------|------|
| `Priority(alpha=(0.5,0.25,0.0,0.25))` | 主方案（v1，diversity 禁用） |
| `UniformPriority()` | R3 ablation（所有 priority=1.0 → 均匀） |
| `RewardPriority()` | R5 ablation（reward 绝对值 priority） |

---

## 6. 采样（TwoLevelSampler）

> **2026-07-23 重大变更**：桶级采样从"quota 比例 + 均匀 + 饥饿加成"改为 **distance 距离加权**。`UniformStrategy` / `QuotaStrategy` / `make_strategy` 注册表已删除归档，桶级只留 `DistanceStrategy`（主方案）与 `BaselineSampler`（CLEAR 基线）。原因见 §6.1。

### 6.1 为什么改成 distance

旧 quota 策略的桶级权重 = `0.7 × soft_target/C + 0.3 × 1/K + starvation`，**不感知当前在训哪个桶**——训 finance 时和训 ops 时回放分布一样。这违背防遗忘的核心：**训新桶时应多回放离它远（易忘）的旧桶**。

distance 策略让回放权重随"当前训练位置"移动：训 finance 时，离 finance 远的桶（如 coding/research）权重高、多回放防遗忘；离 finance 近的桶权重低。这才是定向防遗忘。

### 6.2 桶级采样（DistanceStrategy）

```
centroid = Σ_i  share_i × coords[bucket_i]        # share_i = n_i / batch_size
bucket_weight(b) = distance(b, centroid) / mean(distance)
```

- **当前 batch 的桶分布**：一个 batch 可能混桶（多轮 session 展开成单轮 GRPO 组，或一桶尾部混入下一桶头部）。"当前训练位置" = batch 内各桶按占比加权的**质心**，不是单一桶。
- **距离**：每个桶有 5 维能力坐标（`configs/bucket_coords.json`，由 `scripts/score_bucket_coords.py` 产出）。回放桶 b 的权重 = b 到质心的欧氏距离 / 平均距离。
- **远的桶权重高**：离当前训练位置远 = 遗忘风险高 = 多回放。
- **当前桶权重≈0**：在质心上的桶距离 0，几乎不回放（正在 on-policy 训练，不需要回放自己）。
- **冷启动**：`current_distribution=None`（buffer 还没数据/没跑过 batch）时退化为全桶均匀，保证每个非空桶都能被采到。

### 6.3 桶内采样

按 priority **加权随机**采样（`rng.choices(ids, weights=priorities)`），**非 top-k 贪心**——避免只重复"明星轨迹"。

### 6.4 去重

同一 batch 内同一条轨迹只出现一次。加权随机可能重抽同一个 tid，重试最多 20×batch_size 次，不够则扫剩余轨迹补齐。

### 6.5 CLEAR 基线（BaselineSampler）

`within_bucket_sampling='uniform'` 且 `eviction_type != 'reservoir'` 时走 `BaselineSampler`：**全池均匀随机**，不分桶、不感知 current、不看 priority。这是 Rolnick et al. 2019 的 experience replay 基线——唯一的 CL 机制是"保留旧轨迹并随机回放"，无任何防遗忘加权。作为对照，证明 distance 加权的增益。

---

## 7. 回放权重（TokenWeighting）

### 7.1 当前方案：W0（Flat）

```
w_t = priority × 1.0
↓
batch 内 clip [q_5, q_95]  →  归一化 Σ w_t = 1
```

只有 trajectory 级 priority 起作用，所有 token 等权。

### 7.2 备选方案：W2（U 形块权重）

```
w_t = priority × (γ^block(t) + δ^(K - 1 - block(t))) / 2  ,  γ=δ=0.88
↓
clip + normalize
```

1. 按 OpenAI chat message 边界（assistant / tool role）切分为 K_i 个块
2. 首块和末块 weight ≈ 1.0，中块 ≈ 0.56（γ=δ=0.88 时），形成 U 形
3. 单块消息超 100 token 时二次切分，子块继承父块权重
4. 无法解析 message 结构时 fallback 到等长 K=20 切分

**配置**：`scheme: W0` 设为 flat，`scheme: W2` 启用 U 形。W0 = W2 的 gamma=delta=1 退化（同一代码路径）。

---

## 8. 持久化（dump / load）

SQLite 快照（`buffer_dumps/warmup.sqlite`）：
- 轨迹 + metadata → `trajectories` 表
- 计数器（_step, _seen_counts, _eviction_counts, _rejected_counts）→ `buffer_state` 表
- 单个文件完整恢复 buffer 状态

训练启动时通过 `warmup_path` 预加载，实现 cold-start → L_replay 从 step 0 就有数据。

---

## 9. 配置速查

```yaml
cl:
  lambda_replay: 0.0              # 每实验设定
  replay_batch_size: 512          # 拼在 train_batch(1024) 上，每 mini-batch 生效
  buffer:
    total_capacity: 25000
    alpha: 0.5
    num_buckets: 9
    bucket_names: [workflow, ops, qa, finance, office, communication, safety, coding, research]
    bucket_task_counts: [56, 44, 36, 20, 11, 11, 9, 2, 6]
    bucket_caps: [4915, 4354, 3938, 2935, 2177, 2177, 1969, 928, 1607]
    bucket_floors: [163, 145, 131, 97, 72, 72, 65, 30, 53]
    warmup_path: null
  weighting:
    scheme: W0                    # W0=flat, W2=U-shaped
    gamma: 1.0                    # W2=0.88
    delta: 1.0
    segmenter: message_block
```

---

## 10. 代码索引

| 组件 | 文件 | 关键类/函数 |
|------|------|------|
| 配额分配 | `replay_buffer/bucket.py` | `allocate_quota()`, `BucketReplayBuffer` |
| 采样 | `replay_buffer/sampler.py` | `TwoLevelSampler.sample()`, `DistanceStrategy`, `BaselineSampler` |
| Priority | `replay_buffer/priority.py` | `Priority.compute()`, `UniformPriority`, `RewardPriority` |
| 淘汰 | `replay_buffer/eviction.py` | `Eviction.select_victim()`, `should_evict()` |
| 回放权重 | `replay_buffer/weighting.py` | `TokenWeighting.compute()` |
| 存储 | `replay_buffer/store.py` | `TrajectoryStore`（内存 + SQLite） |
| 训练集成 | `trainer/cl_main.py` | `build_buffer()` |
| 冷启动填桶 | `scripts/warmup_buffer.py` | — |
