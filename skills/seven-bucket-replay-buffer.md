# Skill: 9 桶 Replay Buffer 设计

## 适用场景
- 在持续学习 / 多任务 RL 中需要一个**抗遗忘**的经验回放池，且任务天然分属不同**能力/领域**。
- 需要保证「不同能力不互相侵占」、且 buffer 不退化为「只记得最近高 reward 样本」的滑动窗口。
- 需要把回放池与训练框架（verl/Ray）**解耦**以便独立单测。

## 核心步骤
1. **按能力/领域分桶，不按难度分桶**。难度随模型能力漂移，会让桶定义不稳定；能力/领域是稳定的。本项目 9 桶（ClawEval 官方 category 合并、去多模态、单层无子桶）：workflow / ops / qa / finance / office / communication / safety / coding / research。定义与映射见 `runs/_analysis/capability_buckets/buckets.json`。
2. **Quota = 保底 + 次线性加权**：`target_b = q_min + (C - K·q_min) · w_b`，其中 `w_b ∝ task_count_b^α`（α=0.5 平方根，大桶得更多但不按比例膨胀）。见 `allocate_quota`。
3. **Priority 用抗遗忘信号，不用 reward 绝对值**：4 信号融合 `(forgetting_risk, rarity, diversity, within_bucket_difficulty)`，权重和为 1。reward 整体上升会系统性淘汰旧轨迹 → 退化为滑动窗口，所以 reward-priority 仅作 R5 对照（`RewardPriority`）。
4. **桶内淘汰，禁止跨桶挤出**：over soft_target 时只在本桶内选 victim（priority 模式选最低分；reservoir 模式随机）。
5. **两级采样**：先按 quota 比例（混合均匀）采桶 → 桶内按 priority 加权（或 uniform）采轨迹。采样器**持久化**以保留 starvation 状态。
6. **token 级 U 形块权重（W0/W2）**：`(γ^block + δ^(K_i−1−block))/2`，首尾高中间低；`γ=δ=1` 即均权（W0）。
7. **持久化**：内存 backend 跑实时索引，SQLite 做快照（`dump`/`load`），支持崩溃恢复。

## 关键约束
- `replay_buffer/` **禁止 import verl 或 ray**（保持可独立单测）。
- **桶内淘汰**：`select_victim` 永远不跨桶；hard floor `q_min` 不可跌破（reservoir 模式例外——它自管容量）。
- **Priority ∈ [0,1] 可解释**：每个信号都要归一到 [0,1]（如 `rarity = 1/(1+log(1+count))`），否则加权和失去意义。
- **单桶塌缩**：`num_buckets=1` 时若继承了多桶名单，自动塌缩为单一 `"All"` 桶（R0 CLEAR 基线）。
- **U 形块索引 0 起始**：末端项必须是 `δ^(K_i−1−block)` 才能让末块得 `δ^0=1`，保持对称。
- **diversity 信号在无 embedding 管线时显式置 0**，并在 `stats()` 暴露 active signals，避免「权重非零但恒贡献 0」的隐性稀释。

## 代码锚点
- `replay_buffer/bucket.py`：`BucketReplayBuffer`（quota / add / sample / dump-load）、`allocate_quota`、`_reservoir_add`。
- `replay_buffer/priority.py`：`Priority`（4 信号 + `active_signals`）、`RewardPriority`、`UniformPriority`。
- `replay_buffer/sampler.py`：`TwoLevelSampler`（`within_bucket_sampling`、starvation boost）。
- `replay_buffer/eviction.py`：`Eviction`（`eviction_type` priority/reservoir）。
- `replay_buffer/weighting.py`：`TokenWeighting`（W0=W2(γ=δ=1)、U 形、长块 resplit 无状态泄漏）。
- `replay_buffer/store.py`：`TrajectoryStore`（索引 + `save_sqlite`/`load_sqlite`）。
- 测试：`tests/test_bucket.py`、`test_priority.py`、`test_sampler.py`、`test_eviction.py`、`test_weighting.py`、`test_store.py`。
