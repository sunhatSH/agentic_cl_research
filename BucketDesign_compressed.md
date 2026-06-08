# Continual Learning Replay Buffer 桶结构设计

## 目标
模型持续学习新任务时防止灾难性遗忘。遗忘沿能力类型发生，而非沿难度发生，因此 Replay Buffer 服务于**能力保持**，非高 reward 保留。仅纯文本任务。

---

## 一、设计原则

1. **按能力/领域分桶**，不按难度分桶（难度会漂移）
2. **每桶保底配额**，防高频任务挤空低频桶
3. **桶内淘汰**，禁止跨桶挤出
4. **Priority 不用 reward 绝对值**（reward 整体上升会系统性淘汰旧轨迹，buffer 退化为滑动窗口）

---

## 二、7 桶结构

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

- OfficeQA 单独成桶：办公语境与一般 knowledge 遗忘模式不同，并入会被稀释
- 不纳入 multimodal(4)：模态不同、数据太少、目标不一致

---

## 三、Quota 分配

**保底 + 次线性加权**（推荐 α=0.5 即平方根）：

$$q_i = q_{min} + (C - B \cdot q_{min}) \cdot \frac{n_i^{0.5}}{\sum_j n_j^{0.5}}$$

- $C$=总容量, $B$=桶数(7), $q_{min}$=每桶保底, $n_i$=桶内任务数
- 大桶得更多但不按比例膨胀，小桶有保底不被挤空

**25k 示例**（$q_{min}$=2000）：Workflow≈4319, SysOps≈4272, Dialogue≈3938, Finance≈3337, Communication≈3092, Knowledge≈3047, OfficeQA≈2995

工程上分两层：hard floor（不可跌破）+ soft target（超则加速淘汰，低则加速接纳）

---

## 四、Priority 构成（抗遗忘价值，非 reward）

1. **Forgetting Risk**：当前模型在该轨迹上性能回退程度
2. **Rarity**：桶内低频模式/模板，防热门模板占满
3. **Diversity/Redundancy**：与桶内已有轨迹的重复度（每 query 仅 2 条轨迹，去重尤其重要）
4. **Within-bucket Difficulty**：桶内相对难度，覆盖边界/复杂场景

高 priority = 代表旧能力 + 已出现退化 + 稀有 + 不重复 + 覆盖边界

---

## 五、淘汰：桶内淘汰，禁止跨桶

新轨迹 → 映射到所属桶 → 桶未满直接接纳 → 桶满则只在本桶内淘汰低 priority 轨迹。保证不同能力不互相侵占。

---

## 六、采样：两级采样

1. **采桶**：混合策略——部分按 quota 比例 + 部分按均匀，兼顾大桶覆盖与长尾能力
2. **桶内采轨迹**：按 priority 加权随机采样，不贪心选 top-k，避免只重复"明星轨迹"

---

## 七、Buffer 组织与检索

每条轨迹元数据：trajectory_id, bucket, source_task/category, priority, insert_step, last_replay_step, replay_count, token_length, pattern_id/template_id, 遗忘状态标记

主存储 + 轻量索引（按 ID / bucket / priority / pattern），支持精确查询与训练采样分离。第一版查询：桶全部轨迹、top/bottom-k priority、长期未 replay 轨迹、指定 pattern/task_id 轨迹。

---

## 八、一句话总结

**能力分桶保覆盖 → 保底 quota 防挤空 → 抗遗忘 priority 保质量 → 桶内淘汰防侵占 → 两级采样保 replay 有效性 → 元数据+索引保可观测可控**
