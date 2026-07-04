# Plan — 冷启动数据来源消融实验（Phase 0）

> **状态**：待批准（plan mode 产出）
> **定位**：独立预实验（Phase 0），不进 21 个正式实验、不污染其可比性。选出最佳冷启动数据来源配比后，将该配比**固定**为所有 21 个正式实验的 buffer 预热来源。
> **决策依据**：与 @孙豪 对齐（本次 session）——独立定位、5 臂全扫、双轨验收（冷启动自身指标 + 下游短RL）、更强模型用 `openai/gpt-5`、短RL 固定新任务+同种子、桶拆旧/新两组。

---

## 1. 实验问题

冷启动填 7 桶 replay buffer 时，actor 用哪个模型采集，对下游 CL 训练最好？

- **on-policy 一端**：全用 **Qwen3.6-27B**（sufy 节点）自采 —— 分布与训练策略同源，`L_replay` 分布偏移最小，但 27B 冷启动能力弱、轨迹质量参差。
- **off-policy 一端**：全用 **gpt-5**（更强）采 —— 轨迹质量高、桶灌得满，但行为分布异于 27B，replay 时是强 off-policy，可能引入分布偏移、与 on-policy `L_rl` 打架。
- **折中**：两者按比例混合。

**核心张力**：27B 数据"对味但可能弱"，gpt-5 数据"强但可能不对味"。本实验用数据裁决。

---

## 2. 五个对比臂

统一符号：`P0` = 冷启动数据来源消融的臂前缀。每臂灌满同一份 7 桶 buffer（`total_capacity=25000`, `q_min=2000`），**唯一变量 = 每桶内 27B 采 vs gpt-5 采的轨迹配比**。

| 臂 | 27B : gpt-5 | 角色 |
|----|-------------|------|
| **P0-A** | 100 : 0 | 纯 on-policy（分布同源基线） |
| **P0-B** | 0 : 100 | 纯 off-policy（强模型质量上界） |
| **P0-C** | 50 : 50 | 平分折中 |
| **P0-D** | 70 : 30 | 偏 27B（"27B 为主"假设） |
| **P0-E** | 30 : 70 | 偏 gpt-5（"质量为主"假设） |

**配比落在桶内**：每个桶都按该比例混合两来源（不是"某些桶全 27B、某些桶全 gpt-5"），保证配比是唯一变量、跨桶一致。

**更强模型 = `openai/gpt-5`**（sufy 现选，与 ColdRollout 文档 remote 路一致，零改动）。

---

## 3. 前置改动（采集侧，本实验必需）

**关键前提：采集一次性，五臂共享抽样（不重采）。** 两侧数据在所有配比实验**之前一次性采好**，之后每臂只是从固定的两个池子里**按不同比例抽样**——不重跑采集、不重调模型。省时省钱的核心就在这里：

```
【全实验前，一次性采集（唯一花钱/花时间的步骤）】
  27B(on-policy, sufy qwen3.6-27b) → data/rollouts/local/*.jsonl   (policy=pi0_27b)
  gpt-5.5(off-policy, sufy)         → data/rollouts/remote/*.jsonl   (policy=gpt5)
        │  两池采一次即固定，五臂共用
        ▼
【每臂：纯 CPU 抽样，秒级、零模型调用、零 GPU、可复现(--mix-seed)】
  warmup_buffer.py --ratio-27b {1.0/0.0/0.5/0.7/0.3} → P0-{A..E} buffer
```

`scripts/phase0/run.sh` 的 Stage 1 **只调 `warmup_buffer.py`**（不调任何采集脚本），已实现该语义。

**两池最低采集量**（保证最高配比臂能抽满每桶 `q_min=2000`）：

| 档位 | 27B 池 | gpt-5.5 池 | 说明 |
|------|--------|-----------|------|
| 下限 | ~14k 轨迹 | ~14k 轨迹 | 各自单独够填满一个 25k buffer 的 hard floor（P0-A / P0-B 极端臂各抽一侧） |
| 推荐 | ~20k 轨迹 | ~20k 轨迹 | 混合臂（C/D/E）抽样有余量，各桶 soft target ~80% |

> 两侧都要采够：P0-A(100:0) 只吃 27B 池、P0-B(0:100) 只吃 gpt-5.5 池——任一池不足则对应极端臂抽不满桶。采集量对照 [`Buffer_冷启动数据需求.md`](Buffer_冷启动数据需求.md) §1。

当前 [scripts/collect_rollout.py](../scripts/collect_rollout.py) 落盘 record **原先不含来源标记**——这是做来源消融/事后归因的唯一抓手，必须补：

1. **轨迹 `meta.policy` 标记**：`_traj_to_dict` 增加 `"policy"` 字段（`pi0_27b` / `gpt5`），值由 `--actor` 决定（local→`pi0_27b`，remote→`gpt5`）。
2. **`warmup_buffer.py` 按配比混合**：新增 `--mix "27b=<n>,gpt5=<n>"` 或 `--ratio 0.7`（27B 占比）参数，从两来源目录按桶配额抽样混合，写单个 buffer SQLite；每条轨迹保留 `meta.policy`。
3. **配比可复现**：混合抽样用固定 seed；输出 buffer 附带 `manifest.json` 记录每桶实际 27B/gpt5 条数（验收核对用）。

> 这些改动只在采集/预热侧，不碰 verl / trainer / cl_loss，不影响 21 实验代码。

---

## 4. 验收标准（双轨判定链）

**判定链**：冷启动自身指标先筛 → 下游短RL 做最终裁决。任一臂自身指标不达 gate 直接淘汰，不进短RL（省 GPU）。

### 4.1 轨 1 — 冷启动阶段自身指标（无 GPU、采集后即测）

对每臂采集完的 buffer 计算，全部为**硬 gate**（不达标该臂淘汰）：

| 指标 | 定义 | 验收 gate |
|------|------|-----------|
| **桶配额达标率** | 各桶 `size ≥ q_min=2000` 的桶数 / 7 | **= 7/7**（每桶必须灌到 hard floor） |
| **tool-call 合法率** | 轨迹中可被 hermes parser 正确反解的 tool_call 占比 | **≥ 90%** |
| **judge 有效分命中率** | judge 返回有效分（非 `judge_error`）的轨迹占比 | **≥ 95%** |
| **轨迹多样性** | `eval.metrics.trajectory_diversity` 的 `distinct_4` / `self_bleu_4` | distinct_4 **≥ 0.6** 且 self_bleu_4 **≤ 0.5**（防单一来源模板坍缩） |
| **平均 judge 分** | 该臂 buffer 全轨迹 judge 均分 | 报告值（非 gate，供参考——gpt-5 臂预期更高，不作淘汰依据） |

> 直接复用 [eval/metrics.py](../eval/metrics.py)（`trajectory_diversity` 已实现）+ judge（`trainer/model_reward.py`）。judge 有效分命中率复用 §截断防护的 `judge_error` 标志。

### 4.2 轨 2 — 下游短RL 验收（占 GPU，通过轨 1 的臂才跑）

**公平性设计（关键）**：5 臂**唯一变量 = 冷启动 buffer 来源**，其余全锁死。

- **固定新任务 + 同种子**：5 臂用**同一组固定新任务种子 query + 同一随机种子**跑在线 rollout，消除在线随机性对 CL Score 的污染。
- **桶拆旧/新两组**：7 桶划分为「旧能力桶」（进 buffer 做 replay 防遗忘）+「新任务桶」（只在线学、不进 buffer）。短RL 后量**旧桶遗忘**。
- **同 CL 配置**：5 臂用同一套 CL 超参（建议用 R4 配置：`λ3=0.5` + 抗遗忘 priority + 两级采样；`λ2=0` 排除 KL 干扰），只换 buffer 来源。
- **短程**：每臂 20–30 step（不追求收敛，只看趋势与稳定性）。

验收指标（复用 [eval/metrics.py](../eval/metrics.py)，全部报告 + 关键项设 gate）：

| 指标 | 定义 | 判定 |
|------|------|------|
| **CL Score** | `new_task_perf − α·forgetting`（α=1.0，`cl_score`） | **主裁决指标**：CL Score 最高的臂胜 |
| **Old Task Forgetting** | 旧桶评分相对起点下降（`old_task_forgetting`） | 越低越好；报告 |
| **New Task Performance** | 新任务桶 reward（`new_task_performance`） | 报告（诊断质量-同源权衡） |
| **L_replay 稳定性** | `L_replay / L_rl` 比值曲线（`replay_to_ratio`） | 无发散/剧烈震荡为通过 |
| **Output Entropy** | 前 20 step entropy 下降幅度（`output_entropy`） | **下降 > 50% 判该臂不稳定**（Echo Trap 预警，B4） |

**最终裁决**：通过轨 1 的臂中，**CL Score 最高**者为最佳配比；若 CL Score 接近（差 < 阈值，建议 2%），选 **Old Task Forgetting 更低** 且 **L_replay 更稳** 者（更符合 CL 目标）。

### 4.3 轨 3 —（可选）分布偏移诊断

若轨 2 出现 gpt-5 臂 CL Score 反低（怀疑 off-policy 污染），补：量化 gpt-5 数据相对 27B 策略的 logprob 比值 / KL，定位分布偏移。作解释性证据，非裁决。

---

## 5. 产出

1. **代码**：`collect_rollout.py` 加 `meta.policy`；`warmup_buffer.py` 加 `--ratio`/`--mix` + manifest。
2. **配置**：`configs/phase0/p0-a.yaml … p0-e.yaml`（继承 base，仅 `cl.buffer.warmup_path` 指向对应臂 buffer + 锁死其余变量）。
3. **脚本**：`scripts/phase0/run.sh`（采集 5 臂 buffer → 轨1 gate → 通过者跑短RL → 汇总裁决）。
4. **文档整改**（本需求第二部分）：
   - `doc/source/CL_Design.md`：新增 **Phase 0** 节（实验路线图加 Phase 0 前置分支）；**并在每个已有验收/判定型实验补「验收标准」**（见 §6）。
   - `doc/source/CL_Design.md`：§5 验收清单补来源配比 + manifest 核对项。
   - `doc/archive/Progress.md` / `doc/archive/RunLog.md`：追加 Phase 0 里程碑与运行记录。

---

## 6. 文档整改：给所有"可判定型"实验补验收标准

现状：`doc/source/CL_Design.md` 的实验路线（Phase 1–6）只有**对照轴**，无**验收 gate**——什么样算"通过 / 选中 / 失败"没有量化门槛。本次统一补齐（你的要求："需要给出验收标准的实验下都给出标准，可多试验同时验收"）。

补法（每个 Phase 的实验表后加一张「验收标准」小表，复用 `eval/metrics.py` 指标，可多实验共用同一套 gate）：

| Phase | 补的验收标准（要点） |
|-------|---------------------|
| **Phase 0**（新） | 本文档 §4 双轨判定链 |
| **Phase 1 (B1)** | 遗忘基线：训练不崩（entropy 前100step 降幅 < 50%）；记录 FM 作为下界；`L_replay=0` 校验 |
| **Phase 2 (K*)** | top-2 选择 gate：CL Score 排序 + KL 曲线受控（不发散）；K2-R 交互项判定 KL 在有 replay 时是否冗余的量化门槛 |
| **Phase 3 (R*)** | top-2 选择 gate：R4 vs R3 的 priority 增量、R4 vs R5 的 priority 类型、R0 vs R3 的桶结构增量，各设 CL Score 差异显著门槛 |
| **Phase 4 (C*)** | 组合 vs 单边增量门槛：C1 相对 R-best1 的 CL Score 提升判定；"中庸更优"假说的接受/拒绝标准 |
| **Phase 5 (S*)** | 规模扩展有效性：S2 相对 S1 的 CL Score 提升门槛 + 吞吐/稳定性 gate |
| **Phase 6 (X*)** | 各探索项的触发条件已有；补"探索成功"的接受标准 |

> 统一原则：验收指标一律引用 `eval/metrics.py` 已实现的度量，不新造指标；"多实验同时验收"= 同一 Phase 的多个臂用同一张 gate 表批量判定。

---

## 7. 不做什么（范围边界）

- **不动** verl / trainer / cl_loss / replay_buffer 核心代码（本实验只在采集+预热+配置层）。
- **不做** SFT——gpt-5 数据只进 buffer 做 replay，绝不拿去蒸馏 27B（违背 B2/C1 立论 + 污染可比性）。
- **不**把 Phase 0 计入 21 实验总数；它是前置一次性预实验。
- 短RL 验收依赖 GPU 集群；轨 1（冷启动自身指标）本机/采集机即可跑，先行。
