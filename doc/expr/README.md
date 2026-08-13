# RL 算法迭代 & 实验记录

> 负责人:@孙豪
> 本目录(`doc/expr/`)存放实验的**数据、图、结论**——与 `doc/debug/`(调试排障过程)、
> `doc/archive/RunLog.md`(逐条运行流水)分工不同:这里是**面向结论**的实验档案。
> - `assets/` — 实验结果图(reward/loss 曲线等)
> - `data/`   — 实验数据(每 step 桶信息、metrics 快照、关键数值表)

---

## 一、算法迭代

### ✅ DONE

**1. Reward 算法改进(增加评判维度)**

从单一 correctness 扩展为多维加权,并引入 safety 作为整体门控(乘性):

```python
if task_done:
    reward = (
        0.4 * correctness      # 答案正确性(LLM judge)
        + 0.4 * trajectory     # 轨迹质量(工具调用/无冗余步/推理连贯)
        + 0.2                  # 完成基线分
    )
else:
    reward = 0.4 * trajectory  # 未完成:只按轨迹质量给分

reward = reward * safety       # safety(0~1)作为整体门控,危险操作直接压低
```

- **correctness / trajectory / safety** 均由外部冻结 LLM judge 打分(见 `doc/ops/reward.md`)。
- safety 乘性门控:再高的 correctness+trajectory,若有危险操作也会被 safety 压低 → 抗 reward hacking。
- 未完成任务仍给 trajectory 分(过程有价值),但无完成基线分。

**Reward judge 选型(2026-08-05,n=20 轨迹 × 8 次重复,项目正式 rubric)**

| 模型 | 稳定度(within-traj std) | 延迟均值 | 延迟 max | judge_error | 备注 |
|------|------|------|------|------|------|
| `google/gemini-3.5-flash-lite` | **0.004** | **2.2s** | **5.9s** | 0% | ✅ 主 judge(已采用) |
| `openai/gpt-5.6-luna`(开思考) | 0.036 | 8.9s | 48s | 0% | 稳定度差 gemini 9 倍 |
| `stepfun/step-3.7-flash`(开思考) | 0.038 | 15.0s | 48s | 0% | 稳定度同 gpt 量级,延迟更慢 |
| `deepseek/deepseek-v4-flash`(开思考) | 0.048 | 19.2s | 104s | 0% | 备用 judge |
| `gpt-5.6-luna`(关思考) | 0.045 | 7.1s | 36s | 0% | 关思考后变差 |
| `deepseek-v4-flash`(关思考) | 0.088 | 2.4s | 5.7s | **15%** | 关思考后稳定度翻倍 + 出 judge_error |

结论:**gemini 全面最优**,稳定度碾压、延迟最低、无 error。关思考对 deepseek/gpt 都是负优化
(deepseek 关思考后 15% 请求异常)。**所有模型保持开思考**,主 judge 用 gemini。
选型脚本:`rewardmodel_choose/compare_reward_models.py`,原始结果:`rewardmodel_choose/results/`。

**2. 四个 4 卡旧实验结果(baseline,无 CL 算法)**

用**专门为 Continual Learning 适配的数据**训练,**不加入自己的 CL 算法**(纯 baseline)。
结论:**loss 和 reward 震荡,无明显提升——符合预期**(baseline 没有防遗忘机制,
按桶顺序训练时旧桶能力不被保护,reward 在 0.5~0.85 间震荡不上行)。

- 四实验 b1/k1/k2/k3,各训练到 step ~193-208,reward 均值高度一致(0.60~0.62)。
- ⚠️ 此批为**旧 reward + 修复前代码**跑出,仅作"baseline 无提升"的定性佐证,**不作正式对照数据**。
  正式的 baseline vs CL 对照将用**新 reward 算法 + 16 卡**重跑(本周出),图与数据届时补入
  `assets/` / `data/`。

### 🔲 TODO

1. **迁移到新算法 + 16 卡**:当前结果基于 4 卡 + 旧 reward,需迁到新 reward 算法 + 16 卡
   重新训练,得出算法结论。
2. **CL 算法 vs baseline 对比**:用改进的 Continual Learning 算法训练,对比 reward/loss
   相对 baseline 的进步(证明 buffer 回放 + CL loss 的防遗忘收益)。

---

## 二、RL 实验(工程)

### ✅ DONE

**1. 修复图片进入训练导致崩溃的 bug —— 训练现已稳定**

模型多模态关闭(纯文本)时,agent 工具产出的图片进入推理会打崩 Qwen3.5 的 M-RoPE
(`get_mrope_position` start_idx=None),进而 refcount 泄漏 → pause 死锁 → 整训练 hang
(完整死亡链见 `doc/archive/RunLog.md` §62)。

修复:cutlass-dsl 4.6.1→4.3.4(修 RoundingModeKind)+ 视觉开但冻结(TEXT_MODEL_ONLY=1)
+ 音频关(避 audio_config 崩)+ pause_generation 有界放行(根治死锁)。

- **调整前**:16 卡反复死于 step 6/32(图片死锁),无法持续训练。
- **调整后**:16 卡 baseline 首次跑通,持续推进,reward 健康(0.50~0.68),无死锁、无图片崩。

> 稳定性验证曲线将随**新 reward + 16 卡正式实验**一并给出(本周),存入 `assets/`。

**2. 冷启动采集完成** —— buffer 种子数据就绪(含完整 tool/think 轨迹)。

**3. 解决 OOM 致命点** —— 对齐的 batch/序列 size 一开始过大,到训练阶段 OOM;
降低 size 后可跑,但 9B 模型产生的轨迹长度较长,**被截断/丢弃的数据占比显著增加**(需权衡)。

**4. 加速取舍的代价(已识别)** —— 之前为加速降了 batch_size、降了 rollout max_turn、
降了 max_token,**可能影响长任务表现**。后续正式实验需评估是否放开。

**5. 按桶顺序训练的期望趋势(2026-08-06 首次观察到)** —— b1_16gpu restart with
gpt-5.6-luna + deduction rubric,前 4 步全部在 workflow 桶内。

训练数据桶映射（`datasets/train.parquet`, 45242 行, `shuffle=false`, batch=32）:
```
step 1: rows 0-31   → workflow
step 2: rows 32-63  → workflow
step 3: rows 64-95  → workflow
step 4: rows 96-127 → workflow
```

同桶内 reward 趋势:
```
┌──────┬────────┬───────────┬─────────────┬────────┬─────────┐
│ step │ reward │   range   │  advantage  │  loss  │ entropy │
├──────┼────────┼───────────┼─────────────┼────────┼─────────┤
│   1  │ 0.358  │ 0.00–0.98 │ −2.22~+2.30 │ −0.018 │ 0.336   │
├──────┼────────┼───────────┼─────────────┼────────┼─────────┤
│   2  │ 0.427  │ 0.00–0.93 │ −2.39~+2.09 │ +0.011 │ 0.286   │
├──────┼────────┼───────────┼─────────────┼────────┼─────────┤
│   3  │ 0.458  │ 0.00–0.96 │ −2.10~+2.04 │ −0.123 │ 0.257   │
├──────┼────────┼───────────┼─────────────┼────────┼─────────┤
│   4  │ 0.476  │ 0.00–0.98 │ −2.32~+2.46 │ −0.056 │ 0.275   │
└──────┴────────┴───────────┴─────────────┴────────┴─────────┘
```

同一桶内 reward 持续上升(0.358→0.427→0.458→0.476)，这是正常的学习信号——模型
在工作流编排任务上逐步进步。**当训练切到下一个桶(ops)时，reward 预期会下降或波动**，
这是灾难性遗忘的体现，也是整套持续学习系统要解决的问题。**不是 bug，是基准现象**
——正是这个"换桶下降"才能衡量 CL 算法（九桶 buffer + replay）的防遗忘收益。
记录下来以免后续看到换桶下降时误判为训练出问题。

### 🔲 TODO

1. **用新 reward 算法跑 16 卡对照实验,启动正式实验。**
   本目录需持续存放:
   - 实验图(reward/loss 曲线)→ `assets/`
   - 实验数据 → `data/`:含**每 step 的桶信息**(某 step 用的是哪个桶的数据)、
     metrics 快照、关键数值表,供后续防遗忘评测按桶对齐分析。

---

## 附:数据来源与复现

- metrics 原始:`logs/metrics/qwen35_9b_<exp>/metrics.jsonl`(verl FileLogger)
- 画图脚本:`scripts/plot/plot_metrics_compare.py`(多实验叠加 / 单实验)
- 逐条运行流水:`doc/archive/RunLog.md`
- reward 维度定义:`doc/ops/reward.md`
