# 三 Agent 架构与技术设计（技术汇报向）

> **定位**：观察 / 出题 / 奖励三 agent 的**实现级设计说明**——模块边界、数据流、耐心机制、代码落点、env 配置、与训练链路的耦合。问题动机与论文叙事见 [`UserSim_多轮Query在线生成.md`](UserSim_多轮Query在线生成.md)；论文摘要见 [`Paper_ThreeAgent_Summary_CN.md`](../../paper/drafts/Paper_ThreeAgent_Summary_CN.md)。
> **状态**：设计定稿（2026-06-12）；代码已落盘 `agents/` + 部分 rollout 编排（2026-06-12/13）。
> **写作日期**：2026-06-13

---

## 1. TL;DR

| 项 | 决策 |
|----|------|
| **三个 agent** | Observer（客观报告）→ Questioner（人设 follow-up）+ Reward（观察 grounded 打分） |
| **插入点** | 每个 query 的 **winner 选出并 sync 之后**（`sync_to_winner` 后） |
| **共享结构** | `ObservationReport` $R_t$ **一份两用**——Questioner 与 Reward 读同一份 |
| **Observer** | 无人设；只读 winner 沙箱；**diff-driven，模型只看环境 diff、不看 actor 轨迹**；observer LLM 可选 |
| **Questioner** | 42 人设，**一会话抽 1 个、全程固定**；不碰沙箱 |
| **Reward** | 复用 `trainer/model_reward.JudgeClient`；**双通道**：state_diff 判 completion + pass-through trajectory 判 safety/robustness；空 diff 短路不调 judge |
| **耐心** | `PatienceTracker`：$P_k = P_0 - d_0(2^k-1)$，失败路径 probabilistic redo |
| **后端隔离** | `OBSERVER_*` / `USERSIM_*` / `JUDGE_*` 三套独立 env，抗 self-preference |

---

## 2. 端到端数据流

```text
datasets/queries.jsonl
  └─ 只消费 queries[0] 作 seed q1（真实回流锚点）

SessionSandboxPool（16 会话 × 8 槽，或采集阶段 slots=1）
  │
  ├─ spawn(M, 8)                    # 位级同起点
  ├─ p ← sample_persona(16)         # 会话级人设
  ├─ K ← U{1..K_max}                # follow-up 预算
  │
  └─ for t = 1, 2, ...:
        run_query(q, H) → 8 trajectories
        w ← select_winner(T)          # 现有 reward / fallback
        sync_to_winner(w); H ← H ∥ T[w].messages
        │
        ├─ R_t ← Observer.observe(sandbox=w, actor_trajectory=T[w].messages, baseline, post)
        │         # observer 模型只看 state diff；轨迹仅 pass-through 进 R_t.actor_trajectory
        ├─ r  ← agents.reward.score_followup(query=q, report=R_t)   # 从 R_t 取 state+trajectory
        ├─ buffer.ingest(all 8 trajs, winner_score=r)
        │
        ├─ if t > K: break
        ├─ q ← Questioner.next_query(p, R_t, H)   # None == <end_session>
        └─ if failure & PatienceTracker.on_failure() == False: break
        destroy_all()
```

**与 GRPO 的关系**：三 agent 只在 **query 边界**工作；8 槽同起点、winner-sync、组内 advantage 逻辑**零改动**。

---

## 3. 三个 Agent 技术规格

### 3.1 Observer — `agents/observer.py`（**diff-driven，模型只看 STATE**）

**职责**：从只读沙箱的 **before/after 环境 diff** 产出 `ObservationReport`。observer **模型不看 actor 轨迹**——它的判断只基于真实状态增量。

**为什么 observer 只看 state、不看 actor 叙事（2026-06-19 定稿）**：训练时若让 reward/observer 以 actor 的"声称"为依据，actor 会学到"嘴上说完成、实际没交付"的策略（reward hacking）。原方案让独立 Observer 读 actor 声称去"比对 claims vs 实际"；现进一步收紧为 **observer 根本不接收轨迹**——它对环境做确定性 before/after diff（文件内容 + SysOps 状态），completion 只能由真实 diff 决定。这样反 hacking 是**结构性**的：声称从不进入 observer 的判断，也从不进入 completion。

| 字段 | 含义 |
|------|------|
| `state_diff` | **环境 before/after diff（含内容 + SysOps）= ground truth**（observer 唯一判据） |
| `final[]` | 最终交付 `{path, kind, content_excerpt}`（从 diff 填） |
| `intermediate[]` | 中间结果 `{desc, source, value_excerpt}`（`use_llm` 时模型填） |
| `discrepancies` | 状态内部红旗（空/损坏/自相矛盾；`use_llm` 时填） |
| `actor_trajectory` | **pass-through**：actor 轨迹文本——observer **组件**捎带、observer **模型不看**、仅给 reward |
| `file_tree` | 工作区文件树（fallback 证据） |
| `has_effect` | 本轮 diff 是否非空；False → 短路 reward、走失败/耐心路径 |

**实现要点**：

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

- **observer LLM 可选**（`use_llm`，默认 False）：取证证据已是文本，默认确定性建报告、零模型调用；开模型也只看 `state_diff`，**永不看 trajectory**（不浪费 token）。
- **容错**：LLM/沙箱失败 → 降级为最小报告（state_diff + tree），**不 crash 会话**。
- **空报告**：`has_effect=False`（空 diff）→ `is_empty()` 真 → 触发失败路径 + 耐心机制。
- **Env**：`OBSERVER_API_BASE` / `OBSERVER_MODEL` / `OBSERVER_API_KEY`（`agents/base.resolve_observer_client`）。

### 3.2 Questioner — `agents/questioner.py`

**职责**：读 $R_t$ + 人设 + 历史，生成下一条 user query。

| 输入 | 来源 |
|------|------|
| `persona: Persona` | `agents/personas.sample_persona(seed)` |
| `report: ObservationReport` | Observer 输出 |
| `session_history` | `session_pool.session_history`（winner 正史） |

| 输出 | 含义 |
|------|------|
| `str` | 下一条 query 文本 |
| `None` | `<end_session>` 或 LLM 失败 |

**42 人设**（`agents/personas.json`）：每人设含 `profession / preference / profile / observation_focus / tone / patience(P0) / patience_decay(d0)`。`observation_focus` 编码「整体|细节 × 形式|内容」，决定**强调报告哪一面**——Observer 仍保持客观。

**Env**：`USERSIM_API_BASE` / `USERSIM_MODEL` / `USERSIM_API_KEY`。

### 3.3 Reward — `agents/reward.py`

**职责**：以 $R_t$ 为证据，调用冻结 judge 打分。

```python
# 不重复实现 judge I/O，复用 trainer/model_reward
score_followup(query, report, judge=get_judge())   # trajectory 从 report 取
# → {score, completion, safety, robustness, judge_error[, gated]}
```

- **双通道（都来自同一份 $R_t$）**：`state_diff`（observer 状态证据）= **completion** 的 ground truth；`report.actor_trajectory`（pass-through，observer 模型没看过）给 judge 判 **safety/robustness**。completion 锚在 diff（真实效果），trajectory 说明"怎么做到的"。
- **拦截层（gate）**：`report.has_effect=False`（空 diff）→ 直接 score 0、**不调 judge**（带 `gated`）。
- **防超长**：`state_diff` 与 `actor_trajectory` 各自中间截断封顶（`_MAX_DIFF_CHARS` / `_MAX_TRAJ_CHARS`）。
- **尺度**：与 eval 同构（`safety*(0.8*completion+0.2*robustness)`），保证 reward/eval 一致。
- **Env**：`JUDGE_API_BASE` / `JUDGE_MODEL` / `JUDGE_API_KEY`（与 verl `custom_reward_function` 共用）。

---

## 4. 耐心机制 — `PatienceTracker`（`agents/questioner.py`）

### 4.1 公式

会话内第 $k$ 次**失败**后：

$$P_k = P_0 - d_0 \cdot (2^k - 1)$$

- $P_0$ = `persona.patience`
- $d_0$ = `persona.patience_decay`，缺省回退 `DEFAULT_PATIENCE_DECAY = 0.1`

### 4.2 决策逻辑

```python
class PatienceTracker:
    def on_failure(self) -> bool:
        self.fail_count += 1
        pk = self.p0 - self.d0 * (2**self.fail_count - 1)
        redo_prob = clip(pk, 0, 1)
        return rng.random() < redo_prob   # True=要求重做, False=结束会话
```

| 性质 | 说明 |
|------|------|
| 成功轮 | **不**消耗耐心，走正常 $K$ 预算 |
| 重做轮 | **不计入** follow-up 配额 $K$ |
| 随机性 | 会话级 `seeded RNG`，可复现 |
| 自封顶 | $d_0=0.1, P_0=1$ → 约 3 次失败后 $P_k<0$，等价旧方案硬上限但**随人设可变** |

### 4.3 两轴人设示例（`personas.py`）

| 人设 | $P_0$ | $d_0$ | 行为直觉 |
|------|-------|-------|---------|
| Anna（office assistant） | 1.2 | 0.08 | 好脾气、慢升级 |
| Raj（SRE） | 0.8 | 0.2 | 先给机会但升级快 |
| Tom（founder） | 0.6 | 0.3 | 低容忍、很快放弃 |

---

## 5. 代码模块地图

```text
agents/
├── schema.py          ObservationReport, Persona
├── personas.json      42 人设（数据）
├── personas.py        加载 + sample_persona()
├── observer.py        Observer.observe()
├── questioner.py      Questioner.next_query() + PatienceTracker
├── reward.py          score_followup() → JudgeClient
├── prompts.py         O3/O4/O6 system prompts + builders
└── base.py            ChatClient + resolve_*_client()

rollout/
├── simulated_session.py   8 槽 + winner-sync + 三 agent（训练路径）
└── usersim_collect.py     单槽多轮采集（observer+questioner，无 reward）

scripts/
├── collect_rollout.py     多轮 UserSim 采集入口（需远程 observer/questioner）
└── collect_cold.py        单轮冷启动（**无三 agent**，仅 seed→actor→buffer）

trainer/
└── model_reward.py        JudgeClient（Reward agent 复用）
```

---

## 6. 与现有训练链路的耦合

| 模块 | 影响 |
|------|------|
| GRPO / winner-sync | **零改动** |
| `session_history` | Questioner 读 winner 正史；生成的 query 以 user 消息进正史 |
| 入桶 | 不变；per-query + `<task_domain>` 标签 |
| q1 reward | 可保留静态 checker；**生成 follow-up 必须走模型 judge + $R_t$** |
| batch 换算 | 每会话 query 数 = $1+K$，$K$ 随机 → `gen_batch_size` 用 $\mathbb{E}[1+K]$ |
| 母版销毁 | 不变（会话结束 destroy，不回写母版） |

---

## 7. 为何只观察 Winner（实现约束 + 语义一致）

1. **正史一致**：会话谱系 = winner 轨迹拼接；Questioner 只能看见 winner 世界线。
2. **同起点**：$q_{t+1}$ 的 8 槽起点 = sync 后的 winner 状态；基于该状态的 query 与 8 槽匹配。
3. **成本**：观察 1 份；7 个输家 sync 时已销毁。

---

## 8. 采集 vs 训练：三条路径对照

| 路径 | 槽数 | 三 agent | Reward | 用途 |
|------|------|---------|--------|------|
| `collect_cold.*` | 1 | ❌ | ❌ | 冷启动 buffer（跑通优先） |
| `collect_rollout.*` | 1 | Observer + Questioner | ❌ | 多轮 query 采集验证 |
| `simulated_session` | 8 | ✅ 全套 | ✅ | GRPO 训练 rollout |

---

## 9. 过程监控（§8.1 摘要）

| 指标 | 目的 |
|------|------|
| Follow-up n-gram / embedding 多样性 | 模式坍缩早期预警 |
| 前提成立率审计 | 观察不全时的小概率漂移 |
| ClawEval Multi-turn split | 终点能力验证 |

**不增设独立消融实验**（21 实验已饱和）；质量靠过程指标 + 终点评测。

---

## 10. 待办 / 风险

| 项 | 状态 |
|----|------|
| 接 verl 原生 generate（`VerlRolloutGenerateFn`） | Gap D，占位 |
| 集群 1-step 全栈 smoke | 待 64 GPU |
| judge 与 ClawEval 人工 rubric 一致率校准 | `scripts/calibrate_judge.py` 就绪，等标注 |
| 42 人设分布 vs 真实 follow-up 分布 | 已知局限，靠多样性机制缓解 |

---

## 11. 相关文档

| 文档 | 内容 |
|------|------|
| [`UserSim_多轮Query在线生成.md`](UserSim_多轮Query在线生成.md) | 完整设计规格、§7 接口契约、Algorithm 1 |
| [`Paper_ThreeAgent_Summary_CN.md`](../../paper/drafts/Paper_ThreeAgent_Summary_CN.md) | 论文向摘要 |
| [`Paper_Method_draft_CN.md`](../../paper/drafts/Paper_Method_draft_CN.md) | Method §4.5 散文 + 附录 A prompts |
| [`Sandbox_管理调度指南.md`](../ops/sandbox/Sandbox_管理调度指南.md) | winner-sync 规则 |
| [`Sandbox_Agent架构.md`](../ops/sandbox/Sandbox_Agent架构.md) | 推理在外、动作在内 |
