# 模型配额申请（sufy endpoint）

> **状态**：2026-07-01，沙箱已验证能连 `openai.sufy.com`（沙箱→sufy:443 通、`/v1/models` 返回 118 模型）。
> **用途**：给 sufy 平台申请模型 TPM 配额用。数字为经验值（已降并发 + 含余量），非理论峰值。
> **信源**：模型选型见 [`doc/source/usersim.md`](模型选型.md)；调度见 [`doc/source/训练与推理流程.md`](../source/训练与推理流程.md)。

---

## 0. 一句话

- **endpoint**：`https://openai.sufy.com/v1`（OpenAI 兼容，沙箱可达）
- **真正的大头只有 2 个**：Actor `openai/gpt-5` + Reward `anthropic/claude-4.8-opus`，且**错峰**（reward 在 rollout 结束后才打，不与 actor 同时峰值）
- 其余 5 个模型频次低（questioner 轮换 + observer 默认不调），各 50-200 万 TPM 足够

---

## 1. 并发规模（已降）

| 项 | 值 |
|---|---|
| 同时并发 query | **64**（从 256 降） |
| 每 query 并行路（GRPO 组） | 8 |
| **总并发路数** | **512** |
| 每路 ReAct 步数 | ~6 |
| 每路每分钟轮次 | ~2（rollout ~30s） |

> 降并发原因：256×8=2048 路纯并发 TPM 需求 ~1 亿，sufy 大概率批不到。降到 64×8=512 路，TPM 降到可申请范围，数据总量不变（多分几批跑）。

---

## 2. 模型 / TPM / 数据量

| 模型（sufy） | 角色 | TPM（经验值，含余量） | 调用频次/分钟 | tokens/次 |
|---|---|---|---|---|
| `openai/gpt-5` | **Actor（沙箱内 hermes）** | **1000 万** | ~6K 次（512路×2轮×6步） | ~4K（3K in + 1K out） |
| `anthropic/claude-4.8-opus` | **Reward/Judge** | **400 万** | ~1K 次（burst，与 actor 错峰） | ~10K（diff+traj+rubric in，4K out） |
| `anthropic/claude-sonnet-5` | Questioner 主 | 100 万 | ~128 次 | ~2.5K |
| `deepseek/deepseek-v4-pro` | Questioner 轮换 | 30 万 | ~32 次（轮换池 1/4） | ~2.5K |
| `qwen/qwen3.7-max` | Questioner 轮换 | 30 万 | ~32 次 | ~2.5K |
| `moonshotai/kimi-k2.6` | Questioner 轮换 | 30 万 | ~32 次 | ~2.5K |
| `openai/gpt-5-mini` | Observer（默认不调） | 20 万 | 0（`use_llm=False`） | ~2.5K |

> Actor + Reward 错峰，峰值预算 ≈ **1000 万 TPM**（不是相加 1400 万）。

---

## 3. 数据量

### 单批
- 64 query × 8 路 = **512 条轨迹/批**
- 每批 actor 调用 = 512 × 6步 = ~3K 次
- 每批 reward 调用 = 512 次

### 冷启动（预热 replay buffer）
- buffer 总容量 25K 轨迹，冷启动预热填 ~30-50% = **8K-12K 轨迹**
- 需 **16-24 批 ≈ 1000-1500 query**
- actor 总调用 ≈ 50K-70K 次；reward 总调用 ≈ 8K-12K 次

### 正式训练（持续）
- 21 个实验 × 持续 rollout，长期刷轨迹进 buffer
- 每步 rollout 一批，TPM 消耗稳定在峰值

---

## 4. 给 sufy 的申请清单（可直接转发）

> 需在 `https://openai.sufy.com/v1` 开通以下模型 TPM 配额：
>
> | 模型 | TPM | 用途 |
> |---|---|---|
> | `openai/gpt-5` | 1000 万 | 沙箱内 actor（主负载） |
> | `anthropic/claude-4.8-opus` | 400 万 | reward judge（与 actor 错峰） |
> | `anthropic/claude-sonnet-5` | 100 万 | questioner 主 |
> | `deepseek/deepseek-v4-pro` | 30 万 | questioner 轮换 |
> | `qwen/qwen3.7-max` | 30 万 | questioner 轮换 |
> | `moonshotai/kimi-k2.6` | 30 万 | questioner 轮换 |
> | `openai/gpt-5-mini` | 20 万 | observer（默认不调，余量） |
>
> 峰值并发预算 ≈ 1000 万 TPM（actor + reward 错峰）。
> 预计冷启动采集 ~1500 query（~60K actor 调用 + ~10K judge 调用），之后正式训练持续。

---

## 5. 备注

- **Actor 模型待定**：`openai/gpt-5`（远程对照）vs `qwen/qwen3.6-27b`（与训练基座同源）。若选 qwen3.6-27b，TPM 不变（同量级），但与正式训练 actor 一致更合理。
- **TPM 给不到时的降级**：按 sufy 实际给的 TPM 反推可行并发路数 ≈ TPM / (4K × 12次/路/分钟)。如给 200 万 → ~40 路并发 → 5 query 并发，数据总量不变、多分批。
- **key**：`runtime.env` 的 `AGENT_MODEL_KEY`（sk-...）用于沙箱内 actor；开发机侧 observer/questioner/reward 用同一 key（开发机→sufy 连通性待确认）。
