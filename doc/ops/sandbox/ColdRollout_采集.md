# 冷启动多轮 Rollout 采集（observer + questioner，无奖励）

> **创建**：2026-06-13　**状态**：链路打通（远程 actor 路 small-batch 验证通过），本地 27B 路待全量
> **定位**：用真实回流种子 query，跑多轮 user-sim rollout，采集初始数据。本阶段**不要奖励模型**，但启用**观察 agent + 出题人设 agent**。

## 1. 这一阶段做什么（与 UserSim 设计的差异）

本阶段是 [`UserSim_多轮Query在线生成.md`](../../archive/UserSim_多轮Query在线生成.md) 的**简化采集变体**（2026-06-13 孙豪定）：

| 维度 | 本阶段 | 完整 UserSim 设计 |
|------|--------|-------------------|
| GRPO / 8 槽 / winner | **无**。1 query = 1 rollout，N 沙箱跑 N 条不同 query | 8 槽 + winner-sync |
| 奖励 / judge | **无** | observation-grounded judge |
| 观察 agent (observer) | **有** | 有 |
| 出题人设 agent (questioner) | **有** | 有 |
| 多轮 | 有（observer → questioner 驱动，K~U{1..k_max}） | 有 |

因为无 winner，不能用 `rollout/simulated_session.run_simulated_session`（它依赖 run_query 多槽 + pick_winner + sync）。本阶段新写了 `rollout/usersim_collect.run_usersim_session`（slots=1，单轨迹多轮）。

## 2. 模型选型（本阶段：三方 + actor 双路，均不同）

> **权威选型见 [`模型选型.md`](../../archive/模型选型.md)**（单一信源）。下表为冷启动采集阶段的具体落点（数据去向），选型本身以选型文档为准。

| 角色 | 模型 | 后端 | 数据去向 |
|------|------|------|----------|
| **本地 actor**（正式训练） | Qwen3.6-27B | 本地 verl rollout（lightllm） | `data/rollouts/local/` |
| **沙箱内 actor**（冷启动/hermes） | **openai/gpt-5** | sufy（沙箱内 hermes 调） | `data/rollouts/remote/` |
| **observer** | **openai/gpt-5-mini** | sufy | （三 agent 不存数据，只驱动） |
| **questioner** | **anthropic/claude-sonnet-5** + 轮换池 | sufy | |

三个角色（沙箱内 actor / observer / questioner）刻意用**三个不同模型**，抗 self-preference。两套 actor 数据**分目录存**。模型选型单一信源见 [`模型选型.md`](../../archive/模型选型.md)。

## 3. 远程 API 来源（关键）

sufy（OpenAI 兼容，沙箱可达）：
- base：`https://openai.sufy.com/v1`
- key：`SUFY_API_KEY`（开发机侧 `.env`）/ `AGENT_MODEL_KEY`（沙箱内 `runtime.env`，同一 sufy key）。
- 已验证：沙箱 TCP 443 通、`/v1/models` 返回 117 模型、`openai/gpt-5` chat 验证通过。
- 旧 tokenhub（商汤内网 `172.30.9.145`）沙箱连不上，已弃用。

> **硬约束（孙豪 2026-06-13）**：observer/questioner 必须用远程模型；**若远程 API 不可用则中止并报告**，不得降级跳过继续跑（无 observer = 无报告 = 多轮进行不下去）。

## 4. 运行环境（踩坑后的最终方案）

**Qwen3.6-27B 的架构是 `qwen3_5`（attention+mamba 混合）**，识别/加载它对 vllm 版本有硬要求，几经错配，最终结论：

- **唯一能加载的 vllm = `/mnt/afs_code/ds32_env`（vllm 0.16.0rc，自带 `qwen3_5.py` 建模 + torch2.10）**。
  日志确认：`Resolved architecture: Qwen3_5ForConditionalGeneration`、`get_config OK Qwen3_5Config`。本地 actor 起 vllm 用 `PY=/mnt/afs_code/ds32_env/bin/python`，**在 tmux 里跑**。
- 走不通的路（避坑记录）：
  - `/opt/conda`：vllm0.11 + torch2.9.1 **ABI 不匹配**（`vllm._C undefined symbol`）；且 vllm0.11/0.13 **都没有 `qwen3_5.py`**，根本不支持该架构。
  - vllm0.13 overlay（`envs/vllm013`，已弃用）：要求 `transformers<5`，而 `qwen3_5` 需 transformers≥5.2 → 死结（`ALLOWED_LAYER_TYPES` import 失败）。
  - **结论**：别在旧 vllm 上凑 transformers；用支持 qwen3_5 的新 vllm（ds32_env）。
- 27B 权重在 quarkfs 上加载慢（~15–20 min，一次性，受共享盘 IO 限速）。
- 共享盘 = `/mnt/afs_toolcall/`，8 机共用同一份代码/模型；`ds32_env` 在本地镜像（每台都有）。

## 5. 代码（本次新增）

- `rollout/usersim_collect.py` — `run_usersim_session`：slots=1 单轨迹多轮，observer→questioner，无 winner/reward。
- `scripts/collect_rollout.py` — 入口：抽种子→N 路并发跑 session→jsonl 落盘（每条 session = 轨迹+生成query+observer报告）。`--actor local|remote`。observer/questioner 从 env 解析，缺失 `exit 5`。
- `scripts/collect_rollout.sh` — 启动器：source E2B 凭证→预检 3 远程模型→配 env→（local 路）起 vllm→跑两路到 `data/rollouts/{local,remote}/`。
- `data/cleaning.py` — 文本清洗核心模块：零宽字符剥离 + 乱码检测 + 阈值过滤。`collect_rollout.py` 默认在采集时对 seed query 和轨迹消息做在线清洗（`--no-clean` 可跳过）。
- `scripts/clean_queries.py` — 批量清洗 queries JSONL 文件（采集前预处理）。
- `scripts/clean_buffer.py` — 批量清洗冷启动 buffer SQLite 快照（采集后后处理）。
- `scripts/taskspec_to_queries.py` — **Step 3**：taskspec.yaml → queries.jsonl（`{record_id, queries:[seed_query, *follow_ups]}`）。本脚本的输出是 `collect_cold.py`/`collect_rollout.py` 的 `--queries` 输入。完整 7 步 pipeline 见 [`doc/source/训练与推理流程.md` §6.5](../../source/训练与推理流程.md#65-数据-pipelinetaskspec--训练数据7-步)。
- `scripts/trajectory_to_parquet.py` — **Step 6**：冷启动 trajectory JSONL → verl rl_dataset parquet（`prompt`/`data_source`/`reward_model`/`extra_info` 4 列）。产出填进 `configs/cluster.yaml: data.train_files`。

## 6. 运行命令

```bash
cd /mnt/afs_toolcall/sunhao4/agentic_cl_research
# 两路都跑（先起本地 vllm，再跑 local+remote）
bash scripts/collect_rollout.sh
# 只跑远程（不占 GPU）：
ACTORS=remote bash scripts/collect_rollout.sh
# 小批验证：LIMIT=3 CONCURRENCY=2 ACTORS=remote bash scripts/collect_rollout.sh

# 数据清洗（采集前/后各一步）
python scripts/clean_queries.py --input datasets/queries.jsonl --output datasets/queries_clean.jsonl
python scripts/clean_buffer.py --input logs/cold/buffer.sqlite --output logs/cold/buffer_clean.sqlite
```
日志：`logs/cold/{vllm,rollout_local,rollout_remote}.log`。产物：`data/mock/rollouts/{local,remote}/rollouts_*.jsonl`。

> **Mock 隔离约定（2026-06-13）**：上游正式数据未到位前，当前采集/训练仅用于**验证可行性**，所有**产物**（数据 + 模型）落 `data/mock/` 下，镜像正式结构、与正式产物隔离。**base 模型不隔离**（`/mnt/afs_toolcall/sunhao4/models/Qwen3.6-27B` 是共用输入）。
>
> | 产物 | mock 路径 | 正式路径 | 隔离方式 |
> |------|-----------|----------|----------|
> | rollout 采集数据 | `data/mock/rollouts/{actor}/` | `data/rollouts/` | 脚本 `OUT_BASE` 默认指 mock |
> | replay buffer 快照 | `data/mock/buffer_dumps/` | `buffer_dumps/` | `collect_cold.sh` 的 `OUT` 默认指 mock |
> | 训练 checkpoint | `data/mock/ckpts/` | `ckpts/` | 跑训练时 `trainer.default_local_dir=data/mock/ckpts` 命令行覆盖（**不改 base.yaml**） |
> | base 模型 | （不隔离） | `models/Qwen3.6-27B` | 输入，共用 |
>
> 正式数据到位后：采集加 `OUT_BASE=<repo>/data/rollouts`，训练不传 `default_local_dir` 覆盖即落正式 `ckpts/`。base.yaml 保持干净未改。

## 7. 中间结果（截至 2026-07-01）

- ✅ sufy 连通（沙箱 TCP 443 + 117 模型 + gpt-5 chat 验证）；旧 tokenhub 已弃用。
- ✅ 运行环境凑齐（lightllm verl base + qwen3_5）。
- ✅ **远程 actor 路 small-batch 验证通过**：`--actor remote --backend local --limit 2` → `done=2 failed=0`；产出含多轮、persona（如 "Dr. Lena the researcher"）、questioner 生成的下一轮 query、observer 5 字段报告。链路确认工作。
- ⏳ 本地 27B actor 路 + e2b 真沙箱全量：待起 vllm 后跑（占 8 卡）。
- ✅ 8 机并行：`collect_rollout.py` 已支持 `--node-rank/--num-nodes` 分片参数，各机按 rank 取不同种子子集。

## 8. 待办

| # | 事项 |
|---|------|
| 1 | 起本地 vllm 跑 local actor 路（e2b 真沙箱，全量） |
| 2 | session 失败率监控（remote actor 限流时） |
