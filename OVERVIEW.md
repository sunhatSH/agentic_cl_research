# OVERVIEW — 代码地图

> 一页纸速览：目录职责、数据流、"改 X 看哪个文件"、实验状态、已知问题。
> 协作规范见 `CLAUDE.md`；设计信源见 `doc/source/CL_Design.md`；文档索引见 `doc/README.md`。
> 项目：Continual Learning over Agentic LLM（Qwen3.5-9B，verl 0.8.0，LightLLM，16 卡）。

---

## 目录职责（源码）

| 目录 | 职责 |
|------|------|
| `trainer/` | 训练入口 + verl 无侵入注入。`cl_main.py`(入口) `verl_runner.py`(CLTaskRunner) `cl_loss.py`(CL Loss) `cl_replay_hook_v1.py`(9桶回放 hook + std 指标) `trajectory_adapter_v1.py`(KVBatchMeta→轨迹) `replay_forward.py`(回放行构建) `model_reward*.py`(judge 打分) |
| `replay_buffer/` | 9 桶 Buffer（纯 Python，与 verl 解耦）。`bucket.py` `store.py` `sampler.py` `eviction.py` `priority.py` `weighting.py` |
| `rollout/` `agents/` `inference/` | 采样侧：沙箱客户端 / 三 Agent（observer/questioner/reward）/ 单步生成 |
| `eval/` | ClawEval 评测 |
| `configs/` | 实验配置（三层继承 base→cluster→run/）。见下方 |
| `scripts/` | 训练/数据/画图/沙箱脚本。见下方 |
| `tests/` | ~400 单测（CPU；verl/CUDA smoke 需集群） |
| `docker/sandbox/` | 沙箱镜像 + fs-seeds 种子文件系统 |
| `data_pipeline/` `bin/` | 数据管道工具（独立风格，ruff 排除） |

## 目录职责（产物 / 文档，多为 gitignored）

| 目录 | 职责 |
|------|------|
| `datasets/` | `train.parquet`/`train_aligned.parquet`（训练用，5桶×3200 中等难度）+ `cold_start/`（warmup）+ `_archive/`（旧数据） |
| `logs/metrics/<exp>/` | verl FileLogger 每 step metrics.jsonl（+ metrics.all.jsonl 跨 run 累积） |
| `logs/experiments/figs/` | 画图产物（每实验一文件夹 + config.json，旧图折叠 `{exp}_N/`） |
| `rollouts/training/<exp>/` | 每 step winner + rollout 全量轨迹（每次启动清理） |
| `ckpts/` `buffer_dumps/` | checkpoint / buffer 快照 |
| `doc/` | 设计信源(source/) 运行手册(ops/) 评测(eval/) 调试(debug/) 实验结论(expr/) 归档(archive/) |
| `paper/` `master-thesis/` | 论文 / 学位论文 |

---

## 数据流

```
configs/run/*.yaml
   │  (OmegaConf 继承 base→cluster→run；命令行 --lr 等最高优先级)
   ▼
trainer/cl_main.py  ──build_buffer()──► replay_buffer/BucketReplayBuffer (R 系列; B/K 为 None)
   │
   ▼
trainer/verl_runner.py (CLTaskRunnerV1)
   │  set_loss_fn(cl_loss)  +  install_buffer_hooks_v1(std 指标 + 9桶回放)
   ▼
verl 原生 main_ppo (custom_sync) ──rollout──► RemoteAgentLoopManager
   │                                              │
   │  _update_actor(KVBatchMeta, metrics)          ▼
   │    PRE: 采 buffer winner 掺回放行             rollout/ + agents/ + docker/sandbox
   │    POST: 抽 winner 入 buffer + std 指标 + rollout 记录
   ▼
logs/metrics/<exp>/metrics.jsonl ──► scripts/plot/plot_progress.py ──► logs/experiments/figs/
```

---

## 改 X 功能看哪个文件

| 要改 | 文件 |
|------|------|
| CL Loss（L_rl/L_kl/L_replay/L_ent 组合） | `trainer/cl_loss.py` |
| 回放采样量 / 比例（replay_ratio） | `trainer/cl_replay_hook_v1.py`（PRE 段） |
| 回放行构建（token ids → 训练行） | `trainer/replay_forward.py` + `trainer/replay_batch.py` |
| v1 轨迹/字段提取（bucket/task_id/reward） | `trainer/trajectory_adapter_v1.py` |
| std 指标（reward_std/group_reward_std） | `trainer/cl_replay_hook_v1.py::_merge_std_metrics` |
| 9 桶采样 / 淘汰 / 优先级 / 权重 | `replay_buffer/{sampler,eviction,priority,weighting}.py` |
| reward judge（打分维度/模型） | `trainer/model_reward.py` + `agents/prompts.py` REWARD_RUBRIC |
| 桶坐标 / 训练序 | `configs/bucket_coords.json`(sampler 默认) `bucket_coords_final.json`(建数据) |
| 难度打分 / 建训练集 | `scripts/pipeline/score_difficulty.py` + `scripts/pipeline/build_train.py` |
| 画训练曲线 | `scripts/plot/plot_progress.py` |
| 训练启动 / 拓扑 / lr 传参 | `scripts/train.sh`（--lr 等）+ `scripts/_train_impl.sh` |

---

## configs/ 结构

| 项 | 说明 |
|------|------|
| `base.yaml` `cluster.yaml` `_generated_ppo_trainer.yaml` | 三层继承基底（lr/batch/n 等默认已对齐 B1 实验值：lr=2e-6、batch=32、n=8） |
| `run/*.yaml`（21 个，全 16 卡） | 集群可跑：B1(baseline) / K1-K3,K2-R(KL 消融) / R0-R9(buffer 消融) / b1,r4(旧版)。4gpu/8gpu 变体已删（train.sh 只留 16/32/64 卡预设） |
| `bucket_coords.json` `bucket_coords_final.json` | 前者 sampler 默认，后者建数据用（**其余变体已归 `_bucket_coords_variants/`**） |
| `_legacy_phases/` | 旧 phaseN 配置（已被 run/ 取代，仅回溯） |
| `templates/` | 实验/硬件模板 |

## scripts/ 结构

| 子目录 | 内容 |
|------|------|
| 顶层 | `train.sh` `_train_impl.sh`（训练入口）+ `warmup_buffer.py` `add_reward_fn.py`（pipeline 入口，被 .sh 按名调用） |
| `pipeline/` | 冷启动 / 数据管道 / 难度打分 / 建训练集 |
| `collect/` | 沙箱采集 |
| `data/` `analysis/` | 数据 prep / QC / 桶发现 |
| `plot/` | 画图（plot_progress 主力 + metrics 家族） |
| `serve/` | reward serve / agents harness / 评测 |
| `env/` `sandbox/` | 训练/沙箱凭证 + 镜像 build |
| `_legacy/` | 归档的旧 pipeline/analysis 脚本（历史版本，不再维护） |

---

## 实验状态（2026-08-13）

| 实验 | 类型 | 状态 |
|------|------|------|
| B1 | 纯 PPO baseline | 跑中，coding 桶，reward~0.4，std 指标已落盘 |
| K2 | PPO + KL(0.05) | 跑中，entropy 高于 B1（KL 生效） |
| R0 | CLEAR 单桶 replay | 跑中，buffer 每步 +32、group_std 已出；**回放刚修好待验证**（step2 起应 replay_empty=0） |
| 路线图 | K1-K5 / R3-R9 / C1-C4 / S1-S2 | 见 `doc/source/CL_Design.md` 实验路线 |

结论存 `doc/expr/<exp>/<日期>.md`（图引 `../assets/`）。

## 已知问题 / 待办

- **v1 字段映射**（RunLog §63）：v1 的 bucket/task_id/messages 在 tq 的 field/extra_info，不在 tag。已修 `trajectory_adapter_v1.py`（extra_info 取 bucket/task_id）+ `replay_forward.py`（v1 用 pre-tokenized ids 建回放行，不重 tokenize messages）。**需重启验证 R0 回放非空**。
- **rollout 成功率记录**：`_persist_rollout_status` 目前只在 buffer 启用（R 系列）时触发；B1/K2（无 buffer）不记录——如需全实验记录待独立。
- **回放比例消融**：replay_ratio 当前默认 5（回放占比 16.7%）；CLEAR 原文 50%、设计文档 33%——三档消融待跑。
