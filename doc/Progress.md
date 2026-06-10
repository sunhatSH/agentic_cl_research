# 项目进度

> 本文件是**交付状态**的单一来源。协作流程与分工见 [`ContinualLearning.md`](ContinualLearning.md)；技术设计见 [`CL_Update_Sunhao.md`](CL_Update_Sunhao.md)。

**最后更新：** 2026-06-10  
**当前分支：** `dev_train`  
**verl pin：** `0.8.0`（见 `pyproject.toml`）

---

## 里程碑总览

| ID | 里程碑 | 状态 | 说明 |
|----|--------|------|------|
| M0 | 设计文档 | Done | Loss、7 桶、GPU 调度、实验路线、verl 集成路径 |
| M1 | Replay Buffer v1 | Done | 内存 backend + SQLite 快照持久化，与 verl 解耦，单测 100+ |
| M2 | verl 训练集成 | **Done (code)** | `CLTaskRunner` / `inject_cl_loss` / buffer hooks + fully-async runner scaffold；replay 路径 CUDA smoke 通过，全栈 smoke pending |
| M3 | B1 基线训练 | Blocked | 依赖 GPU 集群（多卡）+ 真实 27B 权重 + 数据；`base.yaml` 已对齐 verl key path |
| M4 | Phase 2–5 消融 | Blocked | 依赖 M3 |
| M5 | ClawEval 评测闭环 | Blocked | 依赖 checkpoint + @杨益博 manifest |

---

## 模块完成度

| 模块 | 进度 | 状态 |
|------|------|------|
| `replay_buffer/` | 100% | Done — 含 reservoir/uniform 开关、单桶塌缩、持久 sampler、SQLite 快照 |
| `trainer/cl_loss.py` | 100% | Done — `make_cl_loss` 按 `is_replay` 分流（replay 行不污染 PPO 分母） |
| `trainer/cl_main.py` | 100% | Done — 入口 `run_cl_ppo`；`build_buffer` 支持 reward/uniform/anti_forgetting |
| `trainer/verl_runner.py` | 100% | Done — `CLTaskRunner`, hooks, `build_trainer`；replay 行双 mask + 跨长度 pad |
| `trainer/verl_async_runner.py` | 100% | Done — Fully Async Policy 对接 scaffold（导入级 + 纯逻辑单测） |
| `trainer/trajectory_adapter.py` | 100% | Done — 回填 `original_logprobs`/`success_rate`，未标注 bucket 跳过 |
| `trainer/replay_batch.py` | 100% | Done — 含 replay warmup 爬升 |
| `trainer/replay_metrics.py` | 100% | Done — buffer 动态日志 + forgetting_risk 回填 + 周期快照（论文证据钩子） |
| `eval/metrics.py` | 100% | Done |
| `eval/run_eval.py` | 40% | 框架有，rollout / manifest 未接（manifest 接口待 P2.4 定形） |
| `configs/` (20 yaml) | 100% | Done — 已重构为 verl Hydra key path，删除死配置 `cl_grpo` |
| `tests/` | ~95% | 144 passed / 1 skipped（新增 replay_metrics 论文证据钩子 7 项） |

---

## 20 实验状态

| Phase | 实验 | 配置 | 训练 | 评测 |
|-------|------|------|------|------|
| 1 | B1 | `configs/phase1/b1.yaml` | 未跑 | 未跑 |
| 2 | K1–K5, K2-R | `configs/phase2/` | 未跑 | 未跑 |
| 3 | R0, R3–R6, R4-w, R4-K | `configs/phase3/` | 未跑 | 未跑 |
| 4 | C1–C4 | `configs/phase4/` | 未跑 | 未跑 |
| 5 | S1, S2 | `configs/phase5/` | 未跑 | 未跑 |
| 6 | X* | 按需 | — | — |

**Checkpoint 产出：** 0

---

## verl 集成验证清单

| 项 | 状态 | 备注 |
|----|------|------|
| verl 版本 pin | Done | `verl==0.8.0`（Python ≥3.10；本机用 uv 建 `.venv`） |
| verl 实际安装 + 导入 | Done | `.venv` (CPython 3.10.20)；`ppo_loss` / `set_loss_fn` / `DataProto.concat` 全部校验存在 |
| 全部引用符号校验 | Done | `main_ppo` / `ray_trainer` / `tensordict_utils` 等 9 模块 ALL-OK |
| `make_cl_loss` no-replay 分支 | Done | `tests/test_cl_loss.py` |
| `compute_replay_loss` 梯度正确 | Done | 改为从 `model_output["log_probs"]` 取 replay 行（带梯度），非 detached 预计算 |
| replay 行拼接 (`_append_replay_rows`) | Done | `DataProto.concat`；双 mask（PPO `response_mask`=0 / `replay_response_mask` 真实 span）+ 跨长度 pad |
| replay 行不污染 PPO loss | Done | replay 行 `response_mask`=0 → `ppo_loss` 自然忽略；L_replay 独占 `replay_response_mask`（B8）|
| buffer 不进 loss 闭包 | Done | 闭包 freevars 仅 `lambda_replay`，cloudpickle 1.2KB |
| `CLTaskRunner` + `set_loss_fn` | Done | `trainer/verl_runner.py` |
| Fully Async Policy 对接 | Done (scaffold) | `trainer/verl_async_runner.py`：CL mixin 子类化 `FullyAsyncTrainer` 底层类后重新 `@ray.remote`；hook `_fit_update_actor` |
| `install_buffer_hooks` | Done | patch `_update_actor`，无全局 monkey-patch |
| replay 路径 CUDA smoke | Done | `pytest -m gpu`：`build_replay_rows`→`select_replay_rows` 在真实 H800 上反向，grad>0 |
| buffer 动态日志 + 周期快照 | Done | `flatten_buffer_stats` 注入 metrics + JSONL；`save_freq` 触发 `buffer.dump`（纯逻辑单测覆盖） |
| forgetting_risk 当前 logprob 前向 | Done (code) / GPU pending | `compute_replay_current_logprobs` 走 verl `compute_log_prob`；off-GPU/假 trainer 优雅降级为 no-op，集群验证 DataProto schema |
| 全栈 1–2 step smoke | **Blocked** | 需多卡集群 + 真实 27B 权重 + 数据集；`scripts/train.sh configs/phase1/b1.yaml ...` 手动跑 |

---

## 外部依赖阻塞

| 依赖 | 负责人 | 阻塞项 |
|------|--------|--------|
| 完整 verl Hydra 配置 | @孙豪 | `base.yaml` 已对齐 key path；集群上仍需与 verl `ppo_trainer` defaults 组合补全 fsdp/optim/rollout engine 等字段 |
| ClawEval 195 任务 manifest | @杨益博 | `eval/run_eval.py` |
| 沙盒 rollout 环境 | @郑乃榕 | 真实 trajectory |
| 用户 query / workspace | @吴健 | 训练数据源 |
| GPU 集群 | — | B1 实测 |

---

## 变更日志

| 日期 | 事件 |
|------|------|
| 2026-06-08 | 仓库骨架、设计文档迁入 `doc/` |
| 2026-06-09 | `e29fd06` — Replay Buffer 核心、20 实验配置、52 单测 |
| 2026-06-09 | M2 verl 集成代码：`CLTaskRunner`, buffer hooks, `doc/Progress.md` |
| 2026-06-09 | verl 0.8.0 装入 `.venv` (py3.10)；修复 replay 梯度(P0)/buffer 序列化(P1)/全局 patch(P2)/权重对齐(P4)；64 passed |
| 2026-06-10 | 系统整理与修复一轮（P0 本地修复 + P1 基线打通 + P2 工程补全）：<br>• weighting 残留状态(A1)、R0 单桶+reservoir/uniform(A2/B5)、持久 sampler(A4)、淘汰 off-by-one(A5)<br>• W0=W2(γ=δ=1) 统一(B1)、R3=`priority_type uniform`、rarity 归一(B3)、S1/S2 固定 `n=2`(B4)<br>• priority 信号回填 `original_logprobs`/`success_rate`(D2)、diversity v1 显式禁用、未标注 bucket 跳过(B12)、replay warmup(B11)<br>• `base.yaml` 重构为 verl key path + 删 `cl_grpo`(B6/B7)、20 yaml 全量校验测试<br>• cl_loss 按 `is_replay` 分流 + replay 行双 mask/跨长度 pad/占位(B8/B9/B10)<br>• Buffer SQLite 快照持久化、Fully Async runner scaffold<br>• 文档修订：U 公式 `δ^(K_i−1−block)`(B2)、GPU 40+24(C1)、成本待重算(C2)、ClawEval General 证据缺口(C4)<br>• 130 passed / 1 skipped；replay 路径在 H800 上通过 CUDA smoke |
| 2026-06-10 | Reward 方案 + judge 规模标注【暂定】：任务判断难度未知，决策门=有标注数据→跑 `calibrate_judge` 看每桶一致率→再定；代码已为暂定设计（judge env 解析、reward 单文件可换），不阻塞 |
| 2026-06-10 | Judge 校准工具链：`eval/judge_agreement.py`（MAE/pearson/kappa/F1 + 按桶 + `rank_judges` 选最小达标）+ `scripts/calibrate_judge.py`（跑候选 endpoint→judge↔人工一致率→排名定 judge）+ `tests/test_judge_agreement.py`（7）。标注数据依赖 ClawEval 人工 rubric（@杨益博 阻塞），工具就绪数据到即跑。206 passed / 1 skipped |
| 2026-06-10 | Reward 反转为模型 judge：放弃规则 reward（尺度不齐+覆盖不到语义桶）→ `trainer/model_reward.py`（抽象 `JudgeClient`，模型不写死，env 解析 `JUDGE_API_BASE/MODEL`）+ `scripts/serve_reward_model.sh`（vLLM 本地冻结 judge）+ `tests/test_model_reward.py`（6）；删 `rule_reward.py`/其测；`base.yaml reward` 改指 model_reward。本地冻结、judge≥策略、默认 32B、ClawEval 一致率校准。199 passed / 1 skipped |
| 2026-06-10 | 沙箱镜像层 + 训练链路 Gap A–D 落地：<br>• 镜像：`Dockerfile`(Node24+OpenClaw+办公工具+persona 种子)、`bin/seed_workspace.sh`(确定性物化)、`agent_entry.sh`、`fs-seeds/`(3 persona)、`openclaw.config.template.json`<br>• 架构文档 `Sandbox_Agent架构.md`(动作内/推理外)、调度文档、`Plan_训练链路补齐.md`<br>• **Gap A** 规则 reward `trainer/rule_reward.py`(safety×(0.8 completion+0.2 robustness))+`base.yaml reward.*`<br>• **Gap B** `scripts/convert_dataset.py` jsonl→parquet(bucket hint/checker 提取/98-2 切分)<br>• **Gap C** `rollout/session_pool.py`+`scheduler.py`(16×8 + winner-sync 状态机, mock 后端)<br>• **Gap D** `rollout/collect.py`(verl 原生字段收集 dry-run → buffer)<br>• 轨迹收集机制定调：用 verl `ToolAgentLoop` 原生字段，不写 proxy<br>• +26 单测，201 passed / 1 skipped |
| 2026-06-10 | 论文证据钩子（`trainer/replay_metrics.py`）：<br>• buffer 动态日志：每步 `flatten_buffer_stats` 注入 verl/wandb metrics + sidecar `logs/buffer_stats/<exp>.jsonl`（各桶 size/fill_ratio/淘汰/reservoir 拒绝/信号权重）<br>• 激活 `forgetting_risk`：replay 行经 `compute_log_prob` 取当前策略 logprob → `per_row_masked_mean` → `backfill_forgetting` 写回 `current_logprobs` 重算优先级（R4-vs-R5 核心证据）<br>• 按 `trainer.save_freq` 自动 `buffer.dump` → `buffer_dumps/<exp>-step-<N>.sqlite`<br>• `build_replay_rows` 新增 `_replay_tids` 对齐 sidecar；`base.yaml` 增 `buffer_stats_log_freq`/`forgetting_update_freq`<br>• +7 单测，144 passed / 1 skipped |

---

## 下一步行动

| 优先级 | 行动 | 负责人 |
|--------|------|--------|
| **P0** | **训练链路补齐（规则 reward / 数据转换 / 沙箱调度 / rollout 桥）——施工图见 [`Plan_训练链路补齐.md`](Plan_训练链路补齐.md)** | @孙豪 |
| P0 | 全栈 GPU smoke：集群上 B1/R4 跑 1–2 step（多卡 + 真实 27B + 数据） | @孙豪 |
| P0 | 集群上 merge verl 完整 Hydra defaults（fsdp/optim/rollout engine） | @孙豪 |
| P1 | ClawEval manifest 接入（接口已在 `eval/run_eval.py` 定形 + fake 单测） | @杨益博 |
| P1 | 沙盒 rollout 提供带 `bucket`/`messages`/`success_rate` 的真实轨迹 | @郑乃榕 |
| P2 | 27B/64 卡 成本与显存重算（C2）；fully-async 全栈联调 | @孙豪 |
