# 项目进度

> 本文件是**交付状态**的单一来源。技术设计见 [`CL_Design.md`](../source/CL_Design.md)。

**最后更新：** 2026-06-19  
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
| M0.5 | Phase 0 冷启动数据来源消融 | **Done (code+doc)** | 5 臂配比（P0-A..E）+ 来源标记 + 配比混合 + 双轨验收 gate；采集/短RL 待集群，见 [`CL_Design.md`](../source/CL_Design.md) |

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
| `tests/` | ~95% | 200 测试函数 / 30 文件（最近一次实跑约 200 passed / 1 GPU skip） |

---

## 21 实验状态

| Phase | 实验 | 配置 | 训练 | 评测 |
|-------|------|------|------|------|
| 1 | B1 | `configs/phase1/b1.yaml` | 未跑 | 未跑 |
| 2 | K1–K5, K2-R | `configs/phase2/` | 未跑 | 未跑 |
| 3 | R0-10k, R0-25k, R3–R6, R4-w, R4-K | `configs/phase3/` | 未跑 | 未跑 |
| 4 | C1–C4 | `configs/phase4/` | 未跑 | 未跑 |
| 5 | S1, S2 | `configs/phase5/` | 未跑 | 未跑 |
| 6 | X* | 按需 | — | — |

**Checkpoint 产出：** 0

### 分阶段实验（2026-08-20 更新，详见 `doc/eval/防遗忘评测方案.md` §11）

21 训练（算法 ablation）之外，防遗忘验证改为**分阶段、可裁剪**方案（脚本已就绪，未跑）：

| 阶段 | 脚本 | 数据 | 训练 | 评测 |
|------|------|------|------|------|
| 第一步主实验 | `scripts/exp1_two_bucket/run.sh` | `datasets/train_cl.parquet`（coding/research 各 6400） | coding→research 续训 200+200，3 方法（baseline/CLEAR/CL） | 评 base + 2 checkpoint × 9 桶 |
| 第二步（可选） | `scripts/exp2_nine_bucket/run.sh` | `datasets/train_exp2.parquet`（7 桶，192 step） | 7 桶少量边训边评，3 方法 | 每桶训完评 9 桶 |
| 评测 runner | `scripts/exp_common/{merge_ckpt,run_eval_cl,run_eval_base}.sh` + `trainer/cl_eval.py` | — | — | 复用训练 RemoteAgentLoopManager 跑 ClawEval + judge 打分 |

**Checkpoint 产出：** 0（第一步训出 `global_step_200` + `global_step_400`）

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
| 用户 query / workspace / 镜像 | @吴健 | 训练数据源（每会话首条 query + 对应沙箱镜像） |
| 多轮种子筛选 + 单/多轮配比（UserSim O1） | @吴健 | `datasets/queries.jsonl` 种子质量 |
| GPU 集群 | — | B1 实测 |

---

## 变更日志

| 日期 | 事件 |
|------|------|
| 2026-07-03 | **Phase 0 冷启动数据来源配比消融**（独立预实验，不进 21；本机，CPU）：新增设计文档 [`CL_Design.md`](../source/CL_Design.md)。<br>• **问题**：冷启动填 7 桶 buffer 时 actor 用 Qwen3.6-27B（on-policy 同源）vs gpt-5（off-policy 更强）vs 按比例混合，哪个对下游 CL 最好——与 21 实验正交的新变量，须正式训练前定死。<br>• **5 臂**：P0-A(100:0) / P0-B(0:100) / P0-C(50:50) / P0-D(70:30) / P0-E(30:70)，唯一变量 = 桶内 27B:gpt5 配比，其余锁死 R4。<br>• **代码**：`collect_rollout.py` 加 `meta.policy` 来源标记；`warmup_buffer.py` 加 `--ratio-27b` 桶内配比混合 + `*.manifest.json`；`configs/phase0/p0-{a..e}.yaml`；`scripts/phase0/{run.sh,gate_coldstart.py}`（三阶段：建 buffer→轨1 gate→短RL）。<br>• **双轨验收**：轨1 冷启动自身指标（桶配额 7/7、tool-call≥90%、judge 有效分≥95%、多样性 distinct_4≥0.6/self_bleu_4≤0.5，无 GPU 即测）先筛 → 轨2 下游短RL（CL Score 主裁，固定新任务+同种子，7 桶拆旧/新）裁决。<br>• **文档整改**：`CL_Design.md` 加 Phase 0 节 + 路线图 + **给 Phase 1–6 各补验收标准表**（统一引用 `eval/metrics.py` 指标、CL Score 显著性门槛 2%）；`Buffer_冷启动数据需求.md` §5 补来源配比核对项。<br>• 采集 + 短RL 待集群（需 sufy 27B/gpt-5 节点 + GPU）。 |
| 2026-06-23 | **thinking 模型截断防护 + B1/R4 方法学一致性**（本机，CPU）：<br>• **截断防护**：模型回复 `finish_reason=length*`（或思考预算吃光 token、content 空且无 tool_calls）时，`agents/base.py::_raise_if_truncated` 统一抛 `TruncatedOutputError`。三 Agent + judge 按各自语义分流——questioner 标 `last_query_was_error`（走 patience/telemetry，不误判为满意 `<end_session>`）、observer 降级到确定性取证报告（不解析半截 JSON）、judge `parse_judge_output` 返回 `(verdict, parsed)` 并在无 verdict JSON 时抛错→`compute_score` 的 `judge_error=1.0`（不再静默全 0 reward）；questioner `max_tokens` 256→512。反 reward-hacking / 反静默退化的工程加固。<br>• **B1/R4 同 rollout**：`configs/run/b1.yaml` 删除 `rollout.agent: null`，B1 改为与 R4 共用同一套 agentic 多轮 rollout（差异仅 CL 项：B1 关 replay/无 KL），同步 `Migration_64GPU.md`——拿"多轮"R4 与"单轮"B1 比遗忘不公平，遗忘基线须同等 rollout 下测。<br>• 测试：`test_agents`+`test_model_reward` 53 passed（+1 截断用例）；全量 287 passed / 7 skipped / 3 failed（3 failed 为既有环境问题：`smoke_1step` 计数 + 缺 verl yaml，与本次无关）。 |
| 2026-06-19 | **模型选型集中化**：新增 [`模型选型.md`](模型选型.md) 作为 actor/observer/questioner/judge 选型的**单一信源**（选型由 @孙豪 拍板：基座 Qwen3.6-27B；远程采集对照 gpt-5；observer gpt-4.1-mini@temp0；questioner claude-sonnet-4-6@temp0.9；judge 本地冻结 32B 默认）。其它文档（`Sandbox_Agent架构.md` §7b、`接口使用_Sandbox与三Agent.md` §2、`ColdRollout_采集.md` §2、`论文向总览` §11、`CLAUDE.md`/`doc/README.md` 索引）改为**引用**该文件，不再各自维护选型；§7b 旧"未拍板/阻塞于标注"措辞改为"选型已定、一致率校准为可选验证"。 |
| 2026-06-19 | 沙箱接口/实现解耦 + 多 Agent harness + observer 取证缺陷定位（本机，无 GPU/E2B）：<br>• **沙箱后端可插拔**：`rollout/sandbox_client.py` 封闭工厂 → 开放注册表（`SandboxClient` Protocol = 接口契约；`register_backend`/`make_sandbox` 按名选后端）；新增 `AliyunSandbox`（阿里云 无影 AgentBay）**留空 stub** —— 换/加沙箱厂商**零改 rollout loop**。配套 `tests/test_sandbox_client.py`(+3)、`pyproject.toml`、`scripts/{sandbox_smoke,collect_rollout,collect_cold}.py`、`configs/base.yaml`。<br>• **observer 实运行取证缺陷**（runtime 证据确认）：主训练路径 `observe()` 没传沙箱、`LocalSandbox` 无持久 FS、只列文件路径不读内容 → observer 退化为"actor 自述复读机"，反 reward-hacking 落空。结论 + 修复计划记入 `CLAUDE.md` TODO#5，**当日即落地 diff-driven**：observer 以沙箱 before/after **内容级 diff** 为 ground truth、actor 声称仅作交叉核对，`LocalSandbox` 改持久 workdir，本机已验证 diff 能读到内容并暴露"声称≠实际"（二进制格式解析 / `watch_dir` / 真实后端连通待集群）。<br>• 新增 [`接口使用_Sandbox与三Agent.md`](../ops/sandbox/接口使用_Sandbox与三Agent.md)（接口怎么用速查 + `ObservationReport`(报告)/`actor_claims`(声明) 消费 + diff-driven 设计）与 `scripts/agents_harness.py`（observer+questioner+reward **离线 harness**，simulated/collect 两模式、mock/real 可切，已本机跑通）。<br>• **协作（其他 Agent，今日）**：文档归档整理（一次性技术报告/汇报/复盘 `doc/` → `paper/refs/` 4 篇 + `scripts/README.md`/`configs/run/README.md`）；`Migration_64GPU.md` +133（集群冷启动/启动指南）、`训练与推理流程.md` 更新；集群启动配置（judge endpoint / B1 与 R4 同走沙箱 / run_phases mock）；健壮性：`trainer/cl_loss.py` `no_padding_2_padding` 失败转 WARNING（防静默错误 replay loss）、`trainer/verl_async_runner.py` Ray actor 解包失败改 fail-fast；`.gitignore` 补 `.hypothesis/`/`*.so`/`node_modules/`；`base.yaml` 20→21 实验。 |
| 2026-06-15 | 多轮 user-sim 采集链路打通（`rollout/usersim_collect.py` + `scripts/collect_rollout.py`）：本地 27B + 远程 GPT 双 actor，observer+questioner 在线多轮，采集结果按 7 桶预热 replay buffer（`scripts/warmup_buffer.py`）并衔接训练。配套 [`ColdRollout_采集.md`](../ops/sandbox/ColdRollout_采集.md) / [`RolloutCollect_技术报告.md`](../../paper/refs/RolloutCollect_技术报告.md)（已归档到 `paper/refs/`）。 |
| 2026-06-13 | 冷启动数据采集（`scripts/collect_cold.py`：本地 vllm 27B → 7 桶 replay buffer）；商汤 SenseCore 8 机×8 卡 H800 上 Qwen3.6-27B 多轮采集集群联调。新增 4 篇文档：[`ColdRollout_采集.md`](../ops/sandbox/ColdRollout_采集.md)（运行手册，本阶段简化变体：1 query=1 rollout、无 GRPO/winner、无奖励，仅 observer+questioner）、[`RolloutCollect_技术报告.md`](../../paper/refs/RolloutCollect_技术报告.md)（已归档到 `paper/refs/`）（系统设计 + rollout 数据结构定义）、[`BugLog_集群采集.md`](BugLog_集群采集.md)（集群采集 bug 库，append-only，首批 10 条）、[`集群推理采集_经验复盘.md`](../../paper/refs/集群推理采集_经验复盘.md)（已归档到 `paper/refs/`）（踩坑复盘，不进论文）。 |
| 2026-06-08 | 仓库骨架、设计文档迁入 `doc/` |
| 2026-06-09 | `e29fd06` — Replay Buffer 核心、20 实验配置、52 单测 |
| 2026-06-09 | M2 verl 集成代码：`CLTaskRunner`, buffer hooks, `doc/archive/Progress.md` |
| 2026-06-09 | verl 0.8.0 装入 `.venv` (py3.10)；修复 replay 梯度(P0)/buffer 序列化(P1)/全局 patch(P2)/权重对齐(P4)；64 passed |
| 2026-06-10 | 系统整理与修复一轮（P0 本地修复 + P1 基线打通 + P2 工程补全）：<br>• weighting 残留状态(A1)、R0 单桶+reservoir/uniform(A2/B5)、持久 sampler(A4)、淘汰 off-by-one(A5)<br>• W0=W2(γ=δ=1) 统一(B1)、R3=`priority_type uniform`、rarity 归一(B3)、S1/S2 固定 `n=2`(B4)<br>• priority 信号回填 `original_logprobs`/`success_rate`(D2)、diversity v1 显式禁用、未标注 bucket 跳过(B12)、replay warmup(B11)<br>• `base.yaml` 重构为 verl key path + 删 `cl_grpo`(B6/B7)、20 yaml 全量校验测试<br>• cl_loss 按 `is_replay` 分流 + replay 行双 mask/跨长度 pad/占位(B8/B9/B10)<br>• Buffer SQLite 快照持久化、Fully Async runner scaffold<br>• 文档修订：U 公式 `δ^(K_i−1−block)`(B2)、GPU 40+24(C1)、成本待重算(C2)、ClawEval General 证据缺口(C4)<br>• 130 passed / 1 skipped；replay 路径在 H800 上通过 CUDA smoke |
| 2026-06-10 | Reward 方案 + judge 规模标注【暂定】：任务判断难度未知，决策门=有标注数据→跑 `calibrate_judge` 看每桶一致率→再定；代码已为暂定设计（judge env 解析、reward 单文件可换），不阻塞 |
| 2026-06-10 | Judge 校准工具链：`eval/judge_agreement.py`（MAE/pearson/kappa/F1 + 按桶 + `rank_judges` 选最小达标）+ `scripts/calibrate_judge.py`（跑候选 endpoint→judge↔人工一致率→排名定 judge）+ `tests/test_judge_agreement.py`（7）。标注数据依赖 ClawEval 人工 rubric（@杨益博 阻塞），工具就绪数据到即跑。206 passed / 1 skipped |
| 2026-06-10 | Reward 反转为模型 judge：放弃规则 reward（尺度不齐+覆盖不到语义桶）→ `trainer/model_reward.py`（抽象 `JudgeClient`，模型不写死，env 解析 `JUDGE_API_BASE/MODEL`）+ `scripts/serve_reward_model.sh`（vLLM 本地冻结 judge）+ `tests/test_model_reward.py`（6）；删 `rule_reward.py`/其测；`base.yaml reward` 改指 model_reward。本地冻结、judge≥策略、默认 32B、ClawEval 一致率校准。199 passed / 1 skipped |
| 2026-06-10 | 沙箱镜像层 + 训练链路 Gap A–D 落地：<br>• 镜像：`Dockerfile`(Node24+OpenClaw+办公工具+persona 种子)、`bin/seed_workspace.sh`(确定性物化)、`agent_entry.sh`、`fs-seeds/`(3 persona)、`openclaw.config.template.json`<br>• 架构文档 `Sandbox_Agent架构.md`(动作内/推理外)、调度文档、`Plan_训练链路补齐.md`<br>• **Gap A** 规则 reward `trainer/rule_reward.py`(safety×(0.8 completion+0.2 robustness))+`base.yaml reward.*`<br>• **Gap B** `scripts/convert_dataset.py` jsonl→parquet(bucket hint/checker 提取/98-2 切分)<br>• **Gap C** `rollout/session_pool.py`+`scheduler.py`(16×8 + winner-sync 状态机, mock 后端)<br>• **Gap D** `rollout/collect.py`(verl 原生字段收集 dry-run → buffer)<br>• 轨迹收集机制定调：用 verl `ToolAgentLoop` 原生字段，不写 proxy<br>• +26 单测，201 passed / 1 skipped |
| 2026-06-12 | 论文主旨 + Introduction 初稿：[`Paper_Intro_draft_CN.md`](../../paper/drafts/Paper_Intro_draft_CN.md) + [`Paper_Intro_draft_EN.md`](../../paper/drafts/Paper_Intro_draft_EN.md)。把整个项目作为**一篇**论文收敛成单一主旨——"持续多轮 agentic RL 防遗忘 = 先在源头产出干净信号、再巩固"——5 个部件全部降格为"实现手段"，贡献列表组织为：①源头级重新表述(两类信号污染) ②干净信号数据/rollout 系统(winner-sync + 三 agent) ③匹配的巩固方法(CL Loss/Buffer/U 形) ④大规模验证；明确叙事纪律(子组件不挣到位置就报负面结果) |
| 2026-06-12 | 论文 Method 章节散文初稿：[`Paper_Method_draft_CN.md`](../../paper/drafts/Paper_Method_draft_CN.md)（中文）+ [`Paper_Method_draft_EN.md`](../../paper/drafts/Paper_Method_draft_EN.md)（英文投稿用），按"干净信号"主线把 §4.1–4.3（CL Loss/Buffer/U 形权重）与 §4.4–4.5（winner-sync/三 agent 多轮）扩写为可直接入稿的正文，配 5 张图占位；中英分文件，不依赖实验结果（Results 留待训练回填） |
| 2026-06-12 | 新增 [`Paper_论文向总览.md`](../../paper/drafts/Paper_论文向总览.md)：把全项目（CL Loss / 7 桶 Buffer / U 形权重 / 沙箱 winner-sync / 模拟用户多轮）收敛成论文骨架（摘要/贡献/方法/实验/系统/局限 + 文档地图），作为新人阅读入口。文档整理：`SandboxRollout.md` 加文档定位提示（§1–3 旧设计已被调度指南取代，现行价值 §4 平台 API）；`项目状态报告.md` 刷新文档清单（18 篇）与推送状态 |
| 2026-06-12 | UserSim 升级为**观察/出题/奖励三 agent 架构**（[`UserSim_多轮Query在线生成.md`](UserSim_多轮Query在线生成.md) §3.2/§3.3）：**观察 agent**（无人设，从 actor 输出判断要收集什么、主动收集中间+最终结果，产出客观报告 R_t）→ 同一份 R_t 喂给 **出题 agent**（有人设，生成下一 query）+ **奖励模型**（据 R_t 中实际产出/效果 + rubric 打分，抗 reward hacking）。出题人设 = **16 个固定人设**（职业/偏好/画像/观察偏好），每会话随机抽 1 个；观察偏好并入人设、取代原轮级视角抽样；观察 agent 无人设。奖励判分 prompt（O4）+ 观察/出题 prompt（O3/O6）留空待实现。接口契约 §7 重写为 Observer/Questioner/Reward 三协议 + ObservationReport 共享结构 + Persona schema；三方各自独立 env 抗 self-preference。代码仍暂缓 |
| 2026-06-11 | 多轮 query 构造方案定稿 → [`UserSim_多轮Query在线生成.md`](UserSim_多轮Query在线生成.md)：真实回流数据只留会话首条 query 作种子，后续 query 由模拟用户 agent 在 winner-sync 边界观察 **winner 单一状态**后在线生成（根除静态多轮数据的前提漂移）；防坍缩 = 轮数压小(1–3 抽样) + 画像库(会话级) + 视角库(轮级)；reward judge 改用 @孙豪 提供的外部 OpenAI 兼容 API（机制走现有 `JudgeClient` env 注入，**地址待提供**），本地 vLLM 降级为 fallback。模拟器/观察 agent **代码暂缓**，接口契约已写入文档 §7；**不增设消融实验**（20 实验已饱和），质量保障 = 过程监控指标（§8.1）+ ClawEval Multi-turn 终点；待设计清单见 §10.1（已分流：O1 种子筛选移交数据侧 @吴健；O5 出题侧安全关闭——已有专门 gate；O2 画像库延后；O3/O4/O6 留空待实现期；O7 全失败兜底已拍板=随机抱怨/结束+抱怨≤3 次/会话） |
| 2026-06-10 | 论文证据钩子（`trainer/replay_metrics.py`）：<br>• buffer 动态日志：每步 `flatten_buffer_stats` 注入 verl/wandb metrics + sidecar `logs/buffer_stats/<exp>.jsonl`（各桶 size/fill_ratio/淘汰/reservoir 拒绝/信号权重）<br>• 激活 `forgetting_risk`：replay 行经 `compute_log_prob` 取当前策略 logprob → `per_row_masked_mean` → `backfill_forgetting` 写回 `current_logprobs` 重算优先级（R4-vs-R5 核心证据）<br>• 按 `trainer.save_freq` 自动 `buffer.dump` → `buffer_dumps/<exp>-step-<N>.sqlite`<br>• `build_replay_rows` 新增 `_replay_tids` 对齐 sidecar；`base.yaml` 增 `buffer_stats_log_freq`/`forgetting_update_freq`<br>• +7 单测，144 passed / 1 skipped |

---

## 下一步行动

| 优先级 | 行动 | 负责人 |
|--------|------|--------|
| **P0** | **训练链路补齐（规则 reward / 数据转换 / 沙箱调度 / rollout 桥）——施工图见 [`Plan_训练链路补齐.md`](../archive/Plan_训练链路补齐.md)** | @孙豪 |
| P0 | 外部 judge API 地址/模型/key 提供并填入 env（`JUDGE_API_BASE/MODEL/KEY`，见 [`UserSim_多轮Query在线生成.md`](UserSim_多轮Query在线生成.md) §6） | @孙豪 |
| P1 | 模拟用户 agent + winner 观察 digest 实现（契约见 UserSim 文档 §7，暂缓启动） | @孙豪 |
| P0 | 全栈 GPU smoke：集群上 B1/R4 跑 1–2 step（多卡 + 真实 27B + 数据） | @孙豪 |
| P0 | 集群上 merge verl 完整 Hydra defaults（fsdp/optim/rollout engine） | @孙豪 |
| P1 | ClawEval manifest 接入（接口已在 `eval/run_eval.py` 定形 + fake 单测） | @杨益博 |
| P1 | 沙盒 rollout 提供带 `bucket`/`messages`/`success_rate` 的真实轨迹 | @郑乃榕 |
| P2 | 27B/64 卡 成本与显存重算（C2）；fully-async 全栈联调 | @孙豪 |
