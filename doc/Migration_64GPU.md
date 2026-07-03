# 64 卡机器迁移 / Session 交接文档

> 本文件用于**跨机器、跨 session 的任务交接**。新 session 上来先读本文件 →
> 再读 [`Progress.md`](Progress.md)（交付状态单一来源）→ 按「冷启动步骤」执行。
> **所有产出（成功 / 失败 / 中间数据 / 结论）必须按 §6 记录到 [`RunLog.md`](RunLog.md)。**

**创建：** 2026-06-10　**源机器：** 单卡 H800（本地开发）　**目标机器：** 64 卡

---

## 1. 一句话目的

把已经写完、在 CPU + 单卡通过约 200 单测的 **Continual-Learning-over-Agentic-LLM**
训练栈（CL Loss + 7 桶 Replay Buffer + verl 0.8.0 无侵入注入），搬到 64 卡机器上
**首次跑通全栈 1–2 step smoke**，然后按 Phase 1→6 路线产出 20 个实验与论文数据。

---

## 2. 现在到了哪一步（交接快照）

| 维度 | 状态 |
|------|------|
| 代码 | **全部完成**。`replay_buffer/` + `trainer/` + `configs/`(21 yaml) + `eval/` + 5 篇 `skills/` |
| 测试 | **约 200 测试函数 / 30 文件**（以集群最近一次 `pytest` 实跑为准；唯一 skip = 全栈 GPU smoke，标 `@pytest.mark.gpu`，需多卡） |
| 论文证据钩子 | 完成：buffer 动态日志 + `forgetting_risk` 回填 + 周期 `buffer.dump`（`trainer/replay_metrics.py`） |
| 唯一阻塞 | **GPU 集群形态联调**：见 §4 待办；细节在 `Progress.md` 的「verl 集成验证清单」 |

> ⚠️ 本机只有 1 张 H800（仅够 replay 路径 CUDA smoke），**全栈训练必须在 64 卡机器上做**。

---

## 3. 新 session 冷启动步骤（Bootstrap）

> 「冷启动」= 新 session 从零把自己拉到可工作状态。**按顺序执行，每步结果写 RunLog。**

```bash
# 0. 定位仓库（git 远端 upstream/dev_train）
cd <repo>/agentic_cl_research && git status && git log --oneline -5

# 1. 读交接 + 状态（先文档后代码）
#    doc/Migration_64GPU.md(本文)  →  doc/Progress.md  →  doc/CL_Update_Sunhao.md
#    架构与约束见仓库根 CLAUDE.md（必读：replay_buffer 不得 import verl/ray；零系数短路；不 fork verl）

# 2. 确认硬件 & 环境
nvidia-smi -L            # 期望 64 张
.venv/bin/python --version   # 期望 CPython 3.10.x；没有则 pip install -e ".[dev]" 重建

# 3. 跑非 GPU 回归，确认搬迁未破坏（期望约 200 passed / 1 skipped）
.venv/bin/python -m pytest -q

# 4. 跑 GPU 标记测试（单卡即可，验证 torch/verl/CUDA 正常）
.venv/bin/python -m pytest -m gpu -q

# 5. 确认外部依赖是否到位（决定能否真训练，见 §7）
#    - 真实 Qwen3.6-27B 权重路径（configs/base.yaml: actor_rollout_ref.model.path）
#    - 训练数据集 train_files/val_files（需含 messages / bucket / success_rate）
#    - verl 完整 Hydra defaults（fsdp/optim/rollout engine）
```

完成 0–5 后，新 session 即「热」，可进入 §4 待办。

### 关于 Buffer 的「冷启动」（区别于上面的 session 冷启动）
Buffer 空时**无需特殊操作**，代码已自处理（`trainer/replay_batch.py`）：
- `len(buffer.store)==0` → `effective_replay_batch_size` 返回 0 → 不追加 replay 行 → `L_replay=0`，纯 RL 跑。
- buffer 渐满时按 `replay_warmup_size` 线性爬坡放大 replay batch，防止小 buffer 被过采样。
- 故 B1（无 buffer）与 R*（有 buffer）启动方式一致，**首个 step 永远先纯 RL**。

---

## 4. 64 卡上待办（按优先级，首项打通 B1 基线）

| 优先级 | 任务 | 验收标准 | 风险点 |
|--------|------|----------|--------|
| **P0-a** | merge verl 完整 Hydra defaults 到 `base.yaml`（fsdp / optim / rollout engine / ref policy） | `validate_config` 通过，无缺字段报错 | key path 已对齐，仅缺引擎字段 |
| **P0-b** | **B1 全栈 smoke**（先 Colocate 64，最简单）：`scripts/train.sh configs/phase1/b1.yaml` 跑 1–2 step | 不崩；wandb 出现 `actor/pg_loss`、`actor/entropy_loss`；产出 step ckpt | DataProto no_padding/padded 形态 |
| **P0-c** | **R4 全栈 smoke**：`scripts/train.sh configs/phase3/r4.yaml` 跑 1–2 step | replay 行不报 shape/KeyError；`actor/replay_loss>0`、`actor/replay_empty` 合理；`logs/buffer_stats/r4.jsonl` 有行 | `_append_replay_rows` concat、`build_replay_rows` response span（chat template offset）、`forgetting` 的 `compute_log_prob` DataProto schema |
| **P1** | 切 **Fully Async Policy 40+24**（`trainer/verl_async_runner.py` scaffold）跑通 | async 下 1–2 step 不崩，staleness 正常 | scaffold 仅导入级验证过 |
| **P1** | 27B/64 卡 **显存 & 同步耗时重算**（替换 doc 里 70B 估算，C2） | 写回 `CL_Update_Sunhao.md` | — |
| **P2** | 跑完整 Phase 1→6（20 实验），逐个产出 ckpt + 评测 | 见 `Progress.md` 20 实验表 | 依赖数据/manifest |

> **建议：先 Colocate 64 跑通形态，再切 40+24 async**——colocate 路径短，能最快暴露 DataProto/response-span 问题。

### 这些点是「代码写好了但只能在 GPU 验」的清单（重点盯）
1. `_append_replay_rows` 的 `DataProto.concat` 在配置引擎下的 padding 模式（no_padding vs padded）。
2. `build_replay_rows` 的真实 response span（当前用全序列近似，集群上须按 chat template offset 重算）。
3. `compute_replay_current_logprobs` 经 verl `compute_log_prob` 取当前 logprob 的 DataProto schema（off-GPU 会优雅降级为 no-op，**GPU 上要确认它真的回填了 `forgetting_risk`**）。

---

## 5. 命令 & 路径速查

```bash
# 单实验训练 / 续训
bash scripts/train.sh configs/phase1/b1.yaml
bash scripts/train.sh configs/phase3/r4.yaml --resume-from ckpts/r4-step-50
# 直接调入口
.venv/bin/python -m trainer.cl_main --config configs/phase3/r4.yaml
# 整个 phase
bash scripts/phase1/run.sh
# 评测
bash scripts/eval.sh ckpts/b1-step-100
# 测试 / lint
.venv/bin/python -m pytest -q            # 非 GPU
.venv/bin/python -m pytest -m gpu -q     # GPU
ruff check . && black --check .
```

| 产物 | 落盘位置（均 gitignore） |
|------|--------------------------|
| 训练 ckpt | `ckpts/<exp>-step-<N>/`（`save_freq` 控制） |
| Buffer 动态日志 | `logs/buffer_stats/<exp>.jsonl`（每步一行，论文画图用） |
| Buffer 快照 | `buffer_dumps/<exp>-step-<N>.sqlite`（按 `save_freq` 自动） |
| wandb | `wandb/`（loss/metrics + `buffer/*` 曲线） |
| 评测结果 | `eval/results/<run_id>/{per_task.json,summary.json}` |
| **人工运行记录** | **`doc/RunLog.md`（见 §6，必须维护）** |

关键开关（`configs/base.yaml` 的 `cl:`）：`buffer_stats_log_freq`、`forgetting_update_freq`、`replay_warmup_size`、`lambda_replay`、`buffer.enabled`。

---

## 6. 记录规范（硬性要求：成功/失败都要留痕 + 解释）

> 用户明确要求：**任何产出的数据和结论，无论失败还是正确，都要有记录，并适当给出解释。**

**做法：** 新 session 每完成一个「可判定结果」的动作（一次 smoke、一次训练、一次评测、一个 bug 的复现/修复），就向 [`RunLog.md`](RunLog.md) **追加一条**，禁止删改历史条目（失败也保留，失败本身是论文/调试的证据）。

每条至少包含：

| 字段 | 说明 |
|------|------|
| 时间 / 机器 / commit | 可复现锚点 |
| 动作 | 跑了什么命令 / 配置 |
| 结果 | ✅成功 / ❌失败 / ⚠️部分；附关键数字（loss、step、报错首行） |
| 产物路径 | ckpt / jsonl / wandb run / 评测 json |
| **解释** | 为什么成功 / 为什么失败、根因、下一步 |

自动化数据（wandb、`buffer_stats/*.jsonl`、`eval/results/*`）负责**量化曲线**；
`RunLog.md` 负责**人能读懂的「发生了什么、为什么」**——两者互补，论文 Results / Appendix 都要用。

---

## 7. 外部依赖阻塞（决定能否「真训练」而非只 smoke）

| 依赖 | 负责人 | 没有它会怎样 |
|------|--------|--------------|
| Qwen3.6-27B 真实权重 | @孙豪 配置路径 | 无法加载 actor，连 smoke 都起不来 |
| 训练数据（含 `messages`/`bucket`/`success_rate` + 会话镜像） | @吴健 / @郑乃榕 | buffer 入桶 & priority 信号缺失，replay 退化 |
| 沙盒 rollout 环境 | @郑乃榕 | 无真实 trajectory |
| ClawEval 195 manifest | @杨益博 | 无法评测遗忘度（接口已定形，缺数据） |
| verl 完整 Hydra defaults | @孙豪 | `validate_config` 报缺字段 |

> 若上述未齐，64 卡 session 仍可做：**形态 smoke（用 dummy/小模型 + 假数据）**，先验证 §4 的三个 GPU-only 点，把结论写进 RunLog，等数据到位再真训练。

---

## 8. 切 session 前确认清单（在本机做）

> 🔴 **当前 git 状态：本 session 全部改动尚未提交！**（40+ 文件 modified + 新增
> `trainer/replay_metrics.py`、`tests/test_replay_metrics.py`、本文、`doc/RunLog.md` 等未跟踪）。
> 分支 `dev_train`，最后提交 `c2e9cc0`。**不 commit/push，64 卡新 session 将看不到这些工作。**

- [ ] **先 `git add` + commit + push 到 `upstream/dev_train`**（运行时产物 ckpts/wandb/logs/.venv/datasets 已 gitignore，不会误提交）
- [x] `doc/Progress.md`、本文、`doc/RunLog.md` 已落盘
- [x] 约 200 passed / 1 skipped 基线（新机器实跑此数即「未搬坏」；以集群实跑为准）

> 新 session 的第一条 RunLog 应是：「在 64 卡机器 `git pull` 后复现 `pytest -q` 结果」。

---

## 附录：集群提交快速参考

> 本节合并自原 `doc/集群训练启动指南.md`（2026-06-19 整理时归档）。覆盖首次提交训练任务前的准备与平台操作。

### A.1 前置条件

| 依赖 | 位置 | 状态 |
|------|------|------|
| 代码 | AFS: `/mnt/afs_toolcall/sunhao4/agentic_cl_research` | ✅ `dev_train` 已 push |
| 模型权重 | AFS: `/mnt/afs_agents/share_models/Qwen/Qwen3.6-27B` | 脚本自动 cp 到 `/tmp/qwen36` |
| Judge 端点 | `configs/agents.yaml` → sufy `anthropic/claude-4.8-opus` | ✅ 已配置 |
| 腾讯/E2B 凭证 | `docker/sandbox/tencent.env` | ✅ 已填写 |
| 镜像 | `qwen36-lightllm` 或基础镜像 + 脚本兜底 pip | 见 §A.3 |

### A.2 环境变量（脚本自动加载）

两个 `.env` 文件由 `start_train.sh` / `run_phases.sh` 自动 source：

**`.env`（训练密钥 + sufy key）：**
```bash
SWANLAB_API_KEY=GDGemFX7c2ruxYtRWzVh0
SUFY_API_KEY=<sufy key>
# Judge/Observer/Questioner 模型与端点统一读 configs/agents.yaml（sufy 网关），
# 不再在 .env 里写 JUDGE_*；下面三行已弃用（保留作旧脚本 fallback）：
# JUDGE_API_BASE=https://openai.sufy.com/v1
# JUDGE_MODEL=anthropic/claude-4.8-opus
# JUDGE_API_KEY=<sufy key>
```

**`docker/sandbox/tencent.env`（腾讯沙箱凭证，R4 实验用）：**
```bash
TENCENT_UIN=100048510516
TENCENTCLOUD_SECRET_ID=AKIDGgxFSVOI9pOmYWWO9MWZUFoEY1QQRK3e
TENCENTCLOUD_SECRET_KEY=...
TENCENTCLOUD_REGION=ap-beijing
E2B_API_KEY=ark_87a5eaa569e168f6dd2c03fe0690b3b7ca5c60c1
E2B_DOMAIN=ap-beijing.tencentags.com
AGS_ROLE_NAME=AgentOS-260506-test
```

### A.3 镜像

推荐使用项目自建镜像（已固化运行时依赖，免去启动时 pip install）：

```
registry.cn-tj-01.sensecore.cn/ccr-zuhu2026/qwen36-lightllm:1.0
```

> 2026-07-01 build + push 完成（commit `3789654`）；**2026-07-03 从旧租户 `ccr-devsfttj` 迁至新租户 `ccr-zuhu2026`**（登录名 `zuhu2026-sunhao4`，retag+push 未重 build，image ID 不变 `8631ec6c9b9b`）。
> base 镜像仍是泽寰 `ccr-devsfttj/verl:cu129_..._0211`（Dockerfile `ARG BASE_IMAGE`，23GB）——重 build 时需确认新租户能否拉旧租户 base，或把 base 也 retag 过去。

若用其他基础镜像，脚本会自动执行兜底 `pip install`（约 2 分钟）。

### A.4 平台提交命令

平台需注入 `RANK`、`MASTER_ADDR`、`MASTER_PORT` 三个环境变量，脚本内部按 `RANK` 区分 head/worker 节点。

| 字段 | 值 |
|------|-----|
| **启动命令** | `bash /mnt/afs_toolcall/sunhao4/agentic_cl_research/scripts/start_train.sh` |
| **节点数** | 8 |
| **每节点 GPU** | 8（共 64 卡） |
| **镜像** | 见 §A.3 |

B1 是纯 RL 遗忘基线（无 KL、无 replay）。B1 与 R4 **共用同一套 agentic 多轮 rollout**（CLSchedulerAgentLoopManager + 16×8 winner-sync + ReAct 沙箱），差异仅在 CL 项——否则拿"多轮执行完成"的 R4 与单轮 B1 比遗忘不公平。最简配置，用来验证全栈链路。

跑别的实验（如 R4）：
```bash
bash /mnt/afs_toolcall/sunhao4/agentic_cl_research/scripts/start_train.sh configs/run/r4.yaml
```

串跑多 phase：
```bash
# 默认：b1 + r4
bash /mnt/afs_toolcall/sunhao4/agentic_cl_research/scripts/run_phases.sh
# 自定义顺序
bash /mnt/afs_toolcall/sunhao4/agentic_cl_research/scripts/run_phases.sh configs/run/b1.yaml configs/run/r4.yaml
```

**平台环境变量注入**：脚本已通过 `RANK` 区分角色：`RANK=0` → 启动 ray head → 跑训练 → ray stop；`RANK≠0` → 启动 ray worker → `--block`。无需额外设置。

### A.5 脚本内部流程

```
1. source .env           → SUFY_API_KEY / SWANLAB_API_KEY（Judge/Observer/Questioner 读 configs/agents.yaml）
2. source tencent.env    → E2B / 腾讯云凭证（R4 沙箱 rollout 用）
3. useradd sunhao4       → 集群容器内建用户（AFS 权限）
4. pip install 兜底       → 镜像已含则秒过
5. cp 模型到 /tmp/qwen36 → 各节点本地盘读，避 AFS 带宽抢占
6. torch.distributed barrier → 等所有节点就绪
7. RANK=0: ray head + 训练 + ray stop
   RANK≠0: ray worker --block
```

### A.6 实验快速对照

| 实验 | 配置 | KL | Replay | Agentic Rollout | 说明 |
|------|------|:--:|:------:|:---------------:|------|
| **B1** | `configs/run/b1.yaml` | ❌ | ❌ | ❌ | 纯 RL 遗忘基线 |
| **R4** | `configs/run/r4.yaml` | ❌ | ✅ (λ=0.5) | ✅ (e2b) | 7 桶 + 抗遗忘 |

配置继承链：`ppo_trainer.yaml` → `base.yaml` → `cluster.yaml` → 各实验 yaml

### A.7 注意事项

**Judge 端点：**
- 当前使用 **anthropic/claude-4.8-opus**（sufy 网关，`configs/agents.yaml: reward`），`max_tokens` 已从 256 提至 4096
- claude-4.8-opus 为 thinking 模型，`max_tokens` 须给足（含 reasoning_tokens），256 会吃光导致 `judge_error=1.0`
- 若 judge 不可达：`parse_judge_output` 返回全 0（completion=0, safety=0, robustness=0），**不报错**，但 reward 全零 → 训练无信号。务必在日志里确认 `judge_error=0`

**Mock Judge 逻辑：**
- `start_train.sh`：`JUDGE_MODEL == "mock-judge"` 才启动 mock（当前 `anthropic/claude-4.8-opus` → 不启动）
- 如需回退 mock：在 `configs/agents.yaml` 把 `reward.model` 改回 `mock-judge`

**B1 与 R4 同走沙箱（方法学一致性）：**
- `b1.yaml` 与 `r4.yaml` 都继承 `cluster.yaml` 的 agentic rollout（CLSchedulerAgentLoopManager + e2b 沙箱），差异仅在 CL 项（B1 关 replay / 无 KL）。
- 旧的 `rollout.agent: null`（B1 单轮）已废弃——拿多轮 R4 和单轮 B1 比遗忘不公平。

**AFS 限制：**
- `HF_HOME` / `HF_DATASETS_CACHE` 指向 `/tmp`（AFS 不支持 `fcntl.flock`）
- Triton / FlashInfer 编译缓存同样指向 `/tmp`（`_node_worker.sh` 已处理）

### A.8 验证 Checklist

提交前在本地确认：
- [ ] `git pull` 拿到最新 `dev_train`
- [ ] `.env` 含 `SUFY_API_KEY`（Judge/Observer/Questioner 共用，模型读 `configs/agents.yaml`）
- [ ] `docker/sandbox/tencent.env` 含 E2B 凭证（R4 需要）
- [ ] 模型权重 `/mnt/afs_agents/share_models/Qwen/Qwen3.6-27B` 存在
- [ ] `curl -sS https://openai.sufy.com/v1/models -H "Authorization: Bearer $SUFY_API_KEY"` 返回 200

提交后确认：
- [ ] rank0 日志出现 `ray status` 输出
- [ ] rank0 日志无 `judge_error=1.0`（说明 judge 端点正常）
- [ ] 训练 step 0 产出 `actor/pg_loss` 等指标
- [ ] swanlab 面板可见 loss 曲线
