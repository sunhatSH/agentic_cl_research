# Sandbox Agent 架构：动作在内、推理在外

> **定位**：回答「沙箱里跑什么、推理在哪、镜像装什么、起实例前要做什么、用户文件系统怎么随机化」。
> 调度（16×8 / winner 同步）见 [`Sandbox_管理调度指南.md`](Sandbox_管理调度指南.md)；平台 API 见 [`SandboxRollout.md`](../../archive/SandboxRollout.md)；运维见 [`Sandbox_腾讯云操作手册.md`](../../archive/Sandbox_腾讯云操作手册.md)。

**最后更新**：2026-06-10 ｜ 决策：agent harness = **OpenClaw**（敲定待选项）

---

## 1. 一句话架构

```text
┌─────────────────────────  GPU 集群（沙箱外）  ─────────────────────────┐
│  策略推理 = Qwen3.6-27B via vLLM（OpenAI 兼容 endpoint）                │
│  —— 只做「下一步动作的 token 生成」，不碰文件系统                        │
└───────────────▲───────────────────────────────────┬───────────────────┘
                │ HTTP（prompt/对话历史）            │ 生成的 assistant 消息
                │                                    │（含 tool_call）
┌───────────────┴────────────────────────────────────▼───────────────────┐
│  腾讯云 Sandbox Instance（沙箱内）                                       │
│  OpenClaw agent harness：                                               │
│    - 维护会话 / 把 query+历史 组 prompt → 调外部 vLLM                    │
│    - 解析模型输出的 tool_call → 在本沙箱执行（bash/read/write/edit/...） │
│    - 把 observation 拼回历史，循环，直到 final answer                    │
│  文件系统 = 随机加载的「用户机器」种子（见 §4）                          │
└──────────────────────────────────────────────────────────────────────┘
```

**推理（thinking）在沙箱外，动作（action / tool 执行）在沙箱内。** OpenClaw 本身「不做思考，是 plumbing」——会话管理、工具注册、动作执行；智能来自外部 LLM。

---

## 2. 为什么这样分

| | 放沙箱外 | 放沙箱内 |
|--|----------|----------|
| **推理（27B 前向/采样）** | ✓ 需要 GPU、要算 logprob 供训练，128 个沙箱不可能各带一份 27B | ✗ |
| **动作执行（装包、写文件、跑命令）** | ✗ 会污染训练机、无隔离 | ✓ 毫秒级快照、可并发数百、天然隔离 |
| **agent loop（OpenClaw）** | 可选 | **选内**：动作紧邻文件系统、与 winner 同步的环境状态一致 |

把 OpenClaw 放沙箱内的关键理由：agent 的每个动作都要落到「那台用户机器」的磁盘上，而 winner 同步、母版派生针对的就是这块磁盘——agent loop 与它执行的磁盘必须在同一个实例里。

---

## 3. 轨迹收集（务必读：千万别自己造 proxy）

### 3.0 轨迹由谁收集 —— 用框架原生，不要事后爬

标准 RL（verl/OpenRLHF/TRL）里，**轨迹是 rollout 生成的副产物**：生成引擎在采样时就顺手返回 token id + logprob，多轮/agentic 时框架自己跑 tool loop，把 assistant token 与 observation token 拼成一条序列，用 `response_mask` 区分谁进 loss。

verl 已自带 `verl/experimental/agent_loop/`（`ToolAgentLoop` = ReAct + tool calling），其 `AgentLoopOutput` **原生**给出整条轨迹：

| 字段 | 含义 |
|------|------|
| `prompt_ids` / `response_ids` | prompt 与「生成+tool 返回」拼接的 token |
| `response_mask` | **1=模型生成 token（进 loss/advantage）；0=observation/tool 返回/padding** |
| `rollout_log_probs` | 行为策略 logprob（`calculate_log_probs` 开启时） |
| `num_turns` / `tool_calls` | 轮数、工具调用数 |

`trainer/trajectory_adapter.py` 读的正是这些原生字段（`rollout_log_probs`/`responses`/`response_mask`）——**buffer 侧早就对齐，轨迹收集这一段框架免费给。** 上一轮提过的「在 vLLM 前架 recording proxy」**仅作 W1 的最后兜底**，不是首选。

### 3.1 两种接线（与 Gap D 对应）

| | W1：OpenClaw 驱动 loop（自管调度） | **W2：verl 驱动生成（推荐收集机制）** |
|--|------------------------------------|----------------------------------------|
| 单步生成/轨迹 | OpenClaw→外部 vLLM HTTP；token+logprob 要靠 proxy 兜底抓 | **verl rollout 原生返回 token+logprob+mask** |
| 沙箱角色 | 完整 agent harness | tool 执行器（verl tool_registry 注册沙箱 tool，OpenClaw 可选） |
| winner 同步 | 好实现（调度自主） | **难**：见 §3.2 冲突 |

### 3.2 关键冲突：winner-sync vs verl 批量 rollout（别忘了）

verl 的范式是「整批 prompt **一次性生成完** → 再统一打分」。而调度指南规则 1 要求 **会话内 query_2 开始前先用 query_1 的 reward 选 winner、同步 8 槽** —— 这是 query 之间的耦合，**verl 单次 rollout 内表达不了**。这正是当初倾向自管调度的原因。

**调和方案（既不造 proxy、又保留 winner-sync）**：

> 由我们编排会话循环（16×8 + 每 query 后 winner 同步），但**每一步生成都调用 verl rollout 引擎的 generate**（原生返回 token+logprob+mask），而不是走 HTTP proxy。即 **「框架负责单步生成与轨迹，我们负责会话级编排」**。

落点：`rollout/scheduler.py` 不自己跑模型，而是复用 verl 的 generate / agent_loop 单步，把每条 query 当一个 episode 收集，episode 之间做 winner-sync。

### 3.3 OpenClaw 的定位（随之而定）

- 若用 verl `ToolAgentLoop` + tool_registry：动作 tool 直接代理到沙箱，**OpenClaw 可不进 loop**（或仅提供 tool 实现）。
- 若坚持 OpenClaw 驱动：回退到 W1 + proxy 兜底抓 logprob。

> 两种接线**镜像完全相同**（都只需 Node+工具+种子），镜像可现在就构建；接线/收集方式由 Gap D 敲定，但**收集机制默认用 verl 原生**。

---

## 4. 随机贴合用户的文件系统环境（Q3）

### 4.1 约束先行：随机 ≠ 组内发散

调度指南规则 1：**同一 query 的 8 个槽必须位级一致**，且都从固定母版派生。所以「随机用户环境」**不能**在镜像 build 时随机（那样母版就不唯一了），也不能让 8 个槽各随机各的。

正确粒度：**按 query/会话随机选一个 persona，对该组 8 个槽施加同一个 persona**。

### 4.2 三层职责

```text
镜像 build（母版，固定）
    └─ 烘焙「persona 种子语料库」/opt/agentic-cl/fs-seeds/ + 物化脚本
       —— 只拷贝语料和脚本，绝不在 build 时物化（保证母版唯一、位级一致）

调度器（沙箱外，决定随机性）
    └─ 给一个 query/会话选 persona：seed = hash(query_id) 或显式 persona_id
       通过 envVars 注入 AGENTIC_CL_PERSONA / AGENTIC_CL_FS_SEED
       —— 同组 8 个槽注入相同值

实例启动（沙箱内，物化）
    └─ bin/seed_workspace.sh 读 env → 从语料库确定性物化该 persona 文件到工作区
       —— 同 seed → 同结果 → 8 槽位级一致
```

### 4.3 Dockerfile 在这件事上做什么

1. `COPY fs-seeds/ /opt/agentic-cl/fs-seeds/`：烘焙 persona 语料库（每个 persona 一个目录 + `manifest.json`）
2. `COPY bin/ /opt/agentic-cl/bin/`：物化脚本 `seed_workspace.sh`（确定性、幂等）
3. **不** `RUN` 物化脚本——物化发生在实例启动，不发生在 build

### 4.4 persona 语料库结构

```text
/opt/agentic-cl/fs-seeds/
├── manifest.json            # 列出所有 persona、对应 bucket、权重
├── finance_analyst/         # Finance 桶：季度 csv、对账 notes
├── sysops_engineer/         # SysOps 桶：配置文件、日志、运维脚本
└── office_assistant/        # OfficeQA 桶：文档、待办、表格
```

`manifest.json` 给每个 persona 标 `buckets`，调度器可按目标 query 的桶选贴合的 persona，或按 seed 均匀随机。语料库随任务演化扩充（新的用户数据可直接做成新 persona）。

### 4.5 确定性物化（关键）

`seed_workspace.sh` 必须满足：**相同 `AGENTIC_CL_PERSONA`/`AGENTIC_CL_FS_SEED` → 字节级相同的工作区**。否则 8 槽起点不一致，破坏 GRPO 组内公平。实现：固定 persona 选择函数 + `cp -a`（保留 mtime）+ 不引入随机文件名/时间戳。

---

## 5. 沙箱需要哪些依赖（Q2/Q4）

| 层 | 依赖 | 用途 |
|----|------|------|
| **Agent harness** | Node 24（或 ≥22.19）+ `openclaw` (npm -g) | 在沙箱内跑 agent loop、执行动作 |
| **核心动作工具** | `git curl wget jq ripgrep unzip zip less procps bash`（apt） | OpenClaw 的 bash/read/write/edit 工具常用 |
| **办公/文档任务** | `poppler-utils pandoc fonts-noto-cjk` + py `pandas numpy openpyxl python-docx pdfplumber` | Finance/OfficeQA 桶的表格、PDF、docx |
| **网络/数据** | py `aiohttp httpx requests pyyaml python-dateutil` | 任务里调 API、解析 |
| **（可选）浏览器** | Chromium + Playwright | ClawEval 网页类任务；纯文本 195 子集**不需要**，默认不装（重） |

依赖只放「动作执行」需要的；**不放任何模型/GPU/推理依赖**（vLLM、torch、27B 权重都在沙箱外）。

---

## 6. 起实例之前要做的前置操作（Q4）

按时间顺序：

| # | 操作 | 在哪做 | 一次性? |
|---|------|--------|---------|
| 1 | **build 母版镜像**：本文件的 Dockerfile → CCR | 构建机 | 一次（任务变了才重建） |
| 2 | **创建 Tool 模板**：用母版镜像建北京区 Tool，网络选 `PUBLIC`（要联网调外部 vLLM/装包） | 控制台/CLI | 一次 |
| 3 | **拉起外部 vLLM**：27B OpenAI 兼容 endpoint，**开 `logprobs`** | GPU 集群 | 每次训练 |
| 4 | **准备 envVars**（起实例时注入，**不**写进镜像 ENV）：<br>· `OPENAI_API_BASE` → vLLM endpoint · `OPENAI_API_KEY` · `OPENCLAW_MODEL`<br>· `AGENTIC_CL_PERSONA`/`AGENTIC_CL_FS_SEED`（同组 8 槽相同）<br>· `AGENTIC_CL_WORKSPACE` 工作区路径 | 调度器 | 每组 |
| 5 | **派生 8 槽**：同 Tool + 同 envVars → 8 个 Instance | 调度器 | 每组 |
| 6 | **实例内**：`seed_workspace.sh` 物化 persona → `agent_entry.sh` 渲染 openclaw 配置（endpoint 来自 env）→ 跑 query | 沙箱内 | 每 query |

> 第 4 步对应「不在 Dockerfile 设 ENV」的快照约束：所有运行时配置走 `envVars`，已由 `rollout/sandbox_env.py` + `configs/sandbox_runtime_env.json` + `docker/sandbox/runtime.env` 支持。

---

## 7. 与镜像文件的映射

| 文件 | 作用 |
|------|------|
| `docker/sandbox/Dockerfile` | 母版：Node+OpenClaw+工具+种子语料 |
| `docker/sandbox/requirements.txt` | 沙箱内 Python 动作依赖 |
| `docker/sandbox/bin/seed_workspace.sh` | 实例启动时确定性物化 persona |
| `docker/sandbox/bin/agent_entry.sh` | 渲染 openclaw 配置 + 跑一次 agent turn |
| `docker/sandbox/openclaw.config.template.json` | OpenClaw 配置模板（endpoint 占位，env 渲染）|
| `docker/sandbox/fs-seeds/` | persona 语料库 + `manifest.json` |

---

## 7b. Reward judge（模型判分）

> **选型权威见 [`模型选型.md`](../../archive/模型选型.md)**（单一信源，由 @孙豪 拍板：reward 走 sufy 托管的 `anthropic/claude-4.8-opus`，冻结）。本节只保留**为什么这么设计**的论证；具体用哪个模型 / endpoint / 校准状态以选型文档为准。一致率校准是**可选验证**，非选型阻塞。

**倾向（2026-06-10，可改）**：reward 用**单一冻结模型 judge**。

| 为什么不用规则 | 说明 |
|----------------|------|
| 尺度不齐 | 规则[0,1] 与语义分分布不同，混用污染跨桶 advantage 基线 |
| 覆盖不到 | Communication/Dialogue/无 gold 的 Knowledge 规则判不了 |
| reward/eval 一致 | ClawEval 本身就是模型 judge（completion/safety/robustness rubric） |

**本地 vs API → sufy 托管冻结**：RL reward 必须是不变的尺子，跑数周/21 实验测遗忘；judge 走 sufy（`anthropic/claude-4.8-opus`），由网关托管保证冻结（不再本地 vLLM 部署，避免版本漂移污染遗忘度量）。判分对象 = ClawEval 三维 rubric，与评测同构。

**模型大小**：anti reward-hacking 要求 judge ≥ 策略（27B）；有 rubric 则核对清单较易。`claude-4.8-opus` 能力远超 27B 策略，用 ClawEval 人工 rubric **一致率**校准，不是凭参数量定。

**代码**：`trainer/model_reward.py`——`JudgeClient` 抽象 + `compute_score` 委托，**模型不写死**，从 `configs/agents.yaml: reward`（`JUDGE_API_BASE`/`JUDGE_MODEL` env 解析）读取；judge 故障记 `judge_error` 不崩 batch。rubric 来源 = `extra_info.checkers`（Gap B 抽取）/ `extra_info.rubric`。

**定 judge = 一致率校准（不是凭参数量）**：

- 工具：`eval/judge_agreement.py`（MAE/pearson/Cohen's kappa/F1 + 按桶分解 + `rank_judges`）+ `scripts/calibrate_judge.py`（跑候选 endpoint→算 judge↔人工一致率→**推荐能过阈值的最小模型**）。
- 流程：①各候选配成不同 sufy 模型 id → ②`calibrate_judge.py --labeled <人工标注> --judges <候选 json>` → ③按 pass kappa 排名，软桶（Communication/Dialogue）一致率掉得多的才往上加大模型。
- **标注数据契约**（每行一条人工评过的轨迹）：

```json
{"task":"...","trajectory":"...","rubric":"...","bucket":"SysOps",
 "human":{"completion":1.0,"safety":1.0,"robustness":0.5},"pass":true}
```

- ⚠️ **阻塞**：ClawEval 人工 rubric 标注数据依赖评测 manifest；到货前可用同 schema 的小规模人工 pilot 集先跑通校准。工具链已就绪，数据一到即可定 judge。

```text
trajectory(solution_str) + task(queries) + rubric(checkers)
        → build_judge_prompt → JudgeClient.score（sufy claude-4.8-opus，冻结）
        → {completion, safety, robustness} → safety*(0.8c+0.2r) → reward
```

---

## 8. 待确认 / PoC

1. **OpenClaw 配置 schema**：`openclaw.config.template.json` 是按文档的最佳猜测，构建后用 `openclaw doctor` 校准实际字段名（provider/baseUrl/model 的确切 key）。
2. **headless 单轮**：确认 `openclaw agent --message ...` 能无 channel、无 daemon 跑通一次 agent turn 并吐出可解析的 trace（轨迹来源）。
3. **logprob 通路**（Gap D 核心）：vLLM OpenAI endpoint 返回的 `logprobs` 能否对齐成 verl 训练所需的 per-token logprob。
4. **镜像体积**：Node+OpenClaw+pandoc 后镜像可能较大，评估冷启动/派生耗时；浏览器默认不装以控体积。
5. **reward judge 选型**：选定 judge 模型 + 起 `serve_reward_model.sh` endpoint + ClawEval 一致率校准（§7b）。
