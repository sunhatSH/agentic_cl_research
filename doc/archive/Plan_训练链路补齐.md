# 训练链路补齐计划（执行版）

> **本文是交给实现者（AI/人）的施工图**：列出 64 卡 27B 正式训练前所有残缺模块的实现规格。
> 状态快照见 `doc/archive/Progress.md`；调度设计见 `doc/ops/sandbox/Sandbox_管理调度指南.md`；CL 设计见 `doc/source/CL_Design.md`。

**写作日期**：2026-06-10（2026-06-10 增补：agent harness 决策 + 沙箱镜像层）
**前提**：64 卡集群 + Qwen3.6-27B 权重已具备（@孙豪 确认）。
**核心决策（已定，不要改）**：
- 奖励用**规则计算**，不用显式 reward model（`reward_model.enable` 保持 false）。
- agent harness = **OpenClaw**（沙箱内执行动作，推理在沙箱外调 vLLM）。架构见 `doc/ops/sandbox/Sandbox_Agent架构.md`。

---

## 0. 现状快照（已完成，不要重做）

| 模块 | 状态 | 证据 |
|------|------|------|
| Replay Buffer 7 桶 | ✅ | `replay_buffer/`，单测 100+ |
| CL Loss 注入 verl | ✅ | `trainer/cl_loss.py` + `verl_runner.inject_cl_loss`，replay 行双 mask，CUDA smoke 通过 |
| **优势计算（GRPO）** | ✅ | `configs/base.yaml` `algorithm.adv_estimator: grpo`，verl 内置组内归一化，组大小 = `rollout.n: 8` |
| buffer 动态日志 / forgetting_risk / 快照 | ✅ | `trainer/replay_metrics.py` |
| 20 实验 yaml | ✅ | `configs/phase1..6/`，校验测试覆盖 |
| 沙箱连通（北京区） | ✅ | `rollout/sandbox_client.E2BSandbox`，REST + `X-Access-Token` |
| **沙箱镜像层（agent harness）** | ✅ code | `docker/sandbox/Dockerfile`（Node24+OpenClaw+工具+persona 种子）、`bin/seed_workspace.sh`（确定性物化，已冒烟）、`agent_entry.sh`、`fs-seeds/`；待 build 验证（§Gap H） |
| 单测基线 | ✅ | 约 200 测试函数 / 30 文件（以集群实跑为准） |

**仓库规约**（实现时必须遵守）：

- Python ≥ 3.10；ruff/black line-length=110；ruff 规则 E,F,W,I,B,UP
- `replay_buffer/` 禁止 import verl / Ray；新增 rollout 侧代码同样保持与 verl 解耦（reward 模块除外，见 Gap A 说明）
- 每个 Gap 完成后跑 `.venv/bin/python -m pytest -q` 全量回归，不允许破坏现有 175 个测试
- 配置一律走 OmegaConf yaml（`configs/base.yaml` 是 verl key-path 覆盖层）

---

## Gap A（P0）：Reward —— 🟡 方案与 judge 规模【暂定】，代码已就位

> **状态（2026-06-10）：reward 方案 + judge 模型规模均暂定，尚未拍板。** 原因：任务/结果的判断难度未知。**决策门 = 先有数据（哪怕几十条人工标注 pilot）→ 跑 `scripts/calibrate_judge.py` 看每桶一致率与分数分布 → 才定 judge 规模 / 是否需要规则补充。** 在此之前不锁定。
>
> 代码已为暂定而设计、不阻塞：judge 模型不写死（env 解析）、reward 是 `compute_score` 单文件 + `JudgeClient` 抽象接口，换方案只换实现不动调用方。
>
> **倾向方案**（2026-06-10，可改）：单一冻结模型 judge。理由：①规则[0,1]与语义分尺度不齐，混用污染跨桶 advantage 基线；②规则覆盖不到 Communication/Dialogue 等语义桶；③ClawEval 本身就是模型 judge，reward 用同构保证 reward/eval 一致。
>
> 实现：`trainer/model_reward.py`（抽象 `JudgeClient`，`compute_score` 委托，**模型不写死**，从 `JUDGE_API_BASE`/`JUDGE_MODEL` 解析）+ `tests/test_model_reward.py`（6 passed，mock judge）+ `scripts/serve_reward_model.sh`（vLLM 起本地冻结 judge，模型路径参数化）+ `base.yaml reward.*` 指向 model_reward。
>
> judge 部署：走 sufy 网关托管的 `anthropic/claude-4.8-opus`（冻结，能力远超 27B 策略以 anti reward-hacking）；需用 ClawEval 人工 rubric 一致率校准。详见 `doc/ops/sandbox/Sandbox_Agent架构.md` reward judge 节。
>
> **待办**：选定 judge 模型 + 起 endpoint + 一致率校准。下方规则方案已废弃，仅留作历史背景。

<details><summary>（已废弃）原规则方案</summary>

### A.1 verl 0.8 接入点（已验证，按此实现）

- `reward.reward_model.enable: false`（verl 默认即 false，**不要**起 reward model worker）
- `reward.reward_manager.name: naive`（verl 默认）——batch 结束后逐样本调用 custom 函数
- 注入键：`reward.custom_reward_function.path` + `.name`，verl 的 `get_custom_reward_fn` 会从该文件动态加载函数

函数签名（verl naive reward manager 约定）：

```python
def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict | None = None) -> float | dict:
    """返回 float，或 {"score": float, ...附加指标}（附加指标会进 metrics）。"""
```

### A.2 新建文件

```
trainer/rule_reward.py          # compute_score 入口 + 规则注册表
tests/test_rule_reward.py       # 纯 Python 单测，不依赖 verl 运行时
```

> 放 `trainer/` 而非 `rollout/`：它是训练侧组件，被 verl 动态加载；但函数体本身保持纯 Python（不 import verl），便于单测。

### A.3 评分公式（对齐 ClawEval，见 `doc/source/ClawEval_Metadata.md`）

\[
score = s_{safety} \times (0.8 \cdot s_{completion} + 0.2 \cdot s_{robustness})
\]

| 分量 | 规则来源 | 取值 |
|------|----------|------|
| `s_safety` | 黑名单规则：危险命令（`rm -rf /`、改系统盘、外发密钥 pattern 等）出现即 0 | {0, 1} |
| `s_completion` | 任务级 checker（见 A.4），从样本 `extra_info["checkers"]` 读取 | [0, 1] |
| `s_robustness` | 格式/过程规则：tool_call JSON 可解析、无未捕获 traceback、轮数 ≤ 上限、final answer 非空 | [0, 1]，各子项等权平均 |

### A.4 checker 规格（数据侧契约，与 Gap B 联动）

每条训练样本的 `extra_info` 携带：

```json
{
  "bucket": "SysOps",
  "checkers": [
    {"type": "regex",       "target": "final_answer", "pattern": "...", "weight": 1.0},
    {"type": "file_exists", "path": "/workspace/out.csv",               "weight": 1.0},
    {"type": "sandbox_assert", "code": "import os; assert ...",         "weight": 2.0},
    {"type": "tool_called", "tool_name": "write_file", "min_times": 1,  "weight": 0.5}
  ]
}
```

- `s_completion` = 通过 checker 的加权分 / 总权重
- `regex` / `tool_called` 在 `solution_str`（完整轨迹文本）上离线判定，**不需要沙箱**
- `file_exists` / `sandbox_assert` 需要在该轨迹对应的沙箱实例里执行 → 由 rollout 侧在**销毁实例前**统一跑 checker，把结果写进轨迹 meta（`checker_results`），`compute_score` 只读结果不重连沙箱（避免 reward 阶段依赖沙箱生命周期）
- **没有 checker 的样本**：fallback = `s_robustness` 风格的过程规则 + 标记 `"checker_missing": 1.0` 进附加指标，便于统计覆盖率

### A.5 配置改动

`configs/base.yaml` 追加：

```yaml
reward:
  reward_manager:
    name: naive
  custom_reward_function:
    path: trainer/rule_reward.py
    name: compute_score
```

### A.6 验收

- 单测：safety 命中→0；完整 checker→加权正确；无 checker→fallback；返回 dict 时附加指标透传
- 集成：`tests/test_verl_smoke.py` 风格——`get_custom_reward_fn(cfg)` 能加载到函数（仅导入级，不跑训练）

</details>

---

## Gap B（P0）：数据转换 —— ✅ 已实现（待填 base.yaml train_files）

> 2026-06-10：`scripts/convert_dataset.py` + `tests/test_convert_dataset.py`（6 passed）。500 条冒烟：checker 覆盖 ~70%、bucket 分布正常、parquet 可读。**待办**：在集群上跑全量 + 把 `data.train_files/val_files` 指向产物。下方为原始规格。

### B.1 输入 / 输出

| | 格式 | 位置 |
|--|------|------|
| 输入 | Queries JSONL（`{"record_id": "...", "queries": [...]}`） | 上游 data-filter 产出 |
| 输出 | verl `rl_dataset` 兼容 parquet | `datasets/train.parquet` / `datasets/val.parquet` |

### B.2 新建文件

```
scripts/convert_dataset.py      # jsonl -> parquet 转换 CLI
tests/test_convert_dataset.py   # 用 3~5 条手造记录测字段映射
```

### B.3 字段映射（verl rl_dataset 约定）

| parquet 列 | 来源 |
|------------|------|
| `prompt` | messages 中首个任务 query 前的上下文（chat template 由 verl/tokenizer 处理，存 messages 列表即可，`data.return_raw_chat: true`） |
| `data_source` | 固定 `"agentic_cl"`（`compute_score` 的第一个参数，用于规则分流） |
| `reward_model.ground_truth` | checker 主答案（无则空串） |
| `extra_info` | `{record_id, bucket, checkers, queries(会话内 query 列表)}` |

- **bucket 标注**：复用 `trainer/domain_tagging.py` 的逻辑给每条 record 打 7 桶标签；打不出的标 `unknown`（buffer 侧已支持跳过未标注桶）
- **train/val 切分**：按 record_id 哈希 98/2，固定种子，保证可复现
- **checker 生成**：首版从 messages 的最后一条 assistant/tool 内容里提取可验证目标（regex 模板）；提取不出的留空走 fallback——**不要阻塞在人工标注上**，覆盖率作为指标输出

### B.4 配置改动

```yaml
data:
  train_files: datasets/train.parquet
  val_files: datasets/val.parquet
  return_raw_chat: true
```

### B.5 验收

- `python scripts/convert_dataset.py --input datasets/queries.jsonl --out-dir datasets/` 产出两个 parquet
- 打印统计：总数、各桶分布、checker 覆盖率
- 单测覆盖：字段齐全、哈希切分稳定、空 messages 跳过

---

## Gap C（P0）：沙箱调度 16×8 —— ✅ 骨架已实现（mock 后端）

> 2026-06-10：`rollout/session_pool.py`（spawn/run_query/pick_winner/sync_to_winner/run_checkers/run_session）+ `rollout/scheduler.py`（16 会话并行）+ `tests/test_session_pool.py`（8 passed）。验证了同起点、winner-sync、兜底规则、不回写母版、16×8 拓扑。**待办**：接真实腾讯后端的 pause→fork sync（§6 D1）+ 128 并发实测。下方为原始规格。

**规格以 `doc/ops/sandbox/Sandbox_管理调度指南.md` 为准**（§4 状态机、§6 平台路径、§7 接口）。要点重述：

- 全局母版固定；每条 queries 会话从母版派生 8 槽
- 会话内每条 query 跑完：选 winner（含同分/全失败兜底）→ 8 槽磁盘 + **对话上下文** 都对齐 winner
- 会话结束销毁全部实例，不回写母版；每 step 16 会话 × 8 槽 = 128 并发

### C.1 新建文件

```
rollout/session_pool.py         # SessionSandboxPool：spawn(8) / run_query / sync_to_winner / destroy_all
rollout/scheduler.py            # RolloutScheduler：16 会话并发编排（asyncio 或线程池）
tests/test_session_pool.py      # 用 LocalSandbox/mock 测状态机，不打真实腾讯云
```

### C.2 接口（与指南 §7 伪代码一致）

```python
class SessionSandboxPool:
    def __init__(self, master_template: str, slots: int = 8, backend: str = "e2b"): ...
    def spawn(self) -> None: ...                       # 从母版起 slots 个实例
    def run_query(self, query: str, agent_fn) -> list[Trajectory]: ...   # 8 路并行
    def sync_to_winner(self, winner_idx: int) -> None: ...               # winner 保活, 仅替换其余 7 槽（绝不 kill winner）
    def run_checkers(self, checkers: list[dict]) -> list[dict]: ...      # Gap A.4：销毁前跑 sandbox checker
    def destroy_all(self) -> None: ...
```

- `sync_to_winner` 走指南 §6 的 **D1 路径**：**winner 保活**，只 kill 输家 7 个 → 从存活 winner fork/覆盖出 7 个替补。**会话内绝不能 kill winner**（它是会话累积状态的唯一活载体，含进程/内存态；kill 了再靠快照重建会断依赖链——已弃用的 D2 杀重建即此坑）
- winner 兜底：同分/全失败 → 随机槽；打分器异常 → 本 query 不入训练、槽位保持上一边界状态
- **PoC 先行**（实现 pool 前必须跑通）：单会话 8 槽 2 query，验证 sync 后 7 个替补与**存活 winner** 文件一致、winner 全程未死；128 并发配额实测

### C.3 验收

- mock 后端单测：状态机走通、兜底规则、destroy 后无泄漏
- 真实后端冒烟脚本（`scripts/sandbox_smoke.py` 扩展 `--session` 模式）：2 槽 × 2 query 全流程

---

## Gap D（P0，决策项）：rollout ↔ verl 的桥 —— ✅ dry-run 骨架已实现

> 2026-06-10：`rollout/collect.py`（`GenStep` 镜像 verl 原生字段 / `make_react_agent_fn` 装配 token+logprob+mask / `ingest_trajectories` 入桶，B12 跳过未标注）+ `tests/test_collect.py`（4 passed）。dry-run 链路：mock 单步生成 → ReAct loop → scheduler 2×2 → 规则打分 → 真实 7 桶 buffer，全程无 GPU。**待办**：`GenerateFn` 接真实 verl rollout generate（拿真 token+logprob）+ `verl_async_runner` 接通 + 集群全栈 smoke。下方为原始规格。

verl 的 rollout（vLLM 生成）需要在生成过程中执行 tool call 并把 observation 拼回上下文。两条路线，**实现者先做 D-2 的可行性确认，再动工**：

**轨迹收集 = 用框架原生，别造 proxy。** verl 自带 `verl/experimental/agent_loop/`（`ToolAgentLoop`），其 `AgentLoopOutput` 原生给出 `prompt_ids`/`response_ids`/`response_mask`（1=生成 token，0=observation）/`rollout_log_probs`/`num_turns`——`trajectory_adapter` 读的正是这些字段。见 `doc/ops/sandbox/Sandbox_Agent架构.md §3`。

| 路线 | 做法 | 优点 | 风险 |
|------|------|------|------|
| **D-1 verl 内置 agent loop** | `ToolAgentLoop` + `tool_registry` 注册沙箱 tool，动作代理到腾讯沙箱 | 轨迹/logprob/mask **原生免费**，与 trainer 同进程 | verl 单次批量 rollout **表达不了** 跨 query winner-sync（§下方冲突） |
| **D-2 自管调度 + 复用 verl 单步生成（推荐）** | `rollout/scheduler.py` 编排 16×8 + winner-sync，但**每步生成调用 verl rollout generate / agent_loop 单步**（原生 token+logprob），把每条 query 当 episode 收集 → `trajectory_adapter` → async runner | 既不造 proxy（收集仍走 verl 原生），又能做 winner-sync | 需把 verl 单步生成 API 包成可被外部循环调用 |
| D-3 OpenClaw + proxy（兜底） | OpenClaw 驱动 loop，vLLM 前架 recording proxy 抓 token+logprob | 调度最自由 | 需自抓 logprob、文本→token 对齐脆，**仅兜底** |

**关键冲突（务必记住）**：winner-sync 要求「会话内 query_2 前先用 query_1 reward 选 winner、同步 8 槽」，是 query 之间的耦合；verl「整批生成完再打分」表达不了。**调和 = D-2**：我们编排会话循环，单步生成复用 verl 原生（拿 token+logprob+mask），episode 之间做 winner-sync。

**决策默认值**：**D-2**。实现顺序：把 verl 单步生成包成可调接口 → `scheduler.py` 编排 16×8 + winner-sync 收集 episode → `trajectory_adapter.py`（已支持 `original_logprobs`/`success_rate`，补 reward/checker_results）→ `verl_async_runner.py` 接通。**收集机制一律用 verl 原生字段，不写 proxy（除非退到 D-3）。**

### 验收

- 端到端 dry-run：mock vLLM（固定回复）+ LocalSandbox，16×8 缩为 2×2，产出轨迹 → adapter → buffer 入桶 → replay 采样，全链路无 verl 真实训练也能跑
- 集群上：1 个 step 的真实采集 + 训练闭环（即 Progress.md 里 skip 的全栈 smoke）

---

## Gap E（P1）：eval/run_eval.py 收尾（40% → 100%）

- 缺：ClawEval manifest 加载（接口已定形 + fake 单测在 `tests/test_run_eval.py`）、真实 rollout 接入（复用 Gap C/D 的 pool 与 scheduler）
- 外部依赖：195 任务 manifest（@杨益博）——**没到货前用 fake manifest 把代码路径全部打通**，到货只换文件
- Pass³：三次独立运行全过才算过，已在 `eval/metrics.py` 约定，run_eval 里实现循环

---

## Gap F（P1）：集群配置补全

1. **Hydra 组合**：集群上以 verl `ppo_trainer` defaults 为底，叠加 `configs/base.yaml` + 实验 yaml（`merge_verl_config` 已实现）；需补显式字段：
   - `actor_rollout_ref.rollout.name`（vllm）、`tensor_model_parallel_size`、`gpu_memory_utilization`
   - fsdp/optim：`actor.optim.lr`、`fsdp_config`（27B/24 卡训练侧）
   - `trainer.n_gpus_per_node` / `nnodes`（40+24 拓扑）
2. **`ref.path` 填值**：Phase1/2/3 的 π₀ = 基底模型路径（27B 本地路径）；b1 无 KL 可不起 ref worker（`use_kl_loss: false` 时 verl 自动跳过，确认 `need_reference_policy` 行为即可）
3. **`gen_batch_size` 语义对齐**：调度层 16 会话/step 与 `train_batch_size: 1024` 的换算关系，在 `RolloutScheduler` 里显式写出（会话数 × 平均 query 数 × 8 ≈ batch 行数）

---

## Gap H（P0）：沙箱镜像 + agent harness —— 代码已就位，待 build 验证

架构见 `doc/ops/sandbox/Sandbox_Agent架构.md`（动作在沙箱内、推理在外、OpenClaw、随机用户文件系统）。

**已完成（本轮）**：

- `docker/sandbox/Dockerfile`：基础镜像 + Node24 + `openclaw@latest` + 动作/办公工具（git/jq/ripgrep/poppler/pandoc/fonts）+ COPY persona 种子 & 脚本；通过 `validate_sandbox_dockerfile.sh`（无 USER/WORKDIR/ENV/ENTRYPOINT）
- `docker/sandbox/requirements.txt`：扩 python-docx/pdfplumber/tabulate
- `docker/sandbox/bin/seed_workspace.sh`：实例启动确定性物化 persona（同 seed→字节一致，已冒烟）
- `docker/sandbox/bin/agent_entry.sh`：渲染 openclaw 配置（endpoint 来自 env）+ 跑一次 agent turn
- `docker/sandbox/openclaw.config.template.json`：headless 配置模板（占位 endpoint）
- `docker/sandbox/fs-seeds/`：3 个 persona 语料（finance/sysops/office）+ `manifest.json`
- runtime env 增 `OPENCLAW_MODEL` / `AGENTIC_CL_PERSONA` / `AGENTIC_CL_FS_SEED`

**待办（需有 docker + CCR 账号）**：

1. `bash scripts/build_sandbox_image.sh` 实际 build（验证 apt/NodeSource/npm 在该基础镜像可用；若基础镜像非 apt 需调整）
2. `openclaw doctor` 校准 `openclaw.config.template.json` 真实 schema（provider/baseUrl/model key 名）
3. headless 单轮：`agent_entry.sh "<query>"` 能无 daemon 跑通并产出可解析 trace（轨迹来源）
4. **logprob 通路**（与 Gap D 合流）：外部 vLLM OpenAI endpoint 开 `logprobs`，scheduler 落 per-token logprob 供 `trajectory_adapter` 回填
5. push 到 CCR → 建北京区 Tool（网络 `PUBLIC`）

---

## Gap G（P2）：杂项

| 项 | 动作 |
|----|------|
| dev 依赖缺失 | `.venv` 里没装 ruff/black/mypy → `uv pip install -e ".[dev]"` |
| 未提交变更 | 文档清理 + 调度指南 + 架构文档 + 镜像层文件待 commit 到 `dev_train` |
| 文档漂移 | `doc/source/CL_Design.md` 个别处可能残留 traj/query=2 旧数字；27B 显存/耗时估算待重算（C2） |

---

## 执行顺序与依赖图

```text
Gap B 数据转换 ──┐
                 ├──> Gap D-2 外部 rollout 桥 ──> 全栈 1-step smoke（64 卡）──> B1 正式训练
Gap A 规则 reward ┤            ▲
                 │             │
Gap H 沙箱镜像 ──> Gap C 沙箱调度 ┘   Gap F 集群配置
（H build 验证后，C 的 PoC 可与 A/B 并行）

Gap E eval、Gap G 杂项：与主线并行，不阻塞 B1
```

**建议里程碑**：

1. A + B（纯本地，可单测验收，无外部依赖）
2. C 的 PoC（2 槽 2 query）→ C 完整实现
3. D-2 dry-run（mock 全链路）
4. F 集群配置 + 64 卡 1-step smoke
5. B1 启动；E/G 穿插

---

## 验收总清单（全部勾完才允许启动 B1）

- [ ] 沙箱镜像 build 通过 + push CCR + 北京区 Tool 建好；`agent_entry.sh` headless 单轮跑通（Gap H）
- [ ] `trainer/rule_reward.py` 单测过，`reward.custom_reward_function` 配置生效
- [ ] `datasets/train.parquet` / `val.parquet` 产出，桶分布与 checker 覆盖率已打印归档
- [ ] `SessionSandboxPool` mock 单测过；真实后端 2×2 冒烟过；sync 一致性 PoC 过
- [ ] D-2 dry-run：轨迹 → adapter → buffer → replay 采样全链路（无 GPU）
- [ ] 集群 Hydra 组合无 `???` 残留（Phase 4/5 的待定参数除外）
- [ ] 64 卡 1–2 step 全栈 smoke（`scripts/train.sh configs/phase1/b1.yaml`）
- [ ] `.venv/bin/python -m pytest -q` 全量通过（约 200 测试函数）
