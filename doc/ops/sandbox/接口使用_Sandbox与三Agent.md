# 接口使用指南：Sandbox 后端 + 三 Agent（报告与声明）

> **定位**：给后续接手的人 / AI agent 的**接口使用速查**——每个接口在哪、签名、怎么调、谁产出谁消费。设计动机见 [`UserSim_三Agent架构与技术设计.md`](../../archive/UserSim_三Agent架构与技术设计.md)（接口契约）与 [`SandboxRollout.md`](../../archive/SandboxRollout.md)（平台 API）。本文只讲**怎么用**。
> **状态**：沙箱接口/实现已解耦；三 Agent 代码已落盘 `agents/`。observer 已落地 **diff-driven、模型只看 state**；reward **双通道**（state 判 completion + pass-through 轨迹判 safety/robustness）。完整设计与"diff 能否到内容"的论证见 [`../paper/refs/Observer_DiffDriven_技术报告.md`](../../../paper/refs/Observer_DiffDriven_技术报告.md)。
> **写作日期**：2026-06-19（2026-06-22 校订）

---

## 0. 速查表（每个接口一行）

| 接口 | 位置 | 签名 / 用法 | 谁产出 → 谁消费 |
|------|------|-------------|------------------|
| `SandboxClient`（Protocol，**接口契约**） | `rollout/sandbox_client.py` | `run_code(code, language="python") -> ExecResult` / `kill() -> None` | rollout loop 只依赖它 |
| `ExecResult`（返回契约） | 同上 | `.stdout / .stderr / .ok` | `run_code` 产出 |
| `make_sandbox`（按名选后端） | 同上 | `make_sandbox(backend="local"\|"e2b"\|"aliyun", **kw) -> SandboxClient` | 调度/采集脚本调用 |
| `register_backend`（开放扩展） | 同上 | `register_backend(name, builder)`；`builder(**kw)->SandboxClient` | 加新厂商时调用 |
| `Observer.observe`（产**报告**） | `agents/observer.py` | `observe(sandbox=None, *, actor_trajectory="", baseline=None, post=None) -> ObservationReport` | 产出 `ObservationReport` |
| `ObservationReport`（**报告**） | `agents/schema.py` | `.state_diff`(ground truth)`/.final/.intermediate/.discrepancies/.actor_trajectory`(pass-through)`/.has_effect`、`.is_empty()` | observer 产 → questioner+reward 消费 |
| `Questioner.next_query`（消费报告） | `agents/questioner.py` | `next_query(persona, report, session_history) -> str\|None` | 读报告 → 出下一条 query |
| `RotatingChatClient`（多模型轮换） | `agents/base.py` | `RotatingChatClient(clients, rotate_every=5)` — 每 N 次调用切换模型 | Questioner 抗模式坍缩 |
| `PatienceTracker`（失败路径） | 同上 | `PatienceTracker(persona, rng).on_failure() -> bool` | 失败轮决定 redo / 结束 |
| `score_followup`（Reward，消费报告） | `agents/reward.py` | `score_followup(*, query, report, judge=None) -> dict`（trajectory 从 `report.actor_trajectory` 取） | 读报告 → 打分 |
| `sample_persona` / `PERSONAS` | `agents/personas.py` | `sample_persona(rng) -> Persona` | 会话级抽 1 个人设 |

---

## 1. Sandbox 层接口（接口与实现已解耦）

`rollout/sandbox_client.py` 显式分三段：**INTERFACE / IMPLEMENTATIONS / REGISTRY**。rollout loop 只认接口 + 按名取后端，换厂商不碰调用方。

### 1.1 接口契约 `SandboxClient` + `ExecResult`

任何后端只要提供这两个方法即可被无缝替换（结构化类型，无需继承）：

```python
class SandboxClient(Protocol):
    def run_code(self, code: str, language: str = "python") -> ExecResult: ...
    def kill(self) -> None: ...

# 返回契约：普通用户代码失败 -> ok=False，错误进 stderr，不要抛异常
@dataclass
class ExecResult:
    stdout: str
    stderr: str
    ok: bool
```

- `run_code`：在**存活实例**里执行代码，返回 `ExecResult`。
- `kill`：释放实例；尽力而为，**teardown 不得抛异常**。

### 1.2 选择后端：`make_sandbox`（按名）

```python
from rollout.sandbox_client import make_sandbox

sb = make_sandbox("local")          # dev/CI
sb = make_sandbox("e2b", template="agentic-cl-code-interpreter", timeout=300)  # 腾讯
res = sb.run_code("print(6*7)")     # -> ExecResult(stdout='42', stderr='', ok=True)
sb.kill()
```

未知名字 → `ValueError`，并列出已注册后端（`['aliyun','e2b','local']`）。

### 1.3 扩展后端：`register_backend`（加厂商 = 一次调用）

```python
from rollout.sandbox_client import register_backend, ExecResult

class MyVendorSandbox:
    def run_code(self, code, language="python") -> ExecResult: ...
    def kill(self) -> None: ...

register_backend("myvendor", lambda **kw: MyVendorSandbox(**kw))
# 之后 make_sandbox("myvendor") 即可，rollout loop 零改动
```

### 1.4 现有后端

| backend | 类 | 状态 | 凭证 / 依赖 |
|---------|----|------|-------------|
| `local` | `LocalSandbox` | ✅ 真（dev/CI） | 无（本地子进程；**持久 workdir：state 跨 `run_code` 不丢，可做 observer diff**，2026-06-19 改） |
| `e2b` | `E2BSandbox` | ✅ 真（集群=腾讯 Agent Runtime） | `E2B_API_KEY` / `E2B_DOMAIN`；裸 REST via httpx |
| `aliyun` | `AliyunSandbox` | 🚧 **stub 留空** | 接口+注册点就绪，body 待回集群用 `wuying-agentbay-sdk` + `AGENTBAY_API_KEY` 填 |

> ✅ `LocalSandbox` 已改持久 workdir（2026-06-19）：observer 的 before/after diff 与 agent 多步状态本机即可验证（见 `CLAUDE.md` TODO#5）。

### 1.5 env / 凭证

```bash
# 腾讯 e2b
export E2B_API_KEY=...   E2B_DOMAIN=ap-beijing.tencentags.com
# 阿里 aliyun（实现后）
export AGENTBAY_API_KEY=...   # pip install wuying-agentbay-sdk
```

---

## 2. 三 Agent 层接口（**报告**）

> **用哪个模型（observer/questioner/judge 的选型）见 [`模型选型.md`](../../archive/模型选型.md)**（单一信源）。本节只讲接口与 env 变量名，不记具体模型。

一句话：**Observer 产出一份 `ObservationReport`（报告），Questioner 和 Reward 各读这同一份报告**。报告核心是 `state_diff`（环境 diff = ground truth）；另有 `actor_trajectory`（actor 轨迹文本，**pass-through**：observer 组件捎带、observer 模型不看、只给 reward）。

### 2.1 `ObservationReport`（**报告**，`agents/schema.py`）

| 字段 | 含义 | 谁用 |
|------|------|------|
| `state_diff: str` | **环境 before/after diff（含内容 + SysOps）= ground truth** | reward 判 completion、questioner 看产出 |
| `final: list[dict]` | 最终交付 `{path, kind, content_excerpt}`（从 diff 填） | reward / questioner |
| `intermediate: list[dict]` | 中间结果 `{desc, source, value_excerpt}`（LLM 填 或 启发式检测） | reward / questioner |
| `discrepancies: str` | 状态内部红旗（空/损坏/自相矛盾；LLM 填 或 启发式检测） | reward 参考 |
| `actor_trajectory: str` | **pass-through**：actor 轨迹文本，**observer 模型不看**，只给 reward 判 safety/robustness | reward |
| `file_tree: str` | 工作区文件树（fallback 证据） | 兜底 |
| `has_effect: bool` | 本轮 diff 是否非空；False → 短路 reward、走失败/耐心路径 | reward gate / driver |
| `is_empty()` | `not has_effect` 或无 final/intermediate → 失败/耐心路径 | session driver |

### 2.2 `actor_trajectory`（**pass-through**）来源与用途

来源 = driver 把 winner 轨迹消息传给 `observe(actor_trajectory=...)`，observer 组件用 `flatten_trajectory` 拍平存进报告。用途 = **只给 reward** 判 safety/robustness。**关键**：observer 模型（LLM）**永远看不到它**（不进 `build_observer_prompt`，不浪费 token）；observer 自身的取证只基于 `state_diff`。结构性反 hacking：声称根本不进 observer 的判断与 completion。

### 2.3 `Observer.observe`（产报告，模型只看 STATE）

```python
from agents.observer import Observer
report = Observer().observe(
    sandbox, actor_trajectory=winner.messages, baseline=pre_snap, post=post_snap
)
# observer 模型只看 state_diff；actor_trajectory 仅 pass-through 给 reward
```

- 容错：LLM/沙箱失败 → 降级为最小报告（state_diff + tree），**不 crash 会话**。
- `use_llm=True`（默认）→ LLM 多轮 tool-use：先看 diff，可调探针工具（read_file/list_dir 等）深入调查，再输出结构化报告；`use_llm=False` → 确定性建报告、零模型调用（降级模式）。两种路径都不看 trajectory。
- env：`OBSERVER_API_BASE` / `OBSERVER_MODEL` / `OBSERVER_API_KEY`（temperature 0，客观）。

### 2.4 `Questioner.next_query` + `PatienceTracker`（消费报告，人设侧）

```python
from agents.questioner import Questioner, PatienceTracker
q = Questioner().next_query(persona, report, pool.session_history)  # str 或 None(=<end_session>)
pt = PatienceTracker(persona, rng); redo = pt.on_failure()           # 失败轮 True=重做 False=结束
```

- persona 的 `observation_focus` 决定**强调报告哪一面**（整体/细节 × 形式/内容）；observer 仍客观。
- persona 的 `tone`（calm/neutral/hot）注入 Questioner system prompt，控制语气风格（2026-06-22 新增）。
- env（单模型）：`USERSIM_API_BASE` / `USERSIM_MODEL` / `USERSIM_API_KEY`（temperature 0.9，抗坍缩）。
- env（多模型轮换，2026-06-22 新增）：
  - `USERSIM_ENDPOINTS`：JSON 数组，每个元素是 `{"base_url": "...", "model": "...", "api_key": "..."}`，`api_key` 可省略（默认 `"sk-local"`）。
  - `USERSIM_ROTATE_EVERY`：每 N 次调用后切换到下一个模型（默认 5）。
  - 设置 `USERSIM_ENDPOINTS` 后，`resolve_questioner_client()` 返回 `RotatingChatClient`（多模型轮换），替代单一 `OpenAIChatClient`；不设置时行为完全不变。
  - 轮换池中的模型应来自**不同厂商/不同系列**（如 claude-sonnet + gpt-4.1 + deepseek-v3），最大化输出风格多样性，防止 follow-up query 的模式坍缩。

### 2.5 `score_followup`（Reward，打分侧，**双通道**）

```python
from agents.reward import score_followup
verdict = score_followup(query=q, report=report, judge=None)  # trajectory 从 report 取
# -> {score, completion, safety, robustness, judge_error[, gated]}；尺度与 ClawEval 同构
```

reward judge 收**两条通道，都来自同一份 $R_t$**：
- **状态通道**：`report.state_diff`（observer 的环境 diff = ground truth）→ 进 rubric，判 **completion**（真实产出）。
- **轨迹通道**：`report.actor_trajectory`（actor 动作文本）→ observer **组件** pass-through 捎带（observer **模型**没看过），判 **safety/robustness**（怎么做的）。

要点：
- **observer 模型不看轨迹**（§2.2/§2.3），但报告把轨迹捎给 reward——completion 只能反映真实 diff（结构性反 hacking：声称根本不进 completion）；轨迹仍供 reward 看"过程"。
- **拦截层（gate）**：`report.has_effect=False`（空 diff，本轮无变化）→ **直接返回 score 0、不调 judge**（带 `gated=1.0`）。
- diff 与 trajectory 各自封顶（`_MAX_DIFF_CHARS` / `_MAX_TRAJ_CHARS`，中间截断）防超长。
- env：`JUDGE_API_BASE` / `JUDGE_MODEL` / `JUDGE_API_KEY`（与 verl `custom_reward_function` 共用）。

### 2.6 数据流

```text
winner.messages ─┐  (轨迹仅 pass-through，observer 模型不看)
                 ▼
sandbox(winner) ─► Observer.observe(sandbox, actor_trajectory=, baseline=, post=) ─► R_t
                       │  R_t = state_diff(事实) + final/intermediate + actor_trajectory(pass-through) + has_effect
                       ├─► Questioner.next_query(persona, R_t, history) ─► 下一条 query
                       └─► score_followup(query, R_t) ─► reward（state→completion, trajectory→safety/robustness）
```

> ⚠️ 三套 env 端点**故意分开**（OBSERVER/USERSIM/JUDGE），抗 self-preference。若三者指向同一模型，保护失效。

---

## 3. 观察证据采集：diff-driven（现行实现）

> 设计动机（为什么从 claim-driven 改 diff-driven 的 6 条理由）、"沙箱 diff 能否到内容"的分层论证，见 [`../paper/refs/Observer_DiffDriven_技术报告.md`](../../../paper/refs/Observer_DiffDriven_技术报告.md)。本节只列**现行机制**与**落地状态**，不重复论证。

### 3.1 机制（确定性取证，observer 模型只看 state）

```text
turn 前  baseline = observer.snapshot(sandbox)   # {fs,sys} 只读快照（仅用 run_code，后端无关）
  ↓ actor turn（winner 在沙箱里做事）
turn 后  post = observer.snapshot(sandbox)
  ↓ diff_snapshots(baseline.fs, post.fs)   → added/modified/removed（按 size+mtime；带内容摘要）
  ↓ extract_binaries(...)                  → 仅本轮变更的 xlsx/docx/pptx/pdf 在沙箱内抽成文本（缺库降级不崩）
  ↓ diff_system(baseline.sys, post.sys)    → 本轮装的包 / 开的端口(LISTEN) / 起的进程
  ↓ state_diff = render(fs diff + sys diff)            # = ground truth
  ↓ use_llm ? 模型只看 state_diff 归纳 : 确定性建报告（默认零模型调用）
  ↓ report.actor_trajectory = flatten(winner_messages)  # pass-through，绝不进 observer prompt
```

- 取证只用 `sandbox.run_code`（后端无关：local/e2b/aliyun 通用），**不绑**某厂商的 `files.*` API。
- 文本类 diff 直接带内容；二进制走**格式提取→文本**；非 FS 效果走 **SysOps 命令探针**。
- `LocalSandbox` 已改持久 workdir（state 跨 `run_code` 不丢），本机即可验证 diff 逻辑。

### 3.2 几层拦截（gate，不需要的就不调用）

| 层 | 调用 | 触发条件（否则跳过） |
|----|------|----------------------|
| L0 | 快照 / diff | 始终（便宜、确定性） |
| L1 | 二进制内容提取 | 本轮**有变更的**富二进制文件（xlsx/docx/pptx/pdf）|
| L2 | observer LLM | `use_llm=True` **且** diff 非空 |
| L3 | reward judge | `report.has_effect=True`（diff 非空）**且** turn>1；空 diff → score 0、不调 judge |
| L4 | questioner LLM | `turn<=k` **且** 未判失败 |
> `has_effect=False` 同时让 `is_empty()` 为真 → 走失败/耐心路径。

### 3.3 落地状态（2026-06-19）

**已实现（本机单测 + 验证）**：持久 workdir、diff 采集（fs + 内容 + 二进制提取 + SysOps）、observer LLM 可选（默认确定性零调用）、性能 Tier1（空 diff 跳过 / 单快照 / (size,mtime) / prompt 瘦身）、reward 双通道 + 空 diff 短路。

**剩余（待集群）**：二进制提取真值（需 openpyxl/pptx 等库 + 真实文件，本机仅验 fallback）；`watch_dir` 抓瞬态中间产物（Tier2，按调查降级未做）；真实 `e2b`/`aliyun` 连通 + 8 槽 winner-sync 下 FS/SYS baseline 正确性（local mock 仅近似）。详见 `CLAUDE.md` TODO#5。

---

## 4. 给 Agent 的使用约定（do / don't）

- **改沙箱后端**：只加类 + `register_backend`，**别动** `session_pool` / `scheduler` / 采集脚本。
- **加新厂商**：照 `AliyunSandbox` stub 的形状实现 `run_code`/`kill`，env 用自己的前缀。
- **用报告**：questioner/reward 一律以 `ObservationReport.state_diff` 为事实基准；`actor_trajectory` 是 reward-only pass-through，**observer 模型不看**、questioner 也不看。
- **三套 env 端点必须分开**，否则 self-preference 保护失效。
- **写完跑** `pytest tests/test_sandbox_client.py tests/test_agents.py` + `ruff check`（集群环境）。
