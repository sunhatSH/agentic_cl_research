# 接口使用指南：Sandbox 后端 + 三 Agent（报告与声明）

> **定位**：给后续接手的人 / AI agent 的**接口使用速查**——每个接口在哪、签名、怎么调、谁产出谁消费。设计动机见 [`UserSim_三Agent架构与技术设计.md`](UserSim_三Agent架构与技术设计.md)（接口契约）与 [`SandboxRollout.md`](SandboxRollout.md)（平台 API）。本文只讲**怎么用**。
> **状态**：沙箱接口/实现已解耦（2026-06-19）；三 Agent 代码已落盘 `agents/`。observer 的"证据采集"现状=claim-driven，目标=diff-driven（§3，回集群施工）。
> **写作日期**：2026-06-19

---

## 0. 速查表（每个接口一行）

| 接口 | 位置 | 签名 / 用法 | 谁产出 → 谁消费 |
|------|------|-------------|------------------|
| `SandboxClient`（Protocol，**接口契约**） | `rollout/sandbox_client.py` | `run_code(code, language="python") -> ExecResult` / `kill() -> None` | rollout loop 只依赖它 |
| `ExecResult`（返回契约） | 同上 | `.stdout / .stderr / .ok` | `run_code` 产出 |
| `make_sandbox`（按名选后端） | 同上 | `make_sandbox(backend="local"\|"e2b"\|"aliyun", **kw) -> SandboxClient` | 调度/采集脚本调用 |
| `register_backend`（开放扩展） | 同上 | `register_backend(name, builder)`；`builder(**kw)->SandboxClient` | 加新厂商时调用 |
| `Observer.observe`（产**报告**） | `agents/observer.py` | `observe(actor_trajectory, sandbox=None) -> ObservationReport` | 产出 `ObservationReport` |
| `ObservationReport`（**报告**） | `agents/schema.py` | `.intermediate/.final/.actor_claims/.discrepancies/.file_tree`、`.is_empty()` | observer 产 → questioner+reward 消费 |
| `Questioner.next_query`（消费报告） | `agents/questioner.py` | `next_query(persona, report, session_history) -> str\|None` | 读报告 → 出下一条 query |
| `PatienceTracker`（失败路径） | 同上 | `PatienceTracker(persona, rng).on_failure() -> bool` | 失败轮决定 redo / 结束 |
| `score_followup`（Reward，消费报告） | `agents/reward.py` | `score_followup(*, query, report, trajectory, judge=None) -> dict` | 读报告 → 打分 |
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

## 2. 三 Agent 层接口（**报告**与**声明**）

一句话：**Observer 产出一份 `ObservationReport`（报告），Questioner 和 Reward 各读这同一份报告**（一份两用）。报告里 `actor_claims` 是 actor 的**声明**（它说自己做了什么），`discrepancies` 是"声明 vs 实际"的差异（反 reward-hacking 的命门）。

### 2.1 `ObservationReport`（**报告**，`agents/schema.py`）

| 字段 | 含义 | 谁用 |
|------|------|------|
| `intermediate: list[dict]` | 中间结果 `{desc, source, value_excerpt}` | reward 核对、questioner 追问 |
| `final: list[dict]` | 最终交付 `{path, kind, content_excerpt}` | reward 判 completion、questioner 反应 |
| `actor_claims: str` | **声明**：actor 轨迹文本（它说自己做了什么） | 与实际对照 |
| `discrepancies: str` | 声明 vs 实际差异（反 hacking 证据） | **reward 据此扣分** |
| `file_tree: str` | 工作区文件树（fallback 证据） | 兜底 |
| `is_empty()` | 无可用证据 → 触发失败/耐心路径 | session driver |

### 2.2 `actor_claims`（**声明**）来源与用途

来源 = `Observer` 把 winner 轨迹消息拍平（`_trajectory_text`）。用途 = 给 reward/questioner 当"actor 自述"，并与沙箱实际比对填 `discrepancies`。**注意：声明 ≠ 事实**——这正是要独立 observer 取证的原因（线上 self-observation 无害，训练里 actor 自观察会被 reward-hack）。

### 2.3 `Observer.observe`（产报告）

```python
from agents.observer import Observer
report = Observer().observe(winner.messages, sandbox=winner_client)  # sandbox 可空 -> 仅 claims
```

- 容错：LLM/沙箱失败 → 降级为最小报告（仅 claims + tree），**不 crash 会话**。
- env：`OBSERVER_API_BASE` / `OBSERVER_MODEL` / `OBSERVER_API_KEY`（temperature 0，客观）。

### 2.4 `Questioner.next_query` + `PatienceTracker`（消费报告，人设侧）

```python
from agents.questioner import Questioner, PatienceTracker
q = Questioner().next_query(persona, report, pool.session_history)  # str 或 None(=<end_session>)
pt = PatienceTracker(persona, rng); redo = pt.on_failure()           # 失败轮 True=重做 False=结束
```

- persona 的 `observation_focus` 决定**强调报告哪一面**（整体/细节 × 形式/内容）；observer 仍客观。
- env：`USERSIM_API_BASE` / `USERSIM_MODEL` / `USERSIM_API_KEY`（temperature 0.9，抗坍缩）。

### 2.5 `score_followup`（Reward，消费报告，打分侧）

```python
from agents.reward import score_followup
verdict = score_followup(query=q, report=report, trajectory=text, judge=None)
# -> {score, completion, safety, robustness, judge_error}；尺度与 ClawEval 同构
```

- 以 `report` 为**事实基准**（非 actor 声称）；`discrepancies` 进 rubric 当反 hacking 信号。
- env：`JUDGE_API_BASE` / `JUDGE_MODEL` / `JUDGE_API_KEY`（与 verl `custom_reward_function` 共用）。

### 2.6 数据流

```text
winner.messages ─┐
                 ├─► Observer.observe(.., sandbox=winner) ─► ObservationReport R_t ─┬─► Questioner.next_query(persona, R_t, history) ─► 下一条 query
sandbox(winner) ─┘                                                                  └─► score_followup(query, R_t, traj) ─► reward
```

> ⚠️ 三套 env 端点**故意分开**（OBSERVER/USERSIM/JUDGE），抗 self-preference。若三者指向同一模型，保护失效。

---

## 3. 观察证据采集：claim-driven → diff-driven（**2026-06-19 已实现**）

> 状态：observer 已落地 diff-driven（`agents/observer.py`：`snapshot()` + `diff_snapshots()` + `observe(.., baseline=)`），`LocalSandbox` 持久 workdir，本机已验证 diff 能读到内容、能暴露"声称 12345 vs 实际 99999"。§3.1 = 被取代的旧方案（留作对照），§3.2/§3.3 = 现行机制，§3.4 = 落地状态与剩余。

### 3.1 旧方案（claim-driven，单次 LLM）的局限 —— 已弃用

`observe()` 是**一次** LLM 调用：把 `actor_claims`（声明文本）+ 一段**固定** `os.walk` 文件树塞进提示，让模型"自己悟该看什么"。系统提示让模型"以声称为线索去读 report.xlsx / 找中间值"，**但代码没给模型任何去取这些信息的能力**：文件树是盲扫（非声称驱动）、**从不读文件内容**、`tool_outputs` 槽位从不填。→ 能判断"该看什么"，不能"真的去看"，"声称值≠实际值"结构上判不出。

### 3.2 现行机制：diff-driven（沙箱状态增量当事实）

把"actor 说了什么"降级为**次要交叉校验输入**，把"沙箱真实改了什么"升为**事实主轴**：

```text
turn 前：snapshot_pre = files.list(workspace, depth=N) + 内容哈希   （或起 watch_dir）
  ↓ 跑 actor turn
turn 后：snapshot_post = files.list(...) + 内容哈希
  ↓ 代码确定性算 diff：
   added    = post - pre
   removed  = pre - post
   modified = 两边都有但 hash/size/mtime 变了
  ↓ 对 added/modified（按声称相关性 + 体积上限过滤）：files.read 拿内容 → 算 difflib 内容差
  ↓ 把结构 diff + 内容 diff 作为 tool_outputs 喂给 observer 模型（模型只负责归纳/核对 discrepancies）
```

两种粒度，可叠加：

- **快照轮询 diff**（before/after `files.list` + `files.read`）：简单稳，但只看**净变化**，会漏 turn 内"建了又被覆盖"的中间产物。
- **事件流 `watch_dir`**（Create/Write/Remove/Rename 实时事件）：能抓**瞬态/被覆盖的中间产物**——正好对上设计里"intermediate 易被后续步骤覆盖"的诉求。

### 3.3 沙箱 diff 能 diff 到**内容**吗？

**能，但要分清三层**（这是问题的核心答案）：

| 想要的 diff | 用什么 API | 能到什么粒度 |
|-------------|-----------|--------------|
| 文件**数量/名称/结构** | `files.list(path, depth)` 前后做集合差 | ✅ 路径增删、大小/类型；**只到名字与元数据，不含内容** |
| 文件**内容** | `files.read(path)` 读取前后两版 → 自己用 `difflib` 算 unified diff | ✅ **能到内容**（行级文本 diff）；需要你把内容拉下来自己算，沙箱不白送 git 式 diff |
| 一把梭的内容 diff | 会话开始 `git init`，每轮后在沙箱里 `git diff` / `git status --porcelain`（via `commands.run`/`execute_command`） | ✅ 最省事的"内容 diff"：一条命令拿到所有跟踪文件的文本差 |
| **二进制**（xlsx/png/db） | `files.read` 取字节 + 格式化解析（openpyxl 读单元格等），或先 `md5` 判变更 | ⚠️ **纯文本 diff 无意义**，必须格式感知解析后再比 |

结论：
- `files.list` 这类**只能 diff 到"文件数量和名称"（+大小/类型元数据）**，给不出内容。
- **要 diff 到内容**，必须 `files.read` 把两版内容读下来自己算（`difflib`），或在沙箱里跑 `git diff`/`diff`。
- E2B（腾讯）有 `files.list`/`files.read`/`watch_dir`/`commands.run`；阿里 AgentBay 有 `session.file_system`/`session.command.execute_command`——两边都够做内容级 diff。

### 3.4 落地状态（2026-06-19）

**已实现（本机验证通过）：**
1. ✅ `LocalSandbox` 持久 workdir —— state 跨 `run_code` 不丢。
2. ✅ `simulated_session` / `usersim_collect`：turn 前 `observer.snapshot(sandbox)` 取 baseline、传 winner 沙箱给 `observe`。
3. ✅ `Observer` diff 采集：只读快照探针（仅用 `run_code`，后端无关）→ `diff_snapshots(pre, post)` 出 added/modified/removed + 内容摘要 → 进 `OBSERVER_SYSTEM` 当 ground truth；`ObservationReport.state_diff` 携带确定性证据。
4. ✅ 提示词：diff = 事实，actor 声称仅交叉核对，`discrepancies` 收"声称不被 diff 支持"的项。

**已实现（2026-06-19 第二批）：**
5. ✅ **性能 Tier1**：空 diff 跳过 observer LLM；每轮 1 次快照（上轮 post 前传作 baseline）；变更检测改 `(size, mtime)` 去掉整文件 sha1；prompt 去冗余 file tree + 封顶。
6. ✅ **(a) 二进制内容提取适配器**：快照对二进制只标记，diff 后**只对本轮变更的** xlsx/xlsm/docx/pptx/pdf 跑沙箱内提取（openpyxl/python-docx/python-pptx/pdfplumber）→ 文本进 diff（`kind=binary→text`）；缺库/解析失败优雅降级为标记，不崩。
7. ✅ **(b) 非 FS 状态命令探针（SysOps）**：`snapshot_system` 取 `{pip, ports(LISTEN), procs}`，`diff_system` 出"本轮装的包 / 开的端口 / 起的进程"；不采集 env 值（防泄密）。
8. ✅ **observer LLM 可选**：`Observer(use_llm=False)` 默认**确定性建报告、零模型调用**（取证证据已是文本）；`use_llm=True` 才调模型做归纳/discrepancy。确定性取证层（快照+diff+提取+sysops）**始终运行**——这正是避免"把原始证据塞给 reward 模型去观察"的高消耗。

**剩余（待集群 / 后续）：**
- 二进制提取真值验证需 openpyxl/pptx 等库 + 真实文件（本机无库，已验证 fallback 不崩）。
- 瞬态/被覆盖的中间产物：当前 before/after 快照只看**净变化**，未上 `watch_dir` 事件流（Tier2 #6，按调查降级）。
- 真实后端（`e2b`/`aliyun`）连通 + 8 槽 winner-sync 下 FS/SYS baseline 正确性（local mock 仅近似）待集群验证。
- discrepancy（声称 vs 实际）在 `use_llm=False` 下留给 reward judge（其 R_t 已含 `state_diff`），不在 observer 重复一遍模型判断。

> 用 §1 的 `e2b`（真后端能读内容）或实现后的 `aliyun` 即可上真实沙箱；本机用 `local`（持久 workdir）已能验证 diff 逻辑。详见 `CLAUDE.md` TODO#5。

---

## 4. 给 Agent 的使用约定（do / don't）

- **改沙箱后端**：只加类 + `register_backend`，**别动** `session_pool` / `scheduler` / 采集脚本。
- **加新厂商**：照 `AliyunSandbox` stub 的形状实现 `run_code`/`kill`，env 用自己的前缀。
- **用报告**：questioner/reward 一律以 `ObservationReport` 为准，**不要**直接信 `actor_claims`（那是"声明"不是事实）。
- **三套 env 端点必须分开**，否则 self-preference 保护失效。
- **写完跑** `pytest tests/test_sandbox_client.py tests/test_agents.py` + `ruff check`（集群环境）。
