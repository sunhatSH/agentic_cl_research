# Observer 取证：claim-driven → diff-driven 技术报告

> **范围**：记录 observer（观察 agent）的取证机制从"actor 声称驱动"改为"沙箱 before/after diff 驱动"的**问题定位、设计、实现与验证**（@孙豪，2026-06-19，本机无 GPU/E2B）。
> **配套**：接口使用见 [`doc/接口使用_Sandbox与三Agent.md`](../../doc/接口使用_Sandbox与三Agent.md) §3；三 agent 设计见 [`doc/UserSim_三Agent架构与技术设计.md`](../../doc/UserSim_三Agent架构与技术设计.md)；状态/TODO 见 `CLAUDE.md` TODO#5；运行记录见 `doc/RunLog.md`（2026-06-19）；论文表述见 `paper/drafts/Paper_Method_draft_*.md` §4.5 + 附录 A.1。
> **代码锚点**：`agents/observer.py`、`agents/prompts.py`、`agents/schema.py`、`rollout/sandbox_client.py`、`rollout/{simulated_session,usersim_collect}.py`、`scripts/agents_harness.py`。

---

## 0. TL;DR

observer 的职责是给 reward 与 questioner 提供一份"实际发生了什么"的客观证据 $R_t$。旧实现让 **actor 的声称**驱动观察（单次 LLM 把"声称文本 + 盲扫文件树"塞进一个提示），用 runtime 证据确认其**结构上看不到真实状态**，退化为"actor 自述复读机"——反 reward-hacking 落空。本次改为 **diff-driven**：以**沙箱 before/after 内容级 diff** 为 ground truth，actor 声称仅作交叉核对。配套把 `LocalSandbox` 改持久 workdir，使本机即可验证。

---

## 1. 背景与问题（runtime 证据）

旧 `observe()` 只做两件事并发一次 LLM 调用：
1. `_trajectory_text(winner.messages)` —— 把 actor 声称拍平成文本；
2. `_probe_file_tree(sandbox)` —— 固定 `os.walk('.')` 列**文件路径**（不读内容）。

用真实 `agents/observer.py` 打桩跑真实路径，确认三处缺陷：

| 假设 | runtime 证据 | 结论 |
|------|--------------|------|
| H1 主训练路径没传沙箱 | `simulated_session` 调 `observe(winner.messages)`，`sandbox_is_none=true` | 主路径 observer 全盲 |
| H2 LocalSandbox 无持久 FS | actor 写文件 `ok=True` 后，探针 `tree_len=0` | 每次 `run_code` 新临时目录，state 不持久 |
| H3 只列路径不读内容 | `file_tree_empty=true`、`tool_outputs` 从不传、唯一证据=声称文本 | 判不了"声称值≠实际值" |

后果：reward 的"按真实效果给分（反 hacking）"沦为"按声称给分"——正是独立 observer 设计想杜绝的攻击面。

---

## 2. 为什么从 claim-driven 改为 diff-driven

| # | 理由 | 说明 |
|---|------|------|
| 1 | **模型会幻觉** | actor 叙述可凭空声称文件/数值；只有沙箱 diff 是事实。 |
| 2 | **声称丢中间产物** | claims 通常只报最终结果，忽略中间产物——而设计最看重"易被后续步骤覆盖的中间态"；diff 不管 actor 提没提都能抓到。 |
| 3 | **反 reward-hacking** | reward 落在声称上 → 策略学到"嘴上说做完、实际没做"；落在真实 diff 上，此攻击面消失（这正是 observer 要独立的原因）。 |
| 4 | **遗漏而非仅造假** | actor 没提到的错误/缺失文件，claim 读者看不见，diff 一定暴露（静默/部分失败）。 |
| 5 | **内容级核对** | "Q3 total = 12345" 能与文件实际字节比对（实际 99999）；claim 读者拿不到内容。 |
| 6 | **确定性** | diff 由代码算（可复现），模型只做归纳/核对，不在证据里再叠一层模型误差。 |

---

## 3. 设计

```text
turn 前：baseline = observer.snapshot(sandbox)         # 只读快照（仅用 run_code，后端无关）
  ↓ 跑 actor turn（winner 在沙箱里做事）
turn 后：post = snapshot(sandbox)
  ↓ diff_snapshots(baseline, post) → added / modified / removed（+ 内容摘要 + before 摘要）
  ↓ _format_changes(diff) → state_diff（ground-truth 证据块）
  ↓ build_observer_prompt(state_diff=…, actor_trajectory=…) → LLM 归纳/核对
  ↓ ObservationReport（intermediate/final/discrepancies + state_diff 携带确定性证据）
```

- **快照探针**：在沙箱里跑一段只读 Python（`os.walk` + 全量 sha1 + 截断文本摘要），打印 `{path: {size, sha, text?}}` 的 JSON。**只依赖 `run_code`**，故 local / e2b / aliyun 后端通用，不绑某厂商的 `files.read`。
- **diff**：纯 Python 计算 added/modified/removed；modified 同时带新旧内容摘要，可量化"改了什么"。
- **无 baseline 兜底**：未给 baseline 时退化为"当前快照内容"（仍是内容级，比旧的盲扫文件树强）；无沙箱时退化为 claims-only。
- **提示词反转**：`OBSERVER_SYSTEM` 明确"**diff = ground truth，authoritative；actor 声称仅 cross-check，不可凌驾于 diff**"，`discrepancies` 收"声称不被 diff 支持"的项，并要求记录"actor 没提但 diff 显示"的真实改动。

---

## 4. 实现

| 文件 | 改动 |
|------|------|
| `agents/observer.py` | 新增 `_SNAPSHOT_PROBE` 只读探针、`snapshot_workspace`/`diff_snapshots`/`_format_changes`/`_format_state`；`Observer.snapshot()`；`observe(traj, sandbox=None, *, baseline=None)` 改为 diff-driven；`parse_observation_report(..., fallback_diff=)` 携带 `state_diff` |
| `agents/prompts.py` | `OBSERVER_SYSTEM` 改 diff-driven；`build_observer_prompt(*, actor_trajectory, state_diff="", file_tree="", tool_outputs="")` 把 diff 置于最前作权威证据 |
| `agents/schema.py` | `ObservationReport` 加 `state_diff: str`（确定性证据，下游 reward 可用） |
| `rollout/sandbox_client.py` | `LocalSandbox` 改**持久 workdir**（`mkdtemp` once，`kill` 时 `rmtree`）——state 跨 `run_code` 不丢 |
| `rollout/simulated_session.py` | turn 前 `observer.snapshot(slot0)` 取 baseline；pick_winner 后传 winner 沙箱给 `observe(..., baseline=)` |
| `rollout/usersim_collect.py` | 同上（单槽：turn 前 snapshot、传 sandbox+baseline） |
| `scripts/agents_harness.py` | 打印 `state_diff`，离线即可看 diff-driven 证据 |

> 向后兼容：`observe([...])`（无沙箱）、`build_observer_prompt(actor_trajectory=, file_tree=)` 旧调用仍可用；移除的 `_probe_file_tree` 无外部引用。

---

## 5. 验证（本机，无 GPU/E2B）

```
# diff 读到内容 + 暴露"声称 vs 实际"
+ ADDED ./report.txt (text, 16B)
    Q3 total = 99999
99999 in prompt: True | 12345 in prompt: True     # observer 同时看到实际值与声称 → 能判 discrepancy
discrepancies: "claimed 12345 but file says 99999"

# LocalSandbox 持久
write a.txt → 另一次 run_code 读回 'hello'          # state 跨调用不丢

# 回归
run_simulated_session 离线: turns=2 trajs=8 ✓        # 无回归
prompt/parse/observe 离线检查 ✓ | ReadLints 无错
```

harness（`--mode collect`）turn1 显示 `+ ADDED ./report.txt … Q3 total = 123`，turn2 显示"(no filesystem changes)"（同内容重写=无净变化），diff 行为正确。

---

## 6. 与反 reward-hacking / 论文的关系

diff-driven 把"一份两用报告 $R_t$"的事实基础从"actor 自述"换成"沙箱真实效果"，使 reward 的 completion 判定**坐实在可验证的产物/数值**上——这是论文 §4.5（三 agent）与贡献 C5（防漂移 + 弱对抗自适应课程）里"reward 落到实际效果、抗嘴上说做完"的工程兑现。论文附录 A.1（observer prompt）已同步更新为 diff-driven 版本。

---

## 7. 剩余 / 待集群

- **二进制产物**（xlsx/png/db）当前只记 size+sha，需 openpyxl 等**格式解析**才能核对"声称的单元格数值"。
- **瞬态/被覆盖的中间产物**：现为 before/after 快照（只看净变化），未上 `watch_dir` 事件流。
- **真实后端**（`e2b`/`aliyun`）连通 + **8 槽 winner-sync 下 FS 级 baseline 正确性**（local mock 不做 FS sync，8 槽 local 仅近似；真实 e2b 下 winner 为活载体则正确）。
