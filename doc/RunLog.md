# 运行记录 / RunLog（append-only）

> **硬性规则：** 任何可判定结果的动作（smoke / 训练 / 评测 / bug 复现与修复）都追加一条；
> **成功与失败都保留，禁止删改历史条目**——失败是调试与论文的证据。
> 量化数据看 wandb / `logs/buffer_stats/*.jsonl` / `eval/results/*`；本文件记「发生了什么、为什么」。
> 交接背景见 [`Migration_64GPU.md`](Migration_64GPU.md)，交付状态见 [`Progress.md`](Progress.md)。

## 条目模板（复制后填写，最新在最上面）

```
### YYYY-MM-DD HH:MM | <机器/卡数> | commit <短hash>
- 动作：<命令 / 配置 / 改动>
- 结果：✅/❌/⚠️ <关键数字：loss、step、通过数、报错首行>
- 产物：<ckpt / *.jsonl / wandb run / eval json 路径>
- 解释：<为什么成功 / 失败根因 / 下一步>
```

---

## 记录（最新在最上面）

### 2026-06-19 03:38 | 本机（在家 macOS，无 E2B/GPU） | commit <pending>
- 动作：理清 observer/reward 的 trajectory 边界，落定**双通道 + pass-through**。observer **模型只看 state（diff）**，永不收 trajectory（不浪费 LLM token）；trajectory 由 observer **组件**捎带为 `ObservationReport.actor_trajectory`（pass-through，不进 observer prompt），reward 从这一份 R_t 读 trajectory 判 safety/robustness、读 state_diff 判 completion。schema 把 `actor_claims` 改名为 `actor_trajectory`（语义=原始轨迹、非"声称"）；`observe(sandbox, *, actor_trajectory=, baseline=, post=)`；`score_followup(query, report, judge)` 不再单独传 trajectory；`build_reward_judge_input` 从 `report.actor_trajectory` 取（封顶）；`_report_block`(questioner) 去掉 actor_claims；`build_observer_prompt`/`OBSERVER_SYSTEM` 状态化（不提 trajectory/claims，输出键去掉 actor_claims）。
- 结果：✅ 端到端验证——observer LLM prompt 不含 trajectory（SECRET_TRAJ_TOKEN 不泄漏）、报告 pass-through 携带、reward 从报告同时拿到 trajectory+state_diff；`test_agents`/`test_simulated_session`/`test_sandbox_client` 可跑用例全过（修了一批残留 `actor_claims` 引用：`_report_block` 生产崩溃点 + 5 处测试构造/断言 + harness mock）；`ReadLints` 无错；harness 两模式跑通。
- 产物：`agents/{schema,observer,prompts,reward}.py`、`rollout/{simulated_session,usersim_collect}.py`、`scripts/agents_harness.py`、`tests/test_agents.py`、`doc/接口使用_Sandbox与三Agent.md` §0/§2/§4。
- 解释：演进——先"reward 不给 trajectory"→"observer 也不给"→澄清为"observer **模型**不给（省 token），但 observer **组件**可 pass-through 给 reward"。最终：observer 纯状态取证，trajectory 走一条不经 observer 模型的旁路通道到 reward。**残留待同步（非代码）**：`paper/latex/sections/A_prompts.tex` O6 prompt、`Paper_Method_draft_{EN,CN}.md` §4.5+附录、`doc/UserSim_*` 设计信源仍是旧"observer 读 claims 比对"叙事。

### 2026-06-19 03:08 | 本机（在家 macOS，无 E2B/GPU） | commit <pending>
- 动作：reward 路径两项收口。(1) **不再给 judge trajectory**——只判当前状态：`build_reward_judge_input` 去掉 trajectory（返回空），judge 以 `state_diff` 为 ground truth；`build_judge_prompt` 空 trajectory 时跳过该段（path A 训练主轨迹不受影响仍带 solution_str）；REWARD_RUBRIC 措辞改为"按当前状态/diff 判，非叙事"；rubric 不再重复结构化报告块（diff 已含内容+SysOps），diff 封顶。(2) **几层拦截**：`ObservationReport.has_effect`（observer 空 diff 时置 False）→ `score_followup` 短路返回 score 0、**不调 judge**（带 gated）；`is_empty()` 改为 diff-aware（has_effect=False 即空，触发失败/耐心路径）。
- 结果：✅ 本机验证——reward 不传轨迹（judge.last_traj==""）、diff+SysOps+discrepancies 进 rubric；空 diff gate 不调 judge（score 0/gated）；build_judge_prompt 空轨迹跳段、非空保留；is_empty diff-aware；`test_agents` 15 + `test_simulated_session` 4 全过（mock agent 改为真写沙箱使 diff 非空）；path A `compute_score` 不受影响；`ReadLints` 无错。
- 产物：`agents/{reward.py,prompts.py,schema.py}`、`trainer/model_reward.py`、`rollout/simulated_session.py`、`tests/{test_agents,test_simulated_session}.py`、`doc/接口使用_Sandbox与三Agent.md` §2.5/§3.4。
- 解释：reward 应落在**真实状态**而非 actor 叙事上——给 trajectory 会把 diff-driven 想绕开的"声称"又带回来、且是最大长度膨胀源。拦截层把"没产生效果的轮"在调 judge 前就短路，省掉昂贵 round-trip 并天然给 0 分。

### 2026-06-19 02:51 | 本机（在家 macOS，无 E2B/GPU） | commit <pending>
- 动作：observer 取证层增强 (a)+(b)+LLM 可选。(a) 二进制内容提取——快照对二进制只标记，diff 后只对**本轮变更的** xlsx/docx/pptx/pdf 跑沙箱内提取（openpyxl/python-docx/python-pptx/pdfplumber）→ 文本进 diff（`kind=binary→text`），缺库/解析失败降级不崩。(b) `snapshot_system`/`diff_system`——本轮装的包/开的端口(LISTEN)/起的进程（不采 env 值，防泄密）。LLM 可选：`Observer(use_llm=False)` 默认确定性建报告零模型调用，确定性取证层始终运行。snapshot 改 `{fs,sys}` bundle，drivers 透传。harness 加 `--use-llm`（默认确定性）。
- 结果：✅ 本机验证——探针均可编译；默认确定性报告 final 从 diff 填（report.csv 内容入 diff）；二进制变更进 final 且 fallback 不崩；diff_system 正确（pandas/8000/nginx）；空 fs+sys diff→最小报告无 LLM；`use_llm=True` 路径完好；3 个 driver 回归（simulated/patience/usersim）通过；`test_agents` 观察用例已切 use_llm=True + 新增确定性用例；`ReadLints` 无错。harness 默认确定性跑通。
- 产物：`agents/observer.py`、`scripts/agents_harness.py`、`tests/test_agents.py`、`doc/接口使用_Sandbox与三Agent.md` §3.4、`CLAUDE.md` TODO#5。
- 解释：调查确认 agent 不只改 workspace（SysOps 装包/起服务 + 产二进制 office 文件），故 (b) 必要、#7(缩范围) 不做。确定性取证（快照+diff+二进制提取+sysops）做成代码层、observer LLM 可选——把"观察"从昂贵的 reward 模型里剥出来，reward/questioner 只读一份已是文本的 R_t。

### 2026-06-19 02:36 | 本机（在家 macOS，无 E2B/GPU） | commit <pending>
- 动作：observer 性能优化 Tier 1（4 项）。#1 空 diff 跳过 observer LLM；#2 每轮 1 次快照（driver 把上轮 post 前传作本轮 baseline，`observe(..., post=)`）；#3 变更检测改 (size, mtime)、探针去掉整文件 sha1（不再每次读全量字节）；#4 prompt 去掉冗余 file tree + 内容/文件数封顶。改 `agents/observer.py` + `rollout/{simulated_session,usersim_collect}.py`。
- 结果：✅ 本机验证 4 项全过——#3 mtime diff 读到内容（report.txt=99999）；#1 空 diff → 0 次 LLM 调用；#2 传 post 时不再触发快照 run_code；#4 prompt 无 file-tree 段。`run_simulated_session`(2 turns/8 trajs)、`run_usersim_session`、observe/parse 回归通过；`ReadLints` 无错。
- 产物：`agents/observer.py`、`rollout/simulated_session.py`、`rollout/usersim_collect.py`。
- 解释：把 observer 每轮开销从"2 次全量哈希快照 + 1 次大 prompt LLM"降到"1 次轻量(stat)快照 + 仅在有变更时 1 次小 prompt LLM"。调查结论：agent 不只改 workspace（SysOps 装包/改系统、产 xlsx/docx/pdf 二进制），故 **#7 不做**；#6(watch_dir) 为后端事件型（离线不可验、8 槽成本、且只覆盖 FS），建议改投"二进制内容提取 + 非 FS 命令探针"（见下次讨论）。

### 2026-06-19 02:14 | 本机（在家 macOS，无 E2B/GPU） | commit <pending>
- 动作：把 observer diff-driven 改造**总结成技术报告** `paper/refs/Observer_DiffDriven_技术报告.md`（问题/6 条理由/设计/实现/验证/剩余）并在 `paper/refs/README.md` 索引；**同步论文**——`paper/latex/sections/A_prompts.tex` O6、`Paper_Method_draft_{EN,CN}.md` 附录 A.1 + §4.5 观察 agent 表述、`Paper_论文向总览.md` §7.2 观察 agent 行 + §10 局限（均把"claim-driven"旧表述改为 diff-driven）。
- 结果：✅ 论文里 observer 提示词与 §4.5/§7.2/§10 表述与代码一致；grep 确认无残留 "claims DRIVE" 旧 prompt。
- 产物：`paper/refs/Observer_DiffDriven_技术报告.md`、`paper/refs/README.md`、`paper/latex/sections/A_prompts.tex`、`paper/drafts/Paper_Method_draft_{EN,CN}.md`、`paper/drafts/Paper_论文向总览.md`。
- 解释：代码改了（claim→diff-driven），论文/附录的 observer prompt 与表述会变 stale，本条把"报告 + 论文"对齐到当前实现，避免投稿稿与代码脱节。

### 2026-06-19 02:00 | 本机（在家 macOS，无 E2B/GPU） | commit <pending>
- 动作：observer 从 **claim-driven 改为 diff-driven**。`agents/observer.py` 加只读快照探针（仅用 `run_code`，后端无关）+ `snapshot()`/`diff_snapshots()`，`observe(traj, sandbox, baseline=)` 出 before/after **内容级** diff；`OBSERVER_SYSTEM`/`build_observer_prompt` 改为 diff=ground truth、actor 声称仅交叉核对；`ObservationReport` 加 `state_diff`。`LocalSandbox` 改**持久 workdir**。`simulated_session`/`usersim_collect` turn 前取 baseline、传 winner 沙箱。harness 打印 state_diff。
- 结果：✅ 本机验证——diff 读到内容（`+ ADDED ./report.txt … Q3 total = 99999`）、observer 提示同时含实际值 99999 与声称 12345（→ 能判 discrepancy）；`LocalSandbox` 跨 `run_code` 持久（写 a.txt→读回 hello）；`run_simulated_session` 离线回归通过（turns=2 trajs=8）、prompt/parse/observe 无回归；`ReadLints` 无错。
- 产物：`agents/{observer.py,prompts.py,schema.py}`、`rollout/{sandbox_client.py,simulated_session.py,usersim_collect.py}`、`scripts/agents_harness.py`、`doc/接口使用_Sandbox与三Agent.md` §3、`CLAUDE.md` TODO#5。
- 解释：为何改——(1) 模型会**幻觉**，声称可造假；(2) 声称只报结果、**丢中间产物**（设计最看重的中间态易被覆盖）；(3) reward 落在声称上=**可被 reward-hack**；(4) diff 由代码**确定性**算，不在证据里引入第二层模型误差；(5) diff 暴露 actor **没提的改动**（静默/部分失败）；(6) 能**内容级核对**"声称值≠实际值"。剩余：二进制格式解析、`watch_dir` 抓瞬态中间产物、真实 e2b/aliyun 连通待集群。

### 2026-06-19 01:50 | 本机（在家 macOS，无 E2B/GPU） | commit <pending>
- 动作：新增多 Agent **离线 harness** `scripts/agents_harness.py`（驱动真实 session driver + 真实 observer/questioner/reward，默认 mock endpoint、`--real` 可切，`--mode simulated|collect`、`--backend local|e2b|aliyun`）；新增接口使用指南 `doc/接口使用_Sandbox与三Agent.md`；`Progress.md` 变更日志补今日条目（含协作方改动）。
- 结果：✅ harness 两模式本机跑通——simulated（8 槽：observer 报告 + questioner 出题 + reward 0.82，16 traj，ended_by=k_budget）、collect（1 槽：observer+questioner，ended_by=end_session）；`ReadLints` 无错。
- 产物：`scripts/agents_harness.py`、`doc/接口使用_Sandbox与三Agent.md`、`doc/Progress.md`。
- 解释：harness 让"三 Agent 协同回路"在无 GPU/无真实模型下**一条命令可见、可调**，且换沙箱后端即验证；真实模型/沙箱连通待集群。

### 2026-06-19 01:36 | 本机（在家 macOS，无 E2B/GPU） | commit <pending>
- 动作：按用户方向收敛——**聚焦"沙箱接口/实现分离"，其余厂商后端留空**。把上一条里写满的 `AliyunSandbox`（AgentBay 真实现）改回**空 stub**（接口契约 + 注册点保留，body 抛 NotImplementedError）；`sandbox_client.py` 显式分三段 INTERFACE / IMPLEMENTATIONS / REGISTRY；撤回 `pyproject.toml` 的 `[aliyun]` 推测依赖（改为注释指引）。
- 结果：✅ 本机手测通过（`local` 跑 6*7=42；未知后端→ValueError 列出注册名；`aliyun`→NotImplementedError(留空)；`register_backend` 可扩展；`AliyunSandbox` 仍满足 run_code/kill 接口形）。
- 产物：`rollout/sandbox_client.py`（INTERFACE/IMPL/REGISTRY 分段 + Aliyun stub）、`tests/test_sandbox_client.py`(aliyun 测改为断言 stub)、`pyproject.toml`/`configs/base.yaml`/`scripts/sandbox_smoke.py`(标注 aliyun=stub)、`CLAUDE.md` TODO#5。
- 解释：交付物 = **接口（`SandboxClient` Protocol）与实现解耦 + 按名选择的开放注册表**；换厂商 = 选 backend 名 / 加一个 `register_backend`。具体 vendor body（阿里等）留空，回集群用真 SDK/凭证填，避免在家凭文档臆测。

### 2026-06-19 01:25 | 本机（在家 macOS，无 E2B/GPU） | commit <pending>
- 动作：回答"沙箱接口与实现是否分离 / 能否从腾讯换阿里"。把沙箱后端从**封闭工厂**（if/else 只认 local/e2b）改成**可插拔注册表**（`register_backend` + 按 name dispatch），新增 `AliyunSandbox`（阿里云 无影 AgentBay，`wuying-agentbay-sdk`，`AGENTBAY_API_KEY`）。
- 结果：✅ 注册表/工厂/AliyunSandbox 逻辑本机手测通过（local→运行 6*7=42；unknown→ValueError 列出已注册名；aliyun 无 key→RuntimeError；register_backend 开放扩展）。`AliyunSandbox` 满足 SandboxClient Protocol（run_code/kill）。⚠️ e2b/aliyun **真连通待回公司**（本机无 httpx/SDK/凭证，也无 pytest/ruff，仅手测逻辑）。
- 产物：`rollout/sandbox_client.py`（registry + AliyunSandbox）、`tests/test_sandbox_client.py`(+3 测)、`pyproject.toml`(`[aliyun]` extra)、`configs/base.yaml` / `rollout/__init__.py` / `scripts/{collect_rollout,collect_cold,sandbox_smoke}.py`(+aliyun choice)。
- 解释：结论 = **接口（Protocol）与实现本就分离，但工厂封闭、且只有 e2b/local**；换阿里需"写一个实现 Protocol 的类 + 注册 + 配自己的 env"，非零代码。本次把工厂改开放并补上 AliyunSandbox，使"换厂商 = 选 backend 名 / 加一个 register_backend 调用"，rollout loop 零改动。

### 2026-06-19 01:15 | 本机（在家 macOS，无 E2B/GPU） | commit <pending>
- 动作：debug 复现"observer 能否观察真实系统/Agent 状态"——给真实 `agents/observer.py` 打 NDJSON 桩，跑真实代码路径（GRPO 路径 `observe(无沙箱)` + 采集路径 `observe(LocalSandbox)`），actor 真写 `report.xlsx`=99999 但自述 12345。
- 结果：❌ **observer 看不到真实状态**：H1 主路径 `sandbox_is_none=true`；H2 即便传 LocalSandbox（agent 写文件 `ok=True`）`tree_len=0`；H3 唯一证据是 actor 自述文本（`file_tree_empty=true`、不读内容、`tool_outputs` 不传）。调试桩已按规范清除（`git diff agents/observer.py` 为空）。
- 产物：无持久产物（instrumentation 已移除）；结论 + 修复计划记入 `CLAUDE.md` TODO#5。
- 解释：单测全绿是因 mock judge 返回写死 JSON；实运行 observer 退化为"actor 自述复读机"，reward 的"按真实效果打分（反 hacking）"沦为"按声称打分"——正是设计想杜绝的攻击面。修复 P0/P1 见 TODO#5，依赖**有持久文件系统的真实沙箱**（回公司）。

### 2026-06-16 10:50 | 4×H800 本机 | commit c36626f
- 动作：vllm 版本升级 0.13.0 → 0.19.0，解决 Qwen3.6-27B 不兼容问题
- 结果：✅ vllm 0.19.0 安装成功 + verl import 正常 + 沙箱连通；❌ GPU 被僵尸 CUDA context 占满无法启动 vllm
- 产物：`/mnt/afs_toolcall/sunhao4/envs/vllm019_venv/`（新 venv），`doc/vllm_upgrade_0.19.md`（技术文档，已归档到 `paper/refs/`）
- 解释：Qwen3.6-27B 是 hybrid linear/full attention 模型，vllm 0.13 不支持 `Qwen3_5ForConditionalGeneration`。升级到 0.19 后 ModelRegistry 已包含该架构。verl 安装时不能带 `[vllm]` extra（会降级到 0.12）。GPU 僵尸进程 PID 869795/881008/892490/901013 占用 4×75GB 显存，需管理员介入释放。

### 2026-06-16 10:30 | 4×H800 本机 | commit c36626f
- 动作：验证腾讯沙箱 E2B 后端连通性
- 结果：✅ `sandbox_smoke.py --backend e2b` 跑通，sandbox execute + GRPO advantage + domain tagging 正常
- 产物：无持久产物（smoke 测试）
- 解释：E2B_API_KEY / E2B_DOMAIN 凭证有效，沙箱可创建实例并执行代码。

### 2026-06-16 10:20 | 4×H800 本机 | commit c36626f
- 动作：从原始数据生成 queries.jsonl
- 结果：✅ 10774 sessions / 143655 queries → queries_clean.jsonl（20 条脏 query 丢弃）
- 产物：`/mnt/afs_toolcall/sunhao4/datasets/juxiaolong_prefix/queries_clean.jsonl`
- 解释：`prepare_queries.py` 提取用户 query，`clean_queries.py` 清洗零宽字符。

### 2026-06-13 | 64 卡集群（8×8 H800，单节点交互） | commit <pending>
- 动作：搭建**冷启动多轮 rollout 采集**链路（observer + questioner，**无奖励**）。新增 `rollout/usersim_collect.py`（slots=1 单轨迹多轮，无 winner/reward）、`scripts/collect_rollout.py`（双 actor 入口，N 路并发，jsonl 落盘）、`scripts/collect_rollout.sh`（启动器：预检远程→起 vllm→两路采集）。模型：远程 actor=gpt-5 / observer=gpt-4.1-mini / questioner=claude-sonnet-4-6（三者不同），本地 actor=Qwen3.6-27B；两套 actor 数据分目录 `data/rollouts/{local,remote}/`。
- 远程 API：tokenhub（`https://tokenhub.sensetime.com/v1`），key 取自 `apodex_research/configs/env_deepseek_v4_pro.env`（本仓库无真实 key）。**硬约束（孙豪）：远程不可用即中止**（observer/questioner 缺则多轮无法进行），启动器预检 3 模型任一非 200 即 exit 5。
- 环境踩坑：`/opt/conda` 的 vllm0.11 与 torch2.9.1 ABI 不匹配（`vllm._C undefined symbol`）→ vllm0.13.0 `--no-deps --target` 装到共享盘 `envs/vllm013` 覆盖 + `PYTHONPATH` 前置 + `LD_LIBRARY_PATH` 补 `/opt/conda` nvidia 库。验证：vllm0.13 + torch2.9.1 + transformers5.2（认 qwen3_5）。
- 结果：✅ 远程三模型连通；✅ 运行环境凑齐；✅ **远程 actor 路 small-batch 验证通过**（`--actor remote --backend local --limit 2` → done=2 failed=0，产出含多轮 + persona（如 "Dr. Lena the researcher"）+ questioner 生成 query + observer 5 字段报告）。
- 产物：`data/rollouts/{local,remote}/rollouts_*.jsonl`（正式）；smoke 在 `/tmp/rollout_smoke/`。日志 `logs/cold/{vllm,rollout_*}.log`。
- 解释：链路确认工作。待办：① 起本地 vllm 跑 local actor + e2b 真沙箱全量；② `collect_rollout.py` 加分片（8 机并行不重复）。详见 `doc/ColdRollout_采集.md`。

### 2026-06-12 | 单卡开发机（conda py3.10 + torch；无 verl） | commit <pending>
- 动作：实现"让 rollout 从我们的调度器走"——把 verl 默认批量 rollout 替换为自管会话调度（16×8 + winner-sync），每步生成调 verl 原生 LLM server。基于对 verl 0.8.0 源码的核实：
  - 发现 verl 官方注入点 `actor_rollout_ref.rollout.agent.agent_loop_manager_class`（`ray_trainer.py:931`，`load_class_from_fqn(fqn,"AgentLoopManager")`）——设一个 yaml key 即替换 rollout manager，**不碰 fit() 一行**，比 monkey-patch 还干净。
  - 单步生成连接点 = `LLMServerClient.generate(request_id,*,prompt_ids,sampling_params)->TokenOutput{token_ids,log_probs}`（`verl/workers/rollout/llm_server.py:180`，async）。
  - verl rollout 输出契约 = prompts[B,P]/responses[B,R]/response_mask/input_ids/attention_mask/position_ids(+rollout_log_probs) + non_tensor（`agent_loop.py:_postprocess`）。
- 改动：
  - `inference/generate.py`：`VerlRolloutGenerateFn` 占位 NotImplementedError → 真实实现（messages→apply_chat_template→`llm_client.generate`→GenStep；asyncio 桥接同步 GenerateFn）。
  - 新建 `trainer/cl_rollout_manager.py`：`trajectories_to_dataproto`（纯函数，Trajectory 列表→verl 契约 DataProto，left-pad prompt/right-pad response）；`extract_queries_from_prompts`（从 gen_batch 取种子 query，优先 raw_prompt 否则 decode）；`CLSchedulerAgentLoopManager`（subclass verl AgentLoopManager，重写 generate_sequences：拆 query→RolloutScheduler.run_step（16×8+winner-sync，agent_fn=make_react_agent_fn(VerlRolloutGenerateFn)）→组装回 DataProto）。verl import 全部延迟（模块级 `__getattr__` 懒构造 `AgentLoopManager` 属性）。
  - `configs/base.yaml`：加注入点（默认注释关闭，按需开 = 走自定义 rollout）。
- 结果：✅ 全量 **227 passed / 7 skipped**；新增 `tests/test_cl_rollout_manager.py` 8 个（fake verl DataProto 验证组装形状/mask/logprob/non_tensor/位置 id + query 提取 + VerlRolloutGenerateFn async 桥接）。修了 fake-verl `sys.modules` 与 test_cl_loss 的串扰（无条件给 verl 模块挂 DataProto）。ruff 新文件全过。
- 解释：⚠️ **纯逻辑（DataProto 组装、query 提取、scheduler 接线、async 桥接）mock 单测通过；真 verl 端到端（真 AgentLoopManager 子类化 + 真 LLM server generate + 真沙箱 16×8 winner-sync + 真 DataProto 回流到 fit）仍待 GPU 集群**——本机无 verl/LLM server。这条线接通了"训练入口 ↔ rollout 沙箱编排"（此前是两套没接的东西，Gap D）。下一步集群：开 base.yaml 注入点 + 真 LLM server 跑 1-step 验证 generate_sequences 出的 DataProto 能喂进 fit 的 old_log_prob/advantage/update。


### 2026-06-12 | 单卡开发机（凭证就位 + 外网可达腾讯北京区） | commit <pending>
- 动作：首次用真实凭证（`docker/sandbox/tencent.env` 已填 TENCENTCLOUD_SECRET_ID/KEY + E2B_API_KEY）启动真实腾讯沙箱（ap-beijing.tencentags.com）。起 1 个 → 8 个 → 128 个（16 会话 × 8 槽，= 一个训练 step 并发峰值）；并验证回收。
- 结果：✅ **首次真实后端跑通**（此前 RunLog 一直记"沙箱起不来"）。
  - 单个：create 0.5s → `print(2+40)`→`42` → kill，1.1s。
  - 128 并发：**128/128 OK，零失败，wall 3.6s**；create min0.51/p500.69/p950.91/max1.45s；单实例端到端 p50 2.78/p95 3.47s；结果全部校验正确。平台无限流、无排队。
  - ❌→✅ **发现并修复回收 bug**：`E2BSandbox.kill()` 用 `_full_id`（`sandboxID-clientID`）DELETE → **404，实例泄漏**（kill 后仍能执行）。实测正确形态 = 裸 `sandboxID` → **204**，删后实例 401 不可达。已改 `kill()` 用 `self._sandbox_id`；连续 3 轮×8 起-kill 无累积泄漏。加回归测试 `test_e2b_kill_deletes_bare_sandbox_id`（mock httpx 断言删 url）。
- 证据：`api.ap-beijing.tencentags.com` 裸 curl HTTP 401（网络通、鉴权头缺）；带 `X-API-KEY` create 204/正常。早先泄漏实例靠 300s timeout 自动过期兜底。另：create `timeout` 最小 300s（传 120 被平台 400 拒），默认值 300 正确。
- 产物：`rollout/sandbox_client.py`（kill 修复）、`tests/test_sandbox_client.py`（回归测试，11 passed）。
- 解释：⚠️ 本轮只验证了"裸实例 create/exec/kill/回收"，**winner-snapshot-fork（会话内 8 槽派生 + winner 同步）尚未在真实平台验证**——那是设计里风险最高的 PoC（平台是否支持进程态 fork），下一步用 `SessionSandboxPool` 真后端跑一条 winner-sync 会话。rollout 规模结论不变：1024×8 / 128 并发（实测 create p50 0.69s，沙箱非瓶颈，支撑设计假设）。


### 2026-06-12 | 单卡开发机（conda py3.10 + torch 2.9.1，临时装 pytest/omegaconf/ruff；无 verl） | commit <pending>
- 动作：修复 Loss 链路三个 bug（接上条审计）。
  - **Bug-1**：`make_cl_loss` 闭包签名改为 `(model_output, data, dp_group=None)` 去掉 config 位置参（verl engine keyword 调用不传 config）；RL 项改为调注入的 `base_loss_fn`，或在 worker 端 lazy 用 `omega_conf_to_dataclass(actor_cfg)` 重建 `ActorConfig` 再 `partial(ppo_loss, config=...)`（复刻 verl engine_workers.py:543/584）。`verl_runner.make_cl_loss_from_cfg` 传 `actor_rollout_ref.actor` 子树（pickle-safe）。
  - **Bug-2**：`compute_replay_loss` 新增 `_to_dense_response_logprobs`，先 `no_padding_2_padding(log_probs, data)` 把 NestedTensor 还原成 dense `[bsz,max_resp_len]` 再按 `is_replay` 选行（mock/dense 输入 no-op 兜底）。
  - **Bug-2b**：`build_replay_rows` 重写为 left-right padded rollout-row 形态——拆 prompt/response，产 `prompts`(左pad)/`responses`(右pad)/`input_ids`/`attention_mask`/`position_ids`，replay mask/weights 改为 response 宽(R)对齐；满足 `no_padding_2_padding` 的 `assert sequence_offsets[-1]==values.shape[0]` 与 `prompt_len>0`。`_append_replay_rows` 改为 prompt 维左 pad、response 维右 pad 分维对齐后重建全序列字段再 `DataProto.concat`。
- 结果：✅ **全量 218 passed / 7 skipped**（skip 全是需 verl/CUDA 的全栈 smoke）。改动相关 test_cl_loss/test_replay_*/test_agents/test_simulated_session 全绿；按新签名更新了 test_cl_loss 的 3 个用例（注入 base_loss_fn / 去 config 位置参）。ruff 对新增+改动文件无 lint 问题（verl_runner 残留 2 个 import 排序是原有代码，未动）。
- 产物：`trainer/cl_loss.py`、`trainer/replay_forward.py`、`trainer/verl_runner.py`、`tests/test_cl_loss.py`。
- 解释：⚠️ **三处修复的纯逻辑在 mock/dense 下验证通过，但"在真 verl NestedTensor + no_padding_2_padding + DataProto.concat + left_right_2_no_padding 全链路端到端"仍需 GPU 集群验证**（本机无 verl，test_verl_smoke/test_buffer_hooks_smoke 仍 skip）。下一步：集群上跑 1-step 全栈 smoke 验证 replay 行真能过 forward 且 log_probs 可选回、断言不触发。


### 2026-06-12 | 单卡开发机（无 verl，conda py3.10 无 pytest） | commit <pending>
- 动作：实现 UserSim 三 agent（doc/UserSim_多轮Query在线生成.md §7 契约，原"暂不实现/prompt 留空"）。新建 `agents/`（observer/questioner/reward/personas/prompts/schema/base）+ `inference/`（generate 边界封装）；在 `rollout/simulated_session.py` 落地 Algorithm 1（`run_simulated_session`，复用现有 `SessionSandboxPool`，不改其代码）。填三个 prompt：O6 Observer（客观无人设）、O3 Questioner（16 人设、防 AI 腔、`<end_session>`）、O4 Reward（observation-grounded、抗 hacking）。Reward 复用 `trainer/model_reward.JudgeClient`，零改动 judge I/O，只加 rubric。
- 结果：✅ mock 冒烟全通过——imports OK（personas=16）；耐心公式 §3.6.5 P1..P4=[0.9,0.7,0.3,-0.5]；observer JSON 解析+兜底；三 prompt 注入校验；reward 复用 ClawEval 聚合（safety*(0.8c+0.2r)=0.48）+ discrepancies 进 rubric + judge 故障兜底；会话编排端到端 2 轮 8 轨迹按 K 预算正常结束。`py_compile` 全部新文件通过。新增 `tests/test_agents.py`(约21)+`tests/test_simulated_session.py`(4)。
- 产物：`agents/*.py`、`inference/*.py`、`rollout/simulated_session.py`、`tests/test_agents.py`、`tests/test_simulated_session.py`；`pyproject.toml` packages.find 补 `rollout*/agents*/inference*`（此前 rollout 也未被打包，一并修）。
- 解释：⚠️ **本机无 pytest/verl，未跑全量 pytest**——验证靠 `/opt/conda/bin/python` 直接 import+逻辑断言。**写测试时一度把耐心 P4 期望写成 -0.1，实际公式 `P_0-d_0(2^k-1)=1-0.1*15=-0.5`，代码对、测试错**，已修测试（doc §3.6.5 例"累计扣减 …1.5"印证）。VerlRolloutGenerateFn 故意 NotImplementedError（generate 接线 = Gap D 待集群）。三 agent 后端各自 env（OBSERVER_/USERSIM_/JUDGE_）抗 self-preference。下一步：集群上接 VerlRolloutGenerateFn + 把 run_simulated_session 接进 scheduler；外部 judge API 地址待 @孙豪 提供。


### 2026-06-12 | 单卡开发机（无 verl/pytest 环境） | commit <pending>
- 动作：从 `cl_main` 入口逐节点审计训练链路对 verl 0.8.0 的 API 兼容性（verl 源码克隆于 `~/verl`，git@github.com:sunhatSH/verl.git）。重点核实 Loss 链路。
- 结果：⚠️ 集成方向正确（不 fork + set_loss_fn + patch _update_actor + 双 mask 均成立），但 **Loss 链路当前不能 work**，定位三个 bug：
  - **Bug-1（致命）**：`cl_loss_with_replay(config, model_output, data, dp_group)` 首参 config，但 verl engine 用 keyword 调 `loss_function(model_output=, data=, dp_group=)` 不传 config（engine 自己用 `partial(ppo_loss, config=actor_config)` 绑定，见 `engine_workers.py:584`、调用点 `fsdp/transformer_impl.py:1282`）。`verl_runner.py:54` 直接 `set_loss_fn(make_cl_loss_from_cfg(cfg))` 未 partial 绑定 → 首次调 loss `TypeError`。
  - **Bug-2（致命）**：`use_remove_padding` 默认 True（`verl/workers/config/model.py:120`），`model_output["log_probs"]` 是 NestedTensor（jagged，`fsdp/transformer_impl.py:1177` `nested_tensor_from_jagged`），非 dense `[N,T]`。`select_replay_rows` 的 `log_probs[rows]` 布尔行索引在 NestedTensor 上不工作。
  - **Bug-2b**：修 Bug-2 的标准做法是先 `no_padding_2_padding` 还原 dense，但其硬断言 `assert sequence_offsets[-1]==values.shape[0]`（`padding.py:131`）要求 replay 行提供正确拆分的 prompts/responses；当前 `build_replay_rows`（`replay_forward.py:192`）把整条序列当 response、无真 prompts → 断言失败。
- 证据（✅ 已确认存在/成立）：`set_loss_fn`(engine_workers.py:491)、`ppo_loss`(workers/utils/losses.py:57)、`_update_actor`(ray_trainer.py:1293/1649)、`actor_rollout_wg`(ray_trainer.py:894)、`DataProto.concat`(protocol.py:917)、`from_single_dict`(protocol.py:480)；"replay 行 response_mask=0 隐身、log_probs 仍在 model_output"成立（engine 前向按 attention_mask 算全 batch，不按 response_mask 裁行）。
- 解释：三个 bug 均为"dense 假设 vs verl 实际 NestedTensor"与"config 未绑定"的接线问题，非设计错误。修复需在有 verl 的环境验证（本机无 pytest/verl）。已在 `trainer/replay_forward.py` 把相关"待集群验证"升级为"已知不兼容"标注。Loss 修复列为独立后续任务，本轮先落审计 + 实现 user-sim 三 agent。


### 2026-06-10 ~16:30 | 单卡 H800（开发机） | commit <pending>
- 动作：按腾讯云文档 [129691](https://cloud.tencent.com/document/product/1814/129691) 本地准备自定义沙盒镜像与 Tool 配置（账号未就绪，先不落盘 build）。
- 结果：✅ `docker/sandbox/Dockerfile` + `requirements.txt` + `image.env.example`；`scripts/{validate,build,push,create_sandbox_tool}.sh`；`configs/sandbox_tool.json`（49999/49983 端口、探针、2C/2Gi）；`doc/Sandbox_Image_Onboarding.md`。`validate_sandbox_dockerfile.sh` 通过；`test_sandbox_dockerfile.py` 3 passed。
- 解释：Dockerfile 仅 `RUN pip install`，无 USER/WORKDIR/ENV/ENTRYPOINT，满足快照约束。真正 `docker build` 需账号登录 CCR 后 `docker pull sandbox-code:latest`。无账号阶段继续 `--backend local` 跑采样链路。


### 2026-06-10 ~16:04 | 单卡 H800（开发机） | commit <pending>
- 动作：验证沙盒能否执行+采样。探测真实沙盒可用性；按 doc §7 建 `SandboxClient` 适配层（local/e2b 双后端）+ GRPO group 采样；写 `scripts/sandbox_smoke.py` 跑通"execute + M 采样 + winner固化 + 领域→bucket"。
- 结果：❌ **真实沙盒起不来**：无 e2b SDK、无 E2B_API_KEY/E2B_DOMAIN 凭证、`tencentags.com:443` 网络超时（外网不通）。✅ **本地后端跑通**：M=6 组里 2 条 emit buggy code(obs=410,reward0,adv-1.41) vs 正确(reward1,adv+0.71)，winner固化按 tid 字典序选定，domain=Finance→bucket。`test_sandbox_client.py` 10 passed；全量 170 passed/1 skipped。
- 产物：`rollout/sandbox_client.py`、`rollout/__init__.py`、`scripts/sandbox_smoke.py`、`tests/test_sandbox_client.py`。cookbook 参考 `/root/workspace/ags-cookbook/examples/mini-rl/main.py`。
- 解释：真实沙盒须在有 SDK+凭证+网络的机器（64 卡集群或联网开发机）`--backend e2b` 跑，循环代码完全一致一行切换。本机用 local 子进程后端等价验证了"执行+采样"链路，与厂商解耦（§7 风险缓解）。


### 2026-06-10 ~16:00 | 设计决策 | commit <pending>
- 动作：定领域/入桶粒度——**per-query（非 per-session）**。
- 结果：✅ 当前代码已是 per-trajectory(=per-query) 解析，**无需改逻辑**。固化契约进 `doc/SandboxRollout.md §5.5` + `domain_tagging` docstring。
- 解释：trajectory 单元本就是 query（GRPO group=同 query 的 M 沙盒）；session 只是上下文来源、跨多桶正常，故"一会话一领域"假设可弃。标签由模型在完整上下文下 emit，追问类 query（"怎么样了"）能被正确归到进行中任务的领域。rollout 硬性要求：每 query 一条 trajectory + 末尾 emit `<task_domain>`，勿合并整段会话。


### 2026-06-10 ~15:52 | 单卡 H800（开发机） | commit <pending>
- 动作：实现"7 桶领域由 LLM 在任务处理时顺带输出"方案（缺口①）。新建 `trainer/domain_tagging.py`：`build_domain_instruction()`（注入 rollout 系统 prompt 的领域指令）+ `parse_domain()`（解析 `<task_domain>NAME</task_domain>`，含别名/取最后一个/校验）。接入 `trajectory_adapter`：无显式 bucket 时从 assistant 文本回收领域，仍无则跳过（B12）。`verl_runner` 传 `valid_buckets=buffer.bucket_names`。
- 结果：✅ 新增 `test_domain_tagging.py`；全量 **160 passed / 1 skipped**；lint 干净。
- 产物：`trainer/domain_tagging.py`、`tests/test_domain_tagging.py`、`trajectory_adapter.py`(回收逻辑)、`verl_runner.py`(传 valid_buckets)、`doc/SandboxRollout.md` §5.3(bucket 来源)。
- 解释：领域作为 rollout 副产物落到 trajectory→bucket，避免单独分类 pass。显式 bucket 字段优先于解析标签。**待 @郑乃榕 在 rollout 系统 prompt 里注入指令**；本机无模型只能验证解析/接入的确定性逻辑。


### 2026-06-10 ~15:45 | 单卡 H800（开发机） | commit <pending>
- 动作：对齐"数据→轨迹"链路；摸清 `datasets/_stage_prefix_pass.jsonl`（10774 session）结构；新建 stage② 提取脚本 `scripts/prepare_queries.py`（drop `<summary>` / 原文保留 / 不打 bucket / 保留 record_id 为 join key）；小样本验证。
- 结果：✅ 50 session → 50 行，795 queries（avg 15.9/session，0 empty）；record#0 正确剔除 8 个 `<summary>` 块剩 5 真实 query。`test_prepare_queries.py` 4 passed。
- 产物：`scripts/prepare_queries.py`、`tests/test_prepare_queries.py`、`.gitignore`(+`datasets/`)。样本输出已删。
- 解释：链路现状——① 原始数据有；② 提取脚本本次补上；③ 沙盒 rollout 仅 doc（外部依赖 @郑乃榕）；④ RL 框架就绪。**仍缺：7 桶 bucket 标签（原始数据无）**，按决定留待单独一步。边界规则：仅 lstrip 后以 `<summary` 开头的 user 消息算压缩块被剔除；"内嵌 summary 但开头是正文"的轮次保留。

### 2026-06-10 ~15:40 | 单卡 H800（开发机） | commit <pending>
- 动作：在本机起 buffer hooks **驱动级集成冒烟**（无模型权重，真 verl DataProto + 假 trainer），跑通 `install_buffer_hooks` 全流程。
- 结果：✅ `test_buffer_hooks_smoke.py` 2 passed；冒烟**暴露并修复 3 个真集成 bug**：(1) `_append_replay_rows` 在 RL batch 带 non_tensor(messages/bucket) 时 `DataProto.concat` 崩 → 给 replay_dp 回填占位 non_tensor 列；(2) ingest 误把 replay 行当新轨迹重复入桶 → 改用 append 前的 `rl_batch`；(3) `trajectory_adapter` 对 numpy 数组做 `a or b` 触发歧义真值崩溃 → 改显式 None 判断。另把 `response_token_ids` 在 adapter 从 responses tensor 回填（消除 weighting flaky）。
- 结果(全量)：✅ 146 passed / 1 skipped。
- 解释：这 3 个 bug 在真集群也会触发（verl non_tensor 一定是 numpy 数组），本机冒烟提前抓到，价值高。**全栈真训练仍 Blocked**（无权重/无外网/单卡），需 64 卡。


### 2026-06-10 ~15:30 | 单卡 H800（开发机） | commit <pending>
- 动作：实现论文证据钩子（`trainer/replay_metrics.py`：buffer 动态日志 + `forgetting_risk` 回填 + 周期 `buffer.dump`），跑全量回归。
- 结果：✅ **144 passed / 1 skipped**（新增 7 项 replay_metrics 单测）；lint 干净。1 skipped = 全栈 GPU smoke（需多卡）。
- 产物：`trainer/replay_metrics.py`、`tests/test_replay_metrics.py`、`configs/base.yaml`(+2 开关)、`doc/Progress.md`(变更日志)。
- 解释：纯「增加观测 + 激活既有功能」，不改 Loss/Buffer 核心语义。`forgetting_risk` 的当前-logprob 前向走 verl `compute_log_prob`，**off-GPU 优雅降级为 no-op，需在 64 卡上确认真回填**。下一步：迁 64 卡跑全栈 smoke。

### 2026-06-10（更早） | 单卡 H800 | —
- 动作：系统整理一轮（P0 本地修复 + P1 基线打通 + P2 工程补全），见 `Progress.md` 变更日志。
- 结果：✅ 130 passed / 1 skipped；replay 路径在 H800 上通过 CUDA smoke（grad>0）。
- 产物：`replay_buffer/`、`trainer/`、20 yaml、5 篇 skills、SQLite 持久化、async runner scaffold。
- 解释：修复 A1/A2/A4/A5/B1/B3/B4/B6/B7/B8/B9/B10/B11/B12/D2 等；详见 `Progress.md`。全栈仍 Blocked on GPU 集群。

---

## 待新 session 在 64 卡机器填写的第一批条目（预留）

- [ ] 复现 `pytest -q`（期望 144 passed / 1 skipped）→ 证明搬迁未破坏
- [ ] `pytest -m gpu`（验证 torch/verl/CUDA）
- [ ] merge verl Hydra defaults 后 `validate_config` 结果
- [ ] B1 全栈 1–2 step smoke（Colocate 64）结果
- [ ] R4 全栈 1–2 step smoke（replay 行 + forgetting 回填）结果
