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
