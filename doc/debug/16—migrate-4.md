# 迁移至 verl 原生 agent_loop 全量问题与修复（2026-07-29 会话）

## A. 架构迁移

| # | 问题 | 根因 | 解决办法 |
|---|---|---|---|
| A1 | 第一版迁移方向错 | 用 RayPPOTrainerV1 桥接 + 默认 RolloutManager，跑的是本地 ToolAgentLoop 非沙箱 harness | 对齐参考脚本: `python -m verl.trainer.main_ppo` + `trainer.use_v1=true` + `trainer.v1.trainer_mode=custom_sync` + `agent_loop_manager_class=RemoteAgentLoopManager` |
| A2 | CL 资产（9桶 buffer + CL loss + observer）需保留 | 迁移后入口变，注入点全换 | 六阶段迁移：CLTaskRunnerV1(复刻 TaskRunnerV1 三步) 注入 CL loss/buffer；CLAgentDataset 注入 fs-seed；ObserverDiffHook 接入沙箱；cl_replay_hook_v1 适配 KVBatchMeta |

## B. 环境 / 依赖

| # | 问题 | 现象 | 解决办法 |
|---|---|---|---|
| B1 | `import verl` 崩 | `transformer_engine has no attribute 'pytorch'`。镜像 flash_attn 是残缺 shim（只有 bert_padding），recipe_custom→megatron→TE 要 `flash_attn.flash_attn_interface`，shim 没有 | 给 shim 补 `flash_attn_interface.py`（转发真 FA3 `flash_attn_3` + `__getattr__` 兜底）+ `_train_impl.sh` PYTHONPATH 前置 AFS shim 目录 |
| B2 | 数据加载崩 | `assert src[-1]` NoneType。v1 `_init_dataloader` 无条件建 val dataset，config `val_files: null` | CLTaskRunnerV1.run 里 val 空/缺失 alias 到 train_files |

## C. 16 卡集群启动

| # | 问题 | 现象 | 解决办法 |
|---|---|---|---|
| C1 | lightllm 起 server 崩端口冲突 | `OSError:98 Address already in use`。recipe_custom `async_lightllm_server` 的 `pd_master_port(1212)` / `multinode_httpmanager_port(12345)` / `multinode_router_gloo_port(20001)` 用 argparse 写死 default；16 卡 = 8 replica / 每节点 4 个同端口撞 | 三端口改 `get_free_port` 动态分配 |
| C2 | observer hook 崩 | `Unknown post-run hook: 'trainer.observer_hook.ObserverDiffHook'`。monkey-patch factory 只在 driver 进程生效，AgentSessionWorker 是独立 ray actor，patch 传不过去 | (a) 改 `hooks/factory.py` 内建 FQN 分支；(b) `verl_runner.py` 把 PYTHONPATH 透传到 ray runtime_env.env_vars |
| C3 | rollout 部分 session 失败 | `ProtocolError: ConnectionState.CLOSED`。腾讯 AGS 入口网关 HTTP/2 GOAWAY（~1000 stream 回收连接），长任务滚动重启时触发 | (a) 禁 e2b HTTP/2 (`sandbox.py` patch `get_transport` → http2=False)；(b) 连接池调大 (`E2B_MAX_KEEPALIVE=1000/MAX=2000`)；(c) worker.py `_run_session_with_timeout` 加 GOAWAY 重采（重试一次） |
| C4 | compute_log_prob 崩 | `RuntimeError: expanded size (512) must match (378)`。`FusedLinearForPPO` 的 chunked 计算与 Qwen3.5 GDN 变长打包 + SP=4 交互导致张量维度不齐；GOAWAY 让部分 session 失败也恶化此问题 | **待 debug 验证**：禁 HTTP/2 + 重采已做，看 512 崩溃是否消失；若仍有则关 `use_fused_kernels` |

## D. 4GPU debug 环境

| # | 问题 | 现象 | 解决办法 |
|---|---|---|---|
| D1 | 4GPU 单机 ray.init 崩 | `ConnectionError: no Ray instance`。config `ray_init.address: auto`（16 卡多机才有 Ray 已起），单机没有 | 4gpu config 去掉 `address: auto`，让 ray 本地起 |

## E. Observer / Reward 审查

| # | 问题 | 解决办法 |
|---|---|---|
| E1 | observer 探针 MAX_FILES=200 截断 → 假 diff 噪声 | → 500 |
| E2 | max_depth=5 漏深目录 | → 7 |
| E3 | 探针没过滤所有 `.` 开头文件 → 缓存噪声 | 过滤所有 `.` 开头目录+文件 |
| E4 | observer 报告用 `str(ObservationReport)` 难读 | 改用 `_format_changes`（带 BEFORE/AFTER 内容的三段式正文）|
| E5 | judge prompt 没明确交叉核对指令 | rubric 加 `## MANDATORY cross-check (anti-reward-hacking)` 段 |
| E6 | actor 轨迹不是完整 JSON messages | judge prompt 里轨迹以完整 OpenAI messages JSON 呈现 |
| E7 | `envd.log` / `jupyter.log` 等运行时日志进 diff | `_is_runtime_file` 加 `*.log` / `*.pid` 后缀过滤 |
| E8 | 另一系统对照：`node-compile-cache` 等缓存进 diff | SKIP 名单扩充 + 探针过滤 `.` 开头目录 |

## F. 改动文件清单

### 本项目（agentic_cl_research）

**新增**: `cl_agent_dataset.py`、`cl_replay_hook_v1.py`、`trajectory_adapter_v1.py`、`observer_hook.py`、`observer_hook_register.py`、`e2b_goaway_patch.py`(已删,功能移入 verl 侧)、`train_manual_2node.sh`、`flash_attn_shim/flash_attn/flash_attn_interface.py`、`configs/run/b1_9b_4gpu.yaml`

**修改**: `verl_runner.py`(CLTaskRunnerV1 + val alias + PYTHONPATH 透传)、`_generated_ppo_trainer.yaml`(覆盖为 verl 最新版,含 v1 段)、`b1_9b_16gpu.yaml`(custom_sync + RemoteAgentLoopManager + fs-seed)、`model_reward_omni.py`(reward 回流)、`observer.py`(MAX_FILES/depth/过滤)、`prompts.py`(rubric 交叉核对)、`train.sh`(4gpu 预设)、`_train_impl.sh`(PYTHONPATH 前置 shim)、`sandbox_client.py`

### verl（dependencies/verl）

| 文件 | 改动 |
|---|---|
| `recipe_custom/agent/runners/hooks/factory.py` | +18 行：FQN 分支，支持 `name="trainer.observer_hook.ObserverDiffHook"` |
| `recipe_custom/agent/runners/sandbox.py` | +30 行：禁 HTTP/2(patch e2b get_transport) + 连接池调大 |
| `recipe_custom/agent/session_worker/worker.py` | +91/-24 行：`_is_goaway` helper + `_run_session_with_timeout` GOAWAY 重采 |
| `recipe_custom/rollout/lightllm/async_lightllm_server.py` | +8 行：三个固定端口改 `get_free_port` 动态分配 |


## G. Sandbox / GOAWAY 修复（腾讯 AGS 网关 HTTP/2 连接回收）

| # | 问题 | 现象 | 解决办法 |
|---|---|---|---|

| 并发方案 | ok | errors | elapsed | rps | p50 | p95 | p99 |
|---|---|---|---|---|---|---|---|
| HTTP/2（多 stream 复用长连接）| 900 | 0 | 0.541s | 1664.6 | 38.20ms | 157.81ms | 170.27ms |
| HTTP/1.1 + keepalive（独立短连接）| 900 | 0 | 8.466s | 106.3 | 677.81ms | 2640.90ms | 3462.54ms |
| **选择 HTTP/1.1** | — | — | 建连开销(~ms) < 命令执行耗时(~s) | — | — | **GOAWAY 消失** → 无 session 重采开销 | — |

| G1 | e2b 硬编码 `http2=True` | 入口网关 ~1000 stream 后 GOAWAY → `RemoteProtocolError: ConnectionTerminated`；`HTTPX_DISABLE_HTTP2` 无效（e2b 不读） | patch e2b `get_transport` / `get_envd_transport` 强制 `http2=False` → HTTP/1.1 |
| G1.1 | HTTP/2 长连接复用 → HTTP/1.1 | HTTP/2 单连接多 stream 复用（长连接），~1000 stream 后网关强制回收导致 GOAWAY。改为 HTTP/1.1 后每个请求独立短连接，网关不会发 GOAWAY，从根源消除 | `sandbox.py` patch e2b 全部 4 条 transport factory（async/sync × get_transport/get_envd_transport）强制 `http2=False`，所有 sandbox API 调用走 HTTP/1.1 |
| G1.2 | 实测 900 并发对比 | 执行命令耗时（秒级）远大于建连开销（毫秒级），HTTP/1.1 增加连接开销不影响整体耗时；但 **HTTP/2 触发 GOAWAY 会导致 session 失败重采（数十秒浪费）**，远大于建连差异 | 见下表 |
| G2 | 连接集中在少数连接上 | GOAWAY 触发更快 | `E2B_MAX_KEEPALIVE_CONNECTIONS=1000` / `E2B_MAX_CONNECTIONS=2000`（import e2b 前设） |
| G3 | GOAWAY 漏网（沙箱内 hermes 调 LLM 那条走不了的） | 部分 session 仍失败 | worker.py `_run_session_with_timeout` GOAWAY 重采：`_is_goaway` 判定 + 重试一次 |
| G4 | 我们的 `sandbox_client.py` 误用 `HTTPX_DISABLE_HTTP2` | 之前用这个变量以为能禁 HTTP/2，实际上 e2b 不理它 | 删掉无效代码，改为注释说明（GOAWAY 是长任务问题，短任务不改；需要时再开连接池/重试）|
| G5 | sandbox 日志被 WARNING 过滤 | `session_id↔sandbox_id` 映射看不到 | `sandbox.py` logger 设 INFO 级（`create_success` 日志可见）|
| G6 | sandbox kill 时偶发 `Instance is in STOPPED` | 沙箱已停止的状态下尝试 kill（无害清理竞态） | 无修（正常现象，`e2b` 自身处理）|

所有 GOAWAY 改动均在 verl 侧 `sandbox.py` + `worker.py`【跨 ray actor 进程天然生效】，不依靠 driver monkey-patch。

## H. 遗留（不阻塞，后续做）

- **fla/causal-conv1d torch fallback**：GDN kernel 慢，CUDA13 编译不出，需重 build 镜像补
- **FlashInferAllReduce disabled / MFU=0**：无害 warning
- **HERMES_TIMEOUT 750s**：部分 agent 任务超时，可通过 hook partial output 打分兜底
- **R 系列 buffer 端到端验证**
- **冷启动完整轨迹数据已就绪**：`cold_start/train.parquet` 含 messages 列，等 SFT/replay 阶段启用
- **C4 512 崩溃是否消失**：待 4GPU debug 验证
