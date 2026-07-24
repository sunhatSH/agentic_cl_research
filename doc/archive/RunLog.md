# 运行记录 / RunLog（append-only）

> **硬性规则：** 任何可判定结果的动作（smoke / 训练 / 评测 / bug 复现与修复）都追加一条；
> **成功与失败都保留，禁止删改历史条目**——失败是调试与论文的证据。
> 量化数据看 wandb / `logs/buffer_stats/*.jsonl` / `eval/results/*`；本文件记「发生了什么、为什么」。
> 交接背景见 [`Migration_64GPU.md`](../ops/Migration_64GPU.md)，交付状态见 [`Progress.md`](Progress.md)。

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

### 2026-07-16 00:34 | 本机 Mac（darwin，无 GPU） | commit 02eaa6e
- 动作：**第二次重 build** + push Qwen36-lightllm 训练镜像（修正）。第一次因远端改动未 pull（本地落后 30 commits）全层 CACHED 推送了旧内容（digest `8631ec6c`）；pull 后 Dockerfile 已含远端 2026-07-15 实机踩坑补齐：transformers 5.8.0→5.12.0、tensordict 区间限定、新增 ray/msgpack/torchdata/protobuf、pyyaml>=6.0.2、自检加 MistralForSequenceClassification。但 `npm install -g @openai/codex` 因 base 镜像无 Node.js 构建失败（exit 127 npm: command not found）。修复：在 codex 前加 `conda install -y -c conda-forge nodejs=22`。构建成功 push。
- 结果：✅ build+push 成功，新 digest `sha256:115f2596c8d8c5b25fe3f7bd6b1a32f749ef46a1138680248898c818055480f0`（8 层新推）。旧 digest `8631ec6c` 已被覆盖。⚠️ `InvalidBaseImagePlatform` 警告同前（base linux/amd64 vs 本机 arm64，不影响）。
- 产物：`registry.cn-tj-01.sensecore.cn/ccr-zuhu2026/qwen36-lightllm:1.0`
- 解释：远端 commit 包含大量依赖版本升级（都是 2026-07-15 训练机实机踩坑验证的），已全量 bake 进镜像。修复提交 `02eaa6e`。

### 2026-07-16 00:29 | 本机 Mac（darwin，无 GPU） | commit b760a8d ~~无效~~（见上条）
- 动作：第一次重 build——但因为本地落后远端 30 commits，Dockerfile 实际是旧版，全部 CACHED 命中，推的还是旧 digest。本次无效，见上条修正。

### 2026-07-03（晚，本机 CPU 开发机） | commit <pending>
- 动作：租户迁移收尾——把**重 build 路径**上残留的旧租户默认值也切到 `ccr-zuhu2026`。前一条（21:07）做的是 retag+push（未重 build），故 `build_and_push.sh` 与 Dockerfile 里仍指向旧租户 `ccr-devsfttj`；本次补齐：① `docker/qwen36-lightllm/build_and_push.sh` 默认 `NAMESPACE ccr-devsfttj→ccr-zuhu2026`、`USERNAME devsfttj-sunhao4→zuhu2026-sunhao4`（原 `NAMESPACE:?` 必填改为默认新租户）。② `docker/qwen36-lightllm/Dockerfile` `ARG BASE_IMAGE` 的 base 由 `ccr-devsfttj/verl:...`→`ccr-zuhu2026/verl:...`（@孙豪 确认 base 已 retag 到新租户）。③ `Migration_64GPU.md:219` 旧的"base 仍在旧租户、重 build 需确认能否跨租户拉"注记 → 改为"base 已迁新租户，重 build 直接拉"。
- 结果：✅ 全仓活配置（脚本/Dockerfile/yaml）里 `ccr-devsfttj` 已清零；剩余引用全在 doc/ 历史与迁移记录中（RunLog 禁改历史，正确保留）。新地址 `registry.cn-tj-01.sensecore.cn/ccr-zuhu2026/qwen36-lightllm:1.0` 一致落地于 build 脚本 + Dockerfile + 启动指南 + Migration。
- 产物：`docker/qwen36-lightllm/build_and_push.sh`、`docker/qwen36-lightllm/Dockerfile`、`doc/ops/Migration_64GPU.md`。
- 解释：区分两条路径——(1) **当前镜像**已在新租户（retag+push 产物，image ID `8631ec6c9b9b` 不变，可直接用）；(2) **未来重 build** 时才会读 build 脚本默认值 + Dockerfile base，本次把这条路径也对齐新租户，避免下次重 build 又落回旧租户或拉不到 base。

### 2026-07-03 21:07 | 本机 Mac（darwin，无 GPU） | commit <pending>
- 动作：训练镜像从旧租户 `ccr-devsfttj` 迁至新租户 `ccr-zuhu2026`。① `docker login registry.cn-tj-01.sensecore.cn --username zuhu2026-sunhao4`（凭证存 `~/.docker/config.json`）。② 本地 `qwen36-lightllm:1.0`（image ID `8631ec6c9b9b`，76.3GB）retag → `registry.cn-tj-01.sensecore.cn/ccr-zuhu2026/qwen36-lightllm:1.0`。③ `docker push`（未重 build，复用之前 `--provenance=false` 产物）。④ 镜像内依赖自检（`docker run --platform linux/amd64 --entrypoint python`）：泽寰栈 5 项版本对齐（transformers 5.8.0 / fla 0.4.2 / TransferQueue 0.1.6 / accelerate 1.13.0 / e2b 2.24.0）、transformers 认 qwen3_5、fla 可 import、verl 训练链路 12 依赖全 OK、lightllm 1.1.0 / megatron / flash_attn 2.7.4 / ray 2.49.1 / numpy 1.26.4 / torch 2.9.0+cu129。⑤ 修 `scripts/start_train.sh` + `run_phases.sh`：`EXPERIMENT_NAME` 从 config 动态读（不再硬编码 `qwen36_27b_b1`），日志目录 `outputs/<exp_name>/` 与 verl ckpt 目录 `checkpoints/RL/<exp_name>/` 同名、各实验独立不互相覆盖。⑥ 改 `doc/archive/集群训练启动指南.md` §3 + `doc/ops/Migration_64GPU.md` 附录 A.3 旧镜像地址 → 新租户。
- 结果：✅ push 成功（exit 0，digest `sha256:8631ec6c9b9b...`，`docker manifest inspect` 远端可查）。✅ 镜像依赖对正式训练链路（lightllm rollout）齐全。⚠️ 两点非缺依赖、设计如此：(1) verl 本体不在镜像——定制版在 `/mnt/afs` 挂载进容器、`PYTHONPATH` 指向 AFS 源码（`start_train.sh:30` 自设，不靠 `dev_env.sh`）；(2) vllm 0.13 不认 qwen3_5 架构——但正式 rollout 走 lightllm 不走 vllm，不影响训练（用户已确认不切 vllm）。
- 产物：远端镜像 `registry.cn-tj-01.sensecore.cn/ccr-zuhu2026/qwen36-lightllm:1.0`；`scripts/start_train.sh`、`scripts/run_phases.sh`（EXPERIMENT_NAME 动态化）；`doc/archive/集群训练启动指南.md`、`doc/ops/Migration_64GPU.md`（镜像地址更新）。
- 解释：旧租户 `ccr-devsfttj`（登录名 `devsfttj-sunhao4`）弃用，迁至新租户 `ccr-zuhu2026`（登录名 `zuhu2026-sunhao4`）。retag+push 不重 build：同一 registry 内跨命名空间共享 blob，多数层 `Mounted from ccr-devsfttj/...` 不重传，只 manifest 提交是新写的，未遇 `manifest invalid`（复用 `--provenance=false` 产物）。**AFS 外挂对训练速度无影响**：权重 `cp -rL` 到 `/tmp/qwen36`、HF/Triton/cache 指 `/tmp`、rollout 是 lightllm 在 GPU 上跑，热路径都不碰 AFS；外挂代价只在启动期（import verl/LightLLM + 拷权重，一次性）。verl/LightLLM 不打包进镜像——维持"改源码不用重 build"灵活性（fully_async policy 仍在迭代）。**剩余阻塞**：`configs/cluster.yaml:81` `data.train_files: ???` 未填（等 @吴健 训练数据 parquet）；提交任务时镜像填新地址 `ccr-zuhu2026/qwen36-lightllm:1.0`。

### 2026-07-03 | 新集群开发机（CPU，无 GPU） | commit <pending>
- 动作：设计并落地 **Phase 0 冷启动数据来源配比消融**（独立预实验，不进 21）。① `scripts/collect_rollout.py` 加 `meta.policy` 来源标记（`pi0_27b`/`gpt5`，`_actor_policy_tag` 按 `--actor` 映射，写进每条 trajectory + session record 顶层）。② `scripts/warmup_buffer.py` 加 `--ratio-27b` 按桶内配比混合两来源（`_mix_by_ratio`：先按 bucket×source 分组、再按比例无重复抽样，固定 `--mix-seed` 可复现）+ 输出 `*.manifest.json` 记录每桶实际 27B/gpt5 条数；来源标记优先读 trajectory `policy`、缺失时从 `{actor}` 目录推断（向后兼容旧数据）。③ 5 臂配置 `configs/phase0/p0-{a..e}.yaml`（100:0 / 0:100 / 50:50 / 70:30 / 30:70，其余锁死 R4）。④ `scripts/phase0/run.sh`（三阶段：建 buffer → 轨1 gate → 短RL）+ `scripts/phase0/gate_coldstart.py`（轨1 冷启动自身指标 gate）。⑤ 新增设计文档 `doc/source/CL_Design.md`；整改 `CL_Design.md`（加 Phase 0 节 + 路线图 + **给 Phase 1–6 各补验收标准表**）、`Buffer_冷启动数据需求.md` §5、`Progress.md`。
- 结果：⏳ 待跑单测验证（本条落地后执行 `pytest`）。设计决策（与 @孙豪 对齐）：独立定位不进 21、5 臂全扫、双轨验收（冷启动自身指标 + 下游短RL）、更强模型 = `openai/gpt-5`、短RL 固定新任务+同种子、7 桶拆旧/新两组。
- 产物：`scripts/collect_rollout.py`、`scripts/warmup_buffer.py`、`configs/phase0/*.yaml`（5）、`scripts/phase0/{run.sh,gate_coldstart.py}`、`doc/source/CL_Design.md`、`doc/source/CL_Design.md`、`doc/source/CL_Design.md`、`doc/archive/Progress.md`。
- 解释：冷启动数据来源（27B on-policy vs gpt-5 off-policy）是一个与现有 21 实验**正交的新变量**——27B 数据"对味但可能弱"、gpt-5 数据"强但可能不对味"（强 off-policy 分布偏移）。它必须在正式训练前定死，故设为 Phase 0 前置预实验。验收难点在于"冷启动数据本身无分数、好坏只在下游显形"，故用双轨判定链：自身指标先筛掉明显差的（无 GPU、采集后即测），短RL 做最终裁决（CL Score 主裁）。**红线**：gpt-5 数据只进 buffer 做 replay，绝不拿去 SFT 蒸馏 27B（违背 B2/C1 立论 + 污染 21 实验可比性）。

### 2026-07-02 ~15:30 | 新集群开发机（CPU，无 GPU，cold env py3.10） | commit <pending>
- 动作：沙箱内 hermes 采集排障 + 首次跑通。① `sandbox_grpo_collect.py --actor hermes` 初测 reward=0/ans 空 → 进沙箱逐层诊断。② 根因定位：`_write_hermes_config` 从开发机侧 `_env("AGENT_MODEL_KEY")` 读 key → 开发机侧未设该 env（key 只在 `docker/sandbox/runtime.env` 里，经 `E2BSandbox.envs=` 注入沙箱内，`load_tencent_env.sh` 未 export）→ 传空 key 给 hermes → hermes 调 sufy 没 key → TimeoutExpired。③ 修复：`_write_hermes_config` 改为沙箱内 Python 直接 `os.environ['AGENT_MODEL_KEY']`（key 在沙箱内，不经开发机进程），删 `--actor-key` 参数。④ 修后第一次真实跑通：`hermes chat -q 'Compute 23*17-19 and write the number to /home/user/result.txt'`（max_turns=6，沙箱内）→ exit 0，创建 `result.txt` 内容 `372`，hermes diff 展示 `+372`。
- 结果：✅ hermes-in-sandbox 真实跑通（沙箱内 hermes 调 sufy `openai/gpt-5` 决策 + 写文件）。沙箱镜像含 hermes v0.16.0 无误。❌ 修复前脚本采集中 hermes 全超时（两个 slot 都 TimeoutExpired）。清理：`/tmp/grpo_*` 四个测试目录。rollout 输出位置定为 `rollouts/cold_start/`（gitignored）。
- 产物：`scripts/sandbox_grpo_collect.py`（`_write_hermes_config` 改沙箱内读 key、`_run_hermes_slot` 去 key 参数、`run_session`/`main` 去 `actor_key`）+ `.gitignore`（+`rollouts/`）+ RunLog 本条。
- 解释：**API 模型（sufy）→ 训练机（本地 27B）迁移答案**：冷启动采集两路 actor 产出**同一 schema 的 trajectory**（messages + token_ids + logprobs + reward + bucket），buffer 不关心谁产的。冷启动用 hermes → sufy（路径 B），预热 buffer 到 ~10K 轨迹；训练机就位后切路径 A（verl + 本地 Qwen3.6-27B），同一份 buffer 继续消费。`doc/source/训练与推理流程.md` §2 已列两条路径。**训练镜像推错位置**：`qwen36-lightllm:1.0` 推到 `registry.cn-tj-01.sensecore.cn/ccr-devsfttj`（和沙箱镜像同一个 registry），应推商汤私有云其他命名空间——@孙豪 后续给正确位置重建。

### 2026-07-01 ~21:10 | 本机（macOS + Docker Desktop，amd64 via buildx） | commit <pending>
- 动作：build + push Qwen3.6-27B 训练镜像 `qwen36-lightllm:1.0` 到天津 SenseCore registry。base = 泽寰 `verl:cu129_lightllm_sandbox_megatron0.14.0_vllm0.13.0R3_torch2.9.0_fa3_te2.5.0_0211`（23GB，含 torch2.9/vllm0.13/megatron0.14/FA3/TE/CUDA12.9）。`docker/qwen36-lightllm/Dockerfile` 在 base 上固化 10 个运行时 pip（tensordict/accelerate1.13/transformers5.8/fla0.4.2/e2b2.24/...）+ verl 训练链路 12 依赖 + 6 项构建期自检。
- 结果：✅ build 成功（selfcheck ALL PASS），⚠️ 首次 push 失败，二次修复后 push 成功。
  - **pull base**：21GB base，公网 ~22MB/s，28 分钟拉完（19:55→20:23）。一度 143KB/s 卡死，重测恢复（瞬时抖动/aoss 限流）。
  - **build**：`NAMESPACE=ccr-devsfttj bash docker/qwen36-lightllm/build_and_push.sh` OK。产物 `qwen36-lightllm:1.0`（76.3GB）。selfcheck 6 项：泽寰栈版本对齐 ✓ / transformers 认 qwen3_5 ✓ / fla import ✓ / verl 训练链路 12 依赖（含 ray 2.49.1）import OK ✓ / numpy 1.26.4 ✓ → **[selfcheck] ALL PASS**。
  - **push 第 1 次失败**：`error from registry: manifest invalid`。根因——Docker Desktop BuildKit 默认加 attestation manifest（provenance），产物是 OCI image index，天津 SenseCore registry (v2) 不认。build 日志铁证 `#11 exporting attestation manifest sha256:cdf81395...`。层全传上去了（多数 `Mounted from ccr-devsfttj/verl` 共享 + 几个 Pushed），manifest 提交被拒。
  - **push 第 2 次修复**：`build_and_push.sh` 加 `docker build --provenance=false`（禁 attestation，产物回归单平台 v2 manifest）+ login 改 `2>/dev/null || skip`（已有凭证时不交互）。重 build（有 cache，几十秒）+ push 成功。
- 产物：`registry.cn-tj-01.sensecore.cn/ccr-devsfttj/qwen36-lightllm:1.0`（天津 registry）。
- 解释 / 踩坑（证据级）：
  - **verl 本体不在镜像里**：定制版 verl 在 `/mnt/afs` 挂载进容器（@孙豪 确认），build 期不可见。诊断层实测：`/opt/conda/bin/python`（base env）`import verl` → `ModuleNotFoundError`；ray 在 base env（2.49.1）。Dockerfile 自检**跳过 verl 本体 import**（只校验其训练链路依赖第 4 项），加注释说明。
  - **attestation manifest 是天津 registry 的坑**：BuildKit 默认开 provenance，产物成 image index；天津 v2 registry 不支持 → `manifest invalid`。`--provenance=false` 根治。同类 SenseCore registry build 都要带这个。
  - **docker login non-TTY**：原脚本 `docker login --username` 在后台跑报 `cannot perform an interactive login from a non-TTY`。改为失败时 fallback 到 `~/.docker/config.json` 已有凭证（pull base 时就存了）。
- 待集群 / 待办：
  - 集群 GPU 机器用此镜像起容器 + 挂载 `/mnt/afs`（含 verl 定制版 + Qwen3.6-27B 权重）跑训练 smoke。
  - `verl` 定制版版本核对（AFS 盘上的 verl 是否与 base 镜像的 vllm0.13/torch2.9 兼容）。

### 2026-06-30 ~21:00 | 本机（macOS + Docker Desktop） | commit <pending>
- 动作：重做沙箱镜像 v2——补 Hermes Agent（v1 缺）。(1) `docker/sandbox/Dockerfile` 加 `ARG HERMES_VERSION=v2026.6.5` + RUN 段：`git clone --depth 1 --branch v2026.6.5 https://github.com/NousResearch/hermes-agent.git /opt/hermes-agent` + `pip install --break-system-packages -e /opt/hermes-agent`（editable，系统 Python 无 venv）。(2) base 从 `agentic-cl-sandbox:v1`(自构建) 换回官方 `sandbox-code:latest`。(3) `image.env` `IMAGE_TAG` v1→v2、base 指向 `sandbox-code:latest`。(4) `configs/sandbox_tool.json` `Image` v1→v2。(5) build + push TCR + 本地容器验证。
- 结果：✅ 全程成功。
  - **TCR login**（关键坑已破）：`tccli tcr CreateInstanceToken --region ap-beijing --RegistryId tcr-hxya4oi8 --cli-unfold-argument` 返回 `Username=100049693643` + `Token`（1h 有效）→ `echo $Token | docker login tcr-rl.tencentcloudcr.com -u 100049693643 --password-stdin` → Login Succeeded。**Username 必须用返回值，不是 'admin'/腾讯云账号 ID**（之前用 admin/sunhao4 均 `unauthorized`）。
  - **pull base**：`docker pull tcr-rl.tencentcloudcr.com/agentos-cl-namespace/sandbox-code:latest` OK（digest `sha256:21b06711...003a`）。
  - **build**：`bash scripts/build_sandbox_image.sh` OK，产物 `tcr-rl.tencentcloudcr.com/agentos-cl-namespace/agentic-cl-sandbox:v2`（8.56GB，amd64 digest `sha256:bd860870...d4f7`）。Hermes 装入证据：build 日志 `Collecting ... (from hermes-agent==0.16.0)`、本地 `docker run` 起 v2 容器 `hermes --version` → **"Hermes Agent v0.16.0 (2026.6.5)"**。
  - **push**：`bash scripts/push_sandbox_image.sh` OK，12 layer 全 Pushed，TCR 侧 `docker manifest inspect` 复核 amd64 digest 一致。
  - **反证 v1 缺 Hermes**：起 v1 实例（Tool `sdt-eya9hqzm` 仍指 v1）`hermes --version` → `command not found`、`import hermes_agent` 报错——证实 v2 的 Hermes 补齐是必要的。
- 产物：v2 镜像已上 TCR（`tcr-rl.tencentcloudcr.com/agentos-cl-namespace/agentic-cl-sandbox:v2`）；`docker/sandbox/Dockerfile`(+Hermes 段)、`docker/sandbox/image.env`(tag v2)、`configs/sandbox_tool.json`(Image v2)。
- 解释 / 踩坑：
  - **Hermes 装 editable（`pip -e`）非 wheel**：与 `docker/verl-hermes-remote-agent/Dockerfile.tencent-ags` 同 recipe，源码留 `/opt/hermes-agent` 便于实例内调试/打补丁；`--break-system-packages` 因 AGS base 用系统 Python（无 venv，snapshot 镜像跑 root）。
  - **`hermes_agent` 模块 import 报错但 `hermes` CLI 可用**：`pip -e` 装的是包名 `hermes-agent`（PyPI 名），console script `hermes` 正常；`import hermes_agent` 失败是因包的 import 名可能不是 `hermes_agent`（待集群实例内 `python3 -c "import hermes; ..."` 核对真实 import 名），不影响训练链路（训练走 `hermes` CLI / verl hermes runner，不直接 import）。
  - **/execute 仍 500**（v1 已知，v2 沿用）：jupyter-server 监听 8888 ≠ 49999，腾讯 base 镜像没按 E2B 方式把 kernel gateway 暴露到 49999。主链路 `commands.run`（49983 envd）通，`/execute` 是附加险不修。
- 待集群 / 待办：
  - 用 v2 重建 Tool（`bash scripts/create_sandbox_via_api.sh custom`，新 Tool 或更新 `sdt-eya9hqzm`）+ 起 v2 实例验证 `hermes --version` + `hermes` 跑通 spawn 同步轨迹。
  - `hermes_agent` 真实 import 名核对（若训练代码直接 import）。
  - TCR Token 1h 过期，集群侧 push 需重新 `CreateInstanceToken`。

### 2026-06-30 ~20:50 | 新集群开发机（CPU，无 GPU，无 docker） | commit 654569d
- 动作：沙箱镜像补 **Hermes Agent** + 基础依赖。查清 hermes 来源：`~/.hermes/hermes-agent` 是 `git@github.com:NousResearch/hermes-agent.git`（已 clone，PyPI 包名 `hermes-agent`，`[project.scripts] hermes=hermes_cli.main:main`，`requires-python >=3.11,<3.14`）；@郑乃榕 的 `verl-hermes-remote-agent` 镜像（Tool `node-python-hermes` `sdt-3o6mpbok` 用的 `tcr-rl.tencentcloudcr.com/rl/verl-hermes-remote-agent:26.6.22`）就是从同 repo `git clone --branch v2026.6.5` + `pip install -e` 装的（见 `Documents/verl/docker/verl-hermes-remote-agent/Dockerfile.tencent-ags`）。决策：OpenClaw + Hermes **两个都装**（文档说训练只用 Hermes，但暂保留 OpenClaw）。改 `docker/sandbox/Dockerfile` + `requirements.txt`，待去有 docker 的机器 build+push。
- 结果：✅ Dockerfile 加 §3b Hermes 安装段（`git clone --depth 1 --branch v2026.6.5` NousResearch/hermes-agent → `/opt/hermes-agent` → `pip install -e --break-system-packages`，镜像系统 Python 无 venv 故用 break-system-packages）；✅ requirements.txt 补 `openai>=1.40` + `fastapi>=0.110` + `uvicorn>=0.30`（修 v1 镜像缺 openai/fastapi）；✅ `validate_sandbox_dockerfile.sh` 通过（无 USER/WORKDIR/ENV/ENTRYPOINT，snapshot 兼容）。⚠️ 本机无 docker，**未 build 验证**——`HERMES_VERSION=v2026.6.5` 是从 verl 镜像抄的 tag，需 build 时确认该 tag 在 NousResearch repo 存在；`pip -e` 装的是 hermes 核心 deps（openai/certifi/httpx/pydantic/jinja2 等 exact-pin），可能与我们 requirements.txt 的 openai 版本冲突（hermes pin `openai==2.24.0`，我们写 `>=1.40`），build 时若解析冲突需对齐。
- 产物：`docker/sandbox/Dockerfile`(+§3b Hermes 段)、`docker/sandbox/requirements.txt`(+openai/fastapi/uvicorn)。
- 解释：镜像层 Hermes 就位（代码），但**未 build**——本机无 docker。下一步：① 去有 docker 的机器 `bash scripts/build_sandbox_image.sh` + `push_sandbox_image.sh`，build 前 `docker login tcr-rl.tencentcloudcr.com`（`tccli tcr CreateInstanceToken --RegistryId tcr-hxya4oi8` 拿临时 token）；② build 时盯 hermes pip 解析是否与 requirements openai 版本打架；③ push 后用新镜像起实例验 `which hermes` + `hermes --version` + `which openclaw`。

### 2026-06-30 ~20:10 | 新集群开发机（CPU，无 GPU，cold env py3.10） | commit 654569d
- 动作：沙箱**双向通信验证** + **沙箱内依赖/Agent 盘点**。① 服务器起 TCP listener(:46653)，沙箱主动连回发 `hello-from-sandbox`，服务器收 `ACK-from-server`。② 探查 v1 镜像实际装了什么。
- 结果：✅ **双向通**——服务器→沙箱(`commands.run` 跑代码)、沙箱→服务器(反向 TCP 连接成功)、沙箱→公网(DNS 解析 `www.baidu.com`→`220.181.111.232`)全通。网络拓扑：沙箱出口 `10.13.64.9`(在 `agent-rl-vpc` 的 `subnet-ljfuh7ln` 内) ↔ 服务器内网 `10.120.2.226`，同 VPC 打通。**之前"只能服务器→沙箱、沙箱回不来"的旧集群问题已解决**(新集群 Tool 绑 `agent-rl-vpc` 后反向链路自然通)。
- 结果(沙箱依赖盘点)：✅ Python 3.12.13、Node v24.18.0、npm 11.16.0、201 个 pip 包(pandas/openpyxl/python-docx/pdfplumber/aiohttp/numpy/pydantic/httpx 齐)；✅ **OpenClaw 2026.6.10** 已装(`openclaw --version` 正常，`which openclaw`=/usr/bin/openclaw，npm 全局包)；✅ **agent harness 脚本就位**：`agent_entry.sh`/`seed_workspace.sh`/`fs-seeds/` 都在 `/opt/agentic-cl/`(非 Dockerfile 里写的 `/app/`)。⚠️ **缺 `openai` 和 `fastapi`** 两个包；⚠️ **没找到 `hermes` 可执行**(`which hermes` 空)——只装了 `openclaw`，没装 hermes；⚠️ runtime env 没注入(`OPENCLAW_MODEL`/`AGENTIC_CL_*` 全空，说明起实例时没传 `envVars` 或 `runtime.env` 没配)。
- 产物：无文件改动(仅探查)；沙箱镜像内 `/opt/agentic-cl/` 是项目 agent 层落点。
- 解释：沙箱**通信全通**。Agent 层**部分就位**——OpenClaw + harness 脚本在，但缺 `hermes`、缺 `openai`/`fastapi` 两个包、runtime env 没注入。下一步：① 确认是否需要 `hermes`(若 hermes 是另一个 agent CLI 则镜像要补装)；② 补 `openai`/`fastapi` 到 `docker/sandbox/requirements.txt` 重建镜像；③ 配 `runtime.env`(OPENCLAW_MODEL/TOKENHUB) 让实例带上推理 endpoint。

### 2026-06-30 ~19:25 | 新集群开发机（CPU，无 GPU，cold env py3.10） | commit 654569d
- 动作：迁入新集群后**新 VPC 沙箱创建+使用全链路冒烟**。① `cold` env 装 `e2b-code-interpreter 2.8.1 + e2b 2.30.0 + tccli 3.1.118.1`（系统 pip 坏、走代理 `http://10.119.176.202:3128` 才下得动 PyPI）。② `tccli ags DescribeSandboxToolList` 确认账号下 20 个 Tool ACTIVE，含本项目 `agentic-cl-sandbox`（ToolId `sdt-eya9hqzm`，4C/8Gi/20Gi、企业版镜像 `tcr-rl.tencentcloudcr.com/agentos-cl-namespace/agentic-cl-sandbox:v1`、VPC `subnet-ljfuh7ln`/`sg-irz5h5ed`、`/init`、envd:49983）。③ 起 VPC 实例 + `commands.run` 跑 `print((23*17)-19)`。
- 结果：✅ 实例连通 6.4s、`stdout='372'` `ok=True`、`kill()` 正常（total 6.8s）；冒烟后 `DescribeSandboxInstanceList` 无残留 `agentic-cl` 实例（已回收）。VPC 网络 + 企业版 TCR 镜像 + RoleArn(`AgentOS-260506-test`) 全链路打通。
- ⚠️ **e2b SDK 前缀校验冲突**：`tencent.env` 里 `E2B_API_KEY=ark_...`（AGS 发的密钥前缀是 `ark_`），但 `e2b_code_interpreter 2.8.1` 的 `validate_api_key` 硬要求 `e2b_` 前缀（正则 `\Ae2b_[0-9a-f]+\Z`），否则 `AuthenticationException`。**根因**：腾讯 AGS 发 `ark_` 前缀的 E2B 兼容 key，与上游 e2b SDK 的前缀校验不一致。**解决**：用 SDK 自带的官方开关 `E2B_VALIDATE_API_KEY=false`（`e2b/connection_config.py:81` 读这个 env，false 时跳过前缀正则校验，**非 monkeypatch**）—— 不换 key、不动 SDK，`ark_` key 直接可用。已固化到 `scripts/load_tencent_env.sh`（train.sh 等都 source 它，rollout worker 自动带上）。`/execute`(49999) 仍不可用、`commands.run`(49983) 正常——与 2026-06-26 踩坑记录一致。
- 产物：`scripts/load_tencent_env.sh`(+`E2B_VALIDATE_API_KEY=false` + 注释)；凭证已就位（`docker/sandbox/tencent.env`）；`tccli` 走 `cold` env(symlink 到 `~/.local/bin`)，CAM 凭证已写入 `~/.tccli/default.credential`。
- 解释：新集群沙箱后端**功能可用 + 鉴权已通**。下一步：① 用 `sandbox_smoke.py --backend e2b -m 2` 跑完整 GRPO 组冒烟；② 8 槽 winner-sync 路径 + observer diff 取证在真实后端验证。

### 2026-06-25 | 本机（开发机，CPU，无 GPU） | commit <pending>
- 动作：数据源切换 + 沙箱/数据方案定稿（设计变更，未跑训练）。(1) **数据源从 sample105_v2（OpenClaw 采集，已弃）切到 `data/taskspecs/`**：每个 task = 1 份声明 `taskspec.yaml`（seed_query/hidden_goal/verifier 判分 rubric/user_profile/available_tools…）+ 1 份初始文件系统 `files/`。workspace↔query 回到 **1:1**（1 task=1 files=1 seed_query），原"1 沙箱↔N 会话"1:n 方案作废。(2) **沙箱 Dockerfile 方案定稿**（`doc/ops/sandbox/沙箱_Dockerfile制作方案.md`）：1 母版镜像 `COPY` 全部 seed（`files/`→`fs-seeds/<task_id>/`），实例启动按 `AGENTIC_CL_PERSONA=<task_id>` 铺开；1 seed→fork 8 容器跑同一 seed_query（GRPO 8 路、位级一致）；母版只装常用依赖、特定依赖 agent 运行时自己装（贴合真实场景）；当前简化版 1 母版，后续多母版/环境扰动留方向。(3) **`data_pipeline/` 1:1→1:n 留空回退**（`extract_initial_queries` 抛 NotImplementedError，待按 taskspec 重写）；`route.py` 撤多余 `first_query_only` 标记。(4) 文档/论文同步：Hermes subagent 方案 §6.6 标 sample105 弃用、`沙箱_实例_Queries对应关系_待定.md` 待定项清理（N/K/schema 作废，仅剩提问上限/结束其余条件/judge 校准）、Method 中英 §4.5 + 论文向总览 数据归属段改为 taskspec。
- 结果：✅ 设计文档定稿；全量 `pytest -q` 306 passed / 7 skipped；ruff 全过。❌ 未跑训练/镜像/rollout（待 §3 自动化脚本 + build + rollout）。
- 产物：`doc/ops/sandbox/沙箱_Dockerfile制作方案.md`（新）、`doc/ops/sandbox/沙箱_实例_Queries对应关系_待定.md`、`doc/source/Hermes_Subagent_训练数据方案.md`、`data_pipeline/extract.py`（留空）、`paper/drafts/{Paper_Method_draft_CN,EN,Paper_论文向总览}.md`、本条记录。
- 解释：数据结构定稿是后续镜像制作 + rollout 采样的前提。taskspec 自带 verifier rubric 可直接当 reward judge 标准、user_profile 驱动 Questioner，比 sample105 干净。下一步：写 taskspec→fs-seeds 自动化脚本 → build 母版镜像 → 跑实例 → rollout 采样 → 据产出轨迹定 rollout 契约。

### 2026-06-23 | 本机（开发机，CPU，无 GPU） | commit <pending>
- 动作：检查并继续优化——(1) **修 3 个失败配置测试**：`tests/test_configs.py` 的 `EXPERIMENT_CONFIGS` glob (`phase*/*.yaml`) 把 2026-06-17 加入的 `configs/phase1/smoke_1step.yaml` 当成第 22 个正式实验，导致 `test_found_all_experiment_configs`(期望 21) + 两个 `smoke_1step` 参数化用例（pin 了集群绝对路径的 verl `_generated_ppo_trainer.yaml`，本机不存在）失败。smoke 配置自述"不是正式实验配置"，故按 `smoke*` stem 前缀从实验花名册排除（非把计数改成 22）。(2) **ruff 核心库 14 项**：autofix F401/I001/UP035（含 `replay_metrics.py` 死 `import torch`、`verl_runner.py` 未用 import），手动给 6 处 `zip()` 加 `strict=`——5 处等长配对用 `strict=True`（bucket names↔targets / signal names↔alpha / samples↔token_weights / prompt↔resp rows / tids↔means，长度失配应暴露 bug），`trajectory_adapter.py` 的 ids↔mask 用 `strict=False`（padding 下可不等长，保留重叠而非崩溃，附注释）。(3) **lint 门禁可用**：1213/1248 个 ruff 报错全集中在 `bin/{detact,clean_zerowidth}.py` 两个 tab 缩进的一次性工具——给 ruff+black 都加 `bin/` exclude，`ruff check .` 从 1248 错→全过。(4) **black 仓库级格式化**：54 文件 reformat（line-length=110，bin/ 已排除）。
- 结果：✅ `ruff check .` 全过、`black --check .` 全过（89 文件）、全量 `pytest -q` **288 passed / 7 skipped**（先前 3 failed 已修；7 skip 全为 verl/CUDA 门控，本机无）。
- 产物：`tests/test_configs.py`（smoke 排除）、`replay_buffer/{bucket,priority}.py`、`trainer/{replay_forward,replay_metrics,trajectory_adapter,verl_runner,verl_async_runner}.py`、`rollout/{sandbox_client,usersim_collect}.py`（ruff）、`pyproject.toml`（ruff+black exclude bin/）、54 文件 black reformat、本条记录。
- 解释：前一条记录里"3 failed 为既有环境问题、与本次无关"的判断成立（确为先前 commit 引入的 test/config 漂移），本次将其修掉——测试花名册应只含正式实验，smoke 探针不该混入。`zip(strict=)` 是 Python 3.10 起的静默截断防护，与本仓"反静默退化"基调一致。`bin/` 排除后 `ruff check .` 重新成为有效门禁。reformat 不改语义、测试不变绿。全栈 GPU / 真实 e2b / Phase4-5 `???` 仍 Blocked on 集群。

### 2026-06-23 | 本机（开发机，CPU，无 GPU） | commit <pending>
- 动作：两件待办收尾——(1) **thinking 模型截断防护**：模型回复因 `finish_reason=length*`（或思考预算吃光 token、content 空且无 tool_calls）被截断时，统一在 `agents/base.py::_raise_if_truncated` 抛 `TruncatedOutputError`，三 Agent + judge 分别按各自语义处理：questioner→标 `last_query_was_error=True`（走 patience/telemetry，不误判为满意 `<end_session>`）；observer→降级到确定性取证报告（不解析半截 JSON）；judge(`model_reward.py`)→`parse_judge_output` 返回 `(verdict, parsed)`，未解析出 verdict JSON 时抛错路由到 `compute_score` 的 except→`judge_error=1.0`（不再静默全 0 reward）；questioner `max_tokens` 256→512。(2) **B1/R4 方法学一致性**：`configs/run/b1.yaml` 删除 `rollout.agent: null`，B1 改为与 R4 共用同一套 agentic 多轮 rollout（差异仅 CL 项：B1 关 replay/无 KL），同步改 `doc/ops/Migration_64GPU.md`。
- 结果：✅ `tests/test_agents.py` + `tests/test_model_reward.py` **53 passed**（含新增 `test_questioner_truncation_is_error_not_end_session` + `parse_judge_output` 双返回值断言）；全量 `pytest -q` **287 passed / 7 skipped / 3 failed**。3 failed 全为**既有环境问题、与本次改动无关**：`test_found_all_experiment_configs` 期望 21 实得 22（committed 的 `configs/phase1/smoke_1step.yaml` 未同步计数，先前 commit 引入）、两个 `smoke_1step` 用例缺 verl `_generated_ppo_trainer.yaml`（本机无 verl）。本机无 ruff/mypy（venv 轻量包），未做仓库级 reformat（避免改动未触碰的历史行）。
- 产物：`agents/{base,observer,questioner}.py`、`trainer/model_reward.py`、`tests/{test_agents,test_model_reward}.py`、`configs/run/b1.yaml`、`doc/{ops/Migration_64GPU,log/Progress}.md`、本条记录。
- 解释：截断防护是**反 reward-hacking / 反静默退化**的工程加固——之前截断的回复（思考模型常见）会被当成"完整答案/满意/全 0 reward"消费，错误不可见；现在统一抛错、各 Agent 显式分流，失败在日志可见。B1/R4 同 rollout 是方法学正确性：拿"多轮执行完成"的 R4 与"单轮"B1 比遗忘不公平，遗忘基线必须同等 rollout 下测。全栈 GPU smoke / 真实 e2b 后端 / Phase 4-5 `???` 参数仍 Blocked on 集群（TODO #4）。

### 2026-06-23 | 本机（开发机，CPU，无 GPU） | commit <pending>
- 动作：收尾 `0622 待办计划.md` 的本机可完成项 + 清理 lint。修 7 个 ruff 错（observer.py 的 E702/F541、verify_endpoints.py 的 F401/F541/E402、verify_binary_extraction.py 的 F401）；给两个 verify 脚本补可执行位；`agents.yaml` yaml 校验通过；CLAUDE.md TODO #3（SWANLAB 硬编码）核对已落地并补标 ✅。
- 结果：✅ 变更文件 `ruff check` 全过；`.venv/bin/python -m pytest` 在本机可跑的 28 个测试文件 **214 passed / 11 skipped**（skip = 缺 torch/omegaconf/verl，需集群环境，与本机一致）；新增配置模块 `agents/config.py` + `configs/agents.yaml`（模型选型单一信源）由 27 个单测覆盖（RotatingChatClient / parse_endpoints / resolve_* / validate_endpoints_distinct / validate_model_distinctness）全过。❌ `tests/test_configs.py`、`test_warmup_preload.py` 因 venv 缺 omegaconf 无法在本机 collect（非本次改动引入，属环境缺包，集群 `pip install -e .[dev]` 后即恢复）。
- 产物：`agents/config.py`、`configs/agents.yaml`、`scripts/{verify_endpoints,verify_binary_extraction}.py`、`agents/observer.py`(lint)、`CLAUDE.md` TODO 标记、本条记录。
- 解释：本机 venv 仅含 pytest/httpx/yaml 等轻量包，torch/verl/omegaconf 需集群装——故能跑的纯逻辑测试（agents/replay_buffer/domain_tagging/adapter/metrics/collect/cleaning/sandbox_client）全过即证明本次改动无回归。剩余阻塞全在集群：Phase 4/5 的 `???` 参数待 Phase 2/3 结果（TODO #4）、全栈 GPU smoke、verl Hydra defaults 补全。无新增硬编码密钥（grep `GDGemFX7` 0 命中）。

### 2026-06-19 03:38 | 本机（在家 macOS，无 E2B/GPU） | commit <pending>
- 动作：理清 observer/reward 的 trajectory 边界，落定**双通道 + pass-through**。observer **模型只看 state（diff）**，永不收 trajectory（不浪费 LLM token）；trajectory 由 observer **组件**捎带为 `ObservationReport.actor_trajectory`（pass-through，不进 observer prompt），reward 从这一份 R_t 读 trajectory 判 safety/robustness、读 state_diff 判 completion。schema 把 `actor_claims` 改名为 `actor_trajectory`（语义=原始轨迹、非"声称"）；`observe(sandbox, *, actor_trajectory=, baseline=, post=)`；`score_followup(query, report, judge)` 不再单独传 trajectory；`build_reward_judge_input` 从 `report.actor_trajectory` 取（封顶）；`_report_block`(questioner) 去掉 actor_claims；`build_observer_prompt`/`OBSERVER_SYSTEM` 状态化（不提 trajectory/claims，输出键去掉 actor_claims）。
- 结果：✅ 端到端验证——observer LLM prompt 不含 trajectory（SECRET_TRAJ_TOKEN 不泄漏）、报告 pass-through 携带、reward 从报告同时拿到 trajectory+state_diff；`test_agents`/`test_simulated_session`/`test_sandbox_client` 可跑用例全过（修了一批残留 `actor_claims` 引用：`_report_block` 生产崩溃点 + 5 处测试构造/断言 + harness mock）；`ReadLints` 无错；harness 两模式跑通。
- 产物：`agents/{schema,observer,prompts,reward}.py`、`rollout/{simulated_session,usersim_collect}.py`、`scripts/agents_harness.py`、`tests/test_agents.py`、`doc/ops/sandbox/接口使用_Sandbox与三Agent.md` §0/§2/§4。
- 解释：演进——先"reward 不给 trajectory"→"observer 也不给"→澄清为"observer **模型**不给（省 token），但 observer **组件**可 pass-through 给 reward"。最终：observer 纯状态取证，trajectory 走一条不经 observer 模型的旁路通道到 reward。**残留待同步（非代码）**：`paper/latex/sections/A_prompts.tex` O6 prompt、`Paper_Method_draft_{EN,CN}.md` §4.5+附录、`doc/source/UserSim_*` 设计信源仍是旧"observer 读 claims 比对"叙事。

### 2026-06-19 03:08 | 本机（在家 macOS，无 E2B/GPU） | commit <pending>
- 动作：reward 路径两项收口。(1) **不再给 judge trajectory**——只判当前状态：`build_reward_judge_input` 去掉 trajectory（返回空），judge 以 `state_diff` 为 ground truth；`build_judge_prompt` 空 trajectory 时跳过该段（path A 训练主轨迹不受影响仍带 solution_str）；REWARD_RUBRIC 措辞改为"按当前状态/diff 判，非叙事"；rubric 不再重复结构化报告块（diff 已含内容+SysOps），diff 封顶。(2) **几层拦截**：`ObservationReport.has_effect`（observer 空 diff 时置 False）→ `score_followup` 短路返回 score 0、**不调 judge**（带 gated）；`is_empty()` 改为 diff-aware（has_effect=False 即空，触发失败/耐心路径）。
- 结果：✅ 本机验证——reward 不传轨迹（judge.last_traj==""）、diff+SysOps+discrepancies 进 rubric；空 diff gate 不调 judge（score 0/gated）；build_judge_prompt 空轨迹跳段、非空保留；is_empty diff-aware；`test_agents` 15 + `test_simulated_session` 4 全过（mock agent 改为真写沙箱使 diff 非空）；path A `compute_score` 不受影响；`ReadLints` 无错。
- 产物：`agents/{reward.py,prompts.py,schema.py}`、`trainer/model_reward.py`、`rollout/simulated_session.py`、`tests/{test_agents,test_simulated_session}.py`、`doc/ops/sandbox/接口使用_Sandbox与三Agent.md` §2.5/§3.4。
- 解释：reward 应落在**真实状态**而非 actor 叙事上——给 trajectory 会把 diff-driven 想绕开的"声称"又带回来、且是最大长度膨胀源。拦截层把"没产生效果的轮"在调 judge 前就短路，省掉昂贵 round-trip 并天然给 0 分。

### 2026-06-19 02:51 | 本机（在家 macOS，无 E2B/GPU） | commit <pending>
- 动作：observer 取证层增强 (a)+(b)+LLM 可选。(a) 二进制内容提取——快照对二进制只标记，diff 后只对**本轮变更的** xlsx/docx/pptx/pdf 跑沙箱内提取（openpyxl/python-docx/python-pptx/pdfplumber）→ 文本进 diff（`kind=binary→text`），缺库/解析失败降级不崩。(b) `snapshot_system`/`diff_system`——本轮装的包/开的端口(LISTEN)/起的进程（不采 env 值，防泄密）。LLM 可选：`Observer(use_llm=False)` 默认确定性建报告零模型调用，确定性取证层始终运行。snapshot 改 `{fs,sys}` bundle，drivers 透传。harness 加 `--use-llm`（默认确定性）。
- 结果：✅ 本机验证——探针均可编译；默认确定性报告 final 从 diff 填（report.csv 内容入 diff）；二进制变更进 final 且 fallback 不崩；diff_system 正确（pandas/8000/nginx）；空 fs+sys diff→最小报告无 LLM；`use_llm=True` 路径完好；3 个 driver 回归（simulated/patience/usersim）通过；`test_agents` 观察用例已切 use_llm=True + 新增确定性用例；`ReadLints` 无错。harness 默认确定性跑通。
- 产物：`agents/observer.py`、`scripts/agents_harness.py`、`tests/test_agents.py`、`doc/ops/sandbox/接口使用_Sandbox与三Agent.md` §3.4、`CLAUDE.md` TODO#5。
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
- 产物：`agents/{observer.py,prompts.py,schema.py}`、`rollout/{sandbox_client.py,simulated_session.py,usersim_collect.py}`、`scripts/agents_harness.py`、`doc/ops/sandbox/接口使用_Sandbox与三Agent.md` §3、`CLAUDE.md` TODO#5。
- 解释：为何改——(1) 模型会**幻觉**，声称可造假；(2) 声称只报结果、**丢中间产物**（设计最看重的中间态易被覆盖）；(3) reward 落在声称上=**可被 reward-hack**；(4) diff 由代码**确定性**算，不在证据里引入第二层模型误差；(5) diff 暴露 actor **没提的改动**（静默/部分失败）；(6) 能**内容级核对**"声称值≠实际值"。剩余：二进制格式解析、`watch_dir` 抓瞬态中间产物、真实 e2b/aliyun 连通待集群。

### 2026-06-19 01:50 | 本机（在家 macOS，无 E2B/GPU） | commit <pending>
- 动作：新增多 Agent **离线 harness** `scripts/agents_harness.py`（驱动真实 session driver + 真实 observer/questioner/reward，默认 mock endpoint、`--real` 可切，`--mode simulated|collect`、`--backend local|e2b|aliyun`）；新增接口使用指南 `doc/ops/sandbox/接口使用_Sandbox与三Agent.md`；`Progress.md` 变更日志补今日条目（含协作方改动）。
- 结果：✅ harness 两模式本机跑通——simulated（8 槽：observer 报告 + questioner 出题 + reward 0.82，16 traj，ended_by=k_budget）、collect（1 槽：observer+questioner，ended_by=end_session）；`ReadLints` 无错。
- 产物：`scripts/agents_harness.py`、`doc/ops/sandbox/接口使用_Sandbox与三Agent.md`、`doc/archive/Progress.md`。
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
- 产物：`/mnt/afs_toolcall/sunhao4/envs/vllm019_venv/`（新 venv），`doc/archive/vllm_upgrade_0.19.md`（技术文档，已归档到 `paper/refs/`）
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
- 解释：链路确认工作。待办：① 起本地 vllm 跑 local actor + e2b 真沙箱全量；② `collect_rollout.py` 加分片（8 机并行不重复）。详见 `doc/ops/sandbox/ColdRollout_采集.md`。

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
- 动作：实现 UserSim 三 agent（doc/source/UserSim_多轮Query在线生成.md §7 契约，原"暂不实现/prompt 留空"）。新建 `agents/`（observer/questioner/reward/personas/prompts/schema/base）+ `inference/`（generate 边界封装）；在 `rollout/simulated_session.py` 落地 Algorithm 1（`run_simulated_session`，复用现有 `SessionSandboxPool`，不改其代码）。填三个 prompt：O6 Observer（客观无人设）、O3 Questioner（16 人设、防 AI 腔、`<end_session>`）、O4 Reward（observation-grounded、抗 hacking）。Reward 复用 `trainer/model_reward.JudgeClient`，零改动 judge I/O，只加 rubric。
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
- 结果：✅ `docker/sandbox/Dockerfile` + `requirements.txt` + `image.env.example`；`scripts/{validate,build,push,create_sandbox_tool}.sh`；`configs/sandbox_tool.json`（49999/49983 端口、探针、2C/2Gi）；`doc/ops/sandbox/Sandbox_Image_Onboarding.md`。`validate_sandbox_dockerfile.sh` 通过；`test_sandbox_dockerfile.py` 3 passed。
- 解释：Dockerfile 仅 `RUN pip install`，无 USER/WORKDIR/ENV/ENTRYPOINT，满足快照约束。真正 `docker build` 需账号登录 CCR 后 `docker pull sandbox-code:latest`。无账号阶段继续 `--backend local` 跑采样链路。


### 2026-06-10 ~16:04 | 单卡 H800（开发机） | commit <pending>
- 动作：验证沙盒能否执行+采样。探测真实沙盒可用性；按 doc §7 建 `SandboxClient` 适配层（local/e2b 双后端）+ GRPO group 采样；写 `scripts/sandbox_smoke.py` 跑通"execute + M 采样 + winner固化 + 领域→bucket"。
- 结果：❌ **真实沙盒起不来**：无 e2b SDK、无 E2B_API_KEY/E2B_DOMAIN 凭证、`tencentags.com:443` 网络超时（外网不通）。✅ **本地后端跑通**：M=6 组里 2 条 emit buggy code(obs=410,reward0,adv-1.41) vs 正确(reward1,adv+0.71)，winner固化按 tid 字典序选定，domain=Finance→bucket。`test_sandbox_client.py` 10 passed；全量 170 passed/1 skipped。
- 产物：`rollout/sandbox_client.py`、`rollout/__init__.py`、`scripts/sandbox_smoke.py`、`tests/test_sandbox_client.py`。cookbook 参考 `/root/workspace/ags-cookbook/examples/mini-rl/main.py`。
- 解释：真实沙盒须在有 SDK+凭证+网络的机器（64 卡集群或联网开发机）`--backend e2b` 跑，循环代码完全一致一行切换。本机用 local 子进程后端等价验证了"执行+采样"链路，与厂商解耦（§7 风险缓解）。


### 2026-06-26 ~21:00 | 本机（macOS） | commit <pending>
- 动作：排查沙箱实例规格"不生效"问题 + 找到真实规格获取方式。
- 背景：Tool `sdt-81jenxfq` 配置 `Resources={4CPU, 8Gi, 20Gi}`，但 E2B 兼容 API `GET /sandboxes` 返回 `cpuCount=2/memoryMB=1024/diskSizeMB=1024`，一度以为规格没生效。
- 结果：✅ **规格其实生效了**，E2B API 字段是假值。
  - **问题原因**：`rollout/sandbox_client.py` 用的 E2B 兼容 API（`api.ap-beijing.tencentags.com`）返回的 `cpuCount/memoryMB/diskSizeMB` 是**固定占位值（永远 2/1024/1024）**，不反映实例真实分配。这是 E2B 兼容层的 bug/限制——它只返回 E2B 默认规格，没同步腾讯侧真实资源。
  - **三方对比**（实例 `qgmr3yskxfb7pnkjidt35uixzujy223m2np2gv2m`，Tool `sdt-81jenxfq`）：
    | 来源 | CPU | 内存 | 磁盘 |
    |---|---|---|---|
    | E2B API `GET /sandboxes` | 2 | 1024 MB | 1024 MB（**假值**）|
    | Tool `CustomConfiguration.Resources` | 4 | 8 Gi | 20 Gi（声明值）|
    | **envd `/metrics`（真实）** | 5 | 8.0 GiB | 20.7 GiB（**铁证**）|
  - envd `/metrics` 返回 `mem_total=8438513664`（8.0GiB）、`disk_total=22202957824`（20.7GiB），与 Tool 配置一致。CPU 显示 5 是宿主逻辑核可见数（cgroup 限额是 CPU 配额，非核数可见性），正常。
- **怎么获取真实规格**（关键）：
  ```bash
  # 起实例后，用官方 SDK 拿 Token（AcquireSandboxInstanceToken）
  curl -H "X-Access-Token: <Token>" \
    https://49983-<InstanceId>.ap-beijing.tencentags.com/metrics
  # 返回 {cpu_count, mem_total, mem_used, disk_total, disk_used, cpu_used_pct, ts}
  ```
  - 端口 **49983**（envd），不是 49999
  - Token 用官方 SDK `AcquireSandboxInstanceToken` 的 `Token`（管控面），不是 `TrafficToken`，不是 E2B API 的 `envdAccessToken`
  - `/metrics` 是 envd 内置接口，返回内核级真实资源（最可靠）
- 解释：
  - 腾云工作人员后台看 Tool 是 4C8G（看的是 `CustomConfiguration.Resources`），跟 envd `/metrics` 一致——**规格一直生效，只是 E2B API 字段骗人**。
  - `rollout/sandbox_client.py` 的 `E2BSandbox` 起实例走 E2B 兼容 API、run_code 走 `49999-{sid}/execute` —— 端口(49999)、Token(envdAccessToken)、路径(/execute) 全错，导致 run_code 500/404。需按官方 SDK 重写（B 阶段）。
- 待办：run_code 正确路径仍在探测（49983 端口 `/execute` 404，要找 envd 0.2.10 的正确代码执行路径）。


### 2026-06-25 ~22:50 | 本机（macOS + Docker Desktop） | commit <pending>
- 动作：企业版 TCR 全链路打通——build → push → 造 custom Tool → 起 RUNNING 实例。
- 结果：✅ 全程成功。
  - TCR 实例 `tcr-rl`（`tcr-hxya4oi8`，公网 `tcr-rl.tencentcloudcr.com`）下新建命名空间 `agentos-cl-sandbox`。
  - `docker login tcr-rl.tencentcloudcr.com`（用 `tccli tcr CreateInstanceToken` 拿临时 Token，1h 有效）→ Login Succeeded。
  - `bash scripts/build_sandbox_image.sh` → OK，tag `tcr-rl.tencentcloudcr.com/agentos-cl-sandbox/agentic-cl-sandbox:v1`（2GB，digest `sha256:0906eebb...aec92`）。
  - `bash scripts/push_sandbox_image.sh` → OK，28 layer 全 Pushed，TCR 侧 `DescribeImages` 复核 digest 一致。
  - `bash scripts/create_sandbox_via_api.sh custom` → 建 Tool `sdt-f4ygdu0a`（ToolName `agentic-cl-sandbox`）+ E2B API Key `ark_9327...`（已提示抄进 tencent.env）+ 起 Instance `a7dptvsoikpf2...` 状态 **RUNNING**。
- 产物：镜像已上 TCR；Tool 已注册；测试实例 RUNNING（10 分钟超时自停）。`configs/sandbox_tool.json` 补实测必需字段（见下）。
- 解释 / 踩坑（证据级）：
  - **个人版 CCR 不可用**：`docker login ccr.ccs.tencentyun.com` 报 `unauthorized: no scope specify`，账号侧个人版访问凭证未配通；改走企业版 TCR 一次通。本项目已统一企业版，个人版弃用。
  - **`sandbox_tool.json` 缺字段致 CreateSandboxTool 失败**：远程版缺 `CustomConfiguration.Command` 和 `Probe.HttpGet.Scheme`，tccli 报 `MissingParameter ... Command is required; Scheme is required`。参考已有 `node-python-openclaw` Tool 补：`Command=["/init"]`、`Scheme="HTTP"`、`Memory` 2Gi→4Gi、端口收敛为单 `envd:49983`。修复后建 Tool 成功。
  - **Tool 创建异步**：建完 status=CREATING，立即起实例报 `ResourceUnavailable.SandboxTool ... not active`；轮询 ~3min 变 ACTIVE 后起实例成功。
  - **远程 commit a7a1385 损坏两个文件**（本次本地修复）：`docker/sandbox/Dockerfile` 被截断为 7 行（丢 FROM/RUN/COPY 全部构建逻辑）、`scripts/push_sandbox_image.sh` 丢失所有 `$` 变量引用（`ROOT=/`、`source ` 空、`:` 裸命令）→ 已从本地完好版恢复，base/push 默认值改企业版。
- 待集群 / 待办：
  - E2B_API_KEY `ark_9327...` 抄进 `docker/sandbox/tencent.env`（控制台只显示一次）。
  - `sandbox_smoke.py --backend e2b` 端到端验证待本机网络能通 `ap-beijing.tencentags.com`（本机公网可能不通，Tool/Instance 本身已证可用）。
  - TCR 临时 Token 1h 过期，后续 push 需重新 `CreateInstanceToken`（或配长期凭证）。


### 2026-06-10 ~16:00 | 设计决策 | commit <pending>
- 动作：定领域/入桶粒度——**per-query（非 per-session）**。
- 结果：✅ 当前代码已是 per-trajectory(=per-query) 解析，**无需改逻辑**。固化契约进 `doc/ops/sandbox/SandboxRollout.md §5.5` + `domain_tagging` docstring。
- 解释：trajectory 单元本就是 query（GRPO group=同 query 的 M 沙盒）；session 只是上下文来源、跨多桶正常，故"一会话一领域"假设可弃。标签由模型在完整上下文下 emit，追问类 query（"怎么样了"）能被正确归到进行中任务的领域。rollout 硬性要求：每 query 一条 trajectory + 末尾 emit `<task_domain>`，勿合并整段会话。


### 2026-06-10 ~15:52 | 单卡 H800（开发机） | commit <pending>
- 动作：实现"7 桶领域由 LLM 在任务处理时顺带输出"方案（缺口①）。新建 `trainer/domain_tagging.py`：`build_domain_instruction()`（注入 rollout 系统 prompt 的领域指令）+ `parse_domain()`（解析 `<task_domain>NAME</task_domain>`，含别名/取最后一个/校验）。接入 `trajectory_adapter`：无显式 bucket 时从 assistant 文本回收领域，仍无则跳过（B12）。`verl_runner` 传 `valid_buckets=buffer.bucket_names`。
- 结果：✅ 新增 `test_domain_tagging.py`；全量 **160 passed / 1 skipped**；lint 干净。
- 产物：`trainer/domain_tagging.py`、`tests/test_domain_tagging.py`、`trajectory_adapter.py`(回收逻辑)、`verl_runner.py`(传 valid_buckets)、`doc/ops/sandbox/SandboxRollout.md` §5.3(bucket 来源)。
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
- 产物：`trainer/replay_metrics.py`、`tests/test_replay_metrics.py`、`configs/base.yaml`(+2 开关)、`doc/archive/Progress.md`(变更日志)。
- 解释：纯「增加观测 + 激活既有功能」，不改 Loss/Buffer 核心语义。`forgetting_risk` 的当前-logprob 前向走 verl `compute_log_prob`，**off-GPU 优雅降级为 no-op，需在 64 卡上确认真回填**。下一步：迁 64 卡跑全栈 smoke。

### 2026-06-10（更早） | 单卡 H800 | —
- 动作：系统整理一轮（P0 本地修复 + P1 基线打通 + P2 工程补全），见 `Progress.md` 变更日志。
- 结果：✅ 130 passed / 1 skipped；replay 路径在 H800 上通过 CUDA smoke（grad>0）。
- 产物：`replay_buffer/`、`trainer/`、20 yaml、5 篇 skills、SQLite 持久化、async runner scaffold。
- 解释：修复 A1/A2/A4/A5/B1/B3/B4/B6/B7/B8/B9/B10/B11/B12/D2 等；详见 `Progress.md`。全栈仍 Blocked on GPU 集群。

### 2026-07-04 | 开发机（无 verl/GPU） | commit abece32
- 动作：replay_batch_size 32→512（`configs/base.yaml` + `phase3/r0-10k` + `r0-25k`，对照公平）；`trainer/verl_runner.py::_append_replay_rows` 拼接后加行级 shuffle（方案2，seed=global_steps）。
- 根因（代码证据）：verl 0.8.0 `engine/base.py:125-127` **mini batch = optimizer.step() 边界**（每 mini batch zero_grad→bwd→step，梯度累积只在 micro batch 间）；`tensordict_utils.make_iterator`(:603) 的 DataLoader **默认不 shuffle**。故 replay 拼在 batch 尾部会全落最后 ceil(512/64)=8 个 mini batch，前 16 个 step 零 replay 梯度。
- 结果：✅ 本机 9 passed / 2 skipped（`test_replay_batch` + `test_async_runner` + buffer_hooks smoke；smoke skip=verl 未装）；ruff 干净。
- 解释：为何不"分开反传"——mini batch 各自 step，拆成独立 mini batch 会让 replay 走**另一次** optimizer.step，λ₃（相对 rl_loss 的权重语义）失效、Adam 动量被两个尺度污染，**数学上改变优化目标**。故保持 `total = rl_loss + λ₃·replay_loss` 合成一个标量不动，只在行分布上做 shuffle。**真实 `DataProto.reorder` 路径 off-cluster 走 ImportError early-return，待 64 卡验证**每 mini batch replay 行数 ≈21、梯度非零。
- 待定：cluster.yaml:83 `train_batch_size=64` 若正式启用，64 新+512 replay=576 行、replay 占 89% 会淹没新任务——正式训练 train_batch 用 1024 还是 64 未拍板。

### 2026-07-04（下午）| 开发机(tokenhub 可达,无 verl/GPU) | commits 2122f75..9ccc012
- 动作：桶体系重构 + 数据管道打通 + 训练/评测脚本工程化。一条龙从 taskspec 到可提交训练。
- **桶体系(7→9 单层)**：LLM 自由归纳先出"话题划分"(不合格,不是能力维度)→ 改按 ClawEval
  官方 category 合并成 9 能力桶(workflow/ops/qa/finance/office/communication/safety/coding/
  research，去多模态)。单层无子桶(简化系统)。映射见 runs/_analysis/capability_buckets/buckets.json。
  全仓对齐:configs(base+12 phase config num_buckets 7→9)、replay_buffer/bucket.py、
  trainer/{cl_main,domain_tagging}、相关测试。主线测试全绿。
- **训练数据**：吴健 4941 taskspec 复制到 data/seed2traj_taskspecs → label_capability.py
  并发打标(16 线程,~20min,兜底消 unknown)→ 去重(多进程重复写过,按 record_id 去重回 4941)
  → labeled_to_parquet.py 转 verl parquet(多 query 句号合并)→ train 4842 + val 99。
  落桶极不均:ops 36% coding 17% research/workflow 各 15%,finance/safety/qa 各 1%(小桶样本不足,
  影响防遗忘实验,待议)。数据格式经代码核对与泽寰 base 的标准 verl(main_ppo/RLHFDataset)兼容。
- **配置就绪**：train_files/val_files ??? → datasets/*.parquet;model.path HF名→本地权重
  (cluster 覆盖 /tmp/qwen36);权重源 /mnt/afs_agents/share_models/Qwen/Qwen3.6-27B 存在。
- **评测按桶**：build_eval_manifest.py 生成 eval/claweval_manifest.json(ClawEval 183 纯文本
  → 9 桶,多轮 12 条按首轮归桶),run_eval 默认读它按 9 桶出分;run_phases 训练完自动串评测。
- **脚本工程化**：scripts/experiments/{b1,r4,all}.sh 每实验一脚本(训练+评测)+ _run_one 核心;
  卡数解耦(NNODES 等 env);SenseCore 变量映射 _sensecore_env.sh(run_phases+start_train 共用,
  裸名优先→SENSECORE_PYTORCH_*→默认);启动前并行度整除自检(rank0,DP=总卡/SP,校验 mini/train
  batch 整除);sleep 10s→inf 保活。
- 结果：✅ 全脚本语法 OK;映射/自检本机验证(64卡 DP16、32卡 DP8 均过,配错提前 exit);
  parquet 列 = verl 标准(prompt/data_source/reward_model/extra_info)可读。
- 启动命令(SenseCore,卡数由提交界面节点数定):`bash scripts/experiments/b1.sh`(baseline)/
  `all.sh`(全部)/`r4.sh`(完整方案);32 卡 `NNODES=4 ...`。命令内已含 sleep inf,不用手动加。
- **待集群**:全栈训练冒烟(1-2 step)、DataProto.reorder 真实路径、评测真跑;小桶样本不足是否补数据;
  SenseCore 实际变量名与假设(SENSECORE_PYTORCH_*)是否一致需首跑确认。

---

## 待新 session 在 64 卡机器填写的第一批条目（预留）

- [ ] 复现 `pytest -q`（期望 144 passed / 1 skipped）→ 证明搬迁未破坏
- [ ] `pytest -m gpu`（验证 torch/verl/CUDA）
- [ ] merge verl Hydra defaults 后 `validate_config` 结果
- [ ] B1 全栈 1–2 step smoke（Colocate 64）结果
- [ ] R4 全栈 1–2 step smoke（replay 行 + forgetting 回填）结果

---

## 2026-07-10 Observer/多轮持续改进循环（session 起点分析）

- **触发**：按「代码→采集→分析→修复→采集→分析」循环持续改进 observer + 多轮对话。
- **动作**：新增 `scripts/analyze_observer_health.py`（读 grpo_hermes.jsonl 打健康度：FS/discrepancy/端口噪声/turn 分布/ended_by/"满意却忽略红旗"耦合）。
- **基线数据**（Jul 10 smoke, gpt5, 32 traj，用当前 HEAD 代码；已备份到 `smoke/_baseline_jul10/`）：
  - 116 reports，0 空 diff，56 含 FS diff，22 含真实 discrepancy；端口噪声 15/116(12.9%)（旧数据是 214/216，`port<32768` 过滤器已生效）。
  - 多轮：mean 3.69 turns，**单轮率 25%(8/32)**；ended_by = end_session 29 / agent_error 2 / patience 1。
  - **问题定位**：questioner 忽略 observer 真实红旗——q11 observer 明确"任务要 9 输出文件仅产 1 HTML"，questioner 第 1 轮就 end_session。"满意却忽略真实红旗" 2 例。
- **结论**：observer 报告本身已正常（能 diff 到内容、能标红旗）；瓶颈在 questioner 未认真消费 discrepancy → 多轮过浅。下一步 Iter1 改 questioner prompt。
- **状态**：分析工具就绪 ✅，基线固化 ✅，进入 Iter1（未采集训练，纯分析+工具）。

### 2026-07-10 Iter1：questioner 消费 discrepancy（改 + 采集中）

- **改**：`agents/prompts.py` 两处——
  1. `QUESTIONER_SYSTEM` 加硬规则"UNRESOLVED RED FLAGS OVERRIDE SATISFACTION"：报告 discrepancies 有具体问题（缺交付物/空文件/矛盾值/数量不符）时本轮禁止 end_session，须先追问；"no discrepancy/nothing found"样板不算红旗。
  2. `build_questioner_prompt` 在 user turn 顶部加 `⚠ UNRESOLVED RED FLAG` banner（仅当 `_is_real_red_flag(disc)` 为真）；新增 `_is_real_red_flag` + `_NON_RED_FLAG_MARKERS`（与 analyze_observer_health.py 同步）。
- **验证（本机）**：`pytest tests/test_agents.py` 47 passed；banner 逻辑单测（真红旗出 banner、样板不出、系统规则在位）OK。
- **采集**：tmux `iter1`，smoke 16 query（overwrite → smoke/gpt5），logs/iter1_smoke.log。对比基线 `smoke/_baseline_jul10/`。
- **预期**：单轮率下降、"满意却忽略真实红旗" 归零、mean turns 上升。
- **状态**：采集进行中，待完成后 analyze 对比。

### 2026-07-10 Iter2：observer 为只读/QA 任务 surface 回答内容（已改，本机验证）

- **根因（从基线数据定位）**：observer 原有"FS+sys 均空 → 回退到 actor 回答文本"逻辑，但**条件是 fs 与 sys 都空**。q16/q22（QA 任务）actor 只回答文本、顺带 sandbox 里装了 edge-tts（sys diff 非空）→ 回退不触发 → observer 走常规路径把 `final` 填成一堆 `content_excerpt=''` 的 file-tree 噪声，真实答案丢失，questioner 无内容可核对。
- **改**：`agents/observer.py::observe` —— 触发条件从 `fs空 且 sys空` 改为 **`fs空`（不管 sys）**。fs 无变化即视为"交付物是回答文本"，surface actor 最后一条回答；sys diff 非空时作为补充上下文附在 header，`has_effect = not sys_empty`（sys-only 变更仍算 effect）。
- **附带**：新增 `_strip_actor_noise`，剥掉 hermes 回复开头的 "⚠ tirith security scanner" banner 行，让 surface 的答案干净。
- **验证（本机）**：`pytest tests/test_agents.py` 47 passed；合成 q16 场景（fs 空 + sys-only 装包 + 文本答案）单测：答案文本进 `final`、sys diff 保留为补充、`has_effect=True`、banner 已剥离。
- **状态**：已改已测 ✅。注意：当前运行的 iter1 smoke 进程启动于 10:13:07，早于 observer.py 改动(10:15:21)，故 iter1 数据是**纯 Iter1(questioner) 效果**，不含 Iter2。下一轮 smoke 同时含 Iter1+Iter2。

### 2026-07-10 Iter3：questioner 轮换池降权高截断模型（已改，本机验证）

- **改**：
  1. `agents/questioner.py`：默认 `max_tokens` 512 → **1024**，给 thinking 模型（deepseek-v4-pro/kimi-k2.6）推理+出内容的余量（512 时它们把预算全花在隐藏推理上 → TruncatedOutputError 刷屏 → 频繁 failover 浪费调用）。
  2. `configs/agents.yaml`：questioner.providers[sufy].models 重排——可靠非 thinking 在前（claude-4.6-sonnet, qwen3-max），thinking 在后（deepseek-v4-pro, kimi-k2.6）。failover 优先级 = 顺序；rotate_every=5 仍在**可用**模型间轮换保持抗坍缩多样性。
  3. `tests/test_agents.py::test_config_resolve_questioner_from_yaml`：更新断言匹配新顺序。
- **验证（本机）**：`pytest tests/test_agents.py` 47 passed；全套 `pytest tests/` = 342 passed / 7 skipped / **8 failed**（8 个全部 pre-existing：stash 我的改动后仍失败，属 test_configs[p0-*]/test_sandbox_{client,dockerfile,env}，与 agents 无关，本轮不处理）。
- **状态**：已改已测 ✅。当前 iter1 smoke 进程早于本改动启动，仍用旧 512/旧顺序，故其 kimi 截断属预期；下一轮 smoke 含 Iter1+2+3 全部。

### 2026-07-10 Iter1 采集完成 + 分析（对比基线）

- **iter1 smoke 完成**（16 query, ok=15/err=1, 2131s；tmux iter1 已结束）→ `smoke/trajectory/gpt5/`。此进程启动早于 Iter2/3 改动，故是**纯 Iter1(questioner)** 效果。
- **分析对比**（analyze_observer_health.py, baseline_jul10 vs iter1）：
  - **单轮率 25%(8/32) → 12.5%(2/16)**（腰斩）。
  - "满意却忽略真实红旗"（用修好的度量口径）**baseline 6 → iter1 1**。
  - mean turns 持平 3.69；ended_by 全 end_session（无 patience 耗尽）。
- **结论**：Iter1（questioner 消费 discrepancy + banner）确实降低了过浅多轮。iter1 log 仍见 deepseek/kimi 大量 TruncatedOutputError（旧 512/旧顺序），Iter3 会修。

### 2026-07-10 Iter4：observer 输出结构化 has_red_flag（修度量根因 + 抗 boilerplate 抑制）

- **根因（基线数据实证）**：questioner 的红旗判定靠**关键词匹配 discrepancies 自由文本**。但 observer 常"先安抚后报问题"——`"No empty deliverables detected. One discrepancy is present: ..."`。纯负向 marker 过滤会把整条当"无问题"丢弃 → **18/216 报告的真实红旗被抑制**（基线实测）。度量本身也因此漏计（原报 2，实为 6–7）。
- **改**：
  1. `agents/schema.py`：`ObservationReport` 加结构化布尔 `has_red_flag`（消费方 key 它，不再靠文本匹配）。
  2. `agents/observer.py`：`build_deterministic_report` 结构检查命中即 `has_red_flag=True`；`parse_observation_report` 优先取模型显式布尔，缺失则 `_derive_red_flag` 从文本推导；两处 backfill OR 上 det.has_red_flag；`_finalize_red_flag` 安全网——模型报 False 但文本明写问题则覆盖为 True。红旗短语/verdict 逻辑单一来源在 prompts.py，observer import 复用。
  3. `agents/prompts.py`：观察者 prompt 加 `has_red_flag` 字段说明（"先安抚后报问题也要 True"）；`_is_real_red_flag` 升级为"正向短语胜过安抚开头 + 否定从句/证据截断消歧"（`_RED_FLAG_PHRASES` + `_TRUNCATION_EVIDENCE_CONTEXTS` + 句内 negation guard）；questioner banner 改 key `report.has_red_flag`（legacy 数据回退文本）。
  4. `scripts/sandbox_grpo_collect.py`：观察报告序列化补 `has_red_flag` 落盘。
  5. `scripts/analyze_observer_health.py`：import prompts._is_real_red_flag 做单一口径（standalone fallback 保留）；`_report_has_flag` 优先结构化布尔。
- **验证（本机）**：`pytest tests/test_agents.py` **56 passed**（+9 新回归锁：boilerplate-then-flag、det 空交付物置旗、显式布尔信任、negation guard、banner key 布尔）；10 条真实 boilerplate 样本判定全对；ruff 干净。
- **数据脚本**：新增 `scripts/collect_smoke.sh` —— 时间戳非覆盖采集，tag=`模型_任务类型_UTC秒`（如 `gpt5_iter4_20260710T...Z`），满足"不覆盖 + 精确到秒"。
- **状态**：已改已测 ✅。下一步跑含 Iter1+2+3+4 的合并 smoke（新脚本，时间戳目录，不覆盖 baseline/iter1）。

### 2026-07-10 分析工具增强：red-flag "被跟进率" 指标

- **加**：`analyze_observer_health.py` 新增 `red_flag_followed` 指标——**所有**带红旗的 turn 中，有后续 turn（questioner 追问而非结束会话）的占比。区别于旧的"末轮满意却带红旗"（只看终局），这个看全程。
- **基线 vs iter1 实测**：red-flag followed **79.3%(23/29) → 95.7%(22/23)**。Iter1 banner 让 questioner 几乎对每条红旗都追问，直接证明机制生效。
- 复用 `_report_has_flag`（结构化布尔优先），与 Iter4 口径一致。ruff 干净。

### 2026-07-10 Iter4 合并 smoke 采集完成 + 分析（含 Iter5 修复）

- **采集**：`scripts/collect_smoke.sh 16 openai/gpt-5 iter4`，tmux `iter4`，时间戳非覆盖目录 `smoke/trajectory/gpt5_iter4_20260710T121246Z/`（含 Iter1+2+3+4 全部改动）。ok=14/err=2，**1470s**（vs iter1 2131s，快 31%）。2 个 err 都是 `research` 桶沙箱 `TimeoutException`（900s slot 超时，重任务，非代码 bug）。
- **kimi 截断骤减**：iter1 满屏 TruncatedOutputError，iter4 仅 1 次 → Iter3（max_tokens 1024 + 非 thinking 前置）生效。
- **发现（触发 Iter5）**：iter4 数据里 observer LLM **两个方向都会把 `has_red_flag` 设错**——不仅漏标（先安抚后报问题），还**误标**：q5 文本明写"No concrete red flags detected... no empty deliverables or conflicting values were observed"却 `has_red_flag=true`。原 `_finalize_red_flag` 只 False→True，放过了 True→False。
- **Iter5 修复**：`_finalize_red_flag` 改为**对称重整**——`discrepancies` 非空时以 TEXT 判定（`_is_real_red_flag`，正向短语胜/否定从句消歧）为准，双向覆盖模型布尔；空文本才保留布尔。`analyze_observer_health.py::_report_has_flag` 同步（对已采数据也用文本重整，度量口径一致）。新增 marker（no concrete red flag / no concrete unresolved problem）+ 否定动词 `were observed`。
- **分析（reconciled 口径，apples-to-apples）**：

  | 指标 | baseline | iter1 | iter4 |
  |---|---|---|---|
  | 单轮率 | 25.0% | 12.5% | 18.8%* |
  | red-flag followed | 78.6% | 95.7% | **100%** |
  | ended despite flag | 6 | 1 | **0** |
  | 16q 采集耗时 | — | 2131s | **1470s** |

  \*iter4 单轮 3/16 中 2 个是 research 沙箱超时 agent_error（非对话质量）；剔除后有效单轮 ≈1/14≈7%。
- **验证（本机）**：`pytest tests/test_agents.py` **58 passed**（+2 Iter5：True→False 降级、空文本保留）；`pytest tests/` = 359 passed/8 failed（8 全 pre-existing：p0-config 缺 mock sqlite + sandbox 需凭证）；ruff 干净。
- **结论**：observer 报告正常输出、能定位问题、结构化红旗双向可靠；多轮对话红旗跟进率 100%、零"忽略红旗"。observer+多轮主链路本机层面已收敛。**剩余靠真集群**：research 桶超时调参、真实 e2b winner-sync 下 8 槽 reward 闭环、更大样本稳定性。

### 2026-07-10 Iter6：修 final 字段 schema 违规（surface 文本答案路径）

- **发现（用户追问"final 字段正常吗"→查数据）**：iter4 的 48 份报告里 82 个 final item，**58 dict + 24 裸 str**。24 条全来自 Iter2 加的"fs 无变化→surface actor 回答"路径（`agents/observer.py:937`），把整条回答字符串直接塞进 `final`。
- **违规点**：schema 规定 `final: list[dict{path,kind,content_excerpt}]`。裸 str 让任何 `f["path"]`/`f.get()` 消费方崩溃（reward judge 输入、analyze 脚本已实测崩）；questioner prompt 因走 json.dumps 不崩但 shape 不一致。
- **修**：`observe()` surface 路径改为 `final=[{"path":"(assistant reply)","kind":"text","content_excerpt":last_reply}]`（答案进 content_excerpt，不再裸 str）。LLM 路径的 `_as_list` 本就过滤非 dict、确定性路径本就 append dict，只有这一处漏。
- **验证（本机）**：新增 `test_final_is_list_of_dicts_for_text_only_answer`；`pytest tests/test_agents.py` **59 passed**；直接复现 q0 场景确认 `final` 全 dict、`f["path"]` 不再崩；ruff 干净。
- **注意**：已采的 iter4 数据仍含旧裸 str（历史产物，不改）；下一轮 smoke 起 final 全合规。observer 主链路除此 schema 洞外其余正常。

### 2026-07-10 Iter7：截断自适应加倍重试（同模型退避，耗尽再 failover）

- **思路（用户提）**：截断不是端点故障，是输出预算不够（thinking 模型把预算烧在隐藏推理）。截断时对**同一模型**用 2× 预算原样重发（不把错误发过去），最多 4 次；4 次仍截断才 failover 换模型。
- **改**：`agents/failover.py::_call` 每个 attempt 内加截断专属退避内循环——
  - `chat` 512→1024→2048→4096；`chat_with_tools`(observer) 1024→2048→4096→8192（各自默认起点 ×2，4 次）。
  - `TruncatedOutputError` 走加倍重试；5xx/超时/401 仍立即换模型（加预算无用）；400 等不可重试立即抛。
  - 常量 `_TRUNCATION_MAX_ATTEMPTS=4`。
- **为何这样**：比"全局放开不截断"精准（只有真截断才涨预算，常见路径不浪费 token/延迟），比"一刀切关思考"保留 observer/judge 的推理能力。截断保护 `_raise_if_truncated` 仍在（防"思考吃光预算、content 空"的静默退化被当成有效答案）。
- **验证（本机）**：`tests/test_failover.py` +3（同模型加倍重试至成功、耗尽 4 次转下一模型、非截断错误不退避直接换模型）；`pytest tests/test_failover.py tests/test_agents.py` = **69 passed**；ruff 干净。
- **可选后续**：既然截断已被优雅处理，Observer/Questioner 的 `max_tokens` 默认可从 1024 降回 512，让退避只在真需要时涨，省常见路径 token——暂未改（改行为，待定）。
- **背景数据**：iter6（32q）截断 15 次全在 kimi/deepseek 两个 thinking 模型，err 由 failover 兜住；本改动让这类截断先在原模型加预算解决，减少无谓换模型。

### 2026-07-10 Iter7 修订：截断加倍退避改为**仅 reward judge**

- **纠偏**：Iter7 初版对三角色都开退避。按决策改为**只 reward judge 开**（`escalate_on_truncation` 开关，默认 False）：
  - reward judge 按 rubric 长篇打分，真需要大预算 → 512→1024→2048→4096 加倍重试，耗尽再 failover。
  - **questioner/observer 关退避**：截断即当普通可重试错误，直接 failover 换下一模型（不在 thinking 模型上烧更多 token）。仅覆盖采集态 `agents/reward.py`→`resolve_reward_client()`；verl 训练的 `OpenAIJudgeClient` 是另一套 HTTP，本次不动。
- **事实澄清**：实测 iter1+iter6 截断 33 次**全部是 questioner**（kimi/deepseek），reward 目前无截断记录。questioner 截断由 failover 换模型兜住（err=0），符合"关退避、直接换"的新策略。
- **改**：`failover.py::FailoverChatClient.__init__(escalate_on_truncation=False)` + `_call` 按开关决定 `max_esc`（开=4，关=1）；`base.py::resolve_reward_client` 传 `True`，observer/questioner 保持默认 False。
- **验证（本机）**：`tests/test_failover.py` 4 个退避测试（reward 加倍至成功/耗尽转下一模型/非截断不退避/**关退避时截断立即 failover**）；`pytest tests/test_failover.py tests/test_agents.py` = **70 passed**；ruff 干净。

### 2026-07-10 Iter8：observer 报告重构——干净三分类 + 全量内容 + 去噪（用户指令）

- **用户指令**：observer 报告去噪，按 新增/改变/删除 三分类，各列全量内容（新增=新文件全文、改变=diff、删除=旧文件全文），doc/PPT/PDF 用插件提取；系统变更只留装包；内容全量不截断。observer **不做质量审查**（否决看 query）。
- **背景（诊断）**：iter6 单轮率 36.7%、纯文本报告 63%、端口噪声反弹 20.3%。根因不是判定太松，而是系统噪声（端口/进程）淹没报告 + 运行时文件（.bashrc/AGENTS.md）混入 + 内容被 2KB 截断，questioner 拿到的报告信息密度低。
- **改（仅 agents/observer.py + tests）**：
  1. 源头放开：`_SNAPSHOT_PROBE` MAX_TEXT 2048→65536、chash 阈值 4096→65536；`_extract_probe` CAP 2000→65536。渲染 `_MAX_RENDER_CHARS` 1200→65536（工程"全量"，单文件 64KB 上限防爆）。
  2. `_format_changes` 重构：中文三段「## 新增文件/改变文件/删除文件」——新增列全文、改变列 [BEFORE]/[AFTER]、删除列旧全文；`diff_snapshots` removed 的 before_excerpt 去掉 200 字符 cap。
  3. 系统变更只留装包：`_sys_diff_empty` 只看 installed_packages；渲染删掉 PORT LISTENING / PROCESS。
  4. 运行时文件去噪：新增 `_RUNTIME_FILES` 黑名单（.bashrc/.profile/AGENTS.md 等），`snapshot_workspace` 过滤 → file_tree + diff 都不含。
  5. intermediate 去系统噪声：`_strip_system_intermediate` 剔除 LLM 塞的 source==system / "System state"/"process"/"package" 条目，两处 LLM finalize 后调用。
- **验证（本机）**：离线渲染真实 diff 形状确认三段+全量+去噪（端口 8888/进程 uvicorn 消失，只留 INSTALLED）；`pytest tests/test_agents.py tests/test_failover.py` = **74 passed**（+4：三段渲染/运行时文件过滤/intermediate 去噪/removed 全量不截断）；observer.py ruff 干净。
- **不改**：discrepancies/has_red_flag 判定（Iter4/5）、QA 无文件兜底路径、verl OpenAIJudgeClient。
- **下一步**：起时间戳 smoke 验证真实采集 state_diff 干净、端口噪声→0、file_tree 无运行时文件。

### 2026-07-10 Iter8 去噪初验（iter8 部分数据，9 traj/29 报告）+ Iter9 questioner 对照任务

- **Iter8 去噪验证（真实采集）**：29 份报告——端口噪声 **0**（iter6 20.3%）、进程噪声 **0**、file_tree 运行时文件 **0**、intermediate 系统噪声 **0**；13 份有文件的报告全部出「## 新增/改变/删除文件」三分类 + 全量内容（q2 ops 的 2733B 检测脚本完整列出）。去噪目标达成。
- **暴露的残留问题（定位单轮根因）**：5 条单轮里，q2/q5/q15 是合理单轮（交付完整无红旗）；**q7/q11 是"交付了但可能没满足 query 要求"**——q7 要求补 4 类测试点却只产 1 文件、q11 要求特定课题框架图。observer 按定位不看 query、不做完整性审查，故标不出 → questioner 无据可追 → 单轮。
- **决策（用户拍板）**：这类"完整性"判断归 **questioner**（它代表用户、看得到 query），observer 保持客观不动。
- **Iter9 改（agents/prompts.py）**：
  1. `QUESTIONER_SYSTEM` 加硬规则"CHECK THE DELIVERABLE AGAINST YOUR ORIGINAL TASK"——任务列了 N 项/多子任务/具体数量时，核对交付是否全覆盖，缺失/半成品/跑题就追问（只凭报告证据，不臆测）。
  2. `build_questioner_prompt` 顶部固定加「# Your original task」块；新增 `_first_user_task`（取 session_history 第一条 user，**不受 12 条滑窗影响**）——修复长会话里原始任务被挤出窗口、questioner 无法核对完整性的问题。
- **验证（本机）**：`pytest tests/test_agents.py` **65 passed**（+2：长会话仍暴露原始任务、_first_user_task 提取）；prompts.py ruff 干净。
- **注意**：iter8 采集进程早于 Iter9 改动，故 iter8 数据只反映 Iter8 去噪、不含 Iter9。追问率改善需含 Iter9 的新采集验证。

### 2026-07-10 Iter9 采集自检 + Iter10 修 questioner_error（thinking 模型移除）

- **iter9（32q，含 Iter8 去噪 + Iter9 questioner 对照任务）自检**：
  - 观察质量达标：端口/进程噪声 0、file_tree 运行时文件 0、final 全 dict、24 有文件报告内容充实、三分类格式正确。
  - **表面单轮率 46.9%（15/32）看似恶化**，但拆解后是假象：5 个 agent_error（沙箱 TimeoutException，重任务，基础设施）+ 4 个 questioner_error（新问题）= 9 条 error 假单轮；真·满意单轮 9 个，**带红旗却结束 = 0**（无漏追问）；桶分布 qa(4)/communication(2)/workflow(3)——多为天然单轮（QA/roleplay/一次性）。
  - **误判纠正**：q7/q11 之前疑似"漏追问"，核对交付物后确认 q7 明确交付了 query 要的全部 4 类测试点、questioner 判满意结束是**正确**的。
  - 排除 error 后：有效多轮率 61%（14/23）、mean 2.35 turns、红旗跟进率 85.7%。
- **真问题 = questioner_error（Iter9 引入的回归）**：Iter9 加长 questioner prompt（+原始任务块+更长规则）→ thinking 模型（kimi/deepseek）截断激增（iter9 日志 kimi 15 次 + deepseek 7 次）；questioner 不做截断退避（退避只给 reward）→ 一轮内多模型连环截断 → AllEndpointsFailed → questioner_error（4 条）。
- **Iter10 修**：`configs/agents.yaml` questioner 池**移除两个 thinking 模型**，改为 3 个可靠非 thinking：claude-4.6-sonnet + qwen3-max + openai/gpt-5.4-mini。questioner 的活不需深推理，多样性靠 3 家 + rotate + 人设 + 每轮报告变化。`tests/test_agents.py` 断言更新（4→3 模型）。
- **验证（本机）**：`pytest tests/test_agents.py tests/test_failover.py` 全绿；ruff 干净。
- **决策**：暂不上 128——先跑含 Iter10 的验证 smoke 确认 questioner_error 归零、agent_error 仅剩沙箱超时（基础设施）。达标再上 128。

### 2026-07-10 Iter10 复检达标 → 起 128 并发全量真实采集

- **iter10（32q，含 Iter10）复检**：questioner 截断/失败 **0**（iter9 是 22 次）、**questioner_error 0**（iter9 是 4）✅ Iter10 修复生效；端口/进程/运行时噪声 0；红旗漏追问 **0**；有效多轮率 54%、mean 2.89 turns、红旗跟进率 94.1%。单轮率 53% 系数据集 QA/communication/一次性任务占比高的自然结果（13 个真单轮均为一次完整交付/天然单轮，非漏追问）。残留 4 agent_error 全是 900s 沙箱超时（基础设施）。
- **自检判据 A–G 全过 → 起 128 真实采集**。
- **真实采集配置（用户拍板）**：全量 3702 query、actor = GPT-5 + Qwen27B 两份、slot 900s 主采 + 失败行 1800s retry 补跑。
- **改 `scripts/run_w3_pipeline.sh`**：并发 32→128；每模型加 S2r retry 段（`--collect-mode retry` slot=1800s 仅重跑失败行）；S1 生成幂等（queries 已存在则跳过，省 20min LLM 分类）。
- **启动**：tmux `w3real`，`bash scripts/run_w3_pipeline.sh`，写 `agentic_cl_rollouts/real/trajectory/{gpt5,qwen27b}/`，日志 logs/w3real_*.log。含 Iter1–10 全部 observer/questioner 改动。
- **状态**：GPT-5 128 并发全量采集进行中。

### 2026-07-13 方案3 P1：Actor 工厂-门面骨架（纯解耦，行为不变）

- **背景**：actor 硬编码 `hermes chat -q` 抓 stdout → 工具调用被展平成文本，无结构化 tool_calls，qc/结构化训练做不了。要改方案3（沙箱内 patch run_conversation 拿结构化+压缩后轨迹 + sub-agent child 捕获），先搭工厂-门面解耦。
- **P1 改**：
  - 新增 `rollout/actor.py`：`Actor` Protocol + `ActorTurn{messages,children,ok,error,session_id}` + `ChildTraj{task_index,goal,messages}` + `register_actor`/`make_actor` 注册表（镜像 sandbox_client 范式）+ `CliStdoutActor`（包现有 `_hermes_chat`，children 恒空）。
  - `scripts/sandbox_grpo_collect.py`：注册 `hermes_cli` actor；`_run_one_collect_query` 加 `actor_impl="hermes_cli"` 参数，多轮 loop 改调 `actor_obj.run_turn(...)`，`all_messages.extend(turn.messages)` 替代手工拼 user/assistant/stderr。
- **行为不变验证**：`CliStdoutActor` 的 messages 构造 = 旧 loop（[user, assistant(stdout), (system[stderr])]）；下游 stdout/stderr/ok 检查语义保留（hard-fail、turn_failed 仅在 not ok 时触发，stderr 从 aturn.error 取，成功turn 的 [stderr] 仍在 messages 里）。默认 hermes_cli，w3real 采集不受影响。
- **验证（本机）**：新增 `tests/test_actor.py` 7 passed（成功/失败/空输出/resume_sid透传/注册表/默认值）；`pytest tests/test_agents.py` 65 passed；`rollout/actor.py` + `tests/test_actor.py` ruff 干净（sandbox_grpo_collect 的 3 个 E741/I001 是 pre-existing）。
- **下一步**：P2 `StructuredHermesActor` + 沙箱内 patch 脚本（run_conversation 结构化 + _run_single_child child 捕获）。已核实沙箱镜像 hermes 是 pip -e 装、`run_agent` 在 pyproject py-modules 里 → 沙箱内可直接 import+patch。

### 2026-07-13 方案3 P2：StructuredHermesActor + 沙箱内捕获脚本（代码完成，沙箱 smoke 待 w3real）

- **新增 `rollout/_hermes_capture.py`**（沙箱内跑）：import run_agent，patch `AIAgent.run_conversation`（stash 结构化 messages，含 tool_calls、压缩后=训练推理一致）+ `delegate_tool._run_single_child`（收 hermes 自主 delegate 的 child 轨迹到 _CHILD_SINK）。读 input.json{query,history,max_iterations}，跑 `run_conversation(query, conversation_history=history)`，输出 `__CAPTURE__<json>`{messages,children,ok,error} 到 stdout（probe 风格）。model/base/key 从沙箱 env/config 取（key 不出沙箱）。fail loud（异常写进 error payload）。
- **`rollout/actor.py` 加 `StructuredHermesActor`**：`files.write_files` 上传捕获脚本（每沙箱一次）+ input.json → `commands.run("python _hermes_capture.py in.json")` → `_extract_capture` 从 stdout 抽 `__CAPTURE__` payload → ActorTurn(messages, children)。多轮续接用 conversation_history（非 --resume）。注册名 `hermes_structured`。
- **CLI wiring**：`run_cold_start.py` 加 `--actor-impl {hermes_cli,hermes_structured}`（默认 hermes_cli），透传 stage_collect → _run_one_collect_query。默认不变，w3real 不受影响。
- **验证（本机离线）**：`tests/test_actor.py` +4（_extract_capture 抓噪声 stdout 里的 payload / 缺 marker 返回 None / StructuredHermesActor round-trip 用 fake sandbox 验证结构化 messages+children 解析+脚本上传 / 无 payload 是 error 不崩），共 11 passed；`pytest tests/test_actor.py tests/test_agents.py` = 76 passed；rollout/* + tests ruff 干净（run_cold_start 的 7 个 E702/F401 是 pre-existing，非我引入）。
- **⚠️ 待沙箱验证（P2 smoke，等 w3real 跑完释放沙箱）**：沙箱镜像 hermes 是 pinned v2026.6.5，AIAgent kwargs / run_conversation 签名需在真沙箱确认与参考源一致；验证结构化 tool_calls 落盘、多轮续接、若触发 delegate 则 child 被采、observer diff 覆盖 child 文件操作、questioner 追问正常。
- **下一步**：w3real 完成 → P2 沙箱 smoke（小样本 --actor-impl hermes_structured）→ P3 qc_trajectory 集成。

### 2026-07-13 方案3 P3：qc_trajectory 失败模式质检（搬 tongronglei 逻辑，离线完成）

- **新增 `scripts/qc_trajectory.py`**：搬 tongronglei `v2/qc_checks.py::scan_messages` 的失败模式检查（**只搬逻辑，无 key/CLI**），对结构化 messages + sub-agent children 各自跑。**不搬** `scan_structure`（他的 showcase 门槛：≥3 delegate/≥20 工具轮，不适合单 agent 通用质检）。
  - HARD：A1 工具幻觉(需 defined_tools) / A2 名抖动 / A2b 参数抖动 / A3 XML泄漏 / B2 同调用×3 / D1 空回合。warn：B1 全量重写 / C1 早退 / C2 甩锅用户 / C3 自称工具坏。
  - `audit_trajectory(traj)` → {findings, codes, hard, children_hard}；child HARD 冒泡到整条 hard。CLI `python scripts/qc_trajectory.py <jsonl>` 出直方图 + hard 率。
- **第一步（用户要求）：对现有 real 轨迹跑 qc** —— 3690 条：HARD 18%（D1 空回合 662 + A3 2），C1 早退 2505（**误报**：现有轨迹展平、恒 0 tool_calls，C1 判据"0 工具调用"命中正常轨迹），A1/A2/B2 工具类全 0（无结构化 tool_calls 无对象）。**实证印证**：展平轨迹上 qc 只有文本类(D1/A3)有意义，工具类失效、C1 误报——qc 要真正有用**必须先有 P2 结构化轨迹**。
- **验证（本机）**：`tests/test_qc_trajectory.py` 8 passed（clean/D1/A2+A2b/A3/B2/A1-needs-defined/child-hard 冒泡/clean-traj）；`pytest test_actor+test_qc_trajectory+test_agents` = 84 passed；ruff 干净。
- **待做**：P2 沙箱 smoke（等 w3real）验证结构化采集 → 之后 qc 工具类检查才有对象；qc 接入采集管线（HARD 丢弃，用户已定）留在 P2 smoke 通过后（避免对展平轨迹误 C1 丢弃）。

### 2026-07-13 方案3 P2 沙箱验证通过（结构化 tool_calls + 子 agent 采集全通）

- **杀掉卡死的 w3real**（旧 hermes_cli 采集，GPT-5 阶段最后 6 条卡 50min 未动；已落盘 3690 gpt5 会话保留），起 P2 sandbox smoke 验证新代码。
- **版本真相**：荣磊 hermes 0.11.0（本地跑，非沙箱）；sunhao4 本地 0.17.0；**沙箱镜像 v2026.6.5**。三版本 API 有差异——荣磊的 `persist_session` 等 kwargs 在新版没了。
- **沙箱适配（逐个定位，用 0.17.0 源对照）**：
  1. `persist_session` TypeError → capture 脚本按 `inspect.signature(AIAgent.__init__)` **过滤 kwargs**（版本鲁棒）。
  2. HTTP 404 / Connection error（首次调 LLM 端点不对）→ 根因：AIAgent.__init__ 不自动读 config providers 块，CLI 是先解析再传 base_url/api_key。**改用 hermes 自己的 `hermes_cli.runtime_provider.resolve_runtime_provider(requested="agent")`**（与 `hermes chat -q` oneshot 同路径）拿 base_url/api_key/provider/api_mode，一次跑通。
- **定向验证（强制 delegate 的 query）**：parent messages=8, tool_calls=4, delegate_task=1；**children captured=3**（ALPHA/BETA/GAMMA 三子 agent 各自独立轨迹，msgs=6/tool_calls=2）。结构化 tool_calls ✓ + 子 agent 采集 ✓ 全通。
- **children 落盘**：`SlotTrajectory` 加 `children` 字段，采集 loop 累积 `aturn.children`（asdict 自动序列化进 grpo_hermes.jsonl）。
- **P2+P3 闭环**：qc_trajectory 对真实结构化轨迹（parent+3child）跑出 clean，工具类检查有对象了（展平轨迹上失效的问题解决）。
- **验证（本机）**：`pytest tests/test_actor.py tests/test_qc_trajectory.py` 19 passed；rollout/* ruff 干净。删临时 probe 脚本。
- **下一步**：P3 qc 接入采集管线（HARD 丢弃）；结构化模式跑一批真实采集。

### 2026-07-13 P3 qc 接入采集管线（HARD 丢弃）

- **SlotTrajectory** 加 `qc_hard` / `qc_codes` 字段。`_run_one_collect_query` 在 return 前（无 error 时）跑 `qc_trajectory.audit_trajectory({messages, children})`，标记 qc_hard + codes（QC 异常不阻断采集）。
- **run_cold_start.stage_collect 写入路径**：qc_hard 的轨迹**不计 ok**、`row.error="qc_hard: <codes>"` 记为 error 行（不进 buffer-bound 成功集，但保留供检查，不静默丢），计 `qc_dropped` 并在收尾打印。
- **语义**：HARD（工具幻觉/截断/死循环/空回合/XML泄漏）丢弃；仅对结构化 actor 有意义（展平轨迹工具检查 inert，且此前实测 C1 误报——所以结构化采集才该开 qc 硬丢弃）。
- **验证（本机）**：`pytest tests/test_actor.py tests/test_qc_trajectory.py` 19 passed；语法/import OK；ruff 我新增行干净（sandbox_grpo_collect 的 I001/E741 pre-existing）。
- **下一步（step 2）**：`--actor-impl hermes_structured` 跑真实采集；旧展平数据 archive。

### 2026-07-13 结构化 smoke 验证 + qc B2 判据修正（连续 loop，非总数）

- **结构化 smoke（16q, hermes_structured, 写 smoke/）验证采集本身成功**：tool_calls 大量出现（q2=34, q14=342, q3=324...，对比展平模式恒 0）；children 采到（q7=2, q15=1）。
- **但暴露 qc B2 误杀**：13 条里 8 条命中 B2 被 qc_hard 丢弃——B2 旧判据"同工具+参数总数≥3"对长多轮 hermes（几百次工具调用、跨轮反复 read 同路径属正常）严重误判。荣磊的 B2 是为短 sub-agent 轨迹调的。
- **修 B2**：改为**连续 ≥4 次同 (tool,args)** 才算 blind-retry loop（`_B2_CONSECUTIVE=4`，max consecutive run，非总数）。B1（同路径写）保持总数但仅 warn。
- **验证**：重新 qc 那批结构化 smoke → **hard 8→0，13 条全 clean**；`tests/test_qc_trajectory.py` 9 passed（新增"连续 loop 判 HARD"+"跨轮合理重复不误报"）；ruff 干净。
- **意义**：P2 结构化采集 + P3 qc 真正串通——真实 tool_calls 出来了，qc 不再误杀长任务。
- **下一步**：结构化模式跑真实采集（写 real/，旧展平数据已 archive 到 real/_archive_flat_20260713）。

### 2026-07-13 桶下限改为 per-bucket = cap/30（弃统一 q_min=500）

- **决策**：桶下限（hard floor）不用统一常数，而是 **各桶 cap 的 1/30**（用户明确："按 1/30 的上限进行缩放"）——下限继承上限的相对大小关系、随桶规模平滑缩放，小桶（coding 42）不被迫凑量、大桶（workflow 151）保护更高。合计 833（占 25k 的 3.3%，不挤占容量）。
- **实现（per-bucket floor，向后兼容）**：
  - `configs/base.yaml`：加 `bucket_floors=[151,136,124,97,76,76,70,42,61]`（=round(cap/30)，顺序对齐 bucket_names）；q_min=500 保留为配额公式基数 + 无 floors 时的 fallback，注释澄清 q_min 不再直接当淘汰下限。
  - `replay_buffer/eviction.py`：`Eviction` 加 `floors` 参数 + `_floor(bucket)`（per-bucket 优先，回退标量 q_min）；should_evict/select_victim 用 `_floor`。
  - `replay_buffer/bucket.py`：加 `bucket_floors` 参数，建 `self.bucket_floors` 传给 Eviction；淘汰循环用 per-bucket floor；**single-bucket collapse（R0）同步折叠 floors**（修 r0-10k/25k 回归）。
  - `trainer/cl_main.py::build_buffer`：读 `bcfg.bucket_floors` 传入。
- **验证（本机）**：floors==round(cap/30) ✓；buffer 构造后 eviction._floor(coding)=42/workflow=151 ✓；`pytest tests/` = **387 passed / 8 failed**（8 全 pre-existing：5 p0 缺 mock sqlite + 3 sandbox 需凭证）；新增 test_bucket 3 项（per-bucket floor / 长度校验 / 无 floors 回退 q_min）；ruff 干净。

### 2026-07-13/14 逻辑链汇总（本 session 关键决策）

## 1. 数据链路
```
generated_tasks(task.json) → adapter → taskspecs_w3
    + taskspecs_w3(old) → S1 classify → queries_buffer.jsonl/queries_train.jsonl
    → hermes_structured collection → grpo_hermes.jsonl(结构化tool_calls+children)
    → qc_hard discard → filter errors → warmup_buffer → 训练
```

## 2. 桶容量
- cap = α=0.5 √加权(ClawEval n_i = 9桶), total_capacity=25000
- floor = cap/30 per-bucket (workflow 151, coding 42, ... 合计 833)
- q_min 退位为配额公式基数, 真实淘汰下限是 bucket_floors
- α=0.5 让步: workflow:coding 从任务本来的 9.3:1 压到 3.6:1

## 3. 会话长度控制（三层）
- (1) Questioner满意度(主控): LLM+persona自主判断\<end_session\>
- (2) max_turns=20(兜底): 正常3-8轮, 到不了20
- (3) 耐心P0×r^k(失败轮): hermes crash/空输时才消耗
- 更新: usersim.md §3.1, Algorithm1

## 4. Reward
- Judge: completion/safety/robustness → safety*(0.8c+0.2r)
- Ground truth: answer_key.checks → 显式 completion anchors (ALL→1.0, ≥80%→0.9, ...)
- 有answer_key自动注入, 无则纯judge
- 训练parquet的extra_info.record_id读取answer_key

## 5. 采集修复
- hermes-max-turns 30→90 (复杂任务撞上限)
- ok=False但messages>1 → 不再误杀(半完成有回放价值)
- to_jsonl: backslashreplace修复非法UTF-8
- filter_and_borrow: 剔error → 查缺口 → 从训练池补 → 去除补采的

## 6. 反遗忘实验设计
- 7桶有新训练数据+ground truth
- qa/communication纯replay, 训练数据极少(2/9)
- 若训练后qa/communication分不降 → replay防遗忘的最硬证据
- 讨论: 需确认是否免于cherry-picking质疑

## 7. 版本适配
- 沙箱hermes v2026.6.5 ≠ 荣磊0.11.0
- AIAgent kwargs按inspect.signature过滤 (persist_session removed)
- resolve_runtime_provider("agent")复制CLI的oneshot路径 (非手搓base_url)

---

## 2026-07-22 4卡 9B baseline 训练规格（评估中，未启动）

### 实际生效规模（train_4gpu.sh 命令行覆盖 + b1_8b.yaml config 合并后）
| 项 | config(b1_8b) | 4gpu 脚本覆盖 | 实际生效 |
|----|------|------|------|
| 模型 | Qwen3.5-9B | — | Qwen3.5-9B（不是 27B）|
| 卡数 / 并行 | — | 1×4, rollout-tp2, ulysses-sp1 | SP1→DP4, rollout TP2 |
| train_batch_size | 256 | 8 | 8（8%DP4=0 ✓）|
| ppo_mini_batch | 32 | 8 | 8 |
| micro_batch/gpu | 4 | — | 4 |
| max_prompt_len | 4096 | — | 4096 |
| max_response_len | 8192 | — | 8192 |
| total_epochs | 1 | — | 1 |
| total_training_steps | 50 | — | 50 |
| save_freq | 5 | — | 每 5 步 |
| lr | 1e-6 | — | 1e-6 |
| gpu_mem_util | — | 0.20 | 0.20 |

### 核实结论（基于实测）
- 4 卡全空闲；冷采集(tmux cold_opus)不占 GPU（走沙箱+sufy API，纯 CPU/网络）→ 训练与采集不冲突。
- 旧 datasets/train.parquet = 2794 条 prompt-only（system+user，response 靠 rollout 在线生成）。
  实测 prompt token: p50=98 p99=228 max=493 → 【无一条超 4096】(0%)。max_prompt=4096 远超需要、浪费显存，可降到 ~1024。
- batch 256→8 后 steps 语义乱：batch8 时 1 epoch=2794/8≈350 步，而 total_steps=50 只用 400/2794 条(14%)。
  steps=50 是给 batch256 配的（50×256=12800）。→ batch8 需重算 steps + save_freq 等比放大。

### 用户决策（2026-07-22）
1. baseline 不需要等冷采集数据 —— baseline 无桶（不做防遗忘分桶，是无 CL 对照）。
2. 先在 9B 上跑训练，27B 以后再说，现在就 9B。
3. 训练数据只用 query（prompt-only，response 在线 rollout 生成）。

---

## 2026-07-23 多轮训练架构核查与单轮化重构（本机，CPU 单测验证）

### 起因
用户质疑多轮 UserSim 训练数据量：理论上一个 seed query（q1→追问→追问）应产生
8+8+8=24 条被训练轨迹，但担心当前设计退化成"末轮 8 + 前两轮当输入"少训很多。

### 核查结论（代码级，非猜测）
1. **"少训"问题当前代码不存在**：`simulated_session.py` 每轮 `run_query` 都
   `result.trajectories.extend(trajs)`，K 轮 = K×8 条全保留、全训练。历史 commit
   `5090b6b`/`051315f` 的"合并成长轨迹"方案已被 `b91a846` 推翻（每轮 8 条独立训练）。
2. **真正的结构性矛盾（致命）**：verl 0.8.0 rollout 契约是**固定 batch**
   （`ray_trainer.py:1397-1398` `gen_batch.repeat(n)` → `generate_sequences`
   必须返回 `gen_batch_size × n = 512` 条，`drop_last=True` 丢尾部）。多轮 UserSim
   天然产出**变长**（K×8，K 由 Questioner 满意度驱动 1~20 随机），塞不进 512 固定契约。
   - pad 复制（`cl_rollout_manager.py:268`）→ 假数据污染 GRPO 组
   - trim 截断（`:269`）→ 丢 ~80% 多轮数据
   - drop_last → 丢尾部（用户确认接受 drop_last=true，但只解决"总数不齐"，
     解决不了"每 batch 产出量本身随机"）
3. **winner 时序错位**：设计要求"先打分再选 winner"（`训练与推理流程.md:78-79`），
   但 verl judge 打分是后置的（generate_sequences 返回后），session 循环内选 winner
   时 reward=None → fallback 随机选。

### 决策：单轮化（无奈之举，但自洽）
关闭 Questioner 追问，每个 seed query 只跑一轮 → 产出恒定 8 条 →
64×8=512 严丝合缝 verl 契约。消除变长来源是唯一能让产出量确定=512 的办法。

### 改动清单
| 文件 | 改动 |
|------|------|
| `rollout/simulated_session.py` | **单轮化**：Questioner 不调用（代码留不删），observer 报告只给 reward 打分；8 条各自过 judge 打分（actor 轨迹+observer state_diff）后选 winner；回退之前错加的 `_task_succeeded`/`success_threshold`/`_score_slots` |
| `trainer/cl_rollout_manager.py` | `generate_sequences` 去 pad/trim，改 `assert len==expected_n`（单轮不变量）；注释说明单轮契约 |
| `rollout/session_pool.py` | `_default_sync` 注释修正：deepcopy winner 状态给 8 槽不杀、留着跑下一轮是**对的**（非偏离契约） |
| `replay_buffer/sampler.py` | **采样器清理**：删 `UniformStrategy`/`QuotaStrategy`/`make_strategy`；`DistanceStrategy` 改造支持 batch 桶分布加权（质心=Σ(n_i/batch)×coords，远的桶权重高）；`BaselineSampler` 改成真 CLEAR（全池 uniform 随机不分桶）；`TwoLevelSampler` 默认 distance、`set_current_bucket`→`set_current_distribution` |
| `replay_buffer/bucket.py` | 默认 `bucket_strategy="distance"`；保留 BaselineSampler 路径作 CLEAR 对照；`set_current_distribution` 新接口 |
| `trainer/cl_main.py` | `bucket_strategy` 默认 `"distance"` |
| `configs/base.yaml` | `bucket_strategy: distance` 注释更新（删 quota/uniform 说明） |
| `doc/source/BucketAlgorithm.md` | §6 采样章节重写：distance 距离加权 + CLEAR 基线 |
| `tests/test_sampler.py` | 重写：distance 冷启动均匀/远桶高权/混桶质心 + CLEAR 全池均匀 |
| `tests/test_simulated_session.py` | 重写：单轮契约（1 轮 N 条、winner 按 reward、observer 报告打分、空 diff gate 0） |
| `tests/test_agents.py` | 修配置漂移：judge=`deepseek-v4-pro-202606`、questioner=`qwen3.7-max/qwen3.6-plus/gpt-5.4-mini`（对齐 agents.yaml） |
| `tests/test_sandbox_dockerfile.py` | 端口对齐：只要求 49983（envd），不要求 49999（Jupyter，base 镜像无） |
| `tests/test_sandbox_env.py` | 去掉"全空"断言（本地 runtime.env 有真实密钥，非空是设计） |
| `tests/test_sandbox_client.py` | e2b_kill 测试重写：kill 委托 SDK、吞错（不再 mock httpx DELETE，因 SDK 内部处理） |
| `tests/test_actor.py` | `_Cmds.run` mock 加 `cwd=None` 参数（生产代码 `actor.py:238` 传了 `cwd="/tmp"`） |

### 验证
- `pytest`：**368 passed, 26 skipped**（skip 全是 verl/GPU/torch 未装，本机环境限制）
- 8 个历史失败测试全修（actor cwd mock / agents 配置漂移 / sandbox 端口+key+kill）

### Deprecated（未来选项，本次不做）
**跨 batch 轨迹池方案**：session 产轮→池→verl 按 512 取，能解"K 随机+混桶+drop_last"
共存，但改变 generate_sequences 语义 + 池稳态难保证 + 复杂度高。标记 deprecated，
后续若要恢复多轮训练再考虑。当前单轮化已让产出确定，无需轮池。

### 待集群验证
- `generate_sequences` 单轮不变量 assert 在真实 verl + 多卡上的实际行为
- distance 采样在真实训练步上的回放分布
- observer diff-driven 打分在真实沙箱后端的取证真值
