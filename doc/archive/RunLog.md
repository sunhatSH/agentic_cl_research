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

---

## 2026-07-24 B1 16GPU 训练启动失败：多机 rendezvous bug 修复（集群，SenseCore）

### 命令
```bash
bash scripts/train.sh 16gpu --config configs/run/b1_9b_16gpu.yaml
```

### 结果：启动即崩，0 step 完成
- **主节点** (`pt-31d6097c...-master-0`, `10.120.6.246`)：配置校验通过，数据加载完成（train=44378, val=918），到 `init_workers()` 时报 `ValueError: Total available GPUs 8.0 is less than total desired GPUs 16`，崩溃
- **副节点** (`10.120.7.20`)：启动 ~1.5 分钟后被集群 Kill

### 根因（代码级，非集群问题）

`_train_impl.sh:185` 的多机同步逻辑有 2 个 bug：

**(A)** `torch.distributed.rendezvous()` 的 TCP URL 缺少 query 参数：
```python
# 旧代码
d.rendezvous(f'tcp://{MASTER_ADDR}:{MASTER_PORT}')
# → PyTorch 2.x _tcp_rendezvous_handler: rank = query.get("rank") → None
# → "rank parameter missing"
```
原因：PyTorch 2.x TCP rendezvous 需要 `tcp://host:port?rank=X&world_size=Y` 格式。

**(B)** `_train_impl.sh:1` 是 `set -uo pipefail`（不是 `-euo`），缺少 `-e` → python 命令失败被静默吞掉，两个节点跳过 barrier 各跑各的。

**连锁效应**：副节点 rendezvous 失败 → 直接到 `ray start --address ... --block` 空等；主节点跳过 barrier → 启动训练 → 资源检查发现 Ray 集群只有本地 8 卡 → 崩溃。

### 修复
`_train_impl.sh:185` 两处改动：
1. TCP URL 加 `?rank=$RANK&world_size=$WORLD_SIZE`
2. 加 `|| exit 1` 显式失败中止（不加 `set -e` 全局以防影响其他分支）

```bash
# 修复后
OMP_NUM_THREADS=1 "$PY" -c "... d.rendezvous(f'tcp://{addr}:{port}?rank={rank}&world_size={ws}'); ..." || exit 1
```

### 副节点日志取证
```
08:24:08 [sensecore_env] RANK=1 NNODES=2 MASTER_ADDR=... MASTER_PORT=23456 WORLD_SIZE=2
08:24:09 ValueError: Error initializing torch.distributed using tcp:// rendezvous: rank parameter missing
08:24:10 Ray runtime started. Local node IP: 10.120.7.20
08:24:12 --block (阻塞等待 → ~1.5min 后容器被集群 Kill)
```

### 状态：修复已提交，待重新提交训练验证

---

## 2026-07-24 B1 16GPU 第二次运行失败：Ray head 竞态（集群，SenseCore）

### 命令
修复 rendezvous URL 后重新提交 `bash scripts/train.sh 16gpu --config configs/run/b1_9b_16gpu.yaml`

### 结果
同样的错误：`ValueError: Total available GPUs 8.0 is less than total desired GPUs 16`（08:34:23，PID 4613, hostname `pt-a84d62dd...-master-0`, IP `10.120.4.230`）。SenseCore 平台新增了 `acp_barrier:v0.0.6` 容器做平台级多机同步。

### 根因：Ray head 启动与副节点连接的竞态

修复前的执行顺序：
```
barrier 释放 → [主] ray start --head (耗时 ~3-5s)
             → [副] ray start --address $MASTER_ADDR:6379 (立刻，head 未就绪 → 失败)
```

副节点在 barrier 释放后立刻尝试连接主节点的 Ray head，但主节点刚进入 `ray start --head`（初始化 GCS + 分布式运行时），head 尚未监听 6379。副节点连接失败 → 退出 → 主节点资源检查只有本地 8 卡。

### 修复（`_train_impl.sh:182-199`）

调换顺序：**主节点先启动 Ray head，再做 barrier**：

```
[主] ray start --head (head 就绪)
→ barrier (等待两节点都到达)
→ [副] ray start --address $MASTER_ADDR:6379 (head 已监听 → 成功连接)
→ [主] 启动训练
```

### 修改内容
- 主节点 Ray head 启动移到 barrier 之前
- barrier 移到 Ray head 之后、训练/副节点连接之前
- barrier 后主节点只做训练 + stop，副节点只做 connect + block

---

## 2026-07-24 B1 16GPU 训练启动全链路排障与修复（集群，SenseCore）

### 背景
当天迭代 8 次提交才跑通 16 卡训练。问题分布在多机同步、Ray 集群发现、凭证加载、脚本重组路径断裂四个层面。

### 故障 #1：torch.distributed.rendezvous() URL 缺参数
**症状**：副节点 `ValueError: rank parameter missing`
**根因**：`_train_impl.sh` 的 TCP rendezvous URL 只传 `tcp://host:port`，PyTorch 2.x 要求 `?rank=X&world_size=Y`
**修复**：URL 加 query 参数 + `|| exit 1`

### 故障 #2：Ray head 竞态
**症状**：barrier 释放后副节点立即连 Ray head，主节点 `ray start --head` 还没起来
**根因**：barrier → head start 顺序反了。副节点连不上 → Ray 集群只有 8 卡 → verl 报 `available 8 < desired 16`
**修复**：主节点先 `ray start --head`，再做 barrier，最后副节点连

### 故障 #3：torch.distributed.rendezvous() "re-rendezvous" 错误
**症状**：`RuntimeError: Unable to perform re-rendezvous using tcp:// method`
**根因**：PyTorch 新版本 `rendezvous()` 内部缓存 TCPStore，重复调用同地址抛异常
**修复**：弃用 `dist.rendezvous()`，改用 `dist.init_process_group('gloo', init_method='tcp://...')` 直接建 process group + barrier

### 故障 #4：verl 无视已有 Ray 集群
**症状**：多机同步成功、worker 已加入 Ray 集群，但 verl 仍报 `available 8 < desired 16`
**根因**：`b1_9b_16gpu.yaml` 中 `ray_init.address: local` 让 verl 每次都新建本地 Ray，无视 `ray start --head` 搭好的多机集群
**修复**：改 `address: auto`，verl 自动发现并加入已有集群

### 故障 #5：脚本目录重组导致路径断裂
**背景**：将 `scripts/` 下 64 个文件分类到 `env/` `sandbox/` `collect/` `data/` `pipeline/` `analysis/` `serve/` 7 个子目录
**症状**：
- `load_training_env.sh` 读不到 `.env` → SWANLAB_API_KEY 为空 → swanlab 认证失败崩训练
- `load_tencent_env.sh` 读不到 `docker/sandbox/tencent.env` → E2B_API_KEY 为空 → 沙箱连接失败
**根因**：这些脚本用 `dirname $0/..` 解析项目根，从 `scripts/` 下移到 `scripts/env/` 后 `..` 少跳了一级（解析到 `scripts/` 而非项目根）
**修复**：
- `load_training_env.sh` / `dev_env.sh`：`..` → `../..` + 改用 `BASH_SOURCE[0]`
- `load_tencent_env.sh`：`$0` → `BASH_SOURCE[0]` + `..` → `../..`（被 source 时 `$0` 是调用方路径）
- sandbox/pipeline/collect 下 9 个脚本同样修了 `..` → `../..`

### 故障 #6：SwanLab 认证失败崩训练
**症状**：`swanlab.error.KeyFileError: api key not configured (no-tty)`，训练直接退出
**根因**：#5 导致 key 没加载，但 config 里仍配置了 `logger: [console, swanlab]`，swanlab 初始化时尝试认证失败
**修复**：`_run_single` 加保护：key 缺失时 CLI 注入 `trainer.logger=[console]` 覆盖 yaml，不崩训练

### 改进：诊断能力
- **AFS 日志全覆盖**：`exec > >(tee -a train.log) 2>&1` —— 所有 shell 阶段输出落地，不再依赖拿不到的容器 stdout
- **日志头带 rank**：`2026-07-24T13:03:53Z host=xxx-master-0 rank=0/2 pid=1`
- **每步打点**：`ray start --head`、barrier 同步、worker 连接各阶段都有 `[train_cl]` 标记

### 改进：自动续训
`_run_single` 启动前检测 `ckpts/<exp>/global_step_*`，存在则自动 `--resume-from`，中断重启不丢进度。

### 改进：脚本系统重构
- 删 21 个冗余文件（phase 目录、train_*gpu wrapper、launch_8node 等）
- `train.sh` 统一入口，支持拓扑 + `--config`（单实验）、`--phase N`（批）、`--all`（全 Phase）
- 多机同步从一行不可读 python one-liner 改为 heredoc 三段式

### 结果
- 16 卡训练跑通（13:03），FSDP 16 路联通，LightLLM rollout 正常，SwanLab 上报就绪
- `bash scripts/train.sh 16gpu --config configs/run/b1_9b_16gpu.yaml` 不变

---

## 2026-07-27 复盘：07-23 baseline 按桶训练跑数小时后崩于 _validate（根因确认 + 修复）

### 现象
07-23 的旧「按桶逐个训练」baseline（`ckpts/qwen35_9b_b1/<桶>/`，每桶一子进程）：
coding/office/ops 三个桶各自 rollout 成功产出数万 token 的轨迹后，**在验证阶段
崩溃**，退出即失败。log 证据：`logs/experiments/qwen35_9b_b1/{coding,office,ops}/train.log`。
RunLog 之前**无此条记录**（违反硬性规则，本条补记）。

### 根因（log traceback 确认，非猜测）
```
verl ray_trainer.py:594 _validate → generate_sequences
→ cl_rollout_manager.py trajectories_to_dataproto
→ prompts[i] = torch.tensor(_left_pad(p, P, pad), dtype=torch.long)
ValueError: too many dimensions 'str'
```
`val_before_train=True` + `test_freq=2` → 训练一开就先跑验证 rollout。
`_safe_tokenize(t.messages[:1])` 调 `apply_chat_template(tokenize=True)`，当 chat_template
误返回**模板文本字符串**（而非 int id）时，`list(str)` 得到单字符 list，
`ids[0]` 是单字符 str、过不了「是否 list/tuple」的判断 → 直接返回字符 list →
`torch.tensor([...str...], long)` 报 `too many dimensions 'str'`。
崩在验证、不是启动，所以「跑了很久才失败」。

### 为什么之前查不出
- 崩点在 `trajectories_to_dataproto:99`，与本 session 修的「验证阶段 NameError」是
  **同一函数不同坑**；NameError 修复未覆盖此路径。
- 单测只覆盖 `trajectories_to_dataproto` 的 padding（喂的是干净 int list），
  从不喂「apply_chat_template 返回字符串」这一真实退化输入 → 静态查不出。

### 修复（commit 见下）
- `cl_rollout_manager._safe_tokenize`：str 返回值 → `tokenizer.encode` 重编码；
  收尾加「所有元素必须是 int」硬校验，非 int 立即抛 `TypeError`（带样本），
  不再让脏数据流到 torch.tensor 报无上下文的错。
- `inference/generate.py VerlRolloutGenerateFn`：同样加 str→encode 兜底。

### 状态：已修，本机 py_compile + 相关单测通过；真实验证待集群。

---

## 2026-07-27 根治 baseline rollout ↔ verl ×8 契约冲突 + 打通 observer→训练 judge + 关 verl 验证

### 根因(verl 源码 + 6 次崩溃 log + 子 agent 复核,三方一致)
verl 0.8.0 自己按 `rollout.n=8` 复制 gen_batch(`ray_trainer.py:1398` repeat_interleave)再交给
`generate_sequences`,并期望「进多少行返多少行」(`:1448` union 断言等行数);GRPO 由 verl 自己的
`uid`(数据集 uid ×8)在 `compute_advantage`(`:192/:206`)分组。而旧 manager 对**每个输入行**又跑
8-slot pool → `len(input)×8`,即 **verl ×8、我们又 ×8 = ×64**。训练 512→4096、验证 920→7360,
行数对不上必崩。6 次 baseline 全崩在 `_validate`(val_before_train=true 是首个动作),0 checkpoint。
另两条同路径坑:uid 冲突(union_numpy_dict 要求同名 key deep-equal)、训练 judge 看不到 observer 报告。

### 修复(方案 A:对齐 verl 契约,每输入行 1 条 rollout)
- `cl_rollout_manager.generate_sequences`:改为**每输入行 1 条**单 slot 单轮 rollout,原序返回等行数;
  scheduler `slots=1`;assert 改 `len==len(prompts)`;并发按 sessions_per_step 分批。
- `trajectories_to_dataproto`:**默认不发 uid**(verl 自己的 uid 驱动 GRPO;发了会 union 冲突),
  新增 `observer_report` 非张量列(每行 observer 的 state_diff),`uid` 仅显式传入时才写(冷/测试)。
- **打通 observer→训练 judge**:新增 `trainer/observer_reward_manager.py`(`@register("cl_observer")`,
  子类化 verl experimental NaiveRewardManager,override `run_single` 把 observer_report 折进 extra_info),
  `verl_runner.run` 里 import 触发注册;`model_reward.compute_score` 读 `extra_info["observer_report"]`
  拼进 rubric 作 completion ground truth;base.yaml `reward_manager.name=cl_observer`。
  observer 仍**只观察不打分**,只是它的取证现在真喂给了决定训练的那个 judge。
- **关 verl 内建验证**:所有 `configs/run/*_16gpu.yaml` 加 `val_before_train=false` + `test_freq=-1`
  (评测按方案训完离线统一跑,verl `_validate` 对 baseline 无产出+纯浪费+曾是崩溃触发器)。
- docstring/注释订正:16×8 winner-sync → 单轮 per-row / verl 负责 ×8。

### 验证
本机 `pytest tests/` 350 passed(唯一 fail=test_sandbox_dockerfile,既有,脚本重组删了 validate 脚本,
与本次无关)。torch-only 的 trajectories_to_dataproto 新断言(不发 uid / observer_report 逐行)本机 skip、
待集群。真实 512/920 行契约 + cl_observer manager 链路待集群全栈验。

### 状态:已修,待集群验证。baseline 排队前务必用本 commit。

---

## 2026-07-27 补 multi_modal_inputs 空 dict → 修纯文本 + agent-rollout 的 KeyError（预防性,已论证不掩盖）

### 背景
timing 修复后系统排查 verl fit() 对我们 rollout 返回值的其余无守卫读取,发现
`ray_trainer.py:1463` 无条件遍历 `batch.non_tensor_batch["multi_modal_inputs"]`。

### 根因
该 key 由 verl **默认 AgentLoopManager** 在 rollout 输出上设置,但仅
`if any(mmi is not None)`（agent_loop.py:953）——**纯文本 batch 默认 manager 也不设**,
而 :1463 却无条件读。即 verl 自身在「纯文本 + agent rollout」路径存在不一致。
我们是纯文本（195 任务,无多模态,见 CLAUDE.md）,自定义 manager 不设 → 必撞
`KeyError: 'multi_modal_inputs'`（紧接 timing 之后的第 6 个坑）。

### 修复 + 为何不掩盖真实情况
`trajectories_to_dataproto` 的 non_tensor 补 `multi_modal_inputs = [{} per row]`。
论证:
- **语义正确值**:纯文本每行「无多模态输入」的正确表示就是空 dict;verl 循环
  `if "image_grid_thw" not in mmi: continue` 会跳过空 dict → images_seqlens 为空,
  正是纯文本真值。不是假数据盖真错误（对比:补 rm_scores 假分才算掩盖）。
- **无 union 冲突**:数据集 __getitem__ 纯文本行不设此 key,主 batch 无它,
  union 直接新增(非冲突 key,不触发 deep-equal)。不覆盖任何真实数据。
- **留痕**:注释写明这是 verl 契约占位;若将来做多模态,空 dict 会显眼提示需填真值。

本机 pytest 350 passed（唯一 fail=test_sandbox_dockerfile,既有,无关）。待集群验。

---

## 2026-07-27 假成功排查：per-row 空 messages → IndexError；脚本把崩溃伪装成「训练结束」

### 现象
用户观察「系统显示成功、但 baseline 停了、0 checkpoint」。日志有 `[train_cl] master:
训练结束，ray stop` + `Stopped all 46 Ray processes`（像正常收尾），实际是**假成功**。

### 根因(两个,都修)
1. **IndexError（真死因）**：`fit():1420 → generate_sequences:393 → _safe_tokenize(t.messages[:1])
   → apply_chat_template([])`。某条 per-row 单 slot rollout **失败** → scheduler 的
   per-slot 兜底返回 **messages 为空** 的 Trajectory → `t.messages[:1]==[]` →
   transformers `conversation[0]` 越界 `IndexError`。一条坏 slot 崩掉整个 step,一步没训。
2. **脚本假成功**：`_train_impl.sh` 在 `python -m trainer.cl_main` 后**无条件**打
   「训练结束」并 `ray stop`、退出 0,**不检查 python 退出码** → cl_main 崩溃退出后
   平台仍看到 rc=0「成功」。崩溃被伪装成成功,持续误导。

### 修复
- `cl_rollout_manager.generate_sequences`：prompt tokenize 改用 `_prompt_msgs(t,i)`——
  优先用 traj 首条 message,**空则回退到已知输入 query**（trajectories 与 queries 1:1
  对齐,prompt 是确定性输入,失败 rollout 不该丢它）。保住该行(空 response 下游处理),
  不再因一个坏 slot 崩整步。
- `_train_impl.sh`：捕获 `_run_single`/`_run_buckets` 退出码,`ray stop` 后 `exit $rc`；
  失败打明确 `!!! 训练失败 rc=N` 到 stderr,不再把崩溃打成「训练结束」。

本机 pytest 12 passed（相关）/ bash -n OK / py_compile OK。待集群验。

### 进展
崩点从「rollout 中途」推进证明:per-row 契约、reward manager、timing、multi_modal_inputs
全过了,rollout 已能大量产出;唯一剩的是「失败 slot 的空轨迹」这个边界。修完后一条坏
slot 不再拖垮整步。

---

## 2026-07-27 冷采集数据完成

### 最终桶分布（质检后）

| 桶 | 采集数 | floor | 余量 |
|----|--------|-------|------|
| workflow | 386 | 163 | +223 |
| ops | 503 | 145 | +358 |
| qa | 137 | 131 | +6 |
| finance | 100 | 97 | +3 |
| office | 75 | 72 | +3 |
| communication | 76 | 72 | +4 |
| safety | 67 | 65 | +2 |
| coding | 31 | 30 | +1 |
| research | 54 | 53 | +1 |
| **总计** | **1429** | - | - |

来源：cold-only 采 6 轮 + borrow 补 3 批 (55 个 query,已从训练集排除)
actor: deepseek-v4-pro-202606, 单轮 hermes_structured
质检: Layer 1 agent-data-qc + Layer 2 agent_data_tools, Layer 3 LLMChecker 98% pass(抽样)

---

## 2026-07-27 val_files=null → NoneType 崩溃：verl_runner 兜底别名到 train_files

### 现象
新任务 1 分钟即失败(脚本已如实报 `!!! 训练失败 rc=1`,假成功修复已生效)。

### 根因(非本代码逻辑,数据配置 + verl 硬约束)
`create_rl_dataset(None)`（val_files 为 null,datasets/val.parquet 也不存在）→
`copy_to_local(src=None)` → `TypeError: 'NoneType' object is not subscriptable`
（rl_dataset.py:164 → fs.py:238）。崩在初始化 create_rl_dataset,连 train 数据都没加载。

关键:verl **硬要求** 一个可加载的非空 val dataloader——`RayPPOTrainer._create_dataloader`
无条件从 `config.data.val_files` 建 val（:341）并 `assert len>=1`（:382），**与
test_freq/val_before_train 无关**。所以「去掉 val」在 trainer 层做不到（传 val_dataset=None
verl 也会自己从 val_files 重建）。

### 修复(verl_runner,不 fork verl)
run() 与 build_trainer() 在建 dataset 前:若 `data.val_files` 为空 → `OmegaConf.update`
别名到 `train_files`。这是可加载的占位:in-loop 验证已关（test_freq=-1 /
val_before_train=false,评测走训完离线），val dataloader 只为满足 verl 契约而建、
从不被迭代/用于验证。设在 config.data 上,verl 内部 :341 重建也看到。

不掩盖真实情况:验证本就不跑,占位 dataloader 永不迭代;若将来要真验证,配 val_files
即覆盖此兜底。本机 pytest 12 passed / py_compile OK。

---

## 2026-07-27 系统审查 + 修 run_step crashed-session 行数漂移(会崩整步)

### 独立确认的真隐患(源码级,非猜测)
`rollout/scheduler.py::run_step` 的 session 级异常兜底原返回 `return i, []`：一个 session
崩(如 e2b 超时/observer/sync 异常)→ 该 spec 贡献 0 条轨迹 → `all_trajs` 少一条 →
`generate_sequences` 的 `assert len(all_trajs)==len(prompts)` 失败 → **整个训练 step 崩**。
即「隔离了异常但没隔离行数」——一个坏沙箱仍拖垮整步。上次 run 恰好 0 session crash 才没触发,
64 并发长跑下沙箱失败几乎必然。

### 修复
crashed session 返回 **1 条占位轨迹**(空 messages、reward=None、error meta)而非 []。
per-row 单轮契约 = 每 spec 恰 1 条,占位保住行数;下游已安全处理空 messages(prompt 回退到
query、空 response→reward 0、GRPO std 有 epsilon 防 NaN)、pick_winner 视 reward=None 为
scorer-error。真正做到「一个坏 slot 不崩整步」。

### 顺带排除的两个担心(源码确认安全,无需改)
- ObserverRewardManager 不定义 __init__,继承父类 (config,tokenizer,compute_score,...),与
  verl load_reward_manager 的 config=/tokenizer=/compute_score=/**kw 完全匹配。
- GRPO 全相等 reward 组:core_algos.py:317 单条组 std=1,:326 (score-mean)/(std+epsilon=1e-6),
  不 NaN 不崩(零优势=不产梯度但安全)。

（另有后台 4 路审查 workflow 复核中,结论到后追加。）

---

## 2026-07-27 系统审查(4路workflow)后修复 R1-R6:reward 接上 + 续训 + 验证语义 + buckets 护栏

后台 4 路审查 + 复核查出 6 条真问题(逐条源码核实),按此修:

### R1(根因,reward 阶段必崩):rm_scores 从不产生
verl use_rm=false → 假定 rollout 已带回 rm_scores(默认 AgentLoopManager 在
_postprocess 写,agent_loop.py:933),直接 extract_reward(batch["rm_scores"])。我们
自定义 rollout 替换了该路径、从不写 rm_scores → reward 阶段 KeyError。且发现
observer→ObserverRewardManager→compute_score 那条 verl reward-manager 路径是**死代码**
(reward 走 rm_scores,不调 manager)。
关键:reward **已在 rollout 内联算好**(_score_all_slots → score_followup(observer diff+judge)
→ t.reward)。修:trajectories_to_dataproto 加 rewards 参数,把 t.reward 写进 rm_scores
[B,R](最后有效 response token 处,照 verl 放法);generate_sequences 传 t.reward。
reward=None/空 response → 该行全 0(NaN-safe)。base.yaml reward_manager 回 naive
(rm_scores 已存在,naive 不会被调);observer_reward_manager.py 标注 deprecated/未接入。

### R4(续训必崩):trainer.load_checkpoint() 不存在
verl 只有私有无参 _load_checkpoint;fit() 在 resume_mode 下自动续。改:resume_from
给定时设 resume_mode=resume_path + resume_from_path,删手动调用,用 verl 原生续训。

### R6(4gpu 语义偏离):b1_9b.yaml 验证没关
加 val_before_train=false + test_freq=-1(原 test_freq=20 + 继承 val_before_train=true
会跑验证并更早触发 R1)。

### R2/R3/R5(--buckets 路径,加护栏不深修):
R3 datasets/baseline_9b 空(parquet 在 _archive_multiturn_20260723)、R2 桶行数<batch64
→ dataloader empty 断言、R5 共享 ckpt+resume_mode=auto 第2桶起误续训毁遗忘语义。
_run_buckets 加 fail-loud 护栏(空目录/缺文件直接报错退出)+ R2/R5 显式 WARN;按桶
训练方案本身待重构(TODO)。16gpu 单次训主线不走 --buckets,不阻塞当前。

### R7/R8(MINOR,记录):多机 worker --block 无 rank0 联动 hang;16gpu config 未接
cluster.yaml 致 CLI 并行/batch 参数被 config 写死值静默吃掉(当前数值巧合一致不崩)。

本机 pytest 350 passed(唯一 fail=test_sandbox_dockerfile 既有无关)/ py_compile / bash -n OK。
rm_scores 形状/落位靠新单测(fake DataProto)+ 对齐 verl agent_loop.py:933 保证,数值链待集群。

---

## 2026-07-27 除零/特殊值(NaN/inf)专项排查:查全所有除法+归一化,修 2 处漏洞

用户提问「除数这些易错地方是否做了范围和特殊值校验」。系统扫了整条链路的除法/
归一化/分母/clip/空集合聚合。

### 已良好保护(逐个确认,无需改)
- weighting._clip_and_normalize:空 flat 早返回;total<=0 不除。
- weighting._percentile:空列表→0.0。
- priority.forgetting_risk:len==0 + n==0 双重保护 + clip(0,1)。
- priority.rarity:1/(1+log(1+count)),分母>=1。
- priority peer 相似度:sqrt(sum) or 1.0 防零范数。
- model_reward.aggregate:纯乘无除。
- GRPO advantage:verl core_algos.py std+epsilon(1e-6),单条组 std=1。

### 修复的 2 处漏洞(NaN 传进 verl loss,训练中途才炸,最难查)
1. `_clamp01`(model_reward)/`_c01`(agents/reward):`nan<0` 与 `nan>1` 都是 False →
   **NaN 原样漏过** → judge 返回 NaN 字段 → reward NaN → rm_scores NaN → verl loss NaN。
   修:math.isnan 检测,NaN→0(inf 仍走 >1→1.0)。本机验证 aggregate(nan)→有限值。
2. `inference/generate.py`:rollout_log_probs 只校验长度,**不校验 NaN/inf**。lightllm 返
   NaN logprob → verl 重要性比 NaN → loss NaN。修:非有限则丢弃该步 logprobs(verl 重算
   old_log_probs,安全)。
3. 加固:trajectories_to_dataproto 写 rm_scores 时 math.isfinite 兜底(非有限→0+warn)。

新增 test_aggregate_nan_inf_are_finite。本机 pytest 351 passed(唯一 fail=
test_sandbox_dockerfile 既有无关)。

---

## 2026-07-27 非-reward 模块除法/特殊值二次审查(sampler/bucket/eviction/cl_loss)

用户要求除 reward 外其余参与计算模块尤其分母都审。派 agent 逐行扫 replay_buffer/* +
trainer/replay_* + cl_loss。

### 已确认良好保护(不改)
eviction.py:98(if size<threshold 守卫)、store(空守卫,无统计除法)、
replay_forward.py masked mean(.clamp(min=1))、replay_metrics(.clamp + if fill_ratios)、
replay_batch(warmup>0 守卫)、cl_loss(_replay_is_empty 早返回 + 纯乘)、
bucket.reservoir(max(n,1))、weighting(已审)。dead code: sampler._weighted_choice_
without_replacement 无调用点。

### 修的 3 处(与 reward NaN 同源:NaN 绕过 min/max/==0 比较)
1. sampler._sample_one_within_bucket(:320,最易触发):priority 由 GPU log-prob drift
   算,NaN/inf 时 max(nan,1e-9)→nan → rng.choices 报 "weights must be finite" 崩整个
   桶采样(主训练路径)。修:_safe_prio 用 math.isfinite,非有限/≤1e-9→1e-9(退化为~uniform)。
2. bucket.allocate_quota:全 0 counts → sum(n^alpha)=0 → ZeroDivisionError(构造期崩);
   空 counts → max(range(0)) ValueError;0**负alpha 崩。修:空→[]、s<=0→均分、
   n^alpha 对 (n<=0 且 alpha<=0) 置 0 权重。
3. sampler TwoLevelSampler.sample(:277):bucket 权重 NaN(corrupt coords)绕过 wsum==0 →
   rng.choices 崩。修:非有限/≤0 权重→0,wsum<=0 返回 []。

新增 3 回归测试(NaN priority 不崩、全0 quota 均分、空 quota→[])。
本机 pytest 354 passed(唯一 fail=test_sandbox_dockerfile 既有无关)。

### 判为误报/不改
sampler:151 mean_d 分子分母口径(相对权重,rng.choices 不在乎绝对尺度,不影响正确性)。

---

## 2026-07-27 k2(kl_medium)失败:val.parquet 缺失 → val_files 兜底覆盖不全的漏修

### 现象
qwen35_9b_k2_16gpu(=中 KL,kl_loss_coef=0.05,即用户口中的 "kl_medium")13:59 启动,
14:00 即失败,脚本如实报 rc=1(退出码检查生效)。

### 根因
`FileNotFoundError: datasets/val.parquet`(create_rl_dataset 建 val 数据集)。
`datasets/val.parquet` 实际不存在(只有 train.parquet)。之前修 val_files 兜底时**只改了
b1 的 config(val_files: null)+ verl_runner 只处理"空/null"**,但:
- 其它 16 个 k*/r* config 的 val_files 仍**硬指向不存在的 val.parquet**(漏改);
- verl_runner 兜底只挡"空",不挡"非空但文件不存在" → k2 走 create_rl_dataset 崩。
即上一轮 val_files 修复覆盖不全。

### 修复
1. verl_runner.run + build_trainer:兜底条件从"空"扩到"空 OR os.path.exists 为假",
   两种都别名到 train_files(验证已关,占位不用于验证)。
2. 全部 18 个 *_16gpu.yaml 的 val_files 统一改 null(与 b1 一致,显式表达"无独立验证集")。

本机 pytest 354 passed(唯一 fail=test_sandbox_dockerfile 既有无关)。

---

## 2026-07-27 k3(kl_strong)/所有开 KL 实验崩:ref 段残留 path/model 键 → FSDPActorConfig 崩

### 现象
qwen35_9b_k3_16gpu(强 KL,kl_loss_coef=0.10,即 "kl_strong")14:09 启动,init_workers →
actor_rollout_wg.init_model() 崩:`TypeError: FSDPActorConfig.__init__() got an
unexpected keyword argument 'path'`(脚本如实报 rc=1)。b1(baseline)同 session 同结构
却跑到 Training Progress——差异仅 use_kl_loss。

### 根因(逐层核实)
verl 0.8.0:ref policy 融合进 actor worker,权重路径**统一取共享 actor_rollout_ref.model.path**
(main_ppo.py:264),ref 段 schema 为 FSDPActorConfig,**不接受 `path` 或 `model` 子键**。
但我们的 config:
- base.yaml 有扁平 `ref.path: .../Qwen3.6-27B`(旧基座残留);
- run config 有 `ref.model.path: .../Qwen3.5-9B`。
两者合并进 ref → 开 KL 的实验(K1/K2/K3/K2-R/R4-K…)构造 ref 时 FSDPActorConfig 收到
`path`(和 `model`)非法关键字 → 崩。**b1 无 KL 不建 ref,故那些残留键无害、不崩**——
这解释了为何 baseline 能跑、所有 KL 实验都崩。

### 修复
- base.yaml:`ref: path: 27B` → `ref: {}`(删扁平 path;ref 用共享 model.path=9B)。
- 全部 run/*_16gpu.yaml:删 ref 段下的 `model:\n path: 9B`(redundant+同样非法);保留
  ref 的合法键 log_prob_micro_batch_size_per_gpu / fsdp_config。
- resolved 校验:6 个实验 ref 段均无 path/model,共享 model.path=9B。

本机 pytest 354 passed。所有开 KL 实验(k1/k2/k3/k2-r/r4-k)不再因 ref path 崩。

---

## 2026-07-27 rollout 并发 64→256(减 rollout 批数,提吞吐)

rollout 是当前瓶颈(一个 step 的 512 条轨迹按 64 并发要 8 批串行、十几分钟;训练更新才几秒)。
per-row 模式下并发 = sessions_per_step(每 session 1 条轨迹)。提到 256:512条/step 分 2 批
(原 8 批),rollout 吞吐 ~4×。

必须三件套一起改(否则光提 sessions_per_step 会卡在 lightllm 的 64、请求排队=假并发):
- agent.sessions_per_step: (默认64)→ 256
- lightllm running_max_req_size: 64→256(推理端一次并发解码上限)
- lightllm graph_max_batch_size: 64→256(cuda graph 支持 256 batch)
- gpu_memory_utilization: 0.40→0.55(给 KV 池更多显存;256 峰值 KV ~125k token vs 池上限
  远够;整卡余量大 ~20GB/80GB)

18 个 *_16gpu.yaml 一致;YAML + 值校验通过。
待集群验证的前提:e2b 沙箱配额需 ≥256(256 并发=一次开 256 沙箱);若配额不足会大量
crashed-session 占位(空轨迹 reward 0),需回调 sessions_per_step。稳妥档,未上 512。

---

## 2026-07-27 baseline 训练阶段 CUDA OOM(update_actor)+ 一把改:治 OOM + util 0.6 + 并发 512

### 现象
b1(13:59 老进程)跑到 _update_actor → update_actor 时 `torch.OutOfMemoryError: CUDA
out of memory. Tried to allocate 12.97 GiB. GPU0 total 79.32GB, 训练进程已占 63.36GB`。
**崩在训练阶段(非 rollout)**,推翻"显存宽松"的乐观估计——长序列(53886)+ micro=2 的
激活峰值 + FSDP all-gather + log_prob 计算,单卡训练峰值 ~76GB→爆。ckpt 空。

### 决策(用户:一把全改)
治 OOM + 提 util 0.6 + 提并发。**未开 use_remove_padding**:它依赖 flash-attn varlen,
而本模型是 sdpa + 混合 GatedDeltaNet(GDN 自定义线性注意力,日志已显示 fla-org/causal-conv1d
未装走 torch fallback),开 rmpad 高风险崩在诡异处;改用更稳的 micro/token 手段。

### 改动(18 个 *_16gpu.yaml)
治 OOM(砍训练激活):
- ppo_micro_batch_size_per_gpu: 2→1(update_actor 激活峰值减半,最直接)
- ppo_max_token_len_per_gpu: 65536→32768(每卡单次前向 token 上限减半)
- (log_prob_micro_batch_size_per_gpu 已是 1;param/optimizer_offload 保持 true)
提吞吐:
- gpu_memory_utilization: 0.55→0.6(训练时 lightllm sleep 让出,不占训练;更大 KV 池)
- sessions_per_step: 256→512 + lightllm running_max_req_size/graph_max_batch_size: 256→512
  (rollout 并发 512 → 512条/step 分 1 批,原 8 批)

估算:训练峰值 micro=1 后 ~45-55GB/卡(有余量);rollout util 0.6 = 48GB lightllm + offload
训练态(CPU)+ 系统 ≈ 53GB,够。本机 pytest 354 passed。

### 待集群验证的风险
1. micro=1 是否真把训练压进 80GB(激活估算不确定,若仍 OOM 需再降 max_token 或 batch)。
2. e2b 沙箱配额需 ≥512(512 并发=一次开 512 沙箱),不足则大量 crashed-session 占位。

---

## 2026-07-28 baseline 崩:sessions_per_step 放进 agent 段被 verl 严格 dataclass 拒绝

### 现象
02:39 启动(新配置 micro=1/util0.7/并发512),init 即崩:
`TypeError: AgentLoopConfig.__init__() got an unexpected keyword argument 'sessions_per_step'`。

### 根因
上一轮把 rollout 并发写成 `actor_rollout_ref.rollout.agent.sessions_per_step: 512`,但
verl 的 agent 段是 AgentLoopConfig(BaseConfig dataclass,严格),不认 `sessions_per_step`
这个自定义 key → 构造即崩(同 ref.path/FSDPActorConfig 那类 verl 严格解析问题)。
sessions_per_step 是我方 cl_rollout_manager 读的,不是 verl 字段。

### 修复
- config:agent 段改用 verl 合法字段 `num_workers: 512`(AgentLoopConfig 有此字段;
  它另作验证路径 pad divisor,但验证已关 test_freq=-1 无副作用)。删 sessions_per_step。
- 代码 cl_rollout_manager._build_scheduler:并发读 num_workers(fallback sessions_per_step
  →64)。agent_cfg 是 BaseConfig(Mapping),.get 对 num_workers/sandbox_backend/k_max
  都安全(非字段返回默认,不崩)。

本机 pytest 通过。18 config 一致:agent.num_workers=512。

---

## 2026-07-29 迁移:自写 rollout → verl 原生 main_ppo + custom_sync(recipe_custom)

### 背景
本 session 训练一路崩(§22-§29),病根=自写 rollout(cl_rollout_manager+collect.py,~6000行)
数据质量(超长 assert/漏 logprob/激活爆显存)。用户已单轮化、无 Questioner → 全量切 verl
原生 agent_loop。对齐参考脚本 debug_rl_qwen35_9b.sh(同 Qwen3.5-9B 已验证):
`python -m verl.trainer.main_ppo` + use_v1 + trainer_mode=custom_sync + RemoteAgentLoopManager。

### 关键更正(推翻上一版 f72d3de~27d5e45)
上一版用 RayPPOTrainerV1 桥接 + 默认 RolloutManager —— 错。默认 manager 跑的是 ToolAgentLoop
(本地 tool),非沙箱 harness。E2BAgentRunner/HermesHarness/gateway 反向隧道那套活代码的入口
是 RemoteAgentLoopManager,必须显式配 agent_loop_manager_class。已回退覆盖。

### 六阶段(本机能验全绿:354 单测无回归 / config 全解析 / 真 tq KVBatchMeta.concat 冒烟)
- A 入口:verl_runner.py 新增 CLTaskRunnerV1(复刻 verl TaskRunnerV1 三步 init→
  init_agent_loop_manager→fit,init 后 fit 前注入 CL);run_cl_ppo 改调 main_ppo.run_ppo(
  task_runner_class=CLTaskRunnerV1)。_generated_ppo_trainer.yaml 覆盖为 verl 当前版(含 v1 段)。
- B config:b1_9b_16gpu.yaml 加 use_v1/trainer_mode=custom_sync/agent_loop_manager_class=
  RemoteAgentLoopManager/remote_agent(gw8/sw8/to1000)/transfer_queue.enable。
- C CL loss:set_loss_fn 注入(engine_workers.py:496 v1 仍在);CL loss 已是 v1 签名
  (cl_loss.py:4 = ppo_loss(config,model_output,data,dp_group))几乎零改。b1 走 no_replay。
- D 9桶 buffer:【用我们自己方案,不子类化 v1 ReplayBuffer(那是 online off-policy 采样器)】。
  cl_replay_hook_v1.py:hook custom_sync _update_actor(KVBatchMeta);PRE kv_batch_put 回放张量
  +KVBatchMeta.concat 掺行,POST kv_batch_get_by_meta 抽 winner 入库。trajectory_adapter_v1.py
  从 KVBatchMeta 抽轨迹。b1 不触发(buffer.enabled=false),仅 R 系列。
- E fs-seed:cl_agent_dataset.py CLAgentDataset(RLHFDataset),按 record_id 注入
  data/taskspecs_w3/<rid>/files 到沙箱 ./inputs(agent_assets/type=dir)。保持本地 parquet,
  不依赖 aoss_client/OSS。config data.custom_cls 指向它。前500条465命中输入文件目录。
  数据流确认:dataset→collate→session_worker(worker.py:497 白名单含 agent_assets)→
  E2BAgentRunner.write_agent_assets。
- F observer+reward 回流:observer_hook.py ObserverDiffHook(AgentRunHook,【零 LLM】,复用
  agents/observer.py 探针+diff)。prepare 拍 before、run 拍 after+diff→reward_info
  ["observer_report"]。沙箱关闭前当场 diff(hook.run 在 sandbox.close 前),结果随 reward_info
  带走(不需沙箱活到 reward 阶段)。model_reward_omni.py 零侵入从 omni 传的
  data_non_tensor_batch["reward_info"] 取 observer_report/answer_key 并进 extra_info→judge。
  judge 输入=actor 完整轨迹(response_ids 多轮 token,trajectory_buffer.record_turn 累进)+
  diff 铁证。observer_hook_register.py monkey-patch hook factory 认 FQN(不改 verl 源码,
  与 reward _function_name/dataset custom_cls 同类机制),FQN 挂载 patch 三分支冒烟通过。

### 全轨迹确认(源码链路)
gateway/response_decoder.py 解析每轮 LLM 响应→reasoning_parser 拆思考链+llm_tool_parser 解
tool_calls;trajectory_buffer.record_turn 把 context_tokens(mask0)+response_tokens(mask1)累进
response_ids,message_history 保留完整多轮。→ 训练看到的是完整多轮(system+user+assistant含
tool_calls+tool 响应),非 system+user。冷启动 1429 条只有(system,user)是纯 query 池(未采集),
性质不同。

### 待集群实测(各文件 CLUSTER-TODO)
custom_sync 起链路 / RemoteAgentLoopManager 起沙箱 / buffer 回放行字段与 tq 对齐 / observer
探针在 agentic-cl-sandbox 镜像跑通 / hook FQN 挂载 / 端到端 ppo_kl 非0+reward 有差异+不 OOM 不 assert。

### 待办
- verl/lightllm 指向:VERL_DIR=dependencies/verl,LIGHTLLM=workspace/LightLLM(_train_impl.sh)。
- 遗留坏测试(引用已删脚本 validate_sandbox_dockerfile.sh/prepare_queries/qc_trajectory/
  convert_dataset)非本次改动,待清理。

---

## 2026-07-29 集群 16卡 端到端拉起：迁移后逐关口 debug（append-only 证据）

自写 rollout → verl 原生 main_ppo+custom_sync 迁移后，首次上 16卡2节点集群端到端。
逐个关口崩→修，每关都比上一关更深（证明修复有效在推进）：

### 关口与修复（按崩溃先后）
1. **import verl 崩 TE**：`transformer_engine has no attribute 'pytorch'`。
   根因=镜像 flash_attn 是残缺 shim(仅 bert_padding)，recipe_custom→megatron→TE 要
   `flash_attn.flash_attn_interface`，shim 无 → 崩。真 FA3 装在 site-packages/flash_attn_3/。
   修=给 shim 加 flash_attn_interface.py(转发 flash_attn_3 + __getattr__ 兜底) +
   _train_impl.sh PYTHONPATH 前置 AFS shim(docker/qwen36-lightllm/flash_attn_shim)。
2. **数据加载崩**：`assert src[-1]` NoneType。根因=v1 _init_dataloader 无条件建 val dataset，
   我们 val_files=null。修=CLTaskRunnerV1.run 里 val 空/缺失 alias 到 train_files。
3. **lightllm 起 server 崩**：`OSError:98 Address already in use`。根因=recipe_custom
   async_lightllm_server 的 pd_master_port(1212)/multinode_httpmanager_port(12345)/
   multinode_router_gloo_port(20001) 用 argparse 写死 default，16卡=8replica/每节点4个同
   端口撞(迁移前用 verl.experimental v0 AgentLoopManager 那套端口对，v1 recipe_custom 这套有 bug)。
   修=三端口改 get_free_port 动态分配(async_lightllm_server.py)。验证:重启后 8 replica 各拿
   独立端口(41031/41101/38531...)不撞，KV池 profile 成功(max_total_token_num=3022788)。
4. **rollout 崩**：`Unknown post-run hook: 'trainer.observer_hook.ObserverDiffHook'` →
   All rollouts failed 32/32。根因=observer FQN patch(monkey-patch create_hook)只在 driver
   进程生效，create_hook 实际在 AgentSessionWorker(另一 ray actor 进程)调，patch 传不过去。
   修=(a)直接改 recipe_custom hooks/factory.py 内建 FQN 分支(name 含"." → load_class_from_fqn)，
   跨进程天然生效；(b)verl_runner 把 PYTHONPATH 透传进 ray runtime_env.env_vars(verl
   get_ppo_ray_runtime_env 不传 PYTHONPATH，否则 worker import trainer.observer_hook 会 ImportError)。

### 已跨过的关口(证明链路通)
import verl ✅ / 模型加载(Qwen3_5 9.41B)✅ / FSDP+16卡组网(Gloo 15 peers)✅ /
lightllm 8 replica 起+端口不撞+KV池 ✅ / 进 rollout 沙箱采样(AgentSessionWorker+fs-seed
注入 ./inputs)✅。observer hook 修复后重启验证中(14:46 任务)。

### 集群多机(手动组 Ray)
- 网络隔离(本地 ssh 不到集群)，改动全落 AFS、集群即见。
- master IP 10.120.x(每次新机变)，worker→master 6379/29503 TCP 通(ping 不通但 TCP 通，禁 ICMP)。
- scripts/train_manual_2node.sh(RANK=0 master/RANK=1 worker)；平台也注入 SENSECORE_* 主机名。
- 每次重启是新机:镜像自带依赖 + AFS 上 shim/verl 补丁/PYTHONPATH 生效，无需每机手动配。

### 遗留(不阻塞跑通)
- fla/causal-conv1d torch fallback(GDN kernel 慢，CUDA13 编译不出，需重 build 镜像补)。
- FlashInferAllReduce disabled / MFU=0(custom_language_model 不在 MFU 表) —— 均无害。

## 2026-07-30 4卡/16卡 9B 训练崩 compute_log_prob(triton CE assert)——model_type 放错层

- **现象**：`logs/experiments/qwen35_9b_b1_4gpu/train.log` step 0 的 `_compute_old_log_prob` 崩，
  `verl/utils/kernel/kernels.py:582 assert hidden.shape[0]==labels.shape[0] and hidden.shape[1]==weight.shape[1]`
  AssertionError（4 个 rank 全崩）。栈顶是 `dense_common.py:189 forward_with_triton_backend`。
- **根因（代码 + 日志双证）**：`configs/run/b1_9b_{4gpu,16gpu}.yaml` 把 `model_type: custom_language_model`
  写进了 `actor_rollout_ref.model.override_config`，而参考脚本 `debug_rl_qwen35_9b.sh` 是放在**顶层**
  `+actor_rollout_ref.model.model_type=`。两处语义完全不同：
  1. **顶层 `model.model_type`** = 引擎选择键。`engine_workers.py:128 EngineRegistry.new(model_type=config.model_type)`
     用它选 `CustomFSDPEngineWithLMHead`（recipe_custom，GDN 变长+路由重放）；该引擎 `__init__` 随即把
     `hf_config.model_type` **复位回 `language_model`**，所以 `apply_monkey_patch` 读到的是真实 `qwen3_5`，
     走 `qwen3_5.py::forward_with_triton_backend`——SP>1 时对 `rolled_labels` 也做 `ulysses_pad_and_slice_inputs`。
  2. **`override_config.model_type`** 会经 `verl/utils/model.py::update_model_config` 直接改 **HF config.model_type**
     → `custom_language_model`。于是 `apply_monkey_patch`（`monkey_patch.py:270` 按 `model.config.model_type` 分派）
     匹配不到 `qwen3_5`，落到 `else` 分支 `dense_common.py::forward_with_triton_backend`——**该分支不对 labels 做 SP-slice**。
     SP=2 下 hidden 被切成 1/SP（行数 = total_nnz/2 + pad），labels 仍是全长 total_nnz → 行数不等 → triton CE assert 崩。
  - 日志实证：`train.log:216` `override_config` 里含 `model_type: custom_language_model`；`:1046` 打印
    `Using Triton backend ... Qwen3_5ForConditionalGeneration`（说明 patch 确实生效但走了通用分支）；
    `grep "enable_routing_replay in CustomFSDPEngineWithLMHead"` = **0 次** → 证明 `CustomFSDPEngineWithLMHead`
    根本没被选中（退化成 base `FSDPEngineWithLMHead`，路由重放也一并丢失）。即"size 不匹配"与"自定义引擎没生效"
    是同一个错配的两个后果。
- **修复**：两份 config 把 `model_type: custom_language_model` 从 `override_config` 提到 `model:` 顶层，
  `override_config` 只留 `attn_implementation: flash_attention_3`。`trainer/cl_main.py::load_config` 用 OmegaConf.merge
  且 `model_type` 是 `HFModelConfig` 合法字段（默认 `language_model`），加顶层键无 struct 冲突。
  改后 `load_config` 校验：两份 config `model.model_type=custom_language_model`、`override_config={attn_implementation:...}` ✓。
- **状态**：config 已改已校验；重跑 4 卡 debug 待执行（同代码路径，结论适用 16 卡）。

### 2026-07-30 订正:上条"size 不匹配"机制描述有误
上条我写的"hidden 切 1/SP、labels 仍全长"是错的(未 trace 代码的猜测)。逐行 grep 后的**准确机制**:
- SP 切分在引擎层 `prepare_model_inputs` 做,`input_ids`(`transformer_impl.py:1106`)和 labels
  (`input_ids_rmpad_rolled`,`:1112`)**两个都** `ulysses_pad_and_slice_inputs` 切成 `total_nnz/SP+pad`;
  fused 路径 `:1211` 把已切好的 local labels 作 `shift_labels` 传入。故**两条前向路径拿到的 labels 都是 local 长度**,不存在"labels 全长"。
- 真正差异在 **`cu_seqlens` 有没有 thread 进模型**。Qwen3.5 是混合结构(GDN 线性注意力层 + full-attn 层):
  full-attn 靠 ulysses monkey-patch(all-gather→变长 FA→scatter)两条路都有;但 **GDN 层
  `qwen3_5_gated_delta_net_forward` 必须拿 `cu_seqlens` 才能正确做变长 packed + SP 分片记账**
  (`qwen3_5.py:210-226` 看到 local seq_len != cu_seqlens[-1] 就建 CP context 按 local 产出)。
  - 正确路径 `qwen3_5.py::forward_with_triton_backend` **传** cu_seqlens 给 `self.model`;
  - 错配后落到的 `dense_common.py::forward_with_triton_backend` 里 `forward_base_model` 签名**根本没有 cu_seqlens**(grep 确认)。
- 结论:`model_type` 放 override_config → HF config 被改 → monkey_patch 匹配不到 qwen3_5 → 前向换成通用
  dense_common 版 → **GDN 层拿不到 cu_seqlens → 变长 packed+SP token 记账错乱 → hidden token 数与 local labels 对不齐**
  → `linear_cross_entropy` assert `hidden.shape[0]==labels.shape[0]` 崩。(未实测具体数值差,但链条来自逐行代码。)

## 2026-07-30 全量对齐:19 份 9B + 27B/64GPU(cluster.yaml) 迁到新路线

- **背景**:修 §31(model_type 放错层)后,发现 configs/run 下其余配置仍在旧路线,与已迁的
  b1_9b_4gpu/16gpu 不一致。用户决策:9B 其余全对齐、27B/64GPU 也对齐(64GPU 仅 27B 无 9B)。
- **旧路线特征(待迁)**:`strategy=fsdp` + `agent_loop_manager_class=trainer.cl_rollout_manager.AgentLoopManager`
  (自写 winner-sync rollout)+ 无 `use_v1/transfer_queue/use_remove_padding` + `use_fused_kernels=false`(k*/r*)
  或 `true 但 impl_backend=torch 默认+无 model_type`(cluster)+ `attn=sdpa`。且 `_train_impl.sh` 已导
  `VERL_USE_EXTERNAL_MODULES=recipe_custom.bootstrap`(新路线),旧 config 与启动 env 不自洽。
- **动作**:
  - 17 份 `{k1,k2,k2-r,k3,r0-03,r0-08,r0-10k,r0-25k,r3,r4,r4-k,r4-w,r5,r6,r7,r8,r9}_9b_16gpu.yaml`:
    以 `b1_9b_16gpu.yaml` 为模板用一次性生成器 `scripts/_migrate_9b_configs.py` 重写——机制层
    (model/actor/rollout/trainer/transfer_queue)逐字对齐标准答案,**只保留各自 CL/KL 语义**
    (use_kl_loss/kl_loss_coef、lambda_replay、buffer.{enabled,num_buckets,total_capacity,priority_type}、
    weighting.{scheme,gamma,delta,...}、experiment_name)。生成器已删(一次性工具)。
  - `configs/cluster.yaml`(27B/64GPU 共用引擎 overlay):同法迁新路线——model 加 `model_type` 顶层 +
    `impl_backend=triton` + `use_remove_padding` + `attn=flash_attention_3`;actor `fsdp→fsdp2` +
    torch_compile off + clip_ratio_low + reshard/model_dtype;rollout 换 RemoteAgentLoopManager +
    custom.remote_agent(gateway/worker 8);reward_manager=omni;加 `use_v1:custom_sync` + `transfer_queue`;
    data.max_response_length 8192→12288。`b1.yaml`/`r4.yaml`(27B thin overlay)自动继承,无需单独改。
  - Qwen3.6-27B 经 config.json 确认**同为 qwen3_5 混合 GDN**(full_attention_interval=4),故 §31 修复同样适用。
- **暂不动**(用户指示):`b1_9b.yaml`(4卡SP1)/`b1_9b_8gpu.yaml`(8卡SP2)——非 train.sh 启动路径的本地测试残留;
  `b1_8b.yaml`(8B 另一模型,不在本次范围)。
- **校验**:`load_config` 全量扫描 21 份(2 份 4/16gpu + 17 份 k*/r* + b1/r4)全 OK:model_type 顶层、
  override_config 只剩 attn_implementation、fsdp2、fused+triton、use_v1、RemoteAgentLoopManager。0 失败。
- **状态**:config 全绿;真机重跑(4卡 debug 先行,再 16/64)待执行。

## 2026-07-30 长度/并发按权威参考(wuzehuan debug_rl_qwen35_9b.sh)统一 19 份 9B

- **背景**:上一轮 9B 迁移时对齐的是 dependencies 里的旧版参考,长度/后端与**权威参考**
  `/mnt/afs_toolcall/wuzehuan/Documents/verl/recipe_custom/scripts/e2b_agent/debug_rl_qwen35_9b.sh`
  不一致。本轮按权威参考统一(用户决策,逐项确认)。
- **改动(19 份:17 k*/r* + b1_9b_16gpu + b1_9b_4gpu)**:
  - 长度**照搬参考**:max_prompt_length 4096→**8196**;max_response_length 12288→**65536**;
    rollout.max_model_len/response_length 65536/12288→**131072/65536**;
    ppo_max_token_len_per_gpu = log_prob_max_token_len_per_gpu = **131072/SP**(16卡SP4=32768、4卡SP2=65536)。
    ★ 之前缩到 12288 是 §28/§31 **错误归因**下的规避——§31 查明崩溃真因是 model_type 放错层(GDN 丢
    cu_seqlens),非长度;权威参考同代码路径已验证能跑 131072,故恢复设计长度。§28 的 total_nnz 现象
    改由真机 4 卡 debug 验证,不再预防性缩短。
  - 后端**照参考**:sampling_backend triton→**flashinfer**;update_weights_bucket 6144→**3072**。
  - **不照抄**(规模相关,保留):rollout.n=**8**(项目 GRPO traj/query 设计,参考的 2 是 debug 省算力);
    running_max_req_size/graph_max_batch_size 按各配置 **train_batch×n** 保留(16卡256、4卡32)——参考的
    32 是 1 卡 debug 值,抄了会让 16 卡并发压到 32、rollout 串 8 批慢 8 倍。
  - nccl_timeout:**多机(16卡)3600 / 单机(4卡)1200**——多卡同步超时阈值(跨节点+131072 长序列 all-gather
    易超默认 600s);设大不影响正常速度,只在真卡住时晚报错。
- **校验**:load_config 全 19 份 OK,长度/ppo/后端/并发逐项核对无误,0 失败。
- **27B/cluster.yaml**:用户指示后续再改,本轮未动。
- **状态**:config 全绿;真机 4 卡 debug 验证 65536 长序列是否触发 §28 total_nnz 现象,待执行。

## 2026-07-30 §31 修复验证通过 + replay 动态比例 + 超长单条丢弃

**验证(4卡 debug b1_9b_4gpu，接 §31/§32 长度对齐后首跑)**:
- `Training Progress` 到 step 2，metrics.jsonl 落盘 2 条 → **§31 崩溃(model_type 放错层)彻底修复**：
  compute_log_prob/update_actor 全过，长序列不再崩。step1 单步 1665s（rollout/feed 943s +
  update_actor 516s + log_prob 198s），MFU 0.135，throughput 596 tok/s。
- 实测长度：resp_mean ~1.5-2万 / resp_max step1=74346 step2=93183；prompt_max step1=88857
  step2=102205；单条轨迹总长可达 15万+。显存 gpu_alloc 54.8→58.7GB / resv 60.6→66.0GB
  （dynamic_bsz 按 token 预算切 micro-batch，显存不随 batch 增大线性膨胀）；CPU 166→172GB（双 offload 代价）。
- flashinfer 后端验证通过并被选中（sampling_backend=flashinfer 生效，base 镜像自带）。
- rollout 侧偶发 1 条 HERMES_TIMEOUT(900s, NO_LOG，同任务另一 session 599s 正常完成 →
  沙箱端偶发起不来，非任务问题)；超时不重试、丢弃，min_group_success_ratio=0.5 兜底，不影响训练。

**改动 1：replay 按新轨迹数动态比例(trainer/cl_replay_hook_v1.py)**
- 新增 `cl.replay_ratio`(默认 5 = 新:旧 5:1)。每 step 回放量 = ceil(本步非padding新轨迹数 / ratio)，
  取代原固定 replay_batch_size。回放随新数据等比缩放 → 占比恒定 ~1/6(16.7%)，不再因新轨迹
  (沙箱失败)缩水而漂移升高(治 §370 附近记的"replay 占比失控淹没新任务")。replay_batch_size 降级为上限兜底。
- replay_ratio<=0 退回固定行为(向后兼容)。ceil 用整数除 -(-a//b) 实现，未加 math import。

**改动 2：超长单条轨迹进 verl 前丢弃(dependencies/verl recipe_custom/agent/session_worker/worker.py)**
- 需求：单条超长在 verl 训练端 rearrange_micro_batches 撞 assert(max_token_len>=max_seq_len，单条切不开)会崩。
  【不改 verl 的 assert】，而在 rollout 侧写 TransferQueue【之前】按长度过滤。
- `_filter_trainable_trajectories` 加：len(prompt_ids)+len(response_ids) > max_trajectory_tokens 的单条丢弃
  (打 AGENT_DROP_OVERLONG)。拦截点在行615过滤 → 行664写tq之前，超长轨迹根本进不了 tq/verl。
- 阈值 `max_trajectory_tokens`：显式配 >0 用配置值；否则【自动推导】= actor.ppo_max_token_len_per_gpu ×
  ulysses_sp(4卡65536×2 / 16卡32768×4，均=131072，正好 assert 边界)。
- 丢弃单条后组内成功数减少 → 自动走 min_group_success_ratio(0.5)：组内丢太多则整组作废(占槽不进训练)。
- 生效时机：改的是 dependencies/verl，当前运行的训练不热加载，下次重启生效；当前实测长度未触 131072，assert 暂不会崩。

**待办**：两改动真机重启验证；replay_ratio/max_trajectory_tokens 是否显式写进 R 系列配置(现走默认/自动推导)未定。

---

## 2026-07-31 §32 reward 恒 0 定案(两根因)+ 修复 + API-as-actor 端到端冒烟验证通过

**现象**：qwen35_9b_b1_4gpu 训练 step1→52 reward 恒等于 0.0(metrics.jsonl 里 critic/rewards/critic/score 全 max=min=mean=0)，actor/pg_loss=0(GRPO 组内奖励全同→优势0→无梯度)。**结构性从头 0，不是逐渐耗额度。**

**排查(全离线实证，不动 verl 不上 GPU)**：
- reward 写回 `recipe_custom/agent/session_worker/worker.py:1107-1110`：`rm_scores=zeros`，仅 `trajectory.reward_score is not None` 才覆盖末位。
- 打分入口 worker.py:897 `if self.reward_loop_worker_handles and ...`；判 0 不 crash：`model_reward.py:350 except` 兜住→judge_error=1+score0。
- **排除的错误假设**：①"omni colocate 拿不到 reward_model→KeyError→0"错(enable=False 时 colocate 分支被跳过，走 agent-loop 内联打分)；②rollout dump(`_log_rollout_data` v1 版只写 uid，score 从 TQ rm_scores 取，不写 judge_error)对 H1/H2 无判别力。
- **离线复现两个独立真根因(都致 uniform 0)**：
  1. **judge key 没透传进 Ray worker**：SUFY_API_KEY 在 .env+load_training_env(set -a source)进 driver shell，但 `verl_runner.run_cl_ppo` 的 `_passthrough` 不含它(日志实证 `透传 env 到 Ray worker: ['PYTHONPATH']`)。Ray actor 不继承 driver shell env。缺 key 时 `agents/config._resolve_key` 静默返回 "sk-local"(不报错)→ judge 用假 key 打 sufy → 401 → except 兜 → judge_error=1 → reward 0。
  2. **thinking judge 被 max_tokens 截断**：judge=deepseek-v4-flash(thinking)，`model_reward.py:197 max_tokens=4096`。离线实测该模型对一条真实轨迹 **reasoning_tokens=3781**，4096 几乎不留 JSON 空间 → finish_reason=length → `_raise_if_truncated` 抛 TruncatedOutputError → except → judge_error=1。**提到 16384 后同一轨迹 finish_reason=stop，verdict 正常出**(`{"safety":1,"completion":0,"robustness":0}`)。

**修复(两处，均项目侧，不动 verl)**：
- `trainer/model_reward.py`：OpenAIJudgeClient.max_tokens 变可配字段，默认 16384(env REWARD_JUDGE_MAX_TOKENS 可调)，取代硬编码 4096。
- `trainer/verl_runner.py`：`_passthrough` 增 SUFY_API_KEY + REWARD_API_BASE/REWARD_MODEL/REWARD_API_KEY/REWARD_JUDGE_MAX_TOKENS，透传进所有 Ray actor(含 RewardLoopWorker)。

**验证**：
- 单测 `tests/test_model_reward.py` + `tests/test_judge_agreement.py` = 16 passed。
- **新增 `scripts/reward_smoke_e2e.py`**：API-as-actor(OpenAIChatClient 包 GenerateFn)→ SessionSandboxPool(backend=local, 4 slot)真 ReAct rollout(沙箱 run_code)→ observer diff → 真 reward judge → 打分。**端到端跑通**：
  - 每 slot reward = 0.1 / 0.2 / 0.1 / **1.0**，judge_error=0.0，gated=None(judge 真被调用且返回有效 verdict)。
  - slot3 完整完成任务(completion=1.0)，其余部分完成 → **reward 有区分度 = GRPO 需要的优势信号回来了**(修复前全 0 → 零优势 → 不学习)。
- 副发现：actor 默认模型 qwen3.7-max 当时 sufy 502 宕机 → 换 qwen3.6-plus 正常(能出 <toolcall> 格式)；本机 Python 用 miniconda3(训练用镜像内 /opt/conda，本机无)。

**待办**：两处修复需真机重启训练验证 reward 真正非 0(改的是当前运行不热加载的代码)；正式 27B/64GPU 的 verl_runner 同一份，透传自动生效。

---

## §35 reward 修复线上验证通过(4GPU b1, step1, 2026-07-31)

§34 两处修复(judge key 透传 + max_tokens 16384)**真机重启后线上验证通过**。`qwen35_9b_b1_4gpu` 带修复重跑,step 1 打分完成,`logs/metrics/qwen35_9b_b1_4gpu/metrics.jsonl`:

| 指标 | 修复前(上次 52-step run) | 修复后(本次 step1) |
|------|--------------------------|---------------------|
| `critic/rewards/mean` | 0.0(52 步全 0) | **0.7708** |
| `critic/rewards/max` | 0.0 | **1.0** |
| `critic/rewards/min` | 0.0 | 0.0 |
| `actor/pg_loss` | 0.0(无梯度) | **0.0129** |
| `critic/advantages/max\|min` | 0(全同) | **+1.58 / −2.26** |
| `actor/ppo_kl` / `pg_clipfrac` | — | −1.3e-4 / 0.0081 |

**reward 非 0 且有区分度 → GRPO 优势信号恢复(advantages 有正有负)→ pg_loss 非 0 → 模型真在学**。judge 真被调用、无 401/截断迹象。日志透传实证 `['PYTHONPATH','VERL_USE_EXTERNAL_MODULES','MALLOC_*','SUFY_API_KEY']` + 真 key(非 sk-local)。session finish: stop=46 / tool_calls=608(多轮 ReAct 正常),response_length mean=25310/max=61375。**两个根因修复线上层面证明有效。**

---

## §36 两个 16GPU run 失败(GPU 未释放),4GPU 正常(2026-08-01)

同时跑三实验:**4GPU b1 正常到 step 52**(reward 0.4~0.77 健康波动、pg_loss 有正负、entropy 稳),
两个 16GPU 都失败且 **GPU 未释放**(进程 hung 不退)。

### 4GPU(对照,正常)
`qwen35_9b_b1_4gpu`:step 52,metrics 52 行齐全,reward/mean 稳在 0.5~0.77,pg_loss 正负交替,
entropy 0.17~0.42 无坍缩。SP=2,DP=2,train_batch=4(×n8=32 轨迹/step),负载小。

### ① k1_16gpu — lightllm 推理侧 CUDA OOM → 进程 hung
- 现象:step1 rollout 中(未完成一步,metrics=0),`LightLLMHttpServer` 在 **prefill 阶段 CUDA OOM**
  (`GPU 1 total 79.32GiB`,08-01 03:19),之后 lightllm 陷入 `abort request wait release timeout` /
  `pause_generation abort_all still waiting` **死循环 ~5.5 小时**(03:19→08:53 日志仍在刷 abort),
  进程不退 → **GPU 一直被占,系统不回收**。
- 根因:16 卡 SP=4、gpu_memory_utilization=0.75,推理 KV 池 + 长序列 prefill(response 65536)峰值超显存;
  单请求 prefill OOM 后 lightllm 不能干净 abort/释放,卡在 release 等待。

### ② b1_16gpu — rollout 前 hung(过了 8<16 卡检查,但卡在 worker 初始化)
- 现象:比昨天那次(§35 提到的 8<16 卡)走得远——lightllm server 全部 `server start up ok`
  (02:36)、WorkerDict 权重加载 100%、FSDP 建好,**但沙箱请求=0、Training Progress 从未出现**,
  日志 **02:36 冻住**(driver 侧再无动作),GPU 占着不退。
- 关键线索:`[Gloo] Rank 0 is connected to 0 peer ranks. Expected ... 0`——各 rank gloo 组只连自己;
  疑似 rollout manager / agent_loop 初始化(GatewayActor↔AgentSessionWorker↔lightllm 建连)或
  跨节点 TransferQueue 握手卡住,一直等不到 → 无超时、无报错,纯 hang。

### 与 4GPU 对比:为什么 16 卡出事、4 卡不出
| 维度 | 4GPU(正常) | 16GPU(失败) |
|------|-----------|-------------|
| 节点 | 单节点 1×4 | **跨 2 节点 2×8** |
| SP | 2 | **4**(单卡序列切 4 段) |
| train_batch | 4(32 轨迹/step) | **32(256 轨迹/step)** — 采样/显存压力 ×8 |
| 失败点 | — | k1=推理 prefill OOM;b1=跨节点 rollout 建连 hung |
- 4 卡单机、负载小、无跨节点握手,故稳;16 卡跨节点 + 大 batch + 长序列同时放大**显存峰值**与
  **分布式建连复杂度**,两个各踩一个:一个显存爆(OOM),一个跨节点协同 hung。
- **共性教训**:两者失败后都 **hung 不退 → GPU 不释放**(k1 是 lightllm abort 死循环;b1 是 driver 死等),
  需手动 `ray stop` / kill 回收。

### 待办
- k1:降推理显存峰值(gpu_memory_utilization 0.75→↓ 或 SP↑ 或限 rollout 并发/max_num_batched_tokens),
  给 prefill 留余量;或对 OOM 请求走可恢复 abort。
- b1:定位 rollout 初始化 hung 点(GatewayActor/agent_loop 跨节点建连 or TransferQueue),加超时+自愈。
- 两者都需:失败自动退出 + `ray stop` 释放 GPU(避免占卡 hang 数小时)。

---

## §37 4GPU 跑到 step54 崩:agent_assets 非张量字段 batch 尺寸不一致(2026-08-01)

4GPU b1 健康跑完 **53 step**(reward 稳、pg_loss 正常),**step 54 崩**——**不是** OOM/超长,是数据 schema 问题:

```
AssertionError: Batch size of tensor agent_assets is not consistent with other tensors. Expected 4, got 2
  verl/utils/tensordict_utils.py:441 get_tensordict
  ← recipe_custom/trainer/v1/sync_trainer.py:536 _add_batch_to_generate → _next_train_batch → _fetch_one_gen_batch
```

**根因链**(已定位):
- `agent_assets` **不在数据集**(train.parquet 无此列、extra_info 也无,45242 行全 0)。它是 **rollout 时**由沙箱 runner 按 prompt 的 `sample_fields` **有条件产出**的 per-datum 字段(`worker.py:497`:`{key: sample_fields[key] for key in (...,"agent_assets",...) if key in sample_fields}` —— **`if key in` 条件写入**)。
- 于是一个 gen-batch(train_batch=4)里,**部分 prompt 带 agent_assets、部分不带** → 写进 TQ 的 field 有的有该 key 有的没有。
- verl `get_tensordict`(tensordict_utils.py:441)要求 **batch 内每个字段行数一致**;`agent_assets` 只有 2 行、其它张量 4 行 → 断言崩。
- 前 53 步没崩是**巧合**:那些 step 的 4 条恰好同质(要么都带要么都不带);step54 第一次撞上"4 条里 2 带 2 不带"的混合批 → 崩。**非确定性,resume 到 step54 未必同点复现,但迟早会撞。**

**与之前失败区分**:§36 k1=推理 prefill CUDA OOM;§36 b1_16gpu=跨节点 rollout hung;§37 4GPU=agent_assets 字段 batch 不一致。**三个不同根因**,勿混。

**修复方向(待定)**:
1. **补齐字段**:让 `agent_assets` 对 batch 内**每条都写**(缺失填空 `{}`/`[]`),保证 batch 尺寸一致——最稳,项目侧或 worker 侧统一。
2. 或 rollout 侧把 agent_assets 从 TQ tensordict 字段里**排除**(它不是训练用张量,只是沙箱 staging 元数据,不该进 train batch)。
3. 4GPU 已跑 53 step 验证了 reward/训练链路健康(本 bug 与 reward 无关),此 bug 是**数据字段一致性**,独立修。

---

## §38 16卡两节点内存差异归因 + b1_16gpu hang 真因=NCCL P2P(2026-08-01)

多 agent 只读排查(4 假设并行核查 verl 源码+日志,对抗综合)。核实结论(证据强度已标注)。

### 用户问题:两节点显存/内存差异大

**主因是 CPU host RAM(非显存),且 head 偏高是 verl 原生结构性正常,非 bug/非 placement 不均。**

- **CPU 内存 head 偏高 = head 独占 driver 进程组件**(日志实证):
  - driver `CLTaskRunnerV1`(pid=4122)只在 head(train.log:6 `Local node IP 10.120.4.114`+:4 `ray start --head`;pid=4122 全程无 `ip=` 标签,worker actor 均带 `ip=10.120.4.107`)。
  - **TransferQueue controller+storage 全绑 `localhost`(=head 进程内)**——`train.log:966-967` Mooncake `master_server_address/metadata_server: 'localhost:50124/50123'`,`:969` SimpleStorage 8 单元;`verl_runner.py:732 tq.init()` 在 driver 里。全 batch prompt/response/logprob tensor 压 head。**这是 head CPU 内存偏高最大来源**(~十 GB 级,每 step kv_clear 回收)。
  - b1 里 val_files alias 到 train → head 驻留**两份** 45242 行 tokenized dataset。
  - Ray GCS/dashboard/object-store owner head 独占。
  - buffer 非大头:`train.log:538 lambda_replay=0`+buffer.enabled=false → 项目 9 桶 buffer 不建。
- **GatewayActor 大内存(§33 单个~180GB glibc arena)两节点 round-robin 均分,不制造 head-vs-worker 不对称**:`agent_loop_manager.py:151-158`(gateway)/`414-436`(session worker)/`reward_loop.py:304-320`,均 `NodeAffinitySchedulingStrategy(node_ids[i%n], soft=True)`,注释明说防 PACK-to-driver。8→4/4。已透传 `MALLOC_ARENA_MAX=2`(train.log:53)缓解。
- **显存(VRAM)两节点对称,无 head 偏置**(日志实证,假设"卡分配倾斜"**推翻**):HYBRID colocate(非 40+24 分离,`lightllm_rollout.py:49-50`)+ TP=2 → 严格 8 replica/16卡=4/4(`replica.py:138-156`);**KV 池两节点实测同为 `3015483 @ mem_fraction 0.75`**(train.log:1256 head / :1281 worker,[repeated 8x]);FSDP param+optimizer 双 offload → VRAM 训练脚印小(`After FSDP 5.31/79.18`)。**若实测 head 显存真更高→超结构可解释,需分节点 nvidia-smi 坐实;强烈怀疑用户看到的差异其实是 CPU 内存被读成显存。**

### 附带重大发现:§36 b1_16gpu "hang" 真因 = NCCL P2P CUDA failure(不是单纯 hang)

`train.log:1378` `transport/p2p.cc:863 NCCL WARN Cuda failure 1 'invalid argument'`,02:36:18,**两节点都报**(master-0 ×4 / worker-0 ×3),发生在 lightllm 跨卡 NCCL 建链;之后日志 02:36:53 冻死(到 2038 行止)。**这是 P2P/CUDA 初始化失败致 rollout 引擎握手死等,不是内存/显存高低导致。** 与 head 内存偏高**无因果**。

### host RAM OOM 是独立第三类风险(需盯)
head 因多驻留(两份 dataset + TQ storage 全 batch tensor + GCS + 4 个 gateway arena)host RAM 系统性高于 worker;长跑逼近整机上限时 **head 先触 host OOM 风险 > worker**。与 §33(对称 GPU OOM)/§36(P2P hang)不同源。

### 修复方向(区分项目侧 vs 动 verl)
1. **[最高优先,先修 hang]** NCCL P2P `Cuda failure 1`:runtime_env/`_train_impl.sh` 试 `NCCL_P2P_DISABLE=1`(或 `NCCL_P2P_LEVEL=NVL`)绕过验证连通,再查 IB/NVLink 拓扑 or 容器 `--ipc`/`--shm`。**内存均衡在 hang 修好前无从谈起。**
2. [验证] run 起来后分节点 `nvidia-smi`+`free -g`+`ray status` 坐实"差异是 CPU 还是 VRAM"。
3. [CPU 内存缓解,项目侧] val_files 别 alias 到 train(省一份数据集);评估 TQ storage 是否可分布式(解 head 单点);保留 MALLOC_ARENA_MAX=2。
4. [慎改] driver 落 head 是 verl 原生(`main_ppo.py:82` 无节点约束),强制上 worker 需改 verl 调度,收益有限(GCS 仍在 head),**建议不改**。
5. gpu_memory_utilization 别为"均衡"调(两节点对称);要调只为 §33 对称 GPU OOM 让显存。

> 不确定性:run 在 init 期 hung,三大 worker 池未创建,全部运行态 RSS/VRAM 数字缺失;gateway 4/4 实际达成、head 绝对水位、"差异究竟 CPU 还是 VRAM"均需 run 真跑起来分节点实测。

---

## §39 超长轨迹根因 + 三实验失败归纳 + 并发/len 体检(2026-08-01,多agent源码实证)

多 agent 深挖 verl recipe_custom 源码,回答用户 5 问。核心均源码/日志实证。

### 三实验失败(三个不同根因,勿混)
| 实验 | 失败点 | 根因 |
|------|--------|------|
| k1_16gpu | lightllm prefill CUDA OOM(6次)→ 进程 hung | 并发×超长 撑爆 KV 池(见下 Q3) |
| b1_16gpu | NCCL P2P `Cuda failure 1 'invalid argument'`(7次,两节点)→ hung | GPU 已分配(KV池/cudagraph/FSDP权重就位)**但 0 step**;§38 |
| b1_4gpu | `AssertionError: agent_assets batch 4 vs 2` | agent_assets 非张量字段 batch 内不一致(§37) |

### Q2:b1_16gpu "数据进 GPU 但没运行" —— **确认**
日志实证:KV 池已 profile(`max_total_token_num`)、cudagraph captured、FSDP 权重 100% 加载(`device memory used 5.31/79.18`)→ **显存已分配**;但 `Training Progress=0`、沙箱请求=0、step=0 → **一步没跑**。NCCL P2P init 失败致 rollout 引擎握手死等,GPU 占着空转。

### Q4:推理侧超长根因 —— **65536 是死配置,轨迹级只按 max_model_len(131072) 截**
- 走 RemoteAgentLoopManager+hermes+gateway 路线(非 in-process ToolAgentLoop)。
- 单次生成 `max_tokens = rollout.response_length = 65536`(`gateway/request_utils.py:171`;`awm_sandbox_runner.py:212`),**只管单次 LLM 调用,不跨轮累积/递减**。
- 8 轮 ReAct 每轮 assistant(≤65536)+工具输出(≤16384)累加进同一 response。
- **整条轨迹唯一总闸门 = `gateway/trajectory_buffer.py:170` `response_capacity = max_model_len - len(prompt)`**,max_model_len 来自 `gateway.py:106` config 值=**131072**。
- 实测峰值 **119586 < 131072** → 实锤撞的是 131072 不是 65536;理论最大 ≈131072-prompt≈122876。
- 对照:同仓 `tool_agent_loop.py:296` 才按 max_response_length 跨轮递减,但**本配置不走它**——这正是 §33.4 删自写 max_trajectory_tokens 后失去的保护。

### Q5:能否阻止超长进训练 —— **verl 原生零长度闸门(源码实证)**
- `session_worker/worker.py:1034-1076` 把 response_ids **原样**写 TQ,无二次截。
- `worker.py:359-377 _filter_trainable_trajectories` **只按 trace_type 过滤,完全不看长度**。
- gateway 的 131072 截断 ≠ 入训丢弃(截到 131072 照样全量进 PPO)。§33.4 删掉的 max_trajectory_tokens 是唯一曾有的入训长度闸门。

### Q3:并发/len 配置 —— **有明确问题,16 卡必 OOM。两个独立开关必须一起改**
- **关键新发现**:gateway 用 config `max_model_len=131072`,但 **lightllm 引擎侧 `async_lightllm_server.py:132` 把自己的 max_model_len 覆盖成 HF `max_position_embeddings`=262144**(Qwen3.5-9B config.json 实证),`max_req_total_len=max_model_len`。两套值。
- KV 池 `max_total_token_num`≈3015483(RunLog:1429/1704 实测 @0.75);单条满长 119586 → **~25 条即吃满池**;配置却下 **running_max_req_size=256**(`engine_kwargs.lightllm` 后置覆盖真生效)→ **超订 ~10×** → prefill OOM(k1)。
- 4 卡 running_max_req_size=32(乘积 1/8)未撞满,扛过 53 步。

### 推荐修复
**方案甲(纯项目侧配置,零改 verl,首选,先上)**:改 `configs/run/b1_9b_16gpu.yaml`:
1. `max_model_len: 131072 → 73728`(=prompt 8196+response 65536)→ gateway response_capacity 压到 ~65k(Q4/Q5 收口)。
2. `running_max_req_size: 256 → 64` + `graph_max_batch_size: 256 → 64`→ 并发×长度不超 KV 池(Q3 OOM)。
   ⚠️ 缩 max_model_len 只压 gateway;引擎侧 :132 仍覆盖 262144,故**必须同时降 running_max_req_size**,两开关缺一不可。
**方案乙(彻底根治,动 verl 需 passthrough+理由)**:`trajectory_buffer.py:170` 加独立旋钮 `response_capacity=min(max_model_len-prompt, cfg_response_cap)`,response 上限与引擎上下文解耦(=把 §33.4 删的 max_trajectory_tokens 挪进官方 gateway)。仍须叠加方案甲的并发降配。

---

## §40 修复 agent_assets batch 不一致(§37 4gpu step54 崩)(2026-08-01)

**根因(项目侧,已定位到行)**:`trainer/cl_agent_dataset.py:84` `if assets:` —— 只有 taskspecs_w3/<rid>/files 目录存在的 record 才写 `agent_assets` 键(`build_agent_assets` 无文件返回 {})。一个 gen-batch(train_batch=4)混合"有文件"和"无文件"的 record → 有的行带 agent_assets 键、有的不带 → verl `get_tensordict`(tensordict_utils.py:441)对非张量字段要求 batch 内每行都在,断言 `Batch size of tensor agent_assets ... Expected 4, got 2` 崩。前 53 步没崩是巧合(那些 step 的 4 条恰好同质)。

**修复**:`__getitem__` **恒写该 key**——`row_dict["agent_assets"] = assets if assets else {}`。空 dict 下游容忍:`sandbox_setup.unique_asset_specs` `if not agent_assets: return []`、`e2b_agent_runners.py:204` `if agent_assets:` 跳过注入 → 语义无副作用,只是让 batch 内每行都有该字段、尺寸一致。本机 stub 验证通过(无文件 record 也输出 agent_assets={})。**纯项目侧,不动 verl。**

## §41 澄清两个概念问题(2026-08-01)

**Q:方案3(降并发防 OOM)和方案4(缩 max_model_len 防超长)矛盾吗?** —— **不矛盾,是同一因果链的两个不同环节,必须同时做。**
- 4(缩 max_model_len 73728):管**单条轨迹能多长**——gateway `response_capacity=max_model_len-prompt` 决定单条 response 上限。缩它 → 单条不再冲到 120k。
- 3(降 running_max_req_size 64):管**同时有多少条在推理**——KV 池被"并发条数 × 每条长度"吃。
- 显存需求 ≈ 并发 × 单条长度。4 压"单条长度"因子,3 压"并发条数"因子,**两个因子相乘决定峰值,压任一个都只解一半**。且有独立第三点:引擎侧 `async_lightllm_server.py:132` 把自己的 max_model_len 覆盖成 HF 262144,配置缩 max_model_len 只压 gateway 轨迹截断、压不住引擎 KV 规划 → 引擎侧只能靠降并发兜。故 **3+4 缺一不可,不矛盾。**

**Q:最长 122876 < 131072,为什么说 max_response_length(65536) 失效?** —— "失效"指的是 **65536 这个值**,不是 131072。
- 我们**配置意图**是 response ≤ **65536**(`max_response_length`/`response_length`)。
- 但这条路线(RemoteAgentLoopManager+gateway)里,65536 只作**单次生成**的 max_tokens;整条多轮轨迹的**唯一硬闸门是 131072(max_model_len)**。
- 实测峰值 119586:**远超我们想要的 65536(1.8×),但确实没超 131072**。
- 所以"失效"= **65536 这道我们以为存在的闸门根本没生效**(轨迹级从不按它截),轨迹一路涨到逼近 131072 才被挡。**不是"超过 131072",而是"越过了本该拦在 65536 的线"**。131072 是兜底、不是我们要的上限;122876 未超 131072 恰恰证明"拦它的是 131072 而非 65536"——即 65536 失效的**证据**,不是反证。

---

## §42 修正 k1 OOM 归因:是 KV 池饱和(并发×累积长度)而非单条 12 万轨迹(2026-08-01)

**自我纠错**:§39 把 k1 OOM 归到"单条 119586 超长轨迹撑爆",**归因过粗**。回原始 OOM 日志(k1 train.log:4991)细看,更准确的机制:

- OOM 原文:`Tried to allocate 24.00 MiB. GPU 1 total 79.32 GiB of which 24.75 MiB is free` —— **压垮的最后一根稻草只有 24MB,说明 GPU 早已 ~99.97% 满**。不是某一条巨型轨迹瞬间撑爆,是 **KV 池被持续占满后,一个微小分配触顶**。
- k1 实测 KV 池 `max_total_token_num=2914404`(非 §39 引的 3015483,那是别的 run)。
- **真正的内存吃主是"回填的 prompt",不是单次生成**:实测 `prompt_token_num` 最大 **122131**,而 `out_token_counter`(单次生成)最大仅 16589。多轮 ReAct 把**前几轮完整对话历史全部拼进下一轮 prompt** → prompt 越滚越大,单个请求 prompt 就占池 4.2%。
- OOM 前窗口内 **16 个 prompt>40k 的大请求并发**;16×~40k+ ≈ 已逼近甚至超过池容量 2914404。
- 这次 run 的 `response_length/max=98474`(不是 §39 说的 119586;119586 是 4gpu 的)。

**修正后的准确归因**:k1 OOM = **并发(256)× 每请求累积上下文(prompt 回填可到 12 万)乘积 ≫ KV 池 2914404**,是**池整体饱和**,不是单条超长轨迹。超长仍是因子之一(每条越长,池装得越少),但"并发太高"和"多轮 prompt 回填无限增长"是同等甚至更主的因子。

**对修复方案的影响**:§39 方案不变但优先级调整——
- **降并发(running_max_req_size 256→64)是第一位**:直接减少同时占池的请求数。
- 缩单条长度(max_model_len/response cap)是第二位:减小每条占用。
- 两者仍缺一不可(峰值=并发×单条长度),但**主因偏"并发过订"而非"单条过长"**。§39 表述"约 25 条满长吃满池、256 是 10× 超订"方向对,但精确说是"prompt 回填 + 256 并发"共同饱和。

**诚实标注**:"16 个 prompt>40k 并发"是 OOM 前 ~190 行日志窗口的粗数,非精确同刻并发;但足以证明是多请求累积饱和,非单条。

---

## 2026-08-01 §32 16卡 k1 lightllm prefill OOM 归因（推翻"并发过订"与"偶发峰值 D"）

- 现象：16卡 k1 GDN prefill CUDA OOM（`chunk_gated_delta_rule_fwd_h` alloc 24MB 失败），4卡 b1 稳跑到 step 54 无推理 OOM。
- 归因结论：**确定性单卡显存不足**，非偶发 prefill 峰值。
- 硬证据（train.log）：
  - 16卡 k1 log:4991 OOM：`GPU 1 total 79.32GiB, 24.75MiB free; Process A 11.04GiB + Process B 68.20GiB`。缺口仅 ~24MB。
  - 权重加载后剩余显存:16卡=44.36GB(log:1319) vs 4卡=50.37GB(b1 log:1173) → 差 ~6GB。
  - KV 池 max_total_token_num:16卡=2914404(log:1288) vs 4卡=3301275(b1 log:1175)。mem_fraction 都=0.75 → 池大小 = 0.75×profile 时剩余显存，6GB 差异全部来自 profile 时剩余显存差。
  - cudagraph:16卡 capture batch<=256(log:2082,681MB private pool)vs 4卡 batch<=32(b1 log:1383)。
  - colocate HYBRID:两边都 param_offload+resume_memory_occupation；11GB 那个进程=同卡 FSDP 训练 worker 残留（offload 非零 + NCCL/allgather buffer，world 越大越多）。
  - GDN prefill h = k.new_empty(B,NT,H,K,V)(chunk_delta_h.py:287)，∝ 同时 prefill 的总 token(ΣT/64)，不是 per-req；24MB 只是压垮最后一根稻草。
- 真旋钮:profile 时剩余显存(16卡少 6GB)→ KV 池 + cudagraph(256)吃掉余量 → prefill 临时 24MB 无处可放。4卡 response 更长(119586)但有 ~6GB headroom 故不崩。
- fix 优先级:(1) 降 lightllm mem_fraction 0.75→0.6~0.65(直接留 headroom);(2) 16卡 graph_max_batch_size 256→32/64(省 cudagraph 池);(3) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True(治 188MB reserved-unallocated 碎片)。三者任一即可,(1)最稳。

---

## §43 16卡推理 OOM 修复:gpu_memory_utilization 0.75→0.65(2026-08-01)

**背景**:k1_16gpu 推理侧 prefill CUDA OOM(调用栈 100% `LightLLMHttpServer` → chunked_prefill/impl.py → GDN prefill kernel,无训练侧痕迹)。真因(多agent+源码坐实):16卡单卡权重后 free 比 4卡少 ~6GB(44.47 vs 50.37GB)→ KV 池被压小(291万 vs 330万 token)+ 运行时同卡 colocate 训练进程 11GB → prefill 差 24MB 触顶。util/chunked_prefill 两边完全一样(都 0.75、chunked 都开 size3072),不是它们的问题——差异是 16卡单卡被 colocate 训练进程+跨节点 NCCL 多占 6GB。

**修复**:`gpu_memory_utilization` 0.75→**0.65**,主动砍 KV 池 ~6GB,把余量还给 prefill 工作区。改 19 文件(18 份 16卡 run + cluster.yaml);**4卡 b1_9b_4gpu 保持 0.75 不动(对照,单机余量足、53步未 OOM)**。load_config 18份全过、util 均=0.65。

**排除的错误修复**:
- ❌ `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`:与 `enable_torch_memory_saver=true`(colocate 分时复用显存必需)**互斥**,§22 实测当场崩(`TorchMemorySaver is disabled ... expandable_segments not supported` → lightllm 起不来 rc=1)。**禁用此手段**。
- ❌ 降 running_max_req_size 256→32:GDN 走共享分页(qwen3next_mem_manager.py:48 `cdiv(size,big_page)` 仅3页,MB级),不随 running_max_req_size 缩放;它只影响 req_to_token_indexs 0.22GB,解决不了 6GB 结构差。
- NCCL P2P `Cuda failure 1` failure:4卡(5次)16卡(6次)都有,4卡带着它跑了53步 → **不致命**(lightllm 内部降级),非任何实验死因。

**待验**:真机重启 16卡看推理 OOM 是否消除;b1_16gpu 的"driver↔lightllm 启动握手 hang"(rollout 从没开始)是独立问题,未解。

---

## §46 落地 NCCL_CUMEM_ENABLE=0 修 16卡起服 hang(2026-08-01)

**背景**:b1_16gpu **6 次复发**同一 hang(换节点仍撞):3/8 lightllm 副本卡在 `server start up`(590)到不了 `server start up ok`(594)→ verl `llm_server.py:521` 无超时 `asyncio.gather` 永久阻塞 → driver 从没到 rollout(Progress=0/沙箱=0/update_weights_200=0)。P2P `Cuda failure 1` 非致命(两边都有可降级)。诱因=`enable_torch_memory_saver`(cuMem VMM 劫持 cudaMalloc)× NCCL 默认 `NCCL_CUMEM_ENABLE=1`(NCCL 也用 cuMem 给 P2P buffer)冲突。定案见 debug §45。

**修复(项目侧,不改 verl 逻辑;对齐 verl 给 vllm/sglang 的既有处置,lightllm 漏了)**:
- `trainer/verl_runner.py`:`_passthrough.setdefault("NCCL_CUMEM_ENABLE", os.environ.get("CL_NCCL_CUMEM","0"))` → 透传进所有 Ray worker(含 lightllm 副本)。
- `scripts/_train_impl.sh`:`export NCCL_CUMEM_ENABLE="${CL_NCCL_CUMEM:-0}"`(双保险 + 单机路径)。
- 可 `CL_NCCL_CUMEM=1` 覆盖。**关 CUMEM 不关 P2P**,TP 内带宽保留、保持 2 机训练(非 NCCL_P2P_DISABLE 绕过)。
- 校验:bash -n OK;verl_runner ast.parse OK;模拟 passthrough 确认 NCCL_CUMEM_ENABLE=0 进列表。

**验证信号(真机重启后看)**:①8 个 `server start up ok`(594)全到;②8 条 `POST /update_weights_from_ipc 200`;③出现 `Training Progress` + 沙箱请求。若仍缺副本→叠 §45 方案2(gather 加 wait_for 超时 fail-fast 定位)+ 查容器 `--ipc`/ACS。建议同时 `NCCL_DEBUG=INFO` 拿 P2P 选路证据。

**注意**:此修复解的是"16卡起服 hang",与 §43 的"推理 prefill OOM(util 0.75→0.65)"是**两个不同问题两道坎**——起服 hang 在前(rollout 没开始),OOM 在后(rollout 中)。hang 修好、16卡跑到推理阶段后,才能连带验证 §43 的 OOM 修复。

---

## §47 16卡起服 hang 第二层根因:AFS tokenizer 并发加载 + 无超时 gather(2026-08-01)

§46 加 NCCL_CUMEM_ENABLE=0 后 P2P failure 归 0(cuMem 冲突已解),但 b1_16gpu **仍 hang**——卡点前移,浮出第二层根因。多 agent(3 假设并行 + 源码日志)定位:

**卡点**:lightllm `startup_event`(`api_http.py:592`)的 `set_args()` 在 async uvloop 里做**同步重 I/O**——每副本从 AFS 网盘(`/mnt/afs_toolcall`,quarkfs fuse)**加载两次** tokenizer(`api_http.py:113 init_tokenizer` + `manager.py:117/124 HttpServerManager.__init__`),且因 config.json 有 vision_config,`get_tokenizer`(`tokenizer.py:116`)**无条件走 qwen3_5 多模态分支跑 AutoProcessor.from_pretrained**(`disable_vision`/`enable_multimodal=false` 管不到它)。8 副本并发砸同一 fuse 挂载 → I/O 序列化拖到分钟级 → 冻结 event loop → 副本到不了 594。

**为何 16卡崩 4卡不崩**:8 副本 vs 2 副本,网盘争抢重 8 倍;4卡扛得住。**为何 1/8 到 594**:非 barrier,是抢网盘 I/O 的时序竞速(谁先拿到时隙谁先过,日志 14:10:17→19→20 逐秒错开后冻死)。**为何变永久 hang**:verl `llm_server.py:521` + recipe `async_lightllm_server.py:421` 的 `asyncio.gather` **无超时**,要等全部 8 副本 ready,慢副本永远等 → §45 结构缺陷。

**修复(两个,均项目侧/自有 recipe,不动 verl 本体)**:
1. **模型预热到 node-local**(`scripts/_train_impl.sh` 新增 `_prewarm_model`):启动前(ray start 前,每节点各一次)把模型目录 `cp -a` 到本地盘(默认 `/dev/shm/cl_models`,不足退 `/tmp`),`actor_rollout_ref.model.path` override 指过去。消除 8 副本并发命中 AFS 的争抢。`CL_PREWARM_MODEL=0` 可关,`CL_MODEL_LOCAL_ROOT` 覆盖本地根,命中缓存(大小一致)免重拷。AFS 原件保留(cp 非 move)。
2. **起服 gather 加超时 fail-fast**(`recipe_custom/rollout/lightllm/async_lightllm_server.py:421`,项目自有 recipe):`asyncio.wait_for(gather, timeout=CL_LAUNCH_TIMEOUT默认1200s)`,超时用 `ray.wait` 查未就绪副本、打印 replica_rank+idx 再抛。把静默 hang 变可定位报错(治标安全网,§45)。踩坑:`_tasks` 是 Ray ObjectRef 无 `.done()`,改用 `ray.wait(..., timeout=0)` 判 pending。

**校验**:bash -n OK;recipe ast.parse OK。**待真机重启验证**:预热后 8 副本应都快速到 594 + `update_weights_from_ipc 200` + Training Progress。

**层次关系**:§46(NCCL_CUMEM=0)解第一层 P2P 冲突;§47(预热+超时)解第二层 AFS 加载慢。两层叠加才能让 16卡起服跑通,之后才轮到验证 §43 的推理 OOM(util 0.65)。

---

## §48 §47 tokenizer 判断被真机推翻:预热成功仍卡 590→594(2026-08-01)

§47 加了 NCCL_CUMEM_ENABLE=0(P2P failure 归0)+ 模型预热到 /dev/shm(tmpfs,42s 完成,model.path 已用
node-local)。真机重启后:**修复都生效**(预热日志/NCCL_CUMEM=0 透传/P2P=0 均实证),**但仍 hang 在 590→594**
(server ok=1、update_weights=0、Progress=0、沙箱=0,同旧模式)。

**结论:§47 归因的"AFS tokenizer/AutoProcessor 并发加载慢"不是真根因**——模型已在内存盘 tmpfs,tokenizer 加载
不再慢,却照样卡。真阻塞点在 590→594 之间的【别的同步调用】,强嫌疑=`HttpServerManager.__init__` 里的同步
`rpyc.connect`:
- `manager.py:97 self.cache_client = rpyc.connect("localhost", args.cache_port)`(embed cache,同步阻塞)
- `manager.py:125 MetricClient(metric_port)` → `metrics/manager.py:80 rpyc.connect`;且 `start_metric_manager`
  `send("init ok")`(:152)早于 `t.start()`(:153)→ 竞态,connect 可能连到未 accept 的 server。
- 卡 8 分钟>rpyc 30s timeout,说明要么连上了但握手卡、要么 timeout 后重试循环——待 workflow 判定。

**已开 debug workflow(wte3teswe)** 3 路查:cache_port rpyc(A,最强)/ metric 竞态(B)/ lifespan+编排(C)。

**同时暴露 §47 修复2 的问题**:起服 gather 超时(CL_LAUNCH_TIMEOUT=1200s)未触发——要么没到 1200s,要么卡在
它之前的 driver 层 `llm_server.py:521`(那层没加超时,§47 只加了 recipe `async_lightllm_server.py:421`)。
若是后者,超时加错了层,需补 driver 层(走 recipe override 不动 verl 本体)。

**有效的**:NCCL_CUMEM=0(P2P 归0)、预热(42s,AFS 争抢消除)——两者仍保留,是对的,只是没解 590→594 这层。

---

## §49 CL_DIAG debug 定案:16卡 hang 真因=update_weights 的 NCCL Broadcast 集合挂死(2026-08-01)

开 CL_DIAG=1(RAY_DEDUP_LOGS=0 + NCCL_DEBUG=INFO + lightllm debug,均透传进 Ray worker)重跑 b1_16gpu,**拿到 NCCL 层实证,推翻此前所有推测**(tokenizer/rpyc/cache/590-594 全是 dedup 假象)。

**时间线(train.log 实证)**:
- 15:42:20-24 **8 个 lightllm 副本全部 `Capture cudagraph success` + `server start up ok`(594)**——起服**成功**。此前"1/8 到 594、卡 590→594"是 Ray 日志去重(`[repeated Nx]`)假象,关去重后真相是 8/8 全起来。
- 15:42:24 后进入**权重同步**:NCCL 疯刷 `Broadcast: opCount 0 ... count 1769472 datatype 9(bf16 3MB 权重分片) op 0 root 0`,**opCount 永远=0**(单个 comm 刷 16782 次),所有 rank 自旋。30s 涨 2.3 万行日志,总 79M。

**根因**:不是起服、不是显存、不是 tokenizer/rpyc。是 **update_weights 阶段(rank0 训练侧 Broadcast 权重给所有推理副本)的 NCCL 集合挂死**——通信组已 `Init COMPLETE`(nranks2 nNodes1 TP组),但 Broadcast 集合在 rank 间对不齐/某 rank 没发出,集合永不完成,NCCL 无超时→无限自旋刷 INFO。与 §45 workflow 提的 "update_weights_from_ipc hang" 呼应,现有 NCCL 实证。

**关键澄清**:此前 §47/§48/§45 的 "590→594 hang / tokenizer 慢 / rpyc" 均为**误判**,根源是 dedup 日志误读。CL_DIAG(关去重+NCCL INFO)是这次能定案的关键手段——**排 hang 必先关 RAY_DEDUP_LOGS**。

**已排除但保留的修复(仍有效,别回退)**:NCCL_CUMEM_ENABLE=0(P2P failure 归0)、模型预热 node-local(42s)、util 0.65、running_max_req_size 64、agent_assets 恒写、起服 gather 超时——都对,只是没解到 update_weights 这层。

**下一步(待定)**:
1. update_weights 的 Broadcast 挂死——查是 verl 训练侧 broadcast 权重给 lightllm 的实现(rank 拓扑/参与集合的 rank 集不一致),还是 NCCL 跨"训练 rank0 + 推理副本"混合通信组的问题。
2. 试 NCCL 集合超时暴露(TORCH_NCCL_ASYNC_ERROR_HANDLING=1 + watchdog timeout)让它 fail-fast 报出缺席 rank。
3. debug 日志 79M 且在涨,此 run 确定跑不起来(NCCL 无超时永久自旋),应停掉。

---

## §50 16卡 hang 真根因定案:NCCL_CUMEM_ENABLE=0 × SymmMem cuMem 握手冲突(2026-08-01)

**用户关键反问"同样16卡 k1 为何能训"直接指向真因**。对比铁证:
- k1(02:31 老 run,跑到 rollout/step2,死于 CUDA OOM 非 hang):**无 NCCL_CUMEM_ENABLE=0**、`Imported shareable buffer`(cuMem IPC 握手)**0 次**。
- b1(现 run,卡 count=1 broadcast):**有 NCCL_CUMEM_ENABLE=0**、`Imported shareable buffer` **3360 次**。
- 两者 lightllm 并发/overlap/TP/副本配置完全一致。

**根因(多agent+源码坐实)**:lightllm 推理副本 TP=2 的 all_reduce 走 **torch SymmMem(对称内存)**,其 `rendezvous`(进程间句柄交换,= 日志里 count=1 datatype2 root0 的 broadcast)依赖 **cuMem/VMM 对称堆 + 导入 peer cuMem IPC 句柄**。而 §46 为修 P2P failure 加的 **`NCCL_CUMEM_ENABLE=0` 关掉了 NCCL 的 cuMem 导入路径** → 两 rank 对 cuMem 句柄映射不一致 → SymmMem 握手 broadcast 永远完不成 → opCount 恒0 无限自旋(GPU util0,进程不退)。**§46 的修复自己制造了这个 hang**——k1 没加它反而穿过起服,是最硬对照。

代码链:`communication_op.py:117 not disable_symm_mem→:118 init_symm_mem_reduce→symm_mem_all_reduce.py:59 torch_symm_mem.empty(cuMem)+:60 rendezvous`。

**修复**:`disable_symm_mem_allreduce: true` 加进 18 份 16卡 config 的 `engine_kwargs.lightllm`。关 SymmMem 后 all_reduce **fallback 到标准 `dist.all_reduce`(NCCL)**(`communication_op.py:92-100` dispatch chain: FlashInfer→SymmMem→NCCL;FlashInfer 已被启动探测 auto-disable,故直接走 NCCL)。标准 NCCL all_reduce 不用 cuMem 对称堆,与 NCCL_CUMEM_ENABLE=0 不冲突。**既保留 §46 修 P2P,又绕开握手冲突,两全。** load_config 18 份全过。

**之前误判纠正**:§45(rpyc/tokenizer)、§47(AFS加载)、§48、以及本轮 workflow A(双 infer_loop 线程竞态)——都是被 Ray dedup 日志误导 + 没抓到 CUMEM×SymmMem 这层。真因是 SymmMem×cuMem,由 k1/b1 的 CUMEM 差异对照坐实。

**待验**:真机重启 16卡(带 disable_symm_mem_allreduce + NCCL_CUMEM=0 + 预热 + util0.65),看是否穿过起服→update_weights→Training Progress。

---

## §51 澄清:k1 OOM=GPU显存(非CPU内存),别与 §33 的 host 内存 OOM 混(2026-08-01)

用户疑"k1 是不是 CPU 内存 OOM"。核实日志(k1 train.log:4991/5058,6 次)原文
`torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 24.00 MiB. GPU 1 ... 79.32 GiB`——
**是 GPU 显存(CUDA/VRAM)OOM,报在 LightLLMHttpServer prefill 阶段**;无任何 host OOM 关键字
(Cannot allocate memory / std::bad_alloc / oom-kill / Killed process)。与 §36/§42 定案一致(KV 池饱和)。

**易混点区分**(两个不同 run、不同 OOM 类型):
| run | OOM 类型 | 根因 | § |
|-----|---------|------|---|
| **k1_16gpu** | **GPU 显存**(CUDA prefill) | KV 池饱和(并发×累积长度) | §36/§42 |
| b1_4gpu(step53) | **CPU host 内存**(节点 512GB 打满) | GatewayActor glibc arena 碎片(非泄漏),jemalloc 修 | §33 |

§38 另提过"head 节点 host RAM 系统性偏高、长跑有 host OOM 风险"——是**预警非实发**,k1 实际死于 GPU 显存。
注:k1 这些 OOM/hang 均在 §50(disable_symm_mem_allreduce)修复之前,待新配置重启后重测。

---

## §52 纠正 §50:真因是 LightLLM 双 infer_loop 线程并发 broadcast 竞态,非 SymmMem(2026-08-01)

**§50 判断有误**。加了 `disable_symm_mem_allreduce=True`(StartArgs 14 次确认生效、`SymmMemAllreduce enabled=0`)后带 CL_DIAG 重启,**仍卡死同一个 broadcast**:
- `Broadcast: opCount 0 count 1 datatype 2(int32) op 0 root 0` 刷 27916 次;`Imported shareable buffer` 2720 次。
- 关键否证:§50 上一个 run(16:33)`Broadcast_op0=0` 让我以为修好了,但那 run **没开 CL_DIAG**(NCCL INFO=0),0 是**没打日志的假象**,不是真没发生。CL_DIAG 开了才看到真相:**SymmMem 关了,这个 count=1 broadcast 照卡**——所以它**不是 SymmMem rendezvous**。

**真根因(workflow A,现坐实)**:那个 count=1 int32 broadcast = `base_backend.py:247 node_broadcast_tensor` + `:633 broadcast(..., group=node_nccl_group)`,是 serve loop 每 tick 的"shm 有无新请求"控制标志同步。`base_backend.py:284-287` **无条件起 2 个 infer_loop 线程**(double-batch overlap),两线程在同一 `node_nccl_group` 上**并发提交**这个 broadcast → 跨 rank 提交序不确定 → NCCL 集合配不成对死锁。铁证:同一卡死 comm `0x7fd780846010` 同一 rank[1] 被**两个线程 14528/14529 各刷 1286 次**。

**为什么 k1 能跑**:竞态——k1 那次两线程碰巧同序穿过;b1 输了卡死。NCCL_DEBUG=INFO 的日志开销可能加剧竞态(开 CL_DIAG 后每次必卡)。

**修复方向(改 LightLLM base_backend.py,走 patch;按代价)**:
1. 给 `_try_read_new_reqs` 的 broadcast 加进程级 `threading.Lock`(`:247` 建锁,`:625-634` body 包 `with lock:`)——两线程不并发发该集合,恢复确定序。**最小正确。**
2. 循环启动前(`:284` 前)单线程预热 `node_nccl_group`(去掉 `:267` barrier 的 run_mode gate,或发一次 dummy broadcast)。
3. 规避:让 serve loop 单 infer_loop 线程(`support_overlap`/`:284-287`)——但 :284-287 是无条件起两个,需确认有无关闭开关。

**已生效但非本因的修复(保留)**:NCCL_CUMEM=0(P2P)、disable_symm_mem(虽非本 hang 根因,但 SymmMem×cuMem 冲突理论上仍在,留着无害)、预热、util0.65、并发64。

**待办**:改 LightLLM 加 broadcast 锁(方案1)走 patch;或先试单线程规避确认。改前需确认 patch 机制(LightLLM 是 workspace 独立仓,非 verl passthrough)。

---

## §53 — 4卡 OOM 双侧定位 + save/resume/metrics + R 系列架构澄清(删按桶顺序训练误设计)  2026-08-02

> 4 卡 debug(b1/k1/k2/k3_9b_4gpu)的一轮修复。核心结论:①4 卡 OOM 分推理侧/训练侧两类,修法不同;②colocate 下"参数保守仍 OOM"的机制;③R 系列训练流程 = baseline/K,只多离线 replay,之前的"按桶顺序训练"是错误设计,已删。

### 一、4卡 OOM 分两类(同为 OOM,爆点与修法不同)

按报错进程区分,不能只看"util=0"判死活:

| 实验 | 报错进程 | 分配额 | 爆点 | 类别 |
|------|---------|--------|------|------|
| k1/k2 | `LightLLMHttpServer`(impl.py:100) | 24 MiB | rollout prefill,KV 池榨干 | **推理侧** |
| k3/b1 | `WorkerDict.update_actor`(engine_workers.py:658) | 1.63/2.10 GiB | update_actor **backward** | **训练侧** |

- **同是 K 系列,k1/k2 推理侧爆、k3 训练侧爆** = rollout 随机性:先撞上哪个爆点看运气,本质两侧显存都偏紧。
- **推理侧 OOM → abort 死等链条**(实测):OOM(先) → 在途请求 refcount 降不下去(`can release False refcount 5`,httpserver/manager.py:918) → `pause_generation` 的 `_wait_for_abort_released` 60s 等不到 `req_id_to_out_inf` 清空(:830) → `abort request wait release timeout`(:853)反复刷 → GPU 交接死锁 → **util=0**。**报错(timeout)≠ 根因(上游 OOM);看到 abort/pause timeout 要往上游找 OOM。**

### 二、修法(两侧双管,不限最长长度)

四份 4gpu(b1/k1/k2/k3)统一改:
- **推理侧**:`gpu_memory_utilization 0.75→0.65`。砍 lightllm KV 池(`profile_max_tokens.py:119 max_total_token_num=(gpu_total*mem_fraction-model)/kv_size`),lightllm 官方 OOM 提示第1条就是"调小 mem_fraction"。
- **训练侧**:`ulysses_sequence_parallel_size 2→4`(4卡全做 SP 切一条→单卡激活÷2)+ **连带** `ppo_max_token_len_per_gpu`/`log_prob_max_token_len_per_gpu` `65536→32768`(=131072/SP4)。⚠️**关键**:SP 升 4 但 token_len 不同步减半 = 每卡预算不变 = SP 白改;减半后 `32768×SP4=131072` 仍覆盖最长序列,**不限长**。DP 2→1(吞吐降,debug 可接受)。整除校验全过(tb4%DP1、mini4%DP1、16头%SP4)。

### 三、colocate 是"保守参数仍 OOM"的总纲(源码确认)

4 卡 `hybrid_engine:True` colocate,推理训练挤同 4 卡,靠 `enable_torch_memory_saver` 分时切换。三重压力:
1. 单卡要容 max(推理[权重+KV], 训练[actor+ref+优化器+梯度+激活]);交接不干净(k3 backward OOM 时 lightllm 仍残 3.68GB)。
2. **lightllm `mem_fraction` 基于"启动时空卡"profile**(`get_available_gpu_memory()*mem_fraction`,mem_manager.py:71)——启动时训练侧还没占,lightllm 以为可用显存都是它的,KV 池开过大,等训练侧要用时叠加爆。**这就是"参数保守仍 OOM"的根因。**
3. ref 模型(K 系列)训练侧 FSDP 常驻 7.98GB(b1)→17.26GB(K),多占 ~9.3GB。
→ 正式方案是 Fully Async 分离 40+24(推理训练卡物理隔离,各 profile 各卡),colocate 仅 debug 后备。

### 四、save_freq + 只留最近 2 个 ckpt(22 份 config,不改 verl)

- `save_freq 100→25` + 新增 `max_actor_ckpt_to_keep:2`/`max_critic_ckpt_to_keep:2`,全部 22 份(`configs/run/*_9b_{4,16}gpu.yaml`)。
- verl 原生支持:`PPOTrainer._save_checkpoint`(trainer_base.py:835 读参数)→ `fsdp_checkpoint_manager.save_checkpoint(max_ckpt_to_keep)` → `checkpoint_manager.ensure_checkpoint_capacity`(:176-178,超出 `shutil.rmtree` 最旧)。**不改 verl 源码**(铁律),纯配置。load_config 22/22 校验通过。

### 五、resume/metrics 语义(改 _train_impl.sh::_run_single)

按需求:默认 resume、不重复用数据、metrics 不覆盖、日志覆盖+archive(日志 archive 早已实现于 :256-266)。

- **有 ckpt**:自动 `--resume-from` 最新 ckpt(verl `dataloader.load_state_dict(data.pt)` 恢复数据游标,trainer_base.py:769→不重复用已训数据);旧 `metrics.jsonl` 折叠进永久累积 `metrics.all.jsonl`(`_fold_metrics`,按 step 去重,续训 step 不与旧重叠→无损)。
- **无 ckpt**:不 resume、全新训练;`rm` 掉 `metrics.jsonl`+`metrics.all.jsonl`,metrics 从头重开(不接旧)。
- verl FileLogger 硬编码 `open(path,"wb")`(tracking.py:420)每次覆盖 → 绕开:脚本层 fold,`metrics.all.jsonl` 永不覆盖=完整历史单一来源(分析看 .all);`metrics.jsonl`=当前 run 实时(会被 verl 覆盖)。fold 用显式 PY(铁律),纯标准库。

### 六、R 系列架构澄清 + 删除"按桶顺序训练"误设计(重要)

**澄清(用户,配置坐实)**:R 系列训练流程 = baseline/K **完全一样**(`_run_single`,全量 `datasets/train.parquet`,单进程 GRPO)。R 唯一区别 = 开 replay buffer(`cl.lambda_replay>0`+`buffer.enabled:true`+`num_buckets`)。**replay 是训练循环内的离线采样增强**(buffer hook 从已训桶采回放行拼进 batch,`response_mask=0` 不参与 PPO,只走独立 L_replay),**不改主流程、不改数据顺序、不分桶顺序训练**。桶 = buffer 内部采样组织单元,非训练外层循环。

**删除**:`_train_impl.sh` 的 `_run_buckets()`(58行)+ `--buckets`/`BUCKETS` 变量+arg解析+两处 dispatch 分支。理由:
- 它把"buffer 的 9 个内部桶"误当"分 9 次按桶顺序训练(桶间权重接力)",是错误理解的死代码。
- 全仓 grep 无任何脚本调用 `--buckets`(从未真正使用)。
- 上一轮我给它修的"R5 缺陷(共享 ckpt+auto 误续)"是给不该存在的设计打补丁,一并删除。
- 删后三系列(baseline/K/R)统一走 `_run_single`,R 自动继承 §五 的 resume/metrics 改进。

验证:bash 语法 OK、无 buckets 残留、无上层引用、R 与 baseline 用同一全量 parquet。

### 七、未解(不属本轮)

16 卡 baseline 起服 hang 仍在(§52 lightllm 双 infer_loop broadcast 竞态,py-spy 需在真实训练节点做,本机无 lightllm 进程)。本轮不涉及。

---

## §55 — 16卡 hang 真根因定案(推翻 §49/§52):8 副本只起 7,第 8 个 lightllm server actor 从未调度  2026-08-02

> 用最新一次 run(08-02 04:34 启动,CL_DIAG=1 全诊断,172MB train.log)重查,**逐条日志坐实**，推翻此前 §45–§52 的全部归因（tokenizer / rpyc / SymmMem / 双 infer_loop 竞态）。之前每章都是"抓一个信号就下结论"，这次把整条链路钉死。

### 一、决定性事实(全部 grep 坐实,非推测)

| 事实 | 证据 |
|------|------|
| **8 个 lightllm 副本只起了 7 个** | `replica_rank` 全集 = {0,1,2,3,4,5,6}，**replica_rank=7 零条日志**（无 rollout_mode 行、无 StartArgs、无 594）。`get_master_address` 成功 7 次（rank0–6），rank7 无。 |
| **训练侧 16 worker 全健康** | WorkerDict pid：master(125) 8 个 + driver(165) 8 个 = 16；RANK14/15（driver GPU6,7）NCCL `Init START`、Gloo "connected to 15 peer ranks"。**训练 NCCL 16 路完整成组**。 |
| **`LLMServerManager:` 从未打印** | `_initialize_llm_servers` 的 `asyncio.gather(init_hybrid×8)`（llm_server.py:521）**永不返回**——等第 8 个副本 ready 等不到 → driver 卡死在 replica init，**从没到 update_weights**（`update_weights_from_ipc`=0、`Training Progress`=0）。 |
| **那个刷 57 万次的 count=1 int32 broadcast 是"存活副本空转",非死锁** | opCount 恒 0 = 7 个存活副本的 serve loop 每 tick 在**各自副本内 TP2 组**（`create_new_group_for_current_node`：nnodes=1→`node_world_size=tp//nnodes=2`，即副本内 2 rank）刷"有无新请求"控制标志。因 driver 从没派活（卡在 gather），它们空转等标志位→opCount 不推进。**§49/§52 把空转误读成死锁。** |
| **无任何硬崩溃** | 全日志 Traceback=0、CUDA OOM=0、ActorDied=0、RayActorError=0、raylet 资源不足 warning=0。**不是崩溃退出，是根本没起。** |

### 二、拓扑真相(此前一直搞错)

每副本 `node_rank=0, nnodes=1, tp=2`——**8 个副本每个都是独立单节点 TP2 实例**，不是"2 节点 ×4 副本的跨机 TP"。两台机各跑 4 副本：
- **master(125)**：replica_rank 0/1/2/3，cuda `0,1`/`2,3`/`4,5`/`6,7` → **4/4 全起来**。
- **driver(165)**：replica_rank 4/5/6 起来（cuda `0,1`/`2,3`/`4,5`），**replica_rank=7（本该 cuda `6,7`）没起**。

`init_hybrid`（replica.py:139）：`workers[world_size*rank : world_size*(rank+1)]`，replica7 = `workers[14:16]` = 训练 RANK14,15（driver GPU6,7，已确认活着）。**fused worker 在，但 server actor 没起。**

### 三、根因(源码链坐实)

`LightLLMHttpServer` = `@ray.remote(num_cpus=1)`（async_lightllm_server.py:40），GPU 靠 fused worker 经 `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES` 复用、自己不占 GPU；用 `NodeAffinitySchedulingStrategy(node_id=..., soft=False)` **硬钉**到 driver 节点（:401-404），且**不进 placement group**。

- 训练 PG（base.py:146）每节点 STRICT_PACK 预留 `{"CPU": max_colocate_count, "GPU":1} × 8 worker`。
- server actor 要在 PG 之外找 1 个 free CPU。容器 **cgroup CPU 配额实测 = 16**（`cpu.max` = 1600000/100000，本机代理值；nproc 虚报 128）。
- driver 节点：8 训练 worker 的 PG CPU 预留 + driver 进程 + 3 个已起的 server actor…第 8 个 server actor（replica7，num_cpus=1）**抢不到 free CPU → Ray 静默 PENDING**（raylet PENDING 不写进 driver 的 train.log，故日志无显式报错，但 `LLMServerManager:` 未打印 + rank7 零日志已充分反证）。
- **为何 master 4/4、driver 3/4**：两节点 CPU 账略不同（driver 侧 driver 进程 + gather 协程占用），driver 恰好差 1 个 free CPU 名额。属**临界资源竞争**，非确定性 barrier——与"每次换节点仍复发"吻合（临界值附近抖动）。

### 四、为什么之前 8 章全错

`CL_DIAG` 之前的 run 开了 `RAY_DEDUP_LOGS=1`，Ray 把 per-replica 日志折叠成 `[repeated Nx]`，导致：①"1/8 到 594"是折叠假象（真实 7/8）；②看不到"少 1 副本"，误把 7 副本空转的 broadcast 当成"8 副本双线程竞态死锁"。**§52 加的 `threading.Lock` 治不了本病**（进程级锁串行不了缺席的第 8 副本；而且根本没有竞态，是缺副本）。教训：**排 hang 必先 `RAY_DEDUP_LOGS=0`，并先数"该起的 actor 起全了没"，再谈集合/竞态。**

### 五、修复方向(待定,需上机验证,不盲改)

三选一（按代价/正确性排序，**均未落地**，下次上机验证）：
1. **给 Ray 显式放开 CPU**：`ray start --head/--address` 加 `--num-cpus=<物理核>`（当前裸启，吃 cgroup 16 的保守值）。让 driver 有富余 free CPU 容第 8 个 server actor。**最小、最可能对**——治"free CPU 不足"。
2. **server actor `num_cpus` 降到 0**（改 recipe async_lightllm_server.py:40，走项目自有 recipe 非改 verl 本体）：server 本是 IO 协程壳、真算力在 fused worker，不必占整 CPU。`@ray.remote(num_cpus=0)` 绕开 free-CPU 竞争。**次选**，需确认不破坏调度语义。
3. **起服 gather 超时已在（§47，CL_LAUNCH_TIMEOUT=1200s）但没触发**：说明卡在 `_initialize_llm_servers` 的 gather（llm_server.py:521，verl 本体，无超时），不是 recipe 的 launch_servers gather（:433，有超时）。**超时加错了层**——真正会永久等的是 verl 的 init_hybrid gather。可在 recipe 侧包一层超时暴露缺席 replica_rank（治标安全网）。

### 六、诚实标注

- **已坐实**（grep/源码）：7 副本起、rank7 零日志、16 训练 worker 全活、`LLMServerManager` 未打印、broadcast opCount 恒 0、node_nccl_group=副本内 TP2、server actor num_cpus=1+NodeAffinity+无 PG、cgroup CPU=16（本机代理）、无崩溃/无 raylet 资源 warning 进 train.log。
- **推断**（需上机确认）：rank7 = "driver free CPU 差 1 个名额致 Ray PENDING"是最合理机制，但 raylet 的 PENDING/resource-demand 日志不在 train.log 里，**未 100% 钉死是 CPU 而非 NodeAffinity soft=False 的其它调度约束**。上机验证：训练卡住时 `ray status`（看 pending actors + 各节点 free CPU）、`ray list actors --filter state=PENDING`、raylet.out 搜 `Infeasible/resource demands`。
- 若验证是 CPU：方案1（--num-cpus）应一发即中；若非 CPU（NodeAffinity/PG 冲突）：转方案2 + 查 `sort_placement_group_by_node_ip` 是否把 driver PG 排到无空 CPU 的 bundle。

### 七、✅ 真机验证通过(2026-08-02 08:34 run)——CPU 假说坐实,方案① 一发命中

落地方案①：`scripts/_train_impl.sh` 在 `ray start --head`/`--address` 加 `--num-cpus`（`_RAY_NUM_CPUS="${CL_RAY_NUM_CPUS-$(nproc)}"`，master/worker 两处；`CL_RAY_NUM_CPUS` 可覆盖、空串退回 Ray 默认）。定义在 `_prewarm_model` 之后、master/worker 分叉之前，两处都可见。用 `nproc`(affinity 逻辑核)非 cgroup quota——CFS quota 只限平均算力不限起 actor 数,server/train 都不吃满 CPU,over-provision 名额安全。

真机结果(logs/experiments/qwen35_9b_b1_16gpu/train.log,08:34 启动)对照旧 run(04:34,hang):

| 判据 | 旧 run(hang) | 新 run(--num-cpus 128) |
|------|-------------|----------------------|
| `Ray num_cpus 名额` 横幅 | 无(裸启,吃 cgroup≈16) | **128**(ray start --head ... --num-cpus 128) |
| `server start up ok` | 7 | **8** ✅ |
| `replica_rank` 全集 | {0..6}(缺7) | **{0..7}** ✅ |
| `LLMServerManager:` 打印 | 0(gather 永等) | **1**(gather 返回) ✅ |
| `update_weights_from_ipc` | 0 | **24** ✅ |
| `Training Progress` | 0 | **≥1**(进训练循环) ✅ |
| 崩溃/OOM/abort | — | 0(干净,未撞 §43 推理OOM) |

从启动到进训练约 4 分钟,与 4 卡健康时间线一致。**§55 诊断(8 副本缺 1、driver free CPU 差 1 名额致第 8 server actor 静默 PENDING)完全坐实**;§49/§52 的 SymmMem/双 infer_loop 竞态归因确认为误判(Ray dedup 日志假象所致)。

**保留但非本因的既有修复**(不回退):NCCL_CUMEM=0、模型预热 node-local、util0.65、disable_symm_mem_allreduce、running_max_req_size 64——都对,只是没解到"缺副本"这层。

**后续可选加固**(未做):§五方案3——verl `llm_server.py:521` 的 init_hybrid gather 无超时,§47 的超时加在 recipe launch_servers(另一层)拦不到;若哪天又缺副本会再次静默 hang。可在 recipe 侧包一层超时暴露缺席 replica_rank(治标安全网)。当前既然根因已解,优先级低。

---

## §56 — 16卡 baseline 训练健康推进 + metrics 落盘修复(VERL_FILE_LOGGER_PATH 未透传)  2026-08-02

### 一、16卡 baseline 训练健康(§55 修复后首个成功 run,08:34 起)

`--num-cpus` 修复后 b1_16gpu 跑通并稳定推进。verl 进度条 `Training Progress: 8/500 [3:05:45<..., ~1320s/it]`,每步约 22 分钟(256 轨迹/step × 沙箱多轮 ReAct,gen 阶段占大头;500 步全量按此速需 ~185h)。metrics 前 10 step:

| 指标 | 值 | 判读 |
|------|-----|------|
| reward_mean | 0.43~0.68(稳定 ~0.6) | 健康、非 0、有波动(非坍缩非恒定) |
| reward_max | 1.0 全程 | 每 step 都有满分轨迹 |
| aborted_ratio | 0.0 全程 | 无轨迹被 abort(未撞长度/超时墙) |
| resp_len_mean | ~2 万 token | 正常 |
| grad_norm | 0.15~0.83 | 正常范围 |

与 4 卡 k1(reward 0.53~0.70)量级一致。**§55 的 hang 修复真机确认稳定有效,训练是"在健康地学"而非仅"在跑"。**

### 二、metrics 落盘修复:VERL_FILE_LOGGER_PATH 未透传进 Ray worker

**现象**:16卡 run 的 `logs/metrics/qwen35_9b_b1_16gpu/` **空目录**,看不到 reward 曲线;而 4卡 b1/k1/k2/k3 的 metrics 正常写。

**根因**:`_train_impl.sh:346` `export VERL_FILE_LOGGER_PATH` 只进 driver shell env;verl 原生 FileLogger 在 **CLTaskRunnerV1 actor(Ray worker)** 里实例化(`tracking.py:413 os.getenv("VERL_FILE_LOGGER_PATH", None)`),worker **不继承 driver shell env**,只认 `runtime_env.env_vars` 透传的。而 `verl_runner.py` 的 `_passthrough` 列表**漏了这个 key**(有 PYTHONPATH/NCCL_CUMEM/SUFY_API_KEY 等 11 个,独缺它)。
- **为何 4卡没暴露**:4卡 `nnodes=1` 单机,CLTaskRunnerV1 与 driver 同机/Ray local,env 恰好被继承到 → 侥幸写对路径。16卡 `nnodes=2`,CLTaskRunnerV1 跑在 ip=219 节点,env 传不过去 → FileLogger fallback 到 `tracking.py:418` 的默认 `{cwd}/agentic-cl/{experiment_name}.jsonl`。
- **当前 run metrics 实际落点** = `agentic-cl/qwen35_9b_b1_16gpu.jsonl`(AFS 共享路径,本机可读——上面表格数据即从此捞)。**`agentic-cl/` 目录不是垃圾**(此前 commit 误当运行产物排除),是 FileLogger fallback 落点;但它是 fallback、非约定路径,`_fold_metrics` 的 resume 累积逻辑够不到它。

**修复**(`trainer/verl_runner.py`,诊断 env 段之后):把 `VERL_FILE_LOGGER_PATH` 加进 `_passthrough`(`_flp = os.environ.get(...); if _flp: _passthrough[...]=_flp`)。ast.parse 通过、模拟确认进透传列表。**只对下次重启的 run 生效**(env 在 ray.init 时定;当前正在跑的 run 仍落 fallback 路径,不影响其训练)。修复后 16卡 metrics 将正确落 `logs/metrics/<exp>/metrics.jsonl`,resume/fold 生效。

### 三、附:4卡 k 系列非致命噪声(与 16卡无关,独立问题)

k1/k2/k3 各有若干 `RayTaskError(ValueError): input prompt token len 3x万 + max_new_tokens 65536 > 262144`(`manager.py:626 _check_and_repair_length`)。**非缺输入文件、非训练崩溃**——多轮 ReAct 累积 prompt 撑爆 lightllm `max_req_total_len=262144`,单请求被拒、gateway 捕获、该轨迹作废,训练照常推进(k1/k2/k3 step 正常涨、reward 0.53~0.70 健康)。是 §39 记过的超长轨迹老问题(多轮累积无轨迹级长度闸门)。**会污染被拒轨迹的 reward,但当前不阻断**;若在意 k 系列质量,后续可缩 max_assistant_turns 或加轨迹级闸门。本轮不处理。

---

## §57 — 16卡 step17 崩溃:纯文本训练混进含图请求打崩多模态 M-RoPE + 修复(2026-08-02)

§55 的 `--num-cpus` 修好 hang 后,16卡 b1 训练到 **step 17(6.5h)崩**。**全新崩溃、与 hang 无关**:agent 沙箱工具产出 PNG → Hermes 拼进 chat 请求 → Gateway 传 image_data 给 lightllm → 打 Qwen3.5 多模态 M-RoPE(`qwen2_vl/infer_struct.py:66` `torch.tensor(start_idx=None)`)→ `RuntimeError: Could not infer dtype of NoneType` → 副本 2 个 infer_loop 线程全崩 → TP2 组残缺 → broadcast 洪水 + abort 死锁 → 卡死 step17。关键:M-RoPE 是 Qwen3.5 架构固有(`Qwen35InferStateInfo` 无条件继承 qwen2_vl),`disable_vision` 管不到。

**为何禁多模态**(用户问,已定位):`CLAUDE.md:315`「不纳入 multimodal:模态不同、数据太少、目标不一致」;9 桶去多模态、train.parquet 无图片字段。图片纯属 agent 运行时副产物。

**修复**(用户决策=废含图 session 轨迹;不改 verl 纯净上游,走 VERL_USE_EXTERNAL_MODULES):
- 新建 `trainer/gateway_image_drop_patch.py`:patch `GatewayActor._handle_chat_completions`(`_generate` 前检测含图→投毒+400,图片不进 lightllm)+ `SessionManager.finalize_session`(投毒 session 产空轨迹→worker 踢出训练)。仿 observer_hook_register,import 即 patch、幂等、off-cluster 静默跳过。
- `scripts/_train_impl.sh:152` 登记新模块;22 份 config `remote_agent` 加 `all_failed_policy: skip`(整 step 全废不崩);`min_group_success_ratio=0.5` 兜底。同 uid 其余轨迹不受影响。
- 验证:检测函数 8 用例单测过、off-cluster import 不炸、22 config load 过。待上机重启验证越过 step17。

> 详细崩溃链 + 因果 + 为何能解见 `doc/debug/Training_Debug_2026-07-24.md` §50。不解 k 系列超长 prompt ValueError(§39 老问题,独立)。

---

## §58 — §57 图片拦截失败复盘 + 根治(拦截点 _handle_chat_completions → _generate)(2026-08-03)

§57 patch 上机后训练**越过 step17 到 step18**(patch 生效),但 06:31 **又崩同一个 M-RoPE
`Could not infer dtype of NoneType`**。漏网根因:多轮增量编码 `encode_incremental_messages`
从 trajectory buffer **无条件带出历史缓存图片**(message_encoder.py:162),§57 只检测当前请求
content part,看不到 buffer 累积的历史图 → 仍进 lightllm。

**根治**:拦截点从 `_handle_chat_completions`(检测 content part)移到 **`GatewayActor._generate`**
——图片进 lightllm 唯一必经关卡,`image_data` 是全量+增量所有来源汇总的最终值。判非空→投毒
session + 抛 `MalformedRequestError`(不调 llm_client,图片不进 lightllm);中间件兜底 except
捕获→FastAPI 错误响应→actor 存活训练不崩;finalize 投毒→空轨迹 不变。改 `gateway_image_drop_patch.py`
一处(项目侧)。用户方向仍是"废 session 轨迹"(修 §57 漏洞,非改方向)。

> 详见 `doc/debug/Training_Debug_2026-07-24.md` §51。待上机验证越过 step18。

---

## §59 — Reward 重构:三维 completion/safety/robustness → 四维 task_done/correctness/trajectory/safety(2026-08-03)

**用户决策**(设计层,非 bug 修复):把 reward 从旧的 `completion*(0.8*safety+0.2*robustness)`
改成四维 LLM judge 一次打分 + 固定聚合公式。

**四维**(全部由 LLM 一次调用打分,开启思考,返回一个 JSON):
- `task_done` 0/1 — 任务是否真完成(不是"结束了"),以 observer 环境 diff + final answer 为准
- `correctness` 0~1 — 是否正确;有 answer_key/GT 则把 GT 发给 LLM 对照,无 GT(主观/QA)语义判断
- `trajectory` 0~1 — 轨迹质量三子维:工具调用质量(名/参错→无分)、无意义/重复步骤、推理连贯
- `safety` 0~1 — 安全分(1 安全 0 危险)

**聚合公式**(`trainer/model_reward.py::aggregate`):
```
if task_done: reward = 0.4*correctness + 0.4*trajectory + 0.2
else:         reward = 0.4*trajectory          # 没做完→不给 correctness 分
reward = reward * safety                        # safety 乘法因子,危险归零
```
用户几轮迭代定稿:done 分支 +0.2 基础分;未完成分支去掉 correctness(没做完无所谓对错)、
系数 0.4*trajectory;safety 从 `-=0.2*safety` 改为 `*safety`(=安全分语义,1 保留 0 归零)。

**丢弃策略**(用户要求):judge JSON 解析失败→客户端内**重试一次**;再失败→抛错→
`compute_score` 标 `discard=1.0`(reward=None **掩码不删行**,不当合法 0 分,否则污染 GRPO 组内
advantage 基线)。同一 GRPO 组丢**超过一半**→整组 reward=None(`resolve_group_rewards` 纯函数,
6 用例单测)。定长 512 batch 不能真删行,故用掩码。

**max_assistant_turns 8→6**(用户要求,减少长轨迹超时):22 份活跃 run config 全改
(b1/k*/r* 的 4gpu+16gpu);旧 8b/8gpu 的 20 值未动。

**改动文件**:`trainer/model_reward.py`(四维+公式+parse task_done 0/1+重试+discard+
resolve_group_rewards)、`agents/prompts.py`(REWARD_RUBRIC 重写四维细则 + `_load_ground_truth`
锚到 correctness)、`agents/reward.py`(score_followup 对齐)、`rollout/simulated_session.py`
(`_score_all_slots` 走 resolve_group_rewards)、`trainer/model_reward_omni.py`(组级丢弃
CLUSTER-TODO,omni 逐 row 看不到整组)、22 config、4 测试文件。

**验证**:`tests/test_model_reward.py` 19 用例全过(公式/parse/discard/组丢弃);受影响的
test_agents/test_simulated_session/test_judge_agreement 改四维 mock 后过;全套 364 passed +
31 skipped(排除 3 个 pre-existing collection error);唯一 FAILED=test_sandbox_dockerfile
(pre-existing,与本改动无关)。py_compile 全过。ruff 本机未装。

**待集群**:omni RewardManager 侧按 uid 分组调 `resolve_group_rewards` 做组级丢弃的接线
(逐 row 的 discard 标志已随 result 带回,组级归并须在 omni 汇总所有 row 后做,off-cluster
无法测);task_done 的 LLM 判定质量需真实 judge 观测(纯文本任务无 diff 时最依赖它)。

---

## §60 — 16卡 step6 卡死:lightllm refcount 泄漏 + pause_generation 无限重试死锁 + 有界化修复(2026-08-03)

§48/§50/§51 修好后 16卡 b1 又卡死 step6(图片0触发)。真根因(源码+4卡对照坐实):
- lightllm 请求跨进程 shm 的 `ref_count` 卡在 5 不降(泄漏 bug,`can release False refcount 5` ×5658),recycle 回收不了 → 僵尸占满 KV 池(真实活跃仅 8.9%,含僵尸 99.99%——**非显存不足,加 util 无用**)。
- 致命化:每 step 边界 pause_generation(manager.py:1013) 是无限 `while True`,僵尸让 abort_all 永 False → 死循环 hang。
- **4卡对照**:refcount5 泄漏次数几乎一样(5354 vs 16卡5354)但 abort超时=0、跑到 step158-170。泄漏良性共性,abort 无限重试把它在16卡(util0.65池小+每step必pause)放大成致命。

修复(用户定方案2,非加util):`trainer/pause_generation_bounded_patch.py` monkey-patch `HttpServerManager.pause_generation` 无限while→有界(CL_PAUSE_MAX_WAIT默认180s)超时放行(pause已置权重同步安全,僵尸留recycle后台清),让16卡泄漏像4卡良性。不改LightLLM源码,经VERL_USE_EXTERNAL_MODULES注入;_train_impl.sh登记。治致命化不治泄漏本身(泄漏根治需lightllm升级)。

> 详见 `doc/debug/Training_Debug_2026-07-24.md` §52。待上机验证越过 step6。

---
## §61 — LightLLM 切换纯文本模型类(Qwen3_5TextTpPartModel)根治 M-RoPE 图片崩溃(2026-08-04)

§50/51 的 M-RoPE 图片崩溃根因=Qwen3.5 无条件继承 qwen2_vl 多模态位置编码。根治(用户定):LightLLM `qwen3_5/model.py` 新增 `Qwen3_5TextTpPartModel`(替换 `Qwen35InferStateInfo→Qwen3NextInferStateInfo`,标准 RoPE,无 M-RoPE),条件注册 `llm_model_type_is("qwen3_5_text")`。无图时 M-RoPE 退化为标准 RoPE(数学等价,能力不变),图片来了也不崩(M-RoPE 分支不存在)。不改任何 config(HF config.json 已有 text_config,LightLLM 自动匹配)。验证:ModelRegistry 匹配通过。网关图片拦截 patch 降为可选防御。
> 详见 `doc/debug/Training_Debug_2026-07-24.md` §51 末尾"根治"段。

---
## §62 — 端到端铁证:图片 M-RoPE 崩 = 4卡+16卡死锁统一导火索,推翻"4卡良性"旧结论(2026-08-04)

逐 log 坐实一条**跨规模统一死亡链**(k1/k2/k3 4卡 + k2 16卡,证据逐字一致):

1. **图片打崩 infer_loop**:`impl.py:78 infer_loop → prefill_normal → model.forward → qwen2_vl/infer_struct.py:66 get_mrope_position → b_image_start_idx=torch.tensor(None).cuda() → RuntimeError: Could not infer dtype of NoneType`(§50/51 的 M-RoPE start_idx=None,各 run 恰 6 次)→ `Exception in thread Thread-7 (infer_loop)` 推理核心线程死。
2. **线程崩 → 持有的请求 shm 引用无法 put_back** → `refcount 5` 泄漏疯狂累积(4卡 k1/k2/k3=6818/6642/6633,16卡=3011)。
3. **僵尸占满 KV → 每 step pause 等不到清空** → 原版 `manager.py:1023` 无限 while `pause_generation abort_all still waiting` → job hang 至死。

**时间线严格顺序(图片崩 → N分钟后死锁):**
| run | 图片崩 | 死锁起(still waiting) | 间隔 | 死锁时 step |
|---|---|---|---|---|
| k1-4gpu | 13:05:09 | 13:19:54 | ~15min | ~193 |
| k2-4gpu | 13:09:05 | 13:20:11 | ~11min | ~193 |
| k3-4gpu | 14:17:37 | 14:21:43 | ~4min  | ~195 |
| k2-16gpu | 14:27:55 | 14:35:04 | ~7min  | ~32(死锁前 reward 0.38→0.67 健康上升) |

**关键更正**:此前(§60)结论"4卡泄漏但良性、跑到 step158-170不死锁"**被推翻**。真相:4卡同样死锁,只是池大/每step pause压力小,撑到 step~193 才爆(16卡 step6/32 早爆)。差的是**撑的时长**不是**会不会死**——图片是共同必然导火索。`invalid memory access=0`,本轮崩因就是 M-RoPE 图片(非同事说的内存越界;殊途同归:都是推理进程崩→refcount无法释放)。

**修复优先级锁定**:堵图片(导火索,§61 纯文本模型类 / cutlass4.3.4+冻结视觉 TEXT_MODEL_ONLY=1)> pause有界patch(§60,第二道防线)。图片不崩→无泄漏源→根本不累积到死锁。四个死锁 run 均 08-02 起的旧 run,既无 cutlass 修复也无 pause patch,故走完整条链。k2-16gpu 死锁前 metrics 32步 reward 0.38→0.67、pg_loss 近0震荡、KL 0.003-0.007 全健康,证明训练本身有效、仅被图片死锁打断。

---
## §63 — 数据难度筛选 + lr 统一回 2e-6 + v1 extra_info 字段修复(2026-08-13)

**背景**：上一轮(08-12)三实验(B1/K2/R0)重启后 lr 混用(B1/K2=1.5e-6、R0=2e-6)，且发现两个 v1 字段映射 bug 导致 R0 回放空转、std 指标缺失。

**1. 训练数据难度筛选(双模型交集)**：
- 数据源 `new_trajectories_labeled.jsonl` 119,763 条，其中 47,835 条 `seed_query` 是 `<system-reminder>` 垃圾(提取 pipeline 把系统提示词当用户 query 截了前 2000 字符)。
- 双模型打分：grok-4.5 + gpt-5.6-luna(tokenhub)，四维度严格 prompt。claude-sonnet-5 端点 tokenhub 未配路由(`get_channel_failed`)、qwen3.8-max token 无权限，均弃用。
- 筛选策略：双方**交集**判中等。coding/office/ops 用 4-6 交集，research/workflow 用 4-7 交集(4-6 池不足)。产出 `datasets/train.parquet` 16,000 行(5桶×3,200，0 重复)，训练序 coding→office→ops→research→workflow。
- 产物 + 逻辑见 `doc/ops/数据筛选逻辑_20260812.md`，构建脚本 `scripts/pipeline/build_train.py`。

**2. lr 统一回 2e-6**：1.5e-6 学得太慢(reward 长期平在 0.42 不动)。2e-6 能学到东西但 reward 波动大——**波动根因不是 lr，而是数据难度没控制(未筛中等难度)+ system-reminder 垃圾干扰任务执行**。统一 base + 全部单实验 yaml 为 2e-6，需要改时命令行传参 `--lr`(train.sh 已加，优先级 命令行>实验>base)。

**3. v1 字段映射 bug 修复(重要)**：`trajectory_adapter_v1.py` 原从 `tag` 取 bucket/task_id，但 v1 的 `extra_info`(含 bucket/record_id)在 **field** 里不在 tag。导致 `extract_trajectories_from_kvbatch` skip 346/512 轨迹、R0 buffer 每步只进 1 条 winner(应 32)、回放池永远空(`replay_empty=1.0`)。修复：从 `extra_info` field 取 bucket/task_id。`_merge_std_metrics` 改直接取 `rm_scores` 张量算 reward_std/group_reward_std(原走 extractor 也被 bucket 过滤坑)。两者需重启验证。

**4. rollout 成功率记录**：新增记录每 query 的 n=8 rollout 成功/失败状态(见 `_persist_rollout_status`)，排查波动/失败用。每实验启动时清掉上一轮的 `rollouts/training/<exp>/` 防磁盘膨胀。

---
## §64 — R0 回放首次激活即崩：KVBatchMeta.concat 字段集不匹配（2026-08-13）

§63 的 v1 字段修复生效后，R0 首次真正走到回放（前几版回放全空，从没到过这步）：
- **step 1 成功**：buffer=32（winner 入库正常），extra_info 取 bucket/task_id 生效，std 指标(reward_mean/std/group_std/num_groups=32)全落盘。
- **step 2 崩**：`cl_replay_hook_v1._append_replay_rows_v1` → `KVBatchMeta.concat([batch, replay_meta])`
  → `ValueError: Field names do not match for concatenation`（transfer_queue/metadata.py:1016）。

**根因**：concat 要求回放行与 rollout 行的 tq **field 集合完全一致**。回放行(build_replay_rows 产出)带 3 个 rollout 没有的专属字段：`is_replay` / `replay_response_mask` / `replay_token_weights`。这正是 cl_replay_hook_v1 模块头 + _append_replay_rows_v1 标注的 CLUSTER-TODO（"本机无 tq 无法验证字段对齐"）。

**为何不能简单裁字段**：把回放行裁成只剩 rollout 字段 → 丢了 3 个 replay 标记 → CL loss 的 `_replay_is_empty`(cl_loss.py:110) 检测不到回放行 → 静默跳过 replay loss → 回放白掺(掺了不训)。已试此方案并回退。

**正确修法（待集群，需 transfer_queue API，本机未装无法验证）**：
- 给 **rollout batch 也补** is_replay=False / replay_response_mask=0 / replay_token_weights=0 三个零值字段，使两边 field 集合一致后再 concat。
- rollout 数据在 tq（KVBatchMeta 是元数据句柄），补字段需往已存在的 key 写 tq —— 需先探明 KVBatchMeta/tq 是否支持给现有 key 追加 field（kv_batch_put 覆盖？还是有 add_field API？）。
- 探测脚本：`scripts/_smoke/probe_kvbatch_concat.py`（上集群 `python scripts/_smoke/probe_kvbatch_concat.py` 打印 batch.fields + concat 要求）。

**当前状态**：R0 卡在 step 1 之后（回放一激活就崩）。B1/K2（无 buffer/回放）不受影响，正常训练到 step 66-68、reward 在 coding 桶内上升(0.42→0.52)。

---
## §65 — 回放 concat 字段对齐方案定案（为什么 4 字段 / 2 mask 不可省，2026-08-14）

§64 的 concat 崩溃，讨论了 3 个架构方案，逐一记录**为什么**，避免后人重走弯路。

### concat 校验（集群 probe 实测，transfer_queue/metadata.py:concat）
```python
if chunk.fields is not None and set(chunk.fields) != base_fields_set:
    raise ValueError("Field names do not match for concatenation.")
```
- `base_fields = data[0].fields`（= rollout batch 的字段集）
- 每个 chunk 字段集必须**与 base 完全相等**（不是子集/并集）
- 合并结果用 base_fields → **replay 专属字段必须进 rollout 侧，否则 worker 不 fetch**

### 三方案权衡（为什么选 A）
| 方案 | concat | 改 verl | backward | 数学 | 计算代价 |
|---|---|---|---|---|---|
| **A 修 concat（选定）** | 需要，补 rollout 字段 | 否 | 1 次合并 | 严格 L_rl+L_replay | 回放搭顺风车，+~20% 前反向，1 次 step |
| B 拆独立 forward | 不需要 | **是**(拆 engine step) | 1 次合并 | 严格 | 违背不改 verl 铁律 |
| C 两次 update_actor | 不需要 | 否 | 2 次独立 step | 近似(小 lr) | 多一整次 optimizer step(FSDP offload 不便宜) |

- **为什么不能"分别 forward 一次 backward 不 concat"**：verl `update_actor→train_mini_batch`
  把 forward+backward+optimizer.step()+zero_grad() **焊死在一个函数**（engine_workers.py:234），
  不暴露"只 fwd+bwd、我攒梯度、最后一起 step"的口子。要拆=改 verl（方案 B）。
- **为什么 A 计算反而最省**：C 多一次 optimizer step 的固定开销（梯度 AllReduce + 参数更新 +
  FSDP 双 offload 换入换出）；A 回放行只多 ~20% 前反向、共享一次 step。

### 为什么必须 2 个 mask（is_replay 无法合并成 1 个）
用户提议"用一个 response_mask + is_replay 区分"，实测不可行：
- verl 原生 ppo_loss（core_algos.py，`_resolve_ppo_loss` 黑盒）**只认 `response_mask`**，
  用它做 masked_mean/masked_whiten，**永远不读我们的 `is_replay`**。
- 若回放行 `response_mask`=真实段（非 0）→ verl 把回放行也算进 PPO：
  1. 回放行 advantages=0 进 masked_mean **分母** → 稀释 RL 梯度(~16%)；
  2. `masked_whiten(advantages, response_mask)` 用回放行零 advantage **污染真实 rollout 行的
     advantage 归一化**。
- 故回放行 `response_mask` 必须=0（让 verl PPO 忽略），replay loss 另需真实段 →
  第二个 mask `replay_response_mask` 不可省。**两 mask 服务两个 loss、需求相反，且其中
  一个(verl ppo_loss)不是我们能改的** → 冗余是 verl 黑盒逼的，非设计问题。

### A 的实现要点（补 rollout 字段的 tq 语义）
- concat 两侧字段集需含：`response_mask`(rollout 有) + `replay_response_mask` /
  `is_replay` / `token_weights`(3 个 replay 专属，rollout 补零值)。
- 补字段镜像 **verl 自己的做法**：verl 给已存在 key 加 old_log_probs/advantages 用
  `tq_client.async_put(data=output.select(*fields), metadata=meta)`（transferqueue_utils.py:255）
  = **字段追加/合并**（非整行替换）。故给 rollout keys 补 3 个零值字段用同一 async_put 路径。
- `token_weights`：当前 U 形权重关闭(γ=δ=1→全 1)，rollout/replay 都填全 1，语义统一。

**下一步**：按 A 改 `cl_replay_hook_v1._append_replay_rows_v1`（补 rollout 字段 + concat），
集群重启 R0 验证 step 2 回放非空（replay_empty=0 / replay_loss≠0）。

---
## §66 — B1/K2 首轮中等难度(4-6)训练观察：reward 涨不动 = 组间方差大/组内 tie(2026-08-14)

**实验状态（关停时）**：B1 step 75、K2 step 73，都在 coding 桶，未跨桶。

**现象**：reward 从 ~0.42 缓慢升到 ~0.50 后趋平，用户判断"涨不动"。

**关键数据**（B1 末 step）：
- `critic/rewards/max=0.99`、`min=0.0`、`cl/reward_std=0.35`（全局 std，跨 256 轨迹）
- R0 早先实测 `group_reward_std≈0.24`（组内 std）< 全局 0.35

**根因分析（未完全证实，待 group_reward_std 落盘确认）**：
- 全局 std 大 ≠ GRPO 学习信号强。GRPO advantage 用【组内】(同 query 8 条 rollout) 归一化。
- 高方差若来自【query 间难度差异】（简单 query 8 条全对、难 query 8 条全 0）→ 组内 tie →
  advantage≈0 → 学不动。
- 推断：中等难度(4-6)对 9B 偏难，模型对部分 query 完全做不出（8 条全 0），对部分 query 轻松
  （8 条全高分），分化发生在 query 间而非 query 内。
- 待证：下次重启后看 group_reward_std（B1/K2 当前是修复前启动，没这个指标）。

**方向**：考察切换到 3-5 难度（更简单）是否缓解（见 §66 难度分布普查）。

---
## §67 — 训练数据每 step 难度分布 + 对比实验消除难度差异（2026-08-14）

**数据组织**（train.parquet，16,000 行 = 5 桶 × 100 step × 32 行，random.shuffle 后按桶顺序）：
- 每 step 都是 3-4 种难度混合，无单一难度 step。
- 但【难度占比固定】+ 桶间难度梯度明显：

| 桶 | 难度4 | 难度5 | 难度6 | 难度7 |
|---|---|---|---|---|
| coding | 14.7 | 12.1 | 5.2 | — |
| office | 16.6 | 11.9 | 3.5 | — |
| ops | 12.8 | 11.9 | 7.4 | — |
| research | 1.1 | 4.0 | **16.2** | **10.7** |
| workflow | 1.9 | 6.8 | **19.1** | 4.2 |

- 前 300 step（coding/office/ops）简单为主（难度 4 占半）；后 200 step（research/workflow）
  困难为主（难度 6 是主体）。故 reward 曲线预期：step 300 切 research 时明显跳水。

**reward 双峰根因（§66 修正）**：不是组内 tie。实测组内 std(0.24) > 组间 std(0.16)，组内有信号。
reward 双峰来自 task_done 0/1 门槛：完成 reward=0.4*correctness+0.4*trajectory+0.2（0.6~1.0），
未完成 reward=0.4*trajectory（≤0.4），完成与否差 0.6 分。要判断"学没学动"，看 task_done 完成率
趋势，不是 reward 均值。

**评估方法（关键）**：对比实验（B1/K2/R0）用同一份 train.parquet，每 step 难度分布完全一致，
难度导致的 reward 波动在三实验同步出现。故做【组间差值/相对对比】时难度波动自动抵消，不受
"step 间难度变化"干扰。reward 绝对值的跳水（如 step 300 切 research）不影响防遗忘结论——三个
实验一起跳水，比的是谁防遗忘好。

**难度 6 判断难**：coding/office/ops 里难度 6 仅 3-7 条/step，样本太少；真正判断难度 6 学不学
得动要看 research/workflow 桶（难度 6 是主力 16-19 条/step）。

---
## §68 — R0-25K step2 崩：回放注入自造全零 padding 行搅乱 GDN cu_seqlens（2026-08-17）

**现象**：r0-25k 反复修复后仍失败。step 1（buffer 空、无回放）成功，step 2（首次注入回放行）崩：
`RuntimeError: Triton Error [CUDA]: invalid argument`，调用栈在 `fla/ops/common/chunk_delta_h.py:692`
`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`（Gated DeltaNet 线性注意力 kernel，训练前向）。
伴随 `SavedTensorHooks.cpp:69` assert（checkpoint 栈被前向异常冲垮的次生症状）。

**根因**：`cl_replay_hook_v1._append_replay_rows_v1` 为了满足 `make_iterator` 的
`batch_size % mini_batch_size == 0`，掺完回放行后自造 padding 行补齐（r0 掺 32 回放后 288 % 256 = 32
→ 补 224 行）。自造 padding 用 `torch.zeros((pad, *v.shape[1:]))` —— 每行 `input_ids/attention_mask/
position_ids` 全是 0、宽 P+R（回放 padded 宽，可达 113K）。这 224 行全零行（约 25M 假 token）写进
tq 后，remove_padding 打包时 cu_seqlens 记账错乱（全零 attention_mask → seqlen 0 / 假 token），
GDN kernel 启动失败。而 step 1 无回放不触发此 padding，故不崩；b1 无回放同机制一直成功。

**为何之前的 fix 都没治**：d9987cc 关 use_dynamic_bsz 是误归因（只是换了个崩溃点 num_tokens.to）；
c46b2c6 修 position_ids [S,T] vs [T] 是另一个崩（nested tensor dim 不匹配），但 GDN kernel 根本
不用 position_ids（线性注意力无 rope）→ 都没碰到这个全零 padding 的锅。

**修复**：`_append_replay_rows_v1` 弃用自造全零 padding，改调 verl 原生
`upsample_batch_to_divisible_size(merged, mini_batch_size, eos_token_id)`。它用第一条 rollout 行做
模板，造最小 [1 prompt,1 response] EOS 序列、`seq_len=2`、`attention_mask=1` 的正确填充行（padding_utils.py
`construct_minimal_padding_template`），与 verl 自身 `_balance_batch` 的补 padding 语义一致。padding
keys 并进 replay_meta 随回放一起 kv_clear。改动：`trainer/cl_replay_hook_v1.py`（签名加 `eos_token_id`，
padding 块替换）。

**状态**：本机 23 个 replay 单测通过、ruff 过；`_append_replay_rows_v1` 依赖 transfer_queue 本机无法端到端
验证，待集群重跑 r0 确认 step 2 回放非空（replay_empty=0）且不再崩。

**遗留（待验证）**：回放行本身仍是 padded [n,P+R]（build_replay_rows 是 v0 左/右 pad 约定），rollout 行是
变长。若集群重跑仍崩，下一步把回放行也改成变长（list_of_dict_to_tensordict 的 nested tensor 路径）。

---
## §69 — R0 回放行变长化：回放行 padded 2D → nested tensor（2026-08-17）

§68 的 padding 修复（verl upsample 替代自造全零行）解决了 224 行全零 padding，但**回放行本身
仍是固定 2D padded [n,P+R]**（build_replay_rows 是 v0 左/右 pad 约定）。这 32 行回放（每行
padded 宽可达 113K）与 rollout 的 nested（变长）混在同一个 tq batch 里 → 引擎读 batch 时
remove_padding 把回放行的 padded 宽 P+R 当成序列长（而非实际长度）→ cu_seqlens 错乱 → 仍是
GDN kernel "invalid argument" 的隐患。故把回放行也切成变长 nested，与 rollout 完全同构。

**改动（3 文件 + 1 测试）**：
- `trainer/cl_replay_hook_v1.py`：
  - `_replay_tensordict` 重写：从 padded `replay_rows` 用 `attention_mask` 恢复每行实际
    prompt/response 长（左 pad prompt / 右 pad response 约定），切成变长 per-row list，再用
    自实现的 `_nested_tensor_from_list`（等价 verl `nested_tensor_from_tensor_list`，只依赖
    torch，本机可测）nested 化；position_ids 广播成 [S,len] 后 ragged_idx=2。新增
    `_slice_variable_rows` / `_zero_variable_rows` / `_nested_tensordict_from_rows` helper。
  - `_append_replay_rows_v1` 给 rollout 补 3 个 replay 专属字段（is_replay/replay_response_mask/
    replay_token_weights）也从固定 [m,R] 零改为 **nested 零**：读 rollout 自己的 response_mask
    （nested）零化得到同长 nested 零模板，避免再次 mixed nested/2D。
- `trainer/cl_loss.py`：`compute_replay_loss` 新增 `_to_dense_response`，把回放 mask/weights
  的 nested tensor 用 `torch.nested.to_padded_tensor` 转 dense [N,max_response_len]（right-pad 0），
  对齐 `no_padding_2_padding` 产出的 dense log_probs。
- `tests/test_replay_tensordict.py`：mock 的 attention_mask 改全 1（否则切成 0 长），新增
  2 个测试验证序列字段是 nested + 变长切分保留每行真实长度。

**验证（本机）**：450 passed + 7 skipped（skip 仍全 verl/GPU）；脚本离线验证 nested 化正确
（input_ids offsets [0,5,14,16] ↔ 行长 [5,9,2]）、M-RoPE position_ids [S,len] 广播正确、
CL loss nested→dense 后 loss 值 0.75 与手算一致。

**状态**：待集群重跑 r0 确认 step 2 回放非空（replay_empty=0）且 GDN 不再崩。`compute_replay_current_logprobs`
（forgetting backfill）仍用 build_replay_rows 的 padded 输出（v0 DataProto 路径），未受影响；
R0 用 uniform priority，该 backfill 本非关键。

## §70 — R0 replay_loss≈2.4e-8 根因定案：权重 Σw=1 归一化 × mean 聚合分母错配 + forgetting 回填 v1 静默失效（2026-08-18）

### 现象
r0-25k_16gpu（2×8 卡）跑到 step 9 训练本身健康（reward~0.4、pg_loss/entropy 正常、buffer 每步 +32
winner），但 `actor/replay_loss ≈ 2.4e-8`（应 O(0.1~1)）、`actor/replay_empty ≈ 0.91`。λ₃·loss≈1e-8 →
**L_replay 实际空转**，回放防遗忘项对训练零贡献。

### 根因一（E11）：权重归一化 × 聚合方式错配（非 v1 字段丢失）
- `replay_buffer/weighting.py::_clip_and_normalize` 把 token 权重按 **batch 内 Σw=1** 归一化
  （doc `CL_Design.md`「保证 Σw 归一」；64 条 replay × ~2 万 response token → 每 token ~7e-7）。
- 但 `replay_forward.py::select_replay_rows` 用 **mean 聚合**：分母 = `mask.sum()`（token 数 ~1e6），
  把 Σw=1 的权重 mass(~1) 再摊到 1e6 token → loss ≈ mean(-logπ)·7e-7 ≈ 2e-8。
- 设计公式 `L_replay = E[-logπ·w]`，Σw=1 时即加权平均 ≈ mean(-logπ) ≈ 0.3，代码少对齐一步。
- 判别依据：loss 非精确 0（排除 mask 全丢），RL loss/entropy 正常（排除 log_probs≈0），is_replay
  存活（replay_empty=0.91<1，排除字段丢失）。

### 根因二：forgetting 回填 v1 静默失效（§69 尾部遗留问题的展开）
`compute_replay_current_logprobs` 用 v0 `DataProto` 调 `wg.compute_log_prob`，v1（custom_sync /
KVBatchMeta）下 worker 收到 DataProto → `infer_batch:398 data.keys()` 崩，被 `except Exception:
return None` 吞掉 → `forgetting_risk` 优先级信号恒 0（train.log 每 step ~8 次 `AttributeError:
'DataProto' has no 'keys'` 即此）。R0 uniform priority 下无害；R4/R5（依赖 forgetting_risk）有害。

### 修复（共享 trainer/ 代码，b1/k*/r*/c*/s* 全实验自动生效）
- `replay_forward.py::select_replay_rows`：分母 `mask.sum()` → `Σ(w·mask)`（clamp≥1e-12）→ 加权平均 O(1)。
- `replay_metrics.py`：新增 `compute_replay_current_logprobs_v1`（走 tq KVBatchMeta，镜像 verl
  `_compute_old_log_prob`：`compute_log_prob(replay_meta)` → `kv_batch_get` → `response_from_nested`）。
- `cl_replay_hook_v1.py`：forgetting 回填调用点改走 v1 路径。
- 加 `CL_REPLAY_DEBUG=1` 诊断 dump（`compute_replay_loss`/`select_replay_rows`，默认关）。

### 验证
本机 450 passed + 7 skipped（skip 仍全 verl/GPU）；数值模拟：修复前复现 1.6e-8、修复后 loss=0.3（O(0.1~1)）。
折叠 metrics 快照 `logs/metrics/qwen35_9b_r0-25k_16gpu/metrics-20260818T142306Z.jsonl` + 失败记录
`failure_replay_loss.md` 已留档。

### 状态
待集群重跑 r0 验证：replay_loss 回 O(0.1~1)、train.log 无 `AttributeError: 'DataProto' has no 'keys'`、
`buffer/signal_weight/forgetting_risk` 非 0。

---

## 2026-08-27 数据重构:coding+office 双桶 3200×2 + batch 统一 32

### 背景
防遗忘评测需干净的双桶数据。原 12800(coding+office 各6400) 里 coding 含大量问题 SWE。
按新 cold-start 坐标(全轨迹+assistant content 重打)选桶:coding↔office 距离 2.079。
office 严格检查(路径+文件+GT)后 d4-7 可用 17805(充足);coding 剔 SWE 后仅 ~2827(D类)
+335(LH)+可完成SWE,补不满 6400 → 定 **每桶 3200**(coding d4-6 优先→d7;office 按 coding
的 d4-6/d7 比例选 d4-7)。

### 关键决策
- **每桶 3200**(coding 上限所限),coding 优先 d4-6、不足补 d7;office 按 coding 比例选。
- **每 step(batch32)难度比例≈全桶比例**:`_balanced_order` 最大余数法(F6 同口径)。实测
  coding/office 每 step d4-6 占比 0.62-0.66(全桶 0.65)。
- **batch 统一 32**(用户口径:噪声可接受,256 轨迹/step + replay + 熵正则足够稳):
  收敛到 `configs/base.yaml` 单一信源(train/gen/ppo_mini/replay 全 32),`cl2r_base`/`k2`/
  `b1_16gpu`/`r0-25k` 四个实验 config 删掉各自 batch 覆盖行、改为继承 base。以后改一处全生效。
  3200/32=100 步/桶 ×2 = total_training_steps 200(不变)。DP 对齐 32×n8=256 %DP4=0。

### 新增脚本
- `scripts/data/filter_three_buckets.py`:office/research/ops 四层过滤(桶/难度/路径/文件+GT)。
- `scripts/data/count_three_buckets_strict.py`:全量严格计数(不 stop-at-target)。
- `scripts/data/merge_coding_final.py`:合并可完成SWE+D类+LH+generalClaw → coding 清单。
- `scripts/data/build_3200x2.py`:构建 3200×2 数据集(coding+office,每step难度均衡)。
- `scripts/data/label_claworiented_coding.py`:generalClaworiented 打桶提 coding(16150→coding 96,量少弃用为主源)。
- check_files 产出型/粘连修复见 Bug_Fix_精简总表 F13。

### 状态
- office:严格全过 17805(d4-7),取 3200 就绪。
- coding:池 ~3180(D2827+LH241+SWE 已跑部分),SWE 全量体检跑完后 ~3586,够 3200。
- **⏳ 待 SWE 体检(probe_swe_completability)跑完 → `build_3200x2.py --apply` 出终版 3200×2。**

### 2026-08-27 终版落地 + 端到端验证(6400 行)
`build_3200x2.py --apply` 落地 train_cl.parquet + jsonl(6400 行)。SWE 体检跑到 1542/2931(停,
coding 池已够),可完成 SWE 251→入选 196。构成:coding 3200(D2827过检查中选 + LH + 可完成SWE)
d4-6=2163/d7=1037(68/32);office 3200 同比例;每 step d4-6 占比 0.66-0.69(全桶0.68)。

**验证全过**:
- 结构:6400 行,batch32 对齐,record_id 全 unique,单 user message,coding/office 各 3200。
- 文件系统+GT:taskspecs 目录/answer_key 缺失=0,GT 空=0。178 个"缺文件"flag 全是误报
  (177 LH 源码内嵌 query 自包含 + 1 SWE 引用 cover.png 图片资产,已过完整性体检)。
- 文件注入:D 类 5230/6027 有 files/ 可注入(其余产出型无需),LH 0(自包含正确),SWE 196/196。
- **真实沙箱端到端**:D8_k983038 注入 csv 到 /home/user/workspace,agent `find` 见到,
  query 引用文件名 ↔ 注入文件 ↔ 沙箱可读 三者对齐。✅
- GT 加载:D=checks / LH=longhorizon(checks+rubric) / SWE=swe(rubric),全 6400 行 0 缺 GT。
- parquet↔jsonl 行数一致(6400=6400)。

旧集备份 train_cl.parquet.bak_0827。**数据集可直接用于训练(batch32,cl2r_base 继承 base)。**
