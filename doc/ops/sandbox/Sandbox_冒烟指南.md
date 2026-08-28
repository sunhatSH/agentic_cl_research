# 沙箱冒烟 — 精简指南（新集群权威版）

> 2026-06-30 在新集群北京区 `ap-beijing` 跑通。本文是当前唯一权威 quick-start。
> 概念见 [`Sandbox_概念与术语.md`](Sandbox_概念与术语.md)，接口见 [`接口使用_Sandbox与三Agent.md`](接口使用_Sandbox与三Agent.md)。

---

## 1. 现状

- **镜像**：`tcr-rl.tencentcloudcr.com/agentos-cl-namespace/agentic-cl-sandbox:v2`（装了 OpenClaw + Hermes Agent + openai/fastapi/pandas 等）
- **Tool**：`agentic-cl-sandbox`（ToolId `sdt-jrpn0rfo`，4C/8Gi/20Gi，VPC `subnet-ljfuh7ln`/`sg-irz5h5ed`，envd:49983，`/init`）
- **VPC**：`agent-rl-vpc`（`vpc-llc6ty0w`）。沙箱有公网出站、能连开发机、能连 sufy（`openai.sufy.com`）——沙箱内 hermes 经 sufy 调模型。旧 tokenhub 弃用。
- **代码执行**：走 `commands.run`（envd 49983），**不走** `/execute`（49999 Jupyter，500）

## 2. 每次开工

```bash
cd /mnt/afs_toolcall/sunhao4/agentic_cl_research
source scripts/load_tencent_env.sh   # 导出 CAM + E2B + E2B_VALIDATE_API_KEY=false
# 需要下包时加: export http_proxy=http://sysagent:c08400bf@10.119.176.202:3128
# tccli 已 symlink 到 ~/.local/bin；e2b/tccli 在 cold env (py3.10)
```

## 3. 查状态

```bash
bash docker/sandbox/ops/ops.sh query                 # 一键查 Tool + Instance
tccli ags DescribeSandboxToolList     --region ap-beijing
tccli ags DescribeSandboxInstanceList --region ap-beijing
```

## 4. 执行代码（最小）

```python
from rollout.sandbox_client import make_sandbox
sb = make_sandbox("e2b", template="agentic-cl-sandbox", timeout=120)
print(sb.run_code("print((23*17)-19)").stdout)  # 372
sb.kill()
```

## 5. 8 路 GRPO 采集（主用法）

```bash
# 框架冒烟（run_code fallback，不依赖模型/hermes）：
python scripts/sandbox_grpo_collect.py --actor run_code --num-queries 2 --slots 2

# 真实（沙箱内 hermes 调模型，需 AGENT_MODEL_BASE/KEY 沙箱可达）：
python scripts/sandbox_grpo_collect.py --actor hermes --num-queries 5 --slots 8
```

模型调用拓扑（两条独立路径）：
- **Actor（沙箱内 hermes）**：hermes 在沙箱里跑，调 `AGENT_MODEL_BASE`（沙箱可达的模型 endpoint）。endpoint+key 经 `runtime.env` 注入沙箱，**不经过开发机**。
- **Observer/Questioner/Reward（开发机侧）**：从 `configs/agents.yaml` 解析，开发机直连 sufy。

详见 [`Sandbox_管理调度指南.md`](Sandbox_管理调度指南.md)。

## 6. 沙箱内 runtime env

| 文件 | 作用 |
|---|---|
| `configs/sandbox_runtime_env.json` | 键清单（值空=待填） |
| `docker/sandbox/runtime.env` | 本地填值（gitignore），覆盖 JSON 同键 |

沙箱内 hermes 调模型用的三个键：
```
AGENT_MODEL_NAME=openai/gpt-5
AGENT_MODEL_BASE=https://openai.sufy.com/v1   # sufy，沙箱已验证可达
AGENT_MODEL_KEY=<sufy key>                     # 机密，只进 runtime.env
```

## 7. custom 镜像 build + push（需要 docker 的机器）

```bash
# 1. 登录 TCR
tccli tcr CreateInstanceToken --cli-unfold-argument --RegistryId tcr-hxya4oi8
docker login tcr-rl.tencentcloudcr.com -u <Username> -p <Token>

# 2. 填好 image.env 后构建推送
bash scripts/validate_sandbox_dockerfile.sh
bash scripts/build_sandbox_image.sh          # 改了 Dockerfile/插件必加 BUILD_NO_CACHE=1
bash scripts/push_sandbox_image.sh

# 3. 改 configs/sandbox_tool.json 的 RoleArn + Image 后
bash scripts/create_sandbox_via_api.sh custom
```

**★ 同 tag（如 v2）覆盖构建后，必须刷新 Tool 的 ImageDigest（2026-08-28 踩坑，耗数小时）**：
Tool 的 `CustomConfiguration.ImageDigest` 在**建 Tool 时把 `:v2` tag 解析成当时的 digest 快照并锁死**。
之后即使 `push_sandbox_image.sh` 用同一个 `:v2` tag 覆盖了 registry（digest 变了），Tool 仍拉**旧
digest** → 起的实例是旧镜像 → 新加的插件/改动不生效（现象：评测/训练里 `web_search` 等一直
undefined，但 registry 的 v2 明明是新的）。**免删更新**（不用清活跃实例）：

```bash
# a. 查 registry 里 v2 现在的 digest（新的）
tccli tcr DescribeImages --RegistryId tcr-hxya4oi8 --NamespaceName agentos-cl-namespace \
  --RepositoryName agentic-cl-sandbox --ImageVersion v2   # 取 ImageInfoList[].Digest
# b. 查 Tool 当前锁的 digest（对比，若不同就要更新）
tccli ags DescribeSandboxToolList --region ap-beijing --ToolIds '["<ToolId>"]'  # CustomConfiguration.ImageDigest
# c. UpdateSandboxTool 把 ImageDigest 换成新的（照抄现有 CustomConfiguration 只改 digest，见下 JSON）
tccli ags UpdateSandboxTool --region ap-beijing --cli-input-json file:///tmp/update_tool.json
# d. AGS 后台预拉新镜像（2GB，几分钟），期间起实例报 409 image is still preparing，正常，重试即可
# e. 验证插件/改动真进镜像（起新实例 ls）：
source scripts/env/load_tencent_env.sh && export E2B_VALIDATE_API_KEY=false
.venv/bin/python scripts/sandbox/verify_web_tools.py     # ✅ 三文件齐 = 生效
```

> ⚠️ **已起的实例不会自动换镜像**：update digest 后，只有【新起】的实例用新镜像。正在跑的
> 评测/训练要**重启**才生效（12840 个 STOPPED 历史实例无需清，不影响 update）。
> `DeleteSandboxTool` 会因"instances still active"失败，所以走 **UpdateSandboxTool**（免删）。


**Dockerfile 约束**：不能写 `USER`/`WORKDIR`/`ENV`/`ENTRYPOINT`（快照启动会失败），只能 `RUN pip install` + `COPY`。见 `scripts/validate_sandbox_dockerfile.sh`。

## 8. 常见卡点

| 现象 | 处理 |
|---|---|
| `AuthenticationException: Invalid API key format` | `E2B_VALIDATE_API_KEY=false`（`load_tencent_env.sh` 已设） |
| `/execute` 500 | 走 `commands.run`（49983），不走 /execute（49999） |
| 沙箱连不上模型 endpoint | 已切 sufy（沙箱可达）；旧 tokenhub 弃用 |
| `tccli: command not found` | 用 `~/.local/bin/tccli`（已 symlink） |
| `docker build` 失败 | 本 Pod 无 docker；在有 docker 的机器跑 `build_sandbox_image.sh` |
| E2B API 返回 2C/1G | 假值，用 envd `/metrics` 拿真实规格（见概念与术语 §G） |
| **改了镜像但改动不生效**（如 `web_search` 一直 undefined，但 registry v2 是新的）| Tool 锁了旧 `ImageDigest`；同 tag 覆盖构建后必须 `UpdateSandboxTool` 刷新 digest + 重启评测/训练（见 §7 ★）。用 `scripts/sandbox/verify_web_tools.py` 起新实例确认插件在不在 |
| `409 image is still preparing` | update digest 后 AGS 后台预拉新镜像（2GB 几分钟），正常，重试即可 |
| **`Permission denied: /home/user/.hermes/cron`**（每 session 崩） | `COPY ... /home/user/.hermes/plugins/` 让 Docker 以 root 建了 `.hermes`，hermes 以 user(uid1000) 跑写不了 → Dockerfile 该 COPY 后必须 `RUN chown -R 1000:1000 /home/user/.hermes`（2026-08-28 回归，已修）|
| **`hermes plugins list` 里没有 web-serper（误判"未加载"）** | `plugins list` **只渲染 standalone 插件**；所有 `kind:backend` 的 web provider（bundled `exa`/`tavily`/… 和我们的 `serper`）都不在该列表里却已正常加载。**别拿它当加载判据**。决定性检查用 `scripts/sandbox/verify_web_tools.py` §4（in-process 问 `PluginManager._plugins['web/serper']` + `web_search_registry._providers` + 各 tool `check_fn`）。2026-08-28 曾据此误报数轮 |


## 9. 相关文件

| 文件 | 作用 |
|---|---|
| `docker/sandbox/tencent.env` | CAM + E2B + 地域（gitignore） |
| `docker/sandbox/runtime.env` | Instance 内运行时 env（gitignore） |
| `configs/sandbox_runtime_env.json` | Instance 内 env 键模板 |
| `configs/sandbox_tool.json` | Tool 配方（VPC + 镜像 + 规格） |
| `scripts/load_tencent_env.sh` | 加载凭证 + `E2B_VALIDATE_API_KEY=false` |
| `scripts/sandbox_grpo_collect.py` | 8 路 GRPO 采集脚本 |
| `rollout/sandbox_client.py` | 沙箱适配（E2BSandbox / LocalSandbox） |

## 10. 官方链接

| 用途 | URL |
|---|---|
| Agent Runtime 控制台 | https://console.cloud.tencent.com/agent-runtime |
| API 密钥 | https://console.cloud.tencent.com/cam/capi |
| CAM 角色 | https://console.cloud.tencent.com/cam/role |
| 容器镜像 TCR | https://console.cloud.tencent.com/tcr |
| 自定义沙箱文档 | https://cloud.tencent.com/document/product/1814/129691 |
