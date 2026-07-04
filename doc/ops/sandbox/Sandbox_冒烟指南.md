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
bash scripts/build_sandbox_image.sh
bash scripts/push_sandbox_image.sh

# 3. 改 configs/sandbox_tool.json 的 RoleArn + Image 后
bash scripts/create_sandbox_via_api.sh custom
```

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
