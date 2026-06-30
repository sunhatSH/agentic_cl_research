# 沙箱冒烟 — 精简指南（新集群权威版）

> 2026-06-30 在新集群北京区 `ap-beijing` 跑通。本文是当前唯一权威 quick-start，旧版细节作废。
> 详细概念见 [`Sandbox_概念与术语.md`](Sandbox_概念与术语.md)，接口见 [`接口使用_Sandbox与三Agent.md`](接口使用_Sandbox与三Agent.md)。

---

## 1. 现状一句话

- **镜像**：`tcr-rl.tencentcloudcr.com/agentos-cl-namespace/agentic-cl-sandbox:v2`（v2，装了 OpenClaw + **Hermes Agent** + openai/fastapi/pandas 等）
- **Tool**：`agentic-cl-sandbox`（ToolId `sdt-jrpn0rfo`，4C/8Gi/20Gi，VPC `subnet-ljfuh7ln`/`sg-irz5h5ed`，envd:49983，`/init`）
- **VPC**：`agent-rl-vpc`（`vpc-llc6ty0w`）。沙箱有公网出站、能连开发机，但**连不上商汤内网 tokenhub**（`172.30.9.145`）——需网络侧打通或给沙箱可达的 endpoint
- **代码执行**：走 `commands.run`（envd 49983），**不走** `/execute`（49999 Jupyter，500）

## 2. 两套密钥（别混）

| 密钥 | 前缀 | 用途 | 在哪 |
|---|---|---|---|
| CAM SecretId/Key | `AKID...` | `tccli` 控制 Tool/Instance | `docker/sandbox/tencent.env` + 已写入 `~/.tccli/default.credential` |
| E2B_API_KEY | `ark_...`（AGS 发的） | e2b SDK 起实例/跑代码 | `docker/sandbox/tencent.env` |

**关键坑**：e2b SDK 默认要求 `e2b_` 前缀，AGS 发的是 `ark_` → `AuthenticationException`。**解决**：`E2B_VALIDATE_API_KEY=false`（SDK 官方开关，已固化在 `load_tencent_env.sh`），**非 monkeypatch**。

## 3. 每次开工

```bash
cd /mnt/afs_toolcall/sunhao4/agentic_cl_research
source scripts/load_tencent_env.sh   # 导出 CAM + E2B + E2B_VALIDATE_API_KEY=false
# 需要下包时加: export http_proxy=http://sysagent:c08400bf@10.119.176.202:3128
# tccli 已 symlink 到 ~/.local/bin；e2b/tccli 在 cold env (py3.10)
```

## 4. 查状态（≈ docker ps / docker images）

```bash
bash docker/sandbox/ops/ops.sh query                 # 一键查 Tool + Instance
tccli ags DescribeSandboxToolList     --region ap-beijing
tccli ags DescribeSandboxInstanceList --region ap-beijing
```

## 5. 执行代码（最小）

```python
from rollout.sandbox_client import make_sandbox
sb = make_sandbox("e2b", template="agentic-cl-sandbox", timeout=120)
print(sb.run_code("print((23*17)-19)").stdout)  # 372
sb.kill()
```

## 6. 8 路 GRPO 采集（主用法）

```bash
# 框架冒烟（run_code fallback，不依赖模型/hermes）：
python scripts/sandbox_grpo_collect.py --actor run_code --num-queries 2 --slots 2

# 真实（沙箱内 hermes 调模型，需 AGENT_MODEL_BASE/KEY 沙箱可达）：
python scripts/sandbox_grpo_collect.py --actor hermes --num-queries 5 --slots 8
```

模型调用拓扑（**两条独立路径**）：
- **Actor（沙箱内 hermes）**：hermes 在沙箱里跑，调 `AGENT_MODEL_BASE`（沙箱可达的模型 endpoint）。endpoint+key 经 `runtime.env` 注入沙箱，**不经过开发机**。
- **Observer/Questioner/Reward（开发机侧）**：从 `configs/agents.yaml` 解析（gpt-4.1-mini / claude-sonnet-4-6 轮换 / claude-opus-4-8-thinking），开发机直连 tokenhub。

详见 [`Sandbox_管理调度指南.md`](Sandbox_管理调度指南.md)。

## 7. 沙箱内 runtime env（起实例注入）

| 文件 | 作用 |
|---|---|
| `configs/sandbox_runtime_env.json` | 键清单（值空=待填） |
| `docker/sandbox/runtime.env` | 本地填值（gitignore），覆盖 JSON 同键 |

沙箱内 hermes 调模型用的三个键：
```
AGENT_MODEL_NAME=gpt-5.1
AGENT_MODEL_BASE=https://tokenhub.sensetime.com/v1   # 必须沙箱可达
AGENT_MODEL_KEY=<tokenhub key>                        # 机密，只进 runtime.env
```
起实例时 `E2BSandbox` 合并上述文件，只把非空键放进 `envVars`。

## 8. 常见卡点

| 现象 | 处理 |
|---|---|
| `AuthenticationException: Invalid API key format` | `E2B_VALIDATE_API_KEY=false`（`load_tencent_env.sh` 已设） |
| `/execute` 500 | 走 `commands.run`（49983），不走 /execute（49999） |
| 沙箱连不上 tokenhub | 沙箱 VPC 无商汤内网路由 → 找网络同事打通 VPC peering，或给沙箱可达的 endpoint |
| `tccli: command not found` | 用 `~/.local/bin/tccli`（已 symlink）；或 `source load_tencent_env.sh` |
| `docker build` 失败 | 本 Pod 无 docker；在有 docker 的机器跑 `build_sandbox_image.sh` |

## 9. 相关文件

| 文件 | 作用 |
|---|---|
| `docker/sandbox/tencent.env` | CAM + E2B + 地域（gitignore） |
| `docker/sandbox/runtime.env` | Instance 内运行时 env（gitignore） |
| `configs/sandbox_runtime_env.json` | Instance 内 env 键模板 |
| `configs/sandbox_tool.json` | Tool 配方（VPC + 镜像 + 规格） |
| `scripts/load_tencent_env.sh` | 加载凭证 + `E2B_VALIDATE_API_KEY=false` |
| `scripts/sandbox_grpo_collect.py` | 8 路 GRPO 采集脚本（hermes / run_code 两 actor） |
| `scripts/sandbox_smoke.py` | 旧版单步冒烟（被 §6 替代） |
| `rollout/sandbox_client.py` | 沙箱适配（E2BSandbox / LocalSandbox） |
