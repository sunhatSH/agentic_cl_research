# 腾讯云 AGS 沙箱实例规格 + run_code 路径踩坑

> 2026-06-26 排查记录。背景：Tool 配置 4C/8Gi/20Gi，但 E2B API 返回 2C/1G，一度以为规格没生效。
> 结论：**规格生效了，E2B API 字段是假值**；`/metrics` 拿真实规格；run_code 路径不是 `/execute`。

---

## 一、规格问题：E2B API 返回假值

### 现象
- Tool `sdt-81jenxfq` 配置 `CustomConfiguration.Resources = {CPU:"4", Memory:"8Gi", Storage:"20Gi"}`
- E2B 兼容 API `GET /sandboxes` 返回 `cpuCount=2 / memoryMB=1024 / diskSizeMB=1024`
- 以为规格没生效

### 原因
`rollout/sandbox_client.py` 用的 **E2B 兼容 API**（`api.ap-beijing.tencentags.com`）返回的 `cpuCount/memoryMB/diskSizeMB` 是**固定占位值（永远 2/1024/1024）**，不反映实例真实分配。这是 E2B 兼容层的 bug/限制——它只返回 E2B 默认规格，没同步腾讯侧真实资源。

### 三方对比（实例 `qgmr3yskxfb7pnkjidt35uixzujy223m2np2gv2m`）

| 来源 | CPU | 内存 | 磁盘 | 可靠度 |
|---|---|---|---|---|
| E2B API `GET /sandboxes` | 2 | 1024 MB | 1024 MB | ❌ 假值 |
| Tool `CustomConfiguration.Resources` | 4 | 8 Gi | 20 Gi | 声明值 |
| **envd `/metrics`（真实）** | 5 | 8.0 GiB | 20.7 GiB | ✅ 铁证 |

- envd `/metrics` 返回 `mem_total=8438513664`（8.0GiB）、`disk_total=22202957824`（20.7GiB），与 Tool 配置一致 → **规格确实生效**。
- CPU 显示 5 是宿主逻辑核可见数（cgroup 限额是 CPU 配额，非核数可见性），正常。
- 腾讯工作人员后台看 Tool 是 4C8G（看的是 `CustomConfiguration.Resources`），跟 `/metrics` 一致。

---

## 二、怎么获取真实规格

### 用的 SDK
**腾讯官方 SDK** `tencentcloud-sdk-python-ags`（不是 E2B 兼容 API）：
```bash
pip install tencentcloud-sdk-python-ags
```
模块路径：`tencentcloud.ags.v20250920`，API endpoint：`ags.tencentcloudapi.com`

### 执行代码（起实例 + 拿 Token + 读 /metrics）

```python
import os
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.ags.v20250920 import ags_client, models
import httpx

# 1. 初始化官方 SDK 客户端（凭证从 docker/sandbox/tencent.env 读）
cred = credential.Credential(
    os.environ["TENCENTCLOUD_SECRET_ID"],
    os.environ["TENCENTCLOUD_SECRET_KEY"],
)
hp = HttpProfile()
hp.endpoint = "ags.tencentcloudapi.com"
cp = ClientProfile()
cp.httpProfile = hp
client = ags_client.AgsClient(cred, "ap-beijing", cp)

# 2. 起实例（用 ToolId 或 ToolName）
req = models.StartSandboxInstanceRequest()
req.ToolId = "sdt-81jenxfq"        # 或 req.ToolName = "agentic-cl-sandbox"
req.Timeout = "15m"
resp = client.StartSandboxInstance(req)
inst_id = resp.Instance.InstanceId
print(f"InstanceId: {inst_id}, Status: {resp.Instance.Status}")

# 3. 拿访问 Token（关键：用官方 SDK 的 AcquireSandboxInstanceToken）
tok_req = models.AcquireSandboxInstanceTokenRequest()
tok_req.InstanceId = inst_id
tok_resp = client.AcquireSandboxInstanceToken(tok_req)
token = tok_resp.Token              # 管控面 Token，访问 49983(envd) 端口用
# traffic_token = tok_resp.TrafficToken  # 数据面 Token，访问其他端口用（/metrics 用不到）

# 4. 读真实规格（envd /metrics 接口）
metrics_url = f"https://49983-{inst_id}.ap-beijing.tencentags.com/metrics"
r = httpx.get(metrics_url, headers={"X-Access-Token": token}, timeout=15)
print(r.status_code, r.json())
# 返回: {"cpu_count":5, "mem_total":8438513664, "mem_used":..., "disk_total":22202957824, ...}
```

### 访问路径（关键）

| 项 | 值 |
|---|---|
| 端口 | **49983**（envd 端口，**不是 49999**） |
| 域名 | `49983-<InstanceId>.ap-beijing.tencentags.com` |
| 路径 | `/metrics`（读规格）/ `/files`（读文件，要 `?username=root&path=<文件路径>`） |
| Header | `X-Access-Token: <Token>`（官方 SDK `AcquireSandboxInstanceToken` 的 `Token`） |
| Token 类型 | **管控面 `Token`**（不是 `TrafficToken`，不是 E2B API 的 `envdAccessToken`） |

### curl 版

```bash
# 拿 Token（用 tccli 或 SDK）
TOKEN=$(python3 -c "..." )  # AcquireSandboxInstanceToken 返回的 Token

# 读真实规格
curl -H "X-Access-Token: $TOKEN" \
  https://49983-<InstanceId>.ap-beijing.tencentags.com/metrics

# 读文件（username=root，path 要是文件不是目录，/proc/* 读不了）
curl -H "X-Access-Token: $TOKEN" \
  "https://49983-<InstanceId>.ap-beijing.tencentags.com/files?username=root&path=/etc/os-release"
```

---

## 三、E2B 兼容 API vs 官方 SDK 对比

| 项 | E2B 兼容 API（旧，有问题） | 官方 SDK（正确） |
|---|---|---|
| endpoint | `api.ap-beijing.tencentags.com` | `ags.tencentcloudapi.com` |
| 起实例 | `POST /sandboxes` + `templateID`(ToolName) | `StartSandboxInstance` + `ToolId` |
| 拿 Token | 起实例返回的 `envdAccessToken` | `AcquireSandboxInstanceToken` 的 `Token` |
| 查规格 | `GET /sandboxes` 的 `cpuCount/memoryMB`（**假值**） | envd `/metrics`（真实） |
| run_code | `POST 49999-{sid}/execute`（**404**） | 待定（49983 端口，路径探测中） |
| 访问端口 | 49999（错） | 49983（envd，对） |

### E2B 兼容 API 获取规格（假值，别用）

```python
# ❌ 这个返回的 cpuCount/memoryMB 是假值（永远 2/1024/1024）
import httpx, os
r = httpx.get(
    "https://api.ap-beijing.tencentags.com/sandboxes",
    headers={"X-API-KEY": os.environ["E2B_API_KEY"]},
    timeout=30,
)
for s in r.json():
    print(s["sandboxID"], s["cpuCount"], s["memoryMB"], s["diskSizeMB"])  # 假值
```

---

## 四、run_code 路径：`/execute` 不通，用 `commands.run`

### `/execute` 是什么 & 为什么不通

`/execute` 是 **E2B code-interpreter 的 Jupyter 内核接口**（`sb.run_code()` 用的），走 Jupyter 内核协议，端口 49999，返回流式 JSON Lines（stdout/stderr/result 富输出）。

**腾讯 AGS base 镜像 `sandbox-code` 没装 Jupyter 内核服务**，所以：
- `POST 49999-{sid}/execute` 永远 `500 Internal Server Error (openresty)`——网关层就挂，后端无 Jupyter 服务
- 换 Token、加 `E2B-Traffic-Access-Token` / `E2b-Sandbox-Port` header 都救不回来
- 连 E2B **官方包** `sb.run_code()` 也 500（实测）——证实是镜像缺 Jupyter，不是代码姿势问题

### 实测验证（实例 `onq5ijuevskkekwmpizu76ae` 等）

| 调用方式 | 端口/路径 | 结果 |
|---|---|---|
| `sb.run_code('print(2+2)')`（E2B 官方包） | 49999/execute | ❌ 500 |
| 手动 `POST 49999-{sid}/execute` + 两 Token | 49999/execute | ❌ 500 |
| 手动 `POST 49983-{sid}/execute` | 49983/execute | ❌ 404 |
| **`sb.commands.run('echo hello')`**（E2B 官方包） | 49983 process gRPC | ✅ **通** |

### 正确姿势：`commands.run`（shell）

`commands.run` 走 envd 的 **process gRPC 接口**（49983 端口），本质开 shell 跑命令。镜像有 `/bin/sh` 即可（base 自带），不依赖 Jupyter。

```python
from e2b_code_interpreter import Sandbox

sb = Sandbox.create(template='agentic-cl-sandbox', timeout=60)
# ✅ 用 commands.run（shell），不要用 run_code（Jupyter，没装）
out = sb.commands.run('python3 -c "print(2+2)"', timeout=30)
print(out.stdout)   # 4
print(out.exit_code)
sb.kill()
```

**附带验证真实规格**（`commands.run` 跑 `nproc`/`free -m`）：
```
nproc  → 5            # 5 核（宿主逻辑核可见，cgroup 限额 4）
free -m → Mem: 8047   # 8 GiB（与 Tool 配置 8Gi 一致，规格生效）
```

### 已验证可用的 envd 接口（49983 端口 + `Token`）

| 路径 | 方法 | Token | 用途 |
|---|---|---|---|
| `/health` | GET | Token | 健康检查（204） |
| `/metrics` | GET | Token | 真实规格 + 资源用量 |
| `/files?username=root&path=<file>` | GET | Token | 读文件（path 要是文件，/proc/* 500 seeker can't seek） |
| process gRPC | - | Token | `commands.run` 走这里（shell 执行） |

### 两套执行方式对比

| | `/execute`（run_code） | `commands.run` |
|---|---|---|
| 协议 | Jupyter 内核 | shell process |
| 端口 | 49999 | 49983 |
| 镜像要求 | 必须装 Jupyter | 有 shell 即可 |
| 腾讯 AGS | ❌ 500（没 Jupyter） | ✅ 通 |
| 输出 | 富（图表/表格/多语言） | 纯文本 stdout/stderr/exit_code |
| 项目需要 | 不需要（跑 agent 动作） | 够用 |

### 解决方案选择

**方案 A（推荐）**：`run_code` 改用 `commands.run` 实现——镜像不用动，agent 跑 shell 动作够用。
**方案 B**：镜像里装 Jupyter（`pip install jupyter ipykernel`），让 `/execute` 通——镜像变大几百 MB，但能拿富输出。本项目不需要富输出，不推荐。

> 若仍要装 Jupyter（方案 B），在 `docker/sandbox/requirements.txt` 加 `jupyter` + `ipykernel`，重 build 镜像即可。`/execute` 走 49999 端口 + `X-Access-Token`(Token) + `E2B-Traffic-Access-Token`(TrafficToken)。

---

## 五、相关文件

| 文件 | 作用 |
|---|---|
| `rollout/sandbox_client.py` | `E2BSandbox` 类（L137-223），待按官方 SDK 重写 |
| `configs/sandbox_tool.json` | Tool 配置（VPC + 4C/8Gi/20Gi） |
| `scripts/sandbox_list_instances.py` | 列实例脚本（E2B API，规格字段假值，注意） |
| `docker/sandbox/tencent.env` | 凭证（SecretId/Key + E2B_API_KEY） |

---

**日期**：2026-06-26
**实例**：`qgmr3yskxfb7pnkjidt35uixzujy223m2np2gv2m`（Tool `sdt-81jenxfq`，VPC 模式）
**结论**：规格生效（/metrics 证实 8Gi/20Gi），E2B API 规格字段假值，run_code 路径待找
