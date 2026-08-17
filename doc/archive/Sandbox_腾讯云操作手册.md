# 腾讯云 Agent Runtime 沙箱 — 操作手册（唯一入口）

> 把配置、命令行、控制台合成一份。**先看 §0 概念，再按 §2 文件地图填配置，按场景走 §4 或 §5。**
>
> **已跑通冒烟的精简步骤**：[`Sandbox_冒烟指南.md`](../ops/sandbox/Sandbox_冒烟指南.md)
>
> 官方文档：[创建自定义沙箱（代码解释器镜像）](https://cloud.tencent.com/document/product/1814/129691)

---

## §0 三个概念（先搞懂再操作）

很多人晕在「镜像 / Tool / Instance」——用 Docker 类比：

| 腾讯云 | 像 Docker 里的 | 干什么 |
|--------|----------------|--------|
| **CCR 镜像** | `docker image` | 环境模板（Python 包装在哪） |
| **沙箱 Tool** | 启动配方 / Compose 配置 | 用哪张镜像、多大 CPU、开哪些端口——**创建一次** |
| **沙箱 Instance** | `docker container` | **真正在跑**的隔离环境，agent 在里面 `run_code` |

关系：

```text
[可选] docker build → 推到 CCR 镜像
         ↓
控制台/API 创建 Tool（配方）  ToolName 例如 agentic-cl-code-interpreter
         ↓
API/SDK 创建 Instance（容器）  每次 rollout / 每个并行环境起一个
         ↓
E2B SDK：run_code / commands.run → 拿 trajectory
```

**两套密钥（别混）：**

| 密钥 | 填在哪 | 用途 |
|------|--------|------|
| CAM **SecretId / SecretKey** | `tencent.env` | `tccli` 调 API：创建 Tool、起 Instance |
| **E2B_API_KEY** | `tencent.env` | `e2b_code_interpreter` SDK 访问已起的 Instance |

**不要用登录密码**做自动化。

**地域（你们 VPC 在北京）：**

```bash
TENCENTCLOUD_REGION=ap-beijing
E2B_DOMAIN=ap-beijing.tencentags.com
```

两行必须成套；`ap-shanghai` / `ap-guangzhou` 等与 `ap-beijing` 不能混用。

---

## §1 本项目两条路线

| 路线 | 要不要自己 build 镜像 | 什么时候用 |
|------|----------------------|------------|
| **A. builtin（推荐起步）** | **不要** | 平台内置 code-interpreter，已能 `run_code` |
| **B. custom 自定义镜像** | **要** | 预装 pandas/aiohttp 等，减少 rollout 时装包 |

**你们当前进度（builtin 已创建）：**

| 项 | 值 |
|----|-----|
| 地域 | 北京 `ap-beijing` |
| Tool 名 | `agentic-cl-code-interpreter` |
| ToolId | 换地域后需在 **北京** 控制台重新创建 Tool（上海创建的 Tool 不可跨地域使用） |
| SDK 模板名 | `Sandbox(template="agentic-cl-code-interpreter")` |

---

## §2 仓库里所有相关文件（填哪个、何时填）

```text
agentic_cl_research/
├── docker/sandbox/
│   ├── tencent.env          ← 【主配置】API 密钥 + E2B + 地域 + 角色名
│   ├── image.env            ← 【仅 custom】CCR 命名空间、镜像 tag
│   ├── Dockerfile           ← custom 镜像：FROM sandbox-code + pip 依赖
│   ├── requirements.txt     ← 要装进镜像的 pip 包
│   └── README.md            ← 本目录速查（细节见本文）
├── configs/
│   ├── sandbox_tool_builtin.json   ← builtin Tool 的 API JSON（一般不用改）
│   └── sandbox_tool.json           ← custom Tool 的 API JSON（push 镜像后改）
├── scripts/
│   ├── load_tencent_env.sh         ← source 加载上面两个 .env
│   ├── create_sandbox_via_api.sh   ← 创建 Tool + E2B Key + 测试 Instance
│   ├── build_sandbox_image.sh      ← custom：docker build
│   ├── push_sandbox_image.sh       ← custom：docker push
│   ├── validate_sandbox_dockerfile.sh
│   ├── sandbox_smoke.py            ← 本地/真沙箱冒烟
│   └── create_sandbox_tool.sh        ← 检查清单
└── rollout/sandbox_client.py       ← Local / E2B 后端适配
```

### `docker/sandbox/tencent.env`（几乎总是要先填）

| 变量 | 填什么 | 去哪拿 |
|------|--------|--------|
| `TENCENT_UIN` | 主账号 ID | 已预填 `100048510516` |
| `TENCENTCLOUD_SECRET_ID` | CAM API 密钥 | [API 密钥管理](https://console.cloud.tencent.com/cam/capi) → 新建 |
| `TENCENTCLOUD_SECRET_KEY` | 同上 | 同上（只显示一次，立刻保存） |
| `TENCENTCLOUD_REGION` | `ap-beijing` | 与 VPC 同地域 |
| `E2B_API_KEY` | `ark_...` 一串 | 跑 `create_sandbox_via_api.sh` 时 `CreateAPIKey` 打印 |
| `E2B_DOMAIN` | `ap-beijing.tencentags.com` | 与 REGION 成套 |
| `AGS_ROLE_NAME` | CAM **角色名** | 仅 **custom** 需要；团队常用 `AgentOS-260506-test` |

### `docker/sandbox/image.env`（仅 custom 镜像）

| 变量 | 填什么 | 去哪拿 |
|------|--------|--------|
| `CCR_NAMESPACE` | 命名空间名 | [容器镜像服务](https://console.cloud.tencent.com/tcr) → 命名空间；本项目建议 `agentos-cl-namespace` |
| `IMAGE_NAME` | `agentic-cl-sandbox` | 默认即可 |
| `IMAGE_TAG` | `v1` | 自己定 |

推完后完整镜像地址示例：

`tcr-rl.tencentcloudcr.com/agentos-cl-namespace/agentic-cl-sandbox:v1`

### `configs/sandbox_tool.json`（仅 custom Tool）

push 镜像后改两处：

- `RoleArn` → `qcs::cam::uin/100048510516:roleName/AgentOS-260506-test`
- `CustomConfiguration.Image` → 上面完整镜像地址

---

## §3 控制台操作（网页端）

### 3.1 创建 API 密钥（第一次必做）

1. 打开 [API 密钥管理](https://console.cloud.tencent.com/cam/capi)
2. **新建密钥** → 复制 SecretId、SecretKey → 写入 `tencent.env`
3. **不要**把登录密码写进文件

### 3.2 容器镜像服务 TCR 企业版（仅 custom 路线）

> ⚠️ **本项目一律使用企业版 TCR（tcr-rl），禁止使用个人版 CCR（ccr.ccs.tencentyun.com）。**

1. 打开 [容器镜像服务](https://console.cloud.tencent.com/tcr) → **企业版**（实例 tcr-rl）
2. 首次 **初始化**
3. **命名空间** → 新建（如 `agentos-cl-namespace`）→ 名字写入 `image.env` 的 `CCR_NAMESPACE`
4. **访问凭证** → 企业版用**临时登录令牌**（不是固定密码）：
   - 控制台「实例 → 访问凭证 → 生成临时登录指令」复制完整命令，或用 tccli：
     ```bash
     tccli tcr CreateInstanceToken --cli-unfold-argument --RegistryId tcr-hxya4oi8
     # 返回 Username + Token（Token 默认 1 小时有效，过期重新生成）
     ```
   - 本机执行（Token 有时效，过期重生成）：
     ```bash
     docker login tcr-rl.tencentcloudcr.com -u <Username> -p <Token>
     ```

### 3.3 CAM 角色（custom 路线；builtin 可跳过）

1. [访问管理 → 角色](https://console.cloud.tencent.com/cam/role) → **新建角色**
2. 载体：**腾讯云产品服务** → **Agent Runtime**
3. 策略：授予 **CCR 读权限**（企业版 CCR 或企业版 TCR）
4. 角色名例如 `AgentOS-260506-test` → 写入 `AGS_ROLE_NAME`
5. 给用户账号 **PassRole** 权限（见官方文档 §三）

### 3.4 Agent Runtime — 创建沙箱 Tool

1. 打开 [Agent Runtime](https://console.cloud.tencent.com/agent-runtime)
2. **左上角地域选「北京」**（与 `tencent.env` 一致）
3. **沙箱工具** → **新建**

**builtin（内置代码解释器）：**

- 类型：**code-interpreter**（不是 custom）
- 名称：`agentic-cl-code-interpreter`（与 SDK `template` 一致）
- 网络：**PUBLIC**（任务要装包/调外网）

**custom（自定义镜像）：**

- 类型：**custom**
- 镜像：填 CCR 完整地址
- CAM 角色：选 §3.3 的角色
- 端口：`49999`（run-code）、`49983`（envd）
- 探针：`http://49999/health`
- 规格：至少 2 CPU、2Gi；启动命令默认 `/init` + `sleep infinity`（Dockerfile 只加依赖时可不改）

### 3.5 创建 E2B API Key（也可命令行创建）

控制台 Agent Runtime 里 **API 密钥**（若有入口），或直接用脚本 `CreateAPIKey`（§4.2）。

### 3.6 创建沙箱实例

Tool 建好后：**沙箱实例** → 新建 → 选 Tool → 设超时。

或用 SDK / `StartSandboxInstance` API（§4.2 脚本会自动起一个测试实例）。

---

## §4 命令行操作（复制即用）

### 4.0 每次开工

```bash
cd /path/to/agentic_cl_research
source scripts/load_tencent_env.sh
```

应看到 `API credentials present (region=ap-beijing)`。

### 4.1 安装依赖

```bash
pip install tccli e2b e2b_code_interpreter
# 或项目 venv：.venv/bin/pip install e2b_code_interpreter
```

### 4.2 路线 A：builtin（不 build 镜像）— 你们已跑通过一次

```bash
bash scripts/create_sandbox_via_api.sh builtin
```

脚本会做三件事：

1. `CreateSandboxTool` → 得到 ToolId
2. `CreateAPIKey` → 打印 `E2B_API_KEY`（抄进 `tencent.env`）
3. `StartSandboxInstance` → 起一个 RUNNING 测试实例

### 4.3 验证沙箱能跑代码

```bash
source scripts/load_tencent_env.sh
python3 scripts/sandbox_smoke.py --backend e2b
```

需要本机能访问 `ap-beijing.tencentags.com`（见 §6 网络）。

### 4.4 路线 B：custom 自定义镜像

```bash
# 1. 登录 CCR
docker login tcr-rl.tencentcloudcr.com

# 2. 拉官方基底（需登录成功）
docker pull tcr-rl.tencentcloudcr.com/agentos-cl-namespace/sandbox-code:latest

# 3. 填好 image.env 后构建推送
bash scripts/validate_sandbox_dockerfile.sh
bash scripts/build_sandbox_image.sh
bash scripts/push_sandbox_image.sh

# 4. 改 configs/sandbox_tool.json 的 RoleArn + Image 后
bash scripts/create_sandbox_via_api.sh custom
```

### 4.5 常用 tccli 查询（排错用）

```bash
source scripts/load_tencent_env.sh

# 列出北京地域所有 Tool
tccli ags DescribeSandboxToolList --region ap-beijing

# 列实例
tccli ags DescribeSandboxInstanceList --region ap-beijing
```

### 4.6 本机无云沙箱时（开发用）

```bash
python3 scripts/sandbox_smoke.py --backend local
```

不走腾讯云，子进程假沙盒，逻辑与采样链路一致。

---

## §5 和训练代码怎么接

```text
沙盒 Instance 里 run_code
    → trajectory（messages / reward / logprobs）
    → trainer/trajectory_adapter 入 7 桶 buffer
    → verl GRPO + CL loss
```

- 领域标签：agent 回复末尾 `<task_domain>Finance</task_domain>`（见 `trainer/domain_tagging.py`）
- 设计细节：`doc/ops/sandbox/SandboxRollout.md`

SDK 最小示例：

```python
import os
from e2b_code_interpreter import Sandbox

os.environ["E2B_API_KEY"] = "..."      # 或 export 后自动读
os.environ["E2B_DOMAIN"] = "ap-beijing.tencentags.com"

sbx = Sandbox(template="agentic-cl-code-interpreter")  # = ToolName
out = sbx.run_code("print((23*17)-19)")
print(out.logs)
sbx.kill()
```

---

## §6 网络（IDC + 北京 VPC）

| 从哪访问 | 需要什么 |
|----------|----------|
| 调 `tccli` 创建 Tool | 能访问 `*.tencentcloudapi.com`（你们可以） |
| E2B SDK 连 Instance | 能访问 `ap-beijing.tencentags.com` |
| 训练机在 IDC、VPC 在北京 | 问网络同事：**VPC 打通 + 内网域名解析**；公网不通就走内网 endpoint |

本机 Pod 曾测：`tencentags` 公网不通 → SDK 可能失败，但 **Tool 创建可以成功**（API 走另一条路）。

---

## §7 填完没填完 — 自检表

### builtin 路线（当前）

| 项 | 状态 |
|----|------|
| `tencent.env` SecretId/Key | 需已填 |
| `tencent.env` REGION + E2B_DOMAIN | 北京成套 |
| `tencent.env` E2B_API_KEY | 需已填 |
| Tool `agentic-cl-code-interpreter` | 已创建 |
| `python3 scripts/sandbox_smoke.py --backend e2b` | 待网络通后验证 |

### custom 路线（以后）

| 项 | 状态 |
|----|------|
| `image.env` CCR_NAMESPACE | 如 `agentos-cl-namespace` |
| docker build + push | 需有 Docker daemon 的机器 |
| `AGS_ROLE_NAME` + `sandbox_tool.json` | 角色 + 镜像地址 |
| `create_sandbox_via_api.sh custom` | 最后一步 |

---

## §8 常见问题

**Q：E2B_API_KEY 和 SecretKey 是同一个吗？**  
不是。SecretKey 给 tccli；E2B_API_KEY 给 SDK，用 `CreateAPIKey` 生成。

**Q：ToolName 和 ToolId 用哪个？**  
SDK `Sandbox(template=...)` 用 **ToolName**（字符串名）。

**Q：AGS_ROLE_NAME 填什么？**  
CAM 角色列表里的 **角色名称**（不是 ARN 整串）。builtin 可留空；custom 用团队 `AgentOS-260506-test` 或自建角色。

**Q：命名空间填 clawgym 还是 agentos-cl-namespace？**  
CL 项目建议 **`agentos-cl-namespace`**（`image.env`）；团队 RL 共用选 `agentos-rl-test260506`。

**Q：Dockerfile 能写 WORKDIR / USER 吗？**  
**不能**（快照启动会失败）。只能 `RUN pip install`。见 `scripts/validate_sandbox_dockerfile.sh`。

**Q：密钥泄露怎么办？**  
控制台禁用旧 Key，新建一对，更新 `tencent.env`。

---

## §9 官方链接速查

| 用途 | URL |
|------|-----|
| Agent Runtime 控制台 | https://console.cloud.tencent.com/agent-runtime |
| API 密钥 | https://console.cloud.tencent.com/cam/capi |
| CAM 角色 | https://console.cloud.tencent.com/cam/role |
| 容器镜像 CCR | https://console.cloud.tencent.com/tcr |
| 自定义沙箱文档 | https://cloud.tencent.com/document/product/1814/129691 |

---

## §10 其他文档

| 文件 | 内容 |
|------|------|
| `doc/ops/sandbox/Sandbox_冒烟指南.md` | 已跑通路径、ops 一键命令 |
| `doc/ops/sandbox/Sandbox_Agent架构.md` | 动作在沙箱内/推理在外、OpenClaw、随机用户文件系统、镜像依赖 |
| `doc/ops/sandbox/Sandbox_管理调度指南.md` | 16×8 调度、会话内 winner 同步、母版策略 |
| `doc/ops/sandbox/SandboxRollout.md` | 平台 API、Tool/Instance、PoC |
| `docker/sandbox/README.md` + `ops/` | 密钥、镜像 build、快捷脚本 |

**账号/镜像/custom 细节以本文 + 冒烟指南为准；调度与 winner 同步见 `Sandbox_管理调度指南.md`。**
