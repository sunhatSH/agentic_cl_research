# 有腾讯云账号后 — 30 分钟上手清单

> **首选入口**：[`Sandbox_腾讯云操作手册.md`](Sandbox_腾讯云操作手册.md)。本文为简短勾选清单。

按顺序做，做完一项勾一项。custom 镜像细节见 [`Sandbox_Image_Onboarding.md`](Sandbox_Image_Onboarding.md)。

> **安全**：请用 **API 密钥（SecretId/SecretKey）** 操作，**不要把登录密码**写进仓库、脚本或聊天。密钥只放在本机 `docker/sandbox/tencent.env`（已 gitignore），或临时 `export`。

---

## 最快创建沙箱（有 API 密钥后一条命令）

```bash
# 1. 控制台创建 API 密钥 https://console.cloud.tencent.com/cam/capi
cp docker/sandbox/tencent.env.example docker/sandbox/tencent.env
# 只填 TENCENTCLOUD_SECRET_ID / TENCENTCLOUD_SECRET_KEY

set -a && source docker/sandbox/tencent.env && set +a
bash scripts/create_sandbox_via_api.sh builtin   # 内置 code-interpreter，无需 docker 镜像
```

脚本会：创建 Tool → 创建 E2B API Key → 起一个测试 Instance。

---

## 你需要先抄下来的 4 个值

做完控制台步骤后，填到 `docker/sandbox/image.env` 和 `configs/sandbox_tool.json`：

| 变量 | 去哪拿 | 填到哪 |
|------|--------|--------|
| **CCR 命名空间** | 容器镜像服务 → 命名空间 | `image.env` 的 `CCR_NAMESPACE` |
| **镜像完整地址** | build+push 后 | `sandbox_tool.json` 的 `CustomConfiguration.Image` |
| **CAM RoleArn** | 访问管理 → 角色 → 复制 ARN | `sandbox_tool.json` 的 `RoleArn` |
| **E2B_DOMAIN** | 沙盒管理员 / 控制台地域 | 环境变量，如 `ap-beijing.tencentags.com` |
| **E2B_API_KEY** | Agent Runtime 控制台 API 密钥 | 环境变量 |

---

## Step 1：容器镜像服务 CCR（≈5 分钟）

1. 控制台打开 **[容器镜像服务](https://console.cloud.tencent.com/tcr)** → 选**个人版**（或企业版）。
2. **初始化**个人版（首次使用）。
3. 创建**命名空间**（例如 `agentic-cl`）→ 记下名字 → `CCR_NAMESPACE`。
4. 右上角 **访问凭证** → 设置/查看密码 → 用于 `docker login`。

```bash
docker login ccr.ccs.tencentyun.com
# 用户名：腾讯云账号 ID
# 密码：访问凭证里设置的密码
```

---

## Step 2：CAM 角色（≈10 分钟，Agent Runtime 文档 §二 §三）

1. **[访问管理 → 角色](https://console.cloud.tencent.com/cam/role)** → 新建角色。
2. 角色载体：**腾讯云产品服务** → 选 **Agent Runtime**。
3. 策略：授予 **CCR 读权限**（个人版）或 TCR（企业版）。文档示例是较宽策略，生产可收紧。
4. 角色名例如 `ags-ccr-agentic-cl` → 完成 → **复制 RoleArn**。
5. **PassRole**：新建自定义策略，允许 `cam:PassRole` 到该 RoleArn → 绑到你的子账号/用户。

RoleArn 形如：`qcs::cam::uin/1234567890:roleName/ags-ccr-agentic-cl`

---

## Step 3：构建并推送镜像（需 **有 Docker daemon 的机器**）

> 当前 Cursor 开发 Pod **不能** docker build（无 daemon）。请在本机笔记本、64 卡训练机、或任意 Linux 上执行。

```bash
cd agentic_cl_research

# 拉官方基底（需 Step 1 登录成功）
docker pull ccr.ccs.tencentyun.com/ags-image/sandbox-code:latest

cp docker/sandbox/image.env.example docker/sandbox/image.env
# 编辑 CCR_NAMESPACE=你的命名空间

bash scripts/validate_sandbox_dockerfile.sh
bash scripts/build_sandbox_image.sh
bash scripts/push_sandbox_image.sh
```

记下输出里的完整 tag，例如：`ccr.ccs.tencentyun.com/agentic-cl/agentic-cl-sandbox:v1`

---

## Step 4：创建沙箱 Tool（控制台 ≈5 分钟）

1. 打开 **[Agent Runtime 控制台](https://console.cloud.tencent.com/agent-runtime)** → **沙箱工具** → **新建**。
2. 对照 `configs/sandbox_tool.json` 填写：
   - 工具名称：`agentic-cl-sandbox`（与 JSON 里 `ToolName` 一致，SDK 用这个名）
   - 类型：**自定义 custom**
   - CAM 角色：Step 2 的角色
   - 镜像地址：Step 3 push 的完整 URL
   - 启动命令：`/init` + 参数 `sleep` / `infinity`（若 Dockerfile 只加依赖，也可留默认）
   - 端口：`49999`（run-code）、`49983`（envd）
   - 探针：`http://49999/health`
   - 规格：2 CPU，2Gi 内存
   - 网络：**PUBLIC**（任务要装包/调 API）
3. 创建成功 → 记下 **ToolId**（`sdt-xxx`）。

也可用 API：把 JSON 里 `REPLACE_*` 改完后 `agr tool create --request @configs/sandbox_tool.json`（若已装 agr CLI）。

---

## Step 5：创建实例 & 验证（需网络通 tencentags）

```bash
pip install e2b-code-interpreter

export E2B_API_KEY=<控制台密钥>
export E2B_DOMAIN=ap-<地域>.tencentags.com   # 问管理员或看控制台

python scripts/sandbox_smoke.py --backend e2b
```

若 Step 5 超时：训练机在 IDC、无公网或 DNS 黑洞 → 走 VPC 打通或公网放行（见 `doc/Migration_64GPU.md` §7）。

本机网络不通时，可先在**能访问公网的机器**上跑 Step 5 验证 Tool/镜像是否正确。

---

## 最短路径（不想自定义镜像）

若暂时不想 build 镜像，可先用平台内置代码解释器：

- 创建 Tool 时选 **ToolType = code-interpreter**（不是 custom）
- 无需 `CustomConfiguration`、无需自己 push 镜像
- SDK：`Sandbox(template="code-interpreter-v1")` 或控制台给出的默认名

自定义镜像的价值是预装 `requirements.txt` 里的 pandas/aiohttp 等，减少 rollout 时装包时间。

---

## 当前环境自检

```bash
bash scripts/create_sandbox_tool.sh
bash scripts/validate_sandbox_dockerfile.sh
```
