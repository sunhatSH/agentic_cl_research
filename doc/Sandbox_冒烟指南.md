# 沙箱冒烟 — 精简指南

> 2026-06-10 在北京 `ap-beijing` 跑通。细节见 [`Sandbox_腾讯云操作手册.md`](Sandbox_腾讯云操作手册.md)。

---

## 1. 概念对照（像 Docker 那样记）

| Docker 里 | 腾讯云沙箱 | 你怎么操作 |
|-----------|------------|------------|
| `docker image` | **CCR 镜像** | `docker build` + `docker push`（仅 custom 路线） |
| Compose / 启动配方 | **沙箱 Tool** | 控制台新建，或 `tccli ags CreateSandboxTool` |
| `docker run` 起的容器 | **沙箱 Instance** | 调 API / SDK 自动起；控制台可看列表 |
| 仓库登录 | **CCR 凭证** | `docker login tcr-rl.tencentcloudcr.com` |
| 云 API 密钥 | **CAM SecretId/Key** | 控制台创建 → `tencent.env` |
| 进容器执行命令 | **run_code / execute** | `rollout/sandbox_client` 或冒烟脚本 |

**两套密钥别混：**

| 密钥 | 用途 | 填在哪 |
|------|------|--------|
| CAM SecretId / SecretKey | `tccli` 建 Tool、查实例 | `tencent.env` |
| E2B_API_KEY | 起沙箱、执行代码 | `tencent.env` |

地域必须成套（本项目 **北京**）：

```bash
TENCENTCLOUD_REGION=ap-beijing
E2B_DOMAIN=ap-beijing.tencentags.com
```

---

## 2. 官网上要做什么

### 2.1 第一次（账号级，做一次）

| 步骤 | 去哪 | 做什么 |
|------|------|--------|
| ① API 密钥 | [CAM → API 密钥](https://console.cloud.tencent.com/cam/capi) | 新建 SecretId/SecretKey → 写入 `docker/sandbox/tencent.env` |
| ② 选地域 | 各控制台左上角 | 一律选 **北京** |
| ③ 开通 Agent Runtime | [Agent Runtime](https://console.cloud.tencent.com/agent-runtime) | 按引导开通（若未开通） |

**不要**把登录密码写进仓库；只用 API 密钥。

### 2.2 跑 builtin 冒烟（当前已跑通路线）

| 步骤 | 控制台 or 命令行 | 说明 |
|------|------------------|------|
| 建 Tool | **二选一** | 见下 §3.2 脚本，或控制台：沙箱工具 → 新建 → 类型 **code-interpreter** → 名 `agentic-cl-code-interpreter` → 网络 **PUBLIC** |
| 建 E2B Key | **二选一** | 脚本 `CreateAPIKey` 打印，或控制台 API 密钥入口 |
| 执行代码 | **命令行** | `sandbox_smoke.py` 或 `make_sandbox('e2b')`（不必在网页点实例） |

控制台里可 **核对**：沙箱工具列表有 `agentic-cl-code-interpreter`；沙箱实例里能看到 RUNNING（脚本会自动起一个测试实例）。

### 2.3 以后要 custom 镜像（预装 pip 依赖）

| 步骤 | 去哪 | 做什么 |
|------|------|--------|
| 命名空间 | [容器镜像 TCR/CCR](https://console.cloud.tencent.com/tcr) | 企业版 → 建命名空间（如 `agentos-cl-sandbox`）→ `image.env` |
| 访问凭证 | 同上 → 访问凭证 | 设密码 → 给有 Docker 的机器 `docker login` |
| CAM 角色 | [CAM → 角色](https://console.cloud.tencent.com/cam/role) | 载体 Agent Runtime + CCR 读权限 → `AGS_ROLE_NAME` |
| 建 custom Tool | Agent Runtime → 沙箱工具 | 类型 custom → 填 CCR 镜像地址 + 角色 + 端口 49999/49983 |

---

## 3. 找谁做什么

| 事项 | 找谁 | 你要说清楚什么 |
|------|------|----------------|
| **VPC / 网络打通** | 网络 / 基础设施同事 | 训练机或开发机能访问 `ap-beijing.tencentags.com` 和 `api.ap-beijing.tencentags.com`；IDC 需 **北京 VPC 打通 + 域名解析**（公网不通则内网 endpoint） |
| **Agent Runtime 配额 / 并发** | 云平台 / 沙箱管理员 | 单账号 Instance 并发上限、是否允许 PUBLIC 网络 |
| **CCR 命名空间 / 团队镜像仓** | 账号管理员或 @孙豪 | 用独立命名空间 `agentos-cl-sandbox` 还是团队共用仓 |
| **CAM 角色 + PassRole** | 有 CAM 权限的同事 | custom 镜像需角色名如 `AgentOS-260506-test` |
| **docker build / push** | 自己有 Docker 的机器 | 当前 Cursor Pod **无 Docker daemon**；用本机、GPU 机或 CI |
| **E2B 域名 / 地域** | 沙箱管理员 | 确认 `E2B_DOMAIN=ap-beijing.tencentags.com` 与控制台地域一致 |
| **Agent 多轮 loop / 工具 harness** | @郑乃榕（工具环境） | 本指南只到「沙箱里跑代码」；完整 trajectory 还要 agent 框架 |
| **评测 / ClawEval** | @杨益博 | 与沙箱冒烟无关，训练后有 ckpt 再评 |

---

## 4. 命令行（类 Docker + 云 API）

### 4.0 每次开工

```bash
cd /path/to/agentic_cl_research
source scripts/load_tencent_env.sh    # 应看到 region=ap-beijing
```

### 4.1 安装 CLI 依赖（一次）

```bash
pip3 install tccli httpx   # tccli=调腾讯云 API；httpx=E2BSandbox 用
# pip 的 e2b_code_interpreter 可选；腾讯云上执行代码走 rollout 适配层，见 §5
```

### 4.2 builtin 冒烟（≈ `docker run` 预制镜像）

```bash
# 建 Tool + E2B Key + 测试实例（打印 E2B_API_KEY → 抄进 tencent.env）
bash scripts/create_sandbox_via_api.sh builtin

# 端到端冒烟
python3 scripts/sandbox_smoke.py --backend e2b -m 2

# 只执行一行（最小）
python3 -c "
from rollout.sandbox_client import make_sandbox
sb = make_sandbox('e2b', template='agentic-cl-code-interpreter')
print(sb.run_code('print((23*17)-19)').stdout)  # 期望 372
sb.kill()
"
```

无云、测逻辑：

```bash
python3 scripts/sandbox_smoke.py --backend local
```

### 4.3 查状态（≈ `docker ps` / `docker images`）

```bash
# 一键查询（推荐，脚本在 docker/sandbox/ops/）
bash docker/sandbox/ops/ops.sh query
bash docker/sandbox/ops/query.sh --tool agentic-cl-code-interpreter
bash docker/sandbox/ops/query.sh --json

# 原始 tccli
source scripts/load_tencent_env.sh
tccli ags DescribeSandboxToolList --region ap-beijing
tccli ags DescribeSandboxInstanceList --region ap-beijing
```

### 4.4 custom 镜像（≈ docker build / push）

**必须在有 Docker daemon 的机器上：**

```bash
source scripts/load_tencent_env.sh

# 登录 CCR（≈ docker login registry）
docker login tcr-rl.tencentcloudcr.com   # 用户名一般是主账号 ID

# 拉官方基底（≈ docker pull 基础镜像）
docker pull tcr-rl.tencentcloudcr.com/ags-image/sandbox-code:latest

# 构建 + 推送（脚本读 image.env）
bash scripts/validate_sandbox_dockerfile.sh
bash scripts/build_sandbox_image.sh
bash scripts/push_sandbox_image.sh

# 改 configs/sandbox_tool.json 后注册 custom Tool
bash scripts/create_sandbox_via_api.sh custom
```

| Docker 习惯 | 本项目命令 |
|-------------|------------|
| `docker login` | `docker login tcr-rl.tencentcloudcr.com` |
| `docker pull` | `docker pull tcr-rl.tencentcloudcr.com/ags-image/sandbox-code:latest` |
| `docker build -t ...` | `bash scripts/build_sandbox_image.sh` |
| `docker push` | `bash scripts/push_sandbox_image.sh` |
| 定义服务/配方 | `tccli ags CreateSandboxTool` 或 `create_sandbox_via_api.sh` |
| `docker run` | `make_sandbox('e2b')` / API `StartSandboxInstance` |
| 进容器执行 | `sb.run_code('...')` |

---

## 5. 我们是怎么跑通的（技术要点）

```text
tencent.env（CAM + E2B + 北京地域）
    → load_tencent_env.sh
    → rollout/sandbox_client.E2BSandbox
        POST api.ap-beijing.tencentags.com/sandboxes     （Header: X-API-KEY）
        → sandboxID + envdAccessToken
        POST 49999-<sandboxID>.ap-beijing.tencentags.com/execute
             （Header: X-Access-Token = envdAccessToken）
    → sandbox_smoke：reward / GRPO / domain
```

**不要**裸用 `e2b_code_interpreter.Sandbox().run_code()`：在腾讯云上会 **401**（host/鉴权与 AGS 不一致）。训练 rollout 统一走 `rollout/sandbox_client.py`。

---

## 6. 成功标志 & 常见卡点

**成功：** 冒烟输出 `obs='372'`、`expected=372`；或一行代码打印 `372`。

| 现象 | 处理 |
|------|------|
| `Sandbox tool ... not found` | 地域错或 Tool 未建 → 控制台选北京，或 `create_sandbox_via_api.sh builtin` |
| `/execute` 401 | 用 `make_sandbox('e2b')`，别裸调 pip SDK |
| 连不上 `tencentags.com` | 找网络同事（§3） |
| 换地域后失败 | 新地域重新建 Tool + Key，更新 `tencent.env` |
| `docker build` 失败 | 换有 Docker 的机器；本 Pod 无 daemon |

---

## 7. 沙箱 Instance 内环境变量（起实例时注入）

与 `tencent.env`（本机调 API）分开：**Instance 里** agent 跑代码时读的环境变量。

| 文件 | 作用 |
|------|------|
| `configs/sandbox_runtime_env.json` | **键清单**（当前值全空，待填） |
| `docker/sandbox/runtime.env.example` | 同上，shell 格式模板 |
| `docker/sandbox/runtime.env` | 本地填写（gitignore）；覆盖 JSON 里同键 |

起沙箱时 `E2BSandbox` 会合并上述文件，**只把非空键**放进 CreateSandbox 的 `envVars`。

```bash
cp docker/sandbox/runtime.env.example docker/sandbox/runtime.env
# 编辑 AGENTIC_CL_WORKSPACE、API 密钥等
bash scripts/print_sandbox_runtime_env.sh   # 预览将注入的 JSON
```

## 8. 相关文件

| 文件 | 作用 |
|------|------|
| `docker/sandbox/tencent.env` | CAM + E2B + 地域 |
| `docker/sandbox/image.env` | CCR 命名空间（custom） |
| `docker/sandbox/runtime.env` | Instance 内运行时 env |
| `configs/sandbox_runtime_env.json` | Instance 内 env 键模板 |
| `scripts/load_tencent_env.sh` | 加载环境变量 |
| `scripts/create_sandbox_via_api.sh` | 建 Tool + Key + 测试实例 |
| `scripts/build_sandbox_image.sh` / `push_sandbox_image.sh` | 类 docker build/push |
| `scripts/sandbox_smoke.py` | 冒烟 |
| `rollout/sandbox_client.py` | 沙箱适配（训练也用） |
