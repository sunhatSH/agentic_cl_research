# 沙盒自定义镜像 — 本地准备 & 账号就绪后操作清单

> **首选入口**：[`Sandbox_腾讯云操作手册.md`](Sandbox_腾讯云操作手册.md)（配置 + 控制台 + 命令行合一）。本文仅展开 **custom 镜像**细节。
>
> 对照腾讯云官方文档：[创建自定义沙箱（基于代码解释器沙箱镜像）](https://cloud.tencent.com/document/product/1814/129691)
>
> **现状（2026-06-10）**：账号可开通；仓库内 Dockerfile/脚本/Tool JSON 已就绪。当前 Cursor 开发 Pod：**无 Docker daemon**（仅 client）、**CCR 443 可达**、**tencentags 公网不可达** → build/push 需在**带 Docker 的机器**上执行；跑沙盒实例仍需 **E2B 凭证 + 训练机到 tencentags 的网络**。

---

## 一、仓库里已准备好的文件

| 路径 | 作用 |
|------|------|
| `docker/sandbox/Dockerfile` | 基于 `sandbox-code:latest` 只加 pip 依赖（快照兼容） |
| `docker/sandbox/requirements.txt` | 本项目 rollout 常用依赖 |
| `docker/sandbox/image.env.example` | CCR 命名空间 / 镜像名 / tag 模板 |
| `scripts/validate_sandbox_dockerfile.sh` | **无需 Docker**，校验 Dockerfile 是否违反快照约束 |
| `scripts/build_sandbox_image.sh` | `docker build --platform=linux/amd64` |
| `scripts/push_sandbox_image.sh` | `docker push` 到 CCR |
| `scripts/create_sandbox_tool.sh` | 创建沙箱 Tool 前的检查清单 |
| `configs/sandbox_tool.json` | Agent Runtime **创建沙箱工具** API/控制台 JSON 模板 |

---

## 二、现在就能做（无账号 / 无 Docker）

```bash
# 校验 Dockerfile 符合快照约束（禁止 USER/WORKDIR/ENV/ENTRYPOINT）
bash scripts/validate_sandbox_dockerfile.sh

# 看工具 JSON 模板里还要填什么
bash scripts/create_sandbox_tool.sh
```

---

## 三、有腾讯云账号后 — 按顺序执行

### Step 0：前置

1. 安装 **Docker**（本机或一台有网络的构建机）。
2. 开通 **容器镜像服务 CCR**（个人版或企业版），记下命名空间。
3. 完成 Agent Runtime 文档中的 **CAM 角色 + PassRole**（见官方文档 §二、§三）。

### Step 1：拉基础镜像 & 登录 CCR

```bash
# 登录 CCR（个人版示例）
docker login ccr.ccs.tencentyun.com

# 拉官方代码解释器基底（必须，自定义镜像 FROM 它）
docker pull ccr.ccs.tencentyun.com/ags-image/sandbox-code:latest
```

### Step 2：配置并构建自定义镜像

```bash
cp docker/sandbox/image.env.example docker/sandbox/image.env
# 编辑：CCR_NAMESPACE、IMAGE_TAG 等

bash scripts/build_sandbox_image.sh
bash scripts/push_sandbox_image.sh
```

构建产物标签形如：`ccr.ccs.tencentyun.com/<namespace>/agentic-cl-sandbox:v1`

### Step 3：填写并创建沙箱 Tool

1. 编辑 `configs/sandbox_tool.json`：
   - `RoleArn` → CAM 角色 ARN
   - `CustomConfiguration.Image` → 上一步推送的完整镜像地址
   - `NetworkConfiguration.NetworkMode` → 训练环境需要外网则 `PUBLIC`
2. 控制台：**Agent Runtime → 沙箱工具 → 新建**，或 API / `agr tool create`。
3. 记下返回的 **ToolId**；E2B SDK 里 `Sandbox(template="agentic-cl-sandbox")` 用的是 **ToolName**（与 JSON 里 `ToolName` 一致）。

`configs/sandbox_tool.json` 已按官方文档配置：

- 端口 **49999**（run_code + `/health` 探针）、**49983**（envd）
- 资源 **2 CPU / 2Gi**（数据分析类任务可改 4Gi，见官方 §资源规格）
- **未写 Command/Args**（Dockerfile 只加依赖 → 冷启动自动用镜像内置 `/init sleep infinity`）

### Step 4：创建实例 & 跑 smoke

```bash
export E2B_API_KEY=<控制台获取>
export E2B_DOMAIN=ap-<region>.tencentags.com   # 地域由管理员告知

pip install e2b-code-interpreter   # 集群 .venv 里装

python scripts/sandbox_smoke.py --backend e2b
```

创建实例 API 示例（官方 §六）：

```json
{ "ToolId": "sdt-xxxx", "Timeout": "30m" }
```

---

## 四、Dockerfile 设计约束（必须遵守）

来自 [官方文档 §快照启动约束](https://cloud.tencent.com/document/product/1814/129691)：

| 约束 | 本项目做法 |
|------|------------|
| 不修改 `USER` | Dockerfile 无 `USER`（保持 root） |
| 不修改 `WORKDIR` | Dockerfile 无 `WORKDIR`（保持 `/`） |
| 环境变量 | 不在 Dockerfile 写 `ENV`；需要时在 Tool API 的 `Env` 里配（自定义程序加 `S6_KEEP_ENV=1`） |
| 不覆盖 `ENTRYPOINT` | 不写 `ENTRYPOINT`；保持基底 `/init` |
| 平台 | 构建时 `--platform=linux/amd64`（脚本已带） |
| 只加依赖 | `RUN pip install -r requirements.txt` |

**错误示例**（会导致快照失败或 run_code 不可用）：

```dockerfile
WORKDIR /app
USER appuser
ENV MY_VAR=hello
ENTRYPOINT ["python", "app.py"]
```

---

## 五、与训练链路的对应关系

```
自定义镜像 (docker/sandbox)
    → 推送 CCR
    → 创建 Tool (configs/sandbox_tool.json)
    → CreateSandboxInstance → InstanceId
    → E2B SDK: Sandbox(template="agentic-cl-sandbox")
    → rollout_group: M 路并行 run_code
    → trajectory → 7 桶 buffer → GRPO + CL loss
```

本机无沙盒时继续用：`python scripts/sandbox_smoke.py --backend local`（子进程假沙盒，逻辑一致）。

---

## 六、常见问题

**Q：没有账号能不能先 build？**  
不能完整 build——`FROM sandbox-code:latest` 需要 CCR 登录后才能 pull。现在只能 `validate_sandbox_dockerfile.sh` + 改 `requirements.txt`。

**Q：ToolName 和 E2B template 是什么关系？**  
SDK 里 `Sandbox(template="agentic-cl-sandbox")` 对应创建 Tool 时的 **ToolName**（不是 ToolId）。

**Q：还要不要找沙盒管理员？**  
账号就绪后仍需：**E2B_API_KEY**、可达的 **E2B_DOMAIN**、以及 IDC→云的网络（见 `doc/Migration_64GPU.md` §7）。镜像和 Tool JSON 你可以自己推/建，网络仍可能要平台同事配合。

---

## 七、变更日志

| 日期 | 说明 |
|------|------|
| 2026-06-10 | 初版：`docker/sandbox/*` + 构建脚本 + `sandbox_tool.json` + 本文 |
