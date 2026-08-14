# Sandbox 概念与术语

> 沙箱文档**入口**。先读本文搞懂概念，再按末尾导航表去具体文档查操作。

---

## A. 沙箱三层：镜像 / Tool / Instance

腾讯云 Agent Runtime 把 Docker 的 `image / container` 二元模型，拆成 **镜像 / Tool / Instance** 三层：

| 腾讯云 | 像 Docker 里的 | 干什么 | 一次还是多次 |
|---|---|---|---|
| **镜像 (Image)** | `docker image` | 环境模板（Python/依赖/种子文件装在哪） | build 一次 |
| **沙箱 Tool** | Compose 配方 | 用哪张镜像 + 多大 CPU + 开哪些端口 + 探针 | **创建一次** |
| **沙箱 Instance** | `docker container` | **真正在跑**的隔离环境，agent 在里面 `run_code` | 每次 rollout 起一个 |

```text
docker build → push 到镜像仓库（见 B）
       ↓
创建 Tool（配方：指定镜像 + CPU + 端口 + 探针）   ← 创建一次
       ↓
创建 Instance（容器：真正跑起来）                ← 每次 rollout 起一个
       ↓
E2B SDK: commands.run → 拿 trajectory
```

**实物对照**（2026-06-25）：镜像 `tcr-rl.tencentcloudcr.com/agentos-cl-namespace/agentic-cl-sandbox:v1`，Tool `sdt-f4ygdu0a`，Instance `a7dptvsoikpf2...`（RUNNING）。

## B. 镜像仓库：实例 → 命名空间 → 镜像

```text
容器镜像服务（console.cloud.tencent.com/tcr）
├── 个人版 CCR  ← host = ccr.ccs.tencentyun.com（本项目弃用）
└── 企业版 TCR  ← 本项目用，按「实例」卖
      └── 实例 tcr-rl（公网 tcr-rl.tencentcloudcr.com，ID tcr-hxya4oi8）
            └── 命名空间 agentos-cl-namespace
                  └── 镜像 agentic-cl-sandbox:v1
```

完整镜像地址 = `tcr-rl.tencentcloudcr.com / agentos-cl-namespace / agentic-cl-sandbox : v1`

**两种 "Instance" 别混**：B 里的「实例」= TCR 镜像仓库实例（存镜像）；A 里的「Instance」= 沙箱实例（跑代码）。

## C. TCR 企业版 vs CCR 个人版

本项目一律用企业版 TCR，个人版 CCR 弃用。

| | 个人版 CCR | 企业版 TCR（本项目用） |
|---|---|---|
| host | `ccr.ccs.tencentyun.com` | `<实例名>.tencentcloudcr.com` |
| 层级 | 命名空间 → 镜像 | **实例 → 命名空间 → 镜像** |
| 登录 | 固定密码 | **临时令牌** `tccli tcr CreateInstanceToken`（1h 有效） |

```bash
# 企业版 docker login
tccli tcr CreateInstanceToken --cli-unfold-argument --RegistryId tcr-hxya4oi8
# 返回 Username + Token
docker login tcr-rl.tencentcloudcr.com -u <Username> -p <Token>
```

## D. 两套密钥（别混）

| 密钥 | 前缀 | 用途 | 在哪 |
|---|---|---|---|
| CAM SecretId/Key | `AKID...` | `tccli` 控制 Tool/Instance | `docker/sandbox/tencent.env` + `~/.tccli/default.credential` |
| E2B_API_KEY | `ark_...` | e2b SDK 起实例/跑代码 | `docker/sandbox/tencent.env` |

**关键坑**：e2b SDK 默认要求 `e2b_` 前缀，AGS 发的是 `ark_` → `AuthenticationException`。解决：`E2B_VALIDATE_API_KEY=false`（SDK 官方开关，已固化在 `load_tencent_env.sh`）。

## E. API 对照（Docker ↔ AGS）

| Docker | AGS CLI | AGS SDK（E2B 兼容） |
|---|---|---|
| `docker run` | `agr instance create --tool-id` | `Sandbox.create(template=...)` |
| `docker exec` | `agr instance code run` | `sandbox.commands.run(...)` |
| `docker stop` | `agr instance pause` | （仅 CLI/REST） |
| `docker start` | `agr instance resume` | （同上） |
| `docker rm` | `agr instance delete` | `sandbox.kill()` |

**关键约束**：
- Tool 是不可变运行模板。`agr tool fork` 仅 Tool→Tool 复制，**不直接支持 Instance→Tool commit**（cookbook CLI 层没有 commit 等价物）。
- 网络模式（`SANDBOX`/`PUBLIC`/`VPC`）在 Tool 创建时定，update 无法切换。
- SDK vs CLI 分工：训练框架（Python）走 SDK；构建 Tool 模板、跨 batch 维护走 CLI/REST。

## F. 代码执行：`commands.run`（不是 `/execute`）

- `/execute`（49999 端口）是 E2B Jupyter 内核接口，腾讯 AGS base 镜像没装 Jupyter → **永远 500**。
- `commands.run`（49983 端口 envd process gRPC）走 shell，有 `/bin/sh` 即可 → **正确姿势**。

```python
from e2b_code_interpreter import Sandbox
sb = Sandbox.create(template='agentic-cl-sandbox', timeout=60)
out = sb.commands.run('python3 -c "print(2+2)"', timeout=30)
print(out.stdout)  # 4
sb.kill()
```

## G. 真实规格获取（E2B API 返回假值）

E2B 兼容 API `GET /sandboxes` 的 `cpuCount/memoryMB/diskSizeMB` 是固定占位值（永远 2/1024/1024），不反映真实分配。用 envd `/metrics` 拿真实值：

```bash
# 起实例后，用官方 SDK 拿 Token（AcquireSandboxInstanceToken）
curl -H "X-Access-Token: <Token>" \
  https://49983-<InstanceId>.ap-beijing.tencentags.com/metrics
# 返回 {cpu_count, mem_total, mem_used, disk_total, disk_used, ...}
```

端口 49983（envd），Token 用 `AcquireSandboxInstanceToken` 的 `Token`（管控面），不是 `TrafficToken`。

## 去哪查什么（导航）

| 想做的事 | 看哪篇 |
|---|---|
| 已跑通路径的精简冒烟步骤 | [`Sandbox_冒烟指南.md`](Sandbox_冒烟指南.md) |
| 动作在沙箱内 / 推理在外 + OpenClaw 架构 | [`Sandbox_Agent架构.md`](Sandbox_Agent架构.md) |
| 16×8 winner-sync 调度 | [`Sandbox_管理调度指南.md`](Sandbox_管理调度指南.md) |
| Sandbox 后端 + 三 Agent 接口怎么用 | [`接口使用_Sandbox与三Agent.md`](接口使用_Sandbox与三Agent.md) |
| 冷启动采集运行手册 | [`ColdRollout_采集.md`](ColdRollout_采集.md) |
| 母版镜像 Dockerfile 制作方案 | [`沙箱_Dockerfile制作方案.md`](沙箱_Dockerfile制作方案.md) |
