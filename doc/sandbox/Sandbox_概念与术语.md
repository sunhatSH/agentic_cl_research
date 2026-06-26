# Sandbox 概念与术语

> 沙箱相关文档的**入口**。先读本文搞懂三组概念，再按末尾导航表去具体文档查操作。
>
> 实物示例（2026-06-25 全链路打通用的真地址）：`tcr-rl.tencentcloudcr.com/agentos-cl-sandbox/agentic-cl-sandbox:v1`

---

## A. 沙箱三层：镜像 / Tool / Instance

腾讯云 Agent Runtime 把 Docker 的 `image / container` 二元模型，拆成 **镜像 / Tool / Instance** 三层。用 Docker 类比最好记：

| 腾讯云 | 像 Docker 里的 | 干什么 | 一次还是多次 |
|---|---|---|---|
| **镜像 (Image)** | `docker image` | 环境模板（Python/依赖/种子文件装在哪） | build 一次 |
| **沙箱 Tool** | Compose 配方 / 启动配方 | 用哪张镜像 + 多大 CPU + 开哪些端口 + 探针 | **创建一次** |
| **沙箱 Instance** | `docker container` | **真正在跑**的隔离环境，agent 在里面 `run_code` | 每次 rollout 起一个 |

### 流向

```text
docker build → push 到镜像仓库（见 B）
       ↓
创建 Tool（配方：指定镜像 + CPU + 端口 + 探针）   ← 创建一次
       ↓
创建 Instance（容器：真正跑起来）                ← 每次 rollout / 每个并行环境起一个
       ↓
E2B SDK: run_code / commands.run → 拿 trajectory
```

### 实物对照（2026-06-25）

| 层 | 实物 |
|---|---|
| 镜像 | `tcr-rl.tencentcloudcr.com/agentos-cl-sandbox/agentic-cl-sandbox:v1` |
| Tool | ToolId `sdt-f4ygdu0a`（ToolName `agentic-cl-sandbox`） |
| Instance | InstanceId `a7dptvsoikpf2...`（状态 RUNNING） |

**关键**：Tool 是「配方」只建一次，Instance 是「容器」每次起一个。一次 build 1 镜像 → 建 1 Tool → 起多个 Instance。

---

## B. 镜像仓库内部结构：实例 → 命名空间 → 镜像

「镜像」存在哪个仓库、哪个命名空间下，是另一层概念，跟 A 的「镜像/Tool/Instance」是正交的两件事。

```text
容器镜像服务（console.cloud.tencent.com/tcr）
│
├── 个人版 CCR  ← 老的，host = ccr.ccs.tencentyun.com（本项目弃用，见 C）
│     └── 命名空间 / 镜像:tag
│
└── 企业版 TCR  ← 本项目用，按「实例」卖
      │
      └── 实例 (Instance)        ← 第1层：独立域名 + 独立访问凭证
            │   本项目 = tcr-rl（公网 tcr-rl.tencentcloudcr.com，ID tcr-hxya4oi8）
            │
            └── 命名空间 (Namespace)  ← 第2层：隔离用
                  │   本项目 = agentos-cl-sandbox
                  │
                  └── 镜像 (Image):tag  ← 第3层：实际 push 的东西
                        本项目 = agentic-cl-sandbox:v1
```

### 完整镜像地址 = 实例域名 / 命名空间 / 镜像名 : tag

```text
tcr-rl.tencentcloudcr.com / agentos-cl-sandbox / agentic-cl-sandbox : v1
        └─ 实例公网域名 ─┘   └─ 命名空间 ─┘   └─ 镜像名 ─┘   └tag┘
```

这四段拼起来就是 `configs/sandbox_tool.json` 里 `CustomConfiguration.Image` 字段填的值，也是造沙箱 Tool 时填的「镜像地址」。

### 概念辨析：两种 "Instance" 别混

- **B 里的「实例」** = TCR 镜像仓库实例（registry instance），是**存镜像**的地方，有独立域名。
- **A 里的「Instance」** = 沙箱实例（sandbox instance），是**跑代码**的容器。

两个都叫 Instance，但一个管镜像存储、一个管容器运行。本文为区分，B 用「TCR 实例」，A 用「沙箱 Instance」。

---

## C. TCR 企业版 vs CCR 个人版

本项目一律用企业版 TCR，个人版 CCR 弃用。

| | 个人版 CCR | 企业版 TCR（本项目用） |
|---|---|---|
| host | `ccr.ccs.tencentyun.com` | `<实例名>.tencentcloudcr.com`（如 `tcr-rl.tencentcloudcr.com`） |
| 层级 | 命名空间 → 镜像 | **实例 → 命名空间 → 镜像**（多一层实例） |
| 登录方式 | 控制台设固定密码，`docker login -u <主账号ID>` | **临时令牌**，`tccli tcr CreateInstanceToken` 拿 Username + Token |
| 凭证时效 | 固定密码长期有效 | 临时 Token 默认 **1 小时**，过期重新生成 |
| 计费 | 免费 | 按实例按月付费 |
| 本项目状态 | ❌ 弃用（login 报 `unauthorized: no scope`，账号侧未配通） | ✅ 全链路打通 |

### 为什么弃用个人版 CCR

`docker login ccr.ccs.tencentyun.com` 报 `unauthorized: no scope specify`——个人版「访问凭证/密码」账号侧没配通，且个人版 host 跟企业版完全不同。企业版 TCR 用 `tccli tcr CreateInstanceToken` 拿临时 token，一次就通。

### 企业版 docker login 命令

```bash
# 1. 拿临时令牌（Token 默认 1 小时有效）
tccli tcr CreateInstanceToken --cli-unfold-argument --RegistryId tcr-hxya4oi8
# 返回 Username + Token

# 2. login（Token 有时效，过期重新生成）
docker login tcr-rl.tencentcloudcr.com -u <Username> -p <Token>
```

也可在控制台「容器镜像服务 → 企业版 → 实例 → 访问凭证 → 生成临时登录指令」直接拿完整命令。

---

## 去哪查什么（导航）

| 想做的事 | 看哪篇 |
|---|---|
| 控制台一步步操作（建实例/命名空间/Tool/Instance） | [`Sandbox_腾讯云操作手册.md`](Sandbox_腾讯云操作手册.md) |
| 已跑通路径的精简冒烟步骤 | [`Sandbox_冒烟指南.md`](Sandbox_冒烟指南.md) |
| 动作在沙箱内 / 推理在外 + OpenClaw 架构 | [`Sandbox_Agent架构.md`](Sandbox_Agent架构.md) |
| 16×8 winner-sync 调度 | [`Sandbox_管理调度指南.md`](Sandbox_管理调度指南.md) |
| 平台 Tool/Instance API 参考 + PoC | [`SandboxRollout.md`](SandboxRollout.md) |
| Sandbox 后端 + 三 Agent 接口怎么用 | [`接口使用_Sandbox与三Agent.md`](接口使用_Sandbox与三Agent.md) |
| 冷启动采集运行手册 | [`ColdRollout_采集.md`](ColdRollout_采集.md) |
| 母版镜像 Dockerfile 制作方案 | [`沙箱_Dockerfile制作方案.md`](沙箱_Dockerfile制作方案.md) |
| 1 沙箱 ↔ N 会话 ↔ N query 关系（待定） | [`沙箱_实例_Queries对应关系_待定.md`](沙箱_实例_Queries对应关系_待定.md) |
| 沙箱操作验证与证据 | [`沙箱操作验证与证据.md`](沙箱操作验证与证据.md) |
