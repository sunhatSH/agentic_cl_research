# 沙盒平台 API 参考（腾讯云 Agent Runtime）

> **定位**：本文仅保留平台 Tool/Instance API 对照与 PoC 清单，作为平台 API 的单一信源。调度逻辑（16×8 + winner-sync）见 [`Sandbox_管理调度指南.md`](../ops/sandbox/Sandbox_管理调度指南.md)；架构（动作内/推理外）见 [`Sandbox_Agent架构.md`](../ops/sandbox/Sandbox_Agent架构.md)；运维见 [`Sandbox_腾讯云操作手册.md`](Sandbox_腾讯云操作手册.md)。

---

## 1. 沙盒 API（与 Docker 对照）

### 1.1 抽象层次

AGS 把 Docker 的 image/container 二元拆成 **Tool / Instance** 两层：

| Docker | AGS |
|---|---|
| Dockerfile / image | **Tool**（模板：镜像、网络、超时、挂载） |
| `docker run` | **Instance**（从 Tool 起的运行体） |

### 1.2 操作对照

| Docker | AGS CLI | AGS SDK（E2B 兼容） |
|---|---|---|
| `docker run` | `agr instance create --tool-id` | `Sandbox.create(template=...)` |
| `docker exec` | `agr instance code run` | `sandbox.run_code(...)` / `sandbox.commands.run(...)` |
| `docker stop`（保留状态） | `agr instance pause` | （pause 仅 CLI/REST 暴露） |
| `docker start`（恢复） | `agr instance resume` | （同上） |
| `docker rm` | `agr instance delete` | `sandbox.kill()` |
| `docker attach` | — | `Sandbox.connect(sandbox_id)` |
| `docker commit` | — | — |
| Tool→Tool 克隆 | `agr tool fork` | — |

### 1.3 关键约束

- **Tool 是不可变的运行模板**。`agr tool fork` 仅 Tool→Tool 复制，**不直接支持"把 Instance 当前状态固化成新 Tool"**——cookbook CLI 层没有 commit 等价物，这是方案的关键 PoC 项（§2）。
- **网络模式**（`SANDBOX` / `PUBLIC` / `VPC`）在 Tool 创建时定，update 无法切换，要换网络必须 fork 新 Tool。CL 训练 query 大概率需要 `PUBLIC`（装包、调外部 API）。
- **SDK vs CLI 分工**：训练框架（Python）日常调用走 SDK；构建 Tool 模板、跨 batch 维护、固化操作走 CLI / RESTful。

---

## 2. PoC 待验证

接入 Phase 1 前必须用真账号实测（参考 ags-cookbook 的 SDK / CLI）：

| # | 验证项 | 决策含义 |
|---|---|---|
| 1 | **`freeze_instance_into_tool` 是否可行**：cookbook CLI 没有 Instance→Tool commit 命令，需核实底层 RESTful API 是否提供 | **阻塞项**。可行→走路径 A；不可行→走路径 B（见下） |
| 2 | 并发 `Sandbox.create` 起 M 个 Instance 的端到端 latency | 必须与 verl GPU 前向同量级，否则 GPU 空转 |
| 3 | 单账号 Instance 并发上限与配额 | 决定 B × M 能开多大 |
| 4 | `pause` 写出快照的大小与延迟 | 决定每 query 维护环境链的存储成本是否可持续 |
| 5 | 网络模式选择（CL query 多大比例需要 `PUBLIC`） | 决定 Tool 的默认 `NetworkMode` |

**实现路径二选一**（验证项 1 的结果决定）：

- **路径 A（首选，依赖 commit 能力）**：每个 query 维护一个 **Tool**，每轮 rollout 固化优胜 Instance 为新 Tool。
- **路径 B（兜底，仅靠 pause/resume）**：每个 query 维护一个 **长寿命 master Instance**（始终 pause 着）；rollout 时 resume master 得到工作实例 → 派生 M-1 个并行实例（API 待确认）→ 选 winner → 旧 master delete、winner pause 成新 master。

---

## 参考

- 腾讯云 Agent Runtime 文档：<https://cloud.tencent.com/document/product/1814/129423>
- AGS Cookbook：<https://github.com/TencentCloudAgentRuntime/ags-cookbook>
- E2B 协议：<https://e2b.dev/>
- 项目内：`doc/source/CL_Design.md`、`doc/source/CL_Design.md`、`doc/source/训练与推理流程.md`
