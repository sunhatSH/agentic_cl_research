# 文档索引（doc/）

> 接手项目：先读 `ops/Migration_64GPU.md` → `archive/Progress.md` → `source/CL_Design.md`。
> 沙箱文档入口：`ops/sandbox/Sandbox_概念与术语.md`。
> 归档文件（冗余/过期/已合并）在 `archive/`，不再维护。

---

## source/ — 信源（论文与代码的上游依据，长期维护）

| 文档 | 内容 |
|------|------|
| [CL_Design.md](source/CL_Design.md) | **主文档**：CL Loss / Replay Buffer（7桶+quota+priority+冷启动数据需求）/ 实验路线（含 Phase 0）/ 评测 / GPU / 精度 / 文献 |
| [usersim.md](source/usersim.md) | **UserSim 单一信源**：模型选型 + 三 agent 架构 + 多轮 query 在线生成 + 42 人设表 |
| [训练与推理流程.md](source/训练与推理流程.md) | 训练循环 + 推理全链路 + verl 0.8.0 集成 + 数据 pipeline |
| [ClawEval_Metadata.md](source/ClawEval_Metadata.md) | 评测基准数据 |
| [Agent轨迹_Schema.md](source/Agent轨迹_Schema.md) | Agent 轨迹 schema（现状 mock vs 目标 buffer/训练） |
| [Hermes_Subagent_训练数据方案.md](source/Hermes_Subagent_训练数据方案.md) | Hermes 同步出入栈 + OpenClaw 主子各自训练方案 |

## ops/ — 运行手册（操作向，按需查阅）

| 文档 | 内容 |
|------|------|
| [Migration_64GPU.md](ops/Migration_64GPU.md) | 跨机器交接 + 冷启动步骤 + 集群提交快速参考（附录） |
| [sandbox/Sandbox_概念与术语.md](ops/sandbox/Sandbox_概念与术语.md) | **沙箱入口**：镜像/Tool/Instance + TCR/CCR + API 对照 + 代码执行 + 真实规格 |
| [sandbox/Sandbox_冒烟指南.md](ops/sandbox/Sandbox_冒烟指南.md) | **沙箱操作唯一入口**：冒烟步骤 + 命令速查 + custom 镜像 build + 常见卡点 |
| [sandbox/Sandbox_Agent架构.md](ops/sandbox/Sandbox_Agent架构.md) | 动作内/推理外 + OpenClaw |
| [sandbox/Sandbox_管理调度指南.md](ops/sandbox/Sandbox_管理调度指南.md) | 16×8 winner-sync 调度 |
| [sandbox/接口使用_Sandbox与三Agent.md](ops/sandbox/接口使用_Sandbox与三Agent.md) | **接口怎么用速查**：Sandbox 后端 + 三 Agent 报告/声明 + diff-driven 设计 |
| [sandbox/ColdRollout_采集.md](ops/sandbox/ColdRollout_采集.md) | 冷启动采集运行手册 |
| [sandbox/沙箱_Dockerfile制作方案.md](ops/sandbox/沙箱_Dockerfile制作方案.md) | 沙箱镜像 Dockerfile 制作方案 |

## eval/ — 评测

| 文档 | 内容 |
|------|------|
| [防遗忘评测方案.md](eval/防遗忘评测方案.md) | 防遗忘评测方案：按桶分组训练 + 统一评测 + 权重后置 |

## archive/ — 归档（冗余/过期/已合并，不再维护）

含过程记录（Progress.md / RunLog.md / BugLog_集群采集.md）、一次性技术报告副本、已合并的启动指南/操作手册/踩坑记录/VerlIntegration/Plan_冷启动/BucketDesign/Buffer_冷启动/UserSim 四件套等，详见 `archive/` 目录。

> 论文产出在 [`../paper/`](../paper/) 目录（drafts 中英 Intro/Method + 总览、latex、refs）。
