# 文档索引（doc/）

> 本目录 21 篇文档按用途分三级。接手项目从「② 运行手册」的 Migration_64GPU.md 开始。

---

## ① 信源（论文与代码的上游依据，长期维护）

| 文档 | 内容 |
|------|------|
| [CL_Update_Sunhao.md](CL_Update_Sunhao.md) | **主文档**：CL Loss / Replay Buffer / 实验路线 / 评测 / 文献 |
| [BucketDesign.md](BucketDesign.md) | 7 桶结构论证（开头含速查节） |
| [VerlIntegration.md](VerlIntegration.md) | verl 0.8.0 集成指导 |
| [UserSim_多轮Query在线生成.md](UserSim_多轮Query在线生成.md) | 三 agent 多轮构造设计规格 |
| [UserSim_三Agent架构与技术设计.md](UserSim_三Agent架构与技术设计.md) | observer/questioner/reward 接口契约 |
| [UserSim_人设库.md](UserSim_人设库.md) | 42 个人设表 + 设计轴 |
| [ClawEval_Metadata.md](ClawEval_Metadata.md) | 评测基准数据 |

## ② 运行手册（操作向，按需查阅）

| 文档 | 内容 |
|------|------|
| [Migration_64GPU.md](Migration_64GPU.md) | 跨机器交接 + 冷启动步骤 |
| [Plan_训练链路补齐.md](Plan_训练链路补齐.md) | 64 卡前 Gap A–H 施工规格 |
| [Sandbox_Agent架构.md](Sandbox_Agent架构.md) | 动作内/推理外 + OpenClaw |
| [Sandbox_管理调度指南.md](Sandbox_管理调度指南.md) | 16×8 winner-sync 调度 |
| [SandboxRollout.md](SandboxRollout.md) | 平台 Tool/Instance API 参考 |
| [Sandbox_腾讯云操作手册.md](Sandbox_腾讯云操作手册.md) | 腾讯云控制台操作步骤 |
| [Sandbox_冒烟指南.md](Sandbox_冒烟指南.md) | 沙箱冒烟精简步骤 |
| [ColdRollout_采集.md](ColdRollout_采集.md) | 冷启动采集运行手册 |
| [Buffer_冷启动数据需求.md](Buffer_冷启动数据需求.md) | replay buffer 冷启动预热数据规格 |

## ③ 过程记录（append-only / 一次性，不主动维护）

| 文档 | 内容 |
|------|------|
| [Progress.md](Progress.md) | 交付状态单一来源 |
| [RunLog.md](RunLog.md) | 运行记录（禁删改历史） |
| [汇报_技术总报告.md](汇报_技术总报告.md) | 一次性汇报稿 |
| [RolloutCollect_技术报告.md](RolloutCollect_技术报告.md) | 采集系统设计/验证报告 |
| [BugLog_集群采集.md](BugLog_集群采集.md) | 集群 bug 库（append-only） |
| [集群推理采集_经验复盘.md](集群推理采集_经验复盘.md) | 踩坑复盘（不进论文） |

> 论文产出在 [`paper/`](../paper/) 目录（drafts 中英 Intro/Method + 总览、latex、refs）。
