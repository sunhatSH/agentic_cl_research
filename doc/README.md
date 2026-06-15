# 文档索引（doc/）

> 本目录 23 篇文档按主题分组。接手项目从「① 接手必读」开始。
> 完整文档关系表见仓库根 [`CLAUDE.md`](../CLAUDE.md)。

## ① 接手必读（状态与交接）

| 文档 | 内容 |
|------|------|
| [Migration_64GPU.md](Migration_64GPU.md) | 跨机器/跨 session 交接：冷启动步骤、当前阻塞 |
| [Progress.md](Progress.md) | 交付状态单一来源：里程碑、模块完成度、21 实验、阻塞 |
| [RunLog.md](RunLog.md) | append-only 运行记录（smoke/训练/评测/bug，禁删改历史） |
| [Plan_训练链路补齐.md](Plan_训练链路补齐.md) | 64 卡正式训练前残缺模块施工规格（Gap A–H） |
| [汇报_技术总报告.md](汇报_技术总报告.md) | 汇报向收敛：设计思路 + 进度 + 取舍 + 局限 |

## ② 核心设计（CL 方法）

| 文档 | 内容 |
|------|------|
| [CL_Update_Sunhao.md](CL_Update_Sunhao.md) | **主文档**：Loss 公式、Replay Buffer、实验路线、评测、文献 |
| [ContinualLearning.md](ContinualLearning.md) | CL Loop 全流程概述 + 多人协作分工 |
| [BucketDesign.md](BucketDesign.md) | 7 桶结构详细论证（为什么这样分桶、quota 推导） |
| [BucketDesign_compressed.md](BucketDesign_compressed.md) | 上篇精简版（仅结论与公式） |
| [VerlIntegration.md](VerlIntegration.md) | verl 集成指导：是否 fork、Buffer 接入、工程结构、风险 |

## ③ 数据采集与多轮 UserSim

| 文档 | 内容 |
|------|------|
| [UserSim_多轮Query在线生成.md](UserSim_多轮Query在线生成.md) | 多轮 query 在线生成（三 agent + 人设耐心机制） |
| [UserSim_三Agent架构与技术设计.md](UserSim_三Agent架构与技术设计.md) | observer/questioner/reward 三 agent 架构与接口契约 |
| [UserSim_人设库.md](UserSim_人设库.md) | 42 个 Questioner 人设表 + 设计轴 |
| [Buffer_冷启动数据需求.md](Buffer_冷启动数据需求.md) | replay buffer 冷启动预热的数据规格 |
| [ColdRollout_采集.md](ColdRollout_采集.md) | 冷启动多轮 rollout 采集运行手册 |
| [RolloutCollect_技术报告.md](RolloutCollect_技术报告.md) | 冷启动采集系统设计/实现/验证 + rollout 数据结构 |
| [BugLog_集群采集.md](BugLog_集群采集.md) | 集群采集 bug 库（append-only） |
| [集群推理采集_经验复盘.md](集群推理采集_经验复盘.md) | 商汤 SenseCore 8×8 H800 上 27B 采集踩坑复盘（不进论文） |

## ④ 沙箱与 Rollout 环境

| 文档 | 内容 |
|------|------|
| [Sandbox_Agent架构.md](Sandbox_Agent架构.md) | OpenClaw agent harness 架构（动作内/推理外） |
| [Sandbox_管理调度指南.md](Sandbox_管理调度指南.md) | 沙箱管理/调度（16×8 winner-sync） |
| [SandboxRollout.md](SandboxRollout.md) | 腾讯 Agent Runtime 采集方案（§4 平台 API 为现行价值） |
| [Sandbox_冒烟指南.md](Sandbox_冒烟指南.md) | 沙箱冒烟手册 |
| [Sandbox_腾讯云操作手册.md](Sandbox_腾讯云操作手册.md) | 腾讯云具体操作步骤 |

## ⑤ 评测

| 文档 | 内容 |
|------|------|
| [ClawEval_Metadata.md](ClawEval_Metadata.md) | ClawEval 评测数据集任务分类、难度分布、模型排名 |

> 论文产出在仓库根 [`paper/`](../paper/) 目录（drafts 中英 Intro/Method + 总览、latex、refs）。
