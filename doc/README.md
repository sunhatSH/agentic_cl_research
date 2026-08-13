# 文档索引（doc/）

> **接手项目**：先读 `ops/Migration_64GPU.md` → `archive/Progress.md` → `source/CL_Design.md`。
> **沙箱文档入口**：`ops/sandbox/Sandbox_概念与术语.md`。
> **论文/学位论文参考**：`refs.md`（唯一信源）+ `重构指南.md`（两套论文资产改造规格）。
> 归档文件（冗余/过期/已合并）在 `archive/`，不再维护。

---

## source/ — 信源（论文与代码的上游依据，长期维护）

| 文档 | 内容 |
|------|------|
| [CL_Design.md](source/CL_Design.md) | **主文档**：CL Loss / Replay Buffer（9桶+quota+priority+冷启动数据需求）/ 实验路线 / 评测 / GPU / 精度 / 文献 |
| [BucketAlgorithm.md](source/BucketAlgorithm.md) | 分桶算法：能力坐标 + max-distance 训练序 |
| [usersim.md](source/usersim.md) | **UserSim 单一信源**：模型选型 + 三 agent 架构 + 多轮 query 在线生成 + 42 人设表 |
| [训练与推理流程.md](source/训练与推理流程.md) | 训练循环 + 推理全链路 + verl 0.8.0 集成 + 数据 pipeline |
| [ClawEval_Metadata.md](source/ClawEval_Metadata.md) | 评测基准数据 |
| [Agent轨迹_Schema.md](source/Agent轨迹_Schema.md) | Agent 轨迹 schema（现状 mock vs 目标 buffer/训练） |
| [Hermes_Subagent_训练数据方案.md](source/Hermes_Subagent_训练数据方案.md) | Hermes 同步出入栈 + OpenClaw 主子各自训练方案 |

## ops/ — 运行手册（操作向，按需查阅）

| 文档 | 内容 |
|------|------|
| [Migration_64GPU.md](ops/Migration_64GPU.md) | 跨机器交接 + 冷启动步骤 + 集群提交快速参考（附录） |
| [9B_16GPU_Config_2026-07-24.md](ops/9B_16GPU_Config_2026-07-24.md) | 9B 16GPU 配置说明 |
| [Concurrency_Policy_by_GPU.md](ops/Concurrency_Policy_by_GPU.md) | 按 GPU 的并发策略 |
| [reward.md](ops/reward.md) | reward 设计（frozen judge 打分） |
| [完整流程_从数据到训练.md](ops/完整流程_从数据到训练.md) | 端到端流程 |
| [冷采集数据管线.md](ops/冷采集数据管线.md) | 冷启动采集管线 |
| [数据清洗.md](ops/数据清洗.md) | 数据清洗规则 |
| [数据筛选逻辑_20260812.md](ops/数据筛选逻辑_20260812.md) | **训练数据难度筛选**：双模型交集 + system-reminder 剥离 + 中等难度定义 |

### ops/sandbox/ — 沙箱

| 文档 | 内容 |
|------|------|
| [Sandbox_概念与术语.md](ops/sandbox/Sandbox_概念与术语.md) | **沙箱入口**：镜像/Tool/Instance + TCR/CCR + API 对照 + 代码执行 + 真实规格 |
| [Sandbox_冒烟指南.md](ops/sandbox/Sandbox_冒烟指南.md) | **沙箱操作唯一入口**：冒烟步骤 + 命令速查 + custom 镜像 build + 常见卡点 |
| [Sandbox_Agent架构.md](ops/sandbox/Sandbox_Agent架构.md) | 动作内/推理外 + OpenClaw |
| [Sandbox_管理调度指南.md](ops/sandbox/Sandbox_管理调度指南.md) | 16×8 winner-sync 调度 |
| [接口使用_Sandbox与三Agent.md](ops/sandbox/接口使用_Sandbox与三Agent.md) | **接口怎么用速查**：Sandbox 后端 + 三 Agent 报告/声明 + diff-driven 设计 |
| [ColdRollout_采集.md](ops/sandbox/ColdRollout_采集.md) | 冷启动采集运行手册 |
| [沙箱_Dockerfile制作方案.md](ops/sandbox/沙箱_Dockerfile制作方案.md) | 沙箱镜像 Dockerfile 制作方案 |

## eval/ — 评测

| 文档 | 内容 |
|------|------|
| [防遗忘评测方案.md](eval/防遗忘评测方案.md) | 防遗忘评测方案：按桶分组训练 + 统一评测 + 权重后置 |
| [训练与评测总思路_产物结构.md](eval/训练与评测总思路_产物结构.md) | 训练/评测产物结构 |

## debug/ — 调试记录（踩坑与修复）

| 文档 | 内容 |
|------|------|
| [Bug_Fix_精简总表.md](debug/Bug_Fix_精简总表.md) | bug 修复精简总表 |
| [16gpu_hang_handoff_2026-08-01.md](debug/16gpu_hang_handoff_2026-08-01.md) | 16GPU hang 交接 |
| [16—migrate-4.md](debug/16—migrate-4.md) | 迁移记录 |
| [LightLLM_pause_abort_deadlock_请教.md](debug/LightLLM_pause_abort_deadlock_请教.md) | LightLLM pause/abort 死锁 |
| [OOM_求助_GPT.md](debug/OOM_求助_GPT.md) | OOM 排查 |
| [Training_Debug_2026-07-24.md](debug/Training_Debug_2026-07-24.md) | 训练调试记录 |

## expr/ — 实验（面向结论，每实验一子目录）

| 文档 | 内容 |
|------|------|
| [README.md](expr/README.md) | 算法迭代 + 实验总说明 |
| [B1/](expr/B1/) | 纯 PPO baseline（每次训练一个日期 .md + 图引 assets/） |
| [K2/](expr/K2/) | PPO + KL 约束 |
| [R0/](expr/R0/) | CLEAR baseline（单桶 replay） |
| assets/ · data/ | 实验图 / 数据（各 .md 用相对路径 `../assets/` 引用） |

## weekly_report/ — 周报

| 文档 | 内容 |
|------|------|
| [20260706-20260712工作.md](weekly_report/20260706-20260712工作.md) | |
| [20260718-20260723工作.md](weekly_report/20260718-20260723工作.md) | |
| [20260724-20260729工作.md](weekly_report/20260724-20260729工作.md) | |
| [20260730-20260731工作.md](weekly_report/20260730-20260731工作.md) | |
| [20260803-20260810工作.md](weekly_report/20260803-20260810工作.md) | |

## 根目录 — 单一信源

| 文档 | 内容 |
|------|------|
| [prompt.md](prompt.md) | 三 Agent 取证 prompt（ENVIRONMENT DIFF） |
| [refs.md](refs.md) | **参考文献唯一信源**（同步到 paper/refs + master-thesis/ref） |
| [重构指南.md](重构指南.md) | 论文重构规格：paper/ + master-thesis/ 两套资产改造 |

## archive/ — 归档（冗余/过期/已合并，不再维护）

含过程记录（Progress.md / RunLog.md / BugLog_集群采集.md）、一次性技术报告副本、已合并的启动指南/操作手册/踩坑记录/VerlIntegration/Plan_冷启动/BucketDesign/Buffer_冷启动/UserSim 四件套/WeeklyReport_20260713 等，详见 `archive/` 目录。

> 论文产出在 [`../paper/`](../paper/) 目录（drafts 中英 Intro/Method + 总览、latex、refs）。
> 学位论文在 [`../master-thesis/`](../master-thesis/) 目录。
