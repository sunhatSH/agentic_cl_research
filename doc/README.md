# 文档索引（doc/）

> **接手项目**：先读 `ops/Migration_64GPU.md` → `archive/Progress.md` → `source/CL_Design.md`。
> **沙箱文档入口**：`ops/sandbox/Sandbox_概念与术语.md`。
> **论文/学位论文参考**：`refs.md`（唯一信源）+ `重构指南.md`（两套论文资产改造规格）。

---

## 目录总览

| 目录 | 定位 | 维护状态 |
|------|------|----------|
| [source/](#source--信源) | 信源：论文与代码的上游依据 | 长期维护 |
| [ops/](#ops--运行手册) | 运行手册（含 sandbox/） | 按需更新 |
| [eval/](#eval--评测) | 评测方案 | 维护 |
| [debug/](#debug--调试记录) | 训练排障记录 | 追加，结论已并入精简总表 |
| [tuning/](#tuning--调优方法) | 调优方法论（怎么归因、怎么调） | 追加 |
| [expr/](#expr--实验) | 实验结论（每实验一子目录） | 追加 |
| [weekly_report/](#weekly_report--周报) | 周报 | 追加 |
| [根目录单一信源](#根目录--单一信源) | prompt / refs / 重构指南 | 维护 |
| [archive/](#archive--归档) | 归档：冗余/过期/已合并 | **不再维护** |

## 接手阅读顺序

1. [ops/Migration_64GPU.md](ops/Migration_64GPU.md) — 交接快照 + 冷启动步骤
2. [archive/Progress.md](archive/Progress.md) — 交付状态单一来源（里程碑 / 模块完成度 / 阻塞）
3. [source/CL_Design.md](source/CL_Design.md) — 核心设计（CL Loss / Buffer / 实验路线）
4. [source/BucketAlgorithm.md](source/BucketAlgorithm.md) — 分桶算法规范
5. [source/usersim.md](source/usersim.md) — 三 Agent 用户模拟
6. [source/训练与推理流程.md](source/训练与推理流程.md) — 训练 + 推理全链路落地
7. [eval/防遗忘评测方案.md](eval/防遗忘评测方案.md) — 评测方法与遗忘度量

---

## source/ — 信源

| 文档 | 内容 |
|------|------|
| [CL_Design.md](source/CL_Design.md) | **主文档**：CL Loss / Replay Buffer（9桶+quota+priority+冷启动数据需求）/ 实验路线 / 评测 / GPU / 精度 / 文献 |
| [BucketAlgorithm.md](source/BucketAlgorithm.md) | 分桶算法：能力坐标 + max-distance 训练序 |
| [usersim.md](source/usersim.md) | **UserSim 单一信源**：模型选型 + 三 agent 架构 + 多轮 query 在线生成 + 42 人设表 |
| [训练与推理流程.md](source/训练与推理流程.md) | 训练循环 + 推理全链路 + verl 0.8.0 集成 + 数据 pipeline |
| [ClawEval_Metadata.md](source/ClawEval_Metadata.md) | 评测基准数据 |
| [Agent轨迹_Schema.md](source/Agent轨迹_Schema.md) | Agent 轨迹 schema（现状 mock vs 目标 buffer/训练） |
| [Hermes_Subagent_训练数据方案.md](source/Hermes_Subagent_训练数据方案.md) | Hermes 同步出入栈 + OpenClaw 主子各自训练方案 |

## ops/ — 运行手册

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

## debug/ — 调试记录

> **先读 [Bug_Fix_精简总表.md](debug/Bug_Fix_精简总表.md)**——已合并去重以下记录，只保留结论。

| 文档 | 内容 |
|------|------|
| [Bug_Fix_精简总表.md](debug/Bug_Fix_精简总表.md) | bug 修复精简总表 |
| [16gpu_hang_handoff_2026-08-01.md](debug/16gpu_hang_handoff_2026-08-01.md) | 16GPU hang 交接（✅ 已解决） |
| [16—migrate-4.md](debug/16—migrate-4.md) | verl 原生 agent_loop 迁移记录 |
| [LightLLM_pause_abort_deadlock_请教.md](debug/LightLLM_pause_abort_deadlock_请教.md) | LightLLM pause/abort 死锁 |
| [OOM_求助_GPT.md](debug/OOM_求助_GPT.md) | OOM 排查 |
| [Training_Debug_2026-07-24.md](debug/Training_Debug_2026-07-24.md) | 训练调试记录 |

## tuning/ — 调优方法

> 调优**方法论**（怎么归因、怎么判 prompt/超参、怎么真机验证），区别于 debug/ 的 bug 结论。

| 文档 | 内容 |
|------|------|
| [Reward_归因方法与调优经验.md](tuning/Reward_归因方法与调优经验.md) | reward 归因方法链（查得分→归类→低分归因→证伪→真机验证→判超参）+ 9 条可复用经验 |

## expr/ — 实验

| 文档 | 内容 |
|------|------|
| [README.md](expr/README.md) | 算法迭代 + 实验总说明 |
| [B1/](expr/B1/) | 纯 PPO baseline（每次训练一个日期 .md + 图引 assets/） |
| [K2/](expr/K2/) | PPO + KL 约束 |
| [R0/](expr/R0/) | CLEAR baseline（单桶 replay） |
| assets/ · data/ | 实验图 / 数据（各 .md 用相对路径 ../assets/ 引用） |

## weekly_report/ — 周报

| 文档 | 内容 |
|------|------|
| [20260706-20260712工作.md](weekly_report/20260706-20260712工作.md) | 冷采集全链路打通 + 数据清洗 |
| [20260718-20260723工作.md](weekly_report/20260718-20260723工作.md) | 4 卡环境适配 + Qwen3.5-9B smoke |
| [20260724-20260729工作.md](weekly_report/20260724-20260729工作.md) | 16 卡 9B baseline 从崩溃到跑通 |
| [20260730-20260731工作.md](weekly_report/20260730-20260731工作.md) | 迁移 verl 原生 agent_loop |
| [20260803-20260810工作.md](weekly_report/20260803-20260810工作.md) | |

## 根目录 — 单一信源

| 文档 | 内容 |
|------|------|
| [prompt.md](prompt.md) | 三 Agent 取证 prompt（ENVIRONMENT DIFF） |
| [refs.md](refs.md) | **参考文献唯一信源**（同步到 paper/refs + master-thesis/ref） |
| [重构指南.md](重构指南.md) | 论文重构规格：paper/ + master-thesis/ 两套资产改造 |

## archive/ — 归档

> **已归档，不再维护。** 冗余 / 过期 / 已合并的文档集中于此，仅作历史参考。核心结论均已上移到上面各目录。

| 分组 | 文件 |
|------|------|
| 交付状态 / 过程记录 | Progress.md、RunLog.md（append-only）、BugLog_集群采集.md |
| 施工图 / 计划 | Plan_训练链路补齐.md、Plan_冷启动数据来源消融.md、0622 待办计划.md |
| Buffer 历史设计 | BucketDesign.md、bucket_buffer.md、Buffer_冷启动数据需求.md（已被 source/BucketAlgorithm.md 取代） |
| UserSim 历史设计 | 模型选型.md、UserSim_三Agent架构与技术设计.md、UserSim_多轮Query在线生成.md、UserSim_人设库.md（已被 source/usersim.md 取代） |
| verl 集成 / 沙箱历史 | VerlIntegration.md、SandboxRollout.md、Sandbox_腾讯云操作手册.md、Sandbox_规格与run_code踩坑.md、沙箱操作验证与证据.md、沙箱_实例_Queries对应关系_待定.md |
| 技术报告 / 复盘 | 汇报_技术总报告.md、RolloutCollect_技术报告.md、集群推理采集_经验复盘.md、集群训练启动指南.md、vllm_upgrade_0.19.md、模型配额申请_sufy.md |
| 周报 / 工作整理（旧） | WeeklyReport_20260713-0717.md、20260805-08_工作整理.md |

> 论文产出在 [../paper/](../paper/) 目录（drafts 中英 Intro/Method + 总览、latex、refs）。
> 学位论文在 [../master-thesis/](../master-thesis/) 目录。
