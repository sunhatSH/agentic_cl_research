# Agentic CL Research

Continual Learning over Agentic LLM 的训练项目——在 GRPO 上叠加 Replay Buffer 与策略约束，在持续学习新任务的同时减小对旧任务能力的遗忘。

## 结构

```
.
├── CLAUDE.md                # AI 协作指南（含完整架构图）
├── README.md                # 本文件
├── pyproject.toml           # 项目元数据与依赖
├── agents/                  # UserSim 三 agent（observer/questioner/reward）
├── bin/                     # 数据管道工具 + 编译工具（clean_zerowidth / detact / pipeline_cpp）
├── configs/                 # 21+ 实验 yaml（按 phase 分子目录 + 3 层继承）
│   ├── base.yaml            #   共享默认配置
│   ├── cluster.yaml         #   集群 64 卡引擎层 overlay
│   ├── run/                 #   集群可运行配置（base + cluster + 实验语义）
│   └── phase<N>/            #   各 phase 实验定义
├── doc/                     # 设计文档（60 篇，索引见 doc/README.md）
├── docker/                  # 镜像构建（sandbox agent runtime / vllm019 / lightllm）
├── eval/                    # ClawEval 评测（195 纯文本任务）
├── inference/               # 单步生成边界（VerlRolloutGenerateFn / HTTP）
├── paper/                   # 论文产出（drafts / latex / refs / assets）
├── replay_buffer/           # 9 桶 Buffer（与 verl 解耦的纯 Python 模块）
├── rollout/                 # 采样侧：沙箱客户端 / 会话池 / 轨迹采集
├── scripts/                 # 训练 / 沙箱 / 数据 / 评测启动脚本（35 个）
├── skills/                  # 可复用方法论（5 篇工程规范）
└── tests/                   # ~290 单元测试 + verl 兼容性 smoke
```

## 快速开始

```bash
# 安装
pip install -e ".[dev]"

# 跑 baseline (Phase 1)
bash scripts/train.sh configs/phase1/b1.yaml

# 评测 ckpt
bash scripts/eval.sh ckpts/b1-step-100
```

## 核心设计

- **CL Loss**：$L_{cl} = \lambda_1 L_{rl} + \lambda_2 L_{kl} + \lambda_3 L_{replay} + \lambda_4 L_{ent}$；$\lambda_4=0.001$ 全程开启防 Echo Trap。
- **9 桶 Buffer**：按能力/领域分桶（不按难度），桶内淘汰禁止跨桶挤出，priority 用抗遗忘信号而非 reward 绝对值。
- **Token 级 w**：W2 主方案 = priority × U 形块权重 $\frac{\gamma^{\text{block}} + \delta^{K_i - \text{block}}}{2}$ + clip + normalize；首尾两端高、中间低（$\gamma=\delta=0.88$）。$\gamma=\delta=1$ 时 U 形退化为均权（W0），超参连续可调。块按动作块切分，$K_i$ 因 trajectory 而异，解析失败或 $K_i=1$ 退化为等长 $K=20$。
- **训练框架**：[verl](https://github.com/volcengine/verl)，**不 fork**——通过 `actor.set_loss_fn` 注入自定义 loss，Buffer 完全外挂。详见 `doc/source/训练与推理流程.md`。

## 实验路线

```
Phase 1 (B1)          建立纯 RL 遗忘基线
   │
   ├── Phase 2 (K1-K5, K2-R)             KL 单独验证
   │
   └── Phase 3 (R0, R3-R6, R4-w, R4-K)   Replay 单独验证
           │
           └── Phase 4 (C1-C4)            KL × Replay 组合
                   │
                   └── Phase 5 (S1, S2)   Rollout 规模扩展
                           │
                           └── Phase 6 (X1-X7)  按需探索
```

共 21 个核心训练（B1 + K1-5/K2-R = 6 + R0-10k/R0-25k/R3-6/R4-w/R4-k = 8 + C1-4 = 4 + S1-2 = 2，Phase 6 按需）。详见 `doc/source/CL_Design.md`。

## 分工

本项目由 @孙豪 独立完成（CL 更新策略、数据采集与清洗、工具环境、评测全链路）。
