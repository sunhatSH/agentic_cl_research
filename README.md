# Agentic CL Research

Continual Learning over Agentic LLM 的训练项目——在 GRPO 上叠加 Replay Buffer 与策略约束，在持续学习新任务的同时减小对旧任务能力的遗忘。

## 结构

```
.
├── CLAUDE.md                # AI 协作指南
├── README.md                # 本文件
├── pyproject.toml           # 项目元数据与依赖
├── doc/                     # 设计文档
│   ├── CL_Update_Sunhao.md      # 主文档：Loss、Buffer、实验路线、评测、参考文献
│   ├── BucketDesign.md          # 7 桶结构详细论证
│   ├── BucketDesign_compressed.md
│   ├── ContinualLearning.md     # CL Loop 全流程
│   ├── ClawEval_Metadata.md     # 评测基准
│   └── VerlIntegration.md       # verl 集成路径
├── replay_buffer/           # 7 桶 Buffer（与 verl 解耦的纯 Python 模块）
│   ├── bucket.py            # 顶层 Buffer + quota 分配
│   ├── priority.py          # 抗遗忘 priority（4 信号融合，非 reward 绝对值）
│   ├── eviction.py          # 桶内淘汰（hard floor + soft target）
│   ├── sampler.py           # 两级采样（采桶 + 桶内 priority 加权）
│   ├── weighting.py         # token-level w_t（W0 / W2）
│   └── store.py             # 主存储 + 索引
├── trainer/                 # CL Loss 与训练入口（基于 verl，零源码改动）
│   ├── cl_loss.py           # cl_loss(config, model_output, data, dp_group)
│   └── cl_main.py           # 训练入口
├── configs/                 # 20 个实验的 yaml（B1, K*, R*, C*, S*）
├── eval/                    # ClawEval 评测（195 文本任务）
├── scripts/                 # 训练 / 评测启动脚本
└── tests/                   # 单元测试 + verl 兼容性 smoke
```

## 快速开始

```bash
# 安装
pip install -e ".[dev]"

# 跑 baseline (Phase 1)
bash scripts/train.sh configs/b1.yaml

# 评测 ckpt
bash scripts/eval.sh ckpts/b1-step-100
```

## 核心设计

- **CL Loss**：$L_{cl} = \lambda_1 L_{rl} + \lambda_2 L_{kl} + \lambda_3 L_{replay} + \lambda_4 L_{ent}$；$\lambda_4=0.001$ 全程开启防 Echo Trap。
- **7 桶 Buffer**：按能力/领域分桶（不按难度），桶内淘汰禁止跨桶挤出，priority 用抗遗忘信号而非 reward 绝对值。
- **Token 级 w**：W2 主方案 = priority × U 形块权重 ($\gamma^{\text{block}} + \delta^{K_i - \text{block}}$) + clip + normalize；首尾两端高、中间低（$\gamma=\delta=0.88$）。块按动作块（`<think>` / `<toolcall>` / `<observation>` / `<final_answer>` 等结构标签）划分，$K_i$ 因 trajectory 而异——具体切分规则待数据到位后定，代码 fallback 用等长 $K=20$。
- **训练框架**：[verl](https://github.com/volcengine/verl)，**不 fork**——通过 `actor.set_loss_fn` 注入自定义 loss，Buffer 完全外挂。详见 `doc/VerlIntegration.md`。

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

共 20 个核心训练。详见 `doc/CL_Update_Sunhao.md`。

## 分工

| 模块 | 负责人 |
|------|--------|
| CL 更新策略 / 调研 | @孙豪 |
| 用户数据获取 | @吴健 |
| 工具环境（Agent Framework） | @郑乃榕 |
| 评测 | @杨益博 |
