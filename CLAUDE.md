# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 仓库性质

Continual Learning over Agentic LLM 的训练项目。仓库所有者：@孙豪。

- **设计文档**：全部位于 `doc/`，是项目的需求与设计依据
- **代码骨架**：`replay_buffer/`、`trainer/`、`configs/`、`eval/`、`scripts/`、`tests/`
- **训练框架**：[verl](https://github.com/volcengine/verl)（pip 安装，不 fork，详见 `doc/VerlIntegration.md`）

## 开发命令

```bash
# 安装（含开发依赖）
pip install -e ".[dev]"

# 运行所有测试
pytest

# 运行单个测试文件 / 单个测试函数
pytest tests/test_bucket.py
pytest tests/test_bucket.py::test_quota_allocation -v

# Lint（ruff）
ruff check .
ruff check --fix .          # 自动修复

# 格式化（black）
black .
black --check .             # 仅检查，不修改

# 类型检查
mypy replay_buffer/ trainer/

# 训练（单实验）
bash scripts/train.sh configs/phase1/b1.yaml
bash scripts/train.sh configs/phase3/r4.yaml --resume-from ckpts/r4-step-50

# 训练（整个 phase）
bash scripts/phase3/run.sh
bash scripts/phase3/run.sh --only r4    # phase 内单个实验

# 评测
bash scripts/eval.sh ckpts/b1-step-100

# 训练入口也可直接调用
python -m trainer.cl_main --config configs/phase1/b1.yaml
```

## 代码架构

```
┌──────────────────────────────────────────────────────────────┐
│ verl (pip install, 不 fork)                                  │
│   RayPPOTrainer → actor.set_loss_fn(cl_loss)  ← 唯一注入点  │
└──────────────────────────────────┬───────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────┐
│ trainer/                                                      │
│   cl_main.py   — 入口：解析 yaml → 构建 buffer → 注入 loss   │
│   cl_loss.py   — make_cl_loss() 返回 verl 兼容的 loss_fn     │
│                  内部调用 compute_replay_loss() 做 replay 前向│
└──────────────────────────────────┬───────────────────────────┘
                                   │ 使用
┌──────────────────────────────────▼───────────────────────────┐
│ replay_buffer/ （纯 Python，不依赖 verl）                     │
│   BucketReplayBuffer  — 顶层接口：add / sample / stats        │
│     ├── Priority       — 4 信号融合的 trajectory 优先级       │
│     ├── TwoLevelSampler— 采桶 → 桶内 priority 加权随机       │
│     ├── TokenWeighting — W0(均权) / W2(U 形块权重)            │
│     ├── eviction.py    — 桶内淘汰策略                         │
│     └── store.py       — 底层存储 + 索引                      │
└──────────────────────────────────────────────────────────────┘

configs/base.yaml        — 所有实验共享的基础配置（OmegaConf 继承）
configs/phase<N>/<exp>.yaml — 按 phase 隔离，每个 yaml 只 override 变化的字段
```

**关键设计约束**：`replay_buffer/` 必须与 verl 完全解耦——不 import verl、不依赖 Ray。这保证 Buffer 可独立单测。

## 编码规范

- Python ≥ 3.10，ruff line-length=110，black line-length=110
- ruff 启用规则：E, F, W, I (isort), B (bugbear), UP (pyupgrade)
- 训练配置用 OmegaConf/yaml（非 argparse dataclass），实验 yaml 继承 `configs/base.yaml`

## 目录结构

```
agentic_cl_research/
├── CLAUDE.md                # 本文件，AI 协作指南
├── README.md                # 项目入口
├── pyproject.toml           # 项目元数据与依赖
├── doc/                     # 设计文档（详见下表）
├── replay_buffer/           # 7 桶 Buffer 实现（与 verl 解耦的纯 Python 模块）
├── trainer/                 # CL Loss 与训练入口（基于 verl，零源码改动）
├── configs/                 # 实验配置（按 phase 分子目录）
│   ├── base.yaml            #   共享默认配置
│   ├── phase1/              #   B1
│   ├── phase2/              #   K1-K5, K2-R
│   ├── phase3/              #   R0, R3-R6, R4-w, R4-K
│   ├── phase4/              #   C1-C4
│   ├── phase5/              #   S1, S2
│   └── phase6/              #   X* (按需)
├── scripts/                 # 训练 / 评测脚本（按 phase 分子目录）
│   ├── train.sh             #   通用单实验入口
│   ├── eval.sh              #   通用评测入口
│   ├── phase1/run.sh        #   启动 Phase 1 全部实验
│   ├── phase2/run.sh        #   启动 Phase 2 全部实验 (--only 选单个)
│   ├── ...
│   └── phase5/run.sh
├── eval/                    # ClawEval 评测
├── tests/                   # 单元测试
│
│ ─── 运行时产物（gitignored）───
├── ckpts/                   # 训练 checkpoint 输出
│   └── <实验名>-step-<N>/
├── buffer_dumps/            # Replay Buffer 序列化快照
├── logs/                    # 训练日志
├── wandb/                   # W&B 实验追踪本地目录
└── eval/results/            # 评测输出结果
```

### 模型存放约定

| 类别 | 存放位置 | 说明 |
|------|----------|------|
| **预训练基底模型** | 仓库外，绝对路径引用 | **基座 = Qwen3.6-27B**（HF: `Qwen/Qwen3.6-27B`）。本地缓存路径如 `/mnt/afs/models/qwen3.6-27b/` 或 HuggingFace cache `~/.cache/huggingface/`。已在 `configs/base.yaml` 的 `model.name` / `model.tokenizer` 中默认指定 |
| **参考策略 $\pi_{ref}$** | `ckpts/` 或仓库外 | `actor_rollout_ref.ref.path` 指定。$\pi_0$ 可指向基底模型路径，$\pi_{t-1}$ 指向上一阶段的 ckpt |
| **训练 checkpoint** | `ckpts/<实验名>-step-<N>/` | trainer 自动写入，`save_freq=25` step 保存一次 |
| **Buffer 快照** | `buffer_dumps/` | 可选持久化，训练中断后可恢复 buffer 状态 |

> 所有运行时产物均已在 `.gitignore` 中排除，不要提交到 git。

## 文档结构与关系

所有设计文档位于 `doc/`：

| 文件 | 内容 | 定位 |
|------|------|------|
| `doc/CL_Update_Sunhao.md` | CL 总设计：Loss 公式、Replay Buffer、实验路线、评测指标、参考文献 | **主文档**，其他文档的上下文依赖 |
| `doc/BucketDesign.md` | Replay Buffer 7 桶结构的详细论证（为什么这样分桶、为什么不用难度分桶、quota 推导过程） | `CL_Update_Sunhao.md` 中 Replay Buffer 部分的完整展开 |
| `doc/BucketDesign_compressed.md` | `BucketDesign.md` 的精简版，仅保留结论和公式 | 快速查阅版 |
| `doc/ContinualLearning.md` | CL Loop 全流程概述：数据获取、更新策略、工具环境、评测 | 多人协作总览，各模块负责人分工 |
| `doc/ClawEval_Metadata.md` | ClawEval 评测数据集的任务分类、难度分布、工具能力层、模型排名 | 评测基准参考 |
| `doc/VerlIntegration.md` | verl 集成指导：是否 fork、Replay Buffer 接入方式、推荐工程结构、风险点 | 实现路径参考 |
| `doc/SandboxRollout.md` | 基于腾讯 Agent Runtime（E2B 兼容）的 trajectory 采集方案：每 query × M 个沙盒、snapshot fork/pause、advantage 选优胜 | rollout 工程方案 |
| `doc/Progress.md` | 里程碑、模块完成度、20 实验状态、外部依赖阻塞、变更日志 | **进度单一来源** |

阅读顺序建议：`ContinualLearning.md`（全貌）→ `CL_Update_Sunhao.md`（技术细节）→ `BucketDesign.md`（分桶论证）→ `ClawEval_Metadata.md`（评测数据）→ `VerlIntegration.md`（落地工程）。

## 核心设计要点

### CL Loss

$$L_{cl} = \lambda_1 L_{rl} + \lambda_2 L_{kl} + \lambda_3 L_{replay} + \lambda_4 L_{ent}$$

- $L_{reg}$（参数 L2 正则）**弃用**，权重为 0，槽位让给 $L_{ent}$
- $\lambda_4 = 0.001$ **所有 Phase 固定开启**，防 Echo Trap（traj/query=2 下关闭 entropy 会训练崩盘），不参与 ablation
- $L_{kl}$ 用 reverse KL：$D_{KL}(\pi_{new} \| \pi_{ref})$

### Replay Buffer 7 桶结构

桶按**能力/领域**划分，不按难度划分（难度会随模型能力漂移）：

1. Workflow [54] — 多步骤任务组织
2. SysOps [52] — 工具使用与系统操作
3. Dialogue [38] — 多轮交互与状态跟踪
4. Finance [18] — 结构化业务规则
5. Communication [12] — 表达与沟通
6. Knowledge/Analysis [11] — 检索与推理
7. OfficeQA [10] — 办公语境问答（单独成桶，不并入 Knowledge）

**不纳入 multimodal(4)**：模态不同、数据太少、目标不一致。纯文本共 195 任务。

### 关键设计约束

- **桶内淘汰，禁止跨桶挤出** — 保证不同能力不互相侵占
- **Priority 不用 reward 绝对值** — reward 整体上升会系统性淘汰旧轨迹，buffer 退化为滑动窗口
- **Quota 分配**：保底 $q_{min}$ + 平方根加权（$\alpha=0.5$），大桶得更多但不按比例膨胀
- **采样**：两级采样（先采桶→桶内采轨迹），混合 quota 比例 + 均匀

### 实验路线

Phase 1→2→4→5→6 + 独立 Phase 3，共 20 个训练，单实验 ~16 GPU-day (8×H100)。完整参数表见 `doc/CL_Update_Sunhao.md`。

### GPU 部署与精度

- **基座模型**：Qwen3.6-27B（HF: `Qwen/Qwen3.6-27B`）。
- **Fully Async Policy 分离 40+24**：推理 40 卡 (5×TP8 vLLM) + 训练 24 卡 FSDP，`staleness_threshold=0.3`。Colocate 64 为后备。
- **BF16 全栈**：FP32 主权重 + FP32 Adam m/v；FP8 不进主路径。
- ⚠️ `doc/CL_Update_Sunhao.md` 中按 70B 估算的显存 / 同步耗时数字待按 27B 重算。
- 详见 `doc/CL_Update_Sunhao.md` § GPU 资源分配与训练流水线 / § 训练精度方案。

### $L_{replay}$ 权重公式

$$w_t^{(i)} = \text{normalize}\Big(\text{clip}\big(\text{priority}_i \cdot \frac{\gamma^{\text{block}(t)} + \delta^{K_i - \text{block}(t)}}{2},\; q_5,\; q_{95}\big)\Big)$$

两维度：Priority（trajectory 级）× **U 形块权重**（首尾两端高、中间低；起步 $\gamma=\delta=0.88$）。块按**动作块**（`<think>` / `<toolcall>` / `<observation>` / `<final_answer>` 等结构标签）划分，$K_i$ 因 trajectory 而异——具体标签集合与切分规则待数据到位后定，代码 fallback 用等长 $K=20$。Phase 3 对照 W0（均权）vs W2（主方案）。

> **2026-06-08 反转**：原方案为单调块衰减 + final_answer boost；改为 U 形是因为"末端的重要性不止 final_answer 一个 token 段，靠近末端的多个块都重要"，单点 boost 抓不住整段。详见 `doc/CL_Update_Sunhao.md`。

## 术语与缩写

| 术语 | 含义 |
|------|------|
| CL | Continual Learning |
| Echo Trap | 多轮 agent RL 中因策略坍缩导致训练崩溃的现象（参考文献 B4） |
| traj/query | 每 query 的 rollout 轨迹数，当前方案为 2（原 8） |
| CLEAR | Rolnick et al. 2019 的 experience replay 基线方法（参考文献 A2） |
| verl | 使用的 RL 训练框架 |
| GRPO | 使用的 RL 算法 |
| $\pi_0$ | 初始策略（模型初始 checkpoint） |
| $\pi_{t-1}$ | 上一阶段 checkpoint 策略 |
| hard floor | 每桶不可跌破的最低配额 $q_{min}$ |
| soft target | 按公式计算的桶目标配额 |

## 评测基准

使用 **ClawEval**（300 任务，3 split：General 161 / Multimodal 101 / Multi-turn 38）。评分公式：$score = s_{safety} \times (0.8 \cdot s_{completion} + 0.2 \cdot s_{robustness})$，Pass³ 标准（三次独立运行全通过）。当前仅使用纯文本任务（195 个）。

## 分工

| 模块 | 负责人 |
|------|--------|
| CL 更新策略 / 调研 | @孙豪 |
| 用户数据获取 | @吴健 |
| 工具环境（Agent Framework） | @郑乃榕 |
| 评测 | @杨益博 |
