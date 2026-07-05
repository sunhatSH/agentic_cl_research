# runs/ — 实验产物根目录

> 所有实验的**配置快照 + 评测结果 + 日志 + 汇总**统一收在这里，每个实验一个自包含子文件夹。
> 权重文件**不在这里**（太大）：真实 checkpoint 在仓库外/顶层 `ckpts/<exp>/`，`runs/<exp>/checkpoints` 是**软链接**指过去。
> 设计依据见 [`doc/eval/训练与评测总思路_产物结构.md`](../doc/eval/训练与评测总思路_产物结构.md)。
>
> 本目录内容 gitignore（运行时产物）；只有本 README 进 git 作结构约定。

## 结构

```
ckpts/                              # ← 真实权重(runs 外, 顶层, gitignore)
└── <exp>/global_step_<N>/

runs/
├── phase0/  p0-a..e   + _phase_summary.json   # 冷启动数据来源消融
├── phase1/  b1        + _phase_summary.json    # 遗忘基线
├── phase2/  k1..k5,k2-r + _phase_summary.json  # KL 单验 → top-2
├── phase3/  r0-10k..r4-k + _phase_summary.json # Replay 单验 → top-2
├── phase4/  c1..c4    + _phase_summary.json     # 组合
├── phase5/  s1,s2     + _phase_summary.json     # 规模
├── phase6/  x*                                  # 按需
├── baseline/                                    # 无 CL 的 27B(终点对决用)
└── _final/                                      # 终点: 持续学习遗忘对决
    ├── forgetting_report.json
    ├── cl_model/{scores.json, per_bucket/*.jsonl}
    ├── baseline/{scores.json, per_bucket/*.jsonl}
    ├── weighting/{linear.json, exponential.json, ...}
    └── all_scores_matrix.csv
```

## 单个实验目录（每实验自包含）

```
runs/phaseN/<exp>/
├── config.snapshot.yaml       # 训练参数快照(该实验完整 config, 可追溯)
├── checkpoints -> ../../../ckpts/<exp>/   # 软链, 真实权重在 runs 外
├── buffer/buffer-final.sqlite # replay buffer 快照
├── eval/                      # 过程评测(该实验训完评一次, phase 选参用)
│   ├── scores.json            #   总账: 每桶得分 + 该实验 vs baseline 两端对比
│   └── per_bucket/            #   细则: 桶内逐题
│       ├── Workflow.jsonl     #     每行 {task_id, passed_all, safety, completion, robustness, reward}
│       ├── SysOps.jsonl
│       └── ...(9 桶)
└── logs/{train.log, metrics.jsonl}
```

## 两层评测

| 层 | 位置 | 时机 | 用途 |
|----|------|------|------|
| 过程评测 | `phaseN/<exp>/eval/` | 每实验训完评一次 | phase 内对比选参; 含 CL vs baseline 两端 |
| 终点对决 | `_final/` | 最后一个 phase 完成后 | CL 最终模型 vs baseline 遗忘对比 + 权重后置 |
