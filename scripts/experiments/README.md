# 实验脚本(每个实验 = 训练 + 按 9 能力桶评测)

每个实验一个脚本，跑完自动评测。卡数解耦(`NNODES` 等环境变量)。

## 启动

| 命令 | 作用 |
|------|------|
| `bash scripts/experiments/b1.sh` | **baseline**(纯 27B RL，不加 CL 算法):训练 + 评测 |
| `bash scripts/experiments/r4.sh` | 完整 CL 方案(9 桶 buffer + priority):训练 + 评测 |
| `bash scripts/experiments/all.sh` | 按顺序跑【所有】实验(当前 b1 + r4) |

前面加 `NNODES=4` 走 32 卡；`RUN_EVAL=0` 只训不评。

## 产物(每个实验)

- 权重: `ckpts/<exp>/`
- 日志: `runs/<phase>/<exp>/logs/{train,eval}.log`
- 评测(按 9 桶): `runs/<phase>/<exp>/eval/scores.json` + `per_bucket/<桶>.jsonl`
- swanlab: project `RL`

## 加新实验

1. 在 `configs/run/` 建 cluster runnable 配置(`defaults: [../_generated_ppo_trainer, ../base, ../cluster]` + 实验语义)
2. 复制 `b1.sh` 改成 `<新实验>.sh`，指向新 config
3. 把 `<新实验>.sh` 加进 `all.sh` 的 `EXPERIMENTS` 列表

> 当前只有 b1/r4 是 cluster runnable。CLAUDE.md 的 21 实验(K*/R0-R6/C*/S*)大多待补 run 配置，
> 且 Phase4/5 参数需前序结果确定(configs/phase4-5 里的 `???`)。
