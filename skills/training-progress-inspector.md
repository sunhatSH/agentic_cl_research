# Skill: 训练进度巡检（某实验 / 某卡数 跑到哪了）

## 适用场景

回答「**某个实验、某个卡数配置，训练进度到哪了、还活着吗**」这类问题，例如：
- 「b1 4 卡跑到第几步了？」
- 「k2 16 卡这个实验多少卡、训到百分之几了？」
- 「r4 16 卡的 ckpt 存到哪一步、进程还在不在？」

只做**只读巡检**：解析配置里的卡数与目标步数、读实时 step、比出进度百分比、判进程存活。不启动、不改配置、不动 ckpt。

## 命名约定（唯一入口，先记牢）

| 概念 | 位置 / 格式 |
|------|-----------|
| 实验 config | `configs/run/<exp>_9b_<N>gpu.yaml`（如 `b1_9b_4gpu.yaml`） |
| 实验名 experiment_name | `qwen35_9b_<exp>_<N>gpu`（config 内 `trainer.experiment_name`） |
| 卡数 | config 内 `trainer.nnodes × trainer.n_gpus_per_node`，也编码在文件名 `_4gpu`/`_16gpu` |
| 目标步数 | config 内 `trainer.total_training_steps`（当前多为 500） |
| 存档间隔 | config 内 `trainer.save_freq`（debug 阶段 25） |
| **已存 ckpt 步数** | `ckpts/<name>/latest_checkpointed_iteration.txt`（纯数字，最新已存 step） |
| **实时训练步数** | `logs/metrics/<name>/metrics.jsonl` **末行的 `step`**（每 step 实时写，比 ckpt 细） |
| 跨 resume 累积 metrics | `logs/metrics/<name>/metrics.all.jsonl`（verl FileLogger 每次启动覆盖 `metrics.jsonl`，永久曲线看这个） |
| 实时日志 | `logs/experiments/<name>/train.log`（多机有 rank 后缀 `train-rank1.log`） |

> ⚠️ `<exp>` 用短横还是下划线要看实际：config 文件名里 `k2-r`、`r0-10k` 带横杠，experiment_name 里同样保留。拿不准就 `ls configs/run/ | grep <exp>` 与 `ls ckpts/ | grep <exp>` 对齐。

## 核心步骤

1. **解析实验**：从用户给的 `<exp>` + 卡数拼出 config 路径与 experiment_name。多个候选时列出让用户确认，别猜。
2. **读卡数**：`nnodes × n_gpus_per_node`（config 权威；文件名 `_Ngpu` 仅作交叉核对，preset 缩放时可能不一致，以 config 为准）。
3. **读进度**（两个数，都要报，区分清楚）：
   - **实时 step** = `metrics.jsonl` 末行 `step`（训练真实进度）
   - **已存 step** = `latest_checkpointed_iteration.txt`（能 resume 的最近存档点）
   - 二者差 = 上次存档后又跑了几步（未落盘、崩了会丢的部分）
4. **算百分比**：`实时 step / total_training_steps`。
5. **判存活**：`train.log` 的 mtime 距今多久。几分钟内在动=大概率活着；几十分钟没动+step 没涨=疑似 hang/挂了（16 卡 hang 是已知坑，见 memory `16gpu-hang-missing-8th-replica`）。

## 一把梭脚本（复制即用）

```bash
# 用法: bash 本段 <exp> <N>gpu    例: b1 4gpu / k2 16gpu / r0-10k 16gpu
EXP="${1:?exp}"; TOPO="${2:?Ngpu}"
ROOT=/mnt/afs_toolcall/sunhao4/workspace/agentic_cl_research
cd "$ROOT"
PY=/mnt/afs_toolcall/sunhao4/miniconda3/bin/python3

CFG="configs/run/${EXP}_9b_${TOPO}.yaml"
[ -f "$CFG" ] || { echo "no config: $CFG"; ls configs/run/ | grep -i "${EXP}"; exit 1; }

NAME=$(grep -E '^\s*experiment_name:' "$CFG" | awk '{print $2}')
NN=$(grep -E '^\s*nnodes:'  "$CFG" | awk '{print $2}')
GP=$(grep -E '^\s*n_gpus_per_node:' "$CFG" | awk '{print $2}')
TOT=$(grep -E '^\s*total_training_steps:' "$CFG" | awk '{print $2}')
SF=$(grep -E '^\s*save_freq:' "$CFG" | awk '{print $2}')
GPUS=$(( ${NN:-0} * ${GP:-0} ))

CKPT_STEP=$(cat "ckpts/$NAME/latest_checkpointed_iteration.txt" 2>/dev/null || echo "无")
MJ="logs/metrics/$NAME/metrics.jsonl"
LIVE_STEP=$($PY - "$MJ" <<'PY' 2>/dev/null || echo "无"
import json,sys
try:
    last=None
    for ln in open(sys.argv[1]):
        ln=ln.strip()
        if ln: last=ln
    print(json.loads(last).get("step","?") if last else "无")
except Exception: print("无")
PY
)
LOG=$(ls -t "logs/experiments/$NAME"/train*.log 2>/dev/null | head -1)
if [ -n "$LOG" ]; then
  AGE=$(( ( $(date +%s) - $(stat -c %Y "$LOG") ) / 60 ))
  ALIVE="日志 ${AGE} 分钟前更新"
else ALIVE="无日志（未启动或已清）"; fi

PCT="?"; [ "$LIVE_STEP" != "无" ] && [ -n "$TOT" ] && PCT=$($PY -c "print(f'{100*$LIVE_STEP/$TOT:.1f}%')" 2>/dev/null)

echo "实验:      $NAME"
echo "卡数:      $GPUS  (${NN}节点 × ${GP}卡)"
echo "进度:      实时 step $LIVE_STEP / $TOT  ($PCT)"
echo "已存 ckpt: step $CKPT_STEP   (save_freq=$SF)"
echo "存活:      $ALIVE"
```

输出示例：
```
实验:      qwen35_9b_b1_4gpu
卡数:      4  (1节点 × 4卡)
进度:      实时 step 103 / 500  (20.6%)
已存 ckpt: step 100   (save_freq=25)
存活:      日志 3 分钟前更新
```

## 关键约束 / 陷阱

- **实时 step ≠ 已存 ckpt step**：metrics 每 step 写、ckpt 每 `save_freq` 存。报进度用实时 step，报「能恢复到哪」用 ckpt step，两者都给，别混。
- **`metrics.jsonl` 会被覆盖**：verl FileLogger 每次启动 `open(path,"wb")`，只含**本次 run**。要完整曲线读同目录 `metrics.all.jsonl`（`_train_impl.sh` resume 前折叠进去的）。
- **卡数以 config 为准**：文件名 `_16gpu` 只是 preset 标签；实际 `nnodes×gpus` 才是权威，preset 缩放（16→4 debug）时可能对不上。
- **step 没涨 + 日志不动 ≠ 一定挂**：可能在一个超长 rollout 里（`session/duration_s/max` 可达数百秒）。判 hang 前先 `tail train.log` 看最后在干嘛；16 卡静默 PENDING 是已知根因（memory `16gpu-hang-missing-8th-replica`）。
- **画曲线**：要看 reward/loss 趋势而非单点，用 `scripts/plot/plot_metrics_nopandas.py <metrics.all.jsonl> -o <outdir>`（无 pandas 依赖）。
- **纯只读**：本 skill 不启动训练（那要 `scripts/train.sh <topo> --config <cfg>`，且必须 tmux/nohup 后台，见项目 CLAUDE.md）。

## 代码锚点

- config 卡数/步数：`configs/run/*_9b_*gpu.yaml` 的 `trainer:` 段
- ckpt 输出：`ckpts/<experiment_name>/{global_step_N/,latest_checkpointed_iteration.txt}`
- metrics 落盘逻辑：`scripts/_train_impl.sh:308-371`（`VERL_FILE_LOGGER_PATH` + all.jsonl 折叠）
- 日志目录：`scripts/_train_impl.sh:252`（`_LOGDIR=logs/experiments/<exp>`）
- 启动/拓扑 preset：`scripts/train.sh`（`TP[Ngpu]` 表）
```
