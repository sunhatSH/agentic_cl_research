# CL 分阶段实验脚本用法

> 一个入口脚本 `scripts/run_cl.sh`，四个动作：train / eval / stage1 / stage2。
> 防止遗忘，记录实验怎么跑、数据在哪、3 方法怎么区分、GPU 怎么传。

## 四个动作

```bash
# train — 训练一段（续训靠 train.sh auto-resume 自动检测 checkpoint）
bash scripts/run_cl.sh train <method> <exp> <total_steps> [override...]
#   例: bash scripts/run_cl.sh train cl cl2r_cl 200
#       bash scripts/run_cl.sh train cl cl2r_exp2_cl 100 data.train_files=.../train_exp2.parquet trainer.save_freq=1

# eval — 评测（评 base 或评某个 checkpoint）
bash scripts/run_cl.sh eval base                              # 评 base 模型（基础能力参照）
bash scripts/run_cl.sh eval <ckpt_actor> <exp> <step>         # merge + 评 checkpoint

# stage1 — 第一部分：coding→research 续训 200+200 + 评 2 checkpoint
bash scripts/run_cl.sh stage1 [method]                        # method 默认 all

# stage2 — 第二部分：7 桶少量边训边评
bash scripts/run_cl.sh stage2 [method]
```

## 3 方法（method）

| method | cl 配置 | 语义 |
|---|---|---|
| `baseline` | `cl.buffer.enabled=false cl.lambda_replay=0` | 无 buffer 无 replay |
| `clear` | （cl2r_base 默认） | 单桶 CLEAR：num_buckets=1, reservoir, W0 |
| `cl` | `num_buckets=9, eviction=fifo, bucket_strategy=distance, W2` | 9 桶 distance + U 形权重 |

## GPU 配置

- **训练**：固定 `train.sh 16gpu`（nnodes=2 gpus=8）。
- **评测**：主动传 `NNODES` + `GPUS_PER_NODE`（**不自动检测**，多节点 nvidia-smi 只查本节点）。
  - debug 机（4 卡）：`NNODES=1 GPUS_PER_NODE=4 bash scripts/run_cl.sh eval ...`
  - 16 卡：`NNODES=2 GPUS_PER_NODE=8`（默认）
- SP=4 / TP=2 保持 config 默认，`total_gpus = NNODES×GPUS_PER_NODE` 须为 4 的倍数。

## 数据

| 文件 | 内容 |
|---|---|
| `datasets/train_cl.parquet` | 第一步：coding 6400 + research 6400（4-6 难度），前 200 step coding / 后 200 research |
| `datasets/train_exp2.parquet` | 第二步：7 桶（office/ops/workflow 各 1600、qa 128、finance 416、safety 640、communication 160），共 192 step |
| `datasets/cold_start/train.parquet` | 冷启动（9 桶，1429 条），replay buffer 回放其他桶用 |

数据构建脚本：`scripts/pipeline/build_train.py`（第一步）、`scripts/pipeline/build_train_exp2.py`（第二步）。

## 评测链路（eval 动作内部）

```
merge(FSDP→HF, verl.model_merger) → 起 Ray → trainer.cl_eval（复用训练 RemoteAgentLoopManager）
→ generate_sequences(ClawEval, validate=True) → replay_buffer.sample(partition="val")
→ extract_trajectories_from_kvbatch 读 judge 四维(task_done/correctness/trajectory/safety)
→ 输出 eval/results/<exp>_step<step>/per_task.json
```

评测结果消费：`eval/run_eval.py`（per-bucket pass rate + CL-Score，`rollout_one_task` 待接批量）。

## 时序（每阶段训练与评测的顺序）

**stage1（第一部分，强信号）**——先训后评、续训再评：

```
评 base（一次，基础能力参照）
for method in baseline clear cl:
  训 coding 200 step        → checkpoint_A (global_step_200)
  续 research 400 step      → checkpoint_B (global_step_400)  [auto-resume]
  评 checkpoint_A（9 桶）     ← 看 coding 涨 + 其他 8 桶泛化下降
  评 checkpoint_B（9 桶）     ← 看 coding 遗忘（核心）+ research 涨
```

**stage2（第二部分，边训边评）**——每桶训完立即评一次：

```
for method in baseline clear cl:
  cum = 0
  for 桶 in [office, ops, workflow, qa, finance, safety, communication]:
    cum += 该桶 step 数
    训到 cum step（save_freq=1，续训靠 auto-resume）
    评 global_step_<cum>（9 桶）   ← 每桶训完立即评，即"边训边评"
```

> 数据顺序：`train_exp2.parquet` 按桶顺序排（office→ops→...，shuffle=false），前 50 step 是 office、接着 50 step ops...，续训 dataloader 游标自动续到下一桶。

**"对某个模型做边训边评"**（以 cl 方法为例）：

```bash
bash scripts/run_cl.sh stage2 cl
# 等价于手动逐桶：
#   bash scripts/run_cl.sh train cl cl2r_exp2_cl 50  data.train_files=.../train_exp2.parquet trainer.save_freq=1
#   bash scripts/run_cl.sh eval  ckpts/cl2r_exp2_cl/global_step_50/actor cl2r_exp2_cl 50
#   bash scripts/run_cl.sh train cl cl2r_exp2_cl 100 ...   # 续训 ops
#   bash scripts/run_cl.sh eval  ckpts/cl2r_exp2_cl/global_step_100/actor cl2r_exp2_cl 100
#   ...
```

## 完整实验流程

```bash
# 1. 评 base（基础能力参照，三组对比第 1 组）
bash scripts/run_cl.sh eval base

# 2. 第一步主实验（3 方法 × coding→research 续训 + 评 2 checkpoint）
bash scripts/run_cl.sh stage1 all

# 3. 第二步（可选，3 方法 × 7 桶边训边评）
bash scripts/run_cl.sh stage2 all
```

三组对比：纯基座（不训）/ 基座+纯RL（baseline）/ 基座+CL（cl）。最终目标：比 CLEAR 新能力提升且防遗忘更好，比纯RL 新能力略降但防遗忘大幅提升。

详见 `doc/eval/防遗忘评测方案.md` §11。
