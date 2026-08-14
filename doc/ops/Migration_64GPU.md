# 64 卡机器迁移 / Session 交接

> 新 session 先读本文件 → 再读 [`Progress.md`](../archive/Progress.md)（交付状态）→ 按「冷启动步骤」执行。
> **所有产出必须记录到 [`RunLog.md`](../archive/RunLog.md)**（成功/失败都保留）。

---

## 1. 交接快照

| 维度 | 状态 |
|------|------|
| 代码 | **全部完成**。`replay_buffer/` + `trainer/` + `configs/`(21 yaml) + `eval/` + 5 篇 `skills/` |
| 测试 | ~290 单测（288 passed + 7 skipped；skip = verl/CUDA 门控，本机无） |
| 论文证据钩子 | 完成：buffer 动态日志 + `forgetting_risk` 回填 + 周期 `buffer.dump` |
| 唯一阻塞 | **GPU 集群形态联调**（64 卡 + 真实 27B 权重 + 数据） |

## 2. 冷启动步骤

```bash
# 0. 定位仓库
cd <repo>/agentic_cl_research && git status && git log --oneline -5

# 1. 确认硬件 & 环境
nvidia-smi -L            # 期望 64 张
.venv/bin/python --version   # 期望 CPython 3.10.x；没有则 pip install -e ".[dev]"

# 2. 跑非 GPU 回归（期望 ~288 passed / 7 skipped）
.venv/bin/python -m pytest -q

# 3. 跑 GPU 标记测试（验证 torch/verl/CUDA）
.venv/bin/python -m pytest -m gpu -q

# 4. 确认外部依赖（决定能否真训练，见 §5）
#    - Qwen3.6-27B 权重路径（configs/base.yaml: actor_rollout_ref.model.path）
#    - 训练数据 train_files/val_files
#    - verl 完整 Hydra defaults（fsdp/optim/rollout engine）
```

Buffer 空时无需特殊操作：`len(buffer.store)==0` → replay_ratio=0 → 纯 RL 跑。B1（无 buffer）与 R*（有 buffer）启动方式一致。

## 3. 64 卡待办

| 优先级 | 任务 | 验收标准 |
|--------|------|----------|
| **P0-a** | merge verl 完整 Hydra defaults | `validate_config` 通过 |
| **P0-b** | **B1 全栈 smoke**（Colocate 64）：`scripts/train.sh configs/phase1/b1.yaml` 跑 1–2 step | 不崩；wandb 出现 `actor/pg_loss`；产出 step ckpt |
| **P0-c** | **R4 全栈 smoke**：`scripts/train.sh configs/phase3/r4.yaml` 跑 1–2 step | replay 行不报 shape/KeyError；`actor/replay_loss>0`；`logs/buffer_stats/r4.jsonl` 有行 |
| P1 | 切 **Fully Async 40+24** 跑通 | async 下 1–2 step 不崩 |
| P1 | 27B/64 卡显存重算（替换 70B 估算） | 写回 `CL_Design.md` |
| P2 | 跑完整 Phase 1→6（21 实验） | 见 `Progress.md` 实验表 |

**GPU-only 验证重点**（代码写好但只能 GPU 验）：
1. `_append_replay_rows` 的 `DataProto.concat` padding 模式
2. `build_replay_rows` 的真实 response span（须按 chat template offset 重算）
3. `compute_replay_current_logprobs` 的 DataProto schema（确认 `forgetting_risk` 真回填）

## 4. 命令速查

```bash
# 训练
bash scripts/train.sh configs/phase1/b1.yaml
bash scripts/train.sh configs/phase3/r4.yaml --resume-from ckpts/r4-step-50
python -m trainer.cl_main --config configs/phase3/r4.yaml

# 评测 / 测试
bash scripts/eval.sh ckpts/b1-step-100
.venv/bin/python -m pytest -q            # 非 GPU
.venv/bin/python -m pytest -m gpu -q     # GPU
ruff check . && black --check .
```

| 产物 | 落盘位置（gitignore） |
|------|----------------------|
| 训练 ckpt | `ckpts/<exp>-step-<N>/` |
| Buffer 日志 | `logs/buffer_stats/<exp>.jsonl` |
| Buffer 快照 | `buffer_dumps/<exp>-step-<N>.sqlite` |
| wandb | `wandb/` |
| 评测结果 | `eval/results/<run_id>/{per_task,summary}.json` |
| **运行记录** | **`doc/archive/RunLog.md`（必须维护）** |

## 5. 外部依赖

| 依赖 | 没有它会怎样 |
|------|-------------|
| Qwen3.6-27B 权重 | 无法加载 actor |
| 训练数据（taskspecs → 冷采集 → parquet） | buffer 入桶 & priority 缺失 |
| 沙箱 rollout 环境 | 无真实 trajectory |
| ClawEval 195 manifest | 无法评测遗忘度 |
| verl Hydra defaults | @孙豪 | `validate_config` 报缺字段 |

依赖未齐仍可做形态 smoke（dummy 模型 + 假数据），验证 §3 的三个 GPU-only 点。

---

## 附录：集群提交

### 前置条件

| 依赖 | 位置 | 状态 |
|------|------|------|
| 代码 | AFS: `/mnt/afs_toolcall/sunhao4/workspace/agentic_cl_research` | ✅ `dev_train` |
| 模型权重 | AFS: `/mnt/afs_agents/share_models/Qwen/Qwen3.6-27B` | 脚本自动 cp 到 `/tmp/qwen36` |
| Judge 端点 | `configs/agents.yaml` → sufy `deepseek-v4-pro-202606` | ✅ |
| 腾讯/E2B 凭证 | `docker/sandbox/tencent.env` | ✅ |
| 镜像 | `registry.cn-tj-01.sensecore.cn/ccr-zuhu2026/qwen36-lightllm:1.0` | ✅ 2026-07-03 迁新租户 |

### 平台提交

| 字段 | 值 |
|------|-----|
| 启动命令 | `bash /mnt/afs_toolcall/sunhao4/workspace/agentic_cl_research/scripts/train <TOPO> --config <CFG>` |
| 节点数 | 8 |
| 每节点 GPU | 8（共 64 卡） |
| 镜像 | 见上表 |

脚本按 `RANK` 区分角色：`RANK=0` → ray head + 训练 + ray stop；`RANK≠0` → ray worker --block。

统一入口 ``scripts/train``（2026-07-24 重构，替换散落的 train_4gpu/8gpu/16gpu/32gpu/64gpu.sh）：
```bash
# 9B baseline（16 卡）
bash scripts/train 16gpu --config configs/run/b1_9b_16gpu.yaml

# 27B 正式训练（64 卡）
bash scripts/train 64gpu --config configs/run/b1.yaml

# dev 机测试（4 卡，自动检测）
bash scripts/train --smoke --config configs/phase1/smoke_1step.yaml
```

``start_train.sh`` → ``scripts/train`` 的 symlink（向后兼容集群旧 Job 配置）。

### 脚本内部流程

```
1. source .env + tencent.env    → SUFY/SWANLAB + E2B 凭证
2. useradd sunhao4              → AFS 权限
3. pip install 兜底              → 镜像已含则秒过
4. cp 模型到 /tmp/qwen36        → 避 AFS 带宽抢占
5. torch.distributed barrier    → 等所有节点就绪
6. RANK=0: ray head + 训练; RANK≠0: ray worker --block
```

### 实验对照

| 实验 | 配置 | KL | Replay | Agentic Rollout | 说明 |
|------|------|:--:|:------:|:---------------:|------|
| **B1** | `configs/run/b1.yaml` | ❌ | ❌ | ❌ | 纯 RL 遗忘基线 |
| **R4** | `configs/run/r4.yaml` | ❌ | ✅ (λ=0.5) | ✅ (e2b) | 9 桶 + 抗遗忘 |

B1 与 R4 **共用同一套 agentic 多轮 rollout**（差异仅在 CL 项），否则拿多轮 R4 和单轮 B1 比遗忘不公平。

配置继承链：`ppo_trainer.yaml` → `base.yaml` → `cluster.yaml` → 各实验 yaml

### 注意事项

- **Judge**：`anthropic/claude-4.8-opus`（thinking 模型），`max_tokens=4096`（须给足，否则 `judge_error=1.0`）。judge 不可达时 reward 全零但不报错——务必确认日志 `judge_error=0`。
- **Mock Judge**：`configs/agents.yaml` 把 `reward.model` 改回 `mock-judge` 即回退。
- **AFS 限制**：`HF_HOME`/`HF_DATASETS_CACHE`/Triton 编译缓存指向 `/tmp`（AFS 不支持 `fcntl.flock`）。

### 提交 Checklist

提交前：`git pull` → `.env` 含 `SUFY_API_KEY` → `tencent.env` 含 E2B 凭证 → 模型权重存在 → `curl sufy /v1/models` 返回 200

提交后：rank0 日志有 `ray status` → 无 `judge_error=1.0` → step 0 产出 `actor/pg_loss` → swanlab 可见 loss 曲线
