# 项目进度

> 本文件是**交付状态**的单一来源。协作流程与分工见 [`ContinualLearning.md`](ContinualLearning.md)；技术设计见 [`CL_Update_Sunhao.md`](CL_Update_Sunhao.md)。

**最后更新：** 2026-06-09  
**当前分支：** `dev_train`  
**verl pin：** `0.8.0`（见 `pyproject.toml`）

---

## 里程碑总览

| ID | 里程碑 | 状态 | 说明 |
|----|--------|------|------|
| M0 | 设计文档 | Done | Loss、7 桶、GPU 调度、实验路线、verl 集成路径 |
| M1 | Replay Buffer v1 | Done | 内存 backend，与 verl 解耦，52+ 单测 |
| M2 | verl 训练集成 | **Done (code)** | `CLTaskRunner` / `inject_cl_loss` / buffer hooks；GPU smoke pending |
| M3 | B1 基线训练 | Blocked | 依赖 GPU 集群 + 完整 verl Hydra 配置 + 模型路径 |
| M4 | Phase 2–5 消融 | Blocked | 依赖 M3 |
| M5 | ClawEval 评测闭环 | Blocked | 依赖 checkpoint + @杨益博 manifest |

---

## 模块完成度

| 模块 | 进度 | 状态 |
|------|------|------|
| `replay_buffer/` | 100% | Done |
| `trainer/cl_loss.py` | 100% | Done — `make_cl_loss` + `compute_replay_loss` (TensorDict 预计算) |
| `trainer/cl_main.py` | 100% | Done — 入口 `run_cl_ppo` |
| `trainer/verl_runner.py` | 100% | Done — `CLTaskRunner`, hooks, `build_trainer` |
| `trainer/trajectory_adapter.py` | 100% | Done |
| `trainer/replay_batch.py` | 100% | Done |
| `eval/metrics.py` | 100% | Done |
| `eval/run_eval.py` | 40% | 框架有，rollout / manifest 未接 |
| `configs/` (20 yaml) | 100% | Done |
| `tests/` | ~95% | 63 用例（含 cl_loss / adapter / verl smoke skip） |

---

## 20 实验状态

| Phase | 实验 | 配置 | 训练 | 评测 |
|-------|------|------|------|------|
| 1 | B1 | `configs/phase1/b1.yaml` | 未跑 | 未跑 |
| 2 | K1–K5, K2-R | `configs/phase2/` | 未跑 | 未跑 |
| 3 | R0, R3–R6, R4-w, R4-K | `configs/phase3/` | 未跑 | 未跑 |
| 4 | C1–C4 | `configs/phase4/` | 未跑 | 未跑 |
| 5 | S1, S2 | `configs/phase5/` | 未跑 | 未跑 |
| 6 | X* | 按需 | — | — |

**Checkpoint 产出：** 0

---

## verl 集成验证清单

| 项 | 状态 | 备注 |
|----|------|------|
| verl 版本 pin | Done | `verl==0.8.0`（Python ≥3.10；本机用 uv 建 `.venv`） |
| verl 实际安装 + 导入 | Done | `.venv` (CPython 3.10.20)；`ppo_loss` / `set_loss_fn` / `DataProto.concat` 全部校验存在 |
| 全部引用符号校验 | Done | `main_ppo` / `ray_trainer` / `tensordict_utils` 等 9 模块 ALL-OK |
| `make_cl_loss` no-replay 分支 | Done | `tests/test_cl_loss.py` |
| `compute_replay_loss` 梯度正确 | Done | 改为从 `model_output["log_probs"]` 取 replay 行（带梯度），非 detached 预计算 |
| replay 行拼接 (`_append_replay_rows`) | Done | `DataProto.concat`，`is_replay` mask + 零 advantage |
| buffer 不进 loss 闭包 | Done | 闭包 freevars 仅 `lambda_replay`，cloudpickle 1.2KB |
| `CLTaskRunner` + `set_loss_fn` | Done | `trainer/verl_runner.py` |
| `install_buffer_hooks` | Done | patch `_update_actor`，无全局 monkey-patch |
| GPU 1–2 step smoke | **Pending** | `pytest -m gpu`；replay 行的 tokenize/DataProto 形态需 GPU 联调 |

---

## 外部依赖阻塞

| 依赖 | 负责人 | 阻塞项 |
|------|--------|--------|
| 完整 verl Hydra 配置 | @孙豪 | `configs/base.yaml` 需与 verl `ppo_trainer` defaults merge |
| ClawEval 195 任务 manifest | @杨益博 | `eval/run_eval.py` |
| 沙盒 rollout 环境 | @郑乃榕 | 真实 trajectory |
| 用户 query / workspace | @吴健 | 训练数据源 |
| GPU 集群 | — | B1 实测 |

---

## 变更日志

| 日期 | 事件 |
|------|------|
| 2026-06-08 | 仓库骨架、设计文档迁入 `doc/` |
| 2026-06-09 | `e29fd06` — Replay Buffer 核心、20 实验配置、52 单测 |
| 2026-06-09 | M2 verl 集成代码：`CLTaskRunner`, buffer hooks, `doc/Progress.md` |
| 2026-06-09 | verl 0.8.0 装入 `.venv` (py3.10)；修复 replay 梯度(P0)/buffer 序列化(P1)/全局 patch(P2)/权重对齐(P4)；64 passed |

---

## 下一步行动

| 优先级 | 行动 | 负责人 |
|--------|------|--------|
| P0 | GPU smoke：B1 跑 1–2 step，验证 FSDP + replay sidecar | @孙豪 |
| P0 | merge verl 完整 Hydra defaults 到实验 yaml | @孙豪 |
| P1 | `compute_log_prob` 联调，补全 replay 前向 | @孙豪 |
| P1 | ClawEval manifest 接入 | @杨益博 |
| P2 | Buffer 持久化 (SQLite/LMDB) | @孙豪 |
