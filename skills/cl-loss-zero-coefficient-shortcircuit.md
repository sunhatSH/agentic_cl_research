# Skill: CL Loss 组合实现与零系数端到端短路

## 适用场景
- 实现由多个加权项组成的复合 loss（本项目：`L_cl = λ1·L_rl + λ2·L_kl + λ3·L_replay + λ4·L_ent`），且各项在不同实验中按 ablation 开关。
- 需要保证「某项权重为 0 时不仅不计入 loss，连**对应的前向/采样/数据拼接都不发生**」（省显存、省时间、避免无谓副作用）。

## 核心步骤
1. **闭包工厂按开关编译两套实现**：`make_cl_loss(replay_enabled, lambda_replay, ...)` 在构建时就决定返回 `cl_loss_no_replay`（只跑 `ppo_loss`）还是 `cl_loss_with_replay`。`enabled = replay_enabled and lambda_replay > 0`。
2. **数据侧也短路**：`build_buffer` 在 `buffer.enabled=false` 或 `lambda_replay==0` 时直接返回 `None`——根本不实例化 buffer（省 ~25k 轨迹内存），buffer hooks 也不挂。
3. **运行时再判空**：即使 replay 分支被编译，若本 step 没有 replay 行（`is_replay.sum()==0`），也跳过 replay 前向并上报 `replay_empty=1`。
4. **固定项不参与 ablation**：`λ4`（entropy）全程固定 0.001 防 Echo Trap，不做开关；`L_kl` 用 reverse KL；`L_reg`（参数正则）弃用，权重恒 0。
5. **指标始终上报**：无论走哪个分支都写 `actor/replay_loss` / `replay_enabled` / `replay_empty`，便于对照实验观察。

## 关键约束
- **零系数 = 端到端跳过**（项目硬规则）：λ=0 的项不采样、不前向、不拼接数据。仅在 loss 里乘 0 是不够的。
- **Priority 的零权信号同样短路**：`Priority.compute` 对权重为 0 的信号不读取对应字段（避免无谓 KeyError / 计算）。
- **复合权重一致性**：`λ` 来自 `cfg.cl`；token weighting 的 `scheme` 与 buffer 的 `priority_type` 是两个独立维度，不要混用（W0 仍保留 priority；真正无 priority 用 `priority_type: uniform`）。
- **可微性**：replay 项必须从 `model_output["log_probs"]`（带梯度）取，不能用 detached 预计算。

## 代码锚点
- `trainer/cl_loss.py`：`make_cl_loss`、`cl_loss_no_replay` / `cl_loss_with_replay`、`_replay_is_empty`、`compute_replay_loss`。
- `trainer/cl_main.py`：`build_buffer`（buffer 级短路）。
- `replay_buffer/priority.py`：`Priority.compute`（信号级短路 + `active_signals`）。
- 测试：`tests/test_cl_loss.py`（`test_make_cl_loss_zero_replay_uses_no_replay_branch`、`test_cl_loss_replay_empty_skips_forward`）、`tests/test_priority.py`（`test_zero_alpha_short_circuits_signal`）。
