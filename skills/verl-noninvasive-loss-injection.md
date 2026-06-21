# Skill: verl 无侵入自定义 Loss 注入

## 适用场景
- 需要在 verl（RayPPOTrainer 或 Fully Async Policy）上叠加**自定义 loss**（如 CL 的 replay 项），但**不想 fork verl 源码**（便于升级、便于多人协作）。
- 需要把外部状态（如 replay buffer）挂到 trainer 上，但**不能让它进入会被 pickle 到 worker 的 loss 闭包**。

## 核心步骤
1. **Loss 注入走 `actor_rollout_wg.set_loss_fn(loss_fn)`**：在 `trainer.init_workers()` 之后调用。`loss_fn(config, model_output, data, dp_group)` 内部先调 verl 的 `ppo_loss` 得到 RL+KL+entropy，再叠加自定义项。
2. **自定义前向数据走「拼行」而非「预计算」**：把 replay 行**追加**到 actor 训练 batch（`DataProto.concat`），让它们经过**同一次 forward**产生带梯度的 `log_probs`，loss 里再按 `is_replay` mask 选出来加权。这样自定义项对当前策略可微。
3. **双 mask 分流**：自定义行的 verl `response_mask` 置 0（让 `ppo_loss` 自然忽略，不污染 RL 分母/KL/entropy），另用 `replay_response_mask` 驱动自定义项。
4. **buffer hooks 用方法包装**：`install_buffer_hooks` 包装 `trainer._update_actor`——pre 阶段拼接 replay 行、post 阶段把本 step 轨迹写回 buffer 并同步 step。不做全局 monkey-patch。
5. **Fully Async Policy**：trainer 是 `@ray.remote` actor，无法从 driver patch；改为**子类化其底层类后重新 `@ray.remote`**（`FullyAsyncTrainer.__ray_metadata__.modified_class`），override `_fit_update_actor` + 在 `init_workers` 里 `set_loss_fn`。

## 关键约束
- **不 fork verl**：只走 `set_loss_fn` + 方法包装；引用 verl 符号前先校验存在（`importlib.util.find_spec`）。
- **buffer 绝不进 loss 闭包**：`make_cl_loss` 不接收 buffer 参数；闭包 freevars 只允许标量/配置，保证 cloudpickle 体积极小且不把 driver-only 对象传到 worker。
- **Ray ActorClass 不能直接被继承**：必须继承底层 plain class 再 `ray.remote(...)` 重新装饰，否则 `ActorClassInheritanceException`。
- **形态对齐**：拼行前把 RL batch 与 replay 行**右 pad 到同一 seq_len**，否则 `DataProto.concat` 形状不匹配。
- **占位字段补齐**：replay 行需补 `old_log_probs`/`ref_log_prob`/`advantages` 零占位，避免开启 KL 时 KeyError。

## 代码锚点
- `trainer/verl_runner.py`：`inject_cl_loss`、`install_buffer_hooks`、`_append_replay_rows`、`CLTaskRunner`、`build_trainer`。
- `trainer/verl_async_runner.py`：`make_cl_fully_async_trainer_cls`、`cl_actor_update`（纯逻辑，可单测）。
- `trainer/cl_loss.py`：`make_cl_loss`（按 `is_replay` 分流）、`compute_replay_loss`。
- `trainer/replay_forward.py`：`build_replay_rows`（双 mask）、`pad_rows_to_seq_len`、`select_replay_rows`。
- 测试：`tests/test_verl_smoke.py`（含 H800 上的 CUDA replay smoke）、`tests/test_async_runner.py`、`tests/test_cl_loss.py`。
