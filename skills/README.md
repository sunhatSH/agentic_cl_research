# Skills — 可复用方法与规范

本目录沉淀本项目（Continual Learning over Agentic LLM）在实现过程中提炼的、**可跨实验/跨项目复用**的方法与工程规范。每个 skill 文档结构统一：

- **适用场景**：什么时候该用这个方法
- **核心步骤**：怎么做
- **关键约束**：不能违反的硬规则（违反即出 bug）
- **代码锚点**：对应实现的文件 / 函数 / 测试

| Skill | 主题 |
|-------|------|
| [nine-bucket-replay-buffer.md](nine-bucket-replay-buffer.md) | 9 桶 Replay Buffer 设计（quota / priority / 两级采样 / 淘汰 / 持久化） |
| [verl-noninvasive-loss-injection.md](verl-noninvasive-loss-injection.md) | verl 无侵入自定义 loss 注入（不 fork，`set_loss_fn` + hooks，含 fully-async） |
| [cl-loss-zero-coefficient-shortcircuit.md](cl-loss-zero-coefficient-shortcircuit.md) | CL Loss 组合实现与零系数端到端短路 |
| [experiment-yaml-conventions.md](experiment-yaml-conventions.md) | 实验 yaml 规范（OmegaConf 继承 + verl Hydra key path + 全量校验） |
| [claweval-forgetting-metrics.md](claweval-forgetting-metrics.md) | ClawEval 评测与遗忘度量（manifest 接口 / Pass^N / CL Score） |
