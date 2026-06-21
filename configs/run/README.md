# configs/run/ — 集群可运行配置

> **第三层 overlay**：在 3 层配置继承体系的顶层。`base.yaml → cluster.yaml → run/*.yaml`

## 为什么有这个目录？

项目有 3 个配置层级，每层解决不同的问题：

```
层 1:  configs/base.yaml           全局共享默认值（模型、loss 系数、buffer 参数、评测）
层 2:  configs/cluster.yaml        64 卡集群引擎层（光 llm 参数、FSDP、Ray、数据路径）
层 3:  configs/run/*.yaml          特定实验的可运行配置（合并 base + cluster + 实验语义）
```

`phase<N>/` 下的 yaml 只继承 `base.yaml`（层 1），不含集群引擎配置。`configs/run/b1.yaml` 和 `configs/run/r4.yaml` 额外引入 `cluster.yaml`（层 2）和 `_generated_ppo_trainer.yaml`，形成完整的 "可提交到集群" 配置。

## 用途

- **集群提交**：`bash scripts/run_phases.sh configs/run/b1.yaml`
- **入口 point**：`run_phases.sh` 默认使用的就是 `configs/run/{b1,r4}.yaml`
- **分离关注点**：phase yaml 只关心算法语义（KL coeff、replay weight），不关心集群参数

## 当前文件

| 文件 | 对应实验 | 说明 |
|------|----------|------|
| [`b1.yaml`](b1.yaml) | Phase 1 / B1 | 纯 RL 遗忘基线，无 KL 无 replay |
| [`r4.yaml`](r4.yaml) | Phase 3 / R4 | BucketDesign 完整版 + priority 加权 |

后续 Phase 如需提交到集群，在此创建对应的 `run/*.yaml`（复制现有模板，改 defaults + 实验字段）。
