# _legacy_phases/ — 旧 phase 配置（已归档）

这些是早期按 `phaseN/` 组织的实验配置，已被 `configs/run/*.yaml` 取代
（run/ 是集群可直接跑的版本：base + cluster + 实验语义合并）。

保留原因：可回溯早期实验设定。**新实验一律用 `configs/run/`，不要用这里的。**

映射关系：
- phase1/b1 → run/b1_9b_16gpu.yaml（B1 baseline）
- phase2/k1-k5,k2-r → run/k*_9b_16gpu.yaml（KL 消融）
- phase3/r0-r6 → run/r*_9b_16gpu.yaml（buffer/replay 消融）
- phase4/c1-c4、phase5/s1-s2 → 待迁到 run/（Phase 4/5 参数由 Phase 2/3 结果定）
