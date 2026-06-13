# Configs — 21 个 CL 消融实验

按 Phase 分子目录组织，每个 yaml 继承 `base.yaml` 并只 override 变化的字段。

## 结构

```
configs/
├── base.yaml              # 共享默认配置，不直接运行
├── phase1/                # Phase 1：纯 RL 遗忘基线 (1 run)
│   └── b1.yaml
├── phase2/                # Phase 2：KL 单独验证 (6 runs)
│   ├── k1.yaml            #   弱 KL + pi_0
│   ├── k2.yaml            #   中 KL + pi_0
│   ├── k3.yaml            #   强 KL + pi_0
│   ├── k4.yaml            #   中 KL + pi_{t-1}
│   ├── k5.yaml            #   强 KL + pi_{t-1}
│   └── k2-r.yaml          #   K2 + replay (与 R4-K 对偶)
├── phase3/                # Phase 3：Replay 单独验证 (8 runs)
│   ├── r0-10k.yaml         #   CLEAR baseline (单 buffer 10k, 原论文)
│   ├── r0-25k.yaml         #   CLEAR + 25k 容量 (隔离容量 vs 桶结构)
│   ├── r3.yaml            #   BucketDesign 基础版
│   ├── r4.yaml            #   BucketDesign 完整版 (+ priority)
│   ├── r5.yaml            #   reward-based priority 对照
│   ├── r4-w.yaml          #   W2 主方案 (priority × U 形块权重)
│   ├── r6.yaml            #   高 lambda_3 = 0.8
│   └── r4-k.yaml          #   R4 + KL (与 K2-R 对偶)
├── phase4/                # Phase 4：KL × Replay 组合 (4 runs)
│   ├── c1.yaml            #   强 KL + 强 Replay
│   ├── c2.yaml            #   弱 KL + 强 Replay
│   ├── c3.yaml            #   强 KL + 弱 Replay
│   └── c4.yaml            #   弱 KL + 弱 Replay
├── phase5/                # Phase 5：Rollout 规模扩展 (2 runs)
│   ├── s1.yaml            #   1024 queries × 8 traj
│   └── s2.yaml            #   4096 queries × 8 traj
└── phase6/                # Phase 6：按需探索 (x*.yaml, 后续创建)
    └── .gitkeep
```

## 使用方式

```bash
# 单实验
bash scripts/train.sh configs/phase3/r4.yaml

# 整个 phase
bash scripts/phase3/run.sh

# phase 内单个实验
bash scripts/phase3/run.sh --only r4

# 恢复训练
bash scripts/train.sh configs/phase3/r4.yaml --resume-from ckpts/r4-step-50
```

## 配置继承

所有 phase yaml 通过 `defaults: [../base]` 继承 `base.yaml`。每个 yaml 只写出与 base 不同的字段。

参数来源见 `doc/CL_Update_Sunhao.md` 的「实验参数组合设计」表。

## 进度追踪

- Phase 1: [x] b1
- Phase 2: [x] k1 [x] k2 [x] k3 [x] k4 [x] k5 [x] k2-r
- Phase 3: [x] r0 [x] r3 [x] r4 [x] r5 [x] r4-w [x] r6 [x] r4-k
- Phase 4: [x] c1 [x] c2 [x] c3 [x] c4
- Phase 5: [x] s1 [x] s2
- Phase 6: (按需创建)

> [x] = yaml 骨架已创建；参数中的 `???` 待前序 Phase 结果确定后填入。
