# Configs for the 20 CL ablation experiments.
#
# Layout:
#   base.yaml       -- shared defaults; do not run directly
#   b1.yaml         -- Phase 1 baseline (pure RL forgetting lower bound)
#   k1..k5.yaml,
#   k2-r.yaml       -- Phase 2 KL ablation (6 runs)
#   r0,r3,r4,r5,
#   r4-w,r6,r4-k    -- Phase 3 Replay ablation (7 runs)
#   c1..c4.yaml     -- Phase 4 KL x Replay grid (4 runs)
#   s1,s2.yaml      -- Phase 5 rollout scale (2 runs)
#   x*.yaml         -- Phase 6 on-demand exploration (created when triggered)
#
# Each leaf config inherits base.yaml and overrides only the parameters that
# vary. The parameter table is in doc/CL_Update_Sunhao.md "实验参数组合设计".

# Skeleton tracking (filled in as configs are created):
# Phase 1: [ ] b1
# Phase 2: [ ] k1 [ ] k2 [ ] k3 [ ] k4 [ ] k5 [ ] k2-r
# Phase 3: [ ] r0 [ ] r3 [ ] r4 [ ] r5 [ ] r4-w [ ] r6 [ ] r4-k
# Phase 4: [ ] c1 [ ] c2 [ ] c3 [ ] c4
# Phase 5: [ ] s1 [ ] s2
