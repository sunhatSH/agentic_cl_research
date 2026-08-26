"""ClawEval Hermes 评测包：Hermes 跑 agent + 官方 grader 评分。

方案 B：训练/评测用 Hermes(verl rollout)跑 agent，评分用官方 ClawEval grader。
- grade.py: 离线评分入口（读 rollout_status → 官方 grader → Pass^3）
- vendor/: 官方 claw_eval 的 grader + models + tasks（拷贝自 qinshilong/claw-eval）
"""
