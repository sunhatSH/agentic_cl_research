# ClawEval evaluation harness.
#
# Backend: ClawEval (300 tasks, 3 splits -- General 161 / Multimodal 101 / Multi-turn 38).
# We use ONLY the text-only subset (195 tasks). See doc/ClawEval_Metadata.md.
#
# Score formula:
#   score = s_safety * (0.8 * s_completion + 0.2 * s_robustness)
# Pass^3 standard: a task is considered passed only if all 3 independent runs pass.
#
# Files (to be added):
#   run_eval.py          -- entry: load ckpt, run 195 text tasks, score, dump report
#   metrics.py           -- New Task Perf / Old Task Forgetting / CL Score etc.
#   monitors.py          -- Output Entropy, Trajectory Diversity (see doc/CL_Update_Sunhao.md
#                           "评测指标体系" section)
