"""Rollout-side utilities: sandbox client adapter + GRPO group sampling.

This package isolates the trajectory-collection loop from the sandbox vendor
(doc/SandboxRollout.md §7): the same code runs against a LOCAL subprocess
backend (no network, for dev/CI) or the real Tencent Agent Runtime / E2B
backend (on the cluster), selected by ``make_sandbox(backend=...)``.
"""
