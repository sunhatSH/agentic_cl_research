"""Rollout-side utilities: sandbox client adapter + GRPO group sampling.

This package isolates the trajectory-collection loop from the sandbox vendor
(doc/SandboxRollout.md §7): the same code runs against a LOCAL subprocess
backend (dev/CI), the Tencent Agent Runtime (``e2b``), or Alibaba 无影 AgentBay
(``aliyun``) -- selected by NAME via ``make_sandbox(backend=...)``. Backends are
decoupled from the loop through an open registry (``register_backend``), so a new
cloud vendor is a one-call addition, not a rollout-loop edit.
"""
