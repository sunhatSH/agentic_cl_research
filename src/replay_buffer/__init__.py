"""Replay Buffer for Continual Learning.

9-bucket experience replay buffer, decoupled from verl. See
``doc/BucketDesign.md`` for the design rationale and ``doc/CL_Update_Sunhao.md``
for the loss-side integration.

Public surface:
    BucketReplayBuffer  -- top-level buffer object
    Priority            -- per-trajectory priority computation
    TwoLevelSampler     -- bucket-level + intra-bucket sampling
    TokenWeighting      -- token-level w_t (priority * (gamma^block + delta^(K_i-block)) / 2)
"""

from replay_buffer.bucket import BucketReplayBuffer
from replay_buffer.priority import Priority
from replay_buffer.sampler import TwoLevelSampler
from replay_buffer.weighting import TokenWeighting

__all__ = ["BucketReplayBuffer", "Priority", "TwoLevelSampler", "TokenWeighting"]
