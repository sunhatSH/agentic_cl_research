"""Token-level weight w_t for L_replay.

Final formula (W2 main scheme, from ``doc/CL_Update_Sunhao.md``):

    w_t^{(i)} = normalize( clip( priority_i * ( gamma^block(t) + delta^(K_i - block(t)) ),
                                 q_5, q_95 ) )

Two independent dimensions composed:
A. Priority (trajectory-level)        -- one value per trajectory
B. U-shaped block weight (token-level coarse) -- gamma^block(t) + delta^(K_i - block(t))
   - First-end exponential: gamma^block(t)            (early decision points)
   - Last-end exponential:  delta^(K_i - block(t))    (final answer + adjacent blocks)
   - Middle blocks are relatively down-weighted.
   - K_i is the per-trajectory block count -- VARIES per trajectory (see below).

Then clip to [5%, 95%] quantiles and normalize so sum_t w_t = 1 within batch.

Why U-shaped (2026-06-08 reversal):
    The end of a trajectory matters in MORE than just the final_answer span --
    the blocks adjacent to the answer (summary-before-conclusion, key
    judgements before decision) are also important. A single-point boost on
    final_answer cannot cover the whole tail; the U-shape lifts the entire
    end region continuously. See doc/CL_Update_Sunhao.md L_replay section.

Block segmentation -- by ACTION BLOCK, not equal-length split:
    Trajectories are split by structural tags (e.g. <think>...</think>,
    <toolcall>...</toolcall>, <observation>...</observation>,
    <final_answer>...</final_answer>). K_i is the number of action blocks in
    trajectory i and DIFFERS across trajectories. The exact tag set and
    nesting / fallback rules are TO BE DETERMINED once real rollout data is
    available.

    Until then, ``segment_action_blocks`` raises NotImplementedError, and the
    weighting falls back to equal-length K=20 splits (controlled by the
    ``segmenter`` argument).

Two scheme variants for Phase 3 ablation:
- W0: uniform 1/(N * |tau_i|)
- W2: above formula (replaces R4-w experiment)
"""


class TokenWeighting:
    """Compute per-token replay weights.

    Args:
        scheme: 'W0' (uniform) or 'W2' (priority * U-shaped block weight + clip).
        gamma: first-end exponential base (default 0.88).
        delta: last-end exponential base  (default 0.88, symmetric U).
        segmenter: 'action_block' (TBD, requires real data) or 'equal_length'
                   (fallback, K=20). Default 'equal_length' until data is in.
        block_types: list of structural tag strings used by the action_block
                     segmenter, e.g. ['<think>', '<toolcall>', '<observation>',
                     '<final_answer>']. Loaded from configs/base.yaml's
                     weighting.block_types. Ignored when segmenter='equal_length'.
        equal_length_K: number of equal-length blocks for the fallback (default 20).
        clip_quantiles: (low, high) tuple for clipping (default (0.05, 0.95)).

    With gamma=delta=0.88, K_i=20:
        endpoints (block 0 / K):  gamma^0 + delta^K  ~= 1.078
        middle    (block K/2):    gamma^(K/2)*2      ~= 0.558
        endpoint/midpoint ratio:  ~1.93x

    Note: as K_i shrinks, the U becomes flatter under fixed gamma. K_i=4 with
    gamma=0.88 yields endpoint/midpoint ~1.04 (almost no U). For short
    trajectories the gamma may need to be lowered (e.g., 0.7) -- to be
    determined by R4-w sweeps once real data is available.
    """

    def __init__(
        self,
        scheme: str = "W2",
        gamma: float = 0.88,
        delta: float = 0.88,
        segmenter: str = "equal_length",
        block_types=None,
        equal_length_K: int = 20,
        clip_quantiles=(0.05, 0.95),
    ):
        raise NotImplementedError

    def compute(self, replay_batch):
        """Return a tensor of shape [batch, max_len] with per-token weights.

        Pipeline:
            1. segment each trajectory into K_i action blocks (or fallback).
            2. for each token, compute gamma^block(t) + delta^(K_i - block(t)).
            3. multiply by priority_i (broadcast across tokens within trajectory).
            4. clip to [q_5, q_95], normalize per batch.
        """
        raise NotImplementedError

    def _segment(self, trajectory):
        """Dispatch to action-block or equal-length segmenter.

        Returns:
            block_ids: int tensor of shape [seq_len], block index per token.
            K_i: total block count for this trajectory.
        """
        raise NotImplementedError

    def _u_shaped_block_weights(self, block_ids, K_i):
        """gamma^block(t) + delta^(K_i - block(t)).

        Symmetric when gamma == delta. Asymmetric U is allowed by setting them
        differently (e.g., delta < gamma to emphasise the tail more).
        """
        raise NotImplementedError

    def _clip_and_normalize(self, w):
        """Clip to quantile range, then normalize per batch."""
        raise NotImplementedError


def segment_action_blocks(trajectory, block_types=None):
    """Split a trajectory into action blocks by structural tags.

    TODO(post-data): implement once real rollout data is in. Open questions:
    - exact tag set: <think> / <toolcall> / <observation> / <final_answer> /
      possibly more (e.g., <plan>, <self_reflection>)
    - nesting rules: how to handle <toolcall> nested inside <think> etc.
    - degenerate cases: trajectories without any structural tags (treat as
      single block? equal-length fallback within the trajectory?)
    - long-block re-splitting: should an extremely long <think> block be
      sub-divided to keep U-shape resolution within it?

    Args:
        trajectory: tokenized trajectory with raw text or pre-parsed tag spans.
        block_types: iterable of recognised tag strings, e.g. ['<think>',
                     '<toolcall>', '<observation>', '<final_answer>']. Default
                     None means use the project default from
                     configs/base.yaml (weighting.block_types).

    Returns:
        block_ids: int tensor of shape [seq_len], block index per token.
        K_i: total block count.
    """
    raise NotImplementedError("Action-block segmentation pending real rollout data.")


def segment_equal_length(seq_len: int, K: int = 20):
    """Fallback: split a trajectory of seq_len tokens into K equal-length blocks.

    Used when ``segmenter='equal_length'`` or before action-block segmenter is
    implemented. Returns block_ids and K (== K argument).
    """
    raise NotImplementedError
