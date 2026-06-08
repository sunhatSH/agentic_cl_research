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

Block segmentation -- action-block PRIMARY, equal-length FALLBACK:
    PRIMARY: Trajectories are split by structural tags (e.g.  此外...完成,
    <toolcall>...</toolcall>, <observation>...</observation>,
    <final_answer>...</final_answer>). K_i is the number of action blocks in
    trajectory i and DIFFERS across trajectories.

    FALLBACK: When the trajectory cannot be parsed into action blocks (no
    structural tags, malformed tags) OR parsing yields K_i == 1 (single block
    = no U-shape possible), the segmenter automatically falls back to
    equal-length K=20 splits. No error raised, no sample dropped -- any
    format gets weights.

    Long-block re-splitting: a single action block exceeding a token threshold
    (default 100) is sub-divided into equal-length sub-blocks. Sub-blocks
    inherit the parent block's U-shape weight (no micro-U within a block).
    This prevents a 500-token  此外 block and a 30-token <toolcall> block
    from having the same resolution.

    The exact tag set and nesting / fallback rules are TO BE DETERMINED once
    real rollout data is available. Until then, ``segment_action_blocks``
    raises NotImplementedError, and the weighting falls back to equal-length
    K=20 splits.

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
        segmenter: 'action_block' (primary, TBD) or 'equal_length'
                   (fallback, K=20). Default 'equal_length' until data is in.
        block_types: list of structural tag strings used by the action_block
                     segmenter, e.g. ['此外', '<toolcall>', '<observation>',
                     '<final_answer>']. Loaded from configs/base.yaml's
                     weighting.block_types. Ignored when segmenter='equal_length'.
        equal_length_K: number of equal-length blocks for the fallback (default 20).
        long_block_threshold: max tokens per action block before re-splitting
                              (default 100). Sub-blocks inherit parent weight.
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
        long_block_threshold: int = 100,
        clip_quantiles=(0.05, 0.95),
    ):
        raise NotImplementedError

    def compute(self, replay_batch):
        """Return a tensor of shape [batch, max_response_len] with per-token weights.

        Scope: response tokens ONLY. The user request (prompt) does not
        participate in L_replay loss and is excluded from block segmentation
        and weight assignment. t=0 is the first response token.

        Pipeline:
            1. Try segment_action_blocks on each trajectory's RESPONSE.
               If K_i >= 2, use action-block segmentation.
               If K_i == 1 or parsing fails, fall back to segment_equal_length.
            2. For long blocks (> long_block_threshold tokens), re-split into
               equal-length sub-blocks inheriting the parent's U-shape weight.
            3. For each response token, compute gamma^block(t) + delta^(K_i - block(t)).
            4. Multiply by priority_i (broadcast across tokens within trajectory).
            5. Clip to [q_5, q_95], normalize per batch.
        """
        raise NotImplementedError

    def _segment(self, trajectory):
        """Dispatch to action-block or equal-length segmenter.

        Tries action-block first; falls back to equal-length when:
          - no structural tags found (unparseable trajectory)
          - tags are malformed / incomplete
          - K_i == 1 (single block = no U-shape possible)

        Returns:
            block_ids: int tensor of shape [seq_len], block index per token.
            K_i: total block count for this trajectory.
        """
        raise NotImplementedError

    def _resplit_long_blocks(self, block_ids, token_counts_per_block, K_i):
        """Re-split blocks exceeding long_block_threshold into equal sub-blocks.

        Sub-blocks inherit the parent block's U-shape weight. No micro-U
        within a block -- the purpose is resolution, not additional weighting.

        Args:
            block_ids: per-token block assignments.
            token_counts_per_block: list of token counts for each block.
            K_i: original block count.

        Returns:
            new_block_ids: updated per-token block assignments.
            new_K_i: updated block count after re-splitting.
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

    Returns K_i >= 2 on success. Caller should fall back to
    segment_equal_length when this returns K_i == 1 or raises.

    TODO(post-data): implement once real rollout data is in. Open questions:
    - exact tag set:  此外 / <toolcall> / <observation> / <final_answer> /
      possibly more (e.g., <plan>, <self_reflection>)
    - nesting rules: how to handle <toolcall> nested inside  此外 etc.
    - degenerate cases: trajectories without any structural tags
    - long-block re-splitting threshold

    Args:
        trajectory: tokenized trajectory with raw text or pre-parsed tag spans.
        block_types: iterable of recognised tag strings, e.g. ['此外',
                     '<toolcall>', '<observation>', '<final_answer>']. Default
                     None means use the project default from
                     configs/base.yaml (weighting.block_types).

    Returns:
        block_ids: int tensor of shape [seq_len], block index per token.
        K_i: total block count (>= 2 on success; == 1 triggers fallback).
    """
    raise NotImplementedError("Action-block segmentation pending real rollout data.")


def segment_equal_length(seq_len: int, K: int = 20):
    """Fallback: split a trajectory of seq_len tokens into K equal-length blocks.

    Used when action-block segmentation fails or yields K_i == 1.
    Returns block_ids and K (== K argument).
    """
    raise NotImplementedError
