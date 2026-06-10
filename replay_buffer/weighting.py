"""Token-level weight w_t for L_replay.

Final formula (W2 main scheme, from ``doc/CL_Update_Sunhao.md``):

    w_t^{(i)} = normalize( clip( priority_i * ( gamma^block(t) + delta^(K_i - block(t)) ) / 2,
                                 q_5, q_95 ) )

Two independent dimensions composed:
A. Priority (trajectory-level)        -- one value per trajectory
B. U-shaped block weight (token-level coarse) -- (gamma^block(t) + delta^(K_i - block(t))) / 2
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

Block segmentation -- message-block PRIMARY, equal-length FALLBACK:
    PRIMARY (message_block): The dataset
    (datasets/_stage_prefix_pass.jsonl) is OpenAI chat format -- each
    rollout is a list of {role, content, tool_calls?} messages. An action
    block = one assistant message (content + optional tool_calls) OR one
    tool message (observation). K_i = number of response messages in the
    trajectory.

    Empirically confirmed on 50 records (3177 assistant msgs): 0 contained
    any <think>/<toolcall>/<final_answer> XML tag; 78% had structured
    `tool_calls` field. So tag-based segmentation does not apply here --
    we segment by message boundaries instead.

    FALLBACK (equal_length): When the trajectory cannot be parsed into
    messages (e.g. flat tokenized response) OR K_i == 1 (single block =
    no U-shape possible), fall back to equal-length K=20 splits. No error
    raised, no sample dropped.

    Long-block re-splitting: a single message exceeding a token threshold
    (default 100) is sub-divided into equal-length sub-blocks. Sub-blocks
    inherit the parent block's U-shape weight (no micro-U within a block).
    This keeps a 500-token assistant turn from out-weighting a 30-token
    tool call.

Two scheme variants for Phase 3 ablation:
- W0: the SAME formula with gamma = delta = 1 (the U-shape degenerates to a
      flat line, so only priority + clip + normalize remain). W0 is therefore
      just W2 with the U-shape turned off -- it still keeps the trajectory
      priority dimension. See doc/CL_Update_Sunhao.md "gamma=delta=1 的等价关系".
- W2: above formula with gamma, delta < 1 (active U-shape).

There is no separate uniform code path: ``scheme='W0'`` simply pins
gamma = delta = 1 and runs the W2 pipeline. To get a TRULY priority-free
uniform replay weight, set the buffer's ``priority_type: uniform`` (R3), which
makes every priority 1.0 so W0 collapses to a flat 1/total weight.
"""

from __future__ import annotations

from collections.abc import Sequence


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Inclusive linear-interpolated percentile on a pre-sorted list. q in [0, 1]."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = q * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


class TokenWeighting:
    """Compute per-token replay weights.

    Args:
        scheme: 'W0' (uniform) or 'W2' (priority * U-shaped block weight + clip).
        gamma: first-end exponential base (default 0.88).
        delta: last-end exponential base  (default 0.88, symmetric U).
        segmenter: 'message_block' (primary, OpenAI chat boundaries) or
                   'equal_length' (fallback, K=20). Default 'message_block'.
        role_boundaries: list of chat roles whose messages are treated as
                         action blocks, e.g. ['assistant', 'tool']. Loaded
                         from configs/base.yaml's weighting.role_boundaries.
                         Ignored when segmenter='equal_length'.
        equal_length_K: number of equal-length blocks for the fallback (default 20).
        long_block_threshold: max tokens per message block before re-splitting
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
        segmenter: str = "message_block",
        role_boundaries: Sequence[str] | None = None,
        equal_length_K: int = 20,
        long_block_threshold: int = 100,
        clip_quantiles: tuple[float, float] = (0.05, 0.95),
    ):
        if scheme not in ("W0", "W2"):
            raise ValueError(f"unknown scheme {scheme!r}")
        if segmenter not in ("message_block", "equal_length"):
            raise ValueError(f"unknown segmenter {segmenter!r}")
        self.scheme = scheme
        # W0 == W2 with the U-shape disabled (gamma=delta=1). One code path.
        if scheme == "W0":
            gamma = 1.0
            delta = 1.0
        self.gamma = gamma
        self.delta = delta
        self.segmenter = segmenter
        self.role_boundaries = tuple(role_boundaries or ("assistant", "tool"))
        self.equal_length_K = equal_length_K
        self.long_block_threshold = long_block_threshold
        self.clip_quantiles = clip_quantiles

    def compute(self, replay_batch: Sequence[dict]) -> list[list[float]]:
        """Return per-token weights for each trajectory in replay_batch.

        Args:
            replay_batch: list of dicts, each with at least:
                response_token_ids: list[int] -- response tokens only (prompt excluded).
                priority: float -- trajectory-level priority.
                messages (optional): list of OpenAI chat messages for the response,
                                     each with role, content, token_span: (start, end).
                                     Required for segmenter='message_block'.

        Returns:
            List of per-token weight lists, one per trajectory.
            W0 and W2 share ONE pipeline; W0 just pins gamma=delta=1 so the
            U-shape is flat and only priority + clip + normalize remain.

        Scope: response tokens ONLY. The user request (prompt) does not
        participate in L_replay loss and is excluded from block segmentation
        and weight assignment. t=0 is the first response token.

        Pipeline:
            1. Segment each response into K_i blocks via message boundaries.
               Fall back to equal_length when K_i == 1 or parsing fails.
            2. Re-split blocks longer than long_block_threshold; sub-blocks
               inherit parent U-shape weight.
            3. Compute U-shape weight per token via _u_shaped_block_weights.
               (W0: gamma=delta=1 -> all blocks weight 1.0.)
            4. Multiply by trajectory priority.
            5. Clip to [q_5, q_95] across batch, normalize per batch.
        """
        return self._w2(replay_batch)

    def _w2(self, replay_batch: Sequence[dict]) -> list[list[float]]:
        all_weights: list[list[float]] = []
        for traj in replay_batch:
            seq_len = len(traj.get("response_token_ids") or [])
            if seq_len == 0:
                all_weights.append([])
                continue
            block_ids, k_i = self._segment(traj)
            block_ids, eff_k, parent_map, orig_k = self._resplit_long_blocks(block_ids, k_i)
            u = self._u_shaped_block_weights(block_ids, eff_k, parent_map, orig_k)
            prio = float(traj.get("priority", 1.0))
            all_weights.append([prio * wt for wt in u])

        return self._clip_and_normalize(all_weights)

    def _segment(self, trajectory: dict) -> tuple[list[int], int]:
        """Dispatch to message-block or equal-length segmenter.

        Returns:
            block_ids: list[int] of length seq_len, block index per token.
            K_i: total block count for this trajectory.
        """
        seq_len = len(trajectory["response_token_ids"])
        if self.segmenter == "message_block":
            try:
                block_ids, k_i = segment_message_blocks(trajectory, self.role_boundaries)
                if k_i >= 2:
                    return block_ids, k_i
            except (KeyError, ValueError, TypeError):
                pass
        return segment_equal_length(seq_len, self.equal_length_K)

    def _resplit_long_blocks(
        self, block_ids: list[int], k_i: int
    ) -> tuple[list[int], int, dict[int, int] | None, int]:
        """Re-split blocks longer than long_block_threshold into equal sub-blocks.

        Sub-blocks inherit the parent block's U-shape weight (block index
        is preserved at the block level via a parent map -- new ids only
        affect resolution within long messages).

        Returns a 4-tuple ``(block_ids, eff_k, parent_map, orig_k)``:
            - block_ids: per-token block index (possibly re-split).
            - eff_k: effective block count after re-splitting.
            - parent_map: new sub-block id -> parent block id, or None when no
              re-split happened. Passed explicitly to ``_u_shaped_block_weights``
              so NO state leaks across trajectories in the same batch (bug A1).
            - orig_k: the pre-resplit block count, used to keep the U-shape
              stable regardless of resolution changes.
        """
        if not block_ids:
            return block_ids, k_i, None, k_i
        # Count tokens per block
        counts: dict[int, int] = {}
        for b in block_ids:
            counts[b] = counts.get(b, 0) + 1
        # If no block is long, nothing to do.
        if all(c <= self.long_block_threshold for c in counts.values()):
            return block_ids, k_i, None, k_i

        new_ids = []
        # Parent-map: new sub-block id -> parent block id (for U weighting).
        parent_map: dict[int, int] = {}
        next_id = 0
        # Walk in original order, allocating sub-blocks contiguously.
        i = 0
        n = len(block_ids)
        while i < n:
            cur = block_ids[i]
            j = i
            while j < n and block_ids[j] == cur:
                j += 1
            run_len = j - i
            if run_len <= self.long_block_threshold:
                new_ids.extend([next_id] * run_len)
                parent_map[next_id] = cur
                next_id += 1
            else:
                # Split into ceil(run_len / threshold) sub-blocks.
                n_sub = (run_len + self.long_block_threshold - 1) // self.long_block_threshold
                base = run_len // n_sub
                rem = run_len % n_sub
                for s in range(n_sub):
                    sub_len = base + (1 if s < rem else 0)
                    new_ids.extend([next_id] * sub_len)
                    parent_map[next_id] = cur
                    next_id += 1
            i = j

        # New eff_k = number of sub-blocks; but each sub-block's U weight is
        # computed from its PARENT id under the ORIGINAL K_i so the tail/head
        # emphasis doesn't shift just because a single message got long.
        return new_ids, next_id, parent_map, k_i

    def _u_shaped_block_weights(
        self,
        block_ids: Sequence[int],
        k_i: int,
        parent_map: dict[int, int] | None = None,
        orig_k: int | None = None,
    ) -> list[float]:
        """(gamma^block(t) + delta^(K_i - block(t))) / 2.

        Division by 2 ensures gamma=delta=1 yields uniform weight 1.0 for all
        blocks (flat / equal-weight baseline W0). This makes gamma and delta
        continuous controls: 1.0 = no U-shape, <1.0 = progressively stronger U.

        ``parent_map`` / ``orig_k`` are passed explicitly (never read from
        instance state) so re-split bookkeeping cannot leak across
        trajectories in the same batch (bug A1).
        """
        if parent_map is not None:
            ok = orig_k if orig_k is not None else k_i
            return [
                (self.gamma ** parent_map[b] + self.delta ** (ok - 1 - parent_map[b])) / 2.0
                for b in block_ids
            ]
        return [
            (self.gamma ** b + self.delta ** (k_i - 1 - b)) / 2.0
            for b in block_ids
        ]

    def _clip_and_normalize(self, weights: list[list[float]]) -> list[list[float]]:
        """Clip to quantile range, then normalize per batch so sum = 1."""
        flat = [w for row in weights for w in row]
        if not flat:
            return weights
        sorted_flat = sorted(flat)
        q_lo, q_hi = self.clip_quantiles
        lo = _percentile(sorted_flat, q_lo)
        hi = _percentile(sorted_flat, q_hi)
        clipped = [[min(max(w, lo), hi) for w in row] for row in weights]
        total = sum(w for row in clipped for w in row)
        if total <= 0:
            return clipped
        return [[w / total for w in row] for row in clipped]


def segment_message_blocks(
    trajectory: dict, role_boundaries: Sequence[str] = ("assistant", "tool")
) -> tuple[list[int], int]:
    """Split a trajectory into action blocks by OpenAI chat message boundaries.

    Each message whose role is in ``role_boundaries`` becomes one action
    block. Assistant turns with both ``content`` and ``tool_calls`` are
    kept as a single block (one thinking + tool call decision unit).

    Args:
        trajectory: dict with at least:
            response_token_ids: list[int]
            messages: list of {role, token_span: (start, end), ...}, with
                      token_span indexing into response_token_ids.

    Returns:
        block_ids: list[int] of length seq_len, block index per token.
        K_i: total block count (>= 2 on success; == 1 triggers fallback).

    Raises:
        KeyError if messages or token_span are missing -- caller catches
        and falls back to equal-length segmentation.
    """
    messages = trajectory["messages"]  # raise KeyError -> caller falls back
    seq_len = len(trajectory["response_token_ids"])
    role_boundaries = set(role_boundaries)

    block_ids = [-1] * seq_len
    k_i = 0
    for m in messages:
        if m.get("role") not in role_boundaries:
            continue
        span = m.get("token_span")
        if span is None:
            continue
        start, end = span
        start = max(0, start)
        end = min(seq_len, end)
        if start >= end:
            continue
        for t in range(start, end):
            block_ids[t] = k_i
        k_i += 1

    # Any token not assigned (gap) inherits the nearest preceding block, or 0 if none.
    last = 0
    for t in range(seq_len):
        if block_ids[t] == -1:
            block_ids[t] = last
        else:
            last = block_ids[t]

    if k_i < 2:
        raise ValueError("K_i < 2; caller should fall back to equal-length")
    return block_ids, k_i


def segment_equal_length(seq_len: int, k: int = 20) -> tuple[list[int], int]:
    """Fallback: split a trajectory of seq_len tokens into K equal-length blocks.

    Used when message-block segmentation fails or yields K_i == 1.

    Returns:
        block_ids: list[int] of length seq_len, block index per token.
        K: total block count (== k argument, or seq_len if shorter).
    """
    if seq_len == 0:
        return [], 0
    k_eff = min(k, seq_len)
    block_size = seq_len / k_eff
    block_ids = [min(int(t / block_size), k_eff - 1) for t in range(seq_len)]
    return block_ids, k_eff
