"""Tests for replay_buffer.weighting — U-shape, segmentation, clip+norm."""

import math

import pytest

from replay_buffer.weighting import (
    TokenWeighting,
    segment_equal_length,
    segment_message_blocks,
)


def test_segment_equal_length_basic():
    block_ids, k = segment_equal_length(seq_len=20, k=4)
    assert k == 4
    assert len(block_ids) == 20
    # Each block ~5 tokens
    assert block_ids[0] == 0
    assert block_ids[5] == 1
    assert block_ids[10] == 2
    assert block_ids[15] == 3
    assert block_ids[-1] == 3


def test_segment_equal_length_clamps_k():
    # seq shorter than K -> K_eff = seq_len
    ids, k = segment_equal_length(seq_len=3, k=10)
    assert k == 3
    assert sorted(set(ids)) == [0, 1, 2]


def test_segment_message_blocks_basic():
    traj = {
        "response_token_ids": list(range(30)),
        "messages": [
            {"role": "assistant", "token_span": (0, 10)},
            {"role": "tool", "token_span": (10, 20)},
            {"role": "assistant", "token_span": (20, 30)},
        ],
    }
    ids, k = segment_message_blocks(traj)
    assert k == 3
    assert ids[5] == 0
    assert ids[15] == 1
    assert ids[25] == 2


def test_segment_message_blocks_excludes_system_user():
    traj = {
        "response_token_ids": list(range(15)),
        "messages": [
            {"role": "user", "token_span": (0, 5)},
            {"role": "assistant", "token_span": (5, 10)},
            {"role": "assistant", "token_span": (10, 15)},
        ],
    }
    ids, k = segment_message_blocks(traj, role_boundaries=("assistant", "tool"))
    assert k == 2


def test_segment_message_blocks_raises_on_single_block():
    traj = {
        "response_token_ids": list(range(10)),
        "messages": [{"role": "assistant", "token_span": (0, 10)}],
    }
    with pytest.raises(ValueError):
        segment_message_blocks(traj)


def test_w0_pins_gamma_delta_to_one():
    # W0 is W2 with the U-shape disabled -> gamma = delta = 1.0.
    tw = TokenWeighting(scheme="W0", gamma=0.5, delta=0.5)
    assert tw.gamma == 1.0
    assert tw.delta == 1.0


def test_w0_keeps_priority_and_normalizes():
    # New W0 semantics (matches doc): flat U-shape BUT priority is retained,
    # then clip + normalize. Equal priority -> uniform; unequal -> proportional.
    tw = TokenWeighting(scheme="W0", clip_quantiles=(0.0, 1.0))
    batch = [
        {"response_token_ids": [1, 2, 3], "priority": 1.0},
        {"response_token_ids": [4, 5], "priority": 1.0},
    ]
    weights = tw.compute(batch)
    flat = [w for row in weights for w in row]
    assert math.isclose(sum(flat), 1.0, abs_tol=1e-6)
    # Equal priority + flat U -> every token weight equal.
    assert all(math.isclose(w, flat[0], abs_tol=1e-9) for w in flat)


def test_w0_proportional_to_priority():
    tw = TokenWeighting(scheme="W0", clip_quantiles=(0.0, 1.0))
    batch = [
        {"response_token_ids": [1, 2], "priority": 1.0},
        {"response_token_ids": [3, 4], "priority": 3.0},
    ]
    weights = tw.compute(batch)
    # Token from the higher-priority trajectory weighs 3x the lower one.
    assert math.isclose(weights[1][0] / weights[0][0], 3.0, rel_tol=1e-6)


def test_u_shape_endpoints_higher_than_middle():
    tw = TokenWeighting(scheme="W2", gamma=0.5, delta=0.5, segmenter="equal_length",
                        equal_length_K=10, long_block_threshold=10_000,
                        clip_quantiles=(0.0, 1.0))
    batch = [{"response_token_ids": list(range(100)), "priority": 1.0}]
    weights = tw.compute(batch)
    w = weights[0]
    # Endpoint tokens (block 0 and block 9) should have higher weight than middle (block 5)
    assert w[0] > w[50]
    assert w[-1] > w[50]


def test_u_shape_symmetric_when_gamma_eq_delta():
    tw = TokenWeighting(scheme="W2", gamma=0.5, delta=0.5, segmenter="equal_length",
                        equal_length_K=10, long_block_threshold=10_000,
                        clip_quantiles=(0.0, 1.0))
    batch = [{"response_token_ids": list(range(100)), "priority": 1.0}]
    w = tw.compute(batch)[0]
    # Block 0 and block 9 should have the same weight when gamma == delta.
    assert math.isclose(w[0], w[-1], abs_tol=1e-6)


def test_gamma_delta_equals_one_yields_flat():
    """gamma=delta=1.0 should give equal U weight 1.0 across all blocks
    (degenerate to W0 modulo priority scaling)."""
    tw = TokenWeighting(scheme="W2", gamma=1.0, delta=1.0, segmenter="equal_length",
                        equal_length_K=10, long_block_threshold=10_000,
                        clip_quantiles=(0.0, 1.0))
    batch = [{"response_token_ids": list(range(100)), "priority": 1.0}]
    w = tw.compute(batch)[0]
    # All weights equal -> normalized to 1/100 each.
    assert all(math.isclose(x, 1.0 / 100, abs_tol=1e-6) for x in w)


def test_normalize_sums_to_one():
    tw = TokenWeighting(scheme="W2", gamma=0.8, delta=0.8,
                        segmenter="equal_length", equal_length_K=5)
    batch = [
        {"response_token_ids": list(range(20)), "priority": 0.5},
        {"response_token_ids": list(range(10)), "priority": 0.9},
    ]
    weights = tw.compute(batch)
    total = sum(w for row in weights for w in row)
    assert math.isclose(total, 1.0, abs_tol=1e-6)


def test_message_block_fallback_to_equal_length():
    tw = TokenWeighting(scheme="W2", segmenter="message_block",
                        equal_length_K=5, clip_quantiles=(0.0, 1.0))
    # No 'messages' key -> KeyError -> fallback to equal_length
    batch = [{"response_token_ids": list(range(20)), "priority": 1.0}]
    weights = tw.compute(batch)
    assert len(weights[0]) == 20


def test_resplit_state_does_not_leak_across_trajectories():
    """Bug A1: a long trajectory that triggers re-splitting must not distort
    the weights of a later short trajectory in the same batch."""
    short = {"response_token_ids": list(range(20)), "priority": 1.0}

    # Short trajectory computed alone (no prior re-split state).
    tw_alone = TokenWeighting(scheme="W2", segmenter="equal_length",
                              equal_length_K=5, long_block_threshold=10,
                              clip_quantiles=(0.0, 1.0))
    alone = tw_alone.compute([short])[0]

    # Same short trajectory placed AFTER a long trajectory that re-splits.
    long_traj = {"response_token_ids": list(range(100)), "priority": 1.0}
    tw_after = TokenWeighting(scheme="W2", segmenter="equal_length",
                              equal_length_K=5, long_block_threshold=10,
                              clip_quantiles=(0.0, 1.0))
    after_long, after_short = tw_after.compute([long_traj, short])

    # The short trajectory's weights must be identical regardless of ordering.
    assert len(after_short) == len(alone)
    # Recompute the short trajectory alone but normalized over both rows is not
    # comparable; instead verify it is internally symmetric (U-shape intact) and
    # not corrupted by the long trajectory's parent_map.
    assert math.isclose(after_short[0], after_short[-1], abs_tol=1e-9)
    # Middle token strictly lower than endpoints (U preserved, not flattened by leak).
    assert after_short[0] > after_short[len(after_short) // 2]
