"""Tests.

Layout:
    test_buckets.py     -- 9-bucket allocation, quota math, eviction rules
    test_priority.py    -- 4-signal priority fusion, anti-reward-monotonicity check
    test_sampler.py     -- two-level sampling distribution, starvation_boost
    test_weighting.py   -- W2 token weight: U-shaped ((gamma^block + delta^(K_i-block))/2),
                           clip+normalize. Action-block segmenter test is
                           pending real data -- equal-length K=20 fallback
                           is testable now.
    test_cl_loss.py     -- cl_loss composition (RL + KL + replay + entropy)
    test_trajectory_adapter.py -- verl batch -> buffer metadata
    test_verl_smoke.py  -- minimal smoke test that custom loss survives a verl
                           train step on toy data (validates the FSDP / grad
                           accumulation risk flagged in doc/VerlIntegration.md
                           section 4)
"""
