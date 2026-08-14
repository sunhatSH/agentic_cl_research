"""Validate all 21 experiment configs load and follow the verl Hydra schema.

These checks are framework-agnostic (no verl import): they assert that our
CL overlay yamls (a) inherit base.yaml cleanly, (b) place actor/model fields
under verl's canonical key paths, (c) contain no dead `cl_grpo` loss_mode, and
(d) keep the CL/buffer/weighting sections internally consistent.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from trainer.cl_main import build_buffer, load_config

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"
# `smoke_*` configs are 4-GPU sanity probes, not formal experiments: they pin an
# absolute verl `_generated_ppo_trainer.yaml` path (cluster-only) and self-declare
# "不是正式实验配置". Exclude them from the experiment roster / schema checks.
# 旧 phaseN 配置 2026-08-13 归位到 _legacy_phases/（被 run/ 取代，仅回溯 + schema 校验）。
EXPERIMENT_CONFIGS = sorted(
    p for p in CONFIG_ROOT.glob("_legacy_phases/phase*/*.yaml") if not p.stem.startswith("smoke")
)


def test_found_all_experiment_configs():
    # 21 = B1(1) + K(6) + R(8) + C(4) + S(2)
    assert len(EXPERIMENT_CONFIGS) == 21, [p.name for p in EXPERIMENT_CONFIGS]


@pytest.mark.parametrize("cfg_path", EXPERIMENT_CONFIGS, ids=lambda p: p.stem)
def test_config_loads_and_matches_verl_schema(cfg_path):
    cfg = load_config(str(cfg_path))

    # (a) No top-level `actor:` -- actor config must live under actor_rollout_ref.
    assert "actor" not in cfg, f"{cfg_path.name}: top-level 'actor' should be actor_rollout_ref.actor"

    # (b) Actor knobs sit on verl's canonical path.
    aro = cfg.get("actor_rollout_ref")
    assert aro is not None and "actor" in aro, f"{cfg_path.name}: missing actor_rollout_ref.actor"
    assert "use_kl_loss" in aro.actor

    # (c) rollout.n = traj/query (GRPO group size).
    assert aro.rollout.n == 8, f"{cfg_path.name}: rollout.n must be 8"

    # (d) No dead cl_grpo loss_mode anywhere.
    dumped = OmegaConf.to_yaml(cfg)
    assert "cl_grpo" not in dumped, f"{cfg_path.name}: dead cl_grpo loss_mode present"

    # (e) Model path lives under actor_rollout_ref.model.path.
    assert "model" in aro and "path" in aro.model

    # (f) CL section present with a valid weighting scheme.
    cl = cfg.get("cl")
    assert cl is not None
    scheme = (cl.get("weighting", {}) or {}).get("scheme", "W2")
    assert scheme in ("W0", "W2"), f"{cfg_path.name}: bad weighting scheme {scheme!r}"


@pytest.mark.parametrize("cfg_path", EXPERIMENT_CONFIGS, ids=lambda p: p.stem)
def test_buffer_builds_when_fully_specified(cfg_path):
    cfg = load_config(str(cfg_path))
    cl = cfg.get("cl", {})
    # Skip configs whose lambda_replay / buffer.enabled are still MISSING (???).
    if OmegaConf.is_missing(cl, "lambda_replay"):
        pytest.skip("lambda_replay unresolved (Phase 4 grid)")
    buf_cfg = cl.get("buffer", {})
    if OmegaConf.is_missing(buf_cfg, "enabled"):
        pytest.skip("buffer.enabled unresolved")

    buf = build_buffer(cfg)
    enabled = buf_cfg.get("enabled", False) and float(cl.get("lambda_replay", 0.0)) > 0
    if not enabled:
        assert buf is None
    else:
        assert buf is not None
        stats = buf.stats()
        # Soft targets sum to total capacity; sizes start at 0.
        assert stats["total_size"] == 0
        assert sum(t for t in buf.soft_target.values()) == buf.total_capacity
