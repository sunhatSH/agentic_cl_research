"""Configuration loader for agents.yaml -- single source of truth for model selection.

Reads ``configs/agents.yaml`` and resolves endpoint configuration (base URL,
model name, API key) for Observer / Questioner / Reward judge. Environment
variables serve as overrides and fallbacks, preserving backward compatibility.

The yaml specifies ``key_env`` fields indicating which env var holds the API
key (keys are never stored in the yaml). Resolution priority:

    1. Direct env var override (OBSERVER_API_BASE, etc.) -- highest priority
    2. agents.yaml values + key_env -> env var for the key
    3. RuntimeError if nothing is configured

This mirrors the existing env-var-only resolution in ``agents/base.py`` and
``trainer/model_reward.py`` but makes the config file the PRIMARY source,
with env vars as the override mechanism.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "agents.yaml"

# Module-level cache: loaded once, invalidated by ``_reload_config``.
_config_cache: dict[str, Any] | None = None


def _load_yaml(path: Path | str | None = None) -> dict[str, Any]:
    """Load and return the agents.yaml dict (cached)."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    p = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    if not p.exists():
        logger.warning("agents.yaml not found at %s -- falling back to env-only resolution", p)
        _config_cache = {}
        return _config_cache
    with open(p) as f:
        _config_cache = yaml.safe_load(f) or {}
    return _config_cache


def _reload_config(path: Path | str | None = None) -> dict[str, Any]:
    """Force-reload the config (used by tests / config hot-reload)."""
    global _config_cache
    _config_cache = None
    return _load_yaml(path)


# --------------------------------------------------------------------------- #
# Resolved endpoint dataclasses                                                #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ResolvedEndpoint:
    """A fully-resolved model endpoint: base_url + model + api_key + temperature."""

    base_url: str
    model: str
    api_key: str
    temperature: float


@dataclass(frozen=True)
class ResolvedQuestionerConfig:
    """Resolved questioner config: rotation pool (possibly empty) + fallback + rotate_every."""

    rotation: list[ResolvedEndpoint]
    fallback: ResolvedEndpoint | None
    rotate_every: int


# --------------------------------------------------------------------------- #
# Resolution helpers                                                           #
# --------------------------------------------------------------------------- #


def _resolve_key(key_env: str, *, prefix_env: str | None = None) -> str:
    """Resolve an API key: try key_env first, then prefix_env, else default."""
    val = os.environ.get(key_env, "").strip()
    if val:
        return val
    if prefix_env:
        val = os.environ.get(prefix_env, "").strip()
        if val:
            return val
    return "sk-local"


def resolve_observer(config_path: Path | str | None = None) -> ResolvedEndpoint:
    """Resolve the Observer endpoint (config-first, env-override).

    Priority:
      1. OBSERVER_API_BASE / OBSERVER_MODEL / TOKENHUB_API_KEY env vars (override)
      2. configs/agents.yaml observer section (key_env for the key)
    """
    cfg = _load_yaml(config_path)
    obs_cfg = cfg.get("observer", {})

    base = os.environ.get("OBSERVER_API_BASE", "").strip() or obs_cfg.get("api_base", "")
    model = os.environ.get("OBSERVER_MODEL", "").strip() or obs_cfg.get("model", "")
    key_env = obs_cfg.get("key_env", "TOKENHUB_API_KEY")
    api_key = _resolve_key(key_env, prefix_env="TOKENHUB_API_KEY")
    temperature = float(os.environ.get("OBSERVER_TEMPERATURE", "") or obs_cfg.get("temperature", 0.0))

    if not base or not model:
        raise RuntimeError(
            "Observer not configured: set OBSERVER_API_BASE + OBSERVER_MODEL env vars, "
            "or configure the observer section in configs/agents.yaml."
        )
    return ResolvedEndpoint(base_url=base, model=model, api_key=api_key, temperature=temperature)


def resolve_questioner(config_path: Path | str | None = None) -> ResolvedQuestionerConfig:
    """Resolve the Questioner config (config-first, env-override).

    Priority:
      1. USERSIM_ENDPOINTS env var (rotation) or USERSIM_API_BASE/MODEL/KEY (single)
      2. configs/agents.yaml questioner section (rotation pool + fallback)
    """
    cfg = _load_yaml(config_path)
    q_cfg = cfg.get("questioner", {})
    rotation_cfg = q_cfg.get("rotation", [])
    fallback_cfg = q_cfg.get("fallback", {})
    rotate_every = int(os.environ.get("USERSIM_ROTATE_EVERY", "") or q_cfg.get("rotate_every", 5))
    temperature = float(os.environ.get("USERSIM_TEMPERATURE", "") or q_cfg.get("temperature", 0.9))

    # Check for env-var rotation (USERSIM_ENDPOINTS) -- overrides yaml rotation
    endpoints_raw = os.environ.get("USERSIM_ENDPOINTS", "").strip()
    if endpoints_raw:
        from agents.base import _parse_endpoints

        entries = _parse_endpoints(endpoints_raw)
        rotation = [
            ResolvedEndpoint(
                base_url=e["base_url"],
                model=e["model"],
                api_key=e["api_key"],
                temperature=temperature,
            )
            for e in entries
        ]
        return ResolvedQuestionerConfig(rotation=rotation, fallback=None, rotate_every=rotate_every)

    # Env-var single-model override (USERSIM_API_BASE / USERSIM_MODEL)
    env_base = os.environ.get("USERSIM_API_BASE", "").strip()
    env_model = os.environ.get("USERSIM_MODEL", "").strip()
    env_key = os.environ.get("TOKENHUB_API_KEY", "").strip()

    if env_base and env_model:
        # Single-model env override -> no rotation
        fallback = ResolvedEndpoint(
            base_url=env_base, model=env_model,
            api_key=env_key or "sk-local", temperature=temperature,
        )
        return ResolvedQuestionerConfig(rotation=[], fallback=fallback, rotate_every=rotate_every)

    # Config-file rotation pool
    if rotation_cfg:
        rotation: list[ResolvedEndpoint] = []
        for entry in rotation_cfg:
            base = entry.get("api_base", "")
            model = entry.get("model", "")
            key_env = entry.get("key_env", "TOKENHUB_API_KEY")
            api_key = _resolve_key(key_env, prefix_env="TOKENHUB_API_KEY")
            if base and model:
                rotation.append(ResolvedEndpoint(
                    base_url=base, model=model, api_key=api_key, temperature=temperature,
                ))
        if rotation:
            fallback = None
            if fallback_cfg:
                fb_key_env = fallback_cfg.get("key_env", "TOKENHUB_API_KEY")
                fb_base_env = fallback_cfg.get("api_base_env", "")
                fb_model_env = fallback_cfg.get("model_env", "")
                fb_base = os.environ.get(fb_base_env, "").strip()
                fb_model = os.environ.get(fb_model_env, "").strip()
                fb_key = _resolve_key(fb_key_env, prefix_env="TOKENHUB_API_KEY")
                if fb_base and fb_model:
                    fallback = ResolvedEndpoint(
                        base_url=fb_base, model=fb_model,
                        api_key=fb_key, temperature=temperature,
                    )
            return ResolvedQuestionerConfig(rotation=rotation, fallback=fallback, rotate_every=rotate_every)

    # Fallback: try the fallback section from yaml
    if fallback_cfg:
        fb_base_env = fallback_cfg.get("api_base_env", "")
        fb_model_env = fallback_cfg.get("model_env", "")
        fb_key_env = fallback_cfg.get("key_env", "TOKENHUB_API_KEY")
        fb_base = os.environ.get(fb_base_env, "").strip()
        fb_model = os.environ.get(fb_model_env, "").strip()
        fb_key = _resolve_key(fb_key_env, prefix_env="TOKENHUB_API_KEY")
        if fb_base and fb_model:
            fallback = ResolvedEndpoint(
                base_url=fb_base, model=fb_model,
                api_key=fb_key, temperature=temperature,
            )
            return ResolvedQuestionerConfig(rotation=[], fallback=fallback, rotate_every=rotate_every)

    raise RuntimeError(
        "Questioner not configured: set USERSIM_ENDPOINTS or USERSIM_API_BASE + USERSIM_MODEL "
        "env vars, or configure the questioner section in configs/agents.yaml."
    )


def resolve_judge(config_path: Path | str | None = None) -> ResolvedEndpoint:
    """Resolve the Reward/Judge endpoint (config-first, env-override).

    Priority:
      1. REWARD_API_BASE / REWARD_MODEL / TOKENHUB_API_KEY env vars (override)
      2. configs/agents.yaml reward section (key_env for the key)
    """
    cfg = _load_yaml(config_path)
    reward_cfg = cfg.get("reward", {})

    base = os.environ.get("REWARD_API_BASE", "").strip() or reward_cfg.get("api_base", "")
    model = os.environ.get("REWARD_MODEL", "").strip() or reward_cfg.get("model", "")
    key_env = reward_cfg.get("key_env", "TOKENHUB_API_KEY")
    api_key = _resolve_key(key_env, prefix_env="TOKENHUB_API_KEY")
    temperature = float(os.environ.get("REWARD_TEMPERATURE", "") or reward_cfg.get("temperature", 0.0))

    if not base or not model:
        raise RuntimeError(
            "Reward not configured: set REWARD_API_BASE + REWARD_MODEL env vars, "
            "or configure the reward section in configs/agents.yaml."
        )
    return ResolvedEndpoint(base_url=base, model=model, api_key=api_key, temperature=temperature)


# --------------------------------------------------------------------------- #
# Validation (anti self-preference check)                                      #
# --------------------------------------------------------------------------- #


def validate_model_distinctness(config_path: Path | str | None = None) -> list[str]:
    """Validate that Observer / Questioner / Reward use different models.

    Anti self-preference (doc §6): the same model observing, asking, AND grading
    would bias the process. Returns a list of warnings (empty if all OK).

    Uses resolved config (config-first + env-override) so it reflects the actual
    runtime configuration.
    """
    warnings: list[str] = []
    configured: dict[str, str] = {}  # label -> model name

    for label, resolver in [("Observer", resolve_observer), ("Reward", resolve_judge)]:
        try:
            ep = resolver(config_path)
            configured[label] = ep.model
        except RuntimeError:
            warnings.append(f"{label} not configured")

    # Questioner: check rotation pool + fallback
    try:
        q_cfg = resolve_questioner(config_path)
        q_models = [ep.model for ep in q_cfg.rotation]
        if q_cfg.fallback:
            q_models.append(q_cfg.fallback.model)
        if q_models:
            configured["Questioner"] = q_models[0]  # primary model for pairwise check
            # Also check pool vs observer/reward
            for label in ["Observer", "Reward"]:
                if label in configured and configured[label] in q_models:
                    warnings.append(
                        f"{label} model '{configured[label]}' also appears in "
                        f"Questioner rotation pool (anti self-preference warning)"
                    )
    except RuntimeError:
        warnings.append("Questioner not configured")

    # Pairwise distinct check (Observer vs Reward)
    labels_ok = list(configured.keys())
    for i, a in enumerate(labels_ok):
        for b in labels_ok[i + 1:]:
            if configured[a] == configured[b]:
                warnings.append(
                    f"{a} and {b} use the same model '{configured[a]}' "
                    f"(anti self-preference violated, doc §6)"
                )

    return warnings
