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


@dataclass(frozen=True)
class ResolvedProvider:
    """One vendor and its ordered list of same-vendor model endpoints (inner loop)."""

    name: str
    endpoints: list[ResolvedEndpoint]


@dataclass(frozen=True)
class ResolvedRole:
    """A role's full failover pool: ordered providers (outer loop), each with
    ordered models (inner loop), plus rotate_every for anti-collapse rotation.

    The two-level failover loop walks providers in order; within a provider it
    walks endpoints in order. ``rotate_every`` (>0) additionally rotates through
    *available* endpoints every N calls for anti mode-collapse (questioner only;
    0 disables rotation).
    """

    role: str
    providers: list[ResolvedProvider]
    rotate_every: int

    def flat_endpoints(self) -> list[ResolvedEndpoint]:
        """All endpoints in failover order (provider-major, model-minor)."""
        return [ep for p in self.providers for ep in p.endpoints]


def _resolve_providers(role_cfg: dict, temperature: float) -> list[ResolvedProvider]:
    """Parse a role's ``providers:`` list into ResolvedProvider objects.

    Each provider: {name, api_base, key_env, models:[...]}. Falls back to the
    legacy single ``api_base``/``model`` or ``rotation:`` shapes when ``providers``
    is absent, so old configs keep working.
    """
    providers_cfg = role_cfg.get("providers")
    out: list[ResolvedProvider] = []

    if providers_cfg:
        for p in providers_cfg:
            base = p.get("api_base", "")
            key_env = p.get("key_env", "TOKENHUB_API_KEY")
            api_key = _resolve_key(key_env, prefix_env="TOKENHUB_API_KEY")
            eps = [
                ResolvedEndpoint(base_url=base, model=m, api_key=api_key, temperature=temperature)
                for m in (p.get("models") or [])
                if base and m
            ]
            if eps:
                out.append(ResolvedProvider(name=p.get("name", base), endpoints=eps))
        return out

    # Legacy: single api_base/model (observer/reward old shape)
    base = role_cfg.get("api_base", "")
    model = role_cfg.get("model", "")
    if base and model:
        key_env = role_cfg.get("key_env", "TOKENHUB_API_KEY")
        api_key = _resolve_key(key_env, prefix_env="TOKENHUB_API_KEY")
        ep = ResolvedEndpoint(base_url=base, model=model, api_key=api_key, temperature=temperature)
        out.append(ResolvedProvider(name=base, endpoints=[ep]))
        return out

    # Legacy: questioner rotation pool (each entry its own single-model provider)
    for entry in role_cfg.get("rotation", []) or []:
        base = entry.get("api_base", "")
        model = entry.get("model", "")
        if not (base and model):
            continue
        key_env = entry.get("key_env", "TOKENHUB_API_KEY")
        api_key = _resolve_key(key_env, prefix_env="TOKENHUB_API_KEY")
        ep = ResolvedEndpoint(base_url=base, model=model, api_key=api_key, temperature=temperature)
        out.append(ResolvedProvider(name=entry.get("name", model), endpoints=[ep]))
    return out


def resolve_role(role: str, config_path: Path | str | None = None) -> ResolvedRole:
    """Resolve any role (observer/questioner/reward) into a two-level failover pool.

    Env-var overrides take priority and produce a single-provider pool:
      - OBSERVER_API_BASE/OBSERVER_MODEL, REWARD_API_BASE/REWARD_MODEL
      - USERSIM_ENDPOINTS (JSON list) or USERSIM_API_BASE/USERSIM_MODEL (questioner)
    Otherwise the ``providers:`` (or legacy) section of agents.yaml is used.
    """
    cfg = _load_yaml(config_path)
    role_cfg = cfg.get(role, {}) or {}

    # role-specific env prefix + temperature
    prefix = {"observer": "OBSERVER", "reward": "REWARD", "questioner": "USERSIM"}.get(role, role.upper())
    default_temp = {"observer": 0.0, "reward": 0.0, "questioner": 0.9}.get(role, 0.0)
    temperature = float(os.environ.get(f"{prefix}_TEMPERATURE", "") or role_cfg.get("temperature", default_temp))
    rotate_every = int(os.environ.get("USERSIM_ROTATE_EVERY", "") or role_cfg.get("rotate_every", 0)) if role == "questioner" else 0

    # ── env overrides → single-provider pool ────────────────────────────────
    if role == "questioner":
        endpoints_raw = os.environ.get("USERSIM_ENDPOINTS", "").strip()
        if endpoints_raw:
            from agents.base import _parse_endpoints

            eps = [
                ResolvedEndpoint(base_url=e["base_url"], model=e["model"], api_key=e["api_key"], temperature=temperature)
                for e in _parse_endpoints(endpoints_raw)
            ]
            if eps:
                return ResolvedRole(role, [ResolvedProvider("env", eps)], rotate_every)
        env_base = os.environ.get("USERSIM_API_BASE", "").strip()
        env_model = os.environ.get("USERSIM_MODEL", "").strip()
        if env_base and env_model:
            ep = ResolvedEndpoint(env_base, env_model, os.environ.get("TOKENHUB_API_KEY", "").strip() or "sk-local", temperature)
            return ResolvedRole(role, [ResolvedProvider("env", [ep])], rotate_every)
    else:
        env_base = os.environ.get(f"{prefix}_API_BASE", "").strip()
        env_model = os.environ.get(f"{prefix}_MODEL", "").strip()
        if env_base and env_model:
            ep = ResolvedEndpoint(env_base, env_model, _resolve_key("TOKENHUB_API_KEY", prefix_env="TOKENHUB_API_KEY"), temperature)
            return ResolvedRole(role, [ResolvedProvider("env", [ep])], rotate_every)

    # ── config-file providers ───────────────────────────────────────────────
    providers = _resolve_providers(role_cfg, temperature)
    if not providers:
        raise RuntimeError(
            f"{role} not configured: set {prefix}_API_BASE + {prefix}_MODEL env vars, "
            f"or configure the {role} section (providers:) in configs/agents.yaml."
        )
    return ResolvedRole(role, providers, rotate_every)


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
    """Resolve the Observer's primary endpoint (first provider, first model).

    Kept for backward compatibility. New code should prefer ``resolve_role`` to
    get the full failover pool.
    """
    return resolve_role("observer", config_path).flat_endpoints()[0]


def resolve_questioner(config_path: Path | str | None = None) -> ResolvedQuestionerConfig:
    """Resolve the Questioner config (backward-compatible view).

    Returns the flattened failover pool as ``rotation`` (all endpoints in
    failover order) with ``fallback=None``. New code should prefer
    ``resolve_role("questioner")`` for the provider-grouped structure.
    """
    role = resolve_role("questioner", config_path)
    return ResolvedQuestionerConfig(
        rotation=role.flat_endpoints(),
        fallback=None,
        rotate_every=role.rotate_every,
    )


def resolve_judge(config_path: Path | str | None = None) -> ResolvedEndpoint:
    """Resolve the Reward/Judge primary endpoint (first provider, first model).

    Kept for backward compatibility. New code should prefer ``resolve_role`` to
    get the full failover pool.
    """
    return resolve_role("reward", config_path).flat_endpoints()[0]


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
        for b in labels_ok[i + 1 :]:
            if configured[a] == configured[b]:
                warnings.append(
                    f"{a} and {b} use the same model '{configured[a]}' "
                    f"(anti self-preference violated, doc §6)"
                )

    return warnings
