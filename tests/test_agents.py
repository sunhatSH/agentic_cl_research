"""Tests for the user-sim three-agent pipeline (doc/UserSim_多轮Query在线生成.md).

All off-network: agents take mock chat clients; the session driver uses a mock
SessionSandboxPool agent_fn. Verifies prompt assembly, JSON parsing, the patience
mechanism (§3.6.5), persona library (§3.5), the Algorithm 1 session flow,
RotatingChatClient (anti mode-collapse), and persona tone injection.
"""

from __future__ import annotations

import random
import threading

import pytest

from agents.base import OpenAIChatClient, RotatingChatClient, _parse_endpoints
from agents.observer import Observer, parse_observation_report
from agents.personas import PERSONAS, sample_persona
from agents.prompts import (
    _TONE_GUIDANCE,
    build_observer_prompt,
    build_questioner_prompt,
    build_reward_judge_input,
)
from agents.questioner import END_SESSION, PatienceTracker, Questioner
from agents.reward import score_followup
from agents.schema import ObservationReport, Persona


class MockChat:
    """Returns a fixed reply, records the messages it was called with."""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[list[dict]] = []

    def chat(self, messages, *, max_tokens: int = 512) -> str:
        self.calls.append(messages)
        return self.reply


# --- personas (§3.5 / §7.5) --------------------------------------------------


def test_persona_library_is_valid():
    assert len(PERSONAS) == 42
    assert len({p.name for p in PERSONAS}) == len(PERSONAS)
    for p in PERSONAS:
        assert p.patience > 0
        assert p.patience_decay > 0
        assert p.observation_focus
        assert p.tone in {"calm", "neutral", "hot"}


def test_sample_persona_is_seeded():
    a = sample_persona(random.Random(0))
    b = sample_persona(random.Random(0))
    assert a.name == b.name


# --- observer (§7.3 / O6) ----------------------------------------------------


def test_observer_parses_json_report():
    raw = (
        '{"intermediate": [{"desc": "sum", "source": "calc.py", "value_excerpt": "42"}], '
        '"final": [{"path": "out.csv", "kind": "csv", "content_excerpt": "a,b"}], '
        '"actor_claims": "wrote out.csv", "discrepancies": "", "file_tree": "out.csv"}'
    )
    # use_llm=True: opt into the model path (default observer is deterministic).
    # The observer is STATE-only -> observe() takes no trajectory.
    obs = Observer(client=MockChat(raw), use_llm=True)
    report = obs.observe()
    assert report.final[0]["path"] == "out.csv"
    assert report.intermediate[0]["value_excerpt"] == "42"
    assert not report.is_empty()


def test_observer_falls_back_on_bad_json():
    obs = Observer(client=MockChat("not json at all"), use_llm=True)
    # bad JSON -> minimal report; must not crash. Trajectory is carried PASS-THROUGH
    # on the report but is NEVER put into the observer prompt (no token waste).
    report = obs.observe(
        actor_trajectory=[{"role": "assistant", "content": "I did stuff"}],
        post={"fs": {"a.txt": {"size": 1, "mtime": 1.0, "ext": ".txt", "text": "x"}}},
    )
    assert "I did stuff" in report.actor_trajectory  # pass-through carried
    assert "a.txt" in report.file_tree
    # the observer LLM prompt must NOT contain the trajectory
    prompt_user = obs._client.calls[-1][1]["content"] if hasattr(obs._client, "calls") else ""
    assert "I did stuff" not in prompt_user


def test_observer_deterministic_report_no_llm():
    # Default (use_llm=False): forensics-only, NO model call; final filled from diff.
    class Boom:
        def chat(self, messages, *, max_tokens=512):
            raise AssertionError("observer must not call the LLM when use_llm=False")

    obs = Observer(client=Boom())
    pre = {"fs": {}, "sys": {}}
    post = {"fs": {"./out.csv": {"size": 3, "mtime": 2.0, "ext": ".csv", "text": "a,b"}}, "sys": {}}
    report = obs.observe(baseline=pre, post=post)
    assert any(f["path"] == "./out.csv" for f in report.final)
    assert "out.csv" in report.state_diff and "a,b" in report.state_diff


def test_parse_observation_report_extracts_embedded_json():
    text = 'prefix {"discrepancies": "x", "final": [], "intermediate": []} suffix'
    report = parse_observation_report(text)
    assert report.discrepancies == "x"


# --- questioner (§7.4 / O3) --------------------------------------------------


def test_questioner_returns_query_text():
    q = Questioner(client=MockChat("The total on page 3 looks off, can you recheck it?"))
    persona = PERSONAS[0]
    report = ObservationReport(final=[{"path": "r.xlsx", "kind": "xlsx", "content_excerpt": "..."}])
    out = q.next_query(persona, report, [{"role": "user", "content": "make a report"}])
    assert out is not None and "page 3" in out


def test_questioner_end_session():
    q = Questioner(client=MockChat(END_SESSION))
    out = q.next_query(PERSONAS[0], ObservationReport(final=[{"path": "x"}]), [])
    assert out is None
    assert not q.last_query_was_error


def test_questioner_api_error_distinguished():
    """API failure sets last_query_was_error=True; <end_session> sets it False."""

    class BoomChat:
        def chat(self, messages, *, max_tokens=512):
            raise RuntimeError("API down")

    q = Questioner(client=BoomChat())
    out = q.next_query(PERSONAS[0], ObservationReport(final=[{"path": "x"}]), [])
    assert out is None
    assert q.last_query_was_error is True

    # Now a satisfied end — must reset the flag
    q2 = Questioner(client=MockChat(END_SESSION))
    out2 = q2.next_query(PERSONAS[0], ObservationReport(final=[{"path": "x"}]), [])
    assert out2 is None
    assert q2.last_query_was_error is False


def test_questioner_truncation_is_error_not_end_session():
    """A truncated reply (thinking model hit max_tokens) is an ERROR, not
    a satisfied '<end_session>'. Without the guard an empty/truncated reply
    would silently end the session."""

    class TruncChat:
        def chat(self, messages, *, max_tokens=512):
            from agents.base import TruncatedOutputError

            raise TruncatedOutputError("finish_reason=length")

    q = Questioner(client=TruncChat())
    out = q.next_query(PERSONAS[0], ObservationReport(final=[{"path": "x"}]), [])
    assert out is None
    # Truncation MUST be flagged as an error (patience/telemetry path), not a
    # satisfied end, otherwise a thinking-model session silently aborts.
    assert q.last_query_was_error is True


# --- patience mechanism (§3.6.5) --------------------------------------------


def test_patience_decays_exponentially():
    # §3.6.5: P_k = P0 * r^k (r = retention rate close to 1). P0=10, r=0.7
    # P1=7, P2=4.9, P3=3.43, P4=2.401
    persona = Persona("t", "", "", "", "detail × content", patience=10.0, patience_decay=0.7)
    pt = PatienceTracker(persona, random.Random(0))
    pt.fail_count = 1
    assert pt.current_patience() == pytest.approx(7.0)
    pt.fail_count = 2
    assert pt.current_patience() == pytest.approx(4.9)
    pt.fail_count = 3
    assert pt.current_patience() == pytest.approx(3.43)
    pt.fail_count = 4
    assert pt.current_patience() == pytest.approx(2.401)


def test_patience_self_caps_around_three_failures():
    # Low patience P0=3, r=0.45: P_k drops below the 0.1 kill threshold quickly
    # -> forced stop after a few failures (§3.6.5 kill threshold = 0.1).
    persona = Persona("t", "", "", "", "x", patience=3.0, patience_decay=0.45)
    pt = PatienceTracker(persona, random.Random(0))
    redos = sum(1 for _ in range(20) if pt.on_failure())
    # never more than a handful of redos before patience falls below 0.1
    assert redos <= 5
    assert pt.current_patience() <= 0.1


def test_patience_high_tolerance_persona_redos_more():
    patient = Persona("p", "", "", "", "x", patience=10.0, patience_decay=0.9)
    impatient = Persona("i", "", "", "", "x", patience=3.0, patience_decay=0.45)
    # deterministic seed; patient persona should survive more failures
    pp = PatienceTracker(patient, random.Random(1))
    ip = PatienceTracker(impatient, random.Random(1))
    pp_redos = sum(1 for _ in range(20) if pp.on_failure())
    ip_redos = sum(1 for _ in range(20) if ip.on_failure())
    assert pp_redos >= ip_redos


# --- reward (§6 / §7.4 / O4) -------------------------------------------------


class MockJudge:
    def __init__(self, verdict, traj_verdict=None):
        self.verdict = verdict
        self.traj_verdict = traj_verdict if traj_verdict is not None else verdict
        self.last_rubric = None
        self.last_trajectory = None
        self.last_traj_rubric = None

    def score(self, *, task, trajectory, rubric, data_source, system=None):
        if system is not None:
            self.last_traj_rubric = rubric
            return self.traj_verdict
        self.last_rubric = rubric
        self.last_trajectory = trajectory
        return self.verdict


def test_reward_uses_observation_report_as_evidence():
    report = ObservationReport(
        final=[{"path": "out.csv", "kind": "csv", "content_excerpt": "a,b,c"}],
        discrepancies="claimed 100 rows but file has 3",
    )
    report.actor_trajectory = "[assistant] ran the tool"  # pass-through channel
    judge = MockJudge(
        {"task_done": 1, "correctness": 0.4, "safety": 1.0},
        traj_verdict={"tool": 1.0, "efficiency": 1.0, "planning": 0.8, "consistency": 1.0, "recovery": 0.5},
    )
    out = score_followup(query="recheck the totals", report=report, judge=judge)
    # trajectory = 0.20*tool + 0.20*efficiency + 0.25*planning + 0.25*consistency + 0.10*recovery
    traj = 0.20 * 1.0 + 0.20 * 1.0 + 0.25 * 0.8 + 0.25 * 1.0 + 0.10 * 0.5
    assert out["trajectory"] == pytest.approx(traj)
    # aggregation: done -> (0.4*correctness + 0.4*trajectory + 0.2) * safety
    assert out["score"] == pytest.approx((0.4 * 0.4 + 0.4 * traj + 0.2) * 1.0)
    # state evidence (discrepancies / report) made it into the judge rubric
    assert "claimed 100 rows" in judge.last_rubric
    # the actor trajectory reaches the judge via the report's pass-through field
    assert "ran the tool" in judge.last_trajectory


def test_reward_judge_error_is_surfaced():
    class Boom:
        def score(self, **kw):
            raise RuntimeError("judge down")

    # has_effect default True -> judge is called -> error surfaced
    out = score_followup(query="q", report=ObservationReport(final=[{"path": "x"}]), judge=Boom())
    assert out["judge_error"] == 1.0
    assert out["score"] == 0.0


def test_reward_gated_on_no_effect_skips_judge():
    # Layer of interception: an empty-diff turn (has_effect False) must NOT call
    # the judge at all -- score 0 for free.
    class Boom:
        def score(self, **kw):
            raise AssertionError("judge must not be called when has_effect is False")

    out = score_followup(query="q", report=ObservationReport(has_effect=False), judge=Boom())
    assert out["score"] == 0.0 and out["task_done"] == 0.0 and out.get("gated") == 1.0


# --- prompt assembly ---------------------------------------------------------


def test_prompts_inject_their_inputs():
    persona = PERSONAS[4]
    report = ObservationReport(final=[{"path": "report.xlsx"}], discrepancies="page 3 empty")

    obs_msgs = build_observer_prompt(state_diff="+ ADDED report.xlsx", file_tree="report.xlsx")
    assert "report.xlsx" in obs_msgs[1]["content"]
    assert "OBJECTIVE" in obs_msgs[0]["content"]

    q_msgs = build_questioner_prompt(persona=persona, report=report, session_history=[])
    assert persona.profession in q_msgs[1]["content"]
    assert "page 3 empty" in q_msgs[1]["content"]
    assert END_SESSION in q_msgs[0]["content"]

    report.actor_trajectory = "[assistant] did it"  # pass-through channel on the report
    r_in = build_reward_judge_input(query="recheck", report=report)
    assert "page 3 empty" in r_in["rubric"]
    assert r_in["task"] == "recheck"
    # trajectory reaches reward via the report's pass-through field
    assert "did it" in r_in["trajectory"]


# --- RotatingChatClient (anti mode-collapse) ----------------------------------


class RecordingChat:
    """Records calls and returns the model name (so we can tell who answered).

    Has a ``model`` attribute to match the ``OpenAIChatClient`` interface
    used by ``RotatingChatClient.current_model``.
    """

    def __init__(self, model_name: str):
        self.model = model_name
        self.call_count = 0

    def chat(self, messages, *, max_tokens: int = 512) -> str:
        self.call_count += 1
        return f"reply from {self.model}"

    def chat_with_tools(self, messages, *, tools=None, max_tokens: int = 1024) -> dict:
        self.call_count += 1
        return {"role": "assistant", "content": f"tool-reply from {self.model}"}


def test_rotating_client_single_model():
    """With one client, rotation is a no-op — always the same model."""
    c = RecordingChat("m1")
    rc = RotatingChatClient([c], rotate_every=3)
    for _ in range(10):
        assert "m1" in rc.chat([])


def test_rotating_client_cycles_through_models():
    """With rotate_every=2, every 2 calls switch to the next model."""
    c1, c2, c3 = RecordingChat("m1"), RecordingChat("m2"), RecordingChat("m3")
    rc = RotatingChatClient([c1, c2, c3], rotate_every=2)
    # Calls 1-2 -> m1, calls 3-4 -> m2, calls 5-6 -> m3, calls 7-8 -> m1 (wrap)
    results = [rc.chat([]) for _ in range(8)]
    assert results[0] == "reply from m1"
    assert results[1] == "reply from m1"
    assert results[2] == "reply from m2"
    assert results[3] == "reply from m2"
    assert results[4] == "reply from m3"
    assert results[5] == "reply from m3"
    assert results[6] == "reply from m1"  # wraps
    assert results[7] == "reply from m1"


def test_rotating_client_current_model():
    """current_model tracks the active model."""
    c1, c2 = RecordingChat("m1"), RecordingChat("m2")
    rc = RotatingChatClient([c1, c2], rotate_every=3)
    assert rc.current_model == "m1"
    for _ in range(3):
        rc.chat([])
    assert rc.current_model == "m2"
    for _ in range(3):
        rc.chat([])
    assert rc.current_model == "m1"  # wraps


def test_rotating_client_requires_at_least_one():
    with pytest.raises(ValueError, match="at least one"):
        RotatingChatClient([], rotate_every=5)


def test_rotating_client_chat_with_tools():
    """chat_with_tools also rotates."""
    c1, c2 = RecordingChat("m1"), RecordingChat("m2")
    rc = RotatingChatClient([c1, c2], rotate_every=2)
    r1 = rc.chat_with_tools([], tools=[{"type": "function", "function": {"name": "x"}}])
    assert "m1" in r1["content"]
    r2 = rc.chat_with_tools([], tools=[{"type": "function", "function": {"name": "x"}}])
    assert "m1" in r2["content"]
    # After 2 calls, should rotate to m2
    r3 = rc.chat_with_tools([], tools=[{"type": "function", "function": {"name": "x"}}])
    assert "m2" in r3["content"]


def test_rotating_client_thread_safety():
    """Concurrent calls don't corrupt the rotation counter."""
    c1, c2 = RecordingChat("m1"), RecordingChat("m2")
    rc = RotatingChatClient([c1, c2], rotate_every=5)
    results: list[str] = []
    lock = threading.Lock()

    def worker():
        for _ in range(50):
            r = rc.chat([])
            with lock:
                results.append(r)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # All 200 calls completed without crash; both models were used
    assert len(results) == 200
    assert any("m1" in r for r in results)
    assert any("m2" in r for r in results)


# --- _parse_endpoints ---------------------------------------------------------


def test_parse_endpoints_json_array():
    raw = (
        '[{"base_url":"http://a/v1","model":"model-a","api_key":"key-a"},'
        '{"base_url":"http://b/v1","model":"model-b","api_key":"key-b"}]'
    )
    entries = _parse_endpoints(raw)
    assert len(entries) == 2
    assert entries[0] == {"base_url": "http://a/v1", "model": "model-a", "api_key": "key-a"}
    assert entries[1] == {"base_url": "http://b/v1", "model": "model-b", "api_key": "key-b"}


def test_parse_endpoints_missing_key_gets_default():
    raw = '[{"base_url":"http://a/v1","model":"model-a"}]'
    entries = _parse_endpoints(raw)
    assert entries[0]["api_key"] == "sk-local"


def test_parse_endpoints_empty_key_gets_default():
    raw = '[{"base_url":"http://a/v1","model":"model-a","api_key":""}]'
    entries = _parse_endpoints(raw)
    assert entries[0]["api_key"] == "sk-local"


def test_parse_endpoints_not_json():
    with pytest.raises(ValueError, match="JSON array"):
        _parse_endpoints("not json at all")


def test_parse_endpoints_not_array():
    with pytest.raises(ValueError, match="non-empty JSON array"):
        _parse_endpoints('{"base_url":"http://a/v1","model":"m"}')


def test_parse_endpoints_empty_array():
    with pytest.raises(ValueError, match="non-empty JSON array"):
        _parse_endpoints("[]")


def test_parse_endpoints_missing_base_url():
    with pytest.raises(ValueError, match="missing 'base_url' or 'model'"):
        _parse_endpoints('[{"model":"m","api_key":"k"}]')


def test_parse_endpoints_missing_model():
    with pytest.raises(ValueError, match="missing 'base_url' or 'model'"):
        _parse_endpoints('[{"base_url":"http://a/v1","api_key":"k"}]')


def test_parse_endpoints_entry_not_object():
    with pytest.raises(ValueError, match="must be an object"):
        _parse_endpoints('["not-an-object"]')


# --- resolve_questioner_client with rotation (env-level) ----------------------


def test_resolve_questioner_client_fallback_no_endpoints(monkeypatch):
    """When USERSIM_ENDPOINTS is not set, falls back to single model."""
    monkeypatch.delenv("USERSIM_ENDPOINTS", raising=False)
    monkeypatch.setenv("USERSIM_API_BASE", "http://localhost:8100/v1")
    monkeypatch.setenv("USERSIM_MODEL", "test-model")
    monkeypatch.setenv("USERSIM_API_KEY", "sk-test")
    from agents.base import resolve_questioner_client
    from agents.failover import FailoverChatClient

    client = resolve_questioner_client()
    # Now a FailoverChatClient (single-endpoint pool from the env override).
    assert isinstance(client, FailoverChatClient)
    assert client.model == "test-model"


def test_resolve_questioner_client_rotation(monkeypatch):
    """When USERSIM_ENDPOINTS is set, returns a FailoverChatClient over the pool."""
    monkeypatch.setenv(
        "USERSIM_ENDPOINTS",
        '[{"base_url":"http://a/v1","model":"model-a","api_key":"key-a"},'
        '{"base_url":"http://b/v1","model":"model-b","api_key":"key-b"}]',
    )
    monkeypatch.setenv("USERSIM_ROTATE_EVERY", "3")
    from agents.base import resolve_questioner_client
    from agents.failover import FailoverChatClient

    client = resolve_questioner_client()
    assert isinstance(client, FailoverChatClient)
    assert client._rotate_every == 3
    # env override -> one "env" provider holding both endpoints
    assert sum(len(p) for p in client._clients) == 2


# --- Persona tone injection ---------------------------------------------------


def test_tone_guidance_injected_in_questioner_prompt():
    """Each tone value produces distinct guidance in the system prompt."""
    for tone_key, guidance_text in _TONE_GUIDANCE.items():
        p = Persona("t", "", "", "", "x", patience=1.0, patience_decay=0.1, tone=tone_key)
        msgs = build_questioner_prompt(persona=p, report=ObservationReport(), session_history=[])
        system = msgs[0]["content"]
        assert guidance_text in system, f"tone={tone_key}: expected guidance in system prompt"
        # persona block includes the tone field
        user = msgs[1]["content"]
        assert f"tone: {tone_key}" in user


def test_tone_hot_makes_prompt_different_from_calm():
    hot = Persona("t", "", "", "", "x", patience=1.0, patience_decay=0.1, tone="hot")
    calm = Persona("t", "", "", "", "x", patience=1.0, patience_decay=0.1, tone="calm")
    hot_msgs = build_questioner_prompt(persona=hot, report=ObservationReport(), session_history=[])
    calm_msgs = build_questioner_prompt(persona=calm, report=ObservationReport(), session_history=[])
    assert hot_msgs[0]["content"] != calm_msgs[0]["content"]
    assert "impatient" in hot_msgs[0]["content"]
    assert "measured" in calm_msgs[0]["content"]


def test_persona_tone_in_library():
    """All personas in the library have a valid tone."""
    valid = {"calm", "neutral", "hot"}
    for p in PERSONAS:
        assert p.tone in valid, f"persona {p.name!r} has invalid tone {p.tone!r}"


# --- validate_endpoints_distinct ------------------------------------------------


def test_validate_endpoints_distinct_ok(monkeypatch):
    """Three different models → no warnings."""
    monkeypatch.setenv("OBSERVER_API_BASE", "http://a/v1")
    monkeypatch.setenv("OBSERVER_MODEL", "observer-model")
    monkeypatch.setenv("USERSIM_API_BASE", "http://b/v1")
    monkeypatch.setenv("USERSIM_MODEL", "questioner-model")
    monkeypatch.setenv("REWARD_API_BASE", "http://c/v1")
    monkeypatch.setenv("REWARD_MODEL", "reward-model")
    from agents.base import validate_endpoints_distinct

    warnings = validate_endpoints_distinct()
    assert warnings == []


def test_validate_endpoints_same_model_warns(monkeypatch):
    """Same model for Observer and Reward → warning."""
    monkeypatch.setenv("OBSERVER_API_BASE", "http://a/v1")
    monkeypatch.setenv("OBSERVER_MODEL", "same-model")
    monkeypatch.setenv("USERSIM_API_BASE", "http://b/v1")
    monkeypatch.setenv("USERSIM_MODEL", "questioner-model")
    monkeypatch.setenv("REWARD_API_BASE", "http://c/v1")
    monkeypatch.setenv("REWARD_MODEL", "same-model")
    from agents.base import validate_endpoints_distinct

    warnings = validate_endpoints_distinct()
    assert any("same model" in w for w in warnings)


def test_validate_endpoints_missing_warns(monkeypatch):
    """Missing env AND config -> warning. With config-first resolution, deleting
    env vars alone is not enough (agents.yaml fills in); we must also point at
    a missing config so the yaml fallback is absent."""
    monkeypatch.delenv("OBSERVER_API_BASE", raising=False)
    monkeypatch.delenv("OBSERVER_MODEL", raising=False)
    monkeypatch.setenv("USERSIM_API_BASE", "http://b/v1")
    monkeypatch.setenv("USERSIM_MODEL", "q-model")
    monkeypatch.setenv("REWARD_API_BASE", "http://c/v1")
    monkeypatch.setenv("REWARD_MODEL", "r-model")
    from agents.base import validate_endpoints_distinct
    from agents.config import _reload_config

    # Point config at a nonexistent file so yaml resolution yields nothing
    _reload_config("/nonexistent/agents.yaml")
    try:
        warnings = validate_endpoints_distinct()
        assert any("Observer not configured" in w for w in warnings)
    finally:
        # Restore default config
        _reload_config(None)


# --- agents/config.py -- config-first resolution ------------------------------


def test_config_resolve_observer_from_yaml():
    """Observer resolves from agents.yaml without any env vars."""
    from agents.config import _reload_config, resolve_observer

    _reload_config(None)  # use default configs/agents.yaml
    try:
        ep = resolve_observer()
        assert ep.base_url == "https://tokenhub.sensetime.com/v1"
        assert ep.model == "gpt-5.4-mini"
        assert ep.temperature == 0.0
    finally:
        _reload_config(None)


def test_config_resolve_judge_from_yaml():
    """Reward judge resolves from agents.yaml without any env vars."""
    from agents.config import _reload_config, resolve_judge

    _reload_config(None)
    try:
        ep = resolve_judge()
        assert ep.base_url == "https://tokenhub.sensetime.com/v1"
        # Judge model is the first in configs/agents.yaml reward.providers[0].models.
        # Kept in sync with the yaml (single source of truth); update both if changed.
        assert ep.model == "gpt-5.6-luna"
        assert ep.temperature == 0.0
    finally:
        _reload_config(None)


def test_config_resolve_questioner_from_yaml():
    """Questioner rotation pool resolves from agents.yaml."""
    from agents.config import _reload_config, resolve_questioner

    _reload_config(None)
    try:
        q_cfg = resolve_questioner()
        assert len(q_cfg.rotation) == 3
        # Rotation pool mirrors configs/agents.yaml questioner.providers[0].models.
        # All reliable NON-thinking models — thinking models (deepseek/kimi) were
        # removed after iter9 showed they over-truncate on long prompts and cause
        # questioner_error via chained failover (2026-07-10 Iter10).
        assert q_cfg.rotation[0].model == "qwen3.7-max"
        assert q_cfg.rotation[1].model == "qwen3.6-plus"
        assert q_cfg.rotation[2].model == "gpt-5.4-mini"
        assert q_cfg.rotate_every == 5
    finally:
        _reload_config(None)


def test_config_env_overrides_yaml(monkeypatch):
    """Env vars take priority over agents.yaml values."""
    from agents.config import _reload_config, resolve_observer

    monkeypatch.setenv("OBSERVER_API_BASE", "http://override/v1")
    monkeypatch.setenv("OBSERVER_MODEL", "override-model")
    _reload_config(None)
    try:
        ep = resolve_observer()
        assert ep.base_url == "http://override/v1"
        assert ep.model == "override-model"
    finally:
        _reload_config(None)


def test_config_missing_yaml_falls_to_env(monkeypatch):
    """When agents.yaml is missing, env-only resolution still works."""
    from agents.config import _reload_config, resolve_observer

    monkeypatch.setenv("OBSERVER_API_BASE", "http://env-only/v1")
    monkeypatch.setenv("OBSERVER_MODEL", "env-model")
    _reload_config("/nonexistent/agents.yaml")
    try:
        ep = resolve_observer()
        assert ep.base_url == "http://env-only/v1"
        assert ep.model == "env-model"
    finally:
        _reload_config(None)


def test_config_validate_distinct_from_yaml():
    """validate_model_distinctness works with config-first resolution."""
    from agents.config import _reload_config, validate_model_distinctness

    _reload_config(None)
    try:
        warnings = validate_model_distinctness()
        # agents.yaml has distinct models -> no self-preference warnings
        assert not any("same model" in w for w in warnings)
    finally:
        _reload_config(None)


def test_config_key_env_resolves_from_env(monkeypatch):
    """key_env in agents.yaml points to TOKENHUB_API_KEY for the shared API key."""
    from agents.config import _reload_config, resolve_observer

    monkeypatch.setenv("TOKENHUB_API_KEY", "my-secret-key")
    _reload_config(None)
    try:
        ep = resolve_observer()
        assert ep.api_key == "my-secret-key"
    finally:
        _reload_config(None)


# --- Iter4: structured has_red_flag verdict (2026-07-10) ---------------------
# Regression lock for the "boilerplate opener suppresses a real flag" bug: the
# observer often writes "No empty deliverables detected. One discrepancy is
# present: ..." and a pure negative-substring filter wrongly dropped it.


def test_is_real_red_flag_positive_phrase_beats_boilerplate_opener():
    from agents.prompts import _is_real_red_flag

    # Reassuring opener THEN a concrete problem -> still a red flag.
    assert _is_real_red_flag(
        "No empty deliverables detected. One discrepancy is present: totals disagree."
    )
    assert _is_real_red_flag(
        "No clear internal contradictions. The only red flag visible is a truncated file."
    )
    # Pure "nothing found" note -> NOT a flag.
    assert not _is_real_red_flag("No clear discrepancies detected from the diff alone.")
    assert not _is_real_red_flag("No explicit discrepancies detected. None detected.")
    assert not _is_real_red_flag("")


def test_deterministic_report_sets_has_red_flag_on_empty_deliverable():
    from agents.observer import build_deterministic_report

    diff = {
        "added": [{"path": "./out.json", "kind": "text", "size": 0, "content_excerpt": ""}],
        "modified": [],
        "removed": [],
    }
    rep = build_deterministic_report(diff=diff, file_tree="out.json", state_diff="+ ADDED out.json")
    assert rep.has_red_flag is True
    assert "empty file" in rep.discrepancies


def test_deterministic_report_clean_has_no_red_flag():
    from agents.observer import build_deterministic_report

    diff = {
        "added": [{"path": "./notes.txt", "kind": "text", "size": 12, "content_excerpt": "hello world"}],
        "modified": [],
        "removed": [],
    }
    rep = build_deterministic_report(diff=diff, file_tree="notes.txt", state_diff="+ ADDED notes.txt")
    assert rep.has_red_flag is False
    assert rep.discrepancies == ""


def test_parse_report_trusts_explicit_has_red_flag_true():
    text = '{"final": [], "intermediate": [], "discrepancies": "totals disagree", "has_red_flag": true}'
    rep = parse_observation_report(text)
    assert rep.has_red_flag is True


def test_parse_report_derives_flag_when_boolean_missing():
    # No has_red_flag key -> derive from text; boilerplate-then-problem must flag.
    text = (
        '{"final": [], "intermediate": [], '
        '"discrepancies": "No empty deliverables detected. One discrepancy is present: X."}'
    )
    rep = parse_observation_report(text)
    assert rep.has_red_flag is True


def test_parse_report_clean_note_no_flag():
    text = '{"final": [], "intermediate": [], "discrepancies": "No clear discrepancies detected."}'
    rep = parse_observation_report(text)
    assert rep.has_red_flag is False


def test_finalize_red_flag_overrides_false_negative():
    from agents.observer import _finalize_red_flag

    rep = ObservationReport(
        discrepancies="No empty deliverables detected. One discrepancy is present: mismatch.",
        has_red_flag=False,
    )
    _finalize_red_flag(rep)
    assert rep.has_red_flag is True


def test_finalize_red_flag_downgrades_false_positive():
    # iter4 data: model set has_red_flag=True while its own text is unambiguously
    # clean ("No concrete red flags detected... no empty deliverables or conflicting
    # values were observed"). The TEXT verdict is authoritative -> downgrade to False.
    from agents.observer import _finalize_red_flag

    rep = ObservationReport(
        discrepancies=(
            "No concrete red flags detected in the provided evidence. The aligned "
            "diagram files exist, and the deprecated files contain non-empty content. "
            "No empty deliverables or conflicting values were observed."
        ),
        has_red_flag=True,
    )
    _finalize_red_flag(rep)
    assert rep.has_red_flag is False


def test_finalize_red_flag_empty_text_keeps_boolean():
    # No text to judge -> leave the (rare deterministic) True as-is.
    from agents.observer import _finalize_red_flag

    rep = ObservationReport(discrepancies="", has_red_flag=True)
    _finalize_red_flag(rep)
    assert rep.has_red_flag is True


def test_questioner_banner_keys_off_has_red_flag():
    # has_red_flag=True with terse discrepancy text -> banner appears.
    report = ObservationReport(
        final=[{"path": "r.json"}],
        discrepancies="totals disagree across files",
        has_red_flag=True,
    )
    msgs = build_questioner_prompt(persona=PERSONAS[0], report=report, session_history=[])
    user = msgs[1]["content"]
    assert "UNRESOLVED RED FLAG" in user
    assert "totals disagree" in user


def test_questioner_no_banner_when_no_flag():
    report = ObservationReport(
        final=[{"path": "r.json"}],
        discrepancies="No clear discrepancies detected.",
        has_red_flag=False,
    )
    msgs = build_questioner_prompt(persona=PERSONAS[0], report=report, session_history=[])
    user = msgs[1]["content"]
    assert "UNRESOLVED RED FLAG" not in user


# --- final-field schema: surfaced text answer must be a dict, not a bare str ---
# Regression for the Iter2 bug where fs-empty QA turns put the raw answer string
# into final, breaking every downstream f["path"] consumer (2026-07-10).


def test_final_is_list_of_dicts_for_text_only_answer():
    obs = Observer(client=None)  # use_llm defaults False -> deterministic path
    pre = {"fs": {}, "sys": {}}
    post = {"fs": {}, "sys": {}}  # no fs change -> surface actor reply
    traj = [
        {"role": "user", "content": "does the module expose onReady?"},
        {"role": "assistant", "content": "Yes, onAppReady is exposed in main.js."},
    ]
    report = obs.observe(baseline=pre, post=post, actor_trajectory=traj)
    assert isinstance(report.final, list)
    for item in report.final:
        assert isinstance(item, dict), f"final item must be dict, got {type(item)}"
        assert set(item.keys()) >= {"path", "kind", "content_excerpt"}
    # the answer text is carried in content_excerpt, not as a bare string
    assert report.final
    assert "onAppReady" in report.final[0]["content_excerpt"]


# --- Observer report restructure (2026-07-10): clean 3-section + full + denoise ---


def test_format_changes_three_sections_full_content_and_denoise():
    from agents.observer import _format_changes

    diff = {
        "added": [{"path": "./a.py", "kind": "text", "size": 5, "content_excerpt": "x = 1"}],
        "modified": [
            {"path": "./c.json", "kind": "text", "size": 9,
             "content_excerpt": '{"n": 2}', "before_excerpt": '{"n": 1}'}
        ],
        "removed": [{"path": "./old.md", "before_excerpt": "old body full"}],
    }
    sys_diff = {"installed_packages": ["pkg==1.0"], "opened_ports": [8888], "started_procs": ["uvicorn"]}
    out = _format_changes(diff, sys_diff)
    # three sections present
    assert "## 新增文件 (ADDED)" in out and "## 改变文件 (MODIFIED)" in out and "## 删除文件 (REMOVED)" in out
    # added full content, removed old content, modified before+after
    assert "x = 1" in out and "old body full" in out
    assert "[BEFORE]" in out and '{"n": 1}' in out and "[AFTER]" in out and '{"n": 2}' in out
    # system: package kept, ports/procs dropped
    assert "INSTALLED pkg==1.0" in out
    assert "8888" not in out and "uvicorn" not in out and "PORT LISTENING" not in out and "PROCESS" not in out


def test_snapshot_workspace_filters_runtime_files(monkeypatch):
    import agents.observer as obs

    fake = {
        "./deliverable.md": {"size": 10, "text": "hi"},
        "./.bashrc": {"size": 5, "text": "x"},
        "./AGENTS.md": {"size": 5, "text": "y"},
    }
    monkeypatch.setattr(obs, "_run_json_probe", lambda sandbox, probe: fake)
    snap = obs.snapshot_workspace(object())
    assert "./deliverable.md" in snap
    assert "./.bashrc" not in snap and "./AGENTS.md" not in snap


def test_strip_system_intermediate_removes_noise():
    from agents.observer import _strip_system_intermediate

    rep = ObservationReport(
        intermediate=[
            {"desc": "System state change: packages installed", "source": "system", "value_excerpt": "edge-tts"},
            {"desc": "running process snapshot", "source": "system", "value_excerpt": "MainThread"},
            {"desc": "intermediate file overwritten later", "source": "./tmp.txt", "value_excerpt": "draft"},
        ]
    )
    _strip_system_intermediate(rep)
    assert len(rep.intermediate) == 1
    assert rep.intermediate[0]["source"] == "./tmp.txt"


def test_removed_file_keeps_full_content_not_capped():
    from agents.observer import diff_snapshots

    long_body = "L" * 5000
    pre = {"./gone.md": {"size": 5000, "mtime": 1.0, "text": long_body}}
    post = {}
    d = diff_snapshots(pre, post)
    assert d["removed"][0]["before_excerpt"] == long_body  # not clipped to 200


# --- Iter9: questioner checks deliverable against the ORIGINAL task -----------


def test_questioner_prompt_surfaces_original_task_even_in_long_session():
    # A long history (> 12 msgs) must still expose turn-1 task at the top.
    history = [{"role": "user", "content": "补充这 4 类测试点：A、B、C、D 再复审"}]
    for i in range(20):
        history.append({"role": "assistant", "content": f"done {i}"})
        history.append({"role": "user", "content": f"followup {i}"})
    report = ObservationReport(final=[{"path": "review_v2.txt", "kind": "text", "content_excerpt": "..."}])
    msgs = build_questioner_prompt(persona=PERSONAS[0], report=report, session_history=history)
    user = msgs[1]["content"]
    assert "# Your original task" in user
    assert "4 类测试点" in user  # original task survived the window
    # system prompt carries the completeness rule
    assert "CHECK THE DELIVERABLE AGAINST YOUR ORIGINAL TASK" in msgs[0]["content"]


def test_first_user_task_extraction():
    from agents.prompts import _first_user_task

    assert _first_user_task([{"role": "user", "content": "task one"},
                             {"role": "assistant", "content": "ok"},
                             {"role": "user", "content": "task two"}]) == "task one"
    assert _first_user_task([{"role": "assistant", "content": "hi"}]) == ""
    assert _first_user_task([]) == ""
