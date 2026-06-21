"""Tests for the user-sim three-agent pipeline (doc/UserSim_多轮Query在线生成.md).

All off-network: agents take mock chat clients; the session driver uses a mock
SessionSandboxPool agent_fn. Verifies prompt assembly, JSON parsing, the patience
mechanism (§3.6.5), persona library (§3.5), and the Algorithm 1 session flow.
"""

from __future__ import annotations

import random

import pytest

from agents.observer import Observer, parse_observation_report
from agents.personas import PERSONAS, sample_persona
from agents.prompts import (
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


# --- patience mechanism (§3.6.5) --------------------------------------------


def test_patience_decays_exponentially():
    # §3.6.5: P_k = P_0 - d_0*(2^k - 1). P0=1.0, d0=0.1
    # cumulative decrement 2^k-1 = 1,3,7,15 -> P1=0.9, P2=0.7, P3=0.3, P4=-0.5
    persona = Persona("t", "", "", "", "detail × content", patience=1.0, patience_decay=0.1)
    pt = PatienceTracker(persona, random.Random(0))
    pt.fail_count = 1
    assert pt.current_patience() == pytest.approx(0.9)
    pt.fail_count = 2
    assert pt.current_patience() == pytest.approx(0.7)
    pt.fail_count = 3
    assert pt.current_patience() == pytest.approx(0.3)
    pt.fail_count = 4
    assert pt.current_patience() == pytest.approx(-0.5)


def test_patience_self_caps_around_three_failures():
    # With P0~1, d0=0.1, patience goes negative by the 4th failure (P4=-0.5)
    # -> forced stop, reproducing the original "≤3 hard cap" as emergent (§3.6.5).
    persona = Persona("t", "", "", "", "x", patience=1.0, patience_decay=0.1)
    pt = PatienceTracker(persona, random.Random(0))
    redos = sum(1 for _ in range(20) if pt.on_failure())
    # never more than a handful of redos before patience is exhausted
    assert redos <= 4
    assert pt.current_patience() < 0


def test_patience_high_tolerance_persona_redos_more():
    patient = Persona("p", "", "", "", "x", patience=1.6, patience_decay=0.07)
    impatient = Persona("i", "", "", "", "x", patience=0.6, patience_decay=0.4)
    # deterministic seed; patient persona should survive more failures
    pp = PatienceTracker(patient, random.Random(1))
    ip = PatienceTracker(impatient, random.Random(1))
    pp_redos = sum(1 for _ in range(10) if pp.on_failure())
    ip_redos = sum(1 for _ in range(10) if ip.on_failure())
    assert pp_redos >= ip_redos


# --- reward (§6 / §7.4 / O4) -------------------------------------------------


class MockJudge:
    def __init__(self, verdict):
        self.verdict = verdict
        self.last_rubric = None

    def score(self, *, task, trajectory, rubric, data_source):
        self.last_rubric = rubric
        self.last_trajectory = trajectory
        return self.verdict


def test_reward_uses_observation_report_as_evidence():
    report = ObservationReport(
        final=[{"path": "out.csv", "kind": "csv", "content_excerpt": "a,b,c"}],
        discrepancies="claimed 100 rows but file has 3",
    )
    report.actor_trajectory = "[assistant] ran the tool"  # pass-through channel
    judge = MockJudge({"completion": 0.4, "safety": 1.0, "robustness": 0.8})
    out = score_followup(query="recheck the totals", report=report, judge=judge)
    # ClawEval aggregation: safety*(0.8*completion + 0.2*robustness)
    assert out["score"] == pytest.approx(1.0 * (0.8 * 0.4 + 0.2 * 0.8))
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
    assert out["score"] == 0.0 and out["completion"] == 0.0 and out.get("gated") == 1.0


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
