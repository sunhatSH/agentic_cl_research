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
    obs = Observer(client=MockChat(raw))
    report = obs.observe([{"role": "assistant", "content": "done, wrote out.csv"}])
    assert report.final[0]["path"] == "out.csv"
    assert report.intermediate[0]["value_excerpt"] == "42"
    assert not report.is_empty()


def test_observer_falls_back_on_bad_json():
    obs = Observer(client=MockChat("not json at all"))
    report = obs.observe([{"role": "assistant", "content": "I did stuff"}])
    # actor text preserved even when the model output is unparseable
    assert "I did stuff" in report.actor_claims


def test_parse_observation_report_extracts_embedded_json():
    text = 'prefix {"actor_claims": "x", "final": [], "intermediate": []} suffix'
    report = parse_observation_report(text)
    assert report.actor_claims == "x"


# --- questioner (§7.4 / O3) --------------------------------------------------


def test_questioner_returns_query_text():
    q = Questioner(client=MockChat("The total on page 3 looks off, can you recheck it?"))
    persona = PERSONAS[0]
    report = ObservationReport(final=[{"path": "r.xlsx", "kind": "xlsx", "content_excerpt": "..."}])
    out = q.next_query(persona, report, [{"role": "user", "content": "make a report"}])
    assert out is not None and "page 3" in out


def test_questioner_end_session():
    q = Questioner(client=MockChat(END_SESSION))
    out = q.next_query(PERSONAS[0], ObservationReport(actor_claims="x"), [])
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
        return self.verdict


def test_reward_uses_observation_report_as_evidence():
    report = ObservationReport(
        final=[{"path": "out.csv", "kind": "csv", "content_excerpt": "a,b,c"}],
        discrepancies="claimed 100 rows but file has 3",
    )
    judge = MockJudge({"completion": 0.4, "safety": 1.0, "robustness": 0.8})
    out = score_followup(
        query="recheck the totals", report=report, trajectory="[assistant] done", judge=judge
    )
    # ClawEval aggregation: safety*(0.8*completion + 0.2*robustness)
    assert out["score"] == pytest.approx(1.0 * (0.8 * 0.4 + 0.2 * 0.8))
    # discrepancies (anti-hacking evidence) made it into the judge rubric
    assert "claimed 100 rows" in judge.last_rubric


def test_reward_judge_error_is_surfaced():
    class Boom:
        def score(self, **kw):
            raise RuntimeError("judge down")

    out = score_followup(query="q", report=ObservationReport(), trajectory="t", judge=Boom())
    assert out["judge_error"] == 1.0
    assert out["score"] == 0.0


# --- prompt assembly ---------------------------------------------------------


def test_prompts_inject_their_inputs():
    persona = PERSONAS[4]
    report = ObservationReport(actor_claims="wrote report.xlsx", discrepancies="page 3 empty")

    obs_msgs = build_observer_prompt(actor_trajectory="I wrote report.xlsx", file_tree="report.xlsx")
    assert "report.xlsx" in obs_msgs[1]["content"]
    assert "OBJECTIVE" in obs_msgs[0]["content"]

    q_msgs = build_questioner_prompt(persona=persona, report=report, session_history=[])
    assert persona.profession in q_msgs[1]["content"]
    assert "page 3 empty" in q_msgs[1]["content"]
    assert END_SESSION in q_msgs[0]["content"]

    r_in = build_reward_judge_input(query="recheck", report=report, trajectory="t")
    assert "page 3 empty" in r_in["rubric"]
    assert r_in["task"] == "recheck"
