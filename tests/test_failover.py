"""Tests for the two-level (provider x model) FailoverChatClient."""

from __future__ import annotations

import json

import httpx
import pytest

from agents.config import ResolvedEndpoint, ResolvedProvider, ResolvedRole
from agents.failover import AllEndpointsFailed, FailoverChatClient


def _ep(model: str) -> ResolvedEndpoint:
    return ResolvedEndpoint(base_url="http://x/v1", model=model, api_key="k", temperature=0.0)


def _role(providers, rotate_every=0):
    return ResolvedRole(role="test", providers=providers, rotate_every=rotate_every)


def _http_error(status: int) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "http://x/v1/chat/completions")
    resp = httpx.Response(status, request=req)
    return httpx.HTTPStatusError("boom", request=req, response=resp)


class _ScriptedClient:
    """Stands in for OpenAIChatClient; each call pops the next scripted outcome."""

    def __init__(self, model, outcomes):
        self.model = model
        self._outcomes = list(outcomes)
        self.calls = 0
        self.max_tokens_seen: list[int] = []  # budget received per call (escalation)

    def chat(self, messages, *, max_tokens=512):
        self.calls += 1
        self.max_tokens_seen.append(max_tokens)
        out = self._outcomes.pop(0) if self._outcomes else "ok:" + self.model
        if isinstance(out, Exception):
            raise out
        return out


def _patch_clients(monkeypatch, mapping):
    """Make FailoverChatClient build _ScriptedClient(model, outcomes) per model.

    Returns a dict {model: _ScriptedClient} so tests can inspect calls / budgets.
    """
    import agents.failover as fo

    created: dict[str, _ScriptedClient] = {}

    def _factory(*, base_url, model, api_key, temperature):
        c = _ScriptedClient(model, mapping.get(model, []))
        created[model] = c
        return c

    monkeypatch.setattr(fo, "OpenAIChatClient", _factory)
    return created


def test_first_model_502_switches_to_same_provider_next(monkeypatch):
    # provider vendor_a: [A(502), B(ok)] -> should return B's answer
    _patch_clients(monkeypatch, {"A": [_http_error(502)], "B": ["ok:B"]})
    role = _role([ResolvedProvider("vendor_a", [_ep("A"), _ep("B")])])
    c = FailoverChatClient(role)
    assert c.chat([]) == "ok:B"
    assert c.model == "B"  # promoted to default


def test_provider_fully_down_switches_provider(monkeypatch):
    # vendor_a: [A(502), B(502)] all down; tokenhub: [C(ok)]
    _patch_clients(
        monkeypatch,
        {"A": [_http_error(502)], "B": [_http_error(503)], "C": ["ok:C"]},
    )
    role = _role([
        ResolvedProvider("vendor_a", [_ep("A"), _ep("B")]),
        ResolvedProvider("tokenhub", [_ep("C")]),
    ])
    c = FailoverChatClient(role)
    assert c.chat([]) == "ok:C"
    assert c.model == "C"


def test_success_updates_default_next_call_starts_there(monkeypatch):
    # First call: A fails, B ok -> default becomes B. Second call: B ok directly.
    _patch_clients(monkeypatch, {"A": [_http_error(502)], "B": ["ok:B1", "ok:B2"]})
    role = _role([ResolvedProvider("vendor_a", [_ep("A"), _ep("B")])])
    c = FailoverChatClient(role)
    assert c.chat([]) == "ok:B1"
    assert c.chat([]) == "ok:B2"  # starts at B now, A not retried


def test_all_endpoints_failed_raises(monkeypatch):
    _patch_clients(monkeypatch, {"A": [_http_error(502)], "B": [_http_error(500)]})
    role = _role([ResolvedProvider("vendor_a", [_ep("A"), _ep("B")])])
    c = FailoverChatClient(role)
    with pytest.raises(AllEndpointsFailed):
        c.chat([])


def test_non_retryable_400_surfaces_immediately(monkeypatch):
    # 400 = bad request; retrying another model would not help -> re-raise as-is
    _patch_clients(monkeypatch, {"A": [_http_error(400)], "B": ["ok:B"]})
    role = _role([ResolvedProvider("vendor_a", [_ep("A"), _ep("B")])])
    c = FailoverChatClient(role)
    with pytest.raises(httpx.HTTPStatusError):
        c.chat([])


def test_disk_state_roundtrip(monkeypatch, tmp_path):
    # First client discovers B works and persists it; a fresh client reads B as default.
    state = tmp_path / "endpoint_state.json"
    _patch_clients(monkeypatch, {"A": [_http_error(502)], "B": ["ok:B", "ok:B2"]})
    role = _role([ResolvedProvider("vendor_a", [_ep("A"), _ep("B")])])
    c1 = FailoverChatClient(role, state_path=str(state))
    assert c1.chat([]) == "ok:B"
    saved = json.loads(state.read_text())
    assert saved["test"]["model_idx"] == 1  # B

    c2 = FailoverChatClient(role, state_path=str(state))
    assert c2.model == "B"  # started from persisted default, no A re-probe


def test_rotation_orthogonal_to_failover(monkeypatch):
    # rotate_every=1 advances default each call among available models.
    _patch_clients(monkeypatch, {"A": ["a1", "a2"], "B": ["b1", "b2"]})
    role = _role([ResolvedProvider("vendor_a", [_ep("A"), _ep("B")])], rotate_every=1)
    c = FailoverChatClient(role)
    r1 = c.chat([])  # rotate: mi 0->1 (B), B ok
    r2 = c.chat([])  # rotate: mi 1->0 (A), A ok
    assert {r1, r2} == {"b1", "a1"}


# --- truncation escalation: retry SAME model with doubled max_tokens ---------
# (2026-07-10) A TruncatedOutputError means the model ran out of output budget,
# not that the endpoint is down. For the REWARD judge (escalate_on_truncation=
# True) we retry the same model at 512->1024->2048->4096 before failing over.
# questioner/observer keep escalation OFF -> truncation fails over immediately.

from agents.base import TruncatedOutputError  # noqa: E402


def _trunc() -> TruncatedOutputError:
    return TruncatedOutputError("cut off by max_tokens")


def test_truncation_retries_same_model_with_doubled_budget(monkeypatch):
    # reward judge (escalate=True): A truncates twice then succeeds -> stays on A,
    # budgets 512,1024,2048.
    created = _patch_clients(monkeypatch, {"A": [_trunc(), _trunc(), "ok:A"]})
    role = _role([ResolvedProvider("vendor_a", [_ep("A"), _ep("B")])])
    c = FailoverChatClient(role, escalate_on_truncation=True)
    assert c.chat([], max_tokens=512) == "ok:A"
    assert c.model == "A"  # never failed over
    assert created["A"].max_tokens_seen == [512, 1024, 2048]
    assert created["B"].calls == 0  # B never touched


def test_truncation_exhausts_escalation_then_next_model(monkeypatch):
    # escalate=True: A truncates 4x (512,1024,2048,4096) -> give up on A, B succeeds.
    created = _patch_clients(
        monkeypatch,
        {"A": [_trunc(), _trunc(), _trunc(), _trunc()], "B": ["ok:B"]},
    )
    role = _role([ResolvedProvider("vendor_a", [_ep("A"), _ep("B")])])
    c = FailoverChatClient(role, escalate_on_truncation=True)
    assert c.chat([], max_tokens=512) == "ok:B"
    assert created["A"].max_tokens_seen == [512, 1024, 2048, 4096]  # 4 escalations
    assert created["A"].calls == 4
    assert c.model == "B"  # promoted


def test_non_truncation_error_does_not_escalate(monkeypatch):
    # A 502 -> immediately next model, NO budget doubling on A (even with escalate).
    created = _patch_clients(monkeypatch, {"A": [_http_error(502)], "B": ["ok:B"]})
    role = _role([ResolvedProvider("vendor_a", [_ep("A"), _ep("B")])])
    c = FailoverChatClient(role, escalate_on_truncation=True)
    assert c.chat([], max_tokens=512) == "ok:B"
    assert created["A"].calls == 1  # tried once, no retry
    assert created["A"].max_tokens_seen == [512]


def test_truncation_without_escalation_fails_over_immediately(monkeypatch):
    # Default (questioner/observer, escalate=False): A truncates ONCE -> straight
    # to B, no budget doubling on A.
    created = _patch_clients(monkeypatch, {"A": [_trunc()], "B": ["ok:B"]})
    role = _role([ResolvedProvider("vendor_a", [_ep("A"), _ep("B")])])
    c = FailoverChatClient(role)  # escalate_on_truncation defaults False
    assert c.chat([], max_tokens=512) == "ok:B"
    assert created["A"].calls == 1  # single attempt, no escalation
    assert created["A"].max_tokens_seen == [512]
    assert c.model == "B"
