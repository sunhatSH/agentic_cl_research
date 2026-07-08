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

    def chat(self, messages, *, max_tokens=512):
        self.calls += 1
        out = self._outcomes.pop(0) if self._outcomes else "ok:" + self.model
        if isinstance(out, Exception):
            raise out
        return out


def _patch_clients(monkeypatch, mapping):
    """Make FailoverChatClient build _ScriptedClient(model, outcomes) per model."""
    import agents.failover as fo

    def _factory(*, base_url, model, api_key, temperature):
        return _ScriptedClient(model, mapping.get(model, []))

    monkeypatch.setattr(fo, "OpenAIChatClient", _factory)


def test_first_model_502_switches_to_same_provider_next(monkeypatch):
    # provider sufy: [A(502), B(ok)] -> should return B's answer
    _patch_clients(monkeypatch, {"A": [_http_error(502)], "B": ["ok:B"]})
    role = _role([ResolvedProvider("sufy", [_ep("A"), _ep("B")])])
    c = FailoverChatClient(role)
    assert c.chat([]) == "ok:B"
    assert c.model == "B"  # promoted to default


def test_provider_fully_down_switches_provider(monkeypatch):
    # sufy: [A(502), B(502)] all down; tokenhub: [C(ok)]
    _patch_clients(
        monkeypatch,
        {"A": [_http_error(502)], "B": [_http_error(503)], "C": ["ok:C"]},
    )
    role = _role([
        ResolvedProvider("sufy", [_ep("A"), _ep("B")]),
        ResolvedProvider("tokenhub", [_ep("C")]),
    ])
    c = FailoverChatClient(role)
    assert c.chat([]) == "ok:C"
    assert c.model == "C"


def test_success_updates_default_next_call_starts_there(monkeypatch):
    # First call: A fails, B ok -> default becomes B. Second call: B ok directly.
    _patch_clients(monkeypatch, {"A": [_http_error(502)], "B": ["ok:B1", "ok:B2"]})
    role = _role([ResolvedProvider("sufy", [_ep("A"), _ep("B")])])
    c = FailoverChatClient(role)
    assert c.chat([]) == "ok:B1"
    assert c.chat([]) == "ok:B2"  # starts at B now, A not retried


def test_all_endpoints_failed_raises(monkeypatch):
    _patch_clients(monkeypatch, {"A": [_http_error(502)], "B": [_http_error(500)]})
    role = _role([ResolvedProvider("sufy", [_ep("A"), _ep("B")])])
    c = FailoverChatClient(role)
    with pytest.raises(AllEndpointsFailed):
        c.chat([])


def test_non_retryable_400_surfaces_immediately(monkeypatch):
    # 400 = bad request; retrying another model would not help -> re-raise as-is
    _patch_clients(monkeypatch, {"A": [_http_error(400)], "B": ["ok:B"]})
    role = _role([ResolvedProvider("sufy", [_ep("A"), _ep("B")])])
    c = FailoverChatClient(role)
    with pytest.raises(httpx.HTTPStatusError):
        c.chat([])


def test_disk_state_roundtrip(monkeypatch, tmp_path):
    # First client discovers B works and persists it; a fresh client reads B as default.
    state = tmp_path / "endpoint_state.json"
    _patch_clients(monkeypatch, {"A": [_http_error(502)], "B": ["ok:B", "ok:B2"]})
    role = _role([ResolvedProvider("sufy", [_ep("A"), _ep("B")])])
    c1 = FailoverChatClient(role, state_path=str(state))
    assert c1.chat([]) == "ok:B"
    saved = json.loads(state.read_text())
    assert saved["test"]["model_idx"] == 1  # B

    c2 = FailoverChatClient(role, state_path=str(state))
    assert c2.model == "B"  # started from persisted default, no A re-probe


def test_rotation_orthogonal_to_failover(monkeypatch):
    # rotate_every=1 advances default each call among available models.
    _patch_clients(monkeypatch, {"A": ["a1", "a2"], "B": ["b1", "b2"]})
    role = _role([ResolvedProvider("sufy", [_ep("A"), _ep("B")])], rotate_every=1)
    c = FailoverChatClient(role)
    r1 = c.chat([])  # rotate: mi 0->1 (B), B ok
    r2 = c.chat([])  # rotate: mi 1->0 (A), A ok
    assert {r1, r2} == {"b1", "a1"}
