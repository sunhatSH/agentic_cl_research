"""Tests for the vendor-isolated sandbox client + GRPO group sampling."""

from __future__ import annotations

import pytest

from rollout.sandbox_client import (
    ExecResult,
    LocalSandbox,
    grpo_advantages,
    make_sandbox,
    register_backend,
    select_winner,
)


def test_local_sandbox_executes_python():
    sb = LocalSandbox(timeout=10)
    res = sb.run_code("print((23 * 17) - 19)")
    sb.kill()
    assert res.ok
    assert res.stdout.strip() == "372"


def test_local_sandbox_reports_failure():
    sb = LocalSandbox(timeout=10)
    res = sb.run_code("raise ValueError('boom')")
    assert not res.ok
    assert "ValueError" in res.stderr


def test_local_sandbox_rejects_non_python():
    res = LocalSandbox().run_code("console.log(1)", language="js")
    assert not res.ok


def test_local_sandbox_timeout():
    sb = LocalSandbox(timeout=1)
    res = sb.run_code("import time; time.sleep(5)")
    assert not res.ok
    assert "timeout" in res.stderr


def test_grpo_advantages_zero_mean():
    advs = grpo_advantages([1.0, 0.0, 1.0, 0.0])
    assert abs(sum(advs)) < 1e-6
    assert advs[0] > 0 and advs[1] < 0


def test_grpo_advantages_empty():
    assert grpo_advantages([]) == []


def test_select_winner_argmax():
    assert select_winner([0.0, 1.0, 0.0]) == 1


def test_select_winner_tiebreak_by_trajectory_id():
    # Two winners (reward 1.0) -> lexicographically smallest id wins.
    idx = select_winner([1.0, 0.0, 1.0], trajectory_ids=["zzz", "aaa", "bbb"])
    assert idx == 2  # "bbb" < "zzz"


def test_make_sandbox_local_and_unknown():
    assert isinstance(make_sandbox("local"), LocalSandbox)
    with pytest.raises(ValueError):
        make_sandbox("nope")


def test_make_sandbox_e2b_without_credentials_raises(monkeypatch):
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    monkeypatch.delenv("E2B_DOMAIN", raising=False)
    with pytest.raises(RuntimeError, match="E2B_API_KEY"):
        make_sandbox("e2b")


def test_make_sandbox_aliyun_is_registered_but_stubbed():
    # 'aliyun' is wired into the registry (the interface/extension point is ready),
    # but the vendor implementation is intentionally left blank (留空) for now, so
    # instantiating it fails fast with NotImplementedError.
    with pytest.raises(NotImplementedError):
        make_sandbox("aliyun")


def test_make_sandbox_unknown_lists_registered_backends():
    # The ValueError enumerates the registry, proving make_sandbox dispatches by name.
    with pytest.raises(ValueError, match="aliyun"):
        make_sandbox("does_not_exist")


def test_register_backend_is_open_for_extension():
    # A new vendor plugs in with one register_backend() call -- no edit to the loop.
    class _CustomSandbox:
        def run_code(self, code, language="python"):
            return ExecResult("custom-ok", "", True)

        def kill(self):
            pass

    register_backend("custom_vendor", lambda **kw: _CustomSandbox())
    sb = make_sandbox("custom_vendor")
    assert isinstance(sb, _CustomSandbox)
    assert sb.run_code("anything").stdout == "custom-ok"


def test_e2b_kill_delegates_to_sdk(monkeypatch):
    """E2BSandbox.kill() delegates to the SDK's kill() and never raises.

    The historical regression (kill DELETE-ing sandboxID-clientID instead of
    the bare sandboxID, leaking instances) is now handled inside the e2b SDK's
    own Sandbox.kill(); our wrapper just calls it best-effort. This test pins
    that contract: kill() calls the SDK kill once and swallows any error.
    """
    from rollout.sandbox_client import E2BSandbox

    # Bypass __init__ (which would hit the network) -- we only test kill().
    sb = E2BSandbox.__new__(E2BSandbox)
    sb._timeout = 300

    calls: list[str] = []

    class _FakeSDKSandbox:
        def kill(self):
            calls.append("kill")

    sb._sb = _FakeSDKSandbox()
    sb._sandbox_id = "sbx123"
    # Must not raise.
    sb.kill()
    assert calls == ["kill"]


def test_e2b_kill_swallows_sdk_error(monkeypatch):
    """kill() must never raise even if the SDK kill() throws."""
    from rollout.sandbox_client import E2BSandbox

    sb = E2BSandbox.__new__(E2BSandbox)
    sb._timeout = 300

    class _FakeSDKSandbox:
        def kill(self):
            raise RuntimeError("network down")

    sb._sb = _FakeSDKSandbox()
    sb._sandbox_id = "sbx123"
    sb.kill()  # must not raise
