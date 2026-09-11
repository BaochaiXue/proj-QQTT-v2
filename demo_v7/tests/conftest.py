"""Keep the demo_v7 unit suite hermetic.

These tests exercise option plumbing, state machines and geometry — none of
them needs a model install. Several of them do construct a real
``OrchestratorSession``, though, and the session fail-fasts on the chosen
backend's local install (the trellis2 conda python, the TRELLIS.2 checkout
and an HF snapshot in the local cache). On a developer box that quietly
passes; on a clean CI runner it turns a unit test into an install check.

The availability probes are stubbed out for the whole suite so the tests
assert what they are about. The probes themselves stay covered where they
belong — by their own tests, which call them directly.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _stub_backend_install_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    from demo_v7.service import backend_options, gaussian_options

    monkeypatch.setattr(
        backend_options, "ensure_trellis2_available", lambda: None
    )
    monkeypatch.setattr(
        gaussian_options, "ensure_triposplat_available", lambda: None
    )
