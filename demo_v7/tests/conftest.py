"""Keep the demo_v7 unit suite hermetic.

These tests exercise option plumbing, state machines and geometry — none of
them needs a model install. Several of them do construct a real
``OrchestratorSession``, though, and startup fail-fasts on machine-local
integrations: the TRELLIS.2 env/checkout/HF snapshot, TripoSplat weights, and
the external Phystwin_shen checkout/config/conda environment. On a developer
box those quietly pass; on a clean CI runner they turn unit tests into install
checks.

Patch only the call sites used by session construction. The probe/validator
functions themselves remain available to their dedicated tests, which call
them directly.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _stub_backend_install_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    from demo_v7.runtime.orchestration import run_config
    from demo_v7.service import backend_options, gaussian_options

    monkeypatch.setattr(
        backend_options, "ensure_trellis2_available", lambda: None
    )
    monkeypatch.setattr(
        gaussian_options, "ensure_triposplat_available", lambda: None
    )
    # OrchestratorRunConfig imports these validators into its own module
    # namespace. Stub that construction boundary, not the underlying module,
    # so validator-specific tests still exercise the real implementation.
    monkeypatch.setattr(
        run_config, "validate_phystwin_shen_repo", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        run_config, "validate_phystwin_shen_settings", lambda *args, **kwargs: None
    )
