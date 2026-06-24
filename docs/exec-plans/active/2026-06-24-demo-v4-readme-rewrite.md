# Demo v4 README Rewrite

## Goal

Rewrite the project root `README.md` as an operator-focused Demo v4 entrypoint
that explains how to run Demo v4, which options matter, where outputs are
written, and how to check that a run succeeded.

## Scope

- Replace the project root README with a Demo v4-first runbook.
- Keep `demo_v4/README.md` as a more detailed Demo v4 reference page.
- Keep the documented commands aligned with the current CLI defaults in
  `demo_v4/realtime_futurephystwin_chunks.py` and
  `services/shape_prior_remote/server.py`.
- Preserve the FuturePhysTwin chunk output contract, READY marker convention,
  shape-prior worker workflow, and common debug variants.

## Non-Goals

- No runtime code changes.
- No changes to Demo 3.2 behavior or shape-prior protocol.
- No hardware validation run.

## Validation

- Run `python demo_v4/realtime_futurephystwin_chunks.py --help`.
- Run `python services/shape_prior_remote/server.py --help`.
- Run a focused smoke test for Demo v4 docs-adjacent behavior if the local
  environment supports it.
