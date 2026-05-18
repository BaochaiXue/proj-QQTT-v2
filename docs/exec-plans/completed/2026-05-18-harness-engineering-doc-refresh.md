# Harness Engineering Doc Refresh

## Goal

Refresh the harness documentation so it is easier for Codex agents and humans
to navigate: small entrypoint, explicit source-of-truth ladder, mechanical
guards, and clear rules for adding harness files or generated artifacts.

## Scope

- Rewrite `scripts/harness/README.md` around agent-first harness engineering:
  repository-local knowledge, progressive disclosure, cataloged entrypoints,
  deterministic checks, and generated-artifact retention.
- Add a compact source-of-truth pointer in
  `docs/generated/harness_engineering_compact_index.md`.
- Keep behavior unchanged.

## Non-Goals

- No harness CLI, runtime, or test behavior changes.
- No generated artifact cleanup.

## Validation

- Run deterministic checks after the doc update.
- Completed: `conda run -n demo_2_max --no-capture-output python scripts/harness/check_harness_catalog.py`
- Completed: `conda run -n demo_2_max --no-capture-output python scripts/harness/check_demo22_boundaries.py`
- Completed: `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
