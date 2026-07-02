# Harness Map

`scripts/harness/` is the repo's agent-legible control surface: thin CLIs,
deterministic guards, probes, benchmarks, and bounded diagnostics for the
single-camera branch.

## Source-Of-Truth Ladder

| Layer | File | Purpose |
| --- | --- | --- |
| Repo charter | `AGENTS.md` | Scope, defaults, invariants, and where to look next. |
| Scope | `docs/SCOPE.md` | In-scope vs out-of-scope boundary for recording, alignment, demos, proxy, and visualization. |
| Architecture | `docs/ARCHITECTURE.md` | Package and entrypoint layering, dependency direction, and formal data-product boundaries. |
| Workflows | `docs/WORKFLOWS.md` | Operator commands and expected manual procedures. |
| Harness catalog | `scripts/harness/_catalog.py` | Machine-checkable list of public harness entrypoints, lifecycle, summaries, and validation profiles. |
| Harness guards | `scripts/harness/guards/check_*.py` | Mechanical enforcement for scope, catalog coverage, visual architecture, and experiment boundaries. |
| Active plans | `docs/exec-plans/active/` | Current intent, decisions, validation, and follow-up state for non-trivial changes. |

## Single-Camera Branch Safety

Single-camera-specific modifications belong on the `single-camera` branch.
Before editing single-camera behavior, confirm `git branch --show-current`
prints `single-camera`; if it does not, switch with `git switch single-camera`.
Do not commit or push single-camera changes directly to `main`. The validated
push target for that work is `git push origin single-camera`.

## Harness Contract

- Public harness scripts are thin entrypoints. Reusable calibration, geometry,
  point-cloud, render, depth, and demo runtime logic belongs under
  `data_process/` or `demo_v5_1/`.
- Every public Python file under `scripts/harness/` must have a `HarnessEntry`
  in `_catalog.py`.
- One-off or research-style workflows live under `scripts/harness/experiments/`.
- Formal recording/alignment code must not import `scripts.harness.experiments`
  or `data_process.visualization.experiments`.
- External repos, checkpoints, TensorRT engines, SAM assets, generated proof
  outputs, and replay datasets stay outside harness code.
- Generated artifacts are evidence, not runtime dependencies.
- Local cache artifacts such as `__pycache__/` should not be committed.

## Current Catalog Shape

`_catalog.py` currently contains 64 entries.

| Lifecycle | Meaning |
| --- | --- |
| `guards` | Deterministic policy checks that keep scope, architecture, catalog, and experiment boundaries enforceable. |
| `validation` | Catalog-driven runners that compose deterministic validation profiles. |
| `diagnostics` | Maintained probes and visualization tools used for current branch debugging and evidence generation. |
| `benchmarks` | External-stack or performance-oriented proof-of-life commands that may need hardware, checkpoints, engines, or local datasets. |
| `experiments` | Isolated research/demo workflows under `scripts/harness/experiments/`; useful as historical or exploratory evidence, not formal runtime dependencies. |
| `support` | Helpers and cleanup tools shared by maintained harness workflows. |

Validation profiles:

| Profile | Coverage | Use |
| --- | --- | --- |
| `smoke` | Small maintained subset | Fast deterministic checks for ordinary docs, guard, and narrow harness edits. |
| `deterministic` | Broader maintained subset | Non-hardware checks for changes that affect catalog shape, path routing, or public CLI coverage. |
| `hardware` | Manual external-stack coverage | Hardware, GPU, GUI, checkpoint, engine, or local dataset checks that are cataloged but not automatic. |
| `exhaustive` | All cataloged deterministic coverage | Full maintained validation sweep before broad harness lifecycle or entrypoint changes. |
| none | Helpers or external-only workflows | Files without direct argparse help coverage or requiring manual/hardware-specific setup. |

## Demo Diagnostics

- `diagnostics/demo/render_demo32_headless_capture.py --panel-mode side-by-side`
  renders the Demo 3.2 fake-live 1x3 side-by-side panel from a saved headless
  capture.

## Add Or Change A Harness Entrypoint

1. Put shared implementation outside `scripts/harness/` first.
2. Add a small CLI, probe, guard, or visualization wrapper in the appropriate harness folder.
3. Register it in `_catalog.py` with lifecycle, summary, and validation profile.
4. Add or update tests when behavior changes.
5. Run the deterministic checks below.

## Generated Artifact Policy

- Prefer one compact validation note per theme over many near-duplicate notes.
- Keep raw logs only when a linked validation note needs them.
- Generated artifacts should not become runtime dependencies.
- Delete obsolete generated artifacts through documented cleanup changes.

## Validation Commands

Use the documented default environment:

```bash
conda run -n demo_2_max --no-capture-output python scripts/harness/guards/check_harness_catalog.py
conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke
conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile deterministic
conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile exhaustive
```

For ordinary doc or narrow harness changes, run the `smoke` profile. Use
`deterministic` when catalog shape, path routing, or public CLI coverage
changes. Use `exhaustive` before broad harness lifecycle migrations or
entrypoint reorganizations.

The smoke unittest batch should list only test modules that exist in the
current checkout. When tests are deleted or moved, update the validation
manifest in the same change instead of leaving stale module names in smoke.
