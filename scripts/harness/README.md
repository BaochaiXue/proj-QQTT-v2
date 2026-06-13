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
| Harness catalog | `scripts/harness/_catalog.py` | Machine-checkable list of public harness entrypoints, categories, summaries, and help coverage. |
| Harness guards | `scripts/harness/check_*.py` | Mechanical enforcement for scope, catalog coverage, visual architecture, and experiment boundaries. |
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
  `data_process/` or `qqtt/demo/`.
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

`_catalog.py` currently contains 62 entries.

| Category | Count | Meaning |
| --- | ---: | --- |
| `checks` | 5 | Repo, scope, architecture, experiment-boundary, and catalog guards. |
| `hardware_external` | 13 | RealSense, FFS, SAM, TensorRT, WSLg/Open3D, and static replay probes. |
| `mask_support` | 4 | SAM mask generation, helper code, object-case registry, and reprojection support. |
| `formal_cleanup` | 1 | Downstream cleanup for `data/different_types/`. |
| `current_compare` | 12 | In-scope aligned RealSense/native-vs-FFS comparison visualizations. |
| `experiments` | 24 | Experiment-only workflows under `scripts/harness/experiments/`. |
| `focused_diagnostics` | 3 | Narrow audits and source diagnostics. |

Help coverage:

| Profile | Entries | Use |
| --- | ---: | --- |
| `quick` | 3 | Fast help checks included in default `check_all.py`. |
| `full` | 52 | Additional help checks included by `check_all.py --full`. |
| none | 10 | Guards, helpers, or shell scripts without direct argparse help coverage. |

## Add Or Change A Harness Entrypoint

1. Put shared implementation outside `scripts/harness/` first.
2. Add a small CLI, probe, guard, or visualization wrapper in the appropriate harness folder.
3. Register it in `_catalog.py` with category, summary, and help profile.
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
conda run -n demo_2_max --no-capture-output python scripts/harness/check_harness_catalog.py
conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py
conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py --full
```

For ordinary doc or narrow harness changes, run the default `check_all.py`.
Use `--full` when the change broadens public CLI surface or harness coverage.
