# Harness Engineering Map

`scripts/harness/` is the repo's agent-legible control surface: thin CLIs,
deterministic guards, probes, benchmarks, and bounded diagnostics. It should
help Codex and humans answer three questions quickly:

1. What workflow exists?
2. Where is the real implementation?
3. Which command proves it still works?

Use this README as a map, not an encyclopedia. Put durable behavior and
architecture in the source-of-truth docs below, keep generated evidence under
`docs/generated/`, and encode repeatable rules in checks.

## Source-Of-Truth Ladder

| Layer | File | Purpose |
| --- | --- | --- |
| Repo charter | `AGENTS.md` | Short injected map for agents: scope, defaults, invariants, and where to look next. |
| Scope | `docs/SCOPE.md` | In-scope vs out-of-scope boundary for recording, alignment, demos, proxy, tracking, and visualization. |
| Architecture | `docs/ARCHITECTURE.md` | Package and entrypoint layering, dependency direction, and formal data-product boundaries. |
| Harness engineering | `docs/HARNESS_ENGINEERING.md` | Agent-first operating model, stable harness interfaces, and Demo 2.3 failure-packet contract. |
| Workflows | `docs/WORKFLOWS.md` | Operator commands and expected manual procedures. |
| Harness catalog | `scripts/harness/_catalog.py` | Machine-checkable list of public harness entrypoints, categories, summaries, and help coverage. |
| Harness guards | `scripts/harness/check_*.py` | Mechanical enforcement for scope, catalog coverage, visual architecture, experiments, and legacy Demo 2.2 boundaries. |
| Current evidence | `docs/generated/harness_engineering_compact_index.md` | Compact index for generated validation artifacts and current harness claims. |
| Plans | `docs/exec-plans/active/`, `docs/exec-plans/completed/` | Versioned intent, decisions, validation, and follow-up state for non-trivial changes. |

If a claim is important enough for a future agent to rely on, make it
repository-local and link it from one of these layers.

## Single-Camera Branch Safety

Single-camera-specific modifications belong on the `single-camera` branch.
Before editing single-camera behavior, confirm `git branch --show-current`
prints `single-camera`; if it does not, switch with `git switch single-camera`.
Do not commit or push single-camera changes directly to `main`. The validated
push target for that work is `git push origin single-camera`.

## Harness Contract

- Public harness scripts are thin entrypoints. Reusable calibration, geometry,
  point-cloud, render, depth, demo runtime, and tracking logic belongs under
  `data_process/`, `qqtt/demo/`, or `qqtt/tracking/`.
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

`_catalog.py` currently contains 75 entries.

| Category | Count | Meaning |
| --- | ---: | --- |
| `checks` | 7 | Repo, scope, architecture, experiment-boundary, legacy Demo 2.2 boundary, harness-engineering, and catalog guards. |
| `hardware_external` | 13 | RealSense, FFS, SAM, TensorRT, WSLg/Open3D, and static replay probes. |
| `mask_support` | 4 | SAM mask generation, helper code, object-case registry, and reprojection support. |
| `formal_cleanup` | 1 | Downstream cleanup for `data/different_types/`. |
| `current_compare` | 12 | In-scope aligned RealSense/native-vs-FFS comparison visualizations. |
| `experiments` | 31 | Experiment-only workflows under `scripts/harness/experiments/`. |
| `focused_diagnostics` | 7 | Narrow audits, overlays, render probes, source diagnostics, Demo 2.3 failure packets, and enhanced PT surface-filter profiling. |

Help coverage:

| Profile | Entries | Use |
| --- | ---: | --- |
| `quick` | 10 | Fast help checks included in default `check_all.py`. |
| `full` | 53 | Additional help checks included by `check_all.py --full`. |
| none | 12 | Guards, helpers, or shell scripts without direct argparse help coverage. |

## Demo 2.3 Failure Packet

For Demo 2.3 FPS or fused-PCD debugging, build a compact packet before changing
runtime code:

```bash
conda run -n demo_2_max --no-capture-output python scripts/harness/summarize_demo23_failure_packet.py \
  --output-json docs/generated/demo23_failure_packet.json \
  --output-md docs/generated/demo23_failure_packet.md
```

The packet pulls together profile JSON, runtime summaries, debug-fusion
calibration reports, and calibration preflight reports. It highlights the
batch-3 builderOptimizationLevel=5 FFS contract, queue/drop pressure, temporal
skew, no-render metric caveats, and calibration risks so future agents begin
from the same evidence.

## Add Or Change A Harness Entrypoint

1. Put shared implementation outside `scripts/harness/` first.
2. Add a small CLI, probe, guard, or visualization wrapper in the appropriate
   harness folder.
3. Register it in `_catalog.py` with category, summary, and help profile.
4. Link the durable claim from `docs/generated/harness_engineering_compact_index.md`
   when the result supersedes older generated reports.
5. Add or update tests when behavior changes.
6. Run the deterministic checks below.

## Generated Artifact Policy

- Prefer one compact validation note per theme over many near-duplicate notes.
- Keep raw logs only when a linked validation note needs them.
- Treat `docs/generated/harness_engineering_compact_index.md` as the current
  claim index; older generated reports remain historical unless linked there.
- Cleanup old generated artifacts through documented cleanup passes, not ad hoc
  deletion from harness scripts.

## Validation Commands

Use the documented default environment:

```bash
conda run -n demo_2_max --no-capture-output python scripts/harness/check_harness_catalog.py
conda run -n demo_2_max --no-capture-output python scripts/harness/check_demo22_boundaries.py
conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py
conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py --full
```

For ordinary doc or narrow harness changes, run the default `check_all.py`.
Use `--full` when the change broadens public CLI surface or harness coverage.

## Agent-First Maintenance

When Codex struggles, do not only patch the immediate symptom. Ask which
repository-local capability was missing:

- Was the map unclear?
- Was the source of truth stale?
- Was an invariant only written in prose when it should be a check?
- Was generated evidence not indexed?
- Was a long-lived plan missing decisions or validation?

Capture the answer in docs or tooling so the next agent can move faster with
less human context.
