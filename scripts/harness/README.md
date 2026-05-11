# Harness Engineering Map

`scripts/harness/` is the operator-facing edge of the repo: CLI wrappers, probes, deterministic guards, and bounded diagnostics. Reusable calibration, geometry, point-cloud, layout, render, and depth logic belongs under `data_process/`.

## Maintenance Contract

- Keep stable public CLIs at their existing paths unless docs and tests move with them.
- Add every new public harness Python file to `scripts/harness/_catalog.py`.
- Keep one-off visualization experiments under `scripts/harness/experiments/`.
- Keep formal recording/alignment code free of `scripts.harness.experiments` and `data_process.visualization.experiments` imports.
- Keep external repos, checkpoints, TensorRT engines, SAM assets, generated proof outputs, and local replay datasets outside harness code.
- Remove local cache artifacts such as `__pycache__/`.

## Source Of Truth

| File | Role |
| --- | --- |
| `_catalog.py` | Compact catalog of every public harness Python entrypoint, category, summary, and help-check profile. |
| `check_harness_catalog.py` | Verifies every public harness Python file is cataloged and categorized correctly. |
| `check_all.py` | Runs the quick/full deterministic validation profiles; help-surface coverage comes from `_catalog.py`. |
| `docs/generated/harness_engineering_compact_index.md` | Compressed index for current generated harness engineering results and claims. |
| `docs/envs.md` | CUDA/toolkit and validated environment policy. |
| `docs/WORKFLOWS.md` | Operator workflows and FFS live-vs-proxy reporting boundary. |

Run:

```bash
python scripts/harness/check_harness_catalog.py
python scripts/harness/check_all.py
python scripts/harness/check_all.py --full
```

## Catalog Summary

Current `_catalog.py` entries: `70`.

| Category | Count | Meaning |
| --- | ---: | --- |
| `checks` | 5 | Repo, scope, architecture, experiment-boundary, and catalog guards. |
| `hardware_external` | 13 | RealSense probes, FFS/SAM/TensorRT proofs, WSLg/Open3D helpers, and static replay benchmarks. |
| `mask_support` | 4 | SAM 3.1 mask generation, helper code, object-case registry, and single-pair reprojection support. |
| `formal_cleanup` | 1 | Downstream cleanup for `data/different_types/`. |
| `current_compare` | 12 | In-scope aligned RealSense/native-vs-FFS comparison visualizations. |
| `experiments` | 31 | Experiment-only workflows under `scripts/harness/experiments/`. |
| `focused_diagnostics` | 4 | Narrow audits, overlays, and source diagnostics. |

Help profile coverage:

| Profile | Entries | Use |
| --- | ---: | --- |
| `quick` | 8 | Fast help checks used by default `check_all.py`. |
| `full` | 52 | Broader help checks used by `check_all.py --full`. |
| none | 10 | Helpers or shell scripts without direct argparse help coverage. |

## Current Boundaries

- FFS defaults and current performance claims are summarized in `docs/generated/harness_engineering_compact_index.md`; use `docs/generated/ffs_live_vs_proxy_boundary.md` for live-vs-proxy wording.
- Shared CUDA 13 toolkit policy lives in `docs/envs.md`; do not duplicate CUDA install guidance here.
- Demo v0.3 remote FFS work is tracked by `docs/exec-plans/active/2026-05-08-demo-v0-3-100kit-staged-remote-ffs.md` and the generated Demo v0.3 validation notes.
- Object/controller PCD filtering caveats are summarized in `docs/generated/harness_engineering_compact_index.md`.
- Object capture lookup should use `scripts.harness.object_case_registry` by `(object_set, round_id)`.

## Adding A Harness File

1. Put reusable implementation outside harness first, usually under `data_process/`.
2. Add the thin CLI/probe/check file under the right harness folder.
3. Add a `HarnessEntry` in `_catalog.py` with the right category and `help_profile`.
4. Run `python scripts/harness/check_harness_catalog.py`.
5. Run `python scripts/harness/check_all.py`; use `--full` for broad changes.

## Retention Policy

- Keep current user-facing CLIs, deterministic checks, hardware probes, and bounded diagnostics.
- Archive or delete obsolete generated results through documented cleanup passes, not from harness scripts.
- Prefer extending `docs/generated/harness_engineering_compact_index.md` over adding more long-form README sections.
