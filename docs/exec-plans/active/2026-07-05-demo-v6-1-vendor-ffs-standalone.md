# Demo v6.1 — Vendor FFS Code for Standalone Release

## Requirement

Problem:
`demo_v6_1/main_data_processing.py` and `demo_v6_1/main_warmup.py`
unconditionally import from the repo-level `services.ffs_remote` package (and
`data_process.depth_backends`), so `import demo_v6_1.*` fails without those
sibling packages present — even in the default `native-realsense` mode that
never touches FFS. Demo v6.1 must be publishable standalone: no imports of
code outside `demo_v6_1` (third-party pip packages excepted).

Required final behavior (user, 2026-07-05):
- Vendor the useful FFS parts into `demo_v6_1/utils`; drop the unused parts.
- No `demo_v6_1` module imports `services.*` or `data_process.*`.

Current state after the later Demo v6.1 legacy cleanup:
- The remote FFS client/protocol vendored during the first standalone pass was
  removed when the unused `ffs_remote` depth source and remote quality side
  channel were deleted.
- `depth_geometry.py` — verbatim from
  `data_process/depth_backends/geometry.py`.
- `ffs_defaults.py` — verbatim from
  `data_process/depth_backends/ffs_defaults.py` (`REPO_ROOT = parents[2]`
  still resolves to the repo root from the new location).
- `fast_foundation_stereo.py` — copy of
  `data_process/depth_backends/fast_foundation_stereo.py` with its one
  intra-repo import repointed to `demo_v6_1.utils.depth_geometry`; heavy
  TensorRT/Torch/FoundationStereo deps stay lazy.

Repointed imports now kept in Demo v6.1: `main_data_processing.py`
(`ffs_defaults` constants and lazy `FastFoundationStereoTensorRTRunner`),
`phystwin_strict_product.py` and `utils/projection.py` (`transform_points`).

Originals in `services/` and `data_process/` are untouched — `demo_v5_1`,
`data_process_origin`, and the FFS servers keep using them.

New test: `tests/test_demo_v6_1_standalone_ffs.py` — AST guard (no
services/data_process import in any demo_v6_1 module), fresh-interpreter
import probe, transform_points, and ffs_defaults checks. Registered in the
validation harness.

## Out of scope (reported, not done)

`demo_v6_1` still imports `qqtt.*` (3 top-level tracking/camera imports in
`main_data_processing.py`: `env.camera.table_calibration`,
`tracking.backends.point_tracker_adapter`, `tracking.sampling`; 2 lazy in the
non-core `others/run_first_order_stitched_futurephystwin.py`) and one lazy
`scripts.harness.experiments.edgetam`. `sam3.*` is an installed pip package
(a legitimate dependency, not repo-local). Fully standalone publishing needs
the `qqtt`/`scripts.harness` couplings addressed too — a larger, separate
effort (qqtt is the PhysTwin tracking/physics core), not part of the FFS
cleanup.

## Validation

- `python -m pytest tests/test_demo_v6_1_standalone_ffs.py -q`
- `python -m pytest tests/ -q` (162 passed)
- `python scripts/harness/validation/run.py --profile smoke`
- `demo_v6_1/main_data_processing.py --help` + `demo_v6_1/main.py --dry-run`
  build with the vendored FFS defaults/choices.
