# AGENTS

## Repo Charter

This `single-camera` branch handles single-camera RealSense preview,
calibration, synchronized recording, aligned case generation, native-vs-FFS
comparison visualization for aligned cases, and sanctioned single-camera
realtime demo/proxy diagnostics built on that camera stream. The `main` branch
remains the protected 3-camera baseline until the user explicitly changes the
repo-wide default.

## Current Experiment Convention

- For current experiment/demo artifacts, the object is `stuffed animal` and the
  controller is `towel`.
- The formal live demo default remains `controller = hand`; `towel` is the
  current non-operator surrogate controller.

## File Map

- `cameras_viewer.py`: live preview / debug entrypoint
- `cameras_calibrate.py`: calibration entrypoint
- `record_data.py`: raw RGB-D recording entrypoint
- `record_data_realtime_align.py`: native realtime RGB-D aligned formal export baseline
- `data_process/record_data_align.py`: trim + align raw cases into `data/`
- `data_process/depth_backends/`: shared FFS geometry + runner used by production alignment and harness scripts
- `data_process/visualization/`: aligned-case comparison visualization package
- `data_process/visualization/experiments/`: experiment-only visualization workflows; formal recording/alignment code must not import this package
- `scripts/harness/diagnostics/demo/realtime_single_camera_pointcloud.py`: branch-default single-camera realtime point-cloud demo entrypoint
- `qqtt/demo/realtime_single_camera_pointcloud.py`: shared single-camera realtime point-cloud demo implementation
- `demo_v3/`: one-camera RealSense masked PCD demo
- `demo_v3_1/`: one-camera RealSense masked PCD demo
- `demo_v3_2/`: one-camera FFS masked PCD demo
- `demo_v3_3/`: one-camera FFS masked PCD demo
- `demo_v4/`: isolated realtime FuturePhysTwin chunk preprocessing demo that
  converts Demo 3.2-style single-camera headless/realtime artifacts into
  FuturePhysTwin-consumable per-window chunk cases
- `qqtt/demo/single_demo_v3_runtime.py`: shared single Demo 3.x launcher
- `services/ffs_remote/`: single-camera remote FFS request/response services for demo/profiling use
- `qqtt/env/camera/`: shared RealSense camera runtime
- `qqtt/env/camera/preflight.py`: record-time probe/preflight decision table
- `env_install/env_install.sh`: camera-only environment setup
- `docs/SCOPE.md`: exact in-scope vs out-of-scope boundary
- `docs/WORKFLOWS.md`: canonical operator workflows
- `docs/ARCHITECTURE.md`: kept package/file structure
- `docs/HARDWARE_VALIDATION.md`: manual real-hardware checklist
- `docs/external-deps.md`: external repo / checkpoint source of truth
- `docs/envs.md`: validated local conda environments
- `docs/exec-plans/`: first-class execution plans for non-trivial changes
- `scripts/harness/guards/check_scope.py`: deterministic repo scope guard
- `scripts/harness/guards/check_visual_architecture.py`: visualization layering / file-size guard
- `scripts/harness/guards/check_experiment_boundaries.py`: guard that keeps experiment-only modules out of formal runtime code
- `scripts/harness/README.md`: grouped harness CLI/probe/check map and retention policy
- `tests/test_record_data_align_smoke.py`: smoke test for aligned-case generation
- `scripts/harness/diagnostics/depth/visual_compare_depth_panels.py`: per-camera aligned native-vs-FFS depth panels
- `scripts/harness/diagnostics/depth/visual_compare_reprojection.py`: aligned native-vs-FFS reprojection compare
- `scripts/harness/diagnostics/depth/visual_compare_depth_video.py`: temporal fused compare
- `scripts/harness/diagnostics/depth/visual_compare_depth_triplet_ply.py`: single-frame native / FFS raw / FFS postprocess fused PLY compare
- `scripts/harness/diagnostics/depth/visual_compare_depth_triplet_video.py`: multi-frame native / FFS raw / FFS postprocess point-cloud video compare
- `scripts/harness/diagnostics/visualization/visual_compare_rerun.py`: multi-frame native-vs-FFS remove-invisible point-cloud export to Rerun + fused PLYs
- `scripts/harness/diagnostics/visualization/visual_compare_turntable.py`: current single-frame professor-facing compare
- `scripts/harness/experiments/ffs/visualize_ffs_static_confidence_panels.py`: static-round masked FFS RGB/depth/confidence boards
- `scripts/harness/experiments/ffs/visualize_ffs_static_confidence_pcd_panels.py`: static-round masked FFS RGB/PCD/confidence boards
- `scripts/harness/diagnostics/visualization/visual_make_professor_triptych.py`: current three-figure professor-facing summary pack
- `scripts/harness/diagnostics/visualization/visual_make_match_board.py`: current point-cloud match board
- `scripts/harness/diagnostics/depth/audit_ffs_left_right.py`: focused FFS left/right ordering audit
- `scripts/harness/diagnostics/visualization/visual_compare_stereo_order_pcd.py`: point-cloud-only current-vs-swapped stereo-order registration board
- `scripts/harness/diagnostics/depth/compare_face_smoothness.py`: fixed face-patch smoothness/noise comparison

## Default Local Environment

- Use `demo_2_max` as the default conda environment for integrated local demo and
  harness work that needs EdgeTAM, RealSense, Fast-FoundationStereo, TensorRT,
  Open3D, and SAM 3.1 in one Python process.
- Prefer `conda run -n demo_2_max --no-capture-output ...` for non-interactive
  commands, or `conda activate demo_2_max` for manual shell workflows.
- Treat `edgetam-max` as an isolated EdgeTAM validation environment, not the
  default RS/FFS demo environment.
- Treat `FFS-SAM-RS` and `FFS-max-sam31-rs` as FFS/SAM/RealSense stack
  environments; do not use them as the default integrated demo environment
  unless a task explicitly asks for that comparison.
- Keep external EdgeTAM repos, FFS repos, SAM checkpoints, and other weights
  outside this repo; reference them by documented local path.

## Python Formatting Defaults

- For Python files touched by Codex, target the project formatter line length
  in `pyproject.toml` and split long code, comments, and docstrings into
  readable multi-line form before finishing.
- Do not leave newly written Python lines over 100 columns unless an
  unavoidable literal, URL, or generated artifact requires it.
- Keep formatting scoped to touched files or snippets; do not run broad
  autoformatting across unrelated files.

## Single-Camera Branch Policy

- All single-camera-specific modifications must be made, committed, and pushed on the `single-camera` branch.
- Before any single-camera change, run `git branch --show-current` and confirm it prints `single-camera`; if it does not, switch with `git switch single-camera` before editing.
- Do not commit or push single-camera changes directly to `main`, and do not merge `single-camera` into `main` unless the user explicitly asks for that merge.
- For single-camera work, the post-validation push target is `git push origin single-camera`, not `git push origin main`.
- Keep `main` protected as the existing 3-camera baseline until the user explicitly changes the repo-wide default.

## Required Workflow For Future Changes

1. Before modifying files, run `git pull --ff-only origin main` and confirm the local branch is up to date with GitHub.
2. For single-camera-specific work, confirm the current branch is `single-camera` before editing.
3. Start with an exec plan under `docs/exec-plans/active/` for any non-trivial change.
4. Keep changes inside the documented camera preview / calibration / recording / alignment core or the sanctioned single-camera demo / proxy diagnostic scope.
5. Update docs and tests in the same change when behavior changes.
6. Run deterministic checks before finishing:
   - `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`
   - use `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile exhaustive` when the change is broad enough that the default smoke profile is not sufficient
7. For external dependency proof-of-life work, record exact commands and outcomes under `docs/generated/`.
8. For FFS changes, keep weights external and validate both deterministic tests and manual hardware outcomes.
9. For comparison visualization changes, validate the calibration loader and non-interactive render path.
10. After committing validated modifications, push them to GitHub with `git push origin single-camera` for single-camera work, otherwise `git push origin main`, unless the user explicitly says not to push.

## Autonomous Goal Execution Policy

- When the user starts or resumes an explicit implementation goal, assume the
  user will be offline until the goal is complete.
- Do not block on design approval, execution-plan approval, or implementation
  approach approval while pursuing that goal. Default to the recommended design
  and continue making concrete progress toward the requested end state.
- Use the exec plan to record decisions, tradeoffs, and validation gates, but do
  not treat the plan as something that requires a separate user approval before
  coding.
- Ask the user only when progress is truly impossible without information that
  cannot be inferred from the repo, external dependency contracts, or the
  stated goal.
- This policy does not override safety and branch invariants: still keep work on
  the correct branch, avoid destructive commands, keep external weights out of
  the repo, update docs/tests with behavior changes, and validate before
  committing/pushing.

## Invariants

- The repo's primary data product stops at `data_process/record_data_align.py`; aligned native-vs-FFS comparison visualization remains an in-scope diagnostic utility built on aligned cases.
- Demo-only segmentation/mask code may exist only in the sanctioned single-camera demo, proxy, visualization, and diagnostic layers; formal recording/alignment code must not depend on those layers or make their artifacts part of the aligned-case compatibility contract.
- Do not reintroduce three-camera demo entrypoints, dual-GPU demo contracts,
  batch-3 demo runtimes, inverse physics, Gaussian Splatting,
  reconstruction/rendering evaluation, robot control, manipulation policy, or
  teleop code. Shape-prior and `final_data.pkl` generation are allowed only in
  the explicitly isolated Demo 3.2 warmup diagnostic and Demo v4
  FuturePhysTwin chunk preprocessing carveouts; they must not alter formal
  recording/alignment products.
- `qqtt/__init__.py` exports only `CameraSystem`.
- `env_install/env_install.sh` stays camera-only.
- Hardware checks remain manual and documented; do not fake them in CI.
- External repos and weights stay outside this repo and are referenced by path.
- `depth/` must remain the canonical compatibility output for aligned cases.
- Comparison visualization is allowed for aligned native-vs-FFS depth inspection and explicitly documented demo/diagnostic artifacts.

## Do Not Change Without Updating Docs / Tests

- camera CLI defaults
- output directory layout for `data_collect/` and `data/`
- metadata fields written by recording / alignment
- scope guard rules

## Deep Docs

- Scope boundary: `docs/SCOPE.md`
- Architecture: `docs/ARCHITECTURE.md`
- Harness map: `scripts/harness/README.md`
- User workflows: `docs/WORKFLOWS.md`
- Manual validation: `docs/HARDWARE_VALIDATION.md`
- Active execution plans: `docs/exec-plans/active/`
