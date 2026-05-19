# AGENTS

## Repo Charter

This repository handles 3-camera RealSense preview, calibration, synchronized recording, aligned case generation, native-vs-FFS comparison visualization for aligned cases, and sanctioned realtime demo/proxy/tracking diagnostics built on those camera streams.

## Current Experiment Convention

- For the current experiment/demo artifacts, the object is `stuffed animal` and the controller is `towel`.
- All current experiment commands, generated artifact names, profiling summaries, tracking overlays, and demo notes should use this convention unless the user explicitly switches to a different case.
- The formal live demo default remains `controller = hand`; `towel` is the current non-operator surrogate controller because no hand operator is present during these experiments.

## File Map

- `cameras_viewer.py`: live preview / debug entrypoint
- `cameras_calibrate.py`: calibration entrypoint
- `record_data.py`: raw RGB-D recording entrypoint
- `record_data_realtime_align.py`: native realtime RGB-D aligned formal export baseline
- `data_process/record_data_align.py`: trim + align raw cases into `data/`
- `data_process/depth_backends/`: shared FFS geometry + runner used by production alignment and harness scripts
- `data_process/visualization/`: aligned-case comparison visualization package
- `data_process/visualization/experiments/`: experiment-only visualization workflows; formal recording/alignment code must not import this package
- `demo_v0_2/`, `demo_v0_3/`: remote FFS replay / proxy experiments
- `demo_v1/`, `demo_v2/`: single-camera realtime point-cloud demo lineage
- `demo_v2_1/`, `demo_v2_1_5/`, `demo_v2_2/`: three-camera semantic fused-PCD demo lineage
- `services/ffs_remote/`: remote FFS request/response and staged proxy services for demo/profiling use
- `qqtt/env/camera/`: shared RealSense camera runtime
- `qqtt/env/camera/preflight.py`: record-time probe/preflight decision table
- `qqtt/tracking/`: demo-oriented tracking backend contracts, probes, lifting, and metrics
- `env_install/env_install.sh`: camera-only environment setup
- `docs/SCOPE.md`: exact in-scope vs out-of-scope boundary
- `docs/WORKFLOWS.md`: canonical operator workflows
- `docs/ARCHITECTURE.md`: kept package/file structure
- `docs/HARNESS_ENGINEERING.md`: agent-first harness engineering map and Demo 2.3 failure-packet contract
- `docs/HARDWARE_VALIDATION.md`: manual real-hardware checklist
- `docs/external-deps.md`: external repo / checkpoint source of truth
- `docs/envs.md`: validated local conda environments
- `docs/exec-plans/`: first-class execution plans for non-trivial changes
- `docs/generated/README.md`: grouped index of generated validation docs and reusable helper assets
- `scripts/harness/check_scope.py`: deterministic repo scope guard
- `scripts/harness/check_visual_architecture.py`: visualization layering / file-size guard
- `scripts/harness/check_experiment_boundaries.py`: guard that keeps experiment-only modules out of formal runtime code
- `scripts/harness/README.md`: grouped harness CLI/probe/check map and retention policy
- `scripts/harness/summarize_demo23_failure_packet.py`: compact Demo 2.3 FPS/fused-PCD failure-packet builder
- `tests/test_record_data_align_smoke.py`: smoke test for aligned-case generation
- `scripts/harness/visual_compare_depth_panels.py`: per-camera aligned native-vs-FFS depth panels
- `scripts/harness/visual_compare_reprojection.py`: aligned native-vs-FFS reprojection compare
- `scripts/harness/visual_compare_depth_video.py`: older temporal fused compare
- `scripts/harness/visual_compare_depth_triplet_ply.py`: single-frame native / FFS raw / FFS postprocess fused PLY compare
- `scripts/harness/visual_compare_depth_triplet_video.py`: multi-frame native / FFS raw / FFS postprocess point-cloud video compare
- `scripts/harness/visual_compare_rerun.py`: multi-frame native-vs-FFS remove-invisible point-cloud export to Rerun + fused PLYs
- `scripts/harness/visual_compare_turntable.py`: current single-frame professor-facing compare
- `scripts/harness/experiments/visualize_ffs_static_confidence_panels.py`: static-round masked FFS RGB/depth/confidence 3x3 experiment boards
- `scripts/harness/experiments/visualize_ffs_static_confidence_pcd_panels.py`: static-round masked FFS RGB/PCD/confidence 3x3 experiment boards
- `scripts/harness/visual_make_professor_triptych.py`: current three-figure professor-facing summary pack
- `scripts/harness/visual_make_match_board.py`: current professor-facing 3-view point-cloud match board
- `scripts/harness/audit_ffs_left_right.py`: focused FFS left/right ordering audit
- `scripts/harness/visual_compare_stereo_order_pcd.py`: point-cloud-only current-vs-swapped stereo-order registration board
- `scripts/harness/compare_face_smoothness.py`: fixed face-patch smoothness/noise comparison

## Default Local Environment

- Use `demo_2_max` as the default conda environment for integrated local demo and harness work that needs EdgeTAM, RealSense, Fast-FoundationStereo, TensorRT, Open3D, and SAM 3.1 in one Python process.
- Prefer `conda run -n demo_2_max --no-capture-output ...` for non-interactive commands, or `conda activate demo_2_max` for manual shell workflows.
- Treat `edgetam-max` as an isolated EdgeTAM validation environment, not the default RS/FFS demo environment.
- Treat `FFS-SAM-RS` and `FFS-max-sam31-rs` as FFS/SAM/RealSense stack environments; do not use them as the default for EdgeTAM + RS + FFS integrated demo work unless a task explicitly asks for that comparison.
- Keep external EdgeTAM repos, FFS repos, SAM checkpoints, and other weights outside this repo; reference them by documented local path.

## Required Workflow For Future Changes

1. Before modifying files, run `git pull --ff-only origin main` and confirm the local branch is up to date with GitHub.
2. Start with an exec plan under `docs/exec-plans/active/` for any non-trivial change.
3. Keep changes inside the documented camera preview / calibration / recording / alignment core or the sanctioned demo / proxy / tracking diagnostic scope.
4. Update docs and tests in the same change when behavior changes.
5. Run deterministic checks before finishing:
   - `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
   - use `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py --full` when the change is broad enough that the default quick profile is not sufficient
6. For external dependency proof-of-life work, record exact commands and outcomes under `docs/generated/`.
7. For FFS changes, keep weights external and validate both deterministic tests and manual hardware outcomes.
8. For comparison visualization changes, validate the calibration loader and non-interactive render path.
9. After committing validated modifications, push them to GitHub with `git push origin main` unless the user explicitly says not to push.

## Invariants

- The repo's primary data product stops at `data_process/record_data_align.py`; aligned native-vs-FFS comparison visualization remains an in-scope diagnostic utility built on aligned cases.
- Demo-only segmentation/mask and tracking code may exist only in the sanctioned demo, proxy, visualization, and tracking-diagnostic layers; formal recording/alignment code must not depend on those layers or make their artifacts part of the aligned-case compatibility contract.
- Do not reintroduce shape prior, inverse physics, Gaussian Splatting, reconstruction/rendering evaluation, robot control, manipulation policy, or teleop code.
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
- Harness engineering: `docs/HARNESS_ENGINEERING.md`, `scripts/harness/README.md`
- User workflows: `docs/WORKFLOWS.md`
- Manual validation: `docs/HARDWARE_VALIDATION.md`
- Active/completed execution plans: `docs/exec-plans/`
