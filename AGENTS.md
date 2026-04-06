# AGENTS

## Repo Charter

This repository handles 3-camera RealSense preview, calibration, synchronized RGB-D recording, and aligned case generation up to `data_process/record_data_align.py`.

## File Map

- `cameras_viewer.py`: live preview / debug entrypoint
- `cameras_calibrate.py`: calibration entrypoint
- `record_data.py`: raw RGB-D recording entrypoint
- `data_process/record_data_align.py`: trim + align raw cases into `data/`
- `data_process/depth_backends/`: shared FFS geometry + runner used by production alignment and harness scripts
- `qqtt/env/camera/`: shared RealSense camera runtime
- `env_install/env_install.sh`: camera-only environment setup
- `docs/SCOPE.md`: exact in-scope vs out-of-scope boundary
- `docs/WORKFLOWS.md`: canonical operator workflows
- `docs/ARCHITECTURE.md`: kept package/file structure
- `docs/HARDWARE_VALIDATION.md`: manual real-hardware checklist
- `docs/external-deps.md`: external repo / checkpoint source of truth
- `docs/envs.md`: validated local conda environments
- `docs/exec-plans/`: first-class execution plans for non-trivial changes
- `scripts/harness/check_scope.py`: deterministic repo scope guard
- `tests/test_record_data_align_smoke.py`: smoke test for aligned-case generation

## Required Workflow For Future Changes

1. Start with an exec plan under `docs/exec-plans/active/` for any non-trivial change.
2. Keep changes inside camera preview / calibration / recording / alignment scope.
3. Update docs and tests in the same change when behavior changes.
4. Run deterministic checks before finishing:
   - `python scripts/harness/check_all.py`
5. For external dependency proof-of-life work, record exact commands and outcomes under `docs/generated/`.
6. For FFS changes, keep weights external and validate both deterministic tests and manual hardware outcomes.

## Invariants

- The repo stops at `data_process/record_data_align.py`.
- Do not reintroduce segmentation, tracking, shape prior, inverse physics, Gaussian Splatting, evaluation, or teleop code.
- `qqtt/__init__.py` exports only `CameraSystem`.
- `env_install/env_install.sh` stays camera-only.
- Hardware checks remain manual and documented; do not fake them in CI.
- External repos and weights stay outside this repo and are referenced by path.
- `depth/` must remain the canonical compatibility output for aligned cases.

## Do Not Change Without Updating Docs / Tests

- camera CLI defaults
- output directory layout for `data_collect/` and `data/`
- metadata fields written by recording / alignment
- scope guard rules

## Deep Docs

- Scope boundary: `docs/SCOPE.md`
- Architecture: `docs/ARCHITECTURE.md`
- User workflows: `docs/WORKFLOWS.md`
- Manual validation: `docs/HARDWARE_VALIDATION.md`
- Active/completed execution plans: `docs/exec-plans/`
