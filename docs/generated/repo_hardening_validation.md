# Repo Hardening Validation

Date: 2026-04-09

Environment:

- repo: `C:\Users\zhang\proj-QQTT`
- conda env: `qqtt-ffs-compat`
- validation python: `C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe`

## What Was Hardened

### Visualization / Compare Contracts

- kept the current compare workflows and artifact names stable
- made calibration-world semantics explicit in compare metadata
- added `scene_overview_calibration_frame.png` to the turntable workflow
- kept current behavior explicit:
  - raw calibration-board `c2w` world is used
  - no semantic-world transform is silently applied

### Calibration Loader

- strengthened `calibration_io.py` validation for:
  - duplicate serial lists
  - non-finite transform values
  - invalid homogeneous bottom rows
  - clearer mapping-mode summaries

### Record-Time Preflight

- moved preflight policy into `qqtt/env/camera/preflight.py`
- made the decision table explicit:
  - `rgbd` allowed directly
  - `stereo_ir` warning-only when unsupported
  - `both_eval` blocked when unsupported
- `record_data.py` now prints explicit operator-facing preflight summaries

### ROI / Object-Compare Contract

- added stronger contract-level tests around:
  - object/context/source alignment
  - turntable frame-contract metadata

### Artifact Writing / Compare Observability

- artifact JSON writing is now more consistently routed through `io_artifacts.py`
- compare metadata explicitly records calibration and frame-contract semantics

## Commands Run

Targeted contract tests:

```bash
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe -m unittest -v tests.test_record_preflight_policy_smoke tests.test_calibration_contract_hardening tests.test_object_compare_contract_smoke tests.test_turntable_frame_contract_smoke
```

Compatibility spot checks:

```bash
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe -m unittest -v tests.test_visual_compare_turntable_smoke tests.test_visual_compare_depth_video_smoke tests.test_recording_metadata_schema_v2
```

Full deterministic validation:

```bash
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe scripts/harness/check_all.py
```

## Results

- targeted contract tests: passed
- compatibility spot checks: passed
- `scripts/harness/check_visual_architecture.py`: passed through `check_all.py`
- `scripts/harness/check_all.py`: passed

## New / Updated Checks

New or expanded validation coverage now includes:

- preflight decision-table policy
- stricter calibration loader validation
- explicit turntable frame-contract metadata
- object/context/source alignment contract
- existing full workflow compatibility checks

## Compatibility Notes

Preserved:

- current main CLI commands
- current aligned-case output naming
- current compare workflows
- current `both_eval` blocked / `stereo_ir` warning-only policy

Behavior now more explicit:

- recording preflight summary is printed before recording
- compare metadata declares calibration-world semantics explicitly
- turntable outputs now include a calibration-frame overview alias

## Remaining Risks

1. `turntable_compare.py` is still the biggest workflow module.
2. `object_compare.py` still mixes pure logic and debug artifact generation.
3. A true semantic-world transform is still not part of the default compare path; the current hardening makes that absence explicit rather than silently faking it.
