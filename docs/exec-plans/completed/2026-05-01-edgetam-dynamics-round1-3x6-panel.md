# 2026-05-01 EdgeTAM Dynamics Round1 3x6 Panel

## Goal

Generate an `ffs_dynamics_round1` time GIF panel that adds EdgeTAM as a sixth
column next to the existing SAM3.1 and SAM2.1 large/base_plus/small/tiny
columns.

## Plan

- Keep the change experiment-only; do not import EdgeTAM from formal
  recording/alignment runtime.
- Add a minimal EdgeTAM video-mask worker that runs under the existing
  `edgetam-max` conda environment.
- Initialize EdgeTAM video tracking from the `SAM3.1 frame0 union mask` for the
  `sloth` target.
- Save EdgeTAM masks in the same mask schema used by the panel renderer.
- Record EdgeTAM propagation-only timing after warmup and label the 3x6 panel
  with ms/frame and FPS.
- Reuse the existing experiment-local FFS depth cache and original camera
  pinhole PCD renderer.

## Validation

- Add deterministic smoke coverage for the generic 3-row panel composer and
  EdgeTAM-compatible mask/timing schema.
- Run the focused unittest and `scripts/harness/check_all.py`.
- Run the full `ffs_dynamics_round1` EdgeTAM 3x6 artifact command.
