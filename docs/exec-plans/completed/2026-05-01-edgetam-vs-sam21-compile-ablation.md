# 2026-05-01 EdgeTAM vs SAM2.1 Compile Fairness Ablation

## Goal

Benchmark EdgeTAM compile modes against SAM2.1 Small/Tiny on the existing RTX
5090 Laptop WSL setup without reinstalling environments or changing production
backend defaults.

## Plan

- Keep the work experiment-only under `scripts/harness/experiments/`.
- Use `edgetam-max` for EdgeTAM and `SAM21-max` for SAM2.1.
- Use `ffs_dynamics_round1_20260414` with the existing SAM3.1 frame0 union mask
  as the initial mask prompt for all modes.
- Benchmark:
  - EdgeTAM eager
  - EdgeTAM `compile_image_encoder=true`
  - guarded manual `torch.compile(image_encoder.forward)`
  - SAM2.1 Small with `vos_optimized=True`
  - SAM2.1 Tiny with `vos_optimized=True`
- Use official-style pass-only timing: one prompt state, 25 propagation runs,
  first 5 as warmup, timed loop body does not threshold/copy/save masks.
- Generate:
  - `docs/generated/edgetam_vs_sam21_speed_ablation.md`
  - `docs/generated/edgetam_vs_sam21_speed_ablation.json`

## Validation

- Run script help/py_compile checks.
- Run the full ablation command.
- Run deterministic harness checks after adding the new experiment CLI to the
  harness catalog.

## Outcome

- Added `scripts/harness/experiments/run_edgetam_vs_sam21_compile_ablation.py`.
- Generated:
  - `docs/generated/edgetam_vs_sam21_speed_ablation.md`
  - `docs/generated/edgetam_vs_sam21_speed_ablation.json`
- Verified the official-style pass-only benchmark on all three cameras from
  `data/dynamics/ffs_dynamics_round1_20260414`.
- Confirmed EdgeTAM eager is slower than SAM2.1 Small/Tiny on this RTX 5090
  Laptop WSL benchmark.
- Confirmed unpatched EdgeTAM `compile_image_encoder=true` fails on this setup
  with a CUDA graph overwritten-output error rooted in the position encoding
  cache path.
- Confirmed a process-local no-position-cache patch lets EdgeTAM compiled image
  encoder complete and reach the fastest measured pass-only throughput, but the
  gain is below the 15% promotion threshold over SAM2.1 Tiny and remains
  experimental.
