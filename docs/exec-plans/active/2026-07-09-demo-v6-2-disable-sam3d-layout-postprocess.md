# Demo v6.2 disable SAM3D layout postprocess

## Requirement

Problem:
The schema-v1 warm-up profile measured 83.795 seconds inside SAM3D Gaussian
layout post-optimization before that vendor step failed. Demo v6.2 consumes
the generated GLB/Gaussian, while the returned layout pose is ignored and the
following align stage independently computes pose and scale.

Required final behavior:
SAM3D generation must skip layout post-optimization while retaining mesh
post-processing, texture baking, generated geometry, and the dedicated
SuperGlue/PnP/ARAP alignment stage.

Inputs:
The same frame-0 RGB/mask, fixed SAM3D seed, model checkpoints, fake-live case,
and dual-GPU assignment used by the measured baseline.

Outputs:
The normal `object.glb`, `object.ply`, aligned `final_mesh.glb`, sampled
`points.npz`, timing profiles, and first formal chunk.

State changes:
The single SAM3D `with_layout_postprocess` call argument changes from true to
false. No compatibility switch or alternate execution path is retained.

Invalid cases:
Missing shape products, failed profile schema validation, or a failed first
formal chunk invalidates the optimization.

Constraints:
Preserve all other SAM3D options and compare the new run against the existing
`outputs_v6_2_profile_20260709_2021` timing baseline. Exact cross-run geometry
equality is not an acceptance gate because the user explicitly removed it.

Unknowns:
No correctness-blocking unknowns remain after the repeated GPU run.

## Plan

- [x] Disable layout post-processing in the one production SAM3D call.
- [x] Add a regression test and update the timing/design documentation.
- [x] Run focused tests and the repository smoke profile.
- [x] Repeat the same fake-live one-chunk run in a fresh output directory.
- [x] Validate timing closure and the normal generated/aligned products.
- [x] Commit and push the validated change to `single-camera`.

## Validation

Branch/setup:

- Current branch: `single-camera`.
- The required `git pull --ff-only origin main` was attempted and refused
  because `single-camera` and `origin/main` have diverged; no merge, rebase,
  or reset was performed.

Deterministic validation:

- Focused timing/cleanup/manifest suite: `21` tests passed.
- Scoped Ruff check and `git diff --check` passed.
- Repository smoke profile: `206` tests passed, including all guards and help
  probes.

GPU fake-live validation:

- Command:
  `conda run -n demo_2_max --no-capture-output python demo_v6_2/main.py
  --base-path outputs_v6_2_profile_no_layout_20260709_2030 --case-prefix
  demo_v6_2_profile --downstream-mode disabled --max-chunks 1`.
- Top-level exit code: `0`; shape prior status: `ready`; shape prior error:
  `null`; first chunk status: `normal`; skipped online publishes: `0`.
- Warm-up decreased from `159.701 s` to `74.042 s`, saving `85.658 s`
  (`53.64%`). The generate critical path decreased from `113.671 s` to
  `28.820 s`; its internal pipeline run decreased from `98.435 s` to
  `13.914 s`.
- The profile schema, seven-stage critical path, four completed subprocess
  profiles, and runtime-to-gate timing identity all passed strict validation.
- The user explicitly removed exact geometry equality from the acceptance
  criteria. The repeated run still produced `object.glb`, `object.ply`,
  aligned `final_mesh.glb`, `points.npz`, and one normal formal chunk.
