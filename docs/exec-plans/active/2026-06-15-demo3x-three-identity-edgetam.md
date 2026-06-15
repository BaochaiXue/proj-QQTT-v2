# Demo 3.x Three-Identity EdgeTAM

## Goal

Split realtime Demo 3.x EdgeTAM tracking identities into `hand_a`, `hand_b`,
and `object`, while keeping the legacy controller PCD/depth path as the union of
the two hand masks.

## Implemented Changes

- Initialize two hand instance masks from first-frame SAM3.1 output.
- Propagate `hand_a`, `hand_b`, and `object` as separate EdgeTAM object ids.
- Preserve `controller_mask = hand_a_mask | hand_b_mask` for PCD generation.
- Label TAPNext++ queries with the first-frame three-target identity.
- Gate query visibility and 3D lift against the current per-target mask.
- Save hand instance masks and query identity labels in headless artifacts.
- Report per-target query counts in live/headless summaries.
- Keep PhysTwin dense query sampling at a 5000-point default.
- Document the Demo 3.2 fake-live three-identity behavior and the two-hand
  frame-0 requirement.

## Validation

- Add unit tests for hand instance splitting, three-id prompts, controller union,
  per-target query labels, per-target gating, and headless artifact fields.
- Run targeted unittest coverage and deterministic harness checks.
- Run Demo 3.2 headless capture and render pcd/tracking videos when feasible.

## Status

- Implemented and validated.

## Results

- PASS: `python -m py_compile qqtt/demo/realtime_masked_edgetam_pcd.py qqtt/demo/single_demo_v3_runtime.py scripts/harness/render_demo32_headless_capture.py tests/test_single_demo_tapnextpp_overlay.py tests/test_single_demo_v3_runtime.py tests/test_demo32_headless_render_helper.py`
- PASS: `python -m unittest tests.test_single_demo_v3_runtime tests.test_single_demo_tapnextpp_overlay tests.test_demo32_headless_render_helper`
- PASS: `python scripts/harness/check_all.py`
- PASS: `python scripts/harness/check_all.py --full`
- PASS: Demo 3 / 3.1 / 3.2 / 3.3 dry-run contracts with `--mode demo` report
  `controller_instance_mode=two-hands`, `edgetam_tracking_identities=['hand_a', 'object', 'hand_b']`,
  and `tracker_query_count=5000`.
- PASS: Demo 3.2 short fake-live headless smoke wrote 17 same-seq paired frames
  under `result/single_demo_v3_2_ffs_masked_pcd/headless_three_identity_smoke_20260615_152544`.
  First saved frame recorded hand query counts `hand_a=418`, `hand_b=504`,
  `object=3840`; the last saved frame recorded `hand_a=372`, `hand_b=419`,
  `object=3810`.
- PASS: Offline helper rendered both `video_pcd_only.mp4` and
  `video_query_phystwin.mp4` from that smoke capture with zero missing query
  frames.
