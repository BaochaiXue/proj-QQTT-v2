# Demo 3 Sidecar Tracking Ownership

## Goal

Remove ambiguity in Demo 3 live tracking ownership before hardware validation.
The shared three-view runtime should own capture, SAM 3.1/HF EdgeTAM masking,
RealSense-depth fusion, and rendering. Demo 3's sidecar should be the only
owner of CoTracker3.

## Scope

- Force shared runtime tracking args to `tracking_backend=none` and
  `tracking_source=cached` when launched from Demo 3.
- Stop passing `--show-tracking-overlay` to the shared runtime from Demo 3.
- Keep Demo 3's own CoTracker sidecar controlled by the public
  `--disable-cotracker` flag.
- Make overlay packet stale checks use publish time, while preserving the source
  input timestamp for diagnostics.
- Update tests and Demo 3 docs.

## Non-Goals

- Do not change Demo 2.2 behavior.
- Do not change RealSense depth-only or batch vision encoder requirements.
- Do not change CoTracker query counts or tracking model behavior.
- Do not run hardware validation in CI.

## Validation

- Targeted Demo 3 contract and CoTracker worker tests.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`.

## Outcome

- Demo 3 shared-runtime argv now always passes
  `--tracking-backend none --tracking-source cached`.
- Demo 3 no longer passes `--show-tracking-overlay` to the shared runtime.
- Demo 3's public `--disable-cotracker` flag still controls only the Demo 3
  sidecar.
- `TrackingOverlayPacket.timestamp_s` now records overlay publish time, and
  `source_timestamp_s` preserves the original input timestamp.
- Demo 3 docs now state that the shared runtime tracking backend is disabled
  and the sidecar is the only CoTracker owner.
- Completed validation:
  - `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_demo3_contract tests.test_demo3_cotracker_worker`
  - shared runtime parser check: `edgetam_batch_vision_encoder=True`,
    `tracking_backend=none`, `tracking_source=cached`,
    `show_tracking_overlay=False`
  - `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
