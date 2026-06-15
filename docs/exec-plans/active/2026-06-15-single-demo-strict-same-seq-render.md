# Single Demo Strict Same-Seq Tracker Render

## Goal

Prevent single Demo 3.x from displaying tracker markers from one frame on top of
point clouds from a different frame. When tracker overlay is enabled, the viewer
must render only complete same-sequence PCD/tracker pairs and hold the previous
complete pair while the next pair is being computed.

## Implementation Notes

- Add an internal paired render packet that validates `pcd.seq == tracker.seq`.
- Keep the existing PCD-only worker/render path for tracker-disabled runs.
- Route tracker-enabled masked PCD runs through one strict paired worker that
  consumes a single `MaskPacket`, computes tracker markers and point clouds from
  that same packet, and publishes only a complete pair.
- Update HUD/debug diagnostics with strict-sync fields.
- Add unit tests for mismatch rejection, matching pair publication, and worker
  selection.

## Validation

- `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_single_demo_tapnextpp_overlay tests.test_single_demo_v3_runtime tests.test_realtime_masked_edgetam_pcd_filter`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
