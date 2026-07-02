# Demo v5.1 Tracking Refactor Per design_spec.md

## Requirement

Problem:
The user rejects the current confidence-ladder tracking
(`StreamingControllerTrackSelector` / `StreamingObjectTrackSelector`:
bundle recovery confidences, dead-reckoning prediction, frozen fallbacks,
`conf >= 0.25` visibility). `demo_v5_1/design_spec.md` (user-authored,
2026-07-02) is the sole spec for the replacement.

Required final behavior (from the spec):
- Frame-0 (warmup frame) query labeling is frozen forever: object query =
  visible at frame 0 and inside the frame-0 object processed mask;
  controller query = visible and inside the controller processed mask.
- One per-frame state `temporary_invalid` covers: tracker-visible but
  mask/depth/motion-gate failure, and tracker invisible / lost. It never
  deletes a query or changes anchor identity.
- Chunk 0 controller selection follows origin strictness (valid at every
  window frame + origin motion consistency with once-fail removal), then
  farthest-point-samples 30 anchors. Anchors, `query_ids`,
  `query_semantic_labels`, `controller_sample_query_ids`,
  `query_schema_hash` never change afterwards.
- A one-time table stores each controller point's nearest 50 controller
  points by first-frame 3D positions (never updated).
- Later chunks: a temporarily-invalid anchor frame is filled by local rigid
  registration — pick the nearest 15 currently-valid of its 50 neighbors,
  estimate a rigid transform from first frame to current frame, apply it to
  the anchor's first-frame position. Identity is never replaced.
- Fewer than 15 valid neighbors among the 50 → raise an exception (spec).
- Motion consistency and detail semantics follow `data_process_origin`
  verbatim (constants 0.01 m radius / 5 neighbors / 0.005 m similarity),
  within realtime constraints.
- No confidence values anywhere in the new path.

Layout: split the new implementation into three files for readability —
`demo_v5_1/tracking.py` (state machine, selection, neighbor table, rigid
recovery, motion filter), `demo_v5_1/segment.py` (per-class semantic mask
gates, frame-0 labeling), `demo_v5_1/ffs.py` (depth/PCD validity + 3D
lifting). The realtime capture side (Stage A: camera, TAPNext++, prepared
frames) is untouched.

Origin-parity decisions folded in:
- Per-frame class-mask gating does NOT subtract the controller mask from
  the object mask (origin keeps the overlap;
  `normalize_processed_mask_frame` remains for non-tracking consumers).
- Object columns: frozen at chunk 0 by frame-0 volume sampling (5 mm voxel,
  shape-prior bounds); per-frame values are direct measurements or zeros
  (origin fills zeros), visibilities honest, no revive/hold-last synthesis.
- Published `*_motions_valid` uses the origin formula; the last row of each
  window stays False (origin's last-frame semantics). The temporary_invalid
  gate applies the motion term only on frames where motion is computable
  (t < T-1).
- `track_process_status` stays warning-only: `normal` when every anchor
  frame is direct, `degraded` when any frame was rigid-proxied; `invalid`
  is no longer produced by tracking (spec failures raise instead).

State changes:
- New: `demo_v5_1/tracking.py`, `demo_v5_1/segment.py`, `demo_v5_1/ffs.py`,
  `tests/test_demo_v5_1_tracking.py`.
- `demo_v5_1/chunk_data_stream.py`: window builders call
  `TrackingRuntime.process_window()`; selector plumbing removed.
- `demo_v5_1/chunk_data_payload.py`: confidence/mode/neighbor-count
  diagnostic keys replaced by `controller_proxied` (+ static recovery
  neighbor table ids); quality manifest fields become direct/proxied
  counts; metrics dict updated.
- `demo_v5_1/phystwin_strict_product.py`: streaming selector classes and
  confidence helpers deleted once unreferenced.

Invalid cases:
- Chunk-0 survivors < 30 → `ControllerSelectionError` (origin asserts the
  same bound offline).
- < 15 valid recovery neighbors → `TrackingRecoveryError` (spec).

Constraints:
- Do not regress realtime: recovery is a 15-point rigid fit per invalid
  anchor frame; neighbor table is built once; motion filter cost unchanged.
- Keep session query schema freezing and `query_schema_hash` semantics.
- Follow repo formatting (<= 100 columns).

Unknowns:
- Chunk-0 all-window strictness with TAPNext++ visibility may reject many
  candidates on real hardware; the spec accepts hard failure. Validated on
  synthetic tests here; real-capture behavior to be observed on next run.

## Plan

- [ ] Implement `demo_v5_1/ffs.py`, `demo_v5_1/segment.py`,
      `demo_v5_1/tracking.py`.
- [ ] Rewire `chunk_data_stream.py` window builders and runtime creation.
- [ ] Replace confidence diagnostics in `chunk_data_payload.py`.
- [ ] Delete unreferenced selector/confidence code in
      `demo_v5_1/phystwin_strict_product.py`.
- [ ] Add `tests/test_demo_v5_1_tracking.py`; update chunk-data tests.
- [ ] Run focused tests + smoke validation.
- [ ] Adversarial review of the diff, then commit/push.

## Validation

- `python -m pytest tests/test_demo_v5_1_tracking.py tests/test_demo_v5_1_chunk_data.py -q`
- `python scripts/harness/validation/run.py --profile smoke`
