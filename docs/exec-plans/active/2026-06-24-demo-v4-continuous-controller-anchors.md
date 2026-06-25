# Demo V4 Continuous Controller Anchors

## Goal

Make Demo v4 online output suitable for true continuous `realtime_phystwin`
online optimization by keeping `final_data.pkl["controller_points"]` as a fixed
ordered controller-anchor trajectory across chunks.

## Context

- Local `realtime_phystwin` does not require controller point coordinates to be
  byte-identical across chunks.
- It does require a stable controller handle count/order for continuous online
  training because the first available controller frame initializes
  controller-object spring topology.
- Current Demo v4 chunks independently run data-process-style FPS selection on
  every chunk, so controller anchor identity can drift even when the shape stays
  `(T, 30, 3)`.
- `data_process_sam3d` first filters controller candidate tracks by first-frame
  semantics, per-frame mask validity, whole-sequence visibility, and local
  motion consistency, then FPS-selects 30 controller candidates.

## Design

1. Keep the existing data quality gates:
   - processed object/controller masks,
   - radius-outlier mask filtering,
   - first-frame semantic query labeling,
   - per-frame semantic visibility,
   - controller motion filtering,
   - FPS selection for the first anchor set.
2. Add a Demo v4 stateful controller-anchor selector:
   - first chunk selects 30 anchors from valid controller candidates using the
     current FPS logic,
   - stores their stable source query ids and last emitted 3D positions,
   - later chunks preserve the same output order by looking up those query ids.
3. If an anchor query is unavailable or fails the controller mask/motion filter
   in a later chunk, revive it from nearby valid controller tracks:
   - predict the first-frame anchor position from surviving anchors at the chunk
     boundary,
   - use inverse-distance KNN/LBS over nearby controller candidate trajectories,
   - keep output shape and ordering fixed.
4. Add manifest metadata for auditability:
   - selected initial anchor query ids,
   - per-chunk direct/revived/fallback counts,
   - per-anchor active query ids when revival is used.
5. Keep FuturePhysTwin/PhysTwin compatibility:
   - do not remove required files or keys,
   - do not change object sampling or shape-prior payloads,
   - keep full tracking/cotracker npz artifacts available for diagnostics.

## Validation

1. Unit tests must fail before implementation:
   - consecutive chunks with different independent FPS candidates should output
     stable first-chunk anchor ids/order,
   - an anchor lost in a later chunk should be revived without changing the
     controller point count or order.
2. Run targeted Demo v4 tests.
3. Run repository smoke validation.
4. Run Demo v4 to produce online output.
5. Run local `realtime_phystwin` online optimization against the Demo v4 online
   output and record the exact command/outcome.

## Non-Goals

- Do not change formal single-camera recording/alignment outputs.
- Do not introduce Gaussian rendering requirements into Demo v4.
- Do not require exact equal controller coordinates across chunk boundaries;
  require stable controller identities/order and compatible spring topology.
