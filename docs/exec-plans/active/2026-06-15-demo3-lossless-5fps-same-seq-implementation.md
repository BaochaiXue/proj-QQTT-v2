# Demo 3.x Lossless 5 FPS Same-Seq Implementation Plan

## Objective

Implement the approved design in
`docs/superpowers/specs/2026-06-15-demo3-lossless-5fps-same-seq-design.md`:
Demo 3.x should process every generated 5 FPS input frame, run PCD and tracker
branches in parallel, publish only ordered same-seq pairs, and fatal on bounded
backlog overflow instead of silently dropping frames.

## Constraints

- Keep the hard visual/output invariant: no rendered or written packet may mix
  PCD seq and tracker seq.
- Do not use latest-wins transport for the formal Demo 3.x tracker-enabled
  lossless path.
- Keep ordinary preview/latest-wins code available outside this path.
- Avoid new public CLI flags unless a required behavior cannot be expressed with
  existing Demo 3.x defaults.
- Preserve the PCD-only visual mode behavior: it may hide tracker marker
  geometry but still runs the tracker-enabled full demo pipeline.
- Keep fake-live default cadence at 5 FPS.

## Phase 1: Lossless Transport Primitives

Add small internal helpers to `qqtt/demo/realtime_masked_edgetam_pcd.py`:

- `OrderedPacketQueue[T]`
  - bounded FIFO queue,
  - monotonically increasing `seq` enforcement,
  - blocking `put`/`get` with stop-event awareness,
  - stats for current length, max length, expected seq, and overflow reason.
- `SameSeqPairer`
  - accepts PCD results and tracker packets independently,
  - buffers by seq with bounded pending maps,
  - emits complete pairs in strict seq order only,
  - raises/records fatal on seq mismatch, missing expected seq timeout, or
    backlog overflow.

Keep these helpers local to the demo runtime unless tests reveal a clean need to
share them.

## Phase 2: Worker Topology

Replace the serial `_strict_paired_worker` topology for tracker-enabled masked
PCD mode with:

- capture worker -> frame queue,
- segmentation worker -> duplicate each `MaskPacket` into PCD and tracker queues,
- PCD worker -> build `PcdBuildResult` -> pairer,
- tracker worker -> build `TrackerMarkerPacket` -> pairer,
- pairer/output worker -> publish `PairedRenderPacket`.

Retain `_build_pcd_packet_from_mask()` and `_build_tracker_marker_packet()` as
the per-frame compute helpers. The implementation should remove the performance
dependency where tracker compute must finish before PCD/filter compute starts.

## Phase 3: Fake-Live Lossless Semantics

Change fake-live capture behavior for Demo 3.x lossless mode:

- enqueue recording frames in seq order at 5 FPS,
- do not overwrite unconsumed frames,
- if `--duration-s` is set, stop offering new frames at duration, then drain all
  already-offered frames before successful exit,
- if duration is 0, process the whole recording unless interrupted.

The source should expose enough counters for tests and debug logs:

- frames offered,
- frames segmented,
- PCD results produced,
- tracker results produced,
- same-seq pairs emitted,
- final drained seq.

## Phase 4: Real-Camera Lossless Semantics

For live real camera Demo 3.x:

- the formal processing task cadence is 5 FPS,
- if the RealSense stream runs faster, sampling to 5 FPS happens before the
  lossless queue,
- every generated 5 FPS task must be processed,
- queue overflow is fatal.

Metadata and debug logs should report the task cadence so operators know the
lossless contract applies to generated 5 FPS tasks, not every raw sensor frame
from a higher-rate stream.

## Phase 5: Render, Headless Output, And Diagnostics

Keep the viewer consuming only complete `PairedRenderPacket` objects. Preserve
the existing UI behavior of holding the previous pair while waiting.

Update diagnostics:

- `lossless=1`,
- `input_fps=5`,
- `expected_seq`,
- `paired_seq`,
- per-stage queue lengths,
- pairer pending PCD/tracker counts,
- max backlog frames,
- fatal backlog stage/reason,
- tracker model/e2e timing,
- PCD/filter timing.

Headless output should write PCD and tracker artifacts only from emitted
same-seq pairs.

## Phase 6: Tests

Add or update tests for:

- ordered queue preserves sequence order,
- ordered queue rejects gaps or out-of-order packets,
- ordered queue overflows at bounded backlog,
- pairer emits only complete same-seq pairs,
- pairer does not emit seq `N+1` while seq `N` is missing,
- pairer accepts PCD-first and tracker-first arrival orders,
- tracker-enabled Demo 3.x starts parallel PCD/tracker/pairer workers,
- fake-live simulated short case processes all offered seq values,
- existing tracker-disabled PCD rendering tests remain unchanged.

## Phase 7: Validation

Run deterministic checks:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest \
  tests.test_single_demo_tapnextpp_overlay \
  tests.test_single_demo_v3_runtime \
  tests.test_realtime_masked_edgetam_pcd_filter \
  tests.test_recorded_rgbd_replay_source

conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py
```

Manual validation:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v3_2/realtime_single_camera_ffs_masked_pcd.py \
  --input-source fake-live \
  --demo-visual-mode pcd \
  --duration-s 90 \
  --debug
```

Expected manual outcome:

- `capture_fps` targets 5 FPS,
- no skipped `paired_seq` values among offered frames,
- backlog remains under threshold,
- PCD and tracker seq always match,
- viewer displays PCD while holding last pair between complete outputs,
- if backlog grows beyond threshold, the run fails loudly with a fatal HUD/log.

## Completion Criteria

- Demo 3.x tracker-enabled masked PCD no longer uses the serial strict paired
  worker as the primary topology.
- Fake-live lossless mode does not silently drop offered frames.
- Same-seq render/output invariant remains enforced.
- Tests and quick deterministic checks pass.
- Implementation is committed and pushed to `origin/single-camera`.
