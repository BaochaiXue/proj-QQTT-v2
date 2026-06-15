# Demo 3.x Lossless 5 FPS Same-Seq Pipeline Design

## Context

Single Demo 3.x currently protects the viewer from displaying mismatched point
clouds and tracker markers by computing tracker markers and PCD packets inside
one strict paired worker. That preserves the visual invariant, but it serializes
work that should be parallel:

```text
mask(seq) -> tracker(seq) -> PCD/filter(seq) -> paired render(seq)
```

The formal three-camera demo expected the PCD/depth branch and tracker branch to
continue independently, with synchronization happening at the render/output
boundary. The single-camera path should follow the same shape:

```text
mask(seq) -> tracker branch -> tracker(seq)
mask(seq) -> PCD branch     -> PCD(seq)
tracker(seq) + PCD(seq) -> ordered same-seq pairer -> render/output(seq)
```

The second issue is that the current transport uses `LatestSlot`, whose intended
behavior is latest-wins preview. That is incompatible with the desired Demo 3.x
contract: fake-live camera input, and later real-camera demo input, must process
every generated 5 FPS frame. If the pipeline cannot keep up, it should fail
explicitly instead of dropping frames silently.

## Goals

- Process every Demo 3.x input frame at the configured 5 FPS demo cadence.
- Keep the hard invariant that PCD geometry and tracker markers are only
  rendered or written when their `seq` values match.
- Restore parallel execution between tracker work and PCD/depth/filter work.
- Preserve the viewer behavior of holding the last complete same-seq pair while
  waiting for the next pair.
- Treat unbounded backlog as a fatal pipeline error, not as permission to skip
  frames.
- Keep ordinary preview/latest-wins utilities available outside the formal
  Demo 3.x lossless path.

## Non-Goals

- Do not disable TAPNext++ for `--demo-visual-mode pcd`; PCD visual mode may
  hide marker geometry, but it still represents the full tracker-enabled demo
  pipeline unless a separate explicit mode is designed later.
- Do not relax same-seq rendering.
- Do not introduce offline batch processing semantics for live camera preview
  tools outside Demo 3.x.
- Do not fake hardware or visual validation in CI.

## Proposed Architecture

Introduce a Demo 3.x `strict_lossless_5fps` runtime policy and replace
latest-wins transport in that policy with bounded ordered queues.

### Components

- `OrderedPacketQueue[T]`
  - FIFO queue keyed by packet `seq`.
  - Enforces monotonically increasing input sequence order.
  - Has a finite `max_backlog_frames` capacity.
  - Fails fast if producers exceed capacity or if consumers observe sequence
    gaps.

- `SameSeqPairer`
  - Accepts `PcdBuildResult` and `TrackerMarkerPacket` from parallel branches.
  - Emits `PairedRenderPacket` only when both sides for the next expected `seq`
    are complete.
  - Holds later results in bounded maps, but never publishes seq `N+1` before
    seq `N`.
  - Fails fast on mismatched packet internals, missing seq timeout, or bounded
    map overflow.

- `LosslessCaptureClock`
  - For fake-live: reads recording frames in numerical sequence and schedules
    them at 5 FPS.
  - For real camera: the demo should configure/acquire 5 FPS input tasks. If
    the physical camera cannot produce a stable 5 FPS task stream, startup or
    backlog checks should fail rather than silently dropping frames.

### Thread Model

```text
capture worker
  -> frame_queue

segmentation worker
  frame_queue -> mask_queue_for_pcd
              -> mask_queue_for_tracker

PCD worker
  mask_queue_for_pcd -> pcd_result -> pairer

tracker worker
  mask_queue_for_tracker -> tracker_packet -> pairer

pairer/output worker
  pairer complete pairs -> paired_render_queue/headless writer

Open3D viewer
  consumes paired_render_queue and holds the last complete pair while waiting
```

The PCD and tracker workers must start from the same `MaskPacket` sequence but
run independently. Synchronization occurs only in `SameSeqPairer`.

## Backlog and Failure Policy

The formal Demo 3.x path should default to a small finite backlog. A good first
default is:

- `input_fps = 5`
- `max_backlog_seconds = 3`
- `max_backlog_frames = 15`

Every ordered queue and pairer map uses this budget. If any stage exceeds the
budget, the runtime records a fatal worker error and closes the viewer with a
HUD message such as:

```text
lossless 5 FPS backlog exceeded
stage=pcd queue_len=16 max=15 expected_seq=123 latest_seq=139
```

This makes performance failures visible. It also prevents quiet behavior changes
where the pipeline appears to run but no longer processes every frame.

## Fake-Live Semantics

Fake-live is no longer a latest-wins camera preview. It is a deterministic
recorded input stream:

- Frame seq starts at 0 and increments by 1.
- Every seq is enqueued.
- The source finishes only after every frame has been offered to the pipeline.
- The demo exits successfully only after every offered frame has produced a
  complete output pair.

For short visual runs with `--duration-s`, the source should stop offering new
frames when the duration expires, then drain all already-offered frames before
exit. Duration limits how many input frames are offered; it does not permit
already-offered frames to be skipped. For full replay/profiling runs, duration
can be 0 or a future explicit `--process-all` mode can be added if needed.

## Real-Camera Semantics

The formal Demo 3.x live mode should generate 5 FPS processing tasks. If the
camera profile exposes only higher FPS streams, the capture worker should sample
or request frames to create a 5 FPS task stream before the lossless queue. The
lossless guarantee applies to those 5 FPS tasks, not to every raw sensor frame
from a 30/60 FPS hardware stream.

If processing falls behind those tasks and backlog exceeds the configured
budget, the demo should fatal. This matches the operator expectation: if a
formal 5 FPS demo cannot process all generated frames, the pipeline is too slow
or incorrectly configured.

## Render and Headless Output

The renderer continues to consume only complete `PairedRenderPacket` objects.
It may hold the previous pair while waiting for the next one, but it must not
advance over missing sequence numbers.

Headless/file outputs use the same paired output path. PCD artifacts and tracker
artifacts for a sequence are written together from the same emitted pair. A
failed or missing sequence is a fatal condition, not a skipped row.

## Diagnostics

HUD and debug logs should include:

- `strict_sync=1`
- `lossless=1`
- `input_fps=5`
- `expected_seq`
- `paired_seq`
- per-stage queue lengths
- pairer pending PCD/tracker counts
- max backlog frames
- backlog fatal reason, when applicable
- tracker `model_ms` and `e2e_ms`
- PCD/filter timing and output point counts

The logs should make it obvious whether slowdowns are due to segmentation,
tracker inference, FFS depth, PCD filtering, pair waiting, or Open3D rendering.

## Testing

Add focused tests for:

- Ordered queue preserves seq order and rejects sequence gaps.
- Ordered queue raises a fatal/overflow signal when backlog exceeds capacity.
- Same-seq pairer does not emit seq `N+1` while seq `N` is incomplete.
- Same-seq pairer emits seq `N` when matching PCD and tracker packets arrive in
  either order.
- Mismatched packet internals are treated as errors.
- Fake-live lossless mode processes all offered sequence numbers in a simulated
  short recording.
- Tracker-enabled Demo 3.x starts parallel PCD/tracker branches plus pairer,
  not the serial strict paired worker.
- Existing tracker-disabled PCD rendering behavior remains covered.

Manual validation should include a fake-live Demo 3.2 run at 5 FPS and confirm:

- no skipped sequence numbers in debug logs,
- `paired_seq` increases monotonically by one,
- backlog stays below threshold,
- render/headless output remains same-seq.

## Open Decisions

- The initial backlog default is 15 frames. This can be tuned after the first
  hardware/fake-live validation run.
- A future explicit CLI such as `--process-all` may make full-recording replay
  clearer, but the first implementation should avoid adding public flags unless
  needed.
- For real camera streams that cannot request 5 FPS directly, the precise
  capture-side sampling policy should be implemented carefully and reported in
  metadata.
