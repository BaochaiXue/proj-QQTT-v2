# Demo 3.2 Side-by-Side Headless and Realtime Panel Design

## Goal

Build a Demo 3.2 real side-by-side panel that can be rendered both offline from
headless capture artifacts and live during fake-live runtime. The panel is a
1x3 video frame:

1. left: original RGB input
2. middle: current filtered PCD projected into the camera view
3. right: tracking markers / tracking overlay

The three columns share one fake-live source timeline. The left RGB column is
allowed to show the latest input frame with no pipeline latency, while the PCD
and tracking columns must use the same strict paired frame. The panel must make
that timing relationship explicit on screen.

## User-Approved Choices

- Implement both offline headless MP4 generation and a runtime realtime panel.
- Use a shared renderer so offline and runtime panels have the same visual
  contract.
- Show latest source RGB on the left; allow `rgb_seq` to lead `paired_seq`.
- Require the middle filtered PCD and right tracking overlay to come from the
  same `paired_seq`.
- Display both `pipeline_latency_ms` and `display_latency_ms`.
- Support tracking overlay backgrounds `target-union` and `rgb`; default to
  `target-union`.

## Scope

In scope:

- Demo 3.2 fake-live headless captures.
- Demo 3.2 fake-live runtime panel display.
- One shared 1x3 BGR panel renderer.
- Metadata/HUD fields needed to explain timing and synchronization.
- Tests for offline rendering, runtime data assembly, and HUD contracts.
- Documentation for the new operator workflow.

Out of scope:

- RealSense live-camera capture changes.
- Demo 3.1, Demo 3.3, or legacy three-camera demo changes.
- Changing the FFS, EdgeTAM, TAPNext++, or filtering algorithms.
- Replacing SAM3.1 initialization or adding shape prior.
- Hardware proof videos. This feature produces the panel; recording a phone
  third-view remains a manual workflow.

## Architecture

Add a shared renderer module:

`qqtt/demo/demo32_side_by_side_panel.py`

This module is pure rendering and small data formatting. It does not read files,
own runtime queues, call FFS, call EdgeTAM, or mutate capture state. It accepts
already loaded panel inputs and returns one BGR `numpy.ndarray`.

Primary public API:

- `SideBySidePanelInputs`
  - `rgb_image_bgr`: latest input RGB frame for the left column.
  - `pcd_panel_bgr`: projected filtered PCD image for the middle column.
  - `tracking_panel_bgr`: tracking overlay image for the right column.
  - `hud`: `SideBySidePanelHud`.
- `SideBySidePanelHud`
  - `rgb_seq`
  - `paired_seq`
  - `rgb_ahead_frames`
  - `input_time_s`
  - `pipeline_latency_ms`
  - `display_latency_ms`
  - `startup_hold_s`
  - `filter_preset`
  - `marker_count`
  - optional `tracking_background`
- `render_side_by_side_panel(inputs, *, output_size=None) -> np.ndarray`

Two callers feed this renderer:

- `scripts/harness/diagnostics/demo/render_demo32_headless_capture.py`
  - adds a side-by-side render mode for offline MP4 export.
  - loads frames, RGB images, PCD artifacts, query trajectories, masks, and
    metadata from a headless capture.
- `qqtt/demo/realtime_masked_edgetam_pcd.py`
  - adds a runtime side-by-side panel path for Demo 3.2 fake-live.
  - uses latest capture RGB for the left column and strict paired output for
    the middle/right columns.

The existing helpers in `render_demo32_headless_capture.py` remain reusable:

- PCD projection uses the same projection behavior as `demo_visual_mode=pcd`.
- Tracking overlay uses the same current-frame query point behavior as
  `demo_visual_mode=tracking`.
- `tracking_background_mask=target-union` applies `object_mask |
  controller_mask`; `tracking_background_mask=rgb` preserves full RGB.

## Data Contract Changes

The panel needs explicit timing data that current artifacts only partly expose.
Add these fields where they can be produced naturally.

### FramePacket

Extend `FramePacket` with source-recording metadata:

- `source_timestamp_s: float | None`
- `source_frame_index: int | None`
- `source_step: int | None`

For fake-live recording replay, populate them from `RecordedRgbdFrameRef`.
For live camera inputs, leave them as `None`.

### Headless frames.jsonl

Add per saved paired frame:

- `source_timestamp_s`
- `source_frame_index`
- `source_step`
- `startup_hold_s`
- `pipeline_latency_ms`
- `filter_preset`
- `marker_count`

Keep existing fields:

- `seq`
- `receive_perf_s`
- `process_done_perf_s`
- `timing`
- `filter_telemetry`
- point and mask counts
- artifact paths

`pipeline_latency_ms` is measured from the paired frame's input receive time to
the moment the strict same-seq PCD/tracker pair is ready. For headless capture,
that is the paired output time, not the later offline render time.

`display_latency_ms` is not stored by headless capture. Offline rendering
computes it when writing the panel frame. Runtime computes it at the moment the
panel frame is displayed or written.

### Headless input RGB timeline

The existing headless `rgb/` directory is a paired artifact written with the PCD
row. It is not enough to reconstruct a no-latency latest-RGB left column when
the pipeline is behind. Add an independent input timeline:

- `input_rgb/<seq>.png`
- `input_frames.jsonl`

Each input row contains:

- `seq`
- `input_rgb_path`
- `source_timestamp_s`
- `source_frame_index`
- `source_step`
- `receive_perf_s`

For fake-live headless capture, write this row when the capture packet is
published, before FFS/segmentation/tracking processing. This timeline is the
offline equivalent of the runtime latest capture slot. It allows offline panel
rendering to show `rgb_seq > paired_seq` when the processing pipeline lags.

### Headless metadata.json

Add capture-level panel defaults:

- `panel_supported: true`
- `panel_sync_policy: "left_latest_rgb_right_strict_same_seq"`
- `tracking_background_default: "target-union"`
- `filter_preset`
- `startup_hold_s`
- `input_rgb_timeline: "input_frames.jsonl"`

The renderer must still work with older captures when possible. If older
captures lack source timestamps, it falls back to `seq / replay_fps` and marks
the summary fallback source as `seq_over_replay_fps`.

## Timing Semantics

The fake-live timeline has two concepts of time:

- `input_time_s`: the original recording timestamp for the displayed source
  frame when available.
- `receive_perf_s`: the runtime monotonic time when the source frame enters the
  pipeline.

For the HUD:

- `rgb_seq`: source sequence shown in the left RGB column.
- `paired_seq`: strict same-seq pipeline frame used by PCD and tracking.
- `rgb_ahead_frames = max(0, rgb_seq - paired_seq)`.
- `input_time_s`: source timestamp of `paired_seq`, because that is the frame
  whose processed output is being judged.
- `pipeline_latency_ms`: paired frame input to same-seq pair ready.
- `display_latency_ms`: paired frame input to panel frame displayed/written.
- `startup_hold_s`: time spent holding replay after frame 0 while first
  segmentation/tracker/PCD pair becomes ready.

Runtime can show a latest RGB frame that has not yet been processed. That is
intentional. The HUD must prevent confusion by always showing both sequences.

## Offline Headless MP4 Flow

Add a side-by-side mode to `render_demo32_headless_capture.py`.

Proposed CLI:

```bash
conda run -n demo_2_max --no-capture-output \
  python scripts/harness/diagnostics/demo/render_demo32_headless_capture.py \
  --capture-dir result/single_demo_v3_2_ffs_masked_pcd/headless_smoke \
  --output result/single_demo_v3_2_ffs_masked_pcd/headless_smoke/video_side_by_side.mp4 \
  --fps 30 \
  --panel-mode side-by-side \
  --tracking-background-mask target-union
```

Rendering algorithm:

1. Load capture metadata and all paired rows from `frames.jsonl`.
2. Build an index of source RGB frames from `input_frames.jsonl` and
   `input_rgb/`.
3. For each output row with `paired_seq`:
   - choose left `rgb_seq` as the latest input frame at the corresponding
     output wall-clock time, if capture has enough timing information;
   - otherwise choose `rgb_seq = paired_seq` as compatibility fallback.
4. Render left panel from `input_rgb/rgb_seq.png`.
5. Render middle panel from `paired_seq` filtered PCD projected into camera
   space.
6. Render right panel from `paired_seq` RGB/mask/query trajectory.
7. Compose the shared 1x3 panel with HUD.
8. Write MP4 and a `.panel_summary.json`.

The offline panel should prefer the no-latency left-column behavior when the
capture contains an input RGB timeline with source timing. For older captures,
it must fail softly into same-seq paired RGB and report:

- `left_rgb_policy: "same_seq_fallback"`
- `left_rgb_fallback_reason`

Summary JSON fields:

- `capture_dir`
- `output`
- `frame_count`
- `fps`
- `panel_mode`
- `left_rgb_policy`
- `sync_policy`
- `tracking_background_mask`
- `input_rgb_frame_count`
- `missing_query_frames`
- `missing_rgb_frames`
- `latency_summary_ms`
- `rendered_counts`

## Runtime Realtime Panel Flow

Add runtime panel support for Demo 3.2 fake-live without replacing the existing
Open3D point-cloud view by default.

Proposed wrapper CLI additions:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v3_2/realtime_single_camera_ffs_masked_pcd.py \
  --input-source fake-live \
  --mode demo \
  --demo-visual-mode tracking \
  --render-mode panel \
  --panel-layout side-by-side \
  --tracking-background-mask target-union
```

Optional video output:

```bash
  --panel-video-output result/single_demo_v3_2_ffs_masked_pcd/realtime_panel.mp4
```

Runtime data flow:

1. Capture worker continues publishing fake-live `FramePacket` objects.
2. Existing strict pairer continues producing same-seq PCD/tracker pairs.
3. Runtime panel loop reads:
   - latest capture frame after the last displayed `rgb_seq` for the left panel;
   - latest strict paired render packet after the last displayed `paired_seq`
     for middle/right.
4. The panel is emitted only when a new paired packet exists. The left RGB can
   be newer than that pair.
5. Middle and right columns always use the paired packet's `paired_seq`.
6. The panel loop displays the composed frame using OpenCV and optionally writes
   the same frame to MP4.

This preserves the current strict same-seq guarantee for the processed panels
while letting the left side communicate realtime input progress.

## HUD Design

The HUD should be compact and repeated in all three columns, with a stronger
global strip at the top or bottom of the full panel.

Required visible fields:

- `rgb_seq=<n>`
- `paired_seq=<n>`
- `rgb_ahead=<n>f`
- `input_t=<seconds>s`
- `pipeline=<ms>ms`
- `display=<ms>ms`
- `startup_hold=<seconds>s`
- `filter=<preset>`
- `markers=<count>`

For the middle panel, optionally add:

- `pcd=obj:<count> ctrl:<count>`

For the right panel, optionally add:

- `tracking_bg=target-union|rgb`

Text rendering should use OpenCV only, with a filled dark translucent
background behind text. The renderer should be deterministic enough for tests:
same inputs produce same output shape and non-empty HUD pixels.

## Error Handling

Offline renderer:

- Fail fast if required capture files are missing for the paired PCD/tracking
  columns.
- For tracking mode with `target-union`, fail fast if `mask_path` is present but
  the mask file is missing or malformed.
- Count missing query trajectories rather than silently reusing an older query.
- For older captures without source timestamp metadata, use same-seq left RGB
  fallback and write the fallback into the summary.

Runtime panel:

- Enable only for fake-live Demo 3.2 in the first implementation.
- If no strict pair is ready, keep waiting while left capture can continue.
- If a paired frame is missing tracking markers, render the tracking column
  without markers and show `markers=0`.
- If video writer cannot open, fail fast before starting long runtime work.

## Testing

Add focused tests rather than relying on hardware.

Unit tests for `qqtt/demo/demo32_side_by_side_panel.py`:

- Composes three same-size panels into one 1x3 BGR image.
- Resizes mismatched input panels to the target cell size.
- Draws HUD text and keeps output dimensions stable.
- Computes `rgb_ahead_frames` correctly.

Offline renderer tests in `tests/test_demo32_headless_render_helper.py`:

- Synthetic capture renders a side-by-side MP4 and `.panel_summary.json`.
- Middle and right panels use the same paired seq.
- Left panel can use a later rgb seq when source timing metadata is available.
- Input RGB timeline rows are preferred over paired artifact RGB for the left
  column.
- Older capture fallback uses same-seq RGB and records the fallback.
- `tracking_background_mask=target-union` and `rgb` both work.
- Missing exact query trajectory increments `missing_query_frames`.

Runtime tests in `tests/test_single_demo_tapnextpp_overlay.py` or a focused new
test module:

- Runtime panel assembler combines latest RGB seq with strict paired seq.
- `rgb_seq > paired_seq` produces the expected HUD fields.
- Panel mode is rejected for unsupported input/demo combinations.
- Panel video writer failure is reported before runtime starts.

Contract tests in `tests/test_single_demo_v3_runtime.py`:

- Dry-run contract includes side-by-side panel support for Demo 3.2 fake-live.
- New CLI flags appear in parser help / dry-run contract.

Validation commands:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest -v \
  tests.test_demo32_headless_render_helper \
  tests.test_single_demo_tapnextpp_overlay \
  tests.test_single_demo_v3_runtime

conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke
```

## Documentation Updates

Update:

- `demo_v3_2/README.md`
  - add headless capture to side-by-side render workflow.
  - add runtime `--render-mode panel` workflow.
  - document left latest RGB vs processed same-seq columns.
- `docs/WORKFLOWS.md`
  - add operator command for generating the panel video.
- `docs/ARCHITECTURE.md`
  - document the shared panel renderer boundary.
- `scripts/harness/README.md`
  - list the side-by-side panel render helper mode.

## Compatibility and Migration

The existing headless capture format remains readable. Older captures without
source timestamps can still render side-by-side panels with same-seq left RGB.
The summary must clearly report that fallback so nobody mistakes it for the
latest-RGB realtime behavior.

Existing `demo_visual_mode=pcd` and `demo_visual_mode=tracking` offline renders
must keep their current behavior. The side-by-side mode is additive.

Existing Open3D runtime point-cloud rendering remains available. The new
runtime panel is selected explicitly through `--render-mode panel`.

## Risks

- The runtime panel may add CPU/video-write overhead. Mitigation: reuse existing
  projected/overlay images where possible, cap render size, and make MP4 writing
  optional.
- Offline left-latest reconstruction depends on saved timing. Mitigation: store
  source timestamps going forward and fall back explicitly for old captures.
- HUD text can clutter the image. Mitigation: use one compact global strip and
  short labels.
- Current dirty branch state contains unrelated changes. Mitigation: implement
  and commit this work in a scoped change, avoiding unrelated files except docs
  and tests required by the feature.

## Acceptance Criteria

- A Demo 3.2 fake-live headless capture can be rendered into a 1x3 MP4.
- Runtime fake-live Demo 3.2 can display the same 1x3 panel live.
- Left panel can show latest RGB while middle/right remain strict same-seq.
- HUD displays seqs, input time, both latencies, startup hold, filter preset,
  and marker count.
- The panel summary reports sync policy and missing/fallback counts.
- Focused tests and smoke validation pass.
