# Demo 3 RealSense CoTracker Runtime Contract

Demo 3 is a realtime visualization demo built on the Demo 2.2 asynchronous
three-camera runtime pattern. It requires exactly three RealSense cameras, uses
RealSense RGB-D as the only depth source, uses HF EdgeTAM masks for semantic
object/controller masks through the shared batch vision encoder path, and runs
CoTracker3 online as a separate async tracking overlay stage.

CoTracker output is used for visualization only. FuturePhysTwin post-processing
artifacts such as `track_process_data.pkl`, inverse-physics data generation,
and final controller point selection are out of scope for Demo 3.

Demo 3 是实时可视化 demo，不是 FuturePhysTwin 数据处理 pipeline。它必须使用三台
RealSense，相机分组和渲染架构学习 Demo 2.2；depth 只用 RealSense，不使用 FFS；mask
来自 HF EdgeTAM，并强制使用 batch vision encoder；CoTracker3 online 作为独立异步
tracking stage 生成可视化 tracking points。CoTracker 慢或未发布时，主 PCD rendering
继续使用最新 fused PCD，不等待 tracking overlay。

## Runtime Shape

```text
3x RealSense capture workers
    -> CaptureGroupBuilder
        -> RealSense RGB-D + HF EdgeTAM masks
            -> FusionWorker
                -> latest fused PCD render slot
            -> CoTracker3OverlayWorker
                -> latest tracking overlay slot
Renderer
    -> render latest fused PCD every frame
    -> overlay latest CoTracker tracks if available
    -> never wait for CoTracker
```

The hot path forbids:

- Fast-FoundationStereo depth
- FFS TensorRT
- FFS remote services
- FFS IR/color alignment workers
- FFS fallback depth
- FuturePhysTwin `track_process_data.pkl`
- inverse physics or final controller point selection

Non-dry-run execution delegates camera capture, timestamp grouping, HF EdgeTAM
masking, RealSense-depth fusion, and Open3D rendering to the shared three-view
runtime used by Demo 2.x. Demo 3 forces that runtime to use RealSense depth and
starts its own sidecar CoTracker3 overlay stage. Demo 2.2 behavior stays
unchanged because the Demo 3 hooks live in the Demo 3 adapter.

## CLI Contract

Dry-run:

```bash
python demo_v3/realtime_three_view_cotracker3_realsense_overlay.py \
  --dry-run \
  --camera-ids 0,1,2
```

Required dry-run fields:

```text
demo = demo3
requires_three_realsense = true
num_cameras = 3
depth_source = realsense
uses_ffs = false
mask_source = hf_edgetam
edgetam_batch_vision_encoder = true
cotracker_backend = cotracker3_online
cotracker_async = true
render_latest_wins = true
render_waited_for_cotracker = false
```

Fail-fast examples:

```bash
python demo_v3/realtime_three_view_cotracker3_realsense_overlay.py --dry-run --camera-ids 0,1
python demo_v3/realtime_three_view_cotracker3_realsense_overlay.py --dry-run --depth-source ffs
```

The first command fails because Demo 3 requires exactly three RealSense cameras.
The second fails because Demo 3 does not support FFS.

## Defaults

- `--preset demo3-realsense-cotracker-highfps`
- `--camera-ids 0,1,2`
- `--depth-source realsense`
- `--mask-source hf-edgetam`
- HF EdgeTAM batch vision encoder enabled
- `--track-mode object-only`
- `--object-prompt "stuffed animal"`
- `--controller-prompt "towel"`
- `--cotracker-backend cotracker3_online`
- `--cotracker-query-count 128`
- `--overlay-max-points-per-camera 30`
- `--overlay-trail-len 16`
- `--overlay-stale-timeout-ms 500`

## CoTracker Overlay Contract

CoTracker3 online is a separate stage. It receives grouped RGB frames and
EdgeTAM masks, samples query points from the semantic mask, converts queries to
CoTracker's `[t, x, y]` input convention through the existing backend, and
stores output tracks as `y,x` so RealSense depth and masks can be indexed
directly.

The live overlay worker uses a latest-only slot. The renderer reads the latest
overlay packet if one exists and proceeds without it when CoTracker is still
warming up, slow, stale, or disabled.

Default live visualization caps the overlay at 30 visible points per camera.
Dense 5000-point CoTracker artifacts remain a benchmark/export diagnostic path,
not the realtime rendering default.

Live overlay query sampling is deterministic random visualization sampling from
the current semantic mask. It is not byte-identical to FuturePhysTwin dense
export sampling. FuturePhysTwin-style `torch.randperm` remains part of the
offline benchmark/export compatibility path.

If an early EdgeTAM mask is empty, Demo 3 does not permanently cache empty
query points. The CoTracker stream initializes on the first non-empty semantic
mask for each camera.

## Profile Fields

Demo 3 profile summaries keep rendering and tracking metrics separate:

```text
rendered_fps
render_loop_fps
capture_group_fps
edgetam_mask_fps
fusion_fps
cotracker_publish_fps
cotracker_model_ms_median
cotracker_model_ms_p95
cotracker_e2e_ms_median
cotracker_e2e_ms_p95
overlay_ms_median
overlay_ms_p95
pcd_fusion_ms_median
pcd_render_ms_median
render_waited_for_cotracker = false
uses_ffs = false
depth_source = realsense
mask_source = hf_edgetam
edgetam_batch_vision_encoder = true
num_realsense_cameras = 3
calibrate_pkl_loaded = true
cotracker_backend = cotracker3_online
cotracker_window_len = 16
cotracker_publish_step = 8
```

The acceptance-critical fields are:

```text
render_waited_for_cotracker = false
uses_ffs = false
depth_source = realsense
num_realsense_cameras = 3
```

## Live Validation

Before opening the live runtime, Demo 3 validates:

- exactly three requested camera ids
- RealSense depth only
- exactly three connected RealSense cameras when `--serials` is omitted
- exactly three requested serials, all connected, when `--serials` is used
- `calibrate.pkl` exists
- calibration metadata, when present, covers all active serials

The live run still depends on hardware for final proof: each camera must produce
color and depth frames, and the shared runtime must load the calibration
transforms and camera intrinsics successfully.
