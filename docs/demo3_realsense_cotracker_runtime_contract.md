# Demo 3 RealSense CoTracker Runtime Contract

Demo 3 is an online-only realtime visualization demo built on the Demo 2.2
asynchronous three-camera runtime pattern. It requires exactly three RealSense
cameras, uses RealSense RGB-D as the only depth source, uses SAM3.1 first-frame
initialization plus HF EdgeTAM online propagation for semantic
object/controller masks through the shared batch vision encoder path, and runs
CoTracker3 online as a separate async tracking overlay stage.

CoTracker output is used for visualization only. FuturePhysTwin post-processing
artifacts such as `track_process_data.pkl`, inverse-physics data generation,
and final controller point selection are out of scope for Demo 3.

Demo 3 tracks the object/controller union. It does not expose object-only or
controller-only live tracking. The only public semantic switch is
`--mode exp|demo`: `exp` uses controller `towel` for the current lab setup and
`demo` uses controller `hand` for the formal live demo.

The rendered overlay is label-filtered after dense tracking. CoTracker samples
and tracks the raw object/controller union, then each query is labeled by its
first-frame object/controller mask membership. By default
`overlay_display_scope = controller`, so the visible overlay keeps only
controller-labeled tracks before applying the 30-points-per-camera cap.

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
starts its own sidecar CoTracker3 overlay stage. The shared runtime tracking
backend is forced to `none` from Demo 3 so CoTracker3 has a single owner: the
Demo 3 sidecar. Demo 2.2 behavior stays unchanged because the Demo 3 hooks live
in the Demo 3 adapter.

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
edgetam_live_session_keep_frames = 64
edgetam_live_session_pruning = true
input_source = live_realsense
offline_mode_available = false
offline_tracking_available = false
init_mode = sam31_first_frame
mask_propagation = hf_edgetam_online
semantic_mode = exp
tracking_mask_scope = object_controller_union
tracking_query_mode = phystwin_dense
tracking_query_count_requested = auto
tracking_query_count_rule = min(union_mask_pixels, 5000)
tracking_sampling = torch_randperm_seed_plus_camera_idx
cotracker_seed = 42
phystwin_dense_compatible = true
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
- `--edgetam-live-session-keep-frames 64`
- shared runtime label override: `Demo 3`
- `--mode exp`
- `--object-prompt "stuffed animal"`
- controller prompt resolved by mode: `towel` for `exp`, `hand` for `demo`
- `--cotracker-backend cotracker3_online`
- `--cotracker-query-mode phystwin_dense`
- `--cotracker-query-count auto`
- `--cotracker-seed 42`
- `--overlay-max-points-per-camera 30`
- `--overlay-display-scope controller`
- `--overlay-trail-len 16`
- `--overlay-stale-timeout-ms 500`
- render object PCD filter defaults to FuturePhysTwin-style volume sampling:
  `--object-point-control phystwin-volume`,
  `--object-volume-voxel-m 0.005`, and
  `--object-volume-points-per-voxel 1`

Demo 3 also exposes the shared three-view runtime diagnostics used by Demo 2.3:

```text
--object-point-control fixed-cap|phystwin-volume
--object-volume-voxel-m
--object-volume-origin world|frame-min|first-stable-frame-min
--object-volume-adaptive / --no-object-volume-adaptive
--object-volume-min-voxel-m
--object-volume-max-voxel-m
--object-volume-target-ms
--object-volume-emergency-max-points
--object-volume-points-per-voxel
--debug-color-by-camera
--debug-save-per-camera-pcd
--debug-save-mask-overlays
--debug-identity-c2w
--debug-invert-c2w
--debug-only-camera-idx
--debug-fusion-max-saved-groups
--gpu-sampling
--gpu-sampling-device-indexes
--point-size
--render-every-n
--render-backend
--render-layer-mode
--render-copy-mode
```

## CoTracker Overlay Contract

CoTracker3 online is a separate stage. It receives grouped RGB frames and
EdgeTAM object/controller union masks, samples query points from that union,
converts queries to CoTracker's `[t, x, y]` input convention through the
existing backend, and stores output tracks as `y,x` so RealSense depth and masks
can be indexed directly.

The live overlay worker uses a latest-only slot. The renderer reads the latest
overlay packet if one exists and proceeds without it when CoTracker is still
warming up, slow, stale, or disabled.

Overlay freshness is measured from the overlay publish time, not the source
capture timestamp, so a slow CoTracker update is not discarded immediately
after it finishes. The source timestamp remains diagnostic metadata.

Default live tracking is FuturePhysTwin-compatible dense sampling: up to 5000
raw query points per camera from `object_mask | controller_mask`, sampled with
torch `randperm(seed + camera_idx)`. Default seed is 42.

Default live visualization first filters displayed tracks to first-frame
controller-labeled queries, then caps the displayed overlay at 30 visible
points per camera. This display cap and label filter do not reduce the raw
CoTracker query count. `--overlay-display-scope object|union` is available for
debug views without changing raw tracking.

The rendered object point cloud is filtered independently from CoTracker
queries. By default Demo 3 keeps up to one representative point per occupied
5mm world voxel, matching the FuturePhysTwin `object_points` volume-sampling
semantics. `--object-volume-points-per-voxel N` can retain more local surface
density per occupied voxel without switching back to arbitrary fixed point
counts. Fixed point caps remain available only through
`--object-point-control fixed-cap` for ablation/debug.

If an early EdgeTAM result contains only object or only controller, Demo 3 does
not initialize CoTracker for that camera yet. The stream initializes only after
both object and controller masks are non-empty, so the first query set cannot
lock into an object-only subset.

## Profile Fields

Demo 3 profile summaries keep rendering and tracking metrics separate:

```text
rendered_fps
render_loop_fps
capture_group_fps
edgetam_mask_fps
fusion_fps
cotracker_publish_fps
cotracker_update_mode
cotracker_update_mode_effective
cotracker_batch_size
cotracker_batch_update_count
cotracker_serial_camera_update_count
cotracker_serial_fallback_count
cotracker_model_ms_median
cotracker_model_ms_p95
cotracker_e2e_ms_median
cotracker_e2e_ms_p95
overlay_ms_median
overlay_ms_p95
object_volume_ms
object_volume_occupied_voxels
object_volume_output_points
object_volume_points_per_voxel
pcd_fusion_ms_median
pcd_render_ms_median
render_waited_for_cotracker = false
uses_ffs = false
depth_source = realsense
mask_source = hf_edgetam
edgetam_batch_vision_encoder = true
edgetam_live_session_keep_frames = 64
edgetam_live_session_pruning = true
num_realsense_cameras = 3
calibrate_pkl_loaded = true
cotracker_backend = cotracker3_online
cotracker_window_len = 16
cotracker_publish_step = 8
tracking_mask_scope = object_controller_union
tracking_query_mode = phystwin_dense
tracking_query_count_requested = auto
tracking_query_count_actual_by_camera
tracking_union_pixels_by_camera
tracking_object_pixels_by_camera
tracking_controller_pixels_by_camera
tracking_sample_object_hits_by_camera
tracking_sample_controller_hits_by_camera
tracking_sample_overlap_hits_by_camera
tracking_sample_background_hits_by_camera
overlay_display_scope
overlay_display_count_by_camera
overlay_display_object_count_by_camera
overlay_display_controller_count_by_camera
```

The acceptance-critical fields are:

```text
render_waited_for_cotracker = false
uses_ffs = false
depth_source = realsense
num_realsense_cameras = 3
tracking_mask_scope = object_controller_union
tracking_query_mode = phystwin_dense
overlay_display_scope = controller
```

Rendered FPS claims must come from `--render-mode pointcloud` runs, not
`--render-mode none` isolation runs. Demo 3 delegates Open3D rendering to the
shared three-view runtime, which stops workers and writes the shared runtime
profile before requesting Open3D teardown on finite-duration runs. On
workstations where Open3D/Filament teardown can still hang or crash, launch the
rendered profile through `scripts/harness/run_wslg_open3d.sh` or set
`QQTT_WSLG_OPEN3D_FAST_EXIT=1`. If fast exit is used, the shared runtime profile
is the durable source of truth; wrapper-level summary output may be bypassed by
the direct process exit.

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
