# Demo 3: RealSense CoTracker Overlay

Demo 3 is an online-only realtime visualization demo. It follows the Demo 2.2
asynchronous three-camera pattern, requires exactly three RealSense cameras,
uses RealSense RGB-D as the only depth source, uses SAM3.1 first-frame
initialization with HF EdgeTAM online mask propagation, and runs CoTracker3
online as a separate async tracking overlay stage.

CoTracker output is used for visualization only. FuturePhysTwin post-processing
artifacts such as `track_process_data.pkl`, inverse physics inputs, and final
controller point selection are out of scope.

Demo 3 no longer exposes object-only tracking. The only public semantic switch
is `--mode exp|demo`; both modes track the object/controller union. `exp` uses
the current lab controller prompt `towel`; `demo` uses the formal live demo
controller prompt `hand`.

Non-dry-run execution now adapts the shared three-view runtime: it opens the
three RealSense cameras, uses HF EdgeTAM masks, fuses RealSense-depth semantic
PCD, forces the HF EdgeTAM batch vision encoder, and starts CoTracker3 as a
sidecar latest-wins overlay stage. Rendering does not wait for CoTracker; stale
or missing overlays are skipped. The shared runtime tracking backend is disabled
from Demo 3, so the sidecar is the only CoTracker owner.

Dry-run contract check:

```bash
python demo_v3/realtime_three_view_cotracker3_realsense_overlay.py \
  --dry-run \
  --camera-ids 0,1,2
```

Live contract defaults:

- exactly three RealSense cameras
- `depth_source = realsense`
- `mask_source = hf_edgetam`
- `init_mode = sam31_first_frame`
- `mask_propagation = hf_edgetam_online`
- `semantic_mode = exp`
- `tracking_mask_scope = object_controller_union`
- `tracking_query_mode = phystwin_dense`
- `tracking_query_count_requested = auto`
- `tracking_query_count_rule = min(union_mask_pixels, 5000)`
- `tracking_sampling = torch_randperm_seed_plus_camera_idx`
- `cotracker_seed = 42`
- `edgetam_batch_vision_encoder = true`
- `cotracker_backend = cotracker3_online`
- `overlay_max_points_per_camera = 30`
- `input_source = live_realsense`
- `offline_mode_available = false`
- `render_waited_for_cotracker = false`
- `uses_ffs = false`

Recommended live validation order:

1. `--preset demo3-realsense-mask-only`
2. injected/fake CoTracker overlay in tests
3. real CoTracker with default `--cotracker-query-count auto`
4. optional readability tuning through `--overlay-max-points-per-camera`

Rendered FPS profiling must use `--render-mode pointcloud`; no-render runs are
upstream isolation only. For finite-duration rendered profiles, the shared
three-view runtime writes its profile before Open3D teardown. If the GUI still
hangs or crashes during teardown on the local workstation, run through
`scripts/harness/run_wslg_open3d.sh` or set `QQTT_WSLG_OPEN3D_FAST_EXIT=1`.
