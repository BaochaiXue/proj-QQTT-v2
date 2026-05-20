# Demo 3.1 Dual-4090 Runtime Contract

Demo 3.1 is a dual-4090 realtime visualization runtime cloned from Demo 3.0.
GPU0 owns the shared three-RealSense capture, SAM3.1/HF EdgeTAM mask,
RealSense-depth fusion, and render path. GPU1 owns a point-tracker backend in a
separate child process. The default validated backend is `cotracker3_online`;
the contract also accepts `trackon2` and `litetracker` so the same Demo 3.1
pipeline can profile alternative online trackers.

No CUDA tensors are transferred between processes. The tracker receives CPU RGB
plus live object/controller union-mask latest-wins packets and returns small CPU
2D track/visibility packets. It does not receive offline video, saved masks,
depth, intrinsics, or camera-to-world transforms. The main process lifts tracks
to world using group-aligned cached RealSense depth, intrinsics, and
camera-to-world transforms through the same semantic projection-grid
backprojection convention used by the fused PCD path.

Demo 3.1 inherits Demo 3.0's online-only object/controller union tracking
semantics. The only public semantic switch is `--mode exp|demo`; both modes
track `object_mask | controller_mask`. `exp` uses controller `towel`, while
`demo` uses controller prompt `human hand` for SAM3.1 while the controller
semantic remains a hand. CoTracker query sampling uses
`phystwin_dense`, but Demo 3.1 defaults to `--cotracker-query-count 4096`
per camera for the batch=3 RTX 4090 path. A full `auto` / 5000-per-view
three-camera batch exceeds the 24GB 4090 memory budget in live profiling. The
controller/towel mask is first capped to
`controller_pcd_max_points_per_camera = 4999` per camera, before both tracking
query selection and fused PCD construction. Query points are then sampled from
the capped object/controller union with the requested per-view budget using
torch `randperm(seed + camera_idx)` with seed 42. Use
`--cotracker-query-count auto` only when exact 5000-per-view dense sampling is
needed and the tracker backend/memory budget can support it. The optional
`--controller-mask-erode-px` parameter shrinks the controller mask before the
tracking union and anchor/trackable-mask path; its implicit default is `1` in
`--mode demo` (human-hand controller prompt) and `0` otherwise. Controller body points are
render-voxel downsampled before Open3D display with
`--controller-render-voxel-m`; this render-only reduction does not touch
LiteTracker input or the red tracking/control markers. Overlay display selection is
separate: raw CoTracker tracks still come from the capped union, but each query
is labeled by first-frame object/controller mask membership and the default
rendered overlay shows controller-labeled tracks only. The display cap is
disabled by default for Demo 3.1: `overlay_max_points_per_camera = 0`
means all visible controller-labeled tracks remain eligible for control-point
selection. The default rendered tracking mark follows a PhysTwin-style anchored
control-handle rule: select up to 16 visible tracking controls per camera,
snap each control to the nearest same-camera, same-semantic fused surface point
within 4 pixels, then draw a red 3D sphere marker with radius 6mm. Direct
2D-track/depth/intrinsics/c2w lifting is disabled by default and is available
under `--tracker-visualization-mode legacy-3d-lift` for debugging. Demo 3.2
uses `--tracker-visualization-mode all-tracks-3d-lift` by default: every
visible LiteTracker point with valid depth becomes a red 3D control marker,
without surface-snap matching, semantic scope-mask rejection, or semantic bbox
rejection.
The Open3D warmup HUD is pipeline-aware: Demo 3.2 reports the FFS/EdgeTAM
path plus LiteTracker query-init and 3D anchors instead of the older fixed
FFS/EdgeTAM-only message.
`--overlay-render-raw-track-points` only affects that legacy debug mode. For
alignment debugging, `--overlay-debug-color-by-camera` colors snapped
overlay/control points by source camera while keeping
`overlay_display_scope=controller`.
The surface-anchor layer follows the display scope: controller overlays snap
only to current controller surface anchors, object overlays to object anchors,
and union overlays to object/controller union anchors. If no same-group surface
anchor is available, or the nearest anchor is outside the pixel snap radius,
the marker is rejected rather than rendered as a detached 3D point.

Camera/mask/PCD work remains asynchronous, but rendered result publication is
gated by CoTracker by default. The main process stores pending PCD packets by
`group_id` in a bounded latest window. When a fresh CoTracker result arrives,
Demo 3.1 first renders the exact matching PCD packet. If that exact packet was
already evicted, it falls back to the nearest pending PCD group by absolute
`group_id` delta and marks the frame as `nearest` in the profile. If CoTracker
is not ready, no pending PCD is available, or the selected PCD has no matching
surface anchors, no new rendered result is published and Open3D keeps the
previous valid frame.
Rendered FPS therefore measures track-ready results, not semantic-only PCD
throughput. Because the tracker result is already the render clock, Demo 3.1
does not expose a render stride option and renders every tracker-ready group.
Use `--no-wait-for-tracking-overlay` only for debugging the semantic PCD before
tracking is available.

The rendered object PCD uses FuturePhysTwin-style world-volume sampling by
default: one representative point per occupied 5mm voxel. This is independent
from CoTracker query sampling and the overlay display cap. Use
`--object-volume-points-per-voxel N` to keep more representatives inside each
occupied voxel when local surface density is more important than render cost.

## Contract Fields

```text
demo = demo3.1
input_source = live_realsense
offline_mode_available = false
offline_tracking_available = false
dual_gpu_enabled = true
required_cuda_devices = 2
mask_gpu_physical = 0
cotracker_gpu_physical = 1
main_cuda_visible_devices = "0"
cotracker_cuda_visible_devices = "1"
uses_ffs = false
depth_source = realsense
mask_source = hf_edgetam
edgetam_batch_vision_encoder = true
edgetam_live_session_keep_frames = 64
edgetam_live_session_pruning = true
init_mode = sam31_first_frame
mask_propagation = hf_edgetam_online
semantic_mode = exp
tracking_mask_scope = object_controller_union
tracking_query_mode = phystwin_dense
tracking_query_count_requested = 4096
tracking_query_count_rule = min(capped_object_controller_union_pixels, 5000)
tracking_sampling = controller_pcd_cap_then_torch_randperm_seed_plus_camera_idx
controller_mask_erode_px = 0
controller_mask_erode_stage = before_tracking_union_and_trackable_filter
controller_mask_erode_applies_to = tracking_input_and_anchor_masks
render_controller_filter = {'render_voxel_m': 0.003, 'render_voxel_downsample': true, 'render_only': true, 'affects_tracking_markers': false}
controller_pcd_max_points_per_camera = 4999
controller_pcd_cap_stage = before_tracking_query_and_fusion
controller_pcd_cap_sampling = stable_coordinate_hash_seed_plus_camera_idx
cotracker_seed = 42
phystwin_dense_compatible = false
wait_for_tracking_overlay = true
tracking_overlay_required_before_first_render = true
tracking_overlay_required_for_render = true
render_requires_new_cotracker_result = true
render_reuses_cached_cotracker_result = false
tracking_overlay_color_rgb = [255, 0, 0]
tracking_overlay_color_mode = solid
tracker_visualization_mode = 3d-surface-markers
tracker_3d_marker_mode = surface_snap
tracker_3d_marker_shape = sphere
tracker_legacy_lift_used = false
tracker_3d_snap_radius_px = 4.0
tracker_3d_marker_radius_m = 0.006
tracker_control_points_per_camera = 16
tracker_control_point_selection = visible-spread
tracking_overlay_lift_method = surface_snap
overlay_lift_mask_scope = controller
overlay_max_points_per_camera = 0
overlay_display_scope = controller
overlay_display_classification = first_frame_mask_membership
overlay_bbox_filter_enabled = true
overlay_bbox_filter_scope = controller
overlay_bbox_filter_margin_m = 0.15
tracking_control_point_markers = true
tracking_control_point_count_requested = 48
tracking_control_points_per_camera = 16
tracking_control_point_radius_m = 0.006
tracking_control_point_sampling = visible-spread_surface_snap
overlay_render_raw_track_points = false
tracking_pending_render_packet_max_groups = 128
tracking_render_packet_match_policy = exact-then-nearest-pending-pcd-by-group-id
cotracker_backend = cotracker3_online
tracker_backend = cotracker3_online
tracker_backend_family = cotracker
tracking_backend_execution_mode = batch-views
tracking_backend_batch_dimension = camera
tracking_backend_batch_size = 3
tracking_backend_batch_supported = true
tracking_backend_batch_auto_selected = false
tracker_batch_query_count_policy = fixed
cotracker_owner = process
cotracker_process_mode = subprocess
cotracker_update_mode = batch
cotracker_batch_fallback_enabled = false
cross_gpu_cuda_tensor_transfer = false
ipc_payload = cpu_numpy_latest_wins
tracking_input_contains_depth = false
tracking_input_contains_intrinsics = false
tracking_input_contains_c2w = false
world_lift_owner = main_process
fusion_mask_policy = latest-reuse
render_waited_for_cotracker = true
render_waited_for_fresh_cotracker_result = true
render_driver = cotracker_child_output
render_trigger = new_cotracker_result
render_waited_for_mask = false
debug_fusion.color_by_camera = false
gpu_sampling.enabled = false
gpu_sampling.device_indexes = [0, 1]
```

`strict` mask policy is available for comparison. In strict mode,
`render_waited_for_mask = true` because fusion requires a matching mask group.

## Profile Fields

Demo 3.1 reports render, fusion, mask reuse, tracking, and GPU ownership
separately so rendered FPS is not confused with true fresh-mask FPS:

Rendered groups are gated on a newly published CoTracker child-process result.
The renderer must not reuse an old CoTracker result as a new rendered tracking
frame. The Open3D HUD shows the displayed depth/FFS cycle time, HF EdgeTAM
inference timing, and the CoTracker batch inference/e2e timing for the rendered
tracking frame.

```text
render_loop_fps
rendered_fps
new_fused_pcd_fps
capture_group_fps
fresh_mask_fps
edgetam_live_session_keep_frames
edgetam_live_session_pruning
mask_reuse_ratio
mask_age_ms_median
mask_age_ms_p95
cotracker_input_fps
cotracker_input_drop_count
cotracker_input_queue_replace_count
cotracker_publish_fps
cotracker_update_mode
cotracker_update_mode_effective
tracker_backend
tracker_backend_family
tracking_backend_execution_mode
tracking_backend_batch_dimension
tracking_backend_batch_size
tracking_backend_batch_enabled
tracking_backend_batch_supported
tracking_backend_batch_support_status
tracking_backend_batch_auto_selected
tracker_batch_query_count_policy
cotracker_batch_size
cotracker_batch_update_count
cotracker_serial_camera_update_count
cotracker_serial_fallback_count
cotracker_model_ms_median
cotracker_model_ms_p95
cotracker_e2e_ms_median
cotracker_e2e_ms_p95
render_requires_new_cotracker_result
render_reuses_cached_cotracker_result
render_waited_for_fresh_cotracker_result
render_driver
render_trigger
rendered_on_new_cotracker_result
overlay_age_ms_median
overlay_age_ms_p95
overlay_render_group_delta_median
overlay_render_group_delta_p95
tracking_overlay_warmup_skipped_render_count
tracking_overlay_first_render_group_id
tracking_overlay_render_blocked_count
tracking_pending_render_packets
tracking_pending_render_packet_max_groups
tracking_pending_render_packet_drop_count
tracking_render_packet_match_policy
tracking_result_exact_render_packet_count
tracking_result_nearest_render_packet_count
tracking_result_without_render_packet_count
tracking_result_without_lift_input_count
tracking_input_mask_reuse_ratio
tracking_input_mask_age_ms_median
tracking_input_mask_age_ms_p95
mask_group_delta_median
mask_group_delta_p95
object_volume_ms
object_volume_occupied_voxels
object_volume_output_points
object_volume_points_per_voxel
gpu0_util_median
gpu0_util_p95
gpu0_mem_used_gb
gpu1_util_median
gpu1_util_p95
gpu1_mem_used_gb
main_process_pid
cotracker_process_pid
tracking_query_count_actual_by_camera
tracking_union_pixels_by_camera
tracking_object_pixels_by_camera
tracking_controller_pixels_by_camera
tracking_sample_object_hits_by_camera
tracking_sample_controller_hits_by_camera
tracking_sample_overlap_hits_by_camera
tracking_sample_background_hits_by_camera
overlay_display_count_by_camera
overlay_display_object_count_by_camera
overlay_display_controller_count_by_camera
overlay_input_points_by_camera
overlay_points_by_camera
overlay_rejected_by_scope_mask_by_camera
overlay_bbox_filter_enabled
overlay_bbox_filter_scope
overlay_bbox_filter_margin_m
overlay_bbox_input_points_by_camera
overlay_bbox_kept_points_by_camera
overlay_bbox_rejected_by_camera
overlay_world_centroid_by_camera_before_bbox
overlay_world_centroid_by_camera
overlay_track_points
tracker_visualization_mode
tracker_3d_marker_mode
tracker_3d_marker_shape
tracker_legacy_lift_used
tracker_surface_anchor_cache_hit
tracker_surface_anchor_group_id
tracker_marker_accepted_by_camera
tracker_marker_rejected_by_camera
tracker_marker_pixel_error_median_by_camera
tracker_marker_pixel_error_p95_by_camera
tracker_marker_layer_by_camera
tracker_marker_points_rendered
tracking_control_point_markers
tracking_control_point_count_requested
tracking_control_points_per_camera
tracking_control_point_count
tracking_control_points_by_camera
tracking_control_point_radius_m
tracking_control_point_sampling
tracking_control_marker_points
tracking_control_point_centroid
overlay_render_raw_track_points
```

Rendered FPS claims must come from `--render-mode pointcloud` runs. A
`--render-mode none` run can isolate mask/fusion/tracking throughput, but it is
not a rendered FPS result. Demo 3.1 delegates Open3D rendering to the shared
three-view runtime, which stops workers and writes the shared runtime profile
before requesting Open3D teardown on finite-duration runs. On workstations where
Open3D/Filament teardown can still hang or crash, launch rendered profiles
through `scripts/harness/run_wslg_open3d.sh` or set
`QQTT_WSLG_OPEN3D_FAST_EXIT=1`. If fast exit is used, the shared runtime profile
is the durable source of truth; wrapper-level summary output may be bypassed by
the direct process exit.

Demo 3.1 exposes the Demo 2.3 fusion and renderer diagnostics through its public
CLI and forwards them to the shared three-view runtime:

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
--controller-render-voxel-m
--debug-color-by-camera
--debug-save-per-camera-pcd
--debug-save-mask-overlays
--debug-identity-c2w
--debug-invert-c2w
--debug-only-camera-idx
--debug-fusion-max-saved-groups
--gpu-sampling
--gpu-sampling-device-indexes
--overlay-display-scope controller|object|union
--overlay-debug-color-by-camera
--overlay-reject-outside-semantic-bbox / --no-overlay-reject-outside-semantic-bbox
--overlay-max-distance-from-controller-m
--tracker-visualization-mode none|3d-surface-markers|2d-debug|legacy-3d-lift|all-tracks-3d-lift
--tracker-3d-snap-radius-px
--tracker-3d-marker-radius-m
--tracker-control-points-per-camera
--tracker-control-point-selection visible-spread|top-visible|mask-stratified
--overlay-control-point-markers / --no-overlay-control-point-markers
--overlay-control-point-count
--overlay-control-point-radius-m
--overlay-render-raw-track-points / --no-overlay-render-raw-track-points
--wait-for-tracking-overlay / --no-wait-for-tracking-overlay
--point-size
--render-backend
--render-layer-mode
--render-copy-mode
```

When `--gpu-sampling` is enabled and `--gpu-sampling-device-indexes` is omitted,
Demo 3.1 samples the physical mask and CoTracker GPU indexes, which are `0,1`
for the default dual-4090 split. The wrapper summary copies the shared runtime
per-device utilization and memory summaries into `gpu0_*` and `gpu1_*` fields.

## Boundaries

- Demo 3.0 remains the stable single-process lineage.
- Demo 3.1 does not build or consume FFS depth.
- Demo 3.1 does not rely on cross-GPU tensor operations or CUDA tensor IPC.
- CoTracker process does not receive depth, intrinsics, or camera-to-world data.
- Demo 3.1 does not expose object-only or controller-only raw live tracking
  modes; `--overlay-display-scope controller|object|union` changes only which
  already-tracked first-frame labels are rendered.
- Demo 3.1 does not expose offline video, cached tracking input, saved-mask
  initialization, or case replay through the live entrypoint.
- CoTracker output is visualization-only tracking overlay data.
