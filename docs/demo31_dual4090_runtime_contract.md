# Demo 3.1 Dual-4090 Runtime Contract

Demo 3.1 is a dual-4090 realtime visualization runtime cloned from Demo 3.0.
GPU0 owns the shared three-RealSense capture, SAM3.1/HF EdgeTAM mask,
RealSense-depth fusion, and render path. GPU1 owns CoTracker3 online in a
separate child process.

No CUDA tensors are transferred between processes. CoTracker receives CPU
RGB plus live object/controller union-mask latest-wins packets and returns
small CPU 2D track/visibility packets. It does not receive offline video,
saved masks, depth, intrinsics, or camera-to-world transforms. The main process
lifts tracks to world using group-aligned cached RealSense depth, intrinsics,
and camera-to-world transforms.

Demo 3.1 inherits Demo 3.0's online-only FuturePhysTwin-compatible tracking
semantics. The only public semantic switch is `--mode exp|demo`; both modes
track `object_mask | controller_mask`. `exp` uses controller `towel`, while
`demo` uses controller `hand`. CoTracker query sampling defaults to
`phystwin_dense`: `min(union_mask_pixels, 5000)` query points per camera,
sampled by torch `randperm(seed + camera_idx)` with seed 42. Overlay display
selection is separate: raw CoTracker tracks still come from the union, but each
query is labeled by first-frame object/controller mask membership and the
default rendered overlay shows controller-labeled tracks only. The display cap
is disabled by default for Demo 3.1: `overlay_max_points_per_camera = 0`
means render all visible controller-labeled tracks selected from the CoTracker
union queries. The visible CoTracker tracking overlay color is high-contrast
red, separate from the semantic PCD object/controller colors.

Camera/mask/PCD work remains asynchronous, but rendered result publication is
gated by CoTracker by default. The main process stores pending PCD packets by
`group_id`; when a fresh CoTracker result arrives, Demo 3.1 renders only the
matching PCD packet after the result can be lifted into red tracking points. If
CoTracker is not ready, or the matching PCD/lift inputs are no longer available,
no new rendered result is published and Open3D keeps the previous valid frame.
Rendered FPS therefore measures track-ready results, not semantic-only PCD
throughput. Use `--no-wait-for-tracking-overlay` only for debugging the semantic
PCD before tracking is available.

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
tracking_query_count_requested = auto
tracking_query_count_rule = min(union_mask_pixels, 5000)
tracking_sampling = torch_randperm_seed_plus_camera_idx
cotracker_seed = 42
phystwin_dense_compatible = true
wait_for_tracking_overlay = true
tracking_overlay_required_before_first_render = true
tracking_overlay_required_for_render = true
tracking_overlay_color_rgb = [255, 0, 0]
overlay_max_points_per_camera = 0
overlay_display_scope = controller
overlay_display_classification = first_frame_mask_membership
cotracker_backend = cotracker3_online
cotracker_owner = process
cotracker_process_mode = subprocess
cross_gpu_cuda_tensor_transfer = false
ipc_payload = cpu_numpy_latest_wins
tracking_input_contains_depth = false
tracking_input_contains_intrinsics = false
tracking_input_contains_c2w = false
world_lift_owner = main_process
fusion_mask_policy = latest-reuse
render_waited_for_cotracker = true
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
cotracker_batch_size
cotracker_batch_update_count
cotracker_serial_camera_update_count
cotracker_serial_fallback_count
cotracker_model_ms_median
cotracker_model_ms_p95
cotracker_e2e_ms_median
cotracker_e2e_ms_p95
overlay_age_ms_median
overlay_age_ms_p95
overlay_render_group_delta_median
overlay_render_group_delta_p95
tracking_overlay_warmup_skipped_render_count
tracking_overlay_first_render_group_id
tracking_overlay_render_blocked_count
tracking_pending_render_packets
tracking_pending_render_packet_drop_count
tracking_result_without_render_packet_count
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
--wait-for-tracking-overlay / --no-wait-for-tracking-overlay
--point-size
--render-every-n
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
