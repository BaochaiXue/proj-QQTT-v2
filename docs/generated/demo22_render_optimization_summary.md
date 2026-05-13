# Demo 2.2 Render Fastpath Summary

## Render Path Audit

```text
current render backend: Open3D GUI Open3DScene + open3d.t.geometry.PointCloud
clear_geometries per frame: no
add_geometry per frame: only when a layer is first added or grows past capacity
remove_geometry per frame: only when a layer becomes empty or legacy-current grows past capacity
new PointCloud per frame: no
Vector3dVector per frame: no
Tensor.from_numpy per frame before change: yes
update_geometry: yes
poll_events/update_renderer: GUI app loop + window.post_redraw
render thread blocks compute: no direct blocking, but GUI posts could queue stale callbacks before coalescing
```

## Implemented

```text
latest-only render buffer: yes
coalesced GUI render posts: yes
combined object/controller display geometry: yes
legacy-inplace backend: yes
tensor-o3d-dlpack CLI/backend hook: yes, experimental
async pinned copy CLI/profile hook: yes
default display LOD: off
compute/filter PCD quality change: no
```

## Profile Additions

Render profile records now include:

```text
queue_wait_ms
gpu_to_cpu_copy_ms
combine_ms
cpu_format_ms
open3d_points_update_ms
open3d_colors_update_ms
open3d_update_geometry_ms
open3d_poll_events_ms
open3d_update_renderer_ms
render_total_ms
render_backpressure_count
render_packets_received/displayed/dropped
```

## Headless Microbenchmark

The checked-in benchmark is a headless packet-format benchmark using synthetic same-quality packets. It verifies the profiler and copy/format path without opening WSLg/Open3D.

```text
source: synthetic
same point count/quality: true
Open3D GUI timing: not measured by headless benchmark
live same-quality profile: still required
```

## Next Live Command

Use the same object/controller quality path, changing only the renderer flags:

```bash
python demo_v2_2/realtime_three_view_async_filtered_fused_pcd.py \
  --edgetam-backend hf_batch_vision_seq_session \
  --edgetam-external-path /home/zhangxinjie/EdgeTAM-HF-batched \
  --experimental-edgetam-batch-vision \
  --track-mode controller-object \
  --object-prompt "stuffed animal" \
  --controller-prompt "towel" \
  --dtype bfloat16 \
  --mask-postprocess cuda-inline \
  --compile-mode vision-reduce-overhead \
  --depth-source ffs_local_batch3 \
  --pcd-mode masked \
  --filter-mode async \
  --render-mode pointcloud \
  --render-every-n 1 \
  --render-backend legacy-inplace \
  --render-layer-mode combined \
  --render-async-latest-only \
  --render-copy-mode async-pinned \
  --render-micro-profile \
  --pcd-color-mode rgb \
  --duration-s 90 \
  --warmup-s 30 \
  --profile-json-output docs/generated/demo22_render_backend_legacy_inplace_same_quality_profile.json
```
