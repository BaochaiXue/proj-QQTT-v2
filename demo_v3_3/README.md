# Demo 3.3 Single-View Shape-Prior Warmup

Demo 3.3 is the experimental shape-prior warmup variant of Demo 3.2. It keeps
the Demo 3.2 FFS + EdgeTAM + TAPNext++/LiteTracker live path unchanged, then
adds one warmup-only canonical reference layer for the current experiment
object: `stuffed animal` with controller `towel`.

During the first valid strict-source tracking input, Demo 3.3 snapshots the
complete RGB-D/mask/calibration bundle. By default the heavy
FuturePhysTwin/SAM3D route starts after live teardown, once the FFS/EdgeTAM and
tracker workers have released GPU memory. Tracker input and first-render warmup
therefore do not wait on SAM3D or compete with it during the latency-sensitive
startup window, while the shape-prior artifact is still generated and
validated from the captured live snapshot. The frame0-only FuturePhysTwin-style
case is written under:

```text
<output-root>/demo33_shape_prior_warmup/<run_id>/case/
```

It then runs the exact single-view SAM3D preprocessing route:

```text
image_upscale.py
segment_util_image.py
data_process_sam3d/shape_prior.py
data_process/align.py
data_process_sam3d/data_process_sample.py --shape_prior
```

The loaded `final_data.pkl` is used only for rendering:

```text
structure_points = object_points[0] + surface_points + interior_points
```

Those points appear as a separate gray canonical reference layer. They do not
change tracker input, live fused PCD, masks, TAPNext++/LiteTracker queries, or
semantic tracking/control markers. Tracked object points render red, tracked
controller points render cyan-blue. Demo 3.3 defaults to
`--overlay-display-scope union` so both tracked classes are visible together;
`--overlay-debug-color-by-camera` temporarily overrides semantic colors for
camera-alignment debugging.

The warmup profile reports `shape_prior_execution_mode =
async_background_thread`, `shape_prior_start_policy = after-teardown`,
`shape_prior_blocks_tracker_input = False`, and
`shape_prior_blocks_first_render = False`. Under
`QQTT_WSLG_OPEN3D_FAST_EXIT=1`, Demo 3.3 records a detached after-teardown
completion worker in `shape_prior_detached_completion_*` fields; that worker
waits for the live process to exit, runs the full route from the captured case,
writes `<profile>_shape_prior_completion.json`, and merges the final
`shape_prior_status` back into the live profile. `--shape-prior-gpu auto`
resolves to the mask/render GPU, and the subprocess uses
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` by default to reduce mesh
decoder fragmentation OOMs. This is a scheduling/resource-placement
optimization only; it does not skip image upscaling, text segmentation, SAM3D,
alignment, sampling, final-data coordinate validation, or render-layer attach.
`--shape-prior-start-policy after-first-render` remains available for
experiments, but on 24 GB GPUs it can OOM while live workers are resident.
`--shape-prior-skip-route-visualizations` is enabled by default for Demo 3.3
and skips only optional FuturePhysTwin diagnostic videos such as
`shape/visualization.mp4`, `final_matching.mp4`, `final_pcd.mp4`, and
`final_data.mp4`. It still runs image upscaling, text segmentation, SAM3D mesh
generation, alignment, sampling, coordinate validation, and render-layer
attachment.

Demo 3.3-specific CLI:

- `--shape-prior-warmup / --no-shape-prior-warmup`, default on
- `--futurephystwin-root`, default `/home/xinjie/FuturePhysTwin`
- `--futurephystwin-python`, default current Python interpreter. Launch Demo 3.3 with
  `conda run --no-capture-output -n demo_3_3_max python ...` to keep live runtime
  and shape-prior route in the same conda environment.
- `--sam3d-root`, default `/home/xinjie/external/sam-3d-objects`
- `--shape-prior-camera-idx`, default `0`
- `--shape-prior-force`
- `--shape-prior-start-policy {after-teardown,after-first-render,immediate}`, default `after-teardown`
- `--shape-prior-gpu`, default `auto`
- `--shape-prior-cuda-alloc-conf`, default `expandable_segments:True`
- `--shape-prior-skip-route-visualizations / --no-shape-prior-skip-route-visualizations`, default on
- `--shape-prior-retry-after-teardown / --no-shape-prior-retry-after-teardown`, default on

Dry-run:

```bash
conda run --no-capture-output -n demo_3_3_max \
  python demo_v3_3/realtime_three_view_litetracker_ffs_dual4090.py \
  --dry-run \
  --camera-ids 0,1,2 \
  --mask-gpu 0 \
  --cotracker-gpu 1 \
  --require-two-cuda \
  --calibrate-path calibrate.pkl
```

Short hardware validation:

```bash
QQTT_WSLG_OPEN3D_FAST_EXIT=1 conda run --no-capture-output -n demo_3_3_max \
  python demo_v3_3/realtime_three_view_litetracker_ffs_dual4090.py \
  --duration-s 60 \
  --camera-ids 0,1,2 \
  --mask-gpu 0 \
  --cotracker-gpu 1 \
  --require-two-cuda \
  --calibrate-path calibrate.pkl \
  --render-mode pointcloud \
  --render-micro-profile \
  --gpu-sampling \
  --gpu-sampling-device-indexes 0,1 \
  --profile-json-output docs/generated/demo33_shape_prior_warmup_60s_profile.json
```

Expected manual result: after warmup, a gray canonical prior appears around
the stuffed animal while red object-tracking markers and cyan-blue controller
tracking markers continue to come only from live depth/tracks. If fast-exit is
enabled, the live profile may first show `shape_prior_status = case_ready`; the
detached completion JSON/log should then advance it to `ready` or an explicit
error after the live workers release GPU memory.
