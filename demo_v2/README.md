# Demo V2: Single-D455 Realtime Viewer

This folder is the active current-week workspace for the standalone
single-camera demo. It streams one Intel RealSense D455 and renders either:

- native RealSense color-aligned RGB-D
- live FFS depth from the D455 IR stereo pair
- HF EdgeTAM tracked controller/object masked point clouds

The demo uses the camera color frame directly: meters, `x` right, `y` down,
`z` forward. It does not read `calibrate.pkl` and does not use any multi-camera
world transform.

## Files

- `realtime_single_camera_pointcloud.py`: single-camera demo entrypoint
- `realtime_masked_edgetam_pcd.py`: Demo 2.0 masked-only EdgeTAM PCD entrypoint
- `run_wslg_open3d.sh`: optional WSLg/Open3D GUI environment wrapper

## Native RealSense Depth

```bash
conda run -n FFS-SAM-RS python demo_v2/realtime_single_camera_pointcloud.py \
  --profile 848x480 \
  --fps 60 \
  --depth-source realsense
```

The default camera view uses the fast image backend, preserves valid aligned
depth pixels, and does not apply far clipping (`--depth-max-m 0.0`).

## FFS Depth

```bash
conda run -n FFS-SAM-RS python demo_v2/realtime_single_camera_pointcloud.py \
  --profile 848x480 \
  --fps 60 \
  --depth-source ffs \
  --view-mode camera \
  --debug
```

FFS mode expects the repo-local two-stage TensorRT engine directory unless
`--ffs-trt-model-dir` is provided. It also expects the external
Fast-FoundationStereo repo path from `--ffs-repo`.

The current FFS reporting/config target is fixed to:

```text
model: 20-30-48
valid_iters: 4
input: 848x480 padded/built as 864x480
TensorRT builderOptimizationLevel: 5
```

## Orbit Point-Cloud View

```bash
conda run -n FFS-SAM-RS python demo_v2/realtime_single_camera_pointcloud.py \
  --depth-source realsense \
  --profile 848x480 \
  --fps 30 \
  --view-mode orbit
```

The script applies the WSLg/Open3D defaults before importing Open3D. The wrapper
is available when you want to force the same environment around another command:

```bash
./demo_v2/run_wslg_open3d.sh conda run -n FFS-SAM-RS python demo_v2/realtime_single_camera_pointcloud.py --view-mode orbit
```

## Demo 2.0 EdgeTAM Masked PCD

`realtime_masked_edgetam_pcd.py` is a copy-and-rewrite demo path for low-latency
masked point clouds. It keeps the single-D455 RealSense/Open3D/latest-wins
structure, but its runtime pipeline is:

```text
RealSense color + IR stereo -> FFS TensorRT depth -> HF EdgeTAM streaming masks -> masked PCD only
```

The default HF EdgeTAM session tracks two objects together:

```text
obj_id=1 controller
obj_id=2 object
```

For setup/debug scenes where the operator hand is not visible, explicitly pass
`--track-mode object-only`. That mode runs SAM3.1 only for the object prompt,
initializes EdgeTAM with `obj_id=2 object`, and leaves the controller PCD empty
instead of failing on a missing hand mask.

This demo requires compiled EdgeTAM. The only accepted runtime mode is
`--compile-mode vision-reduce-overhead`, which compiles the HF
`vision_encoder` and keeps the streaming session / object bookkeeping outside
`torch.compile`.

By default, the demo captures the first live color frame, runs SAM3.1 image
one-frame segmentation on that frame (`Sam3Processor.set_image` plus text
prompt), uses the resulting controller/object masks to initialize EdgeTAM, then
tracks with EdgeTAM only:

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2/realtime_masked_edgetam_pcd.py \
  --profile 848x480 \
  --fps 60 \
  --depth-source ffs \
  --init-mode sam31-first-frame \
  --track-mode controller-object \
  --controller-prompt "hand" \
  --object-prompt "stuffed animal" \
  --compile-mode vision-reduce-overhead \
  --dtype bfloat16 \
  --depth-min-m 0.2 \
  --depth-max-m 1.5 \
  --pcd-max-points 60000 \
  --pcd-color-mode rgb \
  --render-every-n 1 \
  --debug
```

The script intentionally does not render the full-scene point cloud by default,
does not run per-frame SAM3.1, and uses FFS depth by default. SAM3.1 is only
used once on the live first frame through the image path, not through
`propagate_in_video`; after that first-frame initialization, the demo exits its
CUDA autocast context and clears the CUDA cache before continuing with EdgeTAM
streaming. The default FFS
engine path is the repo's `20-30-48 / valid_iters=4 / 848x480 -> 864x480 /
builderOptimizationLevel=5` TensorRT artifact; pass `--depth-source realsense`
only when you need a native-depth fallback. Masked points are colored from the
live RGB frame by default; use `--pcd-color-mode class` to switch back to fixed
controller/object colors. `--init-mode saved-masks` remains available only for
debugging controlled replay-style startup; it is not the default live demo path.

PCD filtering is deliberately split from the hot path. The demo builds a
raw/capped masked PCD every frame; when `--enable-pcd-filter` is set, it submits
the capped object/controller clouds to a latest-wins filter worker every
`--filter-every-n` frames. Rendering uses the latest filtered output if one is
available, otherwise it keeps showing the current raw/capped cloud. This keeps
slow filter frames from blocking capture, FFS, EdgeTAM, or Open3D.

Recommended WSL RTX 5090 live filter policy:

```text
object:     voxel cap -> enhanced-pt
controller: voxel cap -> pt-filter
```

Fast live path:

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2/realtime_masked_edgetam_pcd.py \
  --profile 848x480 \
  --fps 60 \
  --depth-source realsense \
  --track-mode controller-object \
  --pcd-mode masked \
  --render-mode pointcloud \
  --pcd-stride 2 \
  --pcd-max-points 10000 \
  --pcd-color-mode class \
  --render-every-n 2 \
  --enable-pcd-filter \
  --pcd-filter-mode async \
  --filter-every-n 3 \
  --object-filter enhanced-pt \
  --controller-filter pt-filter \
  --object-filter-cap 12000 \
  --controller-filter-cap 12000 \
  --object-filter-voxel-m 0.005 \
  --controller-filter-voxel-m 0.003 \
  --debug
```

Professor-facing local FFS speed preset:

Use this when the demo must stay fully FFS-derived on the local RTX 5090 Laptop
but needs lower Open3D/render load. The preset does not change the FFS engine,
checkpoint, valid iteration count, or EdgeTAM compile mode. It only caps the
rendered masked points and adjusts display latency/point-size defaults:

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2/realtime_masked_edgetam_pcd.py \
  --serial 239222300412 \
  --profile 848x480 \
  --fps 60 \
  --depth-source ffs \
  --demo-preset local-ffs-professor \
  --init-mode sam31-first-frame \
  --track-mode object-only \
  --object-prompt "stuffed animal" \
  --pcd-mode masked \
  --render-mode pointcloud \
  --compile-mode vision-reduce-overhead \
  --dtype bfloat16 \
  --depth-min-m 0.2 \
  --depth-max-m 1.5 \
  --pcd-color-mode rgb \
  --render-every-n 2 \
  --enable-pcd-filter \
  --pcd-filter-mode async \
  --filter-every-n 3 \
  --object-filter enhanced-pt \
  --controller-filter pt-filter \
  --object-filter-cap 20000 \
  --controller-filter-cap 20000 \
  --debug
```

With the current local profiling, the formal no-render FFS path is around
`25 FPS`. If WSLg/Open3D render becomes the visible bottleneck, keep the same
command and add `--render-every-n 2` for a steadier lower-rate display.

## Remote FFS Depth

Use `--depth-source ffs_remote` when the RealSense camera stays on the local
laptop but FFS TensorRT depth should run on a second GPU machine. This is a
remote service offload, not remote CUDA device sharing: the local process still
runs EdgeTAM and UI on the local GPU.

Both client and server environments need `pyzmq` installed.

Build or validate the FFS TensorRT engine on the remote GPU machine before
starting the real server. Do not assume an engine serialized on the 5090 laptop
will run on a 4090; TensorRT engines are not generally portable across GPU
architectures without explicit compatibility settings.

First verify the network path with echo-only mode:

```bash
# Remote GPU machine
conda run --no-capture-output -n demo_2_max \
  python services/ffs_remote/ffs_depth_server.py \
  --bind tcp://0.0.0.0:7001 \
  --echo-only \
  --debug

# Local camera/UI machine
conda run --no-capture-output -n demo_2_max \
  python services/ffs_remote/ffs_depth_client.py \
  --endpoint tcp://<remote_tailscale_ip>:7001 \
  --echo-benchmark \
  --profile 848x480 \
  --fps 30 \
  --duration-s 20 \
  --debug
```

Start the server on the remote GPU machine. The official Demo 2 remote path
must prove the same FFS engine contract as the local quality baseline:
`20-30-48`, `valid_iters=4`, `848x480 -> pad 864x480`,
`builderOptimizationLevel=5`, `max_disp=192`.

```bash
conda run --no-capture-output -n demo_2_max \
  python services/ffs_remote/ffs_depth_server.py \
  --bind tcp://0.0.0.0:7001 \
  --ffs-repo ../Fast-FoundationStereo \
  --ffs-trt-model-dir data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864 \
  --return masked_uv_depth \
  --warmup 20 \
  --debug \
  --strict-engine-contract \
  --required-model 20-30-48 \
  --required-valid-iters 4 \
  --required-height 480 \
  --required-width 864 \
  --required-builder-optimization-level 5 \
  --required-max-disp 192
```

Run the formal local FFS quality baseline on the local camera/UI machine when a
remote direct path is not fast enough. This is the semantically correct
single-machine baseline even if it is only around the mid-20 FPS range:

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2/realtime_masked_edgetam_pcd.py \
  --serial 239222300412 \
  --profile 848x480 \
  --fps 60 \
  --depth-source ffs \
  --init-mode sam31-first-frame \
  --track-mode object-only \
  --object-prompt "stuffed animal" \
  --pcd-mode masked \
  --render-mode pointcloud \
  --compile-mode vision-reduce-overhead \
  --dtype bfloat16 \
  --depth-min-m 0.2 \
  --depth-max-m 1.5 \
  --pcd-max-points 60000 \
  --pcd-color-mode rgb \
  --debug \
  --profile-cuda-events
```

Run the formal remote sparse FFS path only when the network can support it. This
path sends the same-frame EdgeTAM mask plus IR pair to the remote server and
builds the PCD only from returned FFS-derived sparse depth/points. It does not
fall back to native RealSense depth:

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2/realtime_masked_edgetam_pcd.py \
  --serial 239222300412 \
  --profile 848x480 \
  --fps 60 \
  --depth-source ffs_remote \
  --ffs-remote-endpoint tcp://<remote_tailscale_ip>:7001 \
  --ffs-remote-max-inflight 1 \
  --ffs-remote-timeout-ms 5000 \
  --ffs-remote-return masked_uv_depth \
  --init-mode sam31-first-frame \
  --track-mode object-only \
  --object-prompt "stuffed animal" \
  --pcd-mode masked \
  --render-mode pointcloud \
  --compile-mode vision-reduce-overhead \
  --dtype bfloat16 \
  --depth-min-m 0.2 \
  --depth-max-m 1.5 \
  --pcd-max-points 60000 \
  --pcd-color-mode rgb \
  --debug \
  --profile-cuda-events
```

The first full-frame implementation is intentionally conservative:
`--ffs-remote-max-inflight` must be `1`. Each PCD packet uses depth returned for
the same frame `seq`; timed out replies are skipped instead of mixing older
depth with newer masks. `--debug` reports `remote_rtt_ms`,
`remote_server_total_ms`, `remote_request_kb`, and `remote_response_kb` in
addition to `ffs_ms` and `ffs_align_ms`.

Fallback/debug only: if you need a fast UI path while debugging networking, use
native RealSense depth as the main path and enable remote FFS as a low-FPS
comparison side channel. This is not the official professor-facing Demo 2
quality output because the rendered PCD comes from native RealSense depth:

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2/realtime_masked_edgetam_pcd.py \
  --serial 239222300412 \
  --profile 848x480 \
  --fps 60 \
  --depth-source realsense \
  --enable-remote-ffs-quality \
  --remote-ffs-quality-endpoint tcp://<remote_tailscale_ip>:7001 \
  --remote-ffs-quality-return masked_uv_depth \
  --remote-ffs-quality-compress none \
  --remote-ffs-quality-interval-ms 200 \
  --init-mode sam31-first-frame \
  --track-mode object-only \
  --object-prompt "stuffed animal" \
  --pcd-mode masked \
  --compile-mode vision-reduce-overhead \
  --dtype bfloat16 \
  --debug
```

Protocol experiments can use `--return masked_uv_depth` or `--return
masked_xyz` on the server/client utilities, plus `--compress none|zstd|lz4|png`.
For formal Demo 2, sparse modes are official only when they are the main
`--depth-source ffs_remote` path and are computed by the remote FFS opt5 engine.

Object-only startup when no hand is in view:

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2/realtime_masked_edgetam_pcd.py \
  --profile 848x480 \
  --fps 60 \
  --depth-source ffs \
  --demo-preset local-ffs-professor \
  --init-mode sam31-first-frame \
  --track-mode object-only \
  --object-prompt "stuffed animal" \
  --compile-mode vision-reduce-overhead \
  --dtype bfloat16 \
  --depth-min-m 0.2 \
  --depth-max-m 1.5 \
  --pcd-color-mode rgb \
  --debug
```

Profiling isolation runs should be headless and use `--duration-s`:

```bash
# Capture only
python demo_v2/realtime_masked_edgetam_pcd.py --depth-source none --track-mode none --pcd-mode none --render-mode none --duration-s 10 --debug

# EdgeTAM only, no depth/PCD/render
python demo_v2/realtime_masked_edgetam_pcd.py --depth-source none --track-mode object-only --pcd-mode none --render-mode none --duration-s 30 --debug --profile-cuda-events

# FFS only, no EdgeTAM/PCD/render
python demo_v2/realtime_masked_edgetam_pcd.py --depth-source ffs --track-mode none --pcd-mode none --render-mode none --duration-s 30 --debug

# Full compute path without Open3D rendering
python demo_v2/realtime_masked_edgetam_pcd.py --depth-source ffs --track-mode object-only --pcd-mode masked --render-mode none --duration-s 30 --debug --profile-cuda-events
```

Default live timing avoids device-wide CUDA synchronizes around each timed
stage. Pass `--profile-sync` only when a synchronized diagnostic run is needed.
Pass `--profile-cuda-events` to report `cuda_event_model_ms` next to
`wall_model_ms` for EdgeTAM forward timing.

## Moving This Folder

Native RealSense mode can run from this folder as long as the Python environment
has `pyrealsense2`, `open3d`, `numpy`, and optionally `opencv-python` / `numba`.
For FFS mode outside the repo, set:

```bash
export QQTT_REPO_ROOT=/path/to/proj-QQTT-v2
```

or pass explicit `--ffs-repo` and `--ffs-trt-model-dir` paths.
