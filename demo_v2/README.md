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

By default, the demo captures the first live color frame, runs SAM3.1 once on
that frame, uses the resulting controller/object masks to initialize EdgeTAM,
then tracks with EdgeTAM only:

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
used once on the live first frame. The default FFS
engine path is the repo's `20-30-48 / valid_iters=4 / 848x480 -> 864x480 /
builderOptimizationLevel=5` TensorRT artifact; pass `--depth-source realsense`
only when you need a native-depth fallback. Masked points are colored from the
live RGB frame by default; use `--pcd-color-mode class` to switch back to fixed
controller/object colors. `--init-mode saved-masks` remains available only for
debugging controlled replay-style startup; it is not the default live demo path.

Object-only startup when no hand is in view:

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2/realtime_masked_edgetam_pcd.py \
  --profile 848x480 \
  --fps 60 \
  --depth-source ffs \
  --init-mode sam31-first-frame \
  --track-mode object-only \
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

## Moving This Folder

Native RealSense mode can run from this folder as long as the Python environment
has `pyrealsense2`, `open3d`, `numpy`, and optionally `opencv-python` / `numba`.
For FFS mode outside the repo, set:

```bash
export QQTT_REPO_ROOT=/path/to/proj-QQTT-v2
```

or pass explicit `--ffs-repo` and `--ffs-trt-model-dir` paths.
