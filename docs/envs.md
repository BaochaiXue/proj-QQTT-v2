# Environments

## Shared CUDA Toolkit Policy

- Current shared WSL toolkit: `/usr/local/cuda`.
- For CUDA-family extension builds, prefer the shared toolkit:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

- Do not install Linux NVIDIA driver packages inside WSL.
- Keep environment-specific CUDA deviations documented in the active plan that
  needs them.

## `demo_2_max`

- Purpose: default integrated local environment for this branch.
- Expected stack:
  - RealSense / `pyrealsense2`
  - Fast-FoundationStereo
  - TensorRT / Triton as needed by local FFS engines
  - Open3D
  - SAM 3.1 / EdgeTAM dependencies used by the single-camera masked PCD demo
- Expected use:
  - `scripts/harness/diagnostics/demo/realtime_single_camera_pointcloud.py`
  - `demo_v3*`
  - deterministic checks
  - integrated local demo work where switching specialized envs is the main friction
- Validation command:

```bash
conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke
```

- Demo v5 install/check material lives in `demo_v5/env/`. Use
  `demo_v5/env/environment-demo-v5-main.yml` plus
  `demo_v5/env/requirements-demo-v5-main.txt` for the main process and
  `demo_v5/env/environment-demo-v5-shape-prior.yml` plus
  `demo_v5/env/requirements-demo-v5-shape-prior.txt` for the SAM3D worker.

## `FFS-SAM-RS`

- Purpose: FFS/SAM/RealSense stack for viewer, static replay, and visualization
  work that specifically needs that environment.
- Expected use:
  - `cameras_viewer_FFS.py`
  - `scripts/harness/benchmarks/ffs/verify_ffs_demo.py`
  - FFS depth comparison and mask helper experiments
- Default FFS policy:
  - external repo: `../Fast-FoundationStereo`
  - checkpoint family: `20-30-48` unless an experiment explicitly selects another
  - keep recorded `848x480` frames and pad only when the selected engine shape requires it

## `FFS-max-sam31-rs`

- Purpose: FFS max-stack experiments that must keep the local max torch/CUDA/TRT
  stack while also importing QQTT RealSense entrypoints and SAM 3.1 helpers.
- Expected use:
  - explicit FFS/SAM/RealSense comparison tasks
  - not the default integrated demo environment

## `edgetam-max`

- Purpose: isolated EdgeTAM validation environment.
- Expected use:
  - EdgeTAM proof-of-life and model validation outside the default integrated
    single-camera workflow
  - not the default RS/FFS demo environment

## External Assets

External repos, checkpoints, TensorRT engines, and generated proof artifacts stay
outside this repo unless a documented validation plan explicitly says otherwise.
Use `docs/external-deps.md` for source-of-truth paths and acquisition notes.
