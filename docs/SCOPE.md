# Scope

## Repo Charter

This `single-camera` branch handles single-camera RealSense preview,
calibration, synchronized RGB-D recording, aligned case generation,
native-vs-FFS comparison visualization, and sanctioned single-camera
demo/proxy diagnostics built on that camera stream. The `main` branch remains
the protected 3-camera baseline.

The formal data product stops at aligned cases written by
`data_process/record_data_align.py` and realtime native aligned export from
`record_data_realtime_align.py`. Demo, proxy, mask, and visualization code must
remain a diagnostic layer around that data product.

## In Scope

- RealSense preview and debugging
- calibration of a single D400 camera by default
- synchronized RGB-D recording
- trimming and timestamp alignment of recorded cases
- packaging aligned cases under `data/`
- realtime native RGB-D aligned export to the formal `different_types`-style interface
- live FFS preview and single-camera remote FFS proxy services used to evaluate RealSense/FFS throughput
- aligned depth-backend comparison visualization utilities
- explicitly isolated experiment-only aligned-case visualization under
  `data_process/visualization/experiments/` and `scripts/harness/experiments/`
- single-camera realtime point-cloud workflows under `scripts/harness/realtime_single_camera_pointcloud.py` and `demo_v3*`
- SAM/EdgeTAM mask usage when it is scoped to demo initialization or visualization; masks are not part of the formal aligned-case compatibility contract
- manual hardware validation documentation
- deterministic tests and scope guards for the kept workflow

## Out Of Scope

- three-camera demo entrypoints on this branch
- dual-GPU demo contracts
- batch-3 demo runtimes and batch-3 TRT helper scripts
- tracking backend registries, lifted tracking overlays, and tracker benchmark harnesses
- training or vendoring segmentation / tracking models
- shape-prior generation
- scene / point-cloud modeling beyond aligned packaging and sanctioned realtime demo/diagnostic views
- `final_data.pkl`
- inverse physics
- Warp simulation
- Gaussian Splatting
- rendering evaluation
- teleoperation
- robot control, manipulation policy, or simulation-facing production demos
- generated data archives, checkpoints, TensorRT engines, and external repositories as first-class repo contents

## Boundary Rule

Core recording and alignment entrypoints must not import demo, proxy, mask, or
experiment visualization packages. Demo and proxy code may depend on shared
camera runtime, FFS geometry/runners, calibration loaders, and documented
external weights, but it must not make segmentation artifacts part of the
formal aligned-case output contract.

If a file, dependency, CLI, or README section exists only to support downstream
physics, shape priors, tracking backends, reconstruction/rendering evaluation,
robot control, or simulation/manipulation pipelines, it does not belong in this
branch.

Experiment-only FFS depth diagnostics are allowed only when they consume
aligned cases, write diagnostic artifacts outside formal case directories, and
remain separated from recording / alignment runtime imports.
