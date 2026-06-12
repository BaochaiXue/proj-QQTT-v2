# Scope

## Repo Charter

This repository handles 3-camera RealSense preview, calibration, synchronized
RGB-D recording, aligned case generation, native-vs-FFS comparison
visualization, and sanctioned realtime demo/proxy diagnostics built on those
camera streams.

The formal data product still stops at aligned cases written by
`data_process/record_data_align.py` and realtime native aligned export from
`record_data_realtime_align.py`. Demo, proxy, tracking, and visualization code
must remain a diagnostic layer around that data product.

## In Scope

- RealSense preview and debugging
- calibration of a 3-camera D400 setup
- synchronized RGB-D recording
- trimming and timestamp alignment of recorded cases
- packaging aligned cases under `data/`
- realtime native RGB-D aligned export to the formal `different_types`-style interface
- live FFS preview and remote FFS proxy services used to evaluate RealSense/FFS throughput
- aligned depth-backend comparison visualization utilities
- explicitly isolated experiment-only aligned-case visualization under
  `data_process/visualization/experiments/` and `scripts/harness/experiments/`
- demo-only realtime point-cloud workflows under `demo_v*`
- Demo 3.3 may invoke external FuturePhysTwin and MV-SAM3D during warmup to
  render a diagnostic shape-prior reference layer, with cache artifacts written
  under the demo output root and outside formal case directories
- shared tracking diagnostics under `qqtt/tracking` and Demo 3 harness scripts
- SAM/EdgeTAM mask usage when it is scoped to demo initialization, tracking
  diagnostics, or visualization; masks are not part of the formal aligned-case
  compatibility contract
- manual hardware validation documentation
- deterministic tests and scope guard for the kept workflow

## Out Of Scope

- training or vendoring segmentation / tracking models
- shape-prior generation
- scene / point-cloud modeling beyond aligned packaging and sanctioned
  realtime demo/diagnostic views
- `final_data.pkl`
- inverse physics
- Warp simulation
- Gaussian Splatting
- rendering evaluation
- teleoperation
- robot control, manipulation policy, or simulation-facing production demos
- generated data archives, checkpoints, TensorRT engines, and external
  repositories as first-class repo contents

## Boundary Rule

Core recording and alignment entrypoints must not import demo, tracking, or
experiment visualization packages. Demo and proxy code may depend on shared
camera runtime, FFS geometry/runners, calibration loaders, and documented
external weights, but it must not make segmentation/tracking artifacts part of
the formal aligned-case output contract.

If a file, dependency, CLI, or README section exists only to support downstream
physics, shape priors, reconstruction/rendering evaluation, robot control, or
simulation/manipulation pipelines, it does not belong in this repo.

Experiment-only FFS depth diagnostics are allowed only when they consume
aligned cases, write diagnostic artifacts outside formal case directories, and
remain separated from recording / alignment runtime imports.

The only current shape-prior carveout is Demo 3.3 warmup: it may synthesize a
one-frame FuturePhysTwin-style cache and read `final_data.pkl` from that cache
for render-only diagnostics. Formal recording, realtime aligned export, and
`data_process/record_data_align.py` outputs remain unchanged and must not import
or depend on that path.
