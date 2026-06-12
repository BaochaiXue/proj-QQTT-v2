# Repo Scope

This repository remains camera-first. Its formal data product is still camera
preview, calibration, recording, aligned case generation, and native-vs-FFS
comparison. The repo also carries sanctioned realtime demo, remote FFS proxy,
and tracking-diagnostic layers that exercise those camera streams.

## In Scope

- single-camera RealSense viewer / debug by default
- single-camera calibration by default
- single-camera RGB-D recording by default
- aligned case generation up to `data_process/record_data_align.py`
- optional proof-of-life tooling that evaluates an external stereo backend without modifying the main pipeline
- standalone hardware capability probing that supports future camera-only integration decisions
- native-vs-FFS comparison visualization for aligned cases
- live FFS preview, remote FFS proxy services, and replay clients
- realtime point-cloud demo entrypoints under `demo_v*`
- demo-oriented tracking backend probes, benchmarks, and overlays under
  `qqtt/tracking` and `scripts/harness/experiments/`
- SAM/EdgeTAM mask usage when limited to demo initialization, visualization, or
  diagnostic tracking artifacts

## Out Of Scope

- training or vendoring segmentation / tracking models
- shape priors
- inverse physics
- Gaussian Splatting
- downstream simulation
- evaluation pipelines for reconstruction or rendering
- robot control, teleoperation, or manipulation policy demos
- generated data archives, checkpoints, TensorRT engines, and external
  repositories as source-controlled product contents

## Fast-FoundationStereo Boundary

- FFS is external and optional.
- This repo may add proof-of-life scripts and geometry conversion helpers.
- This repo may add standalone stream capability probes and decision-support reports.
- This repo may add aligned-case comparison visualization utilities.
- This repo may add remote FFS proxy services and replay clients when they are
  diagnostics for the camera/FFS path.
- This repo does not vendor the FFS source or weights.
- The recording and alignment entrypoints remain the canonical data-production surface.
