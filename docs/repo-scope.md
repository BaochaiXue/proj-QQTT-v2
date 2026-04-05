# Repo Scope

This repository remains camera-only.

## In Scope

- 3-camera RealSense viewer / debug
- multi-camera calibration
- multi-camera RGB-D recording
- aligned case generation up to `data_process/record_data_align.py`
- optional proof-of-life tooling that evaluates an external stereo backend without modifying the main pipeline

## Out Of Scope

- segmentation
- dense tracking
- shape priors
- inverse physics
- Gaussian Splatting
- downstream simulation
- evaluation pipelines for reconstruction or rendering
- teleoperation / interactive demos

## Fast-FoundationStereo Boundary

- FFS is external and optional.
- This repo may add proof-of-life scripts and geometry conversion helpers.
- This repo does not vendor the FFS source or weights.
- This phase does not change the main recording or alignment entrypoints.

