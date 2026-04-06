# Repo Scope

This repository remains camera-only.

## In Scope

- 3-camera RealSense viewer / debug
- multi-camera calibration
- multi-camera RGB-D recording
- aligned case generation up to `data_process/record_data_align.py`
- optional proof-of-life tooling that evaluates an external stereo backend without modifying the main pipeline
- standalone hardware capability probing that supports future camera-only integration decisions
- native-vs-FFS comparison visualization for aligned cases

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
- This repo may add standalone stream capability probes and decision-support reports.
- This repo may add aligned-case comparison visualization utilities.
- This repo does not vendor the FFS source or weights.
- The recording and alignment entrypoints remain the canonical data-production surface.
