# Scope

## Repo Charter

This repository handles 3-camera RealSense preview, calibration, synchronized RGB-D recording, and aligned case generation up to `data_process/record_data_align.py`.

## In Scope

- RealSense preview and debugging
- calibration of a 3-camera D400 setup
- synchronized RGB-D recording
- trimming and timestamp alignment of recorded cases
- packaging aligned cases under `data/`
- manual hardware validation documentation
- deterministic tests and scope guard for the kept workflow

## Out Of Scope

- segmentation or masks
- object tracking
- shape-prior generation
- scene / point-cloud modeling beyond aligned packaging
- `final_data.pkl`
- inverse physics
- Warp simulation
- Gaussian Splatting
- rendering evaluation
- teleoperation
- downstream demos or visualization for simulation pipelines

## Boundary Rule

If a file, dependency, CLI, or README section exists only to support downstream physics, rendering, tracking, or evaluation, it does not belong in this repo.
