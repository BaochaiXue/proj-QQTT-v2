# 2026-04-21 WSL RealSense RSUSB Proof-of-Life

## Goal

Get the connected Intel RealSense D455 cameras usable from the current WSL2 Ubuntu
workspace by replacing the unreliable pip-wheel Linux backend path with an
RSUSB-backed `librealsense` build, then validate the repo's camera entrypoints
against the attached hardware.

The pass should leave behind:

- a clear diagnosis of the current WSL failure mode
- exact build/install commands for the chosen RealSense backend
- proof-of-life results for device enumeration and at least one repo entrypoint
- generated validation notes under `docs/generated/`

## Non-Goals

- no repo scope changes outside camera preview / calibration / recording support
- no fake CI hardware validation
- no vendoring of external `librealsense` source into this repo
- no Windows-side driver or firmware changes beyond already completed `usbipd` attach

## Local Path Assumptions

- QQTT repo root: `/home/zhangxinjie/proj-QQTT-v2`
- Miniconda base: `/home/zhangxinjie/miniconda3`
- camera env: `record_data_min`
- external librealsense build root: `/home/zhangxinjie/external/librealsense`

## Files To Touch

- new `docs/generated/wsl_realsense_rsusb_validation.md`
- `docs/generated/README.md`
- this exec plan

## Implementation Plan

1. confirm the current failure mode with the existing `pyrealsense2` wheel:
   - `query_devices()` behavior
   - single-camera `pipeline.start()`
   - repo entrypoint failure surface
2. inspect local build prerequisites and choose the external install path for
   an RSUSB-backed `librealsense`
3. build/install `librealsense` with `-DFORCE_RSUSB_BACKEND=ON` outside the repo,
   and make the Python binding visible from `record_data_min`
4. validate the replacement stack with:
   - device enumeration
   - single-camera pipeline start
   - `python cameras_viewer.py`
   - if stable, a minimal `record_data.py --max_frames 1` run
5. document exact commands and outcomes under `docs/generated/`

## Validation Plan

- `/home/zhangxinjie/miniconda3/envs/record_data_min/bin/python -c "import pyrealsense2 as rs; print(len(rs.context().query_devices()))"`
- `/home/zhangxinjie/miniconda3/envs/record_data_min/bin/python <single-camera pipeline probe>`
- `conda run -n record_data_min python cameras_viewer.py`
- `conda run -n record_data_min python record_data.py --num-cam 1 --max_frames 1 --disable-keyboard-listener --serials <serial>`

## Risks

- official `librealsense` upstream explicitly does not support VM installs, and WSL2
  falls into that risk bucket
- RSUSB backend may still have multi-camera limitations
- Python bindings may need to replace or shadow the existing wheel cleanly
- missing native build dependencies may slow or block the build

## Completion Checklist

- [x] confirm the pip-wheel failure mode in WSL
- [ ] verify/build required external dependencies
- [ ] install RSUSB-backed Python binding for `record_data_min`
- [ ] validate enumeration and direct pipeline start
- [ ] validate repo camera entrypoints
- [ ] write generated validation notes

## Progress Log

- 2026-04-21: confirmed all three D455 devices are attached into WSL via `usbipd`
- 2026-04-21: confirmed the stock `record_data_min` env can import `pyrealsense2` but repo camera entrypoints fail with `RuntimeError: failed to set power state`
- 2026-04-21: confirmed a direct `pyrealsense2` `pipeline.start()` probe against an attached serial fails with `RuntimeError('No device connected')`
- 2026-04-21: reviewed upstream docs indicating VM installs are unsupported and RSUSB is the recommended mitigation path for new / unsupported environments
