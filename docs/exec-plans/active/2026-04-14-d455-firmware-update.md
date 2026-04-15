# 2026-04-14 D455 Firmware Update

## Goal

Update the connected D455 cameras to the latest firmware recommended by the current official RealSense SDK/tooling and record the exact commands and outcomes.

## Non-Goals

- no changes to repo runtime defaults
- no calibration, recording, or alignment work in this task
- no firmware downgrade unless required to recover from a failed update

## Files To Touch

- `docs/generated/` firmware update log for commands and outcomes

## Implementation Plan

1. confirm the current and recommended firmware versions reported by the connected devices
2. obtain the official RealSense firmware-update tool from the latest official `librealsense` release
3. inspect the tool's CLI and run updates only for cameras below the recommended version
4. re-enumerate devices and verify the reported firmware versions
5. save commands and outcomes under `docs/generated/`

## Validation Plan

- confirm all three expected serials enumerate after the update
- confirm each device reports `firmware_version == recommended_firmware_version`
