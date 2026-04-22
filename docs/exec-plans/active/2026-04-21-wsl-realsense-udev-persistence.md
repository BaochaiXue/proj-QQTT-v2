# 2026-04-21 WSL RealSense Udev Persistence

## Goal

Replace the temporary manual `chmod` workaround for WSL-attached Intel RealSense D455
raw USB nodes with a persistent `udev` rule so the RSUSB-backed `librealsense` path
comes up ready after the devices are attached to WSL.

The pass should leave behind:

- a narrow `udev` rule for the D455 USB device nodes used by the current WSL flow
- a repo-owned installer script that writes the rule into `/etc/udev/rules.d/`
- updated WSL environment docs that explain the remaining Windows-side `usbipd` attach
  requirement and the one-time Linux-side install step

## Non-Goals

- no change to RealSense capture logic
- no change to Windows-side driver or firmware state
- no vendoring of external `librealsense`
- no attempt to bypass Linux root privileges for system `udev` installation

## Local Path Assumptions

- QQTT repo root: `/home/zhangxinjie/proj-QQTT-v2`
- WSL distro: `Ubuntu`
- camera env: `record_data_min`
- target rule path: `/etc/udev/rules.d/99-qqtt-realsense-wsl.rules`

## Files To Touch

- new `env_install/99-qqtt-realsense-wsl.rules`
- new `env_install/install_wsl_realsense_udev.sh`
- `docs/ARCHITECTURE.md`
- `docs/envs.md`
- `docs/generated/wsl_realsense_rsusb_validation.md`
- this exec plan

## Implementation Plan

1. inspect the current WSL `udev` state and confirm the exact RealSense USB match keys
2. add a repo-owned `udev` rule template that only targets Intel D455 raw USB device nodes
3. add a small installer script that copies the rule into `/etc/udev/rules.d/`, reloads
   `udev`, and retriggers matching devices
4. update environment and validation docs with:
   - the one-time install command
   - the remaining `usbipd --wsl` / `--auto-attach` expectation on Windows
5. run deterministic repo checks after the repo-side changes

## Validation Plan

- `udevadm info -q property -n /dev/bus/usb/002/002`
- `bash env_install/install_wsl_realsense_udev.sh --print-rule`
- `conda run -n qqtt-ffs-compat python scripts/harness/check_all.py`

## Risks

- system rule installation still requires `sudo`
- the current WSL readiness story still depends on Windows-side `usbipd` attach policy
- future cameras with different product IDs would require expanding the rule

## Completion Checklist

- [x] inspect current `udev` match surface
- [ ] add rule template
- [ ] add installer script
- [ ] update docs
- [ ] run deterministic checks
- [ ] install the persistent rule into `/etc/udev/rules.d/`

## Progress Log

- 2026-04-21: confirmed WSL runs `systemd-udevd` and exposes D455 USB nodes as `SUBSYSTEM=usb`, `DEVTYPE=usb_device`
- 2026-04-21: confirmed current user is a member of `plugdev`
- 2026-04-21: confirmed the temporary `chmod` workaround is replacing missing persistent USB-node permissions rather than a repo logic error
