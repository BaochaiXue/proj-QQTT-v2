# 2026-04-28 Official FFS Triton TensorRT Path

## Goal

Align QQTT's two-stage Fast-FoundationStereo TensorRT runtime with the official NVlabs implementation:

- feature TensorRT engine
- Triton GWC volume kernel
- post TensorRT engine

The current QQTT path monkey-patches the official Triton GWC function to `build_gwc_volume_optimized_pytorch1`, which makes the two-stage runtime a mixed TensorRT/PyTorch path and prevents apples-to-apples latency comparisons with official TRT profiles.

## Scope

- Update QQTT two-stage TensorRT runner to use the official `build_gwc_volume_triton` path by default.
- Update the WSL TensorRT validation harness to stop patching Triton to PyTorch.
- Add deterministic checks that the two-stage runner no longer monkey-patches the official GWC function.
- Document the environment requirement and the current local Triton failure if it still occurs.

## Out of Scope

- Rebuilding all static replay matrix artifacts.
- Changing single-engine TensorRT behavior.
- Changing PyTorch FFS behavior.
- Faking hardware validation in CI.

## Validation

1. Run a focused import/unit test for the TensorRT runner setup.
2. Probe official Triton GWC on the local environment.
3. Run `python scripts/harness/check_all.py`.
4. If Triton still fails locally, record the exact failure and next environment action instead of silently falling back to PyTorch.

## Current Result

- Code path updated: QQTT two-stage TensorRT now loads the official Fast-FoundationStereo `TrtRunner` with `core.submodule.build_gwc_volume_triton`; it no longer replaces that function with `build_gwc_volume_optimized_pytorch1`.
- Focused test passed:
  - `/home/zhangxinjie/miniconda3/envs/qqtt-ffs-compat/bin/python -m unittest -v tests.test_ffs_tensorrt_single_engine_smoke`
- Quick harness passed:
  - `/home/zhangxinjie/miniconda3/envs/qqtt-ffs-compat/bin/python scripts/harness/check_all.py`
- Local environment blocker remains:
  - official Triton GWC probe fails in both `qqtt-ffs-compat` and `ffs-standalone`
  - environment: Python `3.10.18`, Torch `2.7.0+cu128`, Triton `3.3.0`
  - failure: `SystemError: PY_SSIZE_T_CLEAN macro must be defined for '#' formats`

Next environment action: create or switch to an official-compatible FFS runtime before using the two-stage TensorRT path for live/viewer inference. The upstream README recommends Python `3.12`, Torch `2.6.0`, TorchVision `0.21.0`, xFormers, and the repo requirements.
