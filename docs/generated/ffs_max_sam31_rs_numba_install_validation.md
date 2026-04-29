# FFS-max-sam31-rs Numba Install Validation

Date: 2026-04-29

Purpose: add optional Numba fused backprojection support for
`scripts/harness/realtime_single_camera_pointcloud.py` without changing the
preserved `torch==2.11.0+cu130`, CUDA 13.0, Open3D, RealSense, or NumPy stack.

Pre-install snapshots:

- `docs/generated/ffs-max-sam31-rs-pre-numba-20260429-explicit.txt`
- `docs/generated/ffs-max-sam31-rs-pre-numba-20260429-pip-freeze.txt`

Environment check before install:

```bash
conda run -n FFS-max-sam31-rs python -c "import sys, numpy as np; print(sys.version); print('numpy', np.__version__)"
conda run -n FFS-max-sam31-rs python -c "import importlib.util; print(importlib.util.find_spec('numba'))"
```

Observed result:

- Python `3.12.13`
- NumPy `2.4.4`
- Numba was not installed

Conda dry run:

```bash
conda install -n FFS-max-sam31-rs numba --dry-run --json
```

Outcome: solver succeeded, but would also link NumPy/MKL-related conda packages.
Because the environment's NumPy/Torch/Open3D/RealSense stack is pip-managed,
the lower-risk install path was to add only pip wheels for Numba and llvmlite.

Install command:

```bash
conda run -n FFS-max-sam31-rs python -m pip install numba
```

Installed packages:

- `numba==0.65.1`
- `llvmlite==0.47.0`

Post-install validation:

```bash
conda run -n FFS-max-sam31-rs python -c "import numba, llvmlite, numpy; print('numba', numba.__version__); print('llvmlite', llvmlite.__version__); print('numpy', numpy.__version__)"
conda run -n FFS-max-sam31-rs python -m unittest -v tests.test_realtime_single_camera_pointcloud_smoke
```

Observed result:

- `numba 0.65.1`
- `llvmlite 0.47.0`
- `numpy 2.4.4`
- realtime single-camera smoke tests passed, including Numba-vs-NumPy
  backprojection equivalence.

Follow-up note: the Numba helper intentionally uses in-process JIT compilation
without `cache=True`. Disk caching was avoided because direct execution as
`python scripts/harness/realtime_single_camera_pointcloud.py` can have a
different import root than package-style test imports, which can make Numba's
cache loader try to import a module path that is not present on `sys.path`.
