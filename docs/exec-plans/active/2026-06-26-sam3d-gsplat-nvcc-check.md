# SAM3D gsplat/nvcc Check Plan

Goal: make Demo v5 fail early when the SAM3D shape-prior environment cannot
build/load `gsplat_cuda`, and warm the extension during validation so the first
demo request does not emit the gsplat/nvcc layout post-optimization error.

## Root Cause

- Historical Demo v5 logs show SAM3D reaching GS layout post optimization, then
  failing while building `gsplat_cuda` because the selected `nvcc` path was
  missing.
- The existing environment checker only imports `gsplat`; that does not trigger
  the CUDA extension path used by SAM3D.
- The checker must validate the actual CUDA compiler path and run a tiny
  `gsplat.rasterization` smoke when CUDA is required.

## Scope

- Keep SAM3D's normal `gsplat` backend active.
- Do not replace SAM3D layout post optimization with an in-code fallback.
- Keep external weights and SAM3D source under `vendor/demo_runtime/`.
- Touch only Demo v5 environment validation/docs and focused tests.

## Implementation Tasks

1. [x] Add failing unit tests for `CUDA_HOME`/`CUDACXX` nvcc validation and for
   shape-prior CUDA checks invoking a gsplat runtime smoke.
2. [x] Add nvcc toolchain checks to `demo_v5/env/check_demo_v5_env.py`.
3. [x] Add a tiny `gsplat.rasterization` CUDA smoke for shape-prior
   `--require-cuda` checks.
4. [x] Update Demo v5 environment docs to explain that the checker warms
   `gsplat_cuda` and that missing nvcc must be fixed before running SAM3D.
5. [x] Run targeted unit tests, shape-prior environment check, py_compile, and
   repo smoke validation.

## Validation

- `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v5_env_check`
- `conda run -n phystwin-max --no-capture-output python demo_v5/env/check_demo_v5_env.py --role shape-prior --require-cuda`
- `conda run -n demo_2_max --no-capture-output python -m py_compile demo_v5/env/check_demo_v5_env.py`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`
