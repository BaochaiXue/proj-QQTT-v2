# Demo 3.3 Single-View SAM3D Shape-Prior Warmup

## Goal

Add Demo 3.3 as an experimental Demo 3.2-derived runtime that builds a
FuturePhysTwin-style single-view original SAM 3D Objects shape prior during
warmup and renders it as a separate canonical reference layer.

## Plan

1. Keep Demo 3.2 unchanged and route `demo_v3_3/` through a new Demo 3.3
   runtime wrapper.
2. Add a demo-only FuturePhysTwin warmup helper that writes a frame0-only case,
   runs the external single-view original SAM 3D Objects route, and loads
   `final_data.pkl`.
3. Add a no-op live-runtime hook that Demo 3.3 overrides at the first valid
   strict-source tracking input.
4. Extend render packets/profile fields with a render-only gray canonical
   reference layer while preserving tracker masks, query inputs, and live PCD
   semantics.
5. Update docs and deterministic tests for Demo 3.3 contracts and helper
   behavior.
6. Optimize warmup flow without lowering quality: snapshot the first valid
   strict-source input, launch a detached after-teardown completion worker when
   fast-exit profiling writes the pre-teardown profile, and give the subprocess
   an explicit `CUDA_VISIBLE_DEVICES` placement plus allocator settings to avoid
   mesh decoder fragmentation OOM. Keep after-first-render as an opt-in
   experiment, but default to teardown execution because live GPU residency
   leaves insufficient 24 GB headroom for original SAM3D mesh decode. Do not
   skip SAM3D, FuturePhysTwin sampling, coordinate validation, or render-layer
   attach.

## Validation

- Focused helper tests for case writing, command order, and final-data loading.
- Contract tests for Demo 3.3 dry-run and Demo 3.2 regression.
- `conda run -n demo_2_max --no-capture-output python -m pytest tests/test_demo31_dual_gpu_contract.py -q`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`

## Progress

- Added `qqtt.demo.demo33_shape_prior_warmup` for synthetic one-frame case
  writing, exact FuturePhysTwin command construction, route execution, and
  `final_data.pkl` canonical structure loading.
- Corrected the default SAM3D root to original `facebookresearch/sam-3d-objects`
  at `/home/xinjie/external/sam-3d-objects`; MV-SAM3D paths are rejected for
  Demo 3.3 warmup.
- Added `qqtt.demo.demo33_runtime` and pointed `demo_v3_3/` at it.
- Added the shared no-op warmup hook, Demo 3.3 override, render-only packet
  fields, gray canonical layer rendering, docs, and deterministic tests.
- Warmup-flow improvement: make shape-prior execution asynchronous with
  first-frame snapshotting, detached after-teardown startup for
  `QQTT_WSLG_OPEN3D_FAST_EXIT=1`, explicit shape-prior GPU placement,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, completion JSON/log
  artifacts, and profile fields showing that it does not block tracker input or
  first render while preserving the full FuturePhysTwin/SAM3D route.
- Validation passed:
  - `conda run -n demo_2_max --no-capture-output python -m pytest tests/test_demo33_shape_prior_warmup.py -q`
  - `conda run -n demo_2_max --no-capture-output python -m pytest tests/test_demo31_dual_gpu_contract.py -q`
  - `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
