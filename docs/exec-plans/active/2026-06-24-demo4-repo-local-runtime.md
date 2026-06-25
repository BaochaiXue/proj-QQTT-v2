# Demo 4 Repo-Local Runtime Execution Plan

## Objective

Remove repo-external default runtime paths from Demo 4 and the Demo 3.2
subsystems it launches, copy referenced external modules/weights/assets into a
repo-local runtime location, and validate only Demo 4 fake-live.

## Approved Defaults

- Runtime asset root: `vendor/demo_runtime/`
- Demo 4 output root: `result/demo_v4/futurephystwin_chunks`
- Shape-prior SAM3D root: `vendor/demo_runtime/sam-3d-objects`
- Shape-prior FuturePhysTwin root: `vendor/demo_runtime/FuturePhysTwin`
- FFS root: `vendor/demo_runtime/Fast-FoundationStereo`
- TAPNext++ root: `vendor/demo_runtime/tapnet`
- TAPNext++ checkpoint:
  `vendor/demo_runtime/checkpoints/tapnextpp/tapnextpp_ckpt.pt`

The user explicitly approved continuing without design approval prompts.

## Checklist

- [x] Confirm branch and upstream workflow.
- [x] Copy external runtime assets into `vendor/demo_runtime/`.
- [x] Remove absolute or parent-repo fallback defaults from Demo 4, shape-prior
      worker, FFS defaults, and TAPNext++ defaults.
- [x] Update tests that assert old defaults.
- [x] Update Demo 4, Demo 3.2, external dependency, and hardware validation
      docs.
- [x] Run focused parser/path tests.
- [x] Run only Demo 4 fake-live as final live validation.
- [x] Record copied asset state and validation outcomes under `docs/generated/`.

## Validation Outcome

- Focused tests passed: `101 tests`, `OK`.
- Demo 4 fake-live completed one 25-frame chunk with default
  `depth_backend=native-realsense`, realtime GPU0, shape-prior worker GPU1, and
  repo-local relative runtime defaults.
- The validated run wrote:
  `result/demo_v4/repo_local_realsense_final_20260624/data/repo_local_realsense_final/final_data.pkl`.
- Shape prior completed through real x4 upscaling, SAM3D inference,
  single-view alignment, and data-process-compatible sampling:
  700 surface points and 1000 interior points.
- All 100 MB or larger model weight/cache files are now stored under
  `vendor/demo_runtime/checkpoints/`; upstream-expected paths are repo-local
  relative symlinks.
- `--warmup-models` was attempted and failed on the local RTX 4090 with a
  24 GB VRAM CUDA OOM during dummy SAM3D decode. The successful validation used
  `--preload-models`, which still keeps model loading off the camera critical
  path.
- Optional gsplat layout post-optimization in `phystwin-max` logged a non-fatal
  `nvcc`/`gsplat_cuda` extension issue; SAM3D still returned ready shape-prior
  points and Demo 4 produced final data.

## Execution Notes

- Do not use symlinks for copied runtime assets.
- Do not copy external `.git` directories.
- Large weight files may remain ignored by extension patterns, but must exist in
  the repo worktree.
- Keep explicit CLI overrides available for diagnostics; only defaults must be
  repo-local.
