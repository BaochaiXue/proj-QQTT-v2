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

- Focused tests passed: `89 tests`, `OK`.
- Demo 4 fake-live launched with repo-local relative defaults.
- GPU validation could not publish a chunk because this session has no CUDA
  device access (`torch.cuda.is_available() == False`, `/dev/nvidia*` absent).
- CPU fallback confirmed EdgeTAM loads from `vendor/demo_runtime/EdgeTAM-hf`
  and then stops because upstream SAM 3.1 requires CUDA.

## Execution Notes

- Do not use symlinks for copied runtime assets.
- Do not copy external `.git` directories.
- Large weight files may remain ignored by extension patterns, but must exist in
  the repo worktree.
- Keep explicit CLI overrides available for diagnostics; only defaults must be
  repo-local.
