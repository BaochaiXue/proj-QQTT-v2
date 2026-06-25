# Demo 4 Repo-Local Runtime Design

## Approval Context

The user requested an approved-by-default implementation: do not stop for
design approval, do not ask for path-layout approval, and assume the
recommended design is accepted. This spec records the decision before code
changes, but it is not a blocking gate.

## Goal

Demo 4 and the Demo 3.2/shape-prior/FFS/TAPNext++ runtime path it launches must
default to repo-local relative assets only. Referenced external modules,
weights, and runtime resources are copied into this repository worktree and the
runtime defaults are updated to use those copies. Final validation runs only
Demo 4 fake-live.

## Recommended Design

Use `vendor/demo_runtime/` as the repo-local runtime asset root. It is inside
the repository worktree and is ignored as a local runtime payload so tens of GB
of third-party source trees and weights are not committed to Git history. The
files must exist in the worktree and runtime defaults must point at them.

Repo-local layout:

```text
vendor/demo_runtime/
  sam-3d-objects/
  FuturePhysTwin/
  Fast-FoundationStereo/
  tapnet/
  EdgeTAM-hf/
  checkpoints/tapnextpp/tapnextpp_ckpt.pt
```

Demo 4 output defaults move from the external FuturePhysTwin checkout to:

```text
result/demo_v4/futurephystwin_chunks
```

This keeps generated chunk output inside the repo worktree while preserving the
existing ignore policy for generated results.

## Runtime Defaults

The following defaults must be repo-local:

- `services/shape_prior_remote/server.py`
  - SAM3D root: `vendor/demo_runtime/sam-3d-objects`
  - FuturePhysTwin root: `vendor/demo_runtime/FuturePhysTwin`
- `demo_v4/realtime_futurephystwin_chunks.py`
  - FuturePhysTwin chunk output root: `result/demo_v4/futurephystwin_chunks`
- `data_process/depth_backends/ffs_defaults.py`
  - FFS repo: `vendor/demo_runtime/Fast-FoundationStereo`
  - FFS model path under that repo
  - FFS subprocess Python default: relative `python`, not a user-home absolute
- `qqtt/demo/realtime_masked_edgetam_pcd.py`
  - EdgeTAM HF model: `vendor/demo_runtime/EdgeTAM-hf`
  - TAPNext++ repo: `vendor/demo_runtime/tapnet`
  - TAPNext++ checkpoint: `vendor/demo_runtime/checkpoints/tapnextpp/tapnextpp_ckpt.pt`
  - no parent-repo fallback

Explicit CLI overrides remain available for debugging, but the default path
contract is repo-local and relative to this repository.

## Copy Policy

Copy required external working trees and weights into `vendor/demo_runtime/`.
Exclude `.git`, Python caches, local logs, and generated output directories.
Do not replace the external repositories with symlinks; the runtime must use
real files inside this repo worktree.

## Scope Boundaries

This change does not rewrite Demo 4 chunk semantics, tracker semantics,
shape-prior alignment, optimization, or fake-live cadence. It only removes
repo-external default paths and gives Demo 4 a repo-local dependency layout.

Docs and tests must be updated where they assert the old absolute defaults.

## Validation

Run focused parser/path tests during implementation. Final live validation is
limited to Demo 4 fake-live, as requested by the user.

The final report must include:

- copied asset roots and whether each exists;
- changed runtime defaults;
- Demo 4 fake-live command and result.

## Self-Review

- No placeholder approvals remain: the layout and defaults are explicit.
- The design does not vendor external `.git` directories.
- The design keeps generated output in ignored `result/` while still inside the
  repo worktree.
- The design preserves explicit CLI overrides but removes repo-external
  defaults and fallbacks.
