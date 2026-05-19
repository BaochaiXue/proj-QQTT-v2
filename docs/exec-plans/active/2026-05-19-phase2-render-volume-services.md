# Phase 2 Render Volume Services

## Goal

Continue the Demo 2.3 / Demo 3 / Demo 3.1 service migration by optimizing
render object filtering and semantic fusion hot paths without changing tracking
or PhysTwin semantics.

## Constraints

- Keep CoTracker semantics unchanged: object/controller union, up to 5000
  queries per camera, and torch randperm seeded by camera.
- Keep object render PCD controlled by FuturePhysTwin-style world-space voxel
  sampling, with 5mm exact mode clearly distinguished from adaptive mode.
- Keep render independent from slow filter/fusion work: latest-wins volume
  output, cheap fallback when volume output is missing, and no renderer wait.
- Do not enable debug PLY/mask overlay work on FPS hot paths by default.

## Plan

- Add a fast exact volume index sampler that computes voxel representatives
  once and gathers points/colors only at the end.
- Extend `ObjectVolumeFilterService` profiling with key/unique/gather timing,
  exactness flags, and latest-wins worker support.
- Implement a production `SemanticFusionService` fast path with cached
  normalized intrinsics grids, flat valid indices, and direct `R,t` transforms.
- Add deterministic tests for exact volume semantics, async latest-wins filter
  behavior, and synthetic semantic fusion correctness.
- Update docs/profile schema for the new service fields.

## Validation

- Focused unittest modules for volume filter and semantic fusion services.
- Touched-module `py_compile`.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`.

## Result

- Implemented fast exact FuturePhysTwin-style volume indices and substage
  timing profile fields.
- Added asynchronous latest-wins object volume filter worker support.
- Implemented `SemanticFusionService` service-fast RGB-D backprojection with
  cached intrinsics grids and offline quality comparison helper.
- Added deterministic unit coverage and included the new service tests in the
  quick harness profile.
