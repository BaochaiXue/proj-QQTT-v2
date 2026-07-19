# Demo v6.2 Shape-Prior Canonical Mesh Cache

## Requirement

Problem:
Demo v6.2 regenerates the same SAM3D canonical `object.glb` on every run even
when the operator is using the same physical object asset.

Required final behavior:

- `shape_prior.object: null` always performs on-the-fly generation and never
  reads or writes the persistent cache.
- A non-null, versioned physical-object identity performs a validated cache
  lookup before stage prewarm. A hit materializes `object.glb` into the current
  case and skips upscale, the second SAM3.1 segmentation, and SAM3D generate.
- A miss runs the existing generation path, validates and atomically publishes
  `object.glb`, then continues through the unchanged align and sample stages.
- `shape_prior.object_prompt` is the single prompt source for the current
  frame-0 observation and cache-miss reconstruction. It is not a cache key.

Inputs:
The YAML object identity, object prompt, external cache root, and the current
frame-0 RGB-D observation.

Outputs:
A run-local `shape/object.glb` with the same downstream layout as before, plus
an immutable schema-v1 cache entry when a cache miss publishes successfully.

State changes:
The persistent cache lives outside the run output root. Cache entries contain
only `object.glb` and its manifest; alignment, sampling, ASAP, and PhysTwin
products remain per-run outputs.

Invalid cases:
Invalid object identities, cache roots under the run output, corrupt entries,
hash mismatches, invalid GLBs, publish conflicts, and materialization failures
fail explicitly. Cache failures never trigger automatic regeneration.

Constraints:
Do not modify alignment, metric scale, symmetry handling, PnP, ARAP, sampling,
ASAP, or PhysTwin contracts. Preserve the seven existing timing stage names.

Unknowns:
No correctness-blocking unknowns. Real SAM3D performance validation remains a
separate GPU run after deterministic validation.

## Review findings to correct

- Pass the configured object prompt even when shape-prior warmup is disabled;
  frame-0 SAM3.1 still needs it for object tracking.
- Release the initial SAM3.1 runtime on a cache hit because the second
  segmentation will not run.
- Validate and hash generated GLBs even when the cache is disabled.
- Attribute cache publication to the generate timing stage instead of leaving
  an unaccounted critical-path gap.
- Validate cache roots against the real run output root rather than a possibly
  overridden capture directory.
- Validate configuration in the orchestrator before output cleanup and again
  at the camera-process boundary.
- Validate the complete manifest contract and keep prewarm stage selection
  explicit and fail-fast.

## Plan

- [x] Correct configuration defaults, validation, and subprocess propagation.
- [x] Harden cache identity, manifest, publication, and materialization rules.
- [x] Correct cache-aware prewarm and SAM3.1 resource lifetime.
- [x] Keep all canonical-mesh preparation inside the existing generate timing
      stage and preserve downstream stages.
- [x] Add request-path, failure-semantics, resource-lifetime, and plumbing
      tests; register them in smoke.
- [x] Update the Demo v6.2 pipeline documentation.
- [x] Run focused tests, scoped static checks, and the repository smoke profile.

## Validation

- Focused shape-prior cache and Demo v6.2 runtime tests.
- CLI/config dry-run checks for cache disabled, miss, hit, and invalid values.
- Scoped compile, Ruff, and `git diff --check`.
- `conda run -n demo_2_max --no-capture-output python
  scripts/harness/validation/run.py --profile smoke`.
- No hardware or SAM3D inference will be claimed unless explicitly run.

## Result

- Focused cache/runtime suite: 60 tests passed.
- Repository smoke profile: 61 tests plus help/guard checks passed.
- Scoped Ruff, Python compilation, and `git diff --check`: passed.
- Real SAM3D generation, real cache-hit latency, and camera/GPU behavior were
  not run as part of this deterministic review.
