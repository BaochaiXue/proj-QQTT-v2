# Demo v6.2 Shape-prior warm-up timing analysis

## Requirement

Problem:
The existing `shape_prior_profile.json` records only the total wall time of
upscale, SAM3.1 segmentation, SAM3D generation, alignment, and sampling. It
cannot distinguish prewarm readiness, model loading, inference, rendering,
matching, ARAP, sampling, and output I/O, so the next optimization target is
not observable.

Required final behavior:
Every successful Demo v6.2 shape-prior warm-up writes one detailed timing
analysis into the existing profile. The analysis must show the sequential
critical path, per-stage substeps, stage shares and ranking, the bottleneck,
and any unattributed wall time. Prewarmed subprocesses must report whether
their initialization finished before frame 0 submitted the stage.

Inputs:
The existing frame-0 request, prewarmed stage workers, SAM3.1 runtime, and the
upscale, SAM3D generate, align, and sample subprocesses.

Outputs:
`capture/shape_prior_profile.json` receives a versioned
`shape_prior_timing` object. Each subprocess also writes its stage profile
under `<shape-prior-case>/shape/timing/` so its internal measurements remain
inspectable independently.

State changes:
Instrumentation and diagnostic sidecars only. Shape generation, alignment,
sample counts, masks, point products, and formal chunk behavior stay
unchanged.

Invalid cases:
Missing, malformed, negative, non-finite, or wrong-stage timing data fails at
the profile boundary instead of producing misleading analysis.

Constraints:
Use `time.perf_counter()` wall durations, keep Python lines readable, preserve
the current single-camera branch and unrelated dirty changes, and do not add a
second warm-up path or profiling mode.

Unknowns:
No correctness-blocking unknowns. A hardware run is still required to produce
new real measurements after instrumentation.

## Plan

- [ ] Define and test the versioned stage/critical-path timing schema.
- [ ] Instrument upscale, SAM3.1 export, SAM3D generation, alignment, and
  sampling at optimization-relevant boundaries.
- [ ] Aggregate subprocess profiles, frame-0 case preparation, and final
  result I/O into `shape_prior_profile.json` with a ranked bottleneck.
- [ ] Document how to read the profile and update focused tests.
- [ ] Run focused validation and the repository smoke profile.

## Validation

Branch/setup:

- Current branch: `single-camera`.
- `HEAD` equals `origin/single-camera` (`0` ahead, `0` behind).
- The required `git pull --ff-only origin main` was attempted and refused
  because `single-camera` and `origin/main` have diverged; no merge or rebase
  was performed.

