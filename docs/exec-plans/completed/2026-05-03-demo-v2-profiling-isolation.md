# Demo V2 Profiling Isolation

## Goal

Add profiling-only switches to explain why live Demo 2 EdgeTAM wall time is
slower than the offline HF EdgeTAM streaming benchmark.

## Contract

- Keep the existing full live demo defaults intact.
- Add isolation modes without importing offline GIF/panel code into the live
  hot path.
- Support headless profiling for capture-only, EdgeTAM-only, FFS-only,
  EdgeTAM+FFS, and full masked PCD runs.
- Record EdgeTAM wall timing separately from optional CUDA-event timing.
- Avoid device-wide `torch.cuda.synchronize()` in the default live hot path;
  make synchronized timing explicit through a profiling flag.
- Keep deterministic tests updated and run `python scripts/harness/check_all.py`.

## Validation

- CLI/default smoke tests for new switches.
- Focused unit tests for object id mapping and timing helpers.
- `python scripts/harness/check_all.py`.
