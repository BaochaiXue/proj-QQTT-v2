# Demo 3.1 Demo 2.3 Diagnostics Sync

## Goal

Sync useful Demo 2.3 diagnostics into Demo 3 / Demo 3.1 without changing Demo
3.1's RealSense-depth + CoTracker process architecture.

## Scope

- Expose and forward shared fusion mismatch debug flags from Demo 3 and Demo
  3.1 into the shared three-view runtime.
- Expose and forward shared GPU sampling flags so Demo 3.1 rendered profiles
  can report both GPUs.
- Let Demo 3 / 3.1 override the shared runtime display/profile label so the
  Open3D window and shared profile do not say Demo 2.1.5.
- Pull shared GPU sampling summaries into Demo 3.1 wrapper summary fields.

## Validation

- Compile Demo 3 / Demo 3.1 runtime modules.
- Run focused Demo 3 / Demo 3.1 contract tests.
- Run Demo 3 / Demo 3.1 dry-runs.
- Run quick deterministic harness.

## Outcome

- Demo 3 and Demo 3.1 now expose and forward the shared fusion mismatch debug
  flags.
- Demo 3 and Demo 3.1 now expose and forward GPU sampling flags.
- Demo 3.1 infers `--gpu-sampling-device-indexes` from its physical mask and
  CoTracker GPUs when the user omits explicit indexes.
- Demo 3.1 wrapper summaries now copy per-device GPU utilization and memory
  metrics from the shared runtime profile.
- Shared runtime windows/profiles can be labeled as Demo 3 or Demo 3.1 through
  forwarded display/version overrides.
- Focused tests, dry-runs, and the quick deterministic harness passed.
