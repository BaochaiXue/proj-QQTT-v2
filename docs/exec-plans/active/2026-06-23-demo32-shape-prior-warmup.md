# Demo 3.2 Shape Prior Warmup

## Goal

Enable Demo 3.2 to start a default-on, fail-soft SAM3D shape-prior warmup from
the first valid same-seq strict RGB-D/object-mask pair, without changing
EdgeTAM, TAPNext++, query identity, current observation PCD, or strict tracking
products.

## Approach

- Add Demo 3.2-only shape-prior CLI and contract fields in the shared Demo 3.x
  wrapper.
- Add a small shape-prior module for snapshot validation, async submission,
  remote-worker protocol, profile fields, and render packet attachment.
- Add a repo-side remote worker service that can load external SAM3D code and
  weights, with a lightweight fake/test path that does not import SAM3D.
- Add single-view canonical-to-observation alignment and validation; do not
  call the old FuturePhysTwin three-camera `align.py`.
- Attach successful results as a gray render/reference layer only. Record
  status and timing in profiles/manifests. Worker failures remain non-fatal.

## Tasks

1. Add focused failing tests for Demo 3.2 CLI defaults, opt-out behavior,
   contract/profile fields, and non-Demo-3.2 isolation.
2. Add failing tests for shape-prior protocol roundtrip, fake worker response,
   and import boundaries that keep SAM3D heavy dependencies out of demo runtime.
3. Add failing tests for single-view alignment on synthetic data and invalid
   validation thresholds.
4. Add failing tests for snapshot gating and async fail-soft manager behavior.
5. Implement the minimal modules and wire runtime render/profile metadata.
6. Update docs, scope carveout, and scope guard rules.
7. Run focused unit tests, then smoke validation with
   `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`.

## Non-goals

- Do not vendor SAM3D/FuturePhysTwin weights or external repositories.
- Do not make shape prior part of formal recording/alignment data products.
- Do not modify tracker/mask/query/PCD filtering decisions based on SAM3D.
