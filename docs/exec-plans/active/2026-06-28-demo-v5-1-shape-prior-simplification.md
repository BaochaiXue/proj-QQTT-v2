# Demo v5.1 Shape-Prior Simplification

## Goal

Make the Demo v5.1 shape-prior path local, linear, and easier to read while
preserving the managed remote SAM3D worker behavior used by the current demo.

## Design

- Keep Demo v5.1 shape-prior code under two files:
  `demo_v5_1/shape_prior.py` and `demo_v5_1/shape_prior_worker.py`.
- Keep the main remote-worker path only: one async submit from the first
  frame-0-like RGB + object mask request with depth/K/c2w for alignment.
- Use one npz ZeroMQ frame for the warmup request/response, not an 8-frame
  protocol.
- Remove old compatibility surfaces, debug-only worker modes, and broad mesh
  fallback handling that obscures the SAM3D flow.
- Keep `qqtt/demo/shape_prior_warmup.py` and `services/shape_prior_remote/`
  for older demo code, but do not import them from Demo v5.1.

## Checklist

- [x] Add failing tests for local runtime/RPC, removed CLI flags, protocol
  frame counts, and sampling helper simplification.
- [x] Add Demo v5.1-local shape-prior runtime and RPC modules.
- [x] Point Demo v5.1 realtime code and worker at the local modules.
- [x] Remove non-main-path CLI flags and worker branches.
- [x] Simplify worker mesh handling and sampling helper fallbacks.
- [x] Drop request-level RGB-D normalization; the shape-prior model
  path treats input as RGB + object mask.
- [x] Collapse runtime/RPC/alignment/sampling helpers into the two-file
  shape-prior path and keep it under 1000 lines.
- [x] Update guards, smoke manifest, and run validation.

## Validation

- Red check: `tests.test_demo_v5_1_shape_prior_simplification` fails before
  implementation because local runtime/RPC modules and removed-flag behavior
  are not present yet.
- Focused unittest, worker `--help`, py_compile, and scope guard passed.
- `git diff --check` on the touched Demo v5.1 shape-prior files, tests,
  scope guard, and validation runner passed.
- Smoke profile passed.
