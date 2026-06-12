# Demo 3.3 Shape-Prior Default On

## Goal

Keep Demo 3.3 running the FuturePhysTwin/SAM3D shape-prior warmup by default.

## Scope

- Set the Demo 3.3 `--shape-prior-warmup` default to enabled.
- Keep `--no-shape-prior-warmup` available for explicit opt-out.
- Update docs and contract tests so the default state is `pending` /
  `async_background_thread`.

## Validation

- Demo 3.3 dry-run reports `shape_prior_warmup_enabled = true`.
- Focused Demo 3.3 contract test passes.
