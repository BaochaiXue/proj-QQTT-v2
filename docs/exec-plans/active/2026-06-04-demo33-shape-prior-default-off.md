# Demo 3.3 Shape-Prior Default Off

## Goal

Make Demo 3.3 stop running the FuturePhysTwin/SAM3D shape-prior warmup by
default, because the generated gray canonical reference layer is not needed in
the live demo path.

## Scope

- Keep the existing shape-prior route available behind explicit
  `--shape-prior-warmup`.
- Change the default contract/profile state to disabled.
- Update Demo 3.3 docs and focused contract tests.

## Validation

- Run the focused Demo 3.3 contract tests.
- Run the Demo 3.3 dry-run in `demo_3_3_max`.
