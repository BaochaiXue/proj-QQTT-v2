# Remove Render Stride From Live Rendered FPS Demos

## Goal

Remove `render_every_n` / `--render-every-n` from Demo 2.2, Demo 2.3, Demo 3, and Demo 3.1 live rendered profiling paths. Rendered FPS should be based on every render-ready packet/result, not an arbitrary group-id stride.

## Scope

- Demo 2.2 and Demo 2.3 wrapper CLIs.
- Shared three-view semantic PCD runtime.
- Demo 3 / Demo 3.1 forwarding and contract docs.
- Legacy single-camera EdgeTAM PCD runtime, so the name is not left behind as an attractive footgun.
- Deterministic smoke tests that asserted the old option existed.

## Validation

- Help/dry-run should no longer expose `--render-every-n`.
- Passing `--render-every-n` to Demo 2.2 / Demo 2.3 / Demo 3.1 should fail as an unknown argument.
- Shared runtime render packet publication should publish every packet.
