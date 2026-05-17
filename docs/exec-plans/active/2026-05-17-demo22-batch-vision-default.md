# Demo 2.2 Batch Vision Default

## Goal

Promote the existing Demo 2.2 batch-vision EdgeTAM path from an explicit
experiment flag to the default async-filter hot path so the next real profile
measures batched vision encoder throughput instead of three serial full
EdgeTAM forwards.

## Scope

- Keep the existing single-owner scheduler and per-camera EdgeTAM video state.
- Enable `--edgetam-batch-vision-encoder` by default only for the Demo 2.2
  async-filter preset, whose contract already uses shared model topology.
- Add public `--edgetam-batch-vision` / `--no-edgetam-batch-vision` aliases
  while preserving the legacy `--experimental-edgetam-batch-vision` flag.
- Keep staged/replicated EdgeTAM modes rejecting batch vision.

## Validation

- Demo 2.2 smoke tests.
- `git diff --check`.
