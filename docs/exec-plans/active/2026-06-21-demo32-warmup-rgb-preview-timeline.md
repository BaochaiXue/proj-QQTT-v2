# Demo 3.2 Warmup RGB Preview Timeline

## Goal

Let fake-live side-by-side panel RGB input keep advancing during warmup while PCD/tracking still wait for the first strict same-seq pair.

## Scope

- Add a preview-only RGB path for Demo 3.2 fake-live panel/headless input timelines.
- Keep official processing frames on the existing `capture_slot` and lossless queues.
- Keep warmup preview frames out of segmentation, PCD, tracking, and strict same-seq pairing.
- Preserve existing offline side-by-side latest-input-RGB selection policy.

## Steps

1. Add tests proving warmup publishes preview frames while official processed frames remain held at seq 0.
2. Add tests proving live panel RGB reads preview frames before falling back to official capture frames.
3. Add tests proving headless input timeline records preview frames without overwriting processed artifacts.
4. Add a color-only recording read path and preview slot.
5. Publish preview frames during fake-live warmup at `--replay-fps`.
6. Resume official processing from the elapsed fake-live source index after the first strict pair.
7. Update Demo 3.2 docs and run focused/smoke validation.
