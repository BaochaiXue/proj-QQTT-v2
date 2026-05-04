# Demo V2 Live SAM3.1 First-Frame Init

## Goal

Make Demo 2.0 initialize masks from the live first frame by default. The
runtime must not rely on saved masks for the normal live path because saved
masks can be out of sync with the current camera pose, object pose, or hand
pose and can poison EdgeTAM tracking from frame 0.

## Required Contract

- Default `--init-mode` is `sam31-first-frame`.
- Default `--track-mode` is `controller-object`; an explicit `--track-mode
  object-only` startup path must skip controller prompt/mask requirements for
  scenes where the operator hand is not currently visible.
- The demo captures frame 0 from the live D455 stream, runs SAM3.1 once on that
  frame, unions controller prompt masks, loads the object prompt mask, and then
  initializes the HF EdgeTAM streaming session.
- `saved-masks` remains available only as an explicit debugging fallback.
- FFS depth remains the default depth source.
- HF EdgeTAM remains compiled-only via `--compile-mode vision-reduce-overhead`.

## Validation

- Focused CLI/default tests.
- `python scripts/harness/check_all.py`.
- Short live smoke with `--init-mode sam31-first-frame` if the current scene has
  the prompted controller/object visible.
