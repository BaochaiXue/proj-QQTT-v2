# 2026-04-29 Still Object/Rope Frame 0 RGB Panel

## Goal

Generate a static RGB panel for:

- still object rounds 1-4
- still rope rounds 1-2
- aligned frame 0
- cameras 0, 1, and 2

## Scope

- Use existing aligned cases under `data/still_object/ffs203048_iter4_trt_level5/` and `data/still_rope/ffs203048_iter4_trt_level5/`.
- Render one RGB column per camera.
- Write the panel and machine-readable summary under `data/experiments/`.
- Do not change formal recording/alignment code.

## Validation

- Confirm every requested case has frame `0.png` for `color` across cameras 0-2.
- Confirm the output image is readable and has the expected six-row by three-column layout.
- Run deterministic harness checks before finishing.
