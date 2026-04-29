# 2026-04-29 Still Object/Rope Frame 0 IR Panel

## Goal

Generate a static IR left/right panel matching the existing static-round IR board style for:

- still object rounds 1-4
- still rope rounds 1-2
- aligned frame 0
- cameras 0, 1, and 2
- IR left and IR right streams

## Scope

- Use existing aligned cases under `data/still_object/ffs203048_iter4_trt_level5/` and `data/still_rope/ffs203048_iter4_trt_level5/`.
- Render raw uint8 grayscale IR frames with no histogram equalization.
- Write the panel and machine-readable summary under `data/experiments/`.
- Do not change formal recording/alignment code.

## Validation

- Confirm every requested case has frame `0.png` for `ir_left` and `ir_right` across cameras 0-2.
- Confirm the output image is readable and has the expected six-row by six-column layout.
- Run deterministic harness checks before finishing.

## Result

- Output root: `data/experiments/still_object_round1_4_still_rope_round1_2_frame0_ir_panel`
- Panel: `still_object_round1_4_still_rope_round1_2_frame0_ir_left_right_panel.png`
- Summary: `summary.json`
- Panel dimensions: `2046x1208`

## Checks

- Output readability check: passed.
- `python scripts/harness/check_all.py`: passed.
