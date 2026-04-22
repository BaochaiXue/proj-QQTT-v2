# 2026-04-22 Upgrade SAM3 Package To SAM3.1-Compatible Code

## Goal

Replace the incompatible `sam3==0.1.3` package in `qqtt-ffs-compat` with official `facebookresearch/sam3` model code that can load `sam3.1_multiplex.pt`.

## Scope

- inspect the current environment and official repo install metadata
- upgrade the environment from PyPI `sam3` to official repo code
- verify `scripts/harness/sam31_mask_helper.py` can initialize the predictor with the downloaded local `sam3.1_multiplex.pt`
- record exact commands and outcomes under `docs/generated/`

## Non-Goals

- no repo-local installer changes
- no attempt to redesign the SAM helper API
- no dependency pin cleanup beyond what is required to get `sam3.1` loading working

## Validation Plan

- verify installed `sam3` package origin/version after replacement
- run a minimal `build_sam31_video_predictor(...)` smoke in `qqtt-ffs-compat`
- write a generated validation note with the final install command and outcome
