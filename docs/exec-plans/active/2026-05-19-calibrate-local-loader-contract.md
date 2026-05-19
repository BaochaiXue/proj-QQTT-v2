# Calibrate Local Loader Contract

## Goal

Verify and harden local `calibrate.pkl` reads after aligned exports began
normalizing `calibrate.pkl` into case camera order. A metadata-aware QQTT loader
must not reorder an already case-ordered aligned calibration a second time.

## Plan

- Keep source/root `calibrate.pkl` metadata semantics unchanged: reference serials describe
  the calibration file order.
- For aligned cases, make `metadata_ext.json["calibration_reference_serials"]` describe
  the emitted aligned `calibrate.pkl` order, which is the case `serial_numbers` order.
- Preserve the original source calibration order in a separate traceability field.
- Add a smoke test that reads an aligned case with the shared local loader and verifies
  `c2ws[camera_idx]` matches the case camera order.

## Validation

- `python -m py_compile data_process/record_data_align.py data_process/aligned_case_metadata.py tests/test_record_data_align_smoke.py`
- `python -m unittest -v tests.test_record_data_align_smoke tests.test_aligned_metadata_loader_smoke`
- `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_record_data_align_smoke tests.test_aligned_metadata_loader_smoke`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
