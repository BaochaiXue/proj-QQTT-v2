# Aligned Calibrate Case Order

## Goal

Make every `data_process/record_data_align.py` aligned-case export write
`calibrate.pkl` in the output case's `metadata["serial_numbers"]` camera order,
not only formal `different_types` exports. The file format remains the same
PhysTwin-compatible `list[(4, 4)] camera_to_world_c2w`.

## Plan

- Update `write_aligned_calibration_file()` to always load and serial-map the source
  calibration before writing the aligned case `calibrate.pkl`.
- Keep `metadata["calibration_reference_serials"]` in aligned metadata for traceability.
- Update smoke coverage so ordinary aligned cases verify case-order calibration output.

## Validation

- `python -m py_compile data_process/record_data_align.py tests/test_record_data_align_smoke.py`
- `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_record_data_align_smoke`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
