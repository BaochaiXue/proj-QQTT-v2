# Demo 3.2 New Fake-Live Case

## Goal

Make Demo 3.2 default fake-live replay use the newly recorded 3-minute
`both_eval` case:

`data_collect/sloth_both_eval_3min_e70_g60_20260621_202627`

Keep all other Demo 3.2 defaults unchanged, including `replay_fps=5`, FFS,
tracking, filters, table-Z behavior, and repo-root `table_calibrate.pkl`.

## Steps

1. Update tests so Demo 3.2 default fake-live contract/delegate point at the
   new case while explicit `--fake-live-case` still overrides.
2. Add a version-aware fake-live default case helper in the Demo 3.x wrapper.
3. Update the lower-level masked PCD fallback/help text to the new case.
4. Update Demo 3.2 docs to describe the new default case and calibration files.
5. Run the focused runtime tests and Demo 3.2 dry-run validation.
