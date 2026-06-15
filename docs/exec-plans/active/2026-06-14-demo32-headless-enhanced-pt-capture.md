# Demo 3.2 Headless Enhanced-PT Capture

## Goal

Allow Demo 3.2 fake-live replay to run the full realtime FFS, EdgeTAM,
TAPNext++, and masked PCD pipeline without opening Open3D, while saving only
enhanced-pt filtered point clouds, color-aligned FFS depth, and query-point
artifacts for later offline video rendering.

## Implementation Notes

- Add a `--headless-capture-dir` CLI surface to the single Demo 3.x runtime and
  the shared masked PCD delegate.
- Treat Demo 3.2/3.3 `fake-live + render-mode none` as headless capture instead
  of disabling tracking and PCD.
- Force headless capture to `--enable-pcd-filter --pcd-filter-mode sync` with
  `enhanced-pt` for both object and controller, failing fast if the user
  explicitly asks for a conflicting filter.
- Save artifacts from actual completed PCD output frames rather than filling in
  every fake camera frame.
- Add an offline helper to render saved filtered PCD plus current TAPNext++
  lifted query points into an MP4. The offline overlay uses light-blue object
  query points and red controller query points without historical trajectory
  lines.

## Validation

- Add focused unit coverage for contract/defaults, validation failures, artifact
  writing, and offline rendering on synthetic data.
- Run the Demo 3.x runtime tests, TAPNext++ overlay tests, and `check_all.py`.
