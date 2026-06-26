# Demo v5 Query Marker Style Plan

## Goal

Make the Demo v5 online viewer and rendered diagnostic videos support both RGB overlay markers and `data_process_sam3d` final-data style rendering through explicit parameters.

## Scope

- Change only Demo v5 visualization/video rendering.
- Keep chunk data products, controller anchor recovery, realtime publishing, and formal recording/alignment products unchanged.
- Preserve existing CLI compatibility while adding mode selection.

## Implementation Notes

- Add `--render-mode rgb-overlay|sam3d-final-data` to `demo_v5/online_points_viewer.py`.
- Add `--output-video PATH` to render existing online chunks to MP4 without opening a live window.
- `rgb-overlay` mode mirrors `data_process_sam3d/utils/visualizer.py::draw_circle`: projected 2D markers are simple filled circles with no black outline, no white rim, and no query-id color switching.
- `sam3d-final-data` mode mirrors `data_process_sam3d/data_process_sample.py::visualize_track`: object points use `plt.cm.rainbow` colors derived from first-frame object point world-y, while controller anchors use solid red Open3D sphere meshes.
- Add `--point-viewer-render-mode` to `demo_v5/realtime_futurephystwin_chunks.py` so live point-viewer windows can choose either mode.

## Validation

- Add focused unit coverage for SAM3D-style object/controller marker colors.
- Add coverage for offline `--output-video` rendering in RGB overlay mode.
- Run the Demo v5 unit tests and smoke validation.
- Re-render latest full chunk-data videos in both modes to `/home/xinjie/下载`.
