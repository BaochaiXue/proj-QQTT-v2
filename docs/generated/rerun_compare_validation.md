# Rerun Compare Validation

## Environment

- conda env: `qqtt-ffs-compat`
- installed optional package:
  - `rerun-sdk==0.31.2`

Install command:

```text
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe -m pip install rerun-sdk
```

## Deterministic Checks

Passed in `qqtt-ffs-compat`:

```text
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe -m unittest -v tests.test_agents_scope_contract_smoke tests.test_ffs_remove_invisible_mask_smoke tests.test_rerun_compare_workflow_smoke
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe scripts/harness/visual_compare_rerun.py --help
```

## Real Data Validation

Single-frame sanity run:

```text
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe scripts/harness\visual_compare_rerun.py --aligned_root C:\Users\zhang\proj-QQTT\data --realsense_case native_30_static --ffs_case ffs_30_static --output_dir C:\Users\zhang\proj-QQTT\data\rerun_compare_sanity_native_30_static_vs_ffs_30_static --frame_start 0 --frame_end 0 --rerun_output rrd_only --max_points_per_camera 30000
```

Result:

- passed
- wrote `pointcloud_compare.rrd`
- wrote 3 fused full-scene PLYs for frame `0000`

Full 30-frame validation run:

```text
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe scripts/harness\visual_compare_rerun.py --aligned_root C:\Users\zhang\proj-QQTT\data --realsense_case native_30_static --ffs_case ffs_30_static --output_dir C:\Users\zhang\proj-QQTT\data\rerun_compare_native_30_static_vs_ffs_30_static --rerun_output rrd_only --max_points_per_camera 30000
```

Result:

- passed
- processed the full shared aligned range:
  - `native_frame_idx = 0..29`
  - `ffs_frame_idx = 0..29`
- wrote:
  - `data/rerun_compare_native_30_static_vs_ffs_30_static/pointcloud_compare.rrd`
  - `data/rerun_compare_native_30_static_vs_ffs_30_static/summary.json`
  - `data/rerun_compare_native_30_static_vs_ffs_30_static/ply_fullscene/*.ply`
- output counts:
  - `30` frames in `summary.json`
  - `90` fused PLY files in `ply_fullscene/`
- RRD size:
  - `126345262` bytes

Frame `0000` summary snapshot with `--max_points_per_camera 30000`:

- `native.fused_point_count = 90000`
- `ffs_remove_1.fused_point_count = 90000`
- `ffs_remove_0.fused_point_count = 90000`
- `ffs_remove_1.remove_invisible_pixel_count = 47320`
- `ffs_remove_0.remove_invisible_pixel_count = 0`

Notes:

- validation used `--rerun_output rrd_only` to keep the run non-interactive while still producing a replayable Rerun timeline
- validation used `--max_points_per_camera 30000` to bound PLY size on the real 30-frame case
- the default CLI behavior remains `--rerun_output viewer_and_rrd`
