# D455 Single-Camera Probe Validation

- Serial used: `239222303506`
- Camera model: `Intel RealSense D455`
- Product line: `D400`
- Probe profile used: `848x480@30`
- Capture mode achieved: `IR-left + IR-right + color`
- Emitter actual state: `1.0`

## Raw Probe Command

```text
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe scripts\harness\probe_d455_ir_pair.py --serial 239222303506 --out_dir data\ffs_proof_of_life\d455_single\239222303506\raw
```

## Raw Probe Outputs

- `data/ffs_proof_of_life/d455_single/239222303506/raw/ir_left.png`
- `data/ffs_proof_of_life/d455_single/239222303506/raw/ir_right.png`
- `data/ffs_proof_of_life/d455_single/239222303506/raw/color.png`
- `data/ffs_proof_of_life/d455_single/239222303506/raw/metadata.json`
- `data/ffs_proof_of_life/d455_single/239222303506/raw/K_ir_left.txt`
- `data/ffs_proof_of_life/d455_single/239222303506/raw/K_color.txt`
- `data/ffs_proof_of_life/d455_single/239222303506/raw/preview_contact_sheet.png`

## Runtime Geometry Snapshot

- `K_ir_left[0,0] = 430.59002685546875`
- `K_color[0,0] = 427.0147399902344`
- `ir_baseline_m = 0.09504302591085434`
- `depth_scale_m_per_unit = 0.0010000000474974513`

## FFS Saved-Pair Command

```text
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe scripts\harness\run_ffs_on_saved_pair.py --sample_dir data\ffs_proof_of_life\d455_single\239222303506\raw --ffs_repo C:\Users\zhang\external\Fast-FoundationStereo --model_path C:\Users\zhang\external\Fast-FoundationStereo\weights\23-36-37\model_best_bp2_serialize.pth --out_dir data\ffs_proof_of_life\d455_single\239222303506\ffs
```

## FFS Outputs

- `data/ffs_proof_of_life/d455_single/239222303506/ffs/disparity_raw.npy`
- `data/ffs_proof_of_life/d455_single/239222303506/ffs/disp_vis.png`
- `data/ffs_proof_of_life/d455_single/239222303506/ffs/depth_ir_left_float_m.npy`
- `data/ffs_proof_of_life/d455_single/239222303506/ffs/cloud.ply`
- `data/ffs_proof_of_life/d455_single/239222303506/ffs/intrinsics_ffs.txt`

## Reprojection Command

```text
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe scripts\harness\reproject_ffs_to_color.py --sample_dir data\ffs_proof_of_life\d455_single\239222303506\raw --ffs_out_dir data\ffs_proof_of_life\d455_single\239222303506\ffs --out_dir data\ffs_proof_of_life\d455_single\239222303506\converted
```

## Converted Outputs

- `data/ffs_proof_of_life/d455_single/239222303506/converted/depth_ir_left_float_m.npy`
- `data/ffs_proof_of_life/d455_single/239222303506/converted/depth_color_aligned_float_m.npy`
- `data/ffs_proof_of_life/d455_single/239222303506/converted/depth_color_aligned_u16.npy`
- `data/ffs_proof_of_life/d455_single/239222303506/converted/conversion_metadata.json`
- `data/ffs_proof_of_life/d455_single/239222303506/converted/quicklook_depth.png`

## Depth Artifact Summary

- `depth_ir_left_float_m.npy`: shape `(480, 848)`, valid pixels `407040`, range `[0.4190, 1.8869]` meters
- `depth_color_aligned_float_m.npy`: shape `(480, 848)`, valid pixels `367095`, range `[0.4213, 1.8587]` meters
- `depth_color_aligned_u16.npy`: shape `(480, 848)`, valid pixels `367095`, range `[421, 1859]` encoded units

## Geometry / Scale Contract

- Stereo source is one D455 internal pair only.
- FFS output first lives in IR-left coordinates.
- Color-aligned output is produced only by explicit IR-left to color reprojection.
- Runtime intrinsics and runtime extrinsics are authoritative.
- The nominal D455 95 mm baseline is documentation only; runtime math uses the SDK-derived baseline.
- Replacement-compatible depth encoding preserves invalid=`0`.
