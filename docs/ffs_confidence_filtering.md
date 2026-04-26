# FFS Confidence Filtering

Confidence filtering is an optional Fast-FoundationStereo alignment mode for
testing whether logits-derived confidence can suppress floating point-cloud
artifacts and bad depth pixels.

## Modes

The PyTorch logits path currently supports:

- `margin`: top-1 probability minus top-2 probability
- `max_softmax`: top-1 probability
- `entropy`: one minus normalized softmax entropy
- `variance`: robust-normalized inverse soft-argmax disparity variance

Larger values mean higher confidence for every mode.

## Output Contract

Confidence filtering is disabled by default. When enabled, FFS depth is still
written as uint16 depth encoded by `depth_scale_m_per_unit`. For the usual
`0.001` scale, this is millimeter depth.

Invalid pixels, low-confidence pixels, out-of-range pixels, and pixels outside
an optional object mask are encoded as `0`.

Canonical/formal depth output remains:

```text
depth/<camera>/<frame>.npy      # ffs backend
depth_ffs/<camera>/<frame>.npy  # both backend
```

Those files contain uint16 depth only. Float-meter depth is not written unless
the existing explicit `--write_ffs_float_m` flag is used. Confidence debug maps
and valid-mask debug maps are also opt-in:

```bash
--write_ffs_confidence_debug
--write_ffs_valid_mask_debug
```

Debug confidence maps are uint8 `0..255`; valid-mask debug PNGs are uint8 with
kept pixels as `255` and rejected pixels as `0`.

## Alignment Usage

```bash
python data_process/record_data_align.py \
  --case_name <case_name> \
  --start 0 \
  --end 120 \
  --depth_backend ffs \
  --ffs_repo /home/zhangxinjie/Fast-FoundationStereo \
  --ffs_model_path /home/zhangxinjie/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth \
  --ffs_scale 1.0 \
  --ffs_valid_iters 4 \
  --ffs_confidence_mode max_softmax \
  --ffs_confidence_threshold 0.6 \
  --ffs_confidence_depth_min_m 0.2 \
  --ffs_confidence_depth_max_m 1.5
```

Filtering happens after FFS depth and the selected confidence map are projected
from IR-left into color coordinates. The confidence projection uses the same
nearest-depth winner rule as depth projection so confidence and depth stay
spatially paired.

## Sweep

```bash
python scripts/harness/run_ffs_confidence_filter_sweep.py \
  --case_name <case_name> \
  --start 0 \
  --end 120 \
  --ffs_repo /home/zhangxinjie/Fast-FoundationStereo \
  --ffs_model_path /home/zhangxinjie/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth \
  --modes margin,max_softmax,entropy,variance \
  --thresholds 0.3,0.4,0.5,0.6,0.7,0.8 \
  --depth_min_m 0.2 \
  --depth_max_m 1.5 \
  --output_root ./data/experiments/ffs_confidence_filter_sweep
```

The sweep writes `results.csv`, `summary.json`, and one experiment directory per
mode/threshold. Summary selection is intentionally conservative:

- keep `valid_ratio_after_confidence >= min_valid_ratio` when possible
- avoid excessive low-confidence rejection
- prefer lower hole ratio
- use outlier metrics only when a future run explicitly adds them

Higher thresholds are not automatically better. They can remove more
low-confidence floating artifacts but also create more holes and reduce object
point count.

## TensorRT Note

Current confidence filtering uses PyTorch classifier logits. TensorRT confidence
would require exporting logits or confidence outputs from the TensorRT graph.
That export is intentionally not part of this first filtering experiment.
