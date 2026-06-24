# Demo v4 Shape-Prior Observation Consistency

## Goal

Make Demo 3.2 / Demo v4 shape-prior alignment use the same quality gate as
FuturePhysTwin offline processing: raw semantic object masks remain the SAM3D
prompt/crop input, while the 3D alignment observation is built from a
depth-valid, PCD-filtered processed object mask.

## Root Cause

The remote shape-prior protocol currently carries one `object_mask`. Demo 3.2
fills it from the raw mask packet, and the worker reuses that same mask for
both SAM3D crop/prompt and `_object_observation_points_world`. FuturePhysTwin
offline `data_process_sam3d/align.py` uses raw mask PNGs for 2D crop/matching
but builds the 3D observation from `pcd/0.npz` plus
`mask/processed_masks.pkl`, where `processed_masks.pkl` applies depth/PCD
validity and radius outlier filtering.

## Implementation Steps

- [x] Extend the shape-prior snapshot/protocol with an optional
  `object_observation_mask`.
- [x] Keep backward compatibility for older seven-frame request payloads by
  falling back to `object_mask` when no observation mask is present.
- [x] Change worker observation lifting to use `object_observation_mask`, while
  SAM3D crop/upscale continues to use raw `object_mask`.
- [x] Build the snapshot observation mask in Demo 3.2 from the current PCD
  build/filter/table-Z output pixels, matching the FuturePhysTwin mask quality
  gate more closely than raw mask/depth lifting.
- [x] Add regression tests for protocol round-trip, worker observation
  priority, and Demo 3.2 snapshot mask separation.
- [x] Update Demo v4 docs to describe raw prompt mask vs processed observation
  mask.

## Validation Commands

```bash
conda run -n demo_2_max --no-capture-output \
  python -m pytest tests/test_demo32_shape_prior_warmup.py \
  tests/test_demo_v4_futurephystwin_chunks.py -q

conda run -n demo_2_max --no-capture-output \
  python scripts/harness/validation/run.py --profile smoke
```

## Acceptance Gates

- Protocol metadata reports both raw prompt mask pixels and observation mask
  pixels for new requests.
- Worker echo-observation excludes raw-mask pixels that were removed from the
  observation mask.
- SAM3D crop/upscale still uses the raw object mask.
- Demo 3.2 shape-prior snapshot carries a non-empty processed observation mask
  when the PCD residual/strict product path can produce one.
