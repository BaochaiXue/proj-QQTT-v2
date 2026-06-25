# External Dependencies

Demo runtime repos, checkpoints, SAM assets, and generated proof artifacts now
use repo-local defaults for this single-camera branch. This file records the
repo-local runtime locations and the external source locations used to refresh
those copies.

## Fast-FoundationStereo

- Default repo-local runtime copy:
  `vendor/demo_runtime/Fast-FoundationStereo`
- Typical checkpoint family: `20-30-48`
- Example checkpoint path:
  - `vendor/demo_runtime/Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth`
- Runtime roles:
  - `cameras_viewer_FFS.py`
  - `data_process/record_data_align.py --depth_backend ffs`
  - `scripts/harness/diagnostics/demo/realtime_single_camera_pointcloud.py --depth-source ffs`
  - native-vs-FFS visualization harnesses

## SAM 3.1

- Runtime role:
  - single-camera masked PCD demo initialization
  - mask helper / visualization diagnostics
- Demo runtime defaults should reference repo-local checkpoints. Prefer
  explicit CLI arguments only for diagnostics against alternate checkpoints.

## EdgeTAM

- Runtime role:
  - optional single-camera mask propagation in demo workflows
- Default repo-local HF snapshot copy:
  - `vendor/demo_runtime/EdgeTAM-hf`
- External cache/source snapshots may still be used to refresh this local copy
  and should be documented in validation notes.

## SAM3D Shape Prior Warmup

- Runtime role:
  - Demo 3.2 optional warmup diagnostic/reference layer
  - remote resident worker for `--shape-prior-execution remote-worker`
  - single-view alignment of SAM3D canonical geometry to the first valid
    same-seq RGB-D + EdgeTAM object mask snapshot
  - object-mask crop and x4 image upscaling before SAM3D inference, matching
    `data_process_sam3d/image_upscale.py`
  - SAM3D Objects single-view is the only supported shape-prior backend in this
    branch
- Default repo-local SAM3D Objects checkout copy:
  - `vendor/demo_runtime/sam-3d-objects`
- Default repo-local FuturePhysTwin checkout copy:
  - `vendor/demo_runtime/FuturePhysTwin`
- Worker entrypoint in this repo:

```bash
python services/shape_prior_remote/server.py \
  --bind tcp://0.0.0.0:7100 \
  --upscale-category "stuffed animal" \
  --preload-models \
  --warmup-models
```

- `--preload-models --warmup-models` makes the worker load the x4 upscaler and
  SAM3D model, then run one deterministic dummy upscaler + SAM3D +
  mesh-conversion request before binding the endpoint. Report cold worker
  startup timing separately from warm request latency.
- Demo 3.2 defaults to a 180000 ms shape-prior request timeout because each real
  request still runs x4 upscaling, SAM3D inference, single-view alignment, and
  data-process-compatible sampling.
- The request metadata includes `table_z_m` and `table_z_above_direction`; the
  worker uses these fields for single-view alignment validation. If older
  protocol/debug requests omit `table_z_above_direction`, the worker falls back
  to `negative` so Demo 3.2 stays compatible with Z-down table-world artifacts.

- Protocol/debug self-test mode that avoids importing SAM3D:

```bash
python services/shape_prior_remote/server.py \
  --bind tcp://0.0.0.0:7100 \
  --echo-observation
```

SAM3D, Stable Diffusion x4 upscaler, PyTorch3D, Kaolin, checkpoints, and model
weights are copied into `vendor/demo_runtime/` for the default Demo runtime
path. Generated route videos and run outputs stay in ignored output locations.
The Demo 3.2 process talks to the worker over the lightweight
`services/shape_prior_remote` protocol; it does not import SAM3D or upscaler
heavy dependencies.

## PhysTwin-Compatible Tracking Products

- Demo 3.2 `--tracking-product-backend phystwin-strict-tracking` targets the
  PhysTwin data contract and postprocessing semantics while keeping the local
  model stack.
- It uses TAPNext++ as the tracker backend, EdgeTAM as the mask backend, and
  RealSense/FFS as the depth backend.
- CoTracker is not a runtime dependency for this mode. Compatibility outputs
  may include a `cotracker/0.npz` path for scripts that expect the PhysTwin
  folder name; manifests mark that path as TAPNext++ data.

## TensorRT Engines

- Runtime role:
  - optional FFS acceleration for viewer, demo, and visualization experiments
- Engines are generated artifacts and must not be committed.
- Store local engine paths under `data/experiments/`, `data/ffs_proof_of_life/`,
  or another ignored/generated location documented by the active plan.
