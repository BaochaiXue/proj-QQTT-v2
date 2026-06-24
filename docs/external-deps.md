# External Dependencies

External repos, checkpoints, TensorRT engines, SAM assets, and generated proof
artifacts stay outside this repo. This file records source-of-truth locations
for the single-camera branch.

## Fast-FoundationStereo

- Default external repo: `../Fast-FoundationStereo`
- Typical checkpoint family: `20-30-48`
- Example checkpoint path:
  - `../Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth`
- Runtime roles:
  - `cameras_viewer_FFS.py`
  - `data_process/record_data_align.py --depth_backend ffs`
  - `scripts/harness/diagnostics/demo/realtime_single_camera_pointcloud.py --depth-source ffs`
  - native-vs-FFS visualization harnesses

## SAM 3.1

- Runtime role:
  - single-camera masked PCD demo initialization
  - mask helper / visualization diagnostics
- Keep checkpoints outside this repo. Prefer environment variables or explicit
  CLI arguments when selecting local checkpoints.

## EdgeTAM

- Runtime role:
  - optional single-camera mask propagation in demo workflows
- Keep the external checkout outside this repo and document the path in the
  relevant validation plan or local environment notes.

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
- Default external SAM3D Objects checkout:
  - `/home/xinjie/external/sam-3d-objects`
- Default FuturePhysTwin checkout reference:
  - `/home/xinjie/FuturePhysTwin`
- Worker entrypoint in this repo:

```bash
python services/shape_prior_remote/server.py \
  --bind tcp://0.0.0.0:7100 \
  --sam3d-root /home/xinjie/external/sam-3d-objects \
  --upscale-category "stuffed animal" \
  --futurephystwin-root /home/xinjie/FuturePhysTwin
```

- Protocol/debug self-test mode that avoids importing SAM3D:

```bash
python services/shape_prior_remote/server.py \
  --bind tcp://0.0.0.0:7100 \
  --echo-observation
```

SAM3D, Stable Diffusion x4 upscaler, PyTorch3D, Kaolin, checkpoints, generated
route videos, and model weights remain external and must not be vendored into
this repo. The Demo 3.2 process talks to the worker over the lightweight
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
