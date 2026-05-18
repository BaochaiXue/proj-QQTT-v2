# demo3-max Native Ubuntu RTX 4090 Environment Validation

Generated: 2026-05-17T23:17:58-04:00

## Summary

- Machine: native Ubuntu desktop with RTX 4090 GPUs.
- Environment: `demo3-max`.
- Clone source: `demo_2_max`.
- Install policy: cloned the existing integrated stack and added only CoTracker3
  runtime support plus external checkpoints.
- Demo 3 depth contract: RealSense depth only; `uses_ffs=false`.
- Live validation status: dry-run passed; live mask/q30/q128 runs are blocked
  until a real root-level `calibrate.pkl` is present.

## Machine

- `nvidia-smi`: NVIDIA-SMI `580.126.09`, driver `580.126.09`, CUDA `13.0`.
- GPUs visible:
  - `NVIDIA GeForce RTX 4090`, bus `00000000:C1:00.0`
  - `NVIDIA GeForce RTX 4090`, bus `00000000:E1:00.0`
- PyTorch selected GPU 0 during probes:
  - name: `NVIDIA GeForce RTX 4090`
  - capability: `(8, 9)`

## Environment

- Conda executable: `/home/xinjie/miniforge3/bin/conda`
- Conda env: `demo3-max`
- Prefix: `/home/xinjie/miniforge3/envs/demo3-max`
- Created with:
  - `/home/xinjie/miniforge3/bin/conda create -y -n demo3-max --clone demo_2_max`
- Activation hook:
  - `/home/xinjie/miniforge3/envs/demo3-max/etc/conda/activate.d/demo3_max_paths.sh`
  - `QQTT_REPO_ROOT=/home/xinjie/proj-QQTT-v2`
  - `EDGETAM_REPO=/home/xinjie/EdgeTAM`
  - `COTRACKER_REPO=/home/xinjie/co-tracker`
  - `CUDA_HOME=/usr/local/cuda`
  - `TORCH_CUDA_ARCH_LIST=8.9`
  - `QQTT_SAM31_CHECKPOINT=/home/xinjie/.cache/huggingface/qqtt_sam31/sam3.1_multiplex.pt`

## Runtime Package Probe

- Python: `3.12.13`
- `torch`: `2.11.0+cu130`
- `torch.version.cuda`: `13.0`
- `torchvision`: `0.26.0+cu130`
- CUDA available: `true`
- `pyrealsense2`: import OK
- `open3d`: `0.19.0`
- `transformers`: `5.7.0`
- `EdgeTamVideoModel`: import OK
- `Sam2VideoProcessor`: import OK
- `sam3`: import OK after adding `pycocotools==2.0.11`
- `cotracker`: import OK from `/home/xinjie/co-tracker/cotracker/__init__.py`
- `cotracker`: `3.0`
- `flow-vis`: `0.1`
- `tensorboard`: `2.20.0`
- `moviepy`: `2.2.1`
- `imageio-ffmpeg`: `0.6.0`
- `setuptools`: `81.0.0`
- `numpy`: `2.4.5`
- `triton`: `3.6.0`

`python -m pip check` reports the inherited SAM 3.1 metadata mismatch:
`sam3 0.1.0 has requirement numpy<2,>=1.26, but you have numpy 2.4.5`.
The cloned base stack already uses the same high-version numpy policy, so this
install did not downgrade numpy.

## External Assets

- CoTracker repo: `/home/xinjie/co-tracker`
- CoTracker commit: `82e02e8029753ad4ef13cf06be7f4fc5facdda4d`
- CoTracker install:
  - `/home/xinjie/miniforge3/bin/conda run --no-capture-output -n demo3-max python -m pip install -e /home/xinjie/co-tracker`
- CoTracker dependency supplement:
  - `matplotlib flow_vis tqdm tensorboard imageio[ffmpeg] moviepy`
- CoTracker3 checkpoints:
  - `/home/xinjie/co-tracker/checkpoints/scaled_online.pth`
    - size: `101695610`
    - sha256: `205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218`
  - `/home/xinjie/co-tracker/checkpoints/baseline_online.pth`
    - size: `101694458`
    - sha256: `8b30b2f239de9987323b729d9115cc5163720a07348a97d045095cd9ebdb7b3a`
- SAM 3.1 checkpoint:
  - `/home/xinjie/.cache/huggingface/qqtt_sam31/sam3.1_multiplex.pt`
  - size: `3502755717`

CoTracker model load check:

```text
CoTrackerOnlinePredictor load OK
step: 8
interp_shape: (384, 512)
params: 25385700
```

## Repo Contract

- `demo_v3/` exists.
- Before this installation pass, `docs/envs.md` did not formally register a
  `demo3-max` or `demo_3_max` environment.
- Demo 3 dry-run contract:
  - `demo = demo3`
  - `requires_three_realsense = true`
  - `num_cameras = 3`
  - `depth_source = realsense`
  - `uses_ffs = false`
  - `mask_source = hf_edgetam`
  - `edgetam_batch_vision_encoder = true`
  - `cotracker_backend = cotracker3_online`
  - `cotracker_async = true`
  - `render_waited_for_cotracker = false`
- Current experiment prompts:
  - object: `stuffed animal`
  - controller: `towel`

## Hardware Probe

`pyrealsense2` sees exactly three RealSense devices:

| Index | Name | Serial | USB |
| --- | --- | --- | --- |
| 0 | Intel RealSense D455 | `239222300781` | `3.2` |
| 1 | Intel RealSense D455 | `239222303506` | `3.2` |
| 2 | Intel RealSense D455 | `239222300412` | `3.2` |

Root-level `calibrate.pkl` is missing. Only the test fixture exists at
`tests/fixtures/record_data_align_minimal/calibrate.pkl`; it was not used as a
live calibration substitute.

## Validation

| Check | Result | Notes |
| --- | --- | --- |
| `git pull --ff-only origin main` | PASS | Already up to date. |
| Base stack import probe | PASS | Torch/CUDA, RealSense, Open3D all import. |
| HF EdgeTAM imports | PASS | `EdgeTamVideoModel`, `Sam2VideoProcessor`. |
| SAM 3.1 import | PASS | Required adding `pycocotools==2.0.11`. |
| CoTracker import | PASS | Editable install from `/home/xinjie/co-tracker`. |
| CoTracker checkpoint load | PASS | `CoTrackerOnlinePredictor` loaded `scaled_online.pth`. |
| `python -m pip check` | WARN | Only `sam3` vs `numpy==2.4.5` metadata mismatch remains. |
| Demo 3 dry-run | PASS | Contract shows RealSense depth and `uses_ffs=false`. |
| RealSense device probe | PASS | Exactly three D455 devices visible. |
| `scripts/harness/check_harness_catalog.py` | PASS | Catalog checks passed. |
| `scripts/harness/check_all.py` | PASS | Quick deterministic checks passed; 253 unittest tests OK. |
| Mask-only live run | BLOCKED | Fails before runtime loop: missing root `calibrate.pkl`. |
| q30 CoTracker live run | BLOCKED | Same calibration blocker. |
| q128 CoTracker live run | BLOCKED | Same calibration blocker. |
| Profile/no-render live run | BLOCKED | Same calibration blocker. |

Mask-only live command attempted:

```bash
timeout 90s /home/xinjie/miniforge3/bin/conda run --no-capture-output -n demo3-max \
  python demo_v3/realtime_three_view_cotracker3_realsense_overlay.py \
  --preset demo3-realsense-mask-only \
  --camera-ids 0,1,2 \
  --duration-s 60 \
  --debug \
  --profile-json-output docs/generated/demo3_4090_mask_only_profile.json
```

Observed failure:

```text
Demo 3 requires calibrate.pkl for three-camera world fusion: calibrate.pkl
```

## Next Manual Step

Create or place the real three-camera calibration file at repo root as
`calibrate.pkl`, then rerun the live sequence:

1. `--preset demo3-realsense-mask-only`
2. `--preset demo3-realsense-cotracker-highfps --cotracker-query-count 30`
3. `--preset demo3-realsense-cotracker-highfps --cotracker-query-count 128`
4. `--preset demo3-realsense-cotracker-profile`
