# Demo 3.1 Max Environment Validation - 2026-05-19

## Environment

- Conda env: `/home/xinjie/miniforge3/envs/demo_3_1_max`
- Created by cloning: `/home/xinjie/miniforge3/envs/demo3-max`
- External repos are outside the QQTT repo:
  - Track-On2: `/home/xinjie/external/track_on`, revision `7e838e8`
  - LiteTracker: `/home/xinjie/external/lite-tracker`, revision `8ffe249`
- External weights are outside the QQTT repo:
  - Track-On2 checkpoint: `/home/xinjie/external/weights/track_on2/trackon2_dinov3_checkpoint.pt`
  - CoTracker3 scaled online weights for LiteTracker: `/home/xinjie/external/weights/cotracker3/scaled_online.pth`

## Core Import Check

Command:

```bash
conda run -n demo_3_1_max --no-capture-output python - <<'PY'
import torch, cv2, numpy, timm, transformers, huggingface_hub, accelerate, mmcv
from mmcv.ops import MultiScaleDeformableAttention
import model.trackon_predictor as trackon_predictor
from src.lite_tracker import LiteTracker
import sam3
print("torch", torch.__version__, torch.version.cuda)
print("cv2", cv2.__version__)
print("numpy", numpy.__version__)
print("timm", timm.__version__)
print("transformers", transformers.__version__)
print("huggingface_hub", huggingface_hub.__version__)
print("accelerate", accelerate.__version__)
print("mmcv", mmcv.__version__)
print("mmcv_ops", MultiScaleDeformableAttention.__name__)
print("trackon_predictor", trackon_predictor.__file__)
print("LiteTracker", LiteTracker.__name__)
print("sam3 ok")
PY
```

Outcome:

```text
torch 2.11.0+cu130 13.0
cv2 4.13.0
numpy 2.4.5
timm 1.0.27
transformers 5.7.0
huggingface_hub 1.15.0
accelerate 1.13.0
mmcv 2.2.0
mmcv_ops MultiScaleDeformableAttention
trackon_predictor /home/xinjie/external/track_on/model/trackon_predictor.py
LiteTracker LiteTracker
sam3 ok
```

## Backend Dry Run Checks

Commands:

```bash
conda run -n demo_3_1_max --no-capture-output python demo_v3_1/realtime_three_view_cotracker3_realsense_overlay_dual4090.py \
  --dry-run --camera-ids 0,1,2 --mask-gpu 0 --cotracker-gpu 1 \
  --cotracker-backend cotracker3_online

conda run -n demo_3_1_max --no-capture-output python demo_v3_1/realtime_three_view_cotracker3_realsense_overlay_dual4090.py \
  --dry-run --camera-ids 0,1,2 --mask-gpu 0 --cotracker-gpu 1 \
  --cotracker-backend trackon2 --tracking-backend-execution-mode auto \
  --trackon2-repo-dir /home/xinjie/external/track_on \
  --trackon2-checkpoint /home/xinjie/external/weights/track_on2/trackon2_dinov3_checkpoint.pt

conda run -n demo_3_1_max --no-capture-output python demo_v3_1/realtime_three_view_cotracker3_realsense_overlay_dual4090.py \
  --dry-run --camera-ids 0,1,2 --mask-gpu 0 --cotracker-gpu 1 \
  --cotracker-backend litetracker --tracking-backend-execution-mode auto \
  --litetracker-repo-dir /home/xinjie/external/lite-tracker \
  --litetracker-weights /home/xinjie/external/weights/cotracker3/scaled_online.pth
```

Outcome:

```text
cotracker3_online dry-run: passed, batch camera mode declared.
trackon2 dry-run: passed, batch camera mode declared.
litetracker dry-run: passed, serial mode declared because batch support is unknown.
All dry-run contracts report tracker_env_name = demo_3_1_max and fps = 30.
```

## External Backend Runtime Notes

- `mmcv==2.2.0` was built from source in `demo_3_1_max` with CUDA ops enabled for RTX 4090 capability `8.9`.
- Track-On2 Python import and `mmcv.ops.MultiScaleDeformableAttention` import both pass.
- LiteTracker source import passes and the CoTracker3 scaled online checkpoint loads into `LiteTracker` with zero missing and zero unexpected keys.
- DINOv3 plus backbone was downloaded through Hugging Face after access was granted:
  `/home/xinjie/.cache/huggingface/hub/models--facebook--dinov3-vits16plus-pretrain-lvd1689m/snapshots/c93d816fc9e567563bc068f01475bec89cc634a6`
- Track-On2 full `Predictor` construction now passes with the downloaded backbone and checkpoint.
- Local Track-On2 repo compatibility note: current `transformers` exposes DINOv3 transformer blocks as `DINOv3ViTModel.model.layer`, while the checked-out Track-On2 code expected `DINOv3ViTModel.layer`. The external repo at `/home/xinjie/external/track_on` was patched to support both locations without registering a duplicate PyTorch module alias. The reproducibility patch is recorded in `docs/generated/trackon2_dinov3_transformers_compat_20260519.patch`.

Track-On2 Predictor validation command:

```bash
conda run -n demo_3_1_max --no-capture-output python - <<'PY'
from model.trackon_predictor import Predictor
ckpt = "/home/xinjie/external/weights/track_on2/trackon2_dinov3_checkpoint.pt"
model = Predictor(checkpoint_path=ckpt, support_grid_size=5)
param_count = sum(p.numel() for p in model.parameters())
print("trackon2 Predictor init ok")
print("param_count", param_count)
print("checkpoint", ckpt)
PY
```

Outcome:

```text
Loaded model weights from /home/xinjie/external/weights/track_on2/trackon2_dinov3_checkpoint.pt
Info: missing (allowed) weights: 235 keys under {'backbone.vit_encoder.dinov3'}
trackon2 Predictor init ok
param_count 52131867
checkpoint /home/xinjie/external/weights/track_on2/trackon2_dinov3_checkpoint.pt
```

## Known Inherited Check

Command:

```bash
conda run -n demo_3_1_max --no-capture-output python -m pip check
```

Outcome:

```text
sam3 0.1.0 has requirement numpy<2,>=1.26, but you have numpy 2.4.5.
```

This mismatch is inherited from the cloned `demo3-max` stack. `sam3` still imports successfully in `demo_3_1_max`.
