# SAM 3.1 Environment Validation

- Date: `2026-04-22`
- Repo root: `/home/zhangxinjie/proj-QQTT-v2`
- Env: `qqtt-ffs-compat`
- GPU: `NVIDIA GeForce RTX 5090 Laptop GPU`
- External checkpoint: `/mnt/c/Users/zhang/external/sam3_checkpoints/sam3.1_multiplex.pt`
- External BPE vocab: `/mnt/c/Users/zhang/external/sam3_checkpoints/bpe_simple_vocab_16e6.txt.gz`

## Goal

Validate whether QQTT's existing `SAM 3.1` sidecar helper can run in the local `qqtt-ffs-compat` env and whether the current external checkpoint can be loaded directly by the upstream `sam3` runtime.

## Environment Install / Repair Commands

```text
conda run -n qqtt-ffs-compat python -m pip install sam3==0.1.3
conda run -n qqtt-ffs-compat python -m pip install psutil
curl -L https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz -o /mnt/c/Users/zhang/external/sam3_checkpoints/bpe_simple_vocab_16e6.txt.gz
```

Observed install caveats:

- `sam3==0.1.3` imported `psutil` at module import time but did not install it as a dependency in this env
- the installed `sam3` wheel did not ship `bpe_simple_vocab_16e6.txt.gz`, so the vocab file was kept external next to the checkpoint

## Validation Commands

Deterministic helper tests:

```text
python -m unittest -v tests.test_sam31_mask_helper_smoke
```

Runtime import sanity:

```text
/home/zhangxinjie/miniconda3/envs/qqtt-ffs-compat/bin/python - <<'PY'
import sam3, torch
from pathlib import Path
print('sam3_version=', getattr(sam3, '__version__', 'unknown'))
print('torch_version=', torch.__version__)
print('cuda_available=', torch.cuda.is_available())
print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
print('checkpoint_exists=', Path('/mnt/c/Users/zhang/external/sam3_checkpoints/sam3.1_multiplex.pt').is_file())
PY
```

Predictor load probe:

```text
/home/zhangxinjie/miniconda3/envs/qqtt-ffs-compat/bin/python - <<'PY'
from scripts.harness.sam31_mask_helper import build_sam31_video_predictor, resolve_sam31_bpe_path
print('resolved_bpe=', resolve_sam31_bpe_path('/mnt/c/Users/zhang/external/sam3_checkpoints/sam3.1_multiplex.pt'))
predictor, checkpoint = build_sam31_video_predictor(
    checkpoint_path='/mnt/c/Users/zhang/external/sam3_checkpoints/sam3.1_multiplex.pt',
    async_loading_frames=False,
    compile_model=False,
    max_num_objects=16,
)
print('predictor_type=', type(predictor).__name__)
print('checkpoint=', checkpoint)
PY
```

Checkpoint structure probe:

```text
/home/zhangxinjie/miniconda3/envs/qqtt-ffs-compat/bin/python - <<'PY'
from collections import Counter
from pathlib import Path
import torch
path = Path('/mnt/c/Users/zhang/external/sam3_checkpoints/sam3.1_multiplex.pt')
ckpt = torch.load(path, map_location='cpu', weights_only=True)
sd = ckpt if isinstance(ckpt, dict) and 'model' not in ckpt else ckpt['model']
prefix_counts = Counter('.'.join(key.split('.')[:2]) for key in sd.keys())
print('state_dict_len=', len(sd))
print(prefix_counts.most_common(5))
PY
```

## Outcomes

### QQTT Helper Compatibility

- `tests.test_sam31_mask_helper_smoke` passed after adding:
  - current-upstream builder name compatibility
  - current-upstream Hugging Face download signature compatibility
  - external BPE vocab path resolution from checkpoint sibling or `QQTT_SAM31_BPE_PATH`
- the helper now works against the current upstream `sam3==0.1.3` API surface instead of only the older `build_sam3_predictor(...)` shape

### Runtime Sanity

- runtime import sanity passed with:
  - `sam3 0.1.3`
  - `torch 2.7.0+cu128`
  - CUDA visible on the local `RTX 5090 Laptop GPU`
- the external checkpoint file and external BPE vocab file were both visible from WSL

### External Checkpoint Load Result

- the predictor load probe did **not** complete successfully for `/mnt/c/Users/zhang/external/sam3_checkpoints/sam3.1_multiplex.pt`
- the failure was a strict `state_dict` mismatch inside upstream `sam3`
- the checkpoint structure probe showed:
  - top-level dict with `1623` parameters
  - dominant prefixes:
    - `detector.backbone` = `769`
    - `tracker.model` = `457`
    - `detector.transformer` = `283`
    - `detector.geometry_encoder` = `76`
    - `detector.segmentation_head` = `28`
- the mismatch included many upstream-expected keys under `tracker.*` and `sam2_convs.*` being missing, while the checkpoint supplied many keys under `tracker.model.*` and `interactive_convs.*`

## Conclusion

The local `qqtt-ffs-compat` env now supports the upstream `sam3` runtime and QQTT's `SAM 3.1` helper path, but the specific external checkpoint `sam3.1_multiplex.pt` is **not** directly loadable as an upstream `sam3==0.1.3` video predictor checkpoint.

Use this setup for:

- validating helper imports and runtime wiring
- loading upstream-compatible `sam3` checkpoints

Do not treat this validation as proof that arbitrary external `sam3.1`-named checkpoints are plug-compatible. The local `multiplex` checkpoint appears to come from a different model layout or forked training/runtime surface.

## Update: Official Repo Upgrade

### Additional Repair Commands

The original `sam3==0.1.3` PyPI install was replaced with the latest official `facebookresearch/sam3` code from GitHub, and the local shell/runtime wiring was updated to point QQTT at the downloaded Hugging Face checkpoint:

```text
conda run -n qqtt-ffs-compat python -m pip uninstall -y sam3
conda run -n qqtt-ffs-compat python -m pip install --no-deps --force-reinstall git+https://github.com/facebookresearch/sam3.git@main
hf download facebook/sam3.1 sam3.1_multiplex.pt --local-dir /home/zhangxinjie/.cache/huggingface/qqtt_sam31
curl -L https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz -o /home/zhangxinjie/.cache/huggingface/qqtt_sam31/bpe_simple_vocab_16e6.txt.gz
```

Persistent shell wiring:

- `hf` is now reachable from the default shell via `/home/zhangxinjie/miniconda3/bin/hf`
- `QQTT_SAM31_CHECKPOINT=/home/zhangxinjie/.cache/huggingface/qqtt_sam31/sam3.1_multiplex.pt`
- the BPE vocab resolves from the official package asset path:
  - `/home/zhangxinjie/miniconda3/envs/qqtt-ffs-compat/lib/python3.10/site-packages/sam3/assets/bpe_simple_vocab_16e6.txt.gz`

### Post-Upgrade Verification

Installed package origin:

```text
conda run -n qqtt-ffs-compat python -m pip show sam3
```

Observed result:

- package name: `sam3`
- installed version string: `0.1.0`
- source: `git+https://github.com/facebookresearch/sam3.git@main`
- resolved commit during install:
  - `2e0009e23f0ad0fbcbd0488df893d30d5c8c2565`

Helper smoke:

```text
bash -lc '/home/zhangxinjie/miniconda3/envs/qqtt-ffs-compat/bin/python - <<\"PY\"
from scripts.harness.sam31_mask_helper import build_sam31_video_predictor, resolve_sam31_bpe_path
print("bpe", resolve_sam31_bpe_path())
predictor, checkpoint = build_sam31_video_predictor(
    checkpoint_path=None,
    async_loading_frames=False,
    compile_model=False,
    max_num_objects=1,
)
print(checkpoint)
if hasattr(predictor, "shutdown"):
    predictor.shutdown()
print("sam31_predictor_init_ok")
PY'
```

Observed result:

- `resolve_sam31_checkpoint_path()` resolved:
  - `/home/zhangxinjie/.cache/huggingface/qqtt_sam31/sam3.1_multiplex.pt`
- `resolve_sam31_bpe_path()` resolved:
  - `/home/zhangxinjie/miniconda3/envs/qqtt-ffs-compat/lib/python3.10/site-packages/sam3/assets/bpe_simple_vocab_16e6.txt.gz`
- predictor initialization completed successfully
- the official repo emitted informational missing-key logs during model construction, but the predictor object was created and the smoke ended with:
  - `sam31_predictor_init_ok`

### Updated Conclusion

After replacing the PyPI wheel with the latest official `facebookresearch/sam3` code, the local `qqtt-ffs-compat` environment can now initialize QQTT's `SAM 3.1` helper against the downloaded Hugging Face `sam3.1_multiplex.pt` checkpoint.

## Update: `FFS-SAM-RS` Clone From `FFS-max`

- Date: `2026-04-28`
- Source env: `FFS-max`
- New env: `FFS-SAM-RS`
- Goal: preserve the newer local FFS stack while adding the official `SAM 3.1` runtime path.
- Checkpoint: `/home/zhangxinjie/.cache/huggingface/qqtt_sam31/sam3.1_multiplex.pt`

### Commands

```text
conda create -y -n FFS-SAM-RS --clone FFS-max
conda run -n FFS-SAM-RS python -m pip install --no-deps --force-reinstall git+https://github.com/facebookresearch/sam3.git@main
conda run -n FFS-SAM-RS python -m pip install iopath fvcore hydra-core ftfy regex
conda run -n FFS-SAM-RS python -m pip install pycocotools ftfy==6.1.1
```

The official `sam3` install resolved GitHub commit:

```text
c97c893969003d3e6803fd5d679f21e515aef5ce
```

### Version Sanity

```text
python_env=FFS-SAM-RS
torch 2.11.0+cu130 cuda 13.0 cuda_available True
tensorrt 10.16.1.11
pyrealsense2 2.57.7.10387
sam3 0.1.0 /home/zhangxinjie/miniconda3/envs/FFS-SAM-RS/lib/python3.12/site-packages/sam3/__init__.py
numpy 2.4.4
ftfy 6.1.1
```

The source env was not modified by the SAM install:

```text
env=FFS-max
torch 2.11.0+cu130 cuda 13.0
pyrealsense2 import_ok
sam3_installed False
```

### Dependency Caveat

`pip check` reports one intentional incompatibility:

```text
sam3 0.1.0 has requirement numpy<2,>=1.26, but you have numpy 2.4.4.
```

This was left unresolved because downgrading NumPy would defeat the purpose of preserving the newer `FFS-max` stack. Treat this env as a validated compatibility probe, not a fully upstream-conformant `sam3` package environment.

### Runtime Probe

QQTT's helper successfully initialized a SAM 3.1 multiplex predictor in `FFS-SAM-RS`:

```text
predictor_type Sam3MultiplexVideoPredictor
checkpoint /home/zhangxinjie/.cache/huggingface/qqtt_sam31/sam3.1_multiplex.pt
sam31_predictor_init_ok
```

A real mask-generation probe also completed using an existing aligned case:

```text
conda run -n FFS-SAM-RS python scripts/harness/generate_sam31_masks.py \
  --case_root data/static/static_ir_only_proj_off_round1_20260428 \
  --output_dir /tmp/ffssamrs_sam31_probe \
  --text_prompt 'stuffed animal' \
  --camera_ids 0 \
  --source_mode frames \
  --ann_frame_index 0 \
  --checkpoint /home/zhangxinjie/.cache/huggingface/qqtt_sam31/sam3.1_multiplex.pt \
  --max_num_objects 1 \
  --overwrite
```

Observed result:

```text
SAM 3.1 masks written to /tmp/ffssamrs_sam31_probe
/tmp/ffssamrs_sam31_probe/mask/0/0/0.png
/tmp/ffssamrs_sam31_probe/mask/0/0/1.png
...
```

The probe output directory was removed after validation.

### Conclusion

`FFS-SAM-RS` can keep the current `FFS-max` Python 3.12 / Torch CUDA 13 / TensorRT 10.16 / RealSense stack and still run the repo's SAM 3.1 helper path. The remaining risk is the official `sam3` metadata requirement for `numpy<2`; the validated helper initialization and one-camera mask generation path work with `numpy 2.4.4`, but broader `sam3` APIs are not exhaustively validated.
