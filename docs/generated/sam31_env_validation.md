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
