# Demo 3 CoTracker3 Full Weights Validation

Generated: 2026-05-17T23:46:30-04:00

## Summary

- Goal: prepare CoTracker3's complete released checkpoint set for Demo 3.0.
- Official checkpoint source: `https://huggingface.co/facebook/cotracker3`.
- External repo: `/home/xinjie/co-tracker` at
  `82e02e8029753ad4ef13cf06be7f4fc5facdda4d`.
- Local checkpoint directory: `/home/xinjie/co-tracker/checkpoints`.
- Result: all four CoTracker3 `.pth` files are present, hashed, locally
  loadable in `demo3-max`, and mirrored into PyTorch Hub's checkpoint cache
  through symlinks.

## Commands

```bash
git pull --ff-only origin main
/home/xinjie/.local/bin/hf download facebook/cotracker3 \
  --local-dir /home/xinjie/co-tracker/checkpoints \
  --include "*.pth"
mkdir -p /home/xinjie/.cache/torch/hub/checkpoints
ln -sf /home/xinjie/co-tracker/checkpoints/scaled_online.pth \
  /home/xinjie/.cache/torch/hub/checkpoints/scaled_online.pth
ln -sf /home/xinjie/co-tracker/checkpoints/scaled_offline.pth \
  /home/xinjie/.cache/torch/hub/checkpoints/scaled_offline.pth
ln -sf /home/xinjie/co-tracker/checkpoints/baseline_online.pth \
  /home/xinjie/.cache/torch/hub/checkpoints/baseline_online.pth
ln -sf /home/xinjie/co-tracker/checkpoints/baseline_offline.pth \
  /home/xinjie/.cache/torch/hub/checkpoints/baseline_offline.pth
```

## Checkpoints

| File | Size bytes | SHA256 |
| --- | ---: | --- |
| `/home/xinjie/co-tracker/checkpoints/scaled_online.pth` | `101695610` | `205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218` |
| `/home/xinjie/co-tracker/checkpoints/scaled_offline.pth` | `101890938` | `2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834` |
| `/home/xinjie/co-tracker/checkpoints/baseline_online.pth` | `101694458` | `8b30b2f239de9987323b729d9115cc5163720a07348a97d045095cd9ebdb7b3a` |
| `/home/xinjie/co-tracker/checkpoints/baseline_offline.pth` | `101889786` | `da09bbac871f7398e5b29c4de5213652658949737bc158840b101678ba8ad1df` |

## Load Probes

`demo3-max` local checkpoint loading:

```text
scaled_online.pth: load_ok params=25385700 interp_shape=(384, 512) step=8
baseline_online.pth: load_ok params=25385700 interp_shape=(384, 512) step=8
scaled_offline.pth: load_ok params=25385700 interp_shape=(384, 512)
baseline_offline.pth: load_ok params=25385700 interp_shape=(384, 512)
```

Demo 3 default backend load path:

```text
backend_load_ok type=CoTrackerOnlinePredictor step=8 load_ms=2930.1
model_device=cuda:0
```

PyTorch Hub code cache was populated at:

```text
/home/xinjie/.cache/torch/hub/facebookresearch_co-tracker_main
```

## Validation

| Check | Result | Notes |
| --- | --- | --- |
| `git pull --ff-only origin main` | PASS | Already up to date. |
| Hugging Face `.pth` reconciliation | PASS | All four CoTracker3 weights present. |
| SHA256 inventory | PASS | Hashes recorded above. |
| Local checkpoint load in `demo3-max` | PASS | Online and offline scaled/baseline models load. |
| Default Demo 3 backend load | PASS | `CoTracker3OnlineBackend(device="cuda")` loads to `cuda:0`. |
| Demo 3 dry-run | PASS | `depth_source=realsense`, `mask_source=hf_edgetam`, `cotracker_backend=cotracker3_online`. |
| `scripts/harness/check_harness_catalog.py` | PASS | Catalog checks passed. |
| `scripts/harness/check_all.py` | PASS | Quick deterministic checks passed; 253 unittest tests OK. |

Current experiment prompts remain `object=stuffed animal` and
`controller=towel`.
