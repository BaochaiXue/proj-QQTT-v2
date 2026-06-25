# Demo 4 Repo-Local Runtime Validation

Generated: 2026-06-24

## Focused Tests

Command:

```bash
conda run -n demo_2_max --no-capture-output \
  python -m unittest tests.test_demo_v4_futurephystwin_chunks tests.test_demo32_shape_prior_warmup
```

Result:

```text
Ran 89 tests in 3.949s
OK
```

## Repo-Local Default Path Check

Command:

```bash
conda run -n demo_2_max --no-capture-output python - <<'PY'
from demo_v4.realtime_futurephystwin_chunks import build_parser as build_v4
from services.shape_prior_remote import server
from data_process.depth_backends import DEFAULT_FFS_REPO
from qqtt.demo import realtime_masked_edgetam_pcd as masked
args = build_v4().parse_args([])
items = {
    "demo4_base": args.futurephystwin_base_path,
    "sam3d_root": server.DEFAULT_SAM3D_ROOT,
    "futurephystwin_root": server.DEFAULT_FUTUREPHYSTWIN_ROOT,
    "ffs_repo": DEFAULT_FFS_REPO,
    "tapnet": masked.DEFAULT_TAPNET_REPO_DIR,
    "tapnextpp_ckpt": masked.DEFAULT_TAPNEXTPP_CHECKPOINT,
    "edgetam_model": Path(masked.DEFAULT_MODEL_ID),
}
for name, path in items.items():
    print(name, path, "absolute=", path.is_absolute(), "exists=", path.exists())
PY
```

Result:

```text
demo4_base result/demo_v4/futurephystwin_chunks absolute= False exists= True
sam3d_root vendor/demo_runtime/sam-3d-objects absolute= False exists= True
futurephystwin_root vendor/demo_runtime/FuturePhysTwin absolute= False exists= True
ffs_repo vendor/demo_runtime/Fast-FoundationStereo absolute= False exists= True
tapnet vendor/demo_runtime/tapnet absolute= False exists= True
tapnextpp_ckpt vendor/demo_runtime/checkpoints/tapnextpp/tapnextpp_ckpt.pt absolute= False exists= True
edgetam_model vendor/demo_runtime/EdgeTAM-hf absolute= False exists= True
```

## Demo 4 Fake-Live Validation Attempt

Command:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v4/realtime_futurephystwin_chunks.py \
  --input-source fake-live \
  --replay-fps 5 \
  --chunk-seconds 5 \
  --max-chunks 1 \
  --capture-extra-seconds 20 \
  --shape-prior-timeout-ms 5000 \
  --shape-prior-chunk-wait-timeout-s 5
```

Result: failed before chunk publication because CUDA is unavailable in the
current session.

Key output:

```text
[tapnextpp-tracker] backend=tapnextpp device=cuda repo=vendor/demo_runtime/tapnet checkpoint=vendor/demo_runtime/checkpoints/tapnextpp/tapnextpp_ckpt.pt
[FATAL] segmentation worker failed: RuntimeError: CUDA device requested but torch.cuda.is_available() is false
futurephystwin_base_path: result/demo_v4/futurephystwin_chunks
mode: full-fake-realtime-camera
```

GPU check:

```text
nvidia-smi: couldn't communicate with the NVIDIA driver
/dev/nvidia*: absent
torch.cuda.is_available(): False
torch.cuda.device_count(): 0
```

## CPU Fallback Path Check

Command:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v4/realtime_futurephystwin_chunks.py \
  --input-source fake-live \
  --replay-fps 5 \
  --chunk-seconds 5 \
  --max-chunks 1 \
  --capture-extra-seconds 20 \
  --shape-prior-timeout-ms 5000 \
  --shape-prior-chunk-wait-timeout-s 5 \
  --demo32-device cpu \
  --demo32-tracker-device cpu \
  --demo32-dtype float32 \
  --no-shape-prior-warmup
```

Result: fake-live started, loaded EdgeTAM from repo-local
`vendor/demo_runtime/EdgeTAM-hf`, and then failed because the upstream SAM 3.1
image model requires CUDA.

Key output:

```text
[tapnextpp-tracker] backend=tapnextpp device=cpu repo=vendor/demo_runtime/tapnet checkpoint=vendor/demo_runtime/checkpoints/tapnextpp/tapnextpp_ckpt.pt
[edgetam] model=vendor/demo_runtime/EdgeTAM-hf device=cpu dtype=float32 track_mode=controller-object compile_mode=vision-reduce-overhead applied=['vision_encoder']
[FATAL] segmentation worker failed: RuntimeError: The upstream SAM 3.1 image model currently requires CUDA.
futurephystwin_base_path: result/demo_v4/futurephystwin_chunks
mode: full-fake-realtime-camera
```

## Conclusion

Demo 4 fake-live was launched using repo-local relative runtime defaults. It did
not fail because of missing absolute `/home/...` model paths, Hugging Face
network fetches, or parent-repo fallbacks. The remaining blocker in this
session is unavailable CUDA/NVIDIA driver access, which prevents SAM 3.1 from
producing the first-frame initialization masks needed for chunk publication.
