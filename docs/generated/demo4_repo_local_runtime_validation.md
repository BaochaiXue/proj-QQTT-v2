# Demo 4 Repo-Local Runtime Validation

Generated: 2026-06-24

## Focused Tests

Command:

```bash
conda run -n demo_2_max --no-capture-output \
  python -m unittest \
  tests.test_phystwin_strict_product \
  tests.test_demo_v4_futurephystwin_chunks \
  tests.test_demo32_shape_prior_warmup
```

Result:

```text
Ran 101 tests in 4.148s
OK
```

## Repo Smoke

Command:

```bash
conda run -n demo_2_max --no-capture-output \
  python scripts/harness/validation/run.py --profile smoke
```

Result:

```text
Ran 301 tests in 3.930s
OK
[validation] smoke checks passed
```

## Repo-Local Runtime Checks

Command:

```bash
find vendor/demo_runtime -type f \
  \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o \
     -name '*.safetensors' -o -name '*.bin' -o -name '*.engine' -o \
     -name '*.onnx' \) \
  -size +99M -not -path 'vendor/demo_runtime/checkpoints/*' \
  -printf '%s\t%p\n' | sort -nr
```

Result: no output. All 100 MB or larger model weight/cache files are under
`vendor/demo_runtime/checkpoints/`.

Runtime upstream-name grep:

```bash
rg -n \
  "stabilityai/stable-diffusion-x4-upscaler|Ruicheng/moge-vitl|yonigozlan/EdgeTAM-hf|facebookresearch/dinov2|source: github|huggingface\.co|hf_hub_download|snapshot_download|repo_id|from_pretrained\(\"[A-Za-z0-9_.-]+/" \
  demo_v3 demo_v3_1 demo_v3_2 demo_v3_3 demo_v4 qqtt/demo \
  services/shape_prior_remote data_process/depth_backends \
  vendor/demo_runtime/sam-3d-objects/sam3d_objects/model/backbone/dit/embedder/dino.py \
  vendor/demo_runtime/sam-3d-objects/checkpoints/hf \
  vendor/demo_runtime/stable-diffusion-x4-upscaler \
  vendor/demo_runtime/MoGe-vitl \
  --glob '!**/__pycache__/**'
```

Result: no output.

## Shape-Prior Worker

Command:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=1 \
conda run -n phystwin-max --no-capture-output \
  python services/shape_prior_remote/server.py \
  --bind tcp://127.0.0.1:7100 \
  --device cuda:0 \
  --preload-models \
  --debug
```

Key ready output:

```text
[shape-prior-worker] ready bind=tcp://127.0.0.1:7100 sam3d_root=vendor/demo_runtime/sam-3d-objects echo=False preload=True warmup=False worker_ready_ms=21728.9
Loading DINO model: dinov2_vitl14_reg from vendor/demo_runtime/dinov2 (source: local)
DINO backbone kwargs: {'weights': 'vendor/demo_runtime/checkpoints/dinov2/dinov2_vitl14_reg4_pretrain.pth'}
```

`--warmup-models` was also attempted on the local RTX 4090. It loaded local
models and local DINO successfully, then failed during the deterministic dummy
SAM3D decode with a 24 GB VRAM CUDA OOM. For this hardware validation the
worker used `--preload-models`; this still moves model loading off the camera
critical path, while the first real request performs upscaling and SAM3D
inference.

The real request completed:

```text
[shape-prior-worker] seq=0 status=ready points=1700 total_ms=25886.0 error=None
```

The worker logged a non-fatal optional GS layout post-optimization error because
`gsplat_cuda` could not load/build without a working `nvcc` in
`phystwin-max`. SAM3D caught that error and still returned aligned shape-prior
points.

## Demo 4 Fake-Live Validation

Command:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v4/realtime_futurephystwin_chunks.py \
  --input-source fake-live \
  --max-chunks 1 \
  --chunk-seconds 5 \
  --replay-fps 5 \
  --capture-extra-seconds 90 \
  --shape-prior-chunk-wait-timeout-s 300 \
  --chunk-poll-interval-s 0.001 \
  --case-prefix repo_local_realsense_final \
  --futurephystwin-base-path result/demo_v4/repo_local_realsense_final_20260624
```

Result: passed. Demo 4 launched Demo 3.2 fake-live with default
`native-realsense`, realtime side on GPU0, shape-prior worker on GPU1, and
repo-local relative runtime paths.

Key output:

```text
chunk_count: 1
depth_backend: native-realsense / realsense
chunk_materialization_source: prepared_phystwin_frame
shape_prior_complete: true
surface_point_count: 700
interior_point_count: 1000
object_point_count: 2020
controller_point_count: 30
first_ready_chunk_wall_s: 43.36636566900415
max_backlog_chunks: 4
demo32_cuda_visible_devices: 0
shape_prior_device: cuda:1
demo32_stop_reason: max_chunks_reached
```

Static `final_data.pkl` was written and loaded successfully:

```text
object_points (25, 2020, 3) float64
object_colors (25, 2020, 3) float64
object_visibilities (25, 2020) bool
object_motions_valid (25, 2020) bool
controller_points (25, 30, 3) float64
surface_points (700, 3) float64
interior_points (1000, 3) float64
```

Shape-prior profile:

```text
shape_prior_status ready
shape_prior_total_ms 25884.87550499849
time_to_shape_prior_ready_ms 41698.905122000724
image_upscale_ms 15753.815698000835
upscaler_model_load_ms 0.0
sam3d_model_load_ms 0.0
sam3d_inference_ms 8706.539059989154
single_view_alignment_ms 1369.767768017482
sampling_ms 53.61547999200411
worker_preloaded_models True
worker_warmed_models False
```

Output roots:

```text
result/demo_v4/repo_local_realsense_final_20260624/repo_local_realsense_final_chunks_manifest.json
result/demo_v4/repo_local_realsense_final_20260624/data/repo_local_realsense_final/final_data.pkl
result/demo_v4/repo_local_realsense_final_20260624/online_data/repo_local_realsense_final/chunks/chunk_000000.pkl
```

## Conclusion

Demo 4 fake-live now completes the repo-local runtime path from fake camera
frames to FuturePhysTwin-compatible `final_data.pkl` using the default
RealSense backend. The remaining environment note is optional gsplat CUDA
extension support in `phystwin-max`; it does not block the validated Demo 4
camera-to-final-data run.
