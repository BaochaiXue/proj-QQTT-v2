# FFS Single-Engine TensorRT WSL Validation

- Date: `2026-04-22`
- Machine: `XinjieZhang`
- GPU: `NVIDIA GeForce RTX 5090 Laptop GPU`
- Environment: `ffs-standalone`
- Scope note: this validates the external Fast-FoundationStereo single-ONNX / single-engine TensorRT route on WSL/Linux and produces a viewer-consumable `864x480` engine directory.

## Commands

Single ONNX export at viewer-compatible engine size:

```text
conda run -n ffs-standalone python /home/zhangxinjie/Fast-FoundationStereo/scripts/make_single_onnx.py --model_dir /home/zhangxinjie/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth --save_path /home/zhangxinjie/proj-QQTT-v2/data/ffs_proof_of_life/trt_single_engine_864x480_wsl --height 480 --width 864 --valid_iters 4 --max_disp 192
```

TensorRT engine build via Python API because `trtexec` was not available on this machine:

```text
/home/zhangxinjie/miniconda3/envs/ffs-standalone/bin/python -c "from pathlib import Path; import sys; sys.path.insert(0, '/home/zhangxinjie/proj-QQTT-v2'); from scripts.harness.verify_ffs_tensorrt_wsl import build_engine_from_onnx; build_engine_from_onnx(onnx_path=Path('/home/zhangxinjie/proj-QQTT-v2/data/ffs_proof_of_life/trt_single_engine_864x480_wsl/fast_foundationstereo.onnx'), engine_path=Path('/home/zhangxinjie/proj-QQTT-v2/data/ffs_proof_of_life/trt_single_engine_864x480_wsl/fast_foundationstereo.engine'), log_path=Path('/home/zhangxinjie/proj-QQTT-v2/data/ffs_proof_of_life/trt_single_engine_864x480_wsl/single_engine_build.log'), workspace_gib=8, fp16=True); print('single-engine TensorRT build complete')"
```

Headless demo validation through the repo-integrated single-engine runner:

```text
/home/zhangxinjie/miniconda3/envs/ffs-standalone/bin/python - <<'PY'
from pathlib import Path
import sys
import cv2
import imageio.v2 as imageio
import numpy as np

ROOT = Path('/home/zhangxinjie/proj-QQTT-v2')
FFS_REPO = Path('/home/zhangxinjie/Fast-FoundationStereo')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(FFS_REPO) not in sys.path:
    sys.path.insert(0, str(FFS_REPO))

from data_process.depth_backends.fast_foundation_stereo import FastFoundationStereoSingleEngineTensorRTRunner
from Utils import depth2xyzmap, o3d, toOpen3dCloud, vis_disparity

out_dir = ROOT / 'data' / 'ffs_proof_of_life' / 'trt_single_engine_864x480_wsl' / 'demo_out'
out_dir.mkdir(parents=True, exist_ok=True)
left_file = FFS_REPO / 'demo_data' / 'left.png'
right_file = FFS_REPO / 'demo_data' / 'right.png'
intrinsic_file = FFS_REPO / 'demo_data' / 'K.txt'
runner = FastFoundationStereoSingleEngineTensorRTRunner(
    ffs_repo=FFS_REPO,
    model_dir=ROOT / 'data' / 'ffs_proof_of_life' / 'trt_single_engine_864x480_wsl',
)
left = imageio.imread(left_file)[..., :3]
right = imageio.imread(right_file)[..., :3]
with open(intrinsic_file, 'r', encoding='utf-8') as handle:
    lines = handle.readlines()
K = np.array(list(map(float, lines[0].rstrip().split())), dtype=np.float32).reshape(3, 3)
baseline = float(lines[1])
result = runner.run_pair(left, right, K_ir_left=K, baseline_m=baseline)
disparity = np.asarray(result['disparity'], dtype=np.float32)
depth = np.asarray(result['depth_ir_left_m'], dtype=np.float32)
K_used = np.asarray(result['K_ir_left_used'], dtype=np.float32)
vis = vis_disparity(disparity, min_val=None, max_val=None, cmap=None, color_map=cv2.COLORMAP_TURBO)
prep_left = cv2.resize(left, (disparity.shape[1], disparity.shape[0]), interpolation=cv2.INTER_LINEAR)
prep_right = cv2.resize(right, (disparity.shape[1], disparity.shape[0]), interpolation=cv2.INTER_LINEAR)
imageio.imwrite(out_dir / 'left.png', prep_left)
imageio.imwrite(out_dir / 'right.png', prep_right)
imageio.imwrite(out_dir / 'disp_vis.png', np.concatenate([prep_left, prep_right, vis], axis=1))
np.save(out_dir / 'depth_meter.npy', depth)
xyz_map = depth2xyzmap(depth, K_used)
pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), prep_left.reshape(-1, 3))
pts = np.asarray(pcd.points)
keep = (pts[:, 2] > 0) & np.isfinite(pts[:, 2]) & (pts[:, 2] <= 100.0)
pcd = pcd.select_by_index(np.where(keep)[0])
o3d.io.write_point_cloud(str(out_dir / 'cloud.ply'), pcd)
print('single-engine demo artifacts written to', out_dir)
PY
```

Short latency profile through the repo-integrated single-engine runner:

```text
/home/zhangxinjie/miniconda3/envs/ffs-standalone/bin/python - <<'PY'
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path('/home/zhangxinjie/proj-QQTT-v2')
FFS_REPO = Path('/home/zhangxinjie/Fast-FoundationStereo')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(FFS_REPO) not in sys.path:
    sys.path.insert(0, str(FFS_REPO))

from data_process.depth_backends.fast_foundation_stereo import FastFoundationStereoSingleEngineTensorRTRunner

runner = FastFoundationStereoSingleEngineTensorRTRunner(
    ffs_repo=FFS_REPO,
    model_dir=ROOT / 'data' / 'ffs_proof_of_life' / 'trt_single_engine_864x480_wsl',
)
left = np.random.randint(0, 256, (480, 864, 3), dtype=np.uint8)
right = np.random.randint(0, 256, (480, 864, 3), dtype=np.uint8)
K = np.array([[1000.0, 0.0, 432.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)
for _ in range(10):
    _ = runner.run_pair(left, right, K_ir_left=K, baseline_m=0.095)
times = []
for _ in range(20):
    start = time.perf_counter()
    _ = runner.run_pair(left, right, K_ir_left=K, baseline_m=0.095)
    times.append((time.perf_counter() - start) * 1000.0)
print('single-engine mean ms', float(np.mean(times)))
print('single-engine min ms', float(np.min(times)))
print('single-engine max ms', float(np.max(times)))
PY
```

Single-camera live viewer smoke:

```text
conda run -n qqtt-ffs-compat python cameras_viewer_FFS.py --max-cams 1 --width 848 --height 480 --duration-s 10 --stats-log-interval-s 5 --ffs_backend tensorrt --ffs_trt_mode single_engine --ffs_repo /home/zhangxinjie/Fast-FoundationStereo --ffs_trt_model_dir /home/zhangxinjie/proj-QQTT-v2/data/ffs_proof_of_life/trt_single_engine_864x480_wsl
```

## Important WSL Notes

- The local machine did not have `trtexec` on `PATH`, so the engine was built with the TensorRT Python API instead.
- The viewer-compatible engine size was fixed at `864x480` so the current `848x480` live viewer can continue to use the same symmetric pad path as the existing two-stage TRT setup.
- This proof-of-life used the same upstream checkpoint family as the existing two-stage WSL validation:
  - `weights/23-36-37/model_best_bp2_serialize.pth`
- The headless demo path used the repo-integrated `FastFoundationStereoSingleEngineTensorRTRunner` rather than the upstream interactive demo script so validation would not block on `cv2.imshow()` / `waitKey(0)`.

## Outputs

- `data/ffs_proof_of_life/trt_single_engine_864x480_wsl/fast_foundationstereo.onnx`
- `data/ffs_proof_of_life/trt_single_engine_864x480_wsl/fast_foundationstereo.yaml`
- `data/ffs_proof_of_life/trt_single_engine_864x480_wsl/fast_foundationstereo.engine`
- `data/ffs_proof_of_life/trt_single_engine_864x480_wsl/single_engine_build.log`
- `data/ffs_proof_of_life/trt_single_engine_864x480_wsl/demo_out/disp_vis.png`
- `data/ffs_proof_of_life/trt_single_engine_864x480_wsl/demo_out/depth_meter.npy`
- `data/ffs_proof_of_life/trt_single_engine_864x480_wsl/demo_out/cloud.ply`

## Results

Export:

- `fast_foundationstereo.onnx` exported successfully at `480x864`
- `fast_foundationstereo.yaml` written successfully

Engine build:

- `fast_foundationstereo.engine` built successfully
- engine size: about `57 MB`
- TensorRT reported engine generation time: about `59.5 s`
- build completed under FP16 with an `8 GiB` workspace limit

Build warnings observed:

- TensorRT reported several FP16 constant-cast clipping warnings to `+/-65504`
- build still completed successfully and produced a loadable engine

Headless demo:

- completed successfully against the upstream demo pair
- wrote `disp_vis.png`, `depth_meter.npy`, and `cloud.ply`
- the demo output directory also contains resized `left.png` and `right.png`

Short latency profile at `864x480`:

- mean latency: about `31.7 ms`
- min latency: about `30.4 ms`
- max latency: about `33.3 ms`

## Follow-Up Boundary

- This proof-of-life does not validate `record_data_align.py` or any aligned-case TRT integration.
- A short single-camera live viewer smoke was completed, but this note still does not claim broad live multi-camera validation.

## Live Viewer Smoke

Observed startup:

- detected camera: `239222303506`
- started at `848x480@30`
- startup note reported padding: `TensorRT engine 864x480; capture 848x480 will be symmetrically padded to 864x480 before inference.`

Observed runtime stats:

- around `t=10.0s`:
  - `capture=33.3`
  - `ffs=11.5`
  - `infer_ms=85.1`
  - `seq_gap=4`

Interpretation:

- the generated single-engine directory is directly consumable by `cameras_viewer_FFS.py`
- the current single-camera viewer path no longer hits the earlier `ModuleNotFoundError: No module named 'Utils'`
- the first `5s` sample window was still in warmup, but the second window produced stable FFS results

## FP32 Follow-Up Build

Goal:

- build a standalone FP32 variant without adding a second `.engine` to the existing FP16 directory

Artifact directory:

- `data/ffs_proof_of_life/trt_single_engine_864x480_wsl_fp32`

Build command:

```text
/home/zhangxinjie/miniconda3/envs/ffs-standalone/bin/python - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, '/home/zhangxinjie/proj-QQTT-v2')
from scripts.harness.verify_ffs_tensorrt_wsl import build_engine_from_onnx
out_dir = Path('/home/zhangxinjie/proj-QQTT-v2/data/ffs_proof_of_life/trt_single_engine_864x480_wsl_fp32')
build_engine_from_onnx(
    onnx_path=out_dir / 'fast_foundationstereo.onnx',
    engine_path=out_dir / 'fast_foundationstereo.engine',
    log_path=out_dir / 'single_engine_build.log',
    workspace_gib=8,
    fp16=False,
)
print('fp32 single-engine TensorRT build complete')
PY
```

Smoke validation command:

```text
/home/zhangxinjie/miniconda3/envs/qqtt-ffs-compat/bin/python - <<'PY'
from pathlib import Path
import sys
import cv2
import imageio.v2 as imageio
import numpy as np

ROOT = Path('/home/zhangxinjie/proj-QQTT-v2')
FFS_REPO = Path('/home/zhangxinjie/Fast-FoundationStereo')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(FFS_REPO) not in sys.path:
    sys.path.insert(0, str(FFS_REPO))

from data_process.depth_backends.fast_foundation_stereo import FastFoundationStereoSingleEngineTensorRTRunner
from Utils import depth2xyzmap, o3d, toOpen3dCloud, vis_disparity

out_dir = ROOT / 'data' / 'ffs_proof_of_life' / 'trt_single_engine_864x480_wsl_fp32' / 'demo_out'
out_dir.mkdir(parents=True, exist_ok=True)
left_file = FFS_REPO / 'demo_data' / 'left.png'
right_file = FFS_REPO / 'demo_data' / 'right.png'
intrinsic_file = FFS_REPO / 'demo_data' / 'K.txt'
runner = FastFoundationStereoSingleEngineTensorRTRunner(
    ffs_repo=FFS_REPO,
    model_dir=ROOT / 'data' / 'ffs_proof_of_life' / 'trt_single_engine_864x480_wsl_fp32',
)
left = imageio.imread(left_file)[..., :3]
right = imageio.imread(right_file)[..., :3]
with open(intrinsic_file, 'r', encoding='utf-8') as handle:
    lines = handle.readlines()
K = np.array(list(map(float, lines[0].rstrip().split())), dtype=np.float32).reshape(3, 3)
baseline = float(lines[1])
result = runner.run_pair(left, right, K_ir_left=K, baseline_m=baseline, audit_mode=True)
print('audit_stats', result['audit_stats'])
PY
```

FP32 outputs:

- `data/ffs_proof_of_life/trt_single_engine_864x480_wsl_fp32/fast_foundationstereo.onnx`
- `data/ffs_proof_of_life/trt_single_engine_864x480_wsl_fp32/fast_foundationstereo.yaml`
- `data/ffs_proof_of_life/trt_single_engine_864x480_wsl_fp32/fast_foundationstereo.engine`
- `data/ffs_proof_of_life/trt_single_engine_864x480_wsl_fp32/single_engine_build.log`
- `data/ffs_proof_of_life/trt_single_engine_864x480_wsl_fp32/demo_out/disp_vis.png`
- `data/ffs_proof_of_life/trt_single_engine_864x480_wsl_fp32/demo_out/depth_meter.npy`
- `data/ffs_proof_of_life/trt_single_engine_864x480_wsl_fp32/demo_out/cloud.ply`

FP32 results:

- build completed successfully with `fp16=False`
- engine size: about `77.7 MB`
- TensorRT reported engine generation time: about `35.8 s`
- upstream demo-pair smoke produced:
  - `finite_ratio=1.0`
  - `positive_ratio=1.0`
  - `min_disparity≈36.76`
  - `max_disparity≈123.58`

Interpretation:

- the FP32 single-engine artifact does not reproduce the earlier “right-edge stripe only” failure seen with the FP16 engine
- keeping FP32 in its own directory preserves the viewer's “exactly one `.engine`” single-engine discovery contract
