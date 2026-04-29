# Environments

## `ffs-standalone`

- Purpose: validate the external Fast-FoundationStereo repo in isolation
- Current validated Python:
  - `3.10`
- Current validated torch stack:
  - `torch==2.7.0+cu128`
  - `torchvision==0.22.0+cu128`
- Current WSL-validated helper packages:
  - `timm==1.0.26`
  - `scikit-image==0.25.2`
  - `PyTurboJPEG==2.2.0`
  - `gdown==6.0.0`
- Current optional TensorRT / ONNX add-on:
  - `onnx==1.21.0`
  - `tensorrt_cu12_bindings==10.16.1.11`
  - `tensorrt_cu12_libs==10.16.1.11`
- Reason for deviation from upstream README:
  - local RTX 5090 Laptop GPU is `sm_120`
  - upstream `torch==2.6.0+cu124` failed with `no kernel image is available for execution on the device`
- Validation command:
  - `conda run -n ffs-standalone python scripts/harness/verify_ffs_demo.py --ffs_repo /home/zhangxinjie/Fast-FoundationStereo --model_path /home/zhangxinjie/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth`
  - `conda run -n ffs-standalone python scripts/harness/benchmark_ffs_configs.py --help`
  - `conda run -n ffs-standalone python scripts/harness/verify_ffs_tensorrt_wsl.py`
- Expected use:
  - official FFS demo
  - checkpoint / import sanity checks
  - saved-pair checkpoint / scale / iteration tradeoff benchmarks
  - two-stage ONNX / TensorRT engine export and headless demo validation on WSL

## `ffs-official-trt`

- Purpose: run the official Fast-FoundationStereo two-stage TensorRT route with Triton GWC enabled
- Created from:
  - `conda create -y -n ffs-official-trt --clone ffs-standalone`
- Current validated Python:
  - `3.10.18`
- Current validated torch stack:
  - `torch==2.7.0+cu128`
  - `torchvision==0.22.0+cu128`
- Current validated Triton:
  - `triton==3.4.0`
- Reason for separate environment:
  - `triton==3.3.0` and `3.3.1` failed the official Fast-FoundationStereo GWC kernel with `SystemError: PY_SSIZE_T_CLEAN macro must be defined for '#' formats`
  - `triton==3.4.0` fixed the local official GWC probe while keeping the RTX 5090-compatible `torch==2.7.0+cu128` stack
- Validation commands:
  - `/home/zhangxinjie/miniconda3/envs/ffs-official-trt/bin/python cameras_viewer_FFS.py --help`
  - official GWC and static-sample two-stage TRT validation are recorded in `docs/generated/ffs_official_twostage_triton_env_validation.md`
- Expected use:
  - official-style two-stage TensorRT inference: `feature_runner.engine -> Triton GWC -> post_runner.engine`
  - live/viewer or static replay tests that specifically need official two-stage TRT latency behavior

## `FFS-SAM-RS`

- Purpose: default environment for new QQTT realtime and visualization experiments that need RealSense, Fast-FoundationStereo, TensorRT, and SAM 3.1 in the same process.
- Current validated Python:
  - `3.12.13`
- Current validated torch stack:
  - `torch==2.11.0+cu130`
  - CUDA available through `torch.version.cuda == 13.0`
- Current validated runtime packages:
  - `pyrealsense2`
  - `sam3`
  - `atomics==1.0.3`
  - `tensorrt==10.16.1.11`
  - `triton==3.6.0`
  - `open3d==0.19.0`
- Default FFS runtime policy:
  - checkpoint: `/home/zhangxinjie/Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth`
  - valid iterations: `4`
  - max disparity: `192`
  - two-stage ONNX/TensorRT artifact: `data/experiments/ffs_trt_static_rounds_848x480_pad864_builderopt5_rtx5090_laptop_20260428/engines/model_20-30-48_iters_4_res_480x864/`
  - builder optimization level: `5`
  - input policy: keep real `848x480` images, edge-pad to `864x480`, then unpad outputs
- Validation note:
  - `docs/generated/sam31_env_validation.md` records SAM 3.1 helper validation.
  - `data/experiments/ffs_trt_static_rounds_848x480_pad864_builderopt5_rtx5090_laptop_20260428/report.md` records the current level-5 TRT speed table.
- Expected use:
  - live `cameras_viewer_FFS.py` TensorRT preview
  - realtime FFS probes
  - visualization harnesses that need both SAM 3.1 masks and FFS depth
  - current operator-side experiments unless a task explicitly calls for legacy `qqtt-ffs-compat`, `ffs-official-trt`, or `FFS-max`

## `FFS-max-sam31-rs`

- Purpose: cloned `FFS-max` environment for FFS max torch/CUDA/TensorRT experiments that also need QQTT RealSense entrypoints and SAM 3.1 masks
- Created from:
  - `conda create -y -n FFS-max-sam31-rs --clone FFS-max`
- Current validated Python:
  - `3.12.13`
- Current preserved torch / CUDA stack:
  - `torch==2.11.0+cu130`
  - `torchvision==0.26.0`
  - `cuda-toolkit==13.0.2`
  - `nvidia-cuda-runtime==13.0.96`
  - `tensorrt-cu13==10.16.1.11`
  - `triton==3.6.0`
- Current RealSense / QQTT camera add-on:
  - `pyrealsense2==2.56.5.9235`
  - `atomics==1.0.3`
  - `pynput==1.8.1`
  - `threadpoolctl==3.6.0`
- Current SAM 3.1 add-on:
  - `sam3==0.1.0` installed from `git+https://github.com/facebookresearch/sam3.git@main`
  - install resolved official commit `c97c893969003d3e6803fd5d679f21e515aef5ce`
  - `QQTT_SAM31_CHECKPOINT=/home/xinjie/.cache/huggingface/qqtt_sam31/sam3.1_multiplex.pt`
- Current caveat:
  - `sam3` declares `numpy<2,>=1.26`, but this clone intentionally preserves `FFS-max`'s `numpy==2.4.4`
  - runtime validation succeeded despite that metadata mismatch; see `docs/generated/ffs_max_sam31_realsense_env_validation.md`
- Validation commands:
  - `conda run -n FFS-max-sam31-rs python -c "import torch; print(torch.__version__, torch.version.cuda)"`
  - `conda run -n FFS-max-sam31-rs python -c "import pyrealsense2 as rs; print(getattr(rs, '__version__', 'import-ok'))"`
  - `conda run -n FFS-max-sam31-rs python -c "from qqtt.env import CameraSystem; print(CameraSystem.__name__)"`
  - `conda run -n FFS-max-sam31-rs python -c "import sam3, sam3.model_builder as mb; print(getattr(sam3, '__version__', 'unknown'), hasattr(mb, 'build_sam3_video_predictor'))"`
  - `conda run -n FFS-max-sam31-rs python -m unittest -v tests.test_sam31_mask_helper_smoke`
  - `conda run -n FFS-max-sam31-rs python scripts/harness/check_all.py`
- Expected use:
  - FFS max-stack experiments that must keep `torch==2.11.0+cu130` and CUDA 13.0
  - QQTT RealSense CLI/runtime imports
  - SAM 3.1 object-mask sidecar generation from aligned cases

## `qqtt-ffs-compat`

- Purpose: run QQTT-side proof-of-life scripts that need both RealSense access and FFS-compatible Python packages
- Current validated Python:
  - `3.10`
- Current validated torch stack:
  - `torch==2.7.0+cu128`
  - `torchvision==0.22.0+cu128`
- Current WSL-validated helper packages:
  - `timm==1.0.26`
  - `scikit-image==0.25.2`
  - `PyTurboJPEG==2.2.0`
  - `gdown==6.0.0`
  - `pyrealsense2==2.56.5.9235`
- Current optional TensorRT add-on:
  - `tensorrt_cu12_bindings==10.16.1.11`
  - `tensorrt_cu12_libs==10.16.1.11`
- Current optional SAM 3 add-on:
  - `sam3==0.1.3`
  - `psutil==7.2.2`
- Current optional visualization add-on:
  - `rerun-sdk==0.31.2`
- Current WSL note:
  - the validated `rerun-sdk==0.31.2` path currently runs with a NumPy 2.x import runtime in this env
  - the current PyPI `sam3` path needed an explicit `psutil` install and an external `bpe_simple_vocab_16e6.txt.gz` vocab file for QQTT's sidecar helper
  - the local external checkpoint `sam3.1_multiplex.pt` did not load as a direct upstream `sam3` video-predictor checkpoint; see `docs/generated/sam31_env_validation.md`
- Validation commands:
  - `conda run -n qqtt-ffs-compat python scripts/harness/probe_d455_ir_pair.py --help`
  - `conda run -n qqtt-ffs-compat python scripts/harness/run_ffs_on_saved_pair.py --help`
  - `conda run -n qqtt-ffs-compat python scripts/harness/reproject_ffs_to_color.py --help`
  - `conda run -n qqtt-ffs-compat python record_data.py --help`
  - `conda run -n qqtt-ffs-compat python data_process/record_data_align.py --help`
  - `conda run -n qqtt-ffs-compat python -c "import tensorrt as trt; print(trt.__version__)"`
  - `conda run -n qqtt-ffs-compat python scripts/harness/visual_compare_rerun.py --help`
  - `conda run -n qqtt-ffs-compat python -m unittest -v tests.test_sam31_mask_helper_smoke`
  - `conda run -n qqtt-ffs-compat python scripts/harness/check_all.py`
  - `conda run -n qqtt-ffs-compat python scripts/harness/check_all.py --full`
- Expected use:
  - D455 raw IR probe
  - saved-pair FFS inference
  - geometry / scale conversion
  - integrated `stereo_ir -> ffs` camera-only workflow
  - optional live TensorRT-backed FFS viewer using prebuilt `864x480` two-stage engines with `848x480 -> 864x480` symmetric pad/unpad handling
  - optional multi-frame Rerun point-cloud diagnostics
  - fast default deterministic validation with optional full validation escalation

## WSL RealSense Raw USB Access

- Purpose: keep RSUSB-backed RealSense access ready in WSL without repeating manual `chmod`
- Current validated Linux-side install command:
  - `sudo bash env_install/install_wsl_realsense_udev.sh`
- Installed rule behavior:
  - matches `SUBSYSTEM=="usb"` and `ENV{DEVTYPE}=="usb_device"`
  - matches Intel D455 `idVendor=8086`, `idProduct=0b5c`
  - sets `GROUP=plugdev` and `MODE=0660`
- Current validated WSL prerequisite:
  - the capture user is in the `plugdev` group
- Current Windows-side prerequisite:
  - devices still need to be bound and attached with `usbipd`
  - for persistent host-side reattach, prefer `usbipd attach --wsl --auto-attach --busid <BUSID>`
- Repo-owned helper assets:
  - `env_install/99-qqtt-realsense-wsl.rules`
  - `env_install/install_wsl_realsense_udev.sh`
