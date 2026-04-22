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
- Reason for deviation from upstream README:
  - local RTX 5090 Laptop GPU is `sm_120`
  - upstream `torch==2.6.0+cu124` failed with `no kernel image is available for execution on the device`
- Validation command:
  - `conda run -n ffs-standalone python scripts/harness/verify_ffs_demo.py --ffs_repo /home/zhangxinjie/Fast-FoundationStereo --model_path /home/zhangxinjie/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth`
  - `conda run -n ffs-standalone python scripts/harness/benchmark_ffs_configs.py --help`
- Expected use:
  - official FFS demo
  - checkpoint / import sanity checks
  - saved-pair checkpoint / scale / iteration tradeoff benchmarks

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
  - `tensorrt-cu12==10.16.1.11`
- Current optional visualization add-on:
  - `rerun-sdk==0.31.2`
- Current WSL note:
  - the validated `rerun-sdk==0.31.2` path currently runs with a NumPy 2.x import runtime in this env
- Validation commands:
  - `conda run -n qqtt-ffs-compat python scripts/harness/probe_d455_ir_pair.py --help`
  - `conda run -n qqtt-ffs-compat python scripts/harness/run_ffs_on_saved_pair.py --help`
  - `conda run -n qqtt-ffs-compat python scripts/harness/reproject_ffs_to_color.py --help`
  - `conda run -n qqtt-ffs-compat python record_data.py --help`
  - `conda run -n qqtt-ffs-compat python data_process/record_data_align.py --help`
  - `conda run -n qqtt-ffs-compat python scripts/harness/visual_compare_rerun.py --help`
  - `conda run -n qqtt-ffs-compat python scripts/harness/check_all.py`
- Expected use:
  - D455 raw IR probe
  - saved-pair FFS inference
  - geometry / scale conversion
  - integrated `stereo_ir -> ffs` camera-only workflow
  - optional live TensorRT-backed FFS viewer using prebuilt two-stage engines
  - optional multi-frame Rerun point-cloud diagnostics
