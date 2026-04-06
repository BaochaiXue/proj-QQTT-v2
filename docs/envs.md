# Environments

## `ffs-standalone`

- Purpose: validate the external Fast-FoundationStereo repo in isolation
- Current validated torch stack:
  - `torch==2.7.0+cu128`
  - `torchvision==0.22.0+cu128`
- Reason for deviation from upstream README:
  - local RTX 5090 Laptop GPU is `sm_120`
  - upstream `torch==2.6.0+cu124` failed with `no kernel image is available for execution on the device`
- Validation command:
  - `conda run -n ffs-standalone python scripts/harness/verify_ffs_demo.py`
- Expected use:
  - official FFS demo
  - checkpoint / import sanity checks

## `qqtt-ffs-compat`

- Purpose: run QQTT-side proof-of-life scripts that need both RealSense access and FFS-compatible Python packages
- Current validated torch stack:
  - `torch==2.7.0+cu128`
  - `torchvision==0.22.0+cu128`
- Validation commands:
  - `conda run -n qqtt-ffs-compat python scripts/harness/probe_d455_ir_pair.py --help`
  - `conda run -n qqtt-ffs-compat python scripts/harness/run_ffs_on_saved_pair.py --help`
  - `conda run -n qqtt-ffs-compat python scripts/harness/reproject_ffs_to_color.py --help`
  - `conda run -n qqtt-ffs-compat python record_data.py --help`
  - `conda run -n qqtt-ffs-compat python data_process/record_data_align.py --help`
- Expected use:
  - D455 raw IR probe
  - saved-pair FFS inference
  - geometry / scale conversion
  - integrated `stereo_ir -> ffs` camera-only workflow
