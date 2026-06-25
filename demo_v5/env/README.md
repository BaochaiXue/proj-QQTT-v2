# Demo v5 Environment Files

This folder contains the install and verification material for Demo v5. Demo v5
uses two environments:

- `demo_2_max`: main camera/fake-camera, tracking, chunk writer, and
  `realtime_phystwin` optimization environment.
- `phystwin-max`: managed SAM3D shape-prior worker environment.

Use existing validated environments when they are already present:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v5/env/check_demo_v5_env.py --role main --require-cuda

conda run -n phystwin-max --no-capture-output \
  python demo_v5/env/check_demo_v5_env.py --role shape-prior --require-cuda
```

Create environments on a new machine:

```bash
bash demo_v5/env/install_demo_v5_env.sh create
```

Or run the steps manually:

```bash
conda env create -f demo_v5/env/environment-demo-v5-main.yml
conda run -n demo_2_max --no-capture-output \
  python -m pip install -r demo_v5/env/requirements-demo-v5-main.txt

conda env create -f demo_v5/env/environment-demo-v5-shape-prior.yml
conda run -n phystwin-max --no-capture-output \
  python -m pip install -r demo_v5/env/requirements-demo-v5-shape-prior.txt
```

Update existing environments:

```bash
bash demo_v5/env/install_demo_v5_env.sh update
```

Or run the steps manually:

```bash
conda env update -f demo_v5/env/environment-demo-v5-main.yml --prune
conda run -n demo_2_max --no-capture-output \
  python -m pip install -r demo_v5/env/requirements-demo-v5-main.txt

conda env update -f demo_v5/env/environment-demo-v5-shape-prior.yml --prune
conda run -n phystwin-max --no-capture-output \
  python -m pip install -r demo_v5/env/requirements-demo-v5-shape-prior.txt
```

GPU wheels are CUDA-stack-sensitive. If PyTorch, PyTorch3D, Kaolin, Warp, or
Open3D fail to install from these requirements on a new machine, install the
wheel matching that machine's CUDA stack first, then rerun the checker. The
validated versions are recorded in `validated-versions-20260625.txt`.

The checker also verifies that repo-local runtime assets exist under
`vendor/demo_runtime/` and that `realtime_phystwin/train_online_zero_then_first.py`
is available.
