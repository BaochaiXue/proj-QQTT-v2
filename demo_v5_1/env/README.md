# Demo v5 Environment Files

This folder contains the install and verification material for Demo v5. Demo v5
uses two environments:

- `demo_2_max`: main camera/fake-camera, tracking, chunk writer, and
  visualizer environment.
- `phystwin-max`: managed SAM3D shape-prior worker environment.

Use existing validated environments when they are already present:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v5/env/check_demo_v5_env.py --role main --require-cuda

conda run -n phystwin-max --no-capture-output \
  python demo_v5/env/check_demo_v5_env.py --role shape-prior --require-cuda
```

The shape-prior `--require-cuda` check validates that `nvcc` is reachable through
`CUDACXX`, `CUDA_HOME/bin/nvcc`, or `PATH`, then runs a tiny
`gsplat.rasterization` call. This intentionally warms `gsplat_cuda` before the
first SAM3D request so a missing compiler or broken JIT cache fails during
environment validation instead of surfacing as a SAM3D layout
postprocess error.

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
If the shape-prior checker reports a missing `nvcc`, rerun
`bash demo_v5/env/install_demo_v5_env.sh update` or install a CUDA toolkit whose
`bin/nvcc` matches the active PyTorch CUDA stack before running SAM3D.

The checker also verifies that repo-local runtime assets exist under
`vendor/demo_runtime/`.
