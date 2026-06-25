# Demo 4 Repo-Local Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Demo 4 and the Demo 3.2 runtime it launches default to repo-local runtime assets, then validate only Demo 4 fake-live.

**Architecture:** Add no new heavy abstraction. Replace external default paths with repo-root-relative defaults, copy external working trees and weights under `vendor/demo_runtime/`, update docs/tests that assert old paths, and run Demo 4 fake-live against the new defaults.

**Tech Stack:** Python `pathlib`, existing argparse defaults, existing Demo 4 unittest suite, shell copy commands, conda `demo_2_max`.

---

### Task 1: Copy Repo-Local Runtime Assets

**Files:**
- Create directory tree: `vendor/demo_runtime/`
- Create generated manifest: `docs/generated/demo4_repo_local_runtime_assets.md`

- [ ] **Step 1: Create destination directories**

Run:

```bash
mkdir -p vendor/demo_runtime/checkpoints/tapnextpp docs/generated
```

Expected: directories exist inside the repo.

- [ ] **Step 2: Copy external runtime trees**

Run:

```bash
rsync -a --delete --exclude .git --exclude __pycache__ --exclude '*.pyc' --exclude wandb --exclude output --exclude outputs /home/xinjie/external/sam-3d-objects/ vendor/demo_runtime/sam-3d-objects/
rsync -a --delete --exclude .git --exclude __pycache__ --exclude '*.pyc' --exclude wandb --exclude output --exclude outputs /home/xinjie/FuturePhysTwin/ vendor/demo_runtime/FuturePhysTwin/
rsync -a --delete --exclude .git --exclude __pycache__ --exclude '*.pyc' --exclude output --exclude outputs /home/xinjie/Fast-FoundationStereo/ vendor/demo_runtime/Fast-FoundationStereo/
rsync -a --delete --exclude .git --exclude __pycache__ --exclude '*.pyc' /home/xinjie/proj-QQTT-v2/external/tapnet/ vendor/demo_runtime/tapnet/
```

Expected: destination trees contain code and weights but no `.git` directories.

- [ ] **Step 3: Copy TAPNext++ checkpoint**

Run:

```bash
cp -a /home/xinjie/proj-QQTT-v2/checkpoints/tapnextpp/tapnextpp_ckpt.pt vendor/demo_runtime/checkpoints/tapnextpp/tapnextpp_ckpt.pt
```

Expected: checkpoint file exists at the repo-local path.

- [ ] **Step 4: Record copied asset state**

Run:

```bash
du -sh vendor/demo_runtime/sam-3d-objects vendor/demo_runtime/FuturePhysTwin vendor/demo_runtime/Fast-FoundationStereo vendor/demo_runtime/tapnet vendor/demo_runtime/checkpoints/tapnextpp/tapnextpp_ckpt.pt
```

Expected: sizes are recorded in `docs/generated/demo4_repo_local_runtime_assets.md`.

### Task 2: Refactor Default Paths

**Files:**
- Modify: `services/shape_prior_remote/server.py`
- Modify: `demo_v4/realtime_futurephystwin_chunks.py`
- Modify: `data_process/depth_backends/ffs_defaults.py`
- Modify: `qqtt/demo/realtime_masked_edgetam_pcd.py`

- [ ] **Step 1: Update shape-prior worker defaults**

Change:

```python
DEFAULT_SAM3D_ROOT = REPO_ROOT / "vendor" / "demo_runtime" / "sam-3d-objects"
DEFAULT_FUTUREPHYSTWIN_ROOT = REPO_ROOT / "vendor" / "demo_runtime" / "FuturePhysTwin"
```

- [ ] **Step 2: Update Demo 4 output default**

Change:

```python
DEFAULT_FUTUREPHYSTWIN_BASE_PATH = Path("result/demo_v4/futurephystwin_chunks")
```

- [ ] **Step 3: Update FFS defaults**

Change:

```python
DEFAULT_FFS_ENV_PYTHON = Path("python")
DEFAULT_FFS_REPO = REPO_ROOT / "vendor" / "demo_runtime" / "Fast-FoundationStereo"
DEFAULT_FFS_MODEL_PATH = DEFAULT_FFS_REPO / "weights" / DEFAULT_FFS_MODEL_NAME / "model_best_bp2_serialize.pth"
```

- [ ] **Step 4: Update TAPNext++ defaults**

Change:

```python
DEFAULT_TAPNET_REPO_DIR = REPO_ROOT / "vendor" / "demo_runtime" / "tapnet"
DEFAULT_TAPNEXTPP_CHECKPOINT = (
    REPO_ROOT / "vendor" / "demo_runtime" / "checkpoints" / "tapnextpp" / "tapnextpp_ckpt.pt"
)
```

### Task 3: Update Tests and Docs

**Files:**
- Modify: `tests/test_demo_v4_futurephystwin_chunks.py`
- Modify: `tests/test_demo32_shape_prior_warmup.py`
- Modify: `demo_v4/README.md`
- Modify: `demo_v3_2/README.md`
- Modify: `docs/external-deps.md`
- Modify: `docs/HARDWARE_VALIDATION.md`

- [ ] **Step 1: Update tests that assert old absolute defaults**

Replace `/home/xinjie/FuturePhysTwin/data/demo_v4_chunks` expectations with
`result/demo_v4/futurephystwin_chunks`.

- [ ] **Step 2: Add path locality assertions**

Add assertions that Demo 4 default base path is not absolute and that shape
prior worker defaults are under `vendor/demo_runtime`.

- [ ] **Step 3: Update operator docs**

Document the worker command without absolute `--sam3d-root` or
`--futurephystwin-root` overrides and document the new repo-local asset root.

### Task 4: Verify and Run Demo 4 Fake-Live

**Files:**
- Generated output: `result/demo_v4/futurephystwin_chunks/`
- Generated proof: `docs/generated/demo4_repo_local_runtime_validation.md`

- [ ] **Step 1: Run focused tests**

Run:

```bash
conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v4_futurephystwin_chunks tests.test_demo32_shape_prior_warmup
```

Expected: tests pass or failures are fixed before continuing.

- [ ] **Step 2: Run only Demo 4 fake-live final validation**

Run a short fake-live Demo 4 command using default repo-local paths:

```bash
conda run -n demo_2_max --no-capture-output python demo_v4/realtime_futurephystwin_chunks.py --input-source fake-live --replay-fps 5 --chunk-seconds 5 --max-chunks 1 --capture-extra-seconds 20 --shape-prior-timeout-s 5
```

Expected: Demo 4 starts and writes/streams at least one fake-live chunk or
reaches a documented dependency/runtime error that is not caused by an
absolute-path default.

- [ ] **Step 3: Record validation**

Write command, exit status, and key output paths to
`docs/generated/demo4_repo_local_runtime_validation.md`.

### Task 5: Commit

**Files:**
- All modified source, tests, docs, and generated proof files.

- [ ] **Step 1: Review status**

Run:

```bash
git status --short --untracked-files=all
```

Expected: source/docs/test files are staged intentionally; large runtime assets
may remain untracked/ignored but must exist on disk.

- [ ] **Step 2: Commit validated changes**

Run:

```bash
git add docs/superpowers/specs/2026-06-24-demo4-repo-local-runtime-design.md docs/superpowers/plans/2026-06-24-demo4-repo-local-runtime.md docs/exec-plans/active/2026-06-24-demo4-repo-local-runtime.md services/shape_prior_remote/server.py demo_v4/realtime_futurephystwin_chunks.py data_process/depth_backends/ffs_defaults.py qqtt/demo/realtime_masked_edgetam_pcd.py tests/test_demo_v4_futurephystwin_chunks.py tests/test_demo32_shape_prior_warmup.py demo_v4/README.md demo_v3_2/README.md docs/external-deps.md docs/HARDWARE_VALIDATION.md docs/generated/demo4_repo_local_runtime_assets.md docs/generated/demo4_repo_local_runtime_validation.md
git commit -m "localize demo v4 runtime assets"
```

Expected: commit succeeds on `single-camera`.
