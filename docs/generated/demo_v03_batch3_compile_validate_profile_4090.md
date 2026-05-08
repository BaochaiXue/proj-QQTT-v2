# Demo v0.3 Batch=3 FFS TensorRT Compile / Validate / Profile on RTX 4090

## Environment

- Machine: Native Ubuntu dual RTX 4090
- GPU used for successful build: RTX 4090 logical GPU 0
- Driver: 570.211.01
- CUDA reported by NVIDIA-SMI: 12.8
- CUDA_HOME: `/usr/local/cuda`
- TORCH_CUDA_ARCH_LIST: `8.9`
- torch: `2.11.0+cu128`
- torch CUDA: `12.8`
- TensorRT: `10.16.1.11`
- Existing services left untouched:
  - `7001` Demo 2 server
  - `7002` Demo v0.2 async server
  - `5201` iperf3 server

## Engine Targets

- Batch1 engine: `/home/xinjie/proj-QQTT-v2/data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864`
- Batch3 engine: `data/experiments/ffs_trt_4090_848x480_pad864_builderopt5_batch3/engines/model_20-30-48_iters_4_res_480x864_batch3`
- Model: `20-30-48`
- valid_iters: `4`
- Engine input: `480x864`
- Capture input: `480x848`
- builderOptimizationLevel: `5`
- max_disp: `192`
- batch_size: `3`

## 100-Kit Replay Data

- Expected path: `result/demo_v0_3_ir_triplet_100kits_848x480`
- Backing target: `/home/xinjie/proj-QQTT-v2/result/demo_v0_3_ir_triplet_100kits_848x480`
- Status: missing on this 4090 machine
- warmup_kits: `20`
- measure_kits: `100`
- warmup included in stats: no

Because the 100-kit replay folder is absent, the validation/profile scripts were implemented and unit-tested but not run to completion on real replay data.

## Implementation

Files added:

- `scripts/ffs_trt/build_batch3_4090_engine.py`
- `scripts/ffs_trt/validate_batch3_4090_engine.py`
- `scripts/ffs_trt/profile_batch3_4090_engine.py`
- `tests/test_ffs_trt_batch3_scripts.py`

Shared/default batch1 behavior changed: no.

Batch3-specific compile notes:

- The external Fast-FoundationStereo repo is not modified.
- The build script patches the upstream Triton GWC helper only inside the build process to replace non-contiguous `.view(...)` use with `.reshape(...)` for batch > 1 export.
- The build script exports ONNX with the legacy TorchScript ONNX exporter (`dynamo=False`) to avoid a PyTorch 2.11 symbol assertion in the post runner.
- The batch3 post ONNX is rewritten to replace two static 1x2 `AveragePool` nodes (`/AveragePool`, `/AveragePool_1`) with equivalent `Reshape + ReduceMean(axis=4)`. This avoids TensorRT 10.16 tactic failure for the large batch3 effective pooling dimension.

## Build

Successful command:

```bash
CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.9 CUDA_HOME=/usr/local/cuda \
conda run --no-capture-output -n demo_2_max \
  python scripts/ffs_trt/build_batch3_4090_engine.py \
  --ffs-repo /home/xinjie/Fast-FoundationStereo \
  --weight /home/xinjie/Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth \
  --out-dir data/experiments/ffs_trt_4090_848x480_pad864_builderopt5_batch3/engines/model_20-30-48_iters_4_res_480x864_batch3 \
  --model 20-30-48 \
  --valid-iters 4 \
  --height 480 \
  --width 864 \
  --capture-height 480 \
  --capture-width 848 \
  --builder-optimization-level 5 \
  --max-disp 192 \
  --batch-size 3 \
  --timing-cache data/experiments/ffs_trt_4090_848x480_pad864_builderopt5_batch3/timing_cache.bin \
  --debug
```

Result: pass.

Static batch check:

```text
mode single_engine not_applicable_or_failed ValueError(...)
mode two_stage batch_size 3
```

Build metadata:

- `feature_runner.engine`: built, static input batch `3`
- `post_runner.engine`: built, static input/output batch `3`
- feature build time: `18.66 s`
- post build time: `706.91 s`
- metadata: `data/experiments/ffs_trt_4090_848x480_pad864_builderopt5_batch3/engines/model_20-30-48_iters_4_res_480x864_batch3/batch3_metadata.json`

Earlier build attempts:

- GPU1 attempt failed because the running 7002 server occupied GPU1 memory and TensorRT skipped large tactics before failing the post runner.
- GPU0 attempt without ONNX pooling rewrite also failed at `/AveragePool_1`.
- GPU0 attempt with the pooling rewrite passed.

## Validate

Command implemented:

```bash
python scripts/ffs_trt/validate_batch3_4090_engine.py \
  --batch1-model-dir /home/xinjie/proj-QQTT-v2/data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864 \
  --batch3-model-dir data/experiments/ffs_trt_4090_848x480_pad864_builderopt5_batch3/engines/model_20-30-48_iters_4_res_480x864_batch3 \
  --replay-dir result/demo_v0_3_ir_triplet_100kits_848x480 \
  --warmup-kits 20 \
  --measure-kits 100 \
  --depth-scale-m-per-unit 0.001 \
  --debug
```

Result: not run to completion.

Reason: the required 100-kit replay folder is missing on this 4090 machine.

## Profile

Command implemented:

```bash
python scripts/ffs_trt/profile_batch3_4090_engine.py \
  --batch1-model-dir /home/xinjie/proj-QQTT-v2/data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864 \
  --batch3-model-dir data/experiments/ffs_trt_4090_848x480_pad864_builderopt5_batch3/engines/model_20-30-48_iters_4_res_480x864_batch3 \
  --replay-dir result/demo_v0_3_ir_triplet_100kits_848x480 \
  --warmup-kits 20 \
  --measure-kits 100 \
  --debug
```

Batch1-only profile result: not run to completion because the required replay folder is missing.

Batch3 profile result: not run to completion because the required replay folder is missing.

## Deterministic Checks

Passed:

```bash
conda run --no-capture-output -n demo_2_max python -m py_compile \
  scripts/ffs_trt/build_batch3_4090_engine.py \
  scripts/ffs_trt/validate_batch3_4090_engine.py \
  scripts/ffs_trt/profile_batch3_4090_engine.py

conda run --no-capture-output -n demo_2_max \
  python -m unittest -v tests.test_ffs_trt_batch3_scripts

conda run --no-capture-output -n demo_2_max \
  python scripts/harness/check_all.py
```

## Decision

- batch3 compile code written: pass
- batch1 behavior unchanged: yes
- 100-kit replay folder found: no
- batch3 build: pass
- batch3 validate 100-kit: not run, missing replay data
- batch3 profile 100-kit: not run, missing replay data
- batch3 usable for server: not yet

Reason: the batch=3 TensorRT engine now compiles and reports static batch `3`, but it must still pass 100-kit real-IR validation and profiling before replacing or augmenting the server path.

Recommended server mode for now: keep existing batch=1 sequential/staged server paths until 100-kit validation/profile data is available and passes.
