# FFS TensorRT Windows Validation

- Date: `2026-04-20`
- Machine: `XinjieZhang`
- GPU: `NVIDIA GeForce RTX 5090 Laptop GPU`
- Driver: `591.90`
- GPU memory: `24463 MiB`
- Environment: `ffs-standalone`
- Python / Torch / ONNX / TensorRT: `3.12.13` / `2.7.0+cu128` / `1.21.0` / `10.16.1.11`
- Scope note: this validates the external Fast-FoundationStereo two-stage ONNX / TensorRT route on Windows only. QQTT production depth-backend behavior was not changed.

## Commands

Python-side dependency install:

```text
conda run -n ffs-standalone python -m pip install --upgrade onnx tensorrt-cu12==10.16.1.11
```

TensorRT Windows SDK download:

```text
curl.exe -L https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.16.1/zip/TensorRT-10.16.1.11.Windows.amd64.cuda-12.9.zip -o C:\Users\zhang\external\TensorRT-10.16.1.11.Windows.amd64.cuda-12.9.zip
Expand-Archive -Path C:\Users\zhang\external\TensorRT-10.16.1.11.Windows.amd64.cuda-12.9.zip -DestinationPath C:\Users\zhang\external
```

`trtexec` smoke:

```text
C:\Users\zhang\external\TensorRT-10.16.1.11\bin\trtexec.exe --help
```

End-to-end proof-of-life:

```text
C:\Users\zhang\miniconda3\envs\ffs-standalone\python.exe scripts\harness\verify_ffs_tensorrt_windows.py
```

## Important Windows Notes

- The local machine did **not** need an additional CUDA Toolkit install for this proof-of-life. The official TensorRT Windows zip plus the Python package were sufficient on this setup.
- Upstream Fast-FoundationStereo TensorRT utilities assume Triton is available for host-side GWC volume construction.
- The repo-local harness works around that Windows gap by:
  - disabling `torch.compile`
  - replacing Triton-only GWC volume calls with the existing PyTorch implementation for ONNX export and TensorRT runtime host preprocessing
- The TensorRT engines themselves are still the standard upstream two-stage engines built by `trtexec`.

## Repo Follow-Up Note For The QQTT Viewer

The proof-of-life in this note remains the validated `640x480` export and engine build. For the current QQTT viewer default capture profile, the intended follow-up engine build target is now:

- capture: `848x480`
- engine: `864x480`
- rationale: upstream `scripts/make_onnx.py` requires dimensions divisible by `32`, so `848x480` cannot be used as the two-stage engine shape directly
- runtime behavior in QQTT: symmetrically replicate-pad the `848x480` IR pair to `864x480`, run TRT, then crop the disparity back to `848x480` before reprojection

That newer `848 -> 864 -> 848` repo path was not part of this original Windows validation run.

## Outputs

- `data\ffs_proof_of_life\trt_two_stage_640x480\feature_runner.onnx`
- `data\ffs_proof_of_life\trt_two_stage_640x480\post_runner.onnx`
- `data\ffs_proof_of_life\trt_two_stage_640x480\onnx.yaml`
- `data\ffs_proof_of_life\trt_two_stage_640x480\feature_runner.engine`
- `data\ffs_proof_of_life\trt_two_stage_640x480\post_runner.engine`
- `data\ffs_proof_of_life\trt_two_stage_640x480\feature_trtexec.log`
- `data\ffs_proof_of_life\trt_two_stage_640x480\post_trtexec.log`
- `data\ffs_proof_of_life\trt_two_stage_640x480\demo_out\disp_vis.png`
- `data\ffs_proof_of_life\trt_two_stage_640x480\demo_out\depth_meter.npy`
- `data\ffs_proof_of_life\trt_two_stage_640x480\demo_out\cloud.ply`

## Results

Dependency verification:

- `import tensorrt` succeeded
- `trt.Builder(trt.Logger())` succeeded
- `trtexec --help` succeeded from the extracted official TensorRT zip

Engine build:

- `feature_runner.engine` built successfully
- `post_runner.engine` built successfully
- `trtexec` detected local compute capability `12.0`

Headless TRT demo:

- completed successfully against the upstream demo pair
- wrote `disp_vis.png`, `depth_meter.npy`, and `cloud.ply`
- resized the upstream `960x540` demo pair to the fixed-engine `640x480` shape recorded in `onnx.yaml`

Latency comparison at `640x480`, `valid_iters=4`, `max_disp=192`:

- TensorRT average after warmup: `40.4 ms`
- PyTorch average after warmup: `66.0 ms`
- Observed speedup: about `1.63x`

## Follow-Up Boundary

- This validation does not cover QQTT `record_data_align.py`, `cameras_viewer_FFS.py`, or any live 3-camera TRT integration.
- This validation also does not cover the newer single-ONNX / single-engine upstream workflow.
- This validation also does not cover the later QQTT-specific `848x480` capture with `864x480` engine pad/unpad path.
