# Fast-FoundationStereo Demo Validation

- Environment: `ffs-standalone`
- Execution mode: `conda-run`
- Exit code: `0`
- Output directory: `C:\Users\zhang\proj-QQTT\data\ffs_proof_of_life\official_demo`

## Command

```text
conda run -n ffs-standalone python C:\Users\zhang\AppData\Local\Temp\tmpc28ou60n\run_demo_wrapper.py C:\Users\zhang\external\Fast-FoundationStereo\scripts\run_demo.py --model_dir C:\Users\zhang\external\Fast-FoundationStereo\weights\23-36-37\model_best_bp2_serialize.pth --left_file C:\Users\zhang\external\Fast-FoundationStereo\demo_data\left.png --right_file C:\Users\zhang\external\Fast-FoundationStereo\demo_data\right.png --intrinsic_file C:\Users\zhang\external\Fast-FoundationStereo\demo_data\K.txt --out_dir C:\Users\zhang\proj-QQTT\data\ffs_proof_of_life\official_demo --scale 1.0 --valid_iters 8 --max_disp 192 --get_pc 1 --remove_invisible 0 --denoise_cloud 0
```

## Verified Outputs

- `disp_vis.png`
- `depth_meter.npy`
- `cloud.ply`

## Stdout

```text

```

## Stderr

```text
args:
{'corr_levels': 2, 'corr_radius': 4, 'hidden_dims': [128], 'low_memory': 0, 'max_disp': 192, 'mixed_precision': True, 'n_downsample': 2, 'n_gru_layers': 1, 'slow_fast_gru': False, 'valid_iters': 8, 'vit_size': 'vitl', 'model_dir': 'C:\\Users\\zhang\\external\\Fast-FoundationStereo\\weights\\23-36-37\\model_best_bp2_serialize.pth', 'left_file': 'C:\\Users\\zhang\\external\\Fast-FoundationStereo\\demo_data\\left.png', 'right_file': 'C:\\Users\\zhang\\external\\Fast-FoundationStereo\\demo_data\\right.png', 'intrinsic_file': 'C:\\Users\\zhang\\external\\Fast-FoundationStereo\\demo_data\\K.txt', 'out_dir': 'C:\\Users\\zhang\\proj-QQTT\\data\\ffs_proof_of_life\\official_demo', 'remove_invisible': 0, 'denoise_cloud': 0, 'denoise_nb_points': 30, 'denoise_radius': 0.03, 'scale': 1.0, 'hiera': 0, 'get_pc': 1, 'zfar': 100}
compile_threads set to 1 for win32
C:\Users\zhang\external\Fast-FoundationStereo\scripts\run_demo.py:64: DeprecationWarning: Starting with ImageIO v3 the behavior of this function will switch to that of iio.v3.imread. To keep the current behavior (and make this warning disappear) use `import imageio.v2 as imageio` or call `imageio.v2.imread` directly.
  img0 = imageio.imread(args.left_file)
C:\Users\zhang\external\Fast-FoundationStereo\scripts\run_demo.py:65: DeprecationWarning: Starting with ImageIO v3 the behavior of this function will switch to that of iio.v3.imread. To keep the current behavior (and make this warning disappear) use `import imageio.v2 as imageio` or call `imageio.v2.imread` directly.
  img1 = imageio.imread(args.right_file)
img0: (540, 960, 3)
Start forward, 1st time run can be slow due to compilation
C:\Users\zhang\external\Fast-FoundationStereo\scripts/..\core\geometry.py:76: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=False):
forward done
PCL saved to C:\Users\zhang\proj-QQTT\data\ffs_proof_of_life\official_demo
Visualizing point cloud. Press ESC to exit.
Cache Metrics: None

TorchDynamo attempted to trace the following frames: [

]
TorchDynamo compilation metrics:
Function, Runtimes (s)
```
