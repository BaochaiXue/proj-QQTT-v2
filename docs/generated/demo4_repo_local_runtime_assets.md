# Demo 4 Repo-Local Runtime Assets

Generated: 2026-06-24

## Copied Asset Roots

```text
225M  vendor/demo_runtime/sam-3d-objects
8.8G  vendor/demo_runtime/FuturePhysTwin
707M  vendor/demo_runtime/Fast-FoundationStereo
4.1M  vendor/demo_runtime/tapnet
54M   vendor/demo_runtime/EdgeTAM-hf
4.8M  vendor/demo_runtime/dinov2
22G   vendor/demo_runtime/checkpoints
```

Sizes above exclude large model bytes moved into the central checkpoints tree.

## Source Paths

```text
/home/xinjie/external/sam-3d-objects
/home/xinjie/FuturePhysTwin
/home/xinjie/Fast-FoundationStereo
/home/xinjie/proj-QQTT-v2/external/tapnet
/home/xinjie/.cache/huggingface/hub/models--yonigozlan--EdgeTAM-hf/snapshots/c266ce53b3fc00f0f495b583f6a116c4e57f53bb
/home/xinjie/.cache/huggingface/hub/models--stabilityai--stable-diffusion-x4-upscaler/snapshots/572c99286543a273bfd17fac263db5a77be12c4c
/home/xinjie/.cache/huggingface/hub/models--Ruicheng--moge-vitl/snapshots/979e84da9415762c30e6c0cf8dc0962896c793df
/home/xinjie/.cache/torch/hub/facebookresearch_dinov2_main
/home/xinjie/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth
/home/xinjie/proj-QQTT-v2/checkpoints/tapnextpp/tapnextpp_ckpt.pt
```

## Copy Policy Used

Runtime working trees were copied into `vendor/demo_runtime/` with `.git`,
`__pycache__`, `*.pyc`, and generated output/log directories excluded. No
external symlinks were used for these runtime asset roots.

All model weight/cache files at or above 100 MB are stored under
`vendor/demo_runtime/checkpoints/`. Runtime source-tree locations that upstream
loaders expect are preserved as repo-local relative symlinks back into that
central checkpoints tree.

Centralized large weight/cache files:

```text
vendor/demo_runtime/checkpoints/sam3d/hf/ss_generator.ckpt
vendor/demo_runtime/checkpoints/sam3d/hf/slat_generator.ckpt
vendor/demo_runtime/checkpoints/sam3d/hf/slat_decoder_mesh.pt
vendor/demo_runtime/checkpoints/sam3d/hf/slat_decoder_mesh.ckpt
vendor/demo_runtime/checkpoints/sam3d/hf/slat_decoder_gs.ckpt
vendor/demo_runtime/checkpoints/sam3d/hf/slat_decoder_gs_4.ckpt
vendor/demo_runtime/checkpoints/sam3d/hf/ss_decoder.ckpt
vendor/demo_runtime/checkpoints/stable-diffusion-x4-upscaler/text_encoder/model.safetensors
vendor/demo_runtime/checkpoints/stable-diffusion-x4-upscaler/unet/diffusion_pytorch_model.safetensors
vendor/demo_runtime/checkpoints/stable-diffusion-x4-upscaler/vae/diffusion_pytorch_model.safetensors
vendor/demo_runtime/checkpoints/MoGe-vitl/model.pt
vendor/demo_runtime/checkpoints/dinov2/dinov2_vitl14_reg4_pretrain.pth
vendor/demo_runtime/checkpoints/FuturePhysTwin/groundedSAM_checkpoints/sam2.1_hiera_large.pt
vendor/demo_runtime/checkpoints/FuturePhysTwin/groundedSAM_checkpoints/groundingdino_swint_ogc.pth
vendor/demo_runtime/checkpoints/Fast-FoundationStereo/weights/onnx/20_26_39/576x960/20_26_39_iters_8_res_576x960.onnx
vendor/demo_runtime/checkpoints/tapnextpp/tapnextpp_ckpt.pt
```

## Presence Checks

```text
vendor/demo_runtime/sam-3d-objects/checkpoints/hf/pipeline.yaml: present
vendor/demo_runtime/Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth: present
vendor/demo_runtime/EdgeTAM-hf/processor_config.json: present
vendor/demo_runtime/EdgeTAM-hf/model.safetensors: present
vendor/demo_runtime/stable-diffusion-x4-upscaler/model_index.json: present
vendor/demo_runtime/MoGe-vitl/model.pt: repo-local symlink into checkpoints
vendor/demo_runtime/dinov2/hubconf.py: present
vendor/demo_runtime/dinov2/dinov2_vitl14_reg4_pretrain.pth: repo-local symlink into checkpoints
vendor/demo_runtime/checkpoints/tapnextpp/tapnextpp_ckpt.pt: present
vendor/demo_runtime/checkpoints contains every >=100 MB model weight/cache file
vendor/demo_runtime/**/.git: absent
vendor/demo_runtime/**/__pycache__: absent at maxdepth 4
```
