# Demo 4 Repo-Local Runtime Assets

Generated: 2026-06-24

## Copied Asset Roots

```text
13G   vendor/demo_runtime/sam-3d-objects
11G   vendor/demo_runtime/FuturePhysTwin
807M  vendor/demo_runtime/Fast-FoundationStereo
4.1M  vendor/demo_runtime/tapnet
54M   vendor/demo_runtime/EdgeTAM-hf
2.4G  vendor/demo_runtime/checkpoints/tapnextpp/tapnextpp_ckpt.pt
```

## Source Paths

```text
/home/xinjie/external/sam-3d-objects
/home/xinjie/FuturePhysTwin
/home/xinjie/Fast-FoundationStereo
/home/xinjie/proj-QQTT-v2/external/tapnet
/home/xinjie/.cache/huggingface/hub/models--yonigozlan--EdgeTAM-hf/snapshots/c266ce53b3fc00f0f495b583f6a116c4e57f53bb
/home/xinjie/proj-QQTT-v2/checkpoints/tapnextpp/tapnextpp_ckpt.pt
```

## Copy Policy Used

Runtime working trees were copied into `vendor/demo_runtime/` with `.git`,
`__pycache__`, `*.pyc`, and generated output/log directories excluded. No
symlinks were used for these runtime asset roots.

## Presence Checks

```text
vendor/demo_runtime/sam-3d-objects/checkpoints/hf/pipeline.yaml: present
vendor/demo_runtime/Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth: present
vendor/demo_runtime/EdgeTAM-hf/processor_config.json: present
vendor/demo_runtime/EdgeTAM-hf/model.safetensors: present
vendor/demo_runtime/checkpoints/tapnextpp/tapnextpp_ckpt.pt: present
vendor/demo_runtime/**/.git: absent
vendor/demo_runtime/**/__pycache__: absent at maxdepth 4
```
