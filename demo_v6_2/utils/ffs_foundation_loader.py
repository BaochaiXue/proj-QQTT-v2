"""Official Fast-FoundationStereo module loader and Triton kernel patching.

Extracted verbatim from ``fast_foundation_stereo.py`` (behavior-preserving split).
``torch`` and the FoundationStereo ``core.*`` modules remain lazy in-method
imports resolved from the ``--ffs-repo`` checkout at runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from demo_v6_2.utils.ffs_tensorrt_infra import _ensure_ffs_repo_on_sys_path


def _disable_torch_compile(torch_module) -> None:
    def identity_compile(fn=None, *args, **kwargs):
        if fn is None:
            def decorator(inner):
                return inner
            return decorator
        return fn

    torch_module.compile = identity_compile


def _patch_batch_safe_gwc_volume_triton(foundation_stereo: Any, submodule: Any) -> None:
    import torch

    triton = getattr(submodule, "triton", None)
    if triton is None:
        raise RuntimeError("Triton is required for the two-stage FFS TensorRT export/runtime path.")
    kernel = getattr(submodule, "_gwc_triton_kernel")

    @torch.no_grad()
    def build_gwc_volume_triton_batch_safe(
        refimg_fea: Any,
        targetimg_fea: Any,
        maxdisp: int,
        num_groups: int,
        normalize: bool = True,
    ) -> Any:
        if triton is None:
            raise RuntimeError("Triton is not available. Please install triton to use build_gwc_volume_triton.")
        batch, channels, height, width = refimg_fea.shape
        assert maxdisp > 0 and channels % num_groups == 0
        group_channels = channels // num_groups
        in_dtype = refimg_fea.dtype if refimg_fea.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32

        if normalize:
            ref_norm = refimg_fea.float().reshape(batch, num_groups, group_channels, height, width).norm(dim=2)
            tar_norm = targetimg_fea.float().reshape(batch, num_groups, group_channels, height, width).norm(dim=2)
            ref_norm = ref_norm.permute(0, 2, 1, 3).reshape(batch * height, num_groups, width).to(in_dtype).contiguous()
            tar_norm = tar_norm.permute(0, 2, 1, 3).reshape(batch * height, num_groups, width).to(in_dtype).contiguous()
        else:
            ref_norm = refimg_fea.new_empty((1, 1, 1), dtype=in_dtype)
            tar_norm = refimg_fea.new_empty((1, 1, 1), dtype=in_dtype)

        ref = refimg_fea.to(in_dtype)
        tar = targetimg_fea.to(in_dtype)
        ref_bhwc = ref.permute(0, 2, 3, 1).reshape(batch * height, width, channels).contiguous()
        tar_bhwc = tar.permute(0, 2, 3, 1).reshape(batch * height, width, channels).contiguous()
        out_bhw = torch.empty((batch * height, num_groups, maxdisp, width), device=ref.device, dtype=in_dtype)
        batch_height = batch * height
        d_eff = min(maxdisp, width)
        grid = lambda meta: (
            batch_height * num_groups,
            triton.cdiv(d_eff, meta["BLOCK_D"]),
            triton.cdiv(width, meta["BLOCK_W"]),
        )
        kernel[grid](
            ref_bhwc,
            tar_bhwc,
            ref_norm,
            tar_norm,
            out_bhw,
            batch_height,
            channels,
            width,
            d_eff,
            num_groups,
            group_channels,
            ref_bhwc.stride(0),
            ref_bhwc.stride(1),
            ref_bhwc.stride(2),
            tar_bhwc.stride(0),
            tar_bhwc.stride(1),
            tar_bhwc.stride(2),
            ref_norm.stride(0),
            ref_norm.stride(1),
            ref_norm.stride(2),
            out_bhw.stride(0),
            out_bhw.stride(1),
            out_bhw.stride(2),
            out_bhw.stride(3),
            NORMALIZE=normalize,
        )
        if d_eff < maxdisp:
            out_bhw[:, :, d_eff:, :] = 0
        return out_bhw.reshape(batch, height, num_groups, maxdisp, width).permute(0, 2, 3, 1, 4).contiguous()

    submodule.build_gwc_volume_triton = build_gwc_volume_triton_batch_safe
    foundation_stereo.build_gwc_volume_triton = build_gwc_volume_triton_batch_safe


def _load_official_tensorrt_foundation_stereo(*, ffs_repo: Path, batch_safe_gwc_volume: bool = False) -> Any:
    _ensure_ffs_repo_on_sys_path(ffs_repo)
    import core.foundation_stereo as foundation_stereo
    import core.submodule as submodule

    if getattr(submodule, "triton", None) is None:
        raise RuntimeError(
            "Official Fast-FoundationStereo two-stage TensorRT requires Triton for the "
            "intermediate GWC volume kernel. Install a compatible official FFS environment "
            "or use --ffs_trt_mode single_engine."
        )
    if batch_safe_gwc_volume:
        _patch_batch_safe_gwc_volume_triton(foundation_stereo, submodule)
    else:
        foundation_stereo.build_gwc_volume_triton = submodule.build_gwc_volume_triton
    return foundation_stereo
