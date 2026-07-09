"""Local Fast-FoundationStereo TensorRT depth runner (vendored for demo_v6_1).

Vendored from ``data_process/depth_backends/fast_foundation_stereo.py`` so
demo_v6_1 is self-contained. The heavy TensorRT / Torch / FoundationStereo
(``core.*``, ``Utils``) dependencies remain lazy in-method imports resolved
from the ``--ffs-repo`` checkout at runtime; only ``--depth-source ffs`` uses
this module.

This module is now a thin re-export shell. The implementation was split into
focused submodules (behavior-preserving, pure relocation); the import path
``demo_v6_2.utils.fast_foundation_stereo`` is preserved so existing callers keep
working unchanged:

- ``ffs_disparity_products``   pure-numpy disparity/depth/confidence products.
- ``ffs_tensorrt_infra``       TensorRT runtime plumbing, image transforms, and
                               pinned input staging buffers.
- ``ffs_foundation_loader``    official FoundationStereo module loader + Triton patch.
- ``ffs_runner_two_stage``     the live ``FastFoundationStereoTensorRTRunner``.
"""

from __future__ import annotations

from demo_v6_2.utils.ffs_disparity_products import (
    build_disparity_products,
    compute_disparity_audit_stats,
    finalize_single_engine_tensorrt_output,
    finalize_tensorrt_disparity_batch_outputs,
    split_disparity_batch_output_maps,
    undo_tensorrt_disparity_transform,
)
from demo_v6_2.utils.ffs_foundation_loader import (
    _disable_torch_compile,
    _load_official_tensorrt_foundation_stereo,
    _patch_batch_safe_gwc_volume_triton,
)
from demo_v6_2.utils.ffs_runner_two_stage import FastFoundationStereoTensorRTRunner
from demo_v6_2.utils.ffs_tensorrt_infra import (
    FFS_INPUT_STAGING_MODES,
    FFS_INPUT_STAGING_PAGEABLE,
    FFS_INPUT_STAGING_PINNED,
    _CachedTensorRTRun,
    _configure_tensorrt_runtime_search_paths,
    _ensure_ffs_repo_on_sys_path,
    _PinnedBatchPairImageInputBuffers,
    _PinnedSinglePairImageInputBuffers,
    apply_tensorrt_image_transform,
    load_tensorrt_model_config,
    resolve_single_engine_tensorrt_model_path,
    resolve_tensorrt_engine_static_batch_size,
    resolve_tensorrt_image_transform,
    resolve_tensorrt_model_config_path,
)

__all__ = [
    "FFS_INPUT_STAGING_MODES",
    "FFS_INPUT_STAGING_PAGEABLE",
    "FFS_INPUT_STAGING_PINNED",
    "FastFoundationStereoTensorRTRunner",
    "apply_tensorrt_image_transform",
    "build_disparity_products",
    "compute_disparity_audit_stats",
    "finalize_single_engine_tensorrt_output",
    "finalize_tensorrt_disparity_batch_outputs",
    "load_tensorrt_model_config",
    "resolve_single_engine_tensorrt_model_path",
    "resolve_tensorrt_engine_static_batch_size",
    "resolve_tensorrt_image_transform",
    "resolve_tensorrt_model_config_path",
    "split_disparity_batch_output_maps",
    "undo_tensorrt_disparity_transform",
    "_CachedTensorRTRun",
    "_PinnedBatchPairImageInputBuffers",
    "_PinnedSinglePairImageInputBuffers",
    "_configure_tensorrt_runtime_search_paths",
    "_disable_torch_compile",
    "_ensure_ffs_repo_on_sys_path",
    "_load_official_tensorrt_foundation_stereo",
    "_patch_batch_safe_gwc_volume_triton",
]
