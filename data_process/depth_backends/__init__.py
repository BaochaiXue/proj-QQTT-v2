from __future__ import annotations

from importlib import import_module


_EXPORT_TO_MODULE = {
    "FfsBenchmarkConfig": ".benchmarking",
    "build_tradeoff_summary": ".benchmarking",
    "build_confidence_filtered_depth_uint16": ".confidence_filtering",
    "compute_reference_depth_metrics": ".benchmarking",
    "expand_benchmark_configs": ".benchmarking",
    "infer_model_label": ".benchmarking",
    "resize_depth_nearest": ".benchmarking",
    "select_tradeoff_result": ".benchmarking",
    "summarize_latency_samples_ms": ".benchmarking",
    "align_depth_to_color": ".geometry",
    "align_ir_scalar_to_color": ".geometry",
    "disparity_to_metric_depth": ".geometry",
    "format_ffs_intrinsic_text": ".geometry",
    "project_to_color": ".geometry",
    "quantize_depth_with_invalid_zero": ".geometry",
    "rasterize_scalar_by_nearest_depth": ".geometry",
    "rasterize_nearest_depth": ".geometry",
    "transform_points": ".geometry",
    "unproject_ir_depth": ".geometry",
    "write_ffs_intrinsic_file": ".geometry",
    "FastFoundationStereoRunner": ".fast_foundation_stereo",
    "FastFoundationStereoSingleEngineTensorRTRunner": ".fast_foundation_stereo",
    "FastFoundationStereoTensorRTRunner": ".fast_foundation_stereo",
    "apply_tensorrt_image_transform": ".fast_foundation_stereo",
    "apply_remove_invisible_mask": ".fast_foundation_stereo",
    "finalize_single_engine_tensorrt_output": ".fast_foundation_stereo",
    "finalize_tensorrt_disparity_batch_outputs": ".fast_foundation_stereo",
    "load_tensorrt_model_config": ".fast_foundation_stereo",
    "normalize_single_engine_tensorrt_image": ".fast_foundation_stereo",
    "resolve_tensorrt_engine_static_batch_size": ".fast_foundation_stereo",
    "resolve_single_engine_tensorrt_model_path": ".fast_foundation_stereo",
    "resolve_tensorrt_model_config_path": ".fast_foundation_stereo",
    "resolve_tensorrt_image_transform": ".fast_foundation_stereo",
    "select_tensorrt_disparity_output": ".fast_foundation_stereo",
    "split_disparity_batch_output_maps": ".fast_foundation_stereo",
    "undo_tensorrt_disparity_transform": ".fast_foundation_stereo",
    "compute_disparity_audit_stats": ".ffs_audit",
    "derive_ir_right_to_color": ".ffs_audit",
    "summarize_left_right_audit": ".ffs_audit",
    "FFS_DEPTH_ARCHIVE_DIR_FFS_BACKEND": ".radius_outlier_filter",
    "FFS_DEPTH_ARCHIVE_DIR_BOTH_BACKEND": ".radius_outlier_filter",
    "FFS_FLOAT_ARCHIVE_DIR": ".radius_outlier_filter",
    "FFS_RADIUS_OUTLIER_FILTER_ARCHIVE_POLICY": ".radius_outlier_filter",
    "FFS_RADIUS_OUTLIER_FILTER_MODE": ".radius_outlier_filter",
    "apply_ffs_radius_outlier_filter_float_m": ".radius_outlier_filter",
    "apply_ffs_radius_outlier_filter_u16": ".radius_outlier_filter",
    "build_ffs_radius_outlier_filter_contract": ".radius_outlier_filter",
}

__all__ = sorted(_EXPORT_TO_MODULE)


def __getattr__(name: str):
    if name not in _EXPORT_TO_MODULE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORT_TO_MODULE[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
