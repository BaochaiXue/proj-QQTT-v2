#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qqtt.demo import demo22_runtime as runtime  # noqa: E402


DEFAULT_PRESET = runtime.PRESET_DEMO22_ASYNC_FILTER_5FPS


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Demo 2.2 local RTX 5090 path: 3 RealSense cameras, local FFS TensorRT, "
            "compiled EdgeTAM tracking, async filtered fused PCD render."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    run = parser.add_argument_group("Run")
    run.add_argument("--duration-s", type=float, default=None, help="Run duration. Use 0 for unlimited.")
    run.add_argument("--warmup-s", type=float, default=None, help="Seconds excluded from FPS/profile summaries.")
    run.add_argument("--profile-json-output", default=None, help="Write the full profile JSON to this path.")
    run.add_argument("--gpu-sampling", action="store_true", help="Record GPU utilization/memory/power samples in the profile.")
    run.add_argument("--gpu-sampling-interval-s", type=float, default=None, help="GPU sampling interval in seconds.")
    run.add_argument("--gpu-sampling-backend", choices=runtime.GPU_SAMPLING_BACKENDS, default=None, help="GPU sampler backend; NVML only.")
    run.add_argument("--gpu-sampling-device-index", type=int, default=None, help="GPU index for the sampler.")
    run.add_argument("--fps", type=int, default=None, help="RealSense RGB+IR capture target FPS.")
    run.add_argument("--dry-run", action="store_true", help="Print the resolved Demo 2.2 runtime contract and exit.")
    run.add_argument("--debug", action="store_true", help="Enable verbose runtime logs.")

    cameras = parser.add_argument_group("Cameras")
    cameras.add_argument("--serials", nargs="*", default=None, help="Optional RealSense serial list.")
    cameras.add_argument("--camera-ids", type=runtime.parse_camera_ids, default=None, help="Camera ids, e.g. 0,1,2.")
    cameras.add_argument("--calibrate-path", default=None, help="Calibration pickle path.")
    cameras.add_argument(
        "--calibration-reference-serials",
        nargs="*",
        default=None,
        help="Optional serial order used by the calibration file.",
    )

    tracking = parser.add_argument_group("Tracking")
    tracking.add_argument(
        "--experiment-mode",
        choices=runtime.EXPERIMENT_MODES,
        default=None,
        help="Semantic mode: controller-object-exp uses towel; demo-mode uses hand.",
    )
    tracking.add_argument("--object-prompt", default=None, help="SAM3.1 first-frame object prompt.")
    tracking.add_argument("--controller-prompt", default=None, help="SAM3.1 first-frame controller prompt.")
    mode = tracking.add_mutually_exclusive_group()
    mode.add_argument("--object-only", action="store_true", help="Track only the object layer.")
    mode.add_argument("--controller-only", action="store_true", help="Track only the controller layer.")
    mode.add_argument("--controller-object", action="store_true", help="Track controller and object layers.")

    view = parser.add_argument_group("Point Cloud")
    view.add_argument("--min-depth-m", type=float, default=None, help="Minimum depth kept in fused PCD.")
    view.add_argument("--max-depth-m", type=float, default=None, help="Maximum depth kept in fused PCD.")
    view.add_argument("--point-size", type=float, default=None, help="Open3D point size.")
    view.add_argument("--render-every-n", type=int, default=None, help="Publish every Nth filtered packet to the renderer.")
    view.add_argument("--render-backend", choices=runtime.RENDER_BACKENDS, default=None, help="Pointcloud renderer backend.")
    view.add_argument("--render-layer-mode", choices=runtime.RENDER_LAYER_MODES, default=None, help="Pointcloud renderer layer mode.")
    view.add_argument("--render-copy-mode", choices=runtime.RENDER_COPY_MODES, default=None, help="Renderer copy/profile mode.")
    view.add_argument("--no-render-async-latest-only", action="store_true", help="Disable coalesced latest-only render posts.")
    view.add_argument("--render-micro-profile", action="store_true", help="Record detailed renderer copy/update timing.")
    view.add_argument("--output-root", default=None, help="Output root for runtime artifacts.")

    overlay = parser.add_argument_group("Tracking Overlay")
    overlay.add_argument("--show-tracking-overlay", action="store_true", help="Enable optional Demo 3 tracking anchor overlay.")
    overlay.add_argument(
        "--tracking-backend",
        choices=("none", "cotracker3_online", "nvofa", "tapnext", "locotrack", "tapir", "vpi_lk", "offline_npz", "cached"),
        default=None,
        help="Tracking backend for optional Demo 3 overlay.",
    )
    overlay.add_argument(
        "--tracking-source",
        choices=("live", "cached", "offline_npz"),
        default=None,
        help="Source for optional Demo 3 overlay tracks.",
    )
    overlay.add_argument("--tracking-num-points", type=int, default=None, help="Sparse tracking query point count.")
    overlay.add_argument("--tracking-overlay-max-points", type=int, default=None, help="Maximum anchors shown in overlay.")
    overlay.add_argument("--tracking-trail-len", type=int, default=None, help="Short 3D trail length in frames.")
    overlay.add_argument("--tracking-update-hz", type=float, default=None, help="Tracking worker update rate cap.")
    overlay.add_argument(
        "--tracking-depth-source",
        choices=("displayed", "native", "ffs"),
        default=None,
        help="Depth source used to lift tracks for overlay.",
    )
    overlay.add_argument("--tracking-output-root", default=None, help="Output root for Demo 3 live tracking artifacts.")

    startup = parser.add_argument_group("Startup")
    startup.add_argument("--no-parallel-init", action="store_true", help="Disable parallel camera/model startup.")
    startup.add_argument("--no-compile-prewarm", action="store_true", help="Skip EdgeTAM compile prewarm during init.")

    experiments = parser.add_argument_group("Explicit Experiments")
    experiments.add_argument(
        "--edgetam-batch-vision",
        dest="edgetam_batch_vision",
        action="store_true",
        default=None,
        help="Batch 3 RGB frames through EdgeTAM image features, then keep per-camera video sessions.",
    )
    experiments.add_argument(
        "--no-edgetam-batch-vision",
        dest="edgetam_batch_vision",
        action="store_false",
        help="Disable the Demo 2.2 preset batch-vision EdgeTAM path for A/B profiling.",
    )
    experiments.add_argument(
        "--experimental-edgetam-batch-vision",
        dest="edgetam_batch_vision",
        action="store_true",
        help="Legacy alias for --edgetam-batch-vision.",
    )
    experiments.add_argument(
        "--experimental-staged-parallel",
        action="store_true",
        help="Use the older staged FFS-then-parallel-EdgeTAM probe instead of the default single-owner path.",
    )
    experiments.add_argument(
        "--experimental-overlapped-stages",
        action="store_true",
        help="Use the cross-group FFS/EdgeTAM/join stage-overlap throughput mode.",
    )
    experiments.add_argument(
        "--ffs-batch-size",
        type=int,
        choices=runtime.FFS_TRT_BATCH_SIZES,
        default=None,
        help="Override FFS TensorRT static batch size.",
    )
    experiments.add_argument(
        "--advanced-help",
        action="store_true",
        help="Show the full internal runtime help with legacy and experiment flags.",
    )

    return parser


def _append_option(args: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        args.extend([flag, str(value)])


def _append_many(args: list[str], flag: str, values: Sequence[object] | None) -> None:
    if values is not None:
        args.append(flag)
        args.extend(str(value) for value in values)


def _format_camera_ids(camera_ids: tuple[int, ...] | None) -> str | None:
    if camera_ids is None:
        return None
    return ",".join(str(camera_id) for camera_id in camera_ids)


def _has_flag(args: Sequence[str], flag: str) -> bool:
    return flag in args or any(str(arg).startswith(f"{flag}=") for arg in args)


def _to_demo22_argv(argv: Sequence[str] | None) -> list[str]:
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = build_arg_parser()
    parsed, passthrough = parser.parse_known_args(raw)
    if parsed.advanced_help:
        return ["--help"]

    translated: list[str] = []
    if parsed.experimental_staged_parallel and not _has_flag(passthrough, "--preset"):
        translated.extend(["--preset", runtime.PRESET_DEMO22_STAGED_PARALLEL_5FPS])

    _append_option(translated, "--duration-s", parsed.duration_s)
    _append_option(translated, "--profile-warmup-exclude-s", parsed.warmup_s)
    _append_option(translated, "--profile-json-output", parsed.profile_json_output)
    _append_option(translated, "--gpu-sampling-interval-s", parsed.gpu_sampling_interval_s)
    _append_option(translated, "--gpu-sampling-backend", parsed.gpu_sampling_backend)
    _append_option(translated, "--gpu-sampling-device-index", parsed.gpu_sampling_device_index)
    _append_option(translated, "--fps", parsed.fps)
    _append_many(translated, "--serials", parsed.serials)
    _append_option(translated, "--camera-ids", _format_camera_ids(parsed.camera_ids))
    _append_option(translated, "--calibrate-path", parsed.calibrate_path)
    _append_many(translated, "--calibration-reference-serials", parsed.calibration_reference_serials)
    _append_option(translated, "--experiment-mode", parsed.experiment_mode)
    _append_option(translated, "--object-prompt", parsed.object_prompt)
    _append_option(translated, "--controller-prompt", parsed.controller_prompt)
    _append_option(translated, "--depth-min-m", parsed.min_depth_m)
    _append_option(translated, "--depth-max-m", parsed.max_depth_m)
    _append_option(translated, "--point-size", parsed.point_size)
    _append_option(translated, "--render-every-n", parsed.render_every_n)
    _append_option(translated, "--render-backend", parsed.render_backend)
    _append_option(translated, "--render-layer-mode", parsed.render_layer_mode)
    _append_option(translated, "--render-copy-mode", parsed.render_copy_mode)
    _append_option(translated, "--output-root", parsed.output_root)
    _append_option(translated, "--tracking-backend", parsed.tracking_backend)
    _append_option(translated, "--tracking-source", parsed.tracking_source)
    _append_option(translated, "--tracking-num-points", parsed.tracking_num_points)
    _append_option(translated, "--tracking-overlay-max-points", parsed.tracking_overlay_max_points)
    _append_option(translated, "--tracking-trail-len", parsed.tracking_trail_len)
    _append_option(translated, "--tracking-update-hz", parsed.tracking_update_hz)
    _append_option(translated, "--tracking-depth-source", parsed.tracking_depth_source)
    _append_option(translated, "--tracking-output-root", parsed.tracking_output_root)
    _append_option(translated, "--ffs-trt-batch-size", parsed.ffs_batch_size)

    if parsed.dry_run:
        translated.append("--dry-run")
    if parsed.debug:
        translated.append("--debug")
    if parsed.gpu_sampling:
        translated.append("--gpu-sampling")
    if parsed.no_render_async_latest_only:
        translated.append("--no-render-async-latest-only")
    if parsed.render_micro_profile:
        translated.append("--render-micro-profile")
    if parsed.object_only:
        translated.extend(["--track-mode", runtime.TRACK_MODE_OBJECT_ONLY])
    if parsed.controller_only:
        translated.extend(["--track-mode", runtime.TRACK_MODE_CONTROLLER_ONLY])
    if parsed.controller_object:
        translated.extend(["--track-mode", runtime.TRACK_MODE_CONTROLLER_OBJECT])
    if parsed.no_parallel_init:
        translated.append("--no-parallel-init")
    if parsed.no_compile_prewarm:
        translated.append("--no-edgetam-prewarm-compile")
    if parsed.edgetam_batch_vision is True:
        translated.append("--edgetam-batch-vision-encoder")
    elif parsed.edgetam_batch_vision is False:
        translated.append("--no-edgetam-batch-vision-encoder")
    if parsed.experimental_overlapped_stages:
        translated.extend(["--gpu-pipeline-mode", runtime.GPU_PIPELINE_MODE_OVERLAPPED_STAGES])
    if parsed.show_tracking_overlay:
        translated.append("--show-tracking-overlay")

    return _with_default_preset([*translated, *passthrough])


def _with_default_preset(argv: Sequence[str] | None) -> list[str]:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--preset" not in args and not any(str(arg).startswith("--preset=") for arg in args):
        return ["--preset", DEFAULT_PRESET, *args]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    return runtime.main(_to_demo22_argv(argv))


if __name__ == "__main__":
    raise SystemExit(main())
