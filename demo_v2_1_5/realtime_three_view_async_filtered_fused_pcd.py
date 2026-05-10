#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo_v2_2 import runtime  # noqa: E402


DEFAULT_PRESET = runtime.PRESET_DEMO215_ASYNC_FILTER_5FPS


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Demo 2.1.5 native-depth path: 3 RealSense cameras, native aligned RealSense depth, "
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
    run.add_argument("--gpu-sampling-backend", choices=runtime.GPU_SAMPLING_BACKENDS, default=None, help="GPU sampler backend.")
    run.add_argument("--gpu-sampling-device-index", type=int, default=None, help="GPU index for the sampler.")
    run.add_argument("--fps", type=int, default=None, help="RealSense RGB-D capture target FPS.")
    run.add_argument("--dry-run", action="store_true", help="Print the resolved Demo 2.1.5 runtime contract and exit.")
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
    tracking.add_argument("--object-prompt", default=None, help="SAM3.1 first-frame object prompt.")
    tracking.add_argument("--controller-prompt", default=None, help="SAM3.1 first-frame controller prompt.")
    mode = tracking.add_mutually_exclusive_group()
    mode.add_argument("--object-only", action="store_true", help="Track only the object layer.")
    mode.add_argument("--controller-object", action="store_true", help="Track controller and object layers.")

    view = parser.add_argument_group("Point Cloud")
    view.add_argument("--min-depth-m", type=float, default=None, help="Minimum depth kept in fused PCD.")
    view.add_argument("--max-depth-m", type=float, default=None, help="Maximum depth kept in fused PCD.")
    view.add_argument("--point-size", type=float, default=None, help="Open3D point size.")
    view.add_argument("--output-root", default=None, help="Output root for runtime artifacts.")

    startup = parser.add_argument_group("Startup")
    startup.add_argument("--no-parallel-init", action="store_true", help="Disable parallel camera/model startup.")
    startup.add_argument("--no-compile-prewarm", action="store_true", help="Skip EdgeTAM compile prewarm during init.")
    startup.add_argument(
        "--warm-cache-only",
        action="store_true",
        help="Warm EdgeTAM/SAM3.1 initialization caches and exit without opening cameras.",
    )
    startup.add_argument(
        "--warm-cache-repeat",
        type=int,
        default=1,
        help="Number of cache warmup passes to run when --warm-cache-only is set.",
    )
    startup.add_argument(
        "--warm-cache-json-output",
        default="docs/generated/demo2_1_5_init_cache_warmup_probe.json",
        help="JSON output path for --warm-cache-only.",
    )

    experiments = parser.add_argument_group("Explicit Experiments")
    experiments.add_argument(
        "--experimental-edgetam-batch-vision",
        action="store_true",
        help="Batch 3 RGB frames through EdgeTAM image features, then keep per-camera video sessions.",
    )
    experiments.add_argument(
        "--experimental-staged-parallel",
        action="store_true",
        help="Use the older staged depth-then-parallel-EdgeTAM probe instead of the default single-owner path.",
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


def _to_demo215_argv(argv: Sequence[str] | None) -> list[str]:
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = build_arg_parser()
    parsed, passthrough = parser.parse_known_args(raw)
    if parsed.advanced_help:
        return ["--help"]

    translated: list[str] = []
    if parsed.experimental_staged_parallel and not _has_flag(passthrough, "--preset"):
        translated.extend(["--preset", runtime.PRESET_DEMO215_STAGED_PARALLEL_5FPS])

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
    _append_option(translated, "--object-prompt", parsed.object_prompt)
    _append_option(translated, "--controller-prompt", parsed.controller_prompt)
    _append_option(translated, "--depth-min-m", parsed.min_depth_m)
    _append_option(translated, "--depth-max-m", parsed.max_depth_m)
    _append_option(translated, "--point-size", parsed.point_size)
    _append_option(translated, "--output-root", parsed.output_root)

    if parsed.dry_run:
        translated.append("--dry-run")
    if parsed.debug:
        translated.append("--debug")
    if parsed.gpu_sampling:
        translated.append("--gpu-sampling")
    if parsed.object_only:
        translated.extend(["--track-mode", runtime.TRACK_MODE_OBJECT_ONLY])
    if parsed.controller_object:
        translated.extend(["--track-mode", runtime.TRACK_MODE_CONTROLLER_OBJECT])
    if parsed.no_parallel_init:
        translated.append("--no-parallel-init")
    if parsed.no_compile_prewarm:
        translated.append("--no-edgetam-prewarm-compile")
    if parsed.experimental_edgetam_batch_vision:
        translated.append("--edgetam-batch-vision-encoder")

    return _with_default_preset([*translated, *passthrough])


def _run_warm_cache(argv: Sequence[str], *, repeat: int, json_output: str | Path) -> int:
    runtime_argv = [arg for arg in argv if arg != "--dry-run"]
    parser = runtime.build_arg_parser()
    args = parser.parse_args(runtime_argv)
    args = runtime.apply_preset_defaults(args, explicit_options=runtime.explicit_cli_options(runtime_argv))
    args.parallel_init = False
    args.edgetam_prewarm_compile = True
    args.edgetam_prewarm_runs = max(1, int(getattr(args, "edgetam_prewarm_runs", 1) or 1))
    args.sam31_cache_init_model = True
    demo_runtime = runtime.Demo22Runtime(args)
    started_s = runtime.time.perf_counter()
    result = demo_runtime.warm_init_caches(repeats=max(1, int(repeat)))
    result["mode"] = "demo2.1.5-init-cache-warmup"
    result["total_wall_ms"] = float(runtime.elapsed_ms(started_s, runtime.time.perf_counter()))
    output_path = Path(json_output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    repeats = result.get("repeats", [])
    last = repeats[-1] if repeats else {}
    print(
        "[demo2.1.5-warm-cache] "
        f"repeats={len(repeats)} total_ms={result['total_wall_ms']:.2f} "
        f"last_repeat_ms={float(last.get('total_ms', 0.0)):.2f} "
        f"json={output_path}",
        flush=True,
    )
    return 0


def _with_default_preset(argv: Sequence[str] | None) -> list[str]:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--preset" not in args and not any(str(arg).startswith("--preset=") for arg in args):
        return ["--preset", DEFAULT_PRESET, *args]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    public_args, _passthrough = build_arg_parser().parse_known_args(raw)
    runtime_argv = _to_demo215_argv(raw)
    if public_args.warm_cache_only:
        return _run_warm_cache(
            runtime_argv,
            repeat=int(public_args.warm_cache_repeat),
            json_output=public_args.warm_cache_json_output,
        )
    return runtime.main(runtime_argv)


if __name__ == "__main__":
    raise SystemExit(main())
