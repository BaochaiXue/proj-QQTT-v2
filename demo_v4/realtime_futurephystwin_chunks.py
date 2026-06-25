#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from typing import Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT_STR = str(REPO_ROOT)
if REPO_ROOT_STR in sys.path:
    sys.path.remove(REPO_ROOT_STR)
sys.path.insert(0, REPO_ROOT_STR)

from demo_v4.headless_chunk_bridge import stream_chunks_from_headless_capture, write_chunks_from_headless_capture


DEFAULT_FUTUREPHYSTWIN_BASE_PATH = Path("result/demo_v4/futurephystwin_chunks")
DEFAULT_INPUT_SOURCE = "fake-live"
DEFAULT_REPLAY_FPS = 5.0
DEFAULT_CHUNK_SECONDS = 5.0
DEFAULT_CHUNK_POLL_INTERVAL_S = 0.001
DEFAULT_DEMO32_LOSSLESS_INPUT_FPS = 5.0
DEFAULT_CASE_PREFIX = "demo_v4"
DEFAULT_DEPTH_BACKEND = "native-realsense"
DEFAULT_MAX_CHUNKS: int | None = None
DEFAULT_CAPTURE_EXTRA_SECONDS = 10.0
DEFAULT_SHAPE_PRIOR_ENDPOINT = "tcp://127.0.0.1:7100"
DEFAULT_MASK_RADIUS_OUTLIER_RADIUS_M = 0.01
DEFAULT_MASK_RADIUS_OUTLIER_NB_POINTS = 40
DEFAULT_REALTIME_GPU_MODE = "single"
DEFAULT_WARMUP_GPU_MODE = "dual"
DEFAULT_GPU_MODE = DEFAULT_REALTIME_GPU_MODE
GPU_MODE_DEMO32_CUDA_VISIBLE_DEVICES = {
    "single": "0",
    "dual": "1",
}
GPU_MODE_SHAPE_PRIOR_DEVICE = {
    "single": "cuda:0",
    "dual": "cuda:1",
}
DEFAULT_DEMO32_DEVICE = "cuda"
DEFAULT_DEMO32_TRACKER_DEVICE = "cuda"
DEFAULT_DEMO32_DTYPE = "bfloat16"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Demo v4 realtime FuturePhysTwin chunk preprocessor. The first "
            "implementation can convert Demo 3.2 headless captures into "
            "FuturePhysTwin-consumable chunk case roots."
        )
    )
    parser.add_argument(
        "--input-source",
        choices=("fake-live", "live"),
        default=DEFAULT_INPUT_SOURCE,
        help="Camera source mode used when Demo v4 launches its own capture.",
    )
    parser.add_argument("--replay-fps", type=float, default=DEFAULT_REPLAY_FPS)
    parser.add_argument(
        "--demo32-source-replay-fps",
        type=float,
        default=None,
        help=(
            "Optional Demo 3.2 fake-live pacing FPS. When omitted, Demo 3.2 uses "
            "--replay-fps; Demo v4 output metadata/window math still use --replay-fps."
        ),
    )
    parser.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS)
    parser.add_argument(
        "--chunk-poll-interval-s",
        type=float,
        default=DEFAULT_CHUNK_POLL_INTERVAL_S,
        help="Polling interval for realtime frames.jsonl chunk tailing.",
    )
    parser.add_argument("--depth-backend", choices=("ir-ffs", "native-realsense"), default=DEFAULT_DEPTH_BACKEND)
    parser.add_argument(
        "--chunk-frame-count",
        type=int,
        default=None,
        help="Override chunk length in frames. Defaults to round(replay_fps * chunk_seconds).",
    )
    parser.add_argument("--futurephystwin-base-path", type=Path, default=DEFAULT_FUTUREPHYSTWIN_BASE_PATH)
    parser.add_argument("--case-prefix", default=DEFAULT_CASE_PREFIX)
    parser.add_argument(
        "--gpu-mode",
        choices=tuple(GPU_MODE_DEMO32_CUDA_VISIBLE_DEVICES),
        default=DEFAULT_REALTIME_GPU_MODE,
        help=(
            "Backward-compatible realtime GPU routing preset. Prefer "
            "--realtime-gpu-mode for new experiments."
        ),
    )
    parser.add_argument(
        "--realtime-gpu-mode",
        choices=tuple(GPU_MODE_DEMO32_CUDA_VISIBLE_DEVICES),
        default=None,
        help="GPU routing preset for the Demo 3.2 camera/fake-camera -> final-data realtime subprocess.",
    )
    parser.add_argument(
        "--warmup-gpu-mode",
        choices=tuple(GPU_MODE_SHAPE_PRIOR_DEVICE),
        default=DEFAULT_WARMUP_GPU_MODE,
        help="GPU routing preset for SAM3D shape-prior warmup device selection.",
    )
    parser.add_argument(
        "--demo32-cuda-visible-devices",
        default=None,
        help="Explicit CUDA_VISIBLE_DEVICES override for the Demo 3.2 subprocess.",
    )
    parser.add_argument(
        "--demo32-device",
        default=DEFAULT_DEMO32_DEVICE,
        help="Segmentation/runtime device passed to Demo 3.2 inside the subprocess CUDA namespace.",
    )
    parser.add_argument(
        "--demo32-tracker-device",
        default=DEFAULT_DEMO32_TRACKER_DEVICE,
        help="TAPNext++ tracker device passed to Demo 3.2 inside the subprocess CUDA namespace.",
    )
    parser.add_argument(
        "--demo32-dtype",
        choices=("bfloat16", "float16", "float32"),
        default=DEFAULT_DEMO32_DTYPE,
        help="Segmentation/runtime dtype passed to Demo 3.2 inside the subprocess CUDA namespace.",
    )
    parser.add_argument(
        "--demo32-lossless-max-backlog-seconds",
        type=float,
        default=None,
        help=(
            "Optional strict lossless replay backlog window passed to Demo 3.2. "
            "Omit it to keep Demo 3.2 defaults."
        ),
    )
    parser.add_argument(
        "--demo32-headless-prepared-only",
        dest="demo32_headless_prepared_only",
        action="store_true",
        help="Ask Demo 3.2 to write only prepared PhysTwin frames needed by Demo v4 chunking.",
    )
    parser.add_argument(
        "--demo32-legacy-headless-artifacts",
        dest="demo32_headless_prepared_only",
        action="store_false",
        help="Keep Demo 3.2 legacy per-frame headless artifacts in addition to prepared PhysTwin frames.",
    )
    parser.set_defaults(demo32_headless_prepared_only=True)
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=DEFAULT_MAX_CHUNKS,
        help=(
            "Optional realtime chunk cap for debug/short validation runs. "
            "Omit it to stream until the fake-live recording or live capture ends."
        ),
    )
    parser.add_argument(
        "--capture-extra-seconds",
        type=float,
        default=DEFAULT_CAPTURE_EXTRA_SECONDS,
        help="Extra Demo 3.2 runtime beyond max_chunks*chunk_seconds to absorb startup/warmup latency.",
    )
    parser.add_argument(
        "--demo32-capture-dir",
        type=Path,
        default=None,
        help="Headless capture directory for the Demo 3.2 realtime subprocess.",
    )
    parser.add_argument(
        "--source-headless-capture",
        type=Path,
        default=None,
        help="Existing Demo 3.2 headless capture directory to chunk without launching capture.",
    )
    parser.add_argument("--surface-points-npy", type=Path, default=None)
    parser.add_argument("--interior-points-npy", type=Path, default=None)
    parser.add_argument(
        "--write-final-pcd",
        dest="write_final_pcd",
        action="store_true",
        help="Write dense per-frame pcd/*.npz into each published chunk case for diagnostics/export.",
    )
    parser.add_argument(
        "--no-write-final-pcd",
        dest="write_final_pcd",
        action="store_false",
        help="Skip dense per-frame pcd/*.npz in chunk cases; final_data/tracking/mask/color remain complete.",
    )
    parser.set_defaults(write_final_pcd=False)
    parser.add_argument(
        "--shape-prior-warmup",
        dest="shape_prior_warmup",
        action="store_true",
        help="Keep SAM3D shape-prior warmup enabled for Demo v4 capture.",
    )
    parser.add_argument(
        "--no-shape-prior-warmup",
        dest="shape_prior_warmup",
        action="store_false",
        help="Disable SAM3D shape-prior warmup.",
    )
    parser.set_defaults(shape_prior_warmup=True)
    parser.add_argument(
        "--shape-prior-start-policy",
        choices=(
            "async-after-first-mask-depth-pair",
            "async-after-first-strict-pair",
            "blocking-before-first-output",
            "after-teardown",
        ),
        default="async-after-first-mask-depth-pair",
    )
    parser.add_argument(
        "--shape-prior-execution",
        choices=("remote-worker", "local-subprocess"),
        default="remote-worker",
    )
    parser.add_argument("--shape-prior-endpoint", default=DEFAULT_SHAPE_PRIOR_ENDPOINT)
    parser.add_argument("--shape-prior-timeout-ms", type=int, default=180000)
    parser.add_argument(
        "--shape-prior-chunk-wait-timeout-s",
        type=float,
        default=300.0,
        help="How long Demo v4 waits for required shape-prior structure points before writing final_data chunks.",
    )
    parser.add_argument(
        "--shape-prior-device",
        default=None,
        help="Explicit shape-prior device override. Defaults from --warmup-gpu-mode.",
    )
    parser.add_argument("--shape-prior-profile-json", type=Path, default=None)
    parser.add_argument(
        "--mask-radius-outlier-filter",
        dest="mask_radius_outlier_filter",
        action="store_true",
        help="Apply data_process_sam3d-style 3D mask radius-outlier refinement before final_data chunking.",
    )
    parser.add_argument(
        "--no-mask-radius-outlier-filter",
        dest="mask_radius_outlier_filter",
        action="store_false",
        help="Disable 3D mask radius-outlier refinement. Intended for tiny synthetic fixtures only.",
    )
    parser.set_defaults(mask_radius_outlier_filter=True)
    parser.add_argument("--mask-radius-outlier-radius-m", type=float, default=DEFAULT_MASK_RADIUS_OUTLIER_RADIUS_M)
    parser.add_argument("--mask-radius-outlier-nb-points", type=int, default=DEFAULT_MASK_RADIUS_OUTLIER_NB_POINTS)
    parser.add_argument(
        "--shape-prior-skip-route-visualizations",
        dest="shape_prior_skip_route_visualizations",
        action="store_true",
    )
    parser.add_argument(
        "--shape-prior-render-route-visualizations",
        dest="shape_prior_skip_route_visualizations",
        action="store_false",
    )
    parser.set_defaults(shape_prior_skip_route_visualizations=True)
    parser.add_argument("--dry-run", action="store_true", help="Print resolved Demo v4 contract and exit.")
    return parser


def resolve_chunk_frame_count(args: argparse.Namespace) -> int:
    chunk_seconds = float(args.chunk_seconds)
    if not math.isfinite(chunk_seconds) or chunk_seconds <= 0.0:
        raise ValueError("chunk seconds must be positive")
    replay_fps = float(args.replay_fps)
    if not math.isfinite(replay_fps) or replay_fps <= 0.0:
        raise ValueError("replay fps must be positive")
    if args.chunk_frame_count is not None:
        value = int(args.chunk_frame_count)
    else:
        value = int(round(replay_fps * chunk_seconds))
    if value <= 0:
        raise ValueError("chunk frame count must be positive")
    return value


def resolve_demo32_source_replay_fps(args: argparse.Namespace) -> float:
    value = args.demo32_source_replay_fps
    fps = float(args.replay_fps if value is None else value)
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("Demo 3.2 source replay fps must be positive")
    return fps


def resolve_realtime_gpu_mode(args: argparse.Namespace) -> str:
    value = getattr(args, "realtime_gpu_mode", None)
    if value is None:
        value = getattr(args, "gpu_mode", DEFAULT_REALTIME_GPU_MODE)
    value = str(value)
    if value not in GPU_MODE_DEMO32_CUDA_VISIBLE_DEVICES:
        raise ValueError(f"unsupported realtime gpu mode: {value!r}")
    return value


def resolve_warmup_gpu_mode(args: argparse.Namespace) -> str:
    value = str(getattr(args, "warmup_gpu_mode", DEFAULT_WARMUP_GPU_MODE))
    if value not in GPU_MODE_SHAPE_PRIOR_DEVICE:
        raise ValueError(f"unsupported warmup gpu mode: {value!r}")
    return value


def resolve_demo32_cuda_visible_devices(args: argparse.Namespace) -> str:
    override = None if args.demo32_cuda_visible_devices is None else str(args.demo32_cuda_visible_devices).strip()
    if override:
        return override
    try:
        return GPU_MODE_DEMO32_CUDA_VISIBLE_DEVICES[resolve_realtime_gpu_mode(args)]
    except KeyError as exc:
        raise ValueError(f"unsupported realtime gpu mode: {resolve_realtime_gpu_mode(args)!r}") from exc


def resolve_shape_prior_device(args: argparse.Namespace) -> str:
    override = getattr(args, "shape_prior_device", None)
    if override is not None and str(override).strip():
        return str(override).strip()
    try:
        return GPU_MODE_SHAPE_PRIOR_DEVICE[resolve_warmup_gpu_mode(args)]
    except KeyError as exc:
        raise ValueError(f"unsupported warmup gpu mode: {resolve_warmup_gpu_mode(args)!r}") from exc


def _load_optional_points(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    arr = np.asarray(np.load(path), dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{path} must contain an Nx3 point array")
    return np.ascontiguousarray(arr, dtype=np.float64)


def resolve_online_dir(args: argparse.Namespace) -> Path:
    return Path(args.futurephystwin_base_path) / "online_data" / str(args.case_prefix)


def resolve_static_data_path(args: argparse.Namespace) -> Path:
    return Path(args.futurephystwin_base_path) / "data" / str(args.case_prefix) / "final_data.pkl"


def _contract(args: argparse.Namespace) -> dict[str, object]:
    return {
        "demo_version": "demo_v4",
        "input_source": str(args.input_source),
        "replay_fps": float(args.replay_fps),
        "demo32_source_replay_fps": resolve_demo32_source_replay_fps(args),
        "demo32_source_replay_fps_override": (
            None if args.demo32_source_replay_fps is None else float(args.demo32_source_replay_fps)
        ),
        "demo32_lossless_input_fps": resolve_demo32_source_replay_fps(args),
        "chunk_seconds": float(args.chunk_seconds),
        "chunk_poll_interval_s": float(args.chunk_poll_interval_s),
        "chunk_frame_count": int(resolve_chunk_frame_count(args)),
        "futurephystwin_base_path": str(args.futurephystwin_base_path),
        "case_prefix": str(args.case_prefix),
        "output_format": "online-primary-static-case",
        "online_dir": str(resolve_online_dir(args)),
        "static_data_path": str(resolve_static_data_path(args)),
        "max_chunks": args.max_chunks,
        "depth_backend": str(args.depth_backend),
        "capture_extra_seconds": float(args.capture_extra_seconds),
        "demo32_capture_dir": None if args.demo32_capture_dir is None else str(args.demo32_capture_dir),
        "gpu_mode": resolve_realtime_gpu_mode(args),
        "realtime_gpu_mode": resolve_realtime_gpu_mode(args),
        "warmup_gpu_mode": resolve_warmup_gpu_mode(args),
        "demo32_cuda_visible_devices": resolve_demo32_cuda_visible_devices(args),
        "demo32_cuda_visible_devices_override": (
            None if args.demo32_cuda_visible_devices is None else str(args.demo32_cuda_visible_devices)
        ),
        "demo32_device": str(args.demo32_device),
        "demo32_tracker_device": str(args.demo32_tracker_device),
        "demo32_dtype": str(args.demo32_dtype),
        "demo32_lossless_max_backlog_seconds": args.demo32_lossless_max_backlog_seconds,
        "demo32_headless_prepared_only": bool(args.demo32_headless_prepared_only),
        "shape_prior_warmup": bool(args.shape_prior_warmup),
        "shape_prior_start_policy": str(args.shape_prior_start_policy),
        "shape_prior_execution": str(args.shape_prior_execution),
        "shape_prior_endpoint": str(args.shape_prior_endpoint),
        "shape_prior_device": resolve_shape_prior_device(args),
        "shape_prior_device_override": None if args.shape_prior_device is None else str(args.shape_prior_device),
        "shape_prior_chunk_wait_timeout_s": float(args.shape_prior_chunk_wait_timeout_s),
        "mask_radius_outlier_filter": bool(args.mask_radius_outlier_filter),
        "mask_radius_outlier_radius_m": float(args.mask_radius_outlier_radius_m),
        "mask_radius_outlier_nb_points": int(args.mask_radius_outlier_nb_points),
        "write_final_pcd": bool(args.write_final_pcd),
        "source_headless_capture": None if args.source_headless_capture is None else str(args.source_headless_capture),
    }


def _demo32_duration_s(args: argparse.Namespace, *, chunk_frame_count: int) -> float:
    if args.max_chunks is None:
        return 0.0
    fps = resolve_demo32_source_replay_fps(args)
    if fps <= 0.0:
        fps = DEFAULT_REPLAY_FPS
    return (float(args.max_chunks) * float(chunk_frame_count) / fps) + float(args.capture_extra_seconds)


def build_demo32_realtime_command(
    args: argparse.Namespace,
    *,
    capture_dir: Path,
    profile_json: Path,
    chunk_frame_count: int,
) -> list[str]:
    script = Path("demo_v3_2") / "realtime_single_camera_ffs_masked_pcd.py"
    demo32_source_replay_fps = resolve_demo32_source_replay_fps(args)
    command = [
        sys.executable,
        str(script),
        "--input-source",
        str(args.input_source),
        "--depth-backend",
        str(args.depth_backend),
        "--duration-s",
        f"{_demo32_duration_s(args, chunk_frame_count=chunk_frame_count):.3f}",
        "--render-mode",
        "none",
        "--headless-capture-dir",
        str(capture_dir),
        "--tracking-product-backend",
        "phystwin-strict-tracking",
        "--track-mode",
        "controller-object",
        "--tracker-backend",
        "tapnextpp",
        "--demo-visual-mode",
        "tracking",
        "--replay-fps",
        str(demo32_source_replay_fps),
        "--device",
        str(args.demo32_device),
        "--dtype",
        str(args.demo32_dtype),
        "--tracker-device",
        str(args.demo32_tracker_device),
    ]
    if args.demo32_lossless_max_backlog_seconds is not None:
        command.extend(
            [
                "--lossless-max-backlog-seconds",
                str(float(args.demo32_lossless_max_backlog_seconds)),
            ]
        )
    if float(demo32_source_replay_fps) != float(DEFAULT_DEMO32_LOSSLESS_INPUT_FPS):
        command.extend(["--lossless-input-fps", str(float(demo32_source_replay_fps))])
    if bool(args.demo32_headless_prepared_only):
        command.append("--headless-prepared-only")
    if bool(args.shape_prior_warmup):
        command.extend(
            [
                "--shape-prior-warmup",
                "--shape-prior-start-policy",
                str(args.shape_prior_start_policy),
                "--shape-prior-execution",
                str(args.shape_prior_execution),
                "--shape-prior-endpoint",
                str(args.shape_prior_endpoint),
                "--shape-prior-timeout-ms",
                str(int(args.shape_prior_timeout_ms)),
                "--shape-prior-device",
                resolve_shape_prior_device(args),
                "--shape-prior-profile-json",
                str(profile_json),
            ]
        )
        if bool(args.shape_prior_skip_route_visualizations):
            command.append("--shape-prior-skip-route-visualizations")
        else:
            command.append("--shape-prior-render-route-visualizations")
    else:
        command.append("--no-shape-prior-warmup")
    return command


def _default_capture_dir(args: argparse.Namespace, base_path: Path) -> Path:
    if args.demo32_capture_dir is not None:
        return Path(args.demo32_capture_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_path / f"{args.case_prefix}_demo32_capture_{stamp}"


def _stop_process(process: subprocess.Popen[bytes]) -> int | None:
    if process.poll() is not None:
        return process.returncode
    try:
        process.terminate()
        return process.wait(timeout=10)
    except Exception:
        try:
            process.kill()
            return process.wait(timeout=10)
        except Exception:
            return process.poll()


def select_validation_chunk_cases(manifests: Sequence[dict[str, object]]) -> list[str]:
    if len(manifests) < 5:
        raise ValueError("at least five chunks are required for second-last and fifth-last validation")
    return [
        str(manifests[-2]["case_name"]),
        str(manifests[-5]["case_name"]),
    ]


def _runtime_chunk_summary(manifests: Sequence[dict[str, object]]) -> dict[str, object]:
    publish_times = [
        float(item["publish_wall_s"])
        for item in manifests
        if item.get("publish_wall_s") is not None
    ]
    intervals = [publish_times[idx] - publish_times[idx - 1] for idx in range(1, len(publish_times))]
    backlog_values = [
        int(item["backlog_chunks"])
        for item in manifests
        if item.get("backlog_chunks") is not None
    ]
    shape_publish_times = [
        float(item["publish_wall_s"])
        for item in manifests
        if item.get("publish_wall_s") is not None
        and bool(item.get("shape_prior_complete") or item.get("shape_prior_target_counts_met"))
    ]
    return {
        "first_ready_chunk_wall_s": publish_times[0] if publish_times else None,
        "first_shape_prior_ready_chunk_wall_s": shape_publish_times[0] if shape_publish_times else None,
        "steady_publish_intervals_s": intervals,
        "steady_state_publish_interval_max_s": max(intervals) if intervals else None,
        "max_backlog_chunks": max(backlog_values) if backlog_values else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    chunk_frame_count = resolve_chunk_frame_count(args)
    if float(args.chunk_poll_interval_s) <= 0.0:
        raise ValueError("--chunk-poll-interval-s must be positive")
    resolve_demo32_source_replay_fps(args)

    if bool(args.dry_run):
        print(json.dumps(_contract(args), indent=2, sort_keys=True))
        return 0

    base_path = Path(args.futurephystwin_base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    if args.source_headless_capture is not None:
        manifests = write_chunks_from_headless_capture(
            args.source_headless_capture,
            base_path=base_path,
            case_prefix=str(args.case_prefix),
            chunk_frame_count=chunk_frame_count,
            fps=int(round(float(args.replay_fps))),
            max_chunks=args.max_chunks,
            surface_points=_load_optional_points(args.surface_points_npy),
            interior_points=_load_optional_points(args.interior_points_npy),
            mask_radius_outlier_filter=bool(args.mask_radius_outlier_filter),
            mask_radius_outlier_radius_m=float(args.mask_radius_outlier_radius_m),
            mask_radius_outlier_nb_points=int(args.mask_radius_outlier_nb_points),
            write_final_pcd=bool(args.write_final_pcd),
        )
        summary = {
            "demo_version": "demo_v4",
            "mode": "source-headless-capture",
            "source_headless_capture": str(args.source_headless_capture),
            "futurephystwin_base_path": str(base_path),
            "case_prefix": str(args.case_prefix),
            "output_format": "online-primary-static-case",
            "online_dir": str(resolve_online_dir(args)),
            "static_data_path": str(resolve_static_data_path(args)),
            "chunk_frame_count": int(chunk_frame_count),
            "max_chunks": args.max_chunks,
            "chunk_count": int(len(manifests)),
            "chunks": manifests,
            "write_final_pcd": bool(args.write_final_pcd),
        }
        summary.update(_runtime_chunk_summary(manifests))
        summary_path = base_path / f"{args.case_prefix}_chunks_manifest.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    capture_dir = _default_capture_dir(args, base_path)
    capture_dir.mkdir(parents=True, exist_ok=True)
    profile_json = Path(args.shape_prior_profile_json) if args.shape_prior_profile_json is not None else capture_dir / "shape_prior_profile.json"
    command = build_demo32_realtime_command(
        args,
        capture_dir=capture_dir,
        profile_json=profile_json,
        chunk_frame_count=chunk_frame_count,
    )
    demo32_env = os.environ.copy()
    cuda_visible_devices = resolve_demo32_cuda_visible_devices(args).strip()
    if cuda_visible_devices:
        demo32_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    process = subprocess.Popen(command, env=demo32_env)
    surface_points = _load_optional_points(args.surface_points_npy)
    interior_points = _load_optional_points(args.interior_points_npy)
    try:
        manifests = stream_chunks_from_headless_capture(
            capture_dir,
            base_path=base_path,
            case_prefix=str(args.case_prefix),
            chunk_frame_count=chunk_frame_count,
            fps=int(round(float(args.replay_fps))),
            max_chunks=args.max_chunks,
            capture_finished=lambda: process.poll() is not None,
            require_shape_prior=bool(args.shape_prior_warmup),
            shape_prior_wait_timeout_s=float(args.shape_prior_chunk_wait_timeout_s),
            poll_interval_s=float(args.chunk_poll_interval_s),
            surface_points=surface_points,
            interior_points=interior_points,
            mask_radius_outlier_filter=bool(args.mask_radius_outlier_filter),
            mask_radius_outlier_radius_m=float(args.mask_radius_outlier_radius_m),
            mask_radius_outlier_nb_points=int(args.mask_radius_outlier_nb_points),
            write_final_pcd=bool(args.write_final_pcd),
        )
    finally:
        return_code = _stop_process(process)
    validation_cases = select_validation_chunk_cases(manifests) if len(manifests) >= 5 else []
    if args.max_chunks is not None and len(manifests) >= int(args.max_chunks):
        stop_reason = "max_chunks_reached"
    elif return_code == 0:
        stop_reason = "demo32_completed"
    elif return_code is None:
        stop_reason = "demo32_status_unknown"
    else:
        stop_reason = "demo32_exited_before_target"
    summary = {
        "demo_version": "demo_v4",
        "mode": "full-fake-realtime-camera" if str(args.input_source) == "fake-live" else "full-live-camera",
        "gpu_mode": resolve_realtime_gpu_mode(args),
        "realtime_gpu_mode": resolve_realtime_gpu_mode(args),
        "warmup_gpu_mode": resolve_warmup_gpu_mode(args),
        "demo32_command": command,
        "demo32_cuda_visible_devices": cuda_visible_devices,
        "demo32_cuda_visible_devices_override": (
            None if args.demo32_cuda_visible_devices is None else str(args.demo32_cuda_visible_devices)
        ),
        "demo32_lossless_max_backlog_seconds": args.demo32_lossless_max_backlog_seconds,
        "demo32_headless_prepared_only": bool(args.demo32_headless_prepared_only),
        "demo32_source_replay_fps": resolve_demo32_source_replay_fps(args),
        "demo32_source_replay_fps_override": (
            None if args.demo32_source_replay_fps is None else float(args.demo32_source_replay_fps)
        ),
        "demo32_lossless_input_fps": resolve_demo32_source_replay_fps(args),
        "shape_prior_device": resolve_shape_prior_device(args),
        "shape_prior_device_override": None if args.shape_prior_device is None else str(args.shape_prior_device),
        "demo32_return_code": return_code,
        "demo32_stop_reason": stop_reason,
        "demo32_capture_dir": str(capture_dir),
        "futurephystwin_base_path": str(base_path),
        "case_prefix": str(args.case_prefix),
        "output_format": "online-primary-static-case",
        "online_dir": str(resolve_online_dir(args)),
        "static_data_path": str(resolve_static_data_path(args)),
        "chunk_frame_count": int(chunk_frame_count),
        "chunk_poll_interval_s": float(args.chunk_poll_interval_s),
        "max_chunks": args.max_chunks,
        "chunk_count": int(len(manifests)),
        "chunks": manifests,
        "validation_chunk_cases": validation_cases,
        "external_shape_prior_points": bool(surface_points is not None or interior_points is not None),
        "write_final_pcd": bool(args.write_final_pcd),
    }
    summary.update(_runtime_chunk_summary(manifests))
    summary_path = base_path / f"{args.case_prefix}_chunks_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if return_code not in (0, None) and not manifests:
        return int(return_code)
    if args.max_chunks is not None and len(manifests) < int(args.max_chunks):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
