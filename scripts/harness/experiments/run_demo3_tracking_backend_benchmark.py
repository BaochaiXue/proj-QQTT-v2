#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np

try:
    import resource
except ImportError:  # pragma: no cover - Windows fallback for help/check paths
    resource = None


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_OUTPUT_ROOT = ROOT / "data" / "experiments" / "demo3_tracking_backend_benchmark"
DEFAULT_NUM_QUERY_POINTS = "100,256,512,1024"
AUTO_QUERY_POINTS = -1
MASK_SOURCE_DEFAULT = "mask_dir"
MASK_SOURCE_PHYSTWIN_UNION = "phystwin_union"
RESULT_COLUMNS = [
    "case_name",
    "backend",
    "available",
    "availability_reason",
    "camera_idx",
    "num_query_points",
    "num_frames",
    "model_ms_median",
    "model_ms_p95",
    "e2e_ms_median",
    "e2e_ms_p95",
    "model_fps",
    "e2e_fps",
    "three_camera_group_fps_serial",
    "gpu_memory_peak_mb",
    "visible_ratio_mean",
    "inside_mask_ratio_mean",
    "depth_valid_ratio_mean",
    "lifted_3d_count_mean",
    "tracking_quality_notes",
    "output_npz_path",
    "notes",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Demo 3 tracking backend benchmark on an aligned case replay.")
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--backends", type=str, default="cotracker3_online")
    parser.add_argument("--cameras", type=str, default="0,1,2")
    parser.add_argument("--num-query-points", type=str, default=DEFAULT_NUM_QUERY_POINTS)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument(
        "--query-mode",
        "--mode",
        dest="query_mode",
        choices=("object_sparse", "object_dense", "controller_sparse", "phystwin_dense"),
        default="object_sparse",
    )
    parser.add_argument("--sampling-strategy", choices=("random", "grid", "uniform_grid", "farthest", "phystwin_random"), default="grid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mask-source",
        choices=(MASK_SOURCE_DEFAULT, MASK_SOURCE_PHYSTWIN_UNION),
        default=MASK_SOURCE_DEFAULT,
        help="Mask layout used for query sampling/metrics. phystwin_union unions mask/{camera}/*/{frame}.png.",
    )
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--depth-source", choices=("native", "ffs"), default="native")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--install-probe-only", action="store_true", help="Only write backend availability/summary files.")
    parser.add_argument("--backend-availability-json", type=Path, default=None)
    parser.add_argument("--require-available", action="store_true", help="Fail if a requested backend is unavailable.")
    parser.add_argument("--write-phystwin-cotracker-dir", action="store_true")
    parser.add_argument(
        "--phystwin-compatible-export",
        action="store_true",
        help="Write root-level cotracker/{camera}.npz outputs and PhysTwin dense metadata.",
    )
    return parser.parse_args(argv)


def _parse_csv_ints(spec: str) -> list[int]:
    return [int(item.strip()) for item in str(spec).split(",") if item.strip()]


def _parse_query_point_requests(spec: str, query_mode: str) -> list[int]:
    normalized_spec = str(spec).strip().lower()
    normalized_mode = str(query_mode).strip().lower()
    if normalized_spec in {"auto", "phystwin", "phystwin_auto"}:
        return [AUTO_QUERY_POINTS]
    if normalized_mode == "phystwin_dense" and normalized_spec == DEFAULT_NUM_QUERY_POINTS:
        return [AUTO_QUERY_POINTS]
    return _parse_csv_ints(spec)


def _format_query_requests(query_requests: list[int]) -> list[int | str]:
    return ["phystwin_auto" if int(item) == AUTO_QUERY_POINTS else int(item) for item in query_requests]


def _parse_csv_strings(spec: str) -> list[str]:
    return [item.strip().lower() for item in str(spec).split(",") if item.strip()]


def _read_backend_availability_json(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "backends" in payload:
        return dict(payload["backends"])
    return dict(payload)


def _resolve_backend_names(spec: str, availability: dict[str, dict[str, Any]] | None = None) -> list[str]:
    from qqtt.tracking.registry import available_backend_names

    normalized = str(spec).strip().lower()
    if normalized == "all":
        return list(available_backend_names())
    if normalized == "auto_highperf":
        candidates = ["nvofa", "vpi_lk", "tapnext", "locotrack", "cotracker3_online"]
        if availability is None:
            return candidates
        selected = [name for name in candidates if bool(availability.get(name, {}).get("available", False))]
        return selected or ["cotracker3_online"]
    return _parse_csv_strings(spec)


def _read_png_sequence(case_root: Path, camera_idx: int, max_frames: int) -> list[np.ndarray]:
    color_dir = case_root / "color" / str(camera_idx)
    frames: list[np.ndarray] = []
    for path in sorted(color_dir.glob("*.png"), key=lambda item: int(item.stem) if item.stem.isdigit() else item.stem):
        if len(frames) >= int(max_frames):
            break
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read color frame: {path}")
        frames.append(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    if frames:
        return frames
    mp4_path = case_root / "color" / f"{camera_idx}.mp4"
    if not mp4_path.exists():
        raise FileNotFoundError(f"No PNG sequence or mp4 found for camera {camera_idx} under {case_root / 'color'}")
    cap = cv2.VideoCapture(str(mp4_path))
    try:
        while len(frames) < int(max_frames):
            ok, image_bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise FileNotFoundError(f"No frames read from {mp4_path}")
    return frames


def _find_mask_path(mask_root: Path, camera_idx: int, frame_idx: int) -> Path | None:
    candidates = (
        mask_root / str(camera_idx) / f"{frame_idx}.png",
        mask_root / str(camera_idx) / f"{frame_idx:06d}.png",
        mask_root / str(camera_idx) / f"{frame_idx}.npy",
        mask_root / str(camera_idx) / f"{frame_idx:06d}.npy",
        mask_root / str(camera_idx) / "0" / f"{frame_idx}.png",
        mask_root / str(camera_idx) / "0" / f"{frame_idx:06d}.png",
        mask_root / str(camera_idx) / "0" / f"{frame_idx}.npy",
        mask_root / str(camera_idx) / "0" / f"{frame_idx:06d}.npy",
        mask_root / f"{camera_idx}_{frame_idx}.png",
        mask_root / f"{camera_idx}_{frame_idx}.npy",
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_single_mask_file(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path)) > 0
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    return np.asarray(mask) > 0


def _phystwin_union_mask_paths(mask_root: Path, camera_idx: int, frame_idx: int) -> list[Path]:
    frame_tokens = (str(frame_idx), f"{frame_idx:06d}")
    paths: list[Path] = []
    camera_root = mask_root / str(camera_idx)
    for token in frame_tokens:
        paths.extend(sorted(camera_root.glob(f"*/{token}.png")))
        paths.extend(sorted(camera_root.glob(f"*/{token}.npy")))
    return paths


def _load_phystwin_union_mask(mask_root: Path, camera_idx: int, frame_idx: int, shape_hw: tuple[int, int]) -> np.ndarray:
    paths = _phystwin_union_mask_paths(mask_root, camera_idx, frame_idx)
    if not paths and frame_idx != 0:
        paths = _phystwin_union_mask_paths(mask_root, camera_idx, 0)
    if not paths:
        raise FileNotFoundError(f"No PhysTwin-style masks found under {mask_root / str(camera_idx)} for frame {frame_idx}.")
    union = np.zeros(shape_hw, dtype=bool)
    for path in paths:
        mask = _load_single_mask_file(path)
        if mask.shape != shape_hw:
            raise ValueError(f"Mask shape {mask.shape} from {path} does not match frame shape {shape_hw}.")
        union = np.logical_or(union, mask)
    return union


def _load_mask(
    mask_root: Path | None,
    camera_idx: int,
    frame_idx: int,
    shape_hw: tuple[int, int],
    *,
    mask_source: str,
) -> np.ndarray:
    if mask_root is None:
        return np.ones(shape_hw, dtype=bool)
    if str(mask_source) == MASK_SOURCE_PHYSTWIN_UNION:
        return _load_phystwin_union_mask(mask_root, camera_idx, frame_idx, shape_hw)
    path = _find_mask_path(mask_root, camera_idx, frame_idx)
    if path is None:
        return np.ones(shape_hw, dtype=bool)
    mask = _load_single_mask_file(path)
    if mask.shape != shape_hw:
        raise ValueError(f"Mask shape {mask.shape} from {path} does not match frame shape {shape_hw}.")
    return mask


def _latency_summary(ms: float) -> dict[str, float]:
    value = float(ms)
    return {
        "model_ms_median": value,
        "model_ms_p95": value,
        "e2e_ms_median": value,
        "e2e_ms_p95": value,
        "model_fps": 1000.0 / value if value > 0 else 0.0,
        "e2e_fps": 1000.0 / value if value > 0 else 0.0,
    }


def _load_lift_context(case_root: Path) -> dict[str, Any] | None:
    try:
        from data_process.visualization.calibration_io import load_calibration_transforms
        from data_process.visualization.io_case import get_case_intrinsics, get_depth_scale_list, load_case_metadata

        metadata = load_case_metadata(case_root)
        return {
            "metadata": metadata,
            "intrinsics": get_case_intrinsics(metadata),
            "depth_scales": get_depth_scale_list(metadata, len(metadata["serial_numbers"])),
            "c2w": load_calibration_transforms(
                case_root / "calibrate.pkl",
                serial_numbers=metadata["serial_numbers"],
                calibration_reference_serials=metadata.get("calibration_reference_serials", metadata["serial_numbers"]),
            ),
        }
    except Exception:
        return None


def _compute_lift_metrics(
    *,
    case_root: Path,
    lift_context: dict[str, Any] | None,
    camera_idx: int,
    depth_source: str,
    result: Any,
    masks: list[np.ndarray],
) -> dict[str, Any]:
    if lift_context is None:
        return {"depth_valid_ratio_mean": 0.0, "lifted_3d_count_mean": 0.0}
    try:
        from data_process.visualization.io_case import load_depth_frame
        from qqtt.tracking.lift import lift_tracks_to_world
        from qqtt.tracking.metrics import compute_3d_lift_metrics

        metadata = lift_context["metadata"]
        depth_kind = "realsense" if depth_source == "native" else "ffs"
        lifted_frames = []
        for frame_idx in range(min(result.tracks_yx.shape[0], len(masks))):
            _, depth, _ = load_depth_frame(
                case_dir=case_root,
                metadata=metadata,
                camera_idx=camera_idx,
                frame_idx=frame_idx,
                depth_source=depth_kind,
                use_float_ffs_depth_when_available=True,
            )
            lifted_frames.append(
                lift_tracks_to_world(
                    tracks_yx_t=result.tracks_yx[frame_idx],
                    visibility_t=result.visibility[frame_idx],
                    depth_uint16=depth,
                    depth_scale_m_per_unit=float(lift_context["depth_scales"][camera_idx] or 1.0),
                    mask=masks[frame_idx],
                    K=lift_context["intrinsics"][camera_idx],
                    c2w=lift_context["c2w"][camera_idx],
                    camera_idx=camera_idx,
                )
            )
        return compute_3d_lift_metrics(lifted_frames)
    except Exception as exc:
        return {"depth_valid_ratio_mean": 0.0, "lifted_3d_count_mean": 0.0, "lift_notes": f"{type(exc).__name__}: {exc}"}


def _effective_mask_source(query_mode: str, mask_source: str) -> str:
    if str(query_mode).strip().lower() == "phystwin_dense" and str(mask_source) == MASK_SOURCE_DEFAULT:
        return MASK_SOURCE_PHYSTWIN_UNION
    return str(mask_source)


def _effective_mask_root(*, case_root: Path, mask_dir: Path | None, mask_source: str) -> Path | None:
    if mask_dir is not None:
        return mask_dir.resolve()
    if str(mask_source) == MASK_SOURCE_PHYSTWIN_UNION:
        return case_root / "mask"
    return None


def _sample_query_points_for_request(
    mask: np.ndarray,
    *,
    requested_points: int,
    query_mode: str,
    sampling_strategy: str,
    seed: int,
) -> np.ndarray:
    from qqtt.tracking.sampling import sample_phystwin_dense, sample_query_points_from_mask

    normalized_mode = str(query_mode).strip().lower()
    if normalized_mode == "phystwin_dense" and int(requested_points) == AUTO_QUERY_POINTS:
        return sample_phystwin_dense(mask, seed=seed)
    if int(requested_points) == AUTO_QUERY_POINTS:
        raise ValueError("auto query count is only supported with --query-mode phystwin_dense.")
    strategy = str(sampling_strategy)
    if normalized_mode == "phystwin_dense" and strategy == "grid":
        strategy = "phystwin_random"
    return sample_query_points_from_mask(
        mask,
        num_points=int(requested_points),
        strategy=strategy,
        seed=seed,
        strict=normalized_mode == "phystwin_dense",
    )


def _write_outputs(output_dir: Path, rows: list[dict[str, Any]], availability: dict[str, dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in RESULT_COLUMNS})
    (output_dir / "availability.json").write_text(json.dumps(availability, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {"row_count": len(rows), "availability": availability, "results_csv": str(output_dir / "results.csv")}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Demo 3 Tracking Backend Benchmark", "", f"- rows: {len(rows)}"]
    for name, item in availability.items():
        state = "available" if item["available"] else "unavailable"
        lines.append(f"- {name}: {state} - {item['reason']}")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _max_rss_mb() -> float:
    if resource is None:
        return 0.0
    # Linux reports ru_maxrss in KiB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _torch_cuda_peak_mb() -> float:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def _write_profile_report(output_dir: Path, profile: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "profile.json").write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Demo 3 Tracking Benchmark Profile",
        "",
        f"- case: {profile.get('case_name', '')}",
        f"- backends: {', '.join(profile.get('backends', []))}",
        f"- cameras: {', '.join(str(item) for item in profile.get('cameras', []))}",
        f"- query_points: {', '.join(str(item) for item in profile.get('query_counts', []))}",
        f"- frames_requested: {profile.get('frames_requested', 0)}",
        f"- total_wall_ms: {profile.get('total_wall_ms', 0.0):.3f}",
        f"- frame_load_ms_total: {profile.get('frame_load_ms_total', 0.0):.3f}",
        f"- mask_load_ms_total: {profile.get('mask_load_ms_total', 0.0):.3f}",
        f"- max_rss_mb: {profile.get('max_rss_mb', 0.0):.3f}",
        f"- torch_cuda_peak_mb: {profile.get('torch_cuda_peak_mb', 0.0):.3f}",
        "",
        "## Serial Group FPS",
        "",
    ]
    group_rows = [row for row in profile.get("row_profiles", []) if row.get("camera_idx") == "all"]
    if group_rows:
        lines.append("| Backend | Points | Group FPS | E2E p50 ms | E2E p95 ms | Notes |")
        lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
        for row in group_rows:
            lines.append(
                "| {backend} | {points} | {fps:.3f} | {p50:.3f} | {p95:.3f} | {notes} |".format(
                    backend=row.get("backend", ""),
                    points=row.get("num_query_points", 0),
                    fps=float(row.get("three_camera_group_fps_serial", 0.0) or 0.0),
                    p50=float(row.get("e2e_ms_median", 0.0) or 0.0),
                    p95=float(row.get("e2e_ms_p95", 0.0) or 0.0),
                    notes=str(row.get("notes", "")),
                )
            )
    else:
        lines.append("No serial group rows were produced.")
    lines.extend(["", "## Per-Camera Rows", ""])
    camera_rows = [row for row in profile.get("row_profiles", []) if row.get("camera_idx") != "all"]
    if camera_rows:
        lines.append("| Backend | Camera | Points | Frames | E2E ms | Visible | Inside Mask | Depth Valid | Lifted | Notes |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for row in camera_rows:
            lines.append(
                "| {backend} | {camera} | {points} | {frames} | {e2e:.3f} | {visible:.3f} | {inside:.3f} | {depth:.3f} | {lifted:.3f} | {notes} |".format(
                    backend=row.get("backend", ""),
                    camera=row.get("camera_idx", ""),
                    points=row.get("num_query_points", 0),
                    frames=row.get("num_frames", 0),
                    e2e=float(row.get("e2e_ms_median", 0.0) or 0.0),
                    visible=float(row.get("visible_ratio_mean", 0.0) or 0.0),
                    inside=float(row.get("inside_mask_ratio_mean", 0.0) or 0.0),
                    depth=float(row.get("depth_valid_ratio_mean", 0.0) or 0.0),
                    lifted=float(row.get("lifted_3d_count_mean", 0.0) or 0.0),
                    notes=str(row.get("notes", "")),
                )
            )
    else:
        lines.append("No per-camera rows were produced.")
    (output_dir / "profile.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    from qqtt.tracking.io import save_cotracker_like_npz
    from qqtt.tracking.metrics import compute_2d_track_metrics
    from qqtt.tracking.registry import check_backend_availability, create_backend

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    case_root = Path(args.case_root).resolve()
    output_dir = Path(args.output_root).resolve() / case_root.name
    preloaded_availability = _read_backend_availability_json(args.backend_availability_json.resolve()) if args.backend_availability_json else None
    backends = _resolve_backend_names(str(args.backends), preloaded_availability)
    camera_indices = _parse_csv_ints(args.cameras)
    query_requests = _parse_query_point_requests(str(args.num_query_points), str(args.query_mode))
    mask_source = _effective_mask_source(str(args.query_mode), str(args.mask_source))
    mask_root = _effective_mask_root(case_root=case_root, mask_dir=args.mask_dir, mask_source=mask_source)
    phystwin_mode = str(args.query_mode).strip().lower() == "phystwin_dense"
    phystwin_export = bool(args.write_phystwin_cotracker_dir or args.phystwin_compatible_export or phystwin_mode)
    availability = {name: item.to_dict() for name, item in check_backend_availability(backends).items()}
    if preloaded_availability:
        for name in backends:
            if name in preloaded_availability:
                availability[name] = dict(preloaded_availability[name])
    if args.require_available:
        missing = [name for name, item in availability.items() if not item["available"]]
        if missing:
            raise RuntimeError(f"Required tracking backend(s) unavailable: {missing}")

    if bool(args.install_probe_only):
        _write_outputs(output_dir, [], availability)
        return {"output_dir": str(output_dir), "rows": [], "availability": availability}

    benchmark_start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    frame_load_ms_total = 0.0
    mask_load_ms_total = 0.0
    frame_cache: dict[int, list[np.ndarray]] = {}
    mask_cache: dict[int, list[np.ndarray]] = {}
    for camera_idx in camera_indices:
        load_start = time.perf_counter()
        frames = _read_png_sequence(case_root, camera_idx, int(args.frames))
        frame_load_ms_total += (time.perf_counter() - load_start) * 1000.0
        frame_cache[camera_idx] = frames
        shape_hw = frames[0].shape[:2]
        mask_start = time.perf_counter()
        mask_cache[camera_idx] = [
            _load_mask(mask_root, camera_idx, frame_idx, shape_hw, mask_source=mask_source)
            for frame_idx in range(len(frames))
        ]
        mask_load_ms_total += (time.perf_counter() - mask_start) * 1000.0

    lift_context = _load_lift_context(case_root)
    normalized_mode = str(args.query_mode)

    for backend_name in backends:
        backend_available = bool(availability[backend_name]["available"])
        backend = None
        backend_warmup_error = ""
        backend_load_ms = 0.0
        if backend_available:
            try:
                backend = create_backend(backend_name, device=str(args.device))
                warmup_camera = camera_indices[0]
                warmup_start = time.perf_counter()
                backend.initialize(
                    frame_cache[warmup_camera][:1],
                    np.zeros((1, 2), dtype=np.float32),
                    masks=mask_cache[warmup_camera][:1],
                )
                backend_load_ms = (time.perf_counter() - warmup_start) * 1000.0
            except Exception as exc:
                backend_available = False
                backend_warmup_error = f"{type(exc).__name__}: {exc}"
        for requested_points in query_requests:
            group_row_indices: list[int] = []
            camera_e2e_ms: list[float] = []
            camera_query_counts: list[int] = []
            for camera_idx in camera_indices:
                frames = frame_cache[camera_idx]
                masks = mask_cache[camera_idx]
                query_points = _sample_query_points_for_request(
                    masks[0],
                    requested_points=int(requested_points),
                    query_mode=str(args.query_mode),
                    sampling_strategy=str(args.sampling_strategy),
                    seed=int(args.seed),
                )
                camera_query_counts.append(int(len(query_points)))
                row: dict[str, Any] = {
                    "case_name": case_root.name,
                    "backend": backend_name,
                    "available": backend_available,
                    "availability_reason": availability[backend_name]["reason"],
                    "camera_idx": int(camera_idx),
                    "num_query_points": int(len(query_points)),
                    "num_frames": int(len(frames)),
                    "model_ms_median": 0.0,
                    "model_ms_p95": 0.0,
                    "e2e_ms_median": 0.0,
                    "e2e_ms_p95": 0.0,
                    "model_fps": 0.0,
                    "e2e_fps": 0.0,
                    "three_camera_group_fps_serial": 0.0,
                    "gpu_memory_peak_mb": 0.0,
                    "visible_ratio_mean": 0.0,
                    "inside_mask_ratio_mean": 0.0,
                    "depth_valid_ratio_mean": 0.0,
                    "lifted_3d_count_mean": 0.0,
                    "tracking_quality_notes": "",
                    "output_npz_path": "",
                    "notes": "" if backend_available else (backend_warmup_error or availability[backend_name]["reason"]),
                }
                if not backend_available or len(query_points) == 0:
                    rows.append(row)
                    group_row_indices.append(len(rows) - 1)
                    continue

                start = time.perf_counter()
                try:
                    if backend is None:
                        raise RuntimeError("Backend was not initialized.")
                    result = backend.track_sequence(
                        frames_rgb=frames,
                        query_points_yx=query_points,
                        camera_idx=camera_idx,
                        output_shape_hw=frames[0].shape[:2],
                    )
                except Exception as exc:
                    row["available"] = False
                    row["notes"] = f"{type(exc).__name__}: {exc}"
                    rows.append(row)
                    group_row_indices.append(len(rows) - 1)
                    continue

                elapsed_ms = (time.perf_counter() - start) * 1000.0
                camera_e2e_ms.append(elapsed_ms)
                row.update(_latency_summary(float(result.stats.get("model_run_ms", elapsed_ms))))
                row["gpu_memory_peak_mb"] = _torch_cuda_peak_mb()
                if backend_load_ms:
                    row["notes"] = f"backend_load_ms={backend_load_ms:.3f}"
                e2e = _latency_summary(elapsed_ms)
                row["e2e_ms_median"] = e2e["e2e_ms_median"]
                row["e2e_ms_p95"] = e2e["e2e_ms_p95"]
                row["e2e_fps"] = e2e["e2e_fps"]

                metrics_2d = compute_2d_track_metrics(result.tracks_yx, result.visibility, masks=masks)
                metrics_3d = _compute_lift_metrics(
                    case_root=case_root,
                    lift_context=lift_context,
                    camera_idx=camera_idx,
                    depth_source=str(args.depth_source),
                    result=result,
                    masks=masks,
                )
                existing_notes = str(row.get("notes", ""))
                lift_notes = str(metrics_3d.get("lift_notes", ""))
                combined_notes = "; ".join(item for item in (existing_notes, lift_notes) if item)
                row.update(
                    {
                        "visible_ratio_mean": metrics_2d["visible_ratio_mean"],
                        "inside_mask_ratio_mean": metrics_2d["inside_mask_ratio_mean"],
                        "depth_valid_ratio_mean": metrics_3d["depth_valid_ratio_mean"],
                        "lifted_3d_count_mean": metrics_3d["lifted_3d_count_mean"],
                        "tracking_quality_notes": "NVOFA/VPI are propagation baselines, not long-term TAP" if backend_name in {"nvofa", "vpi_lk"} else "",
                        "notes": combined_notes,
                    }
                )

                backend_dir = output_dir / backend_name / f"points_{len(query_points)}"
                npz_path = backend_dir / f"cam{camera_idx}.npz"
                metadata = {
                    "backend": backend_name,
                    "image_size": [int(frames[0].shape[0]), int(frames[0].shape[1])],
                    "query_mode": normalized_mode,
                    "depth_source": str(args.depth_source),
                    "mask_source": mask_source,
                    "phystwin_compatible": bool(phystwin_mode),
                }
                save_cotracker_like_npz(result, npz_path, camera_idx=camera_idx, metadata=metadata)
                save_cotracker_like_npz(result, backend_dir / "cotracker_like" / f"{camera_idx}.npz", camera_idx=camera_idx, metadata=metadata)
                if phystwin_export:
                    save_cotracker_like_npz(
                        result,
                        output_dir / "cotracker" / f"{camera_idx}.npz",
                        camera_idx=camera_idx,
                        metadata=metadata,
                    )
                row["output_npz_path"] = str(npz_path)
                (backend_dir / f"benchmark_cam{camera_idx}.json").write_text(
                    json.dumps(row, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                rows.append(row)
                group_row_indices.append(len(rows) - 1)

            group_time_s = sum(camera_e2e_ms) / 1000.0
            group_fps = float(len(camera_e2e_ms) / group_time_s) if group_time_s > 0 else 0.0
            for row_idx in group_row_indices:
                rows[row_idx]["three_camera_group_fps_serial"] = group_fps
            rows.append(
                {
                    "case_name": case_root.name,
                    "backend": backend_name,
                    "available": backend_available,
                    "availability_reason": availability[backend_name]["reason"],
                    "camera_idx": "all",
                    "num_query_points": int(np.median(camera_query_counts)) if camera_query_counts else 0,
                    "num_frames": min(len(frame_cache[idx]) for idx in camera_indices),
                    "model_ms_median": 0.0,
                    "model_ms_p95": 0.0,
                    "e2e_ms_median": float(np.median(camera_e2e_ms)) if camera_e2e_ms else 0.0,
                    "e2e_ms_p95": float(np.percentile(camera_e2e_ms, 95)) if camera_e2e_ms else 0.0,
                    "model_fps": 0.0,
                    "e2e_fps": group_fps,
                    "three_camera_group_fps_serial": group_fps,
                    "gpu_memory_peak_mb": 0.0,
                    "visible_ratio_mean": 0.0,
                    "inside_mask_ratio_mean": 0.0,
                    "depth_valid_ratio_mean": 0.0,
                    "lifted_3d_count_mean": 0.0,
                    "tracking_quality_notes": "serial group aggregate",
                    "output_npz_path": "",
                    "notes": "serial scheduling" if backend_available else availability[backend_name]["reason"],
                }
            )

    _write_outputs(output_dir, rows, availability)
    profile = {
        "case_name": case_root.name,
        "case_root": str(case_root),
        "output_dir": str(output_dir),
        "backends": backends,
        "cameras": camera_indices,
        "query_counts": _format_query_requests(query_requests),
        "query_mode": str(args.query_mode),
        "mask_source": mask_source,
        "mask_root": None if mask_root is None else str(mask_root),
        "phystwin_compatible_export": phystwin_export,
        "frames_requested": int(args.frames),
        "frame_load_ms_total": float(frame_load_ms_total),
        "mask_load_ms_total": float(mask_load_ms_total),
        "total_wall_ms": float((time.perf_counter() - benchmark_start) * 1000.0),
        "max_rss_mb": _max_rss_mb(),
        "torch_cuda_peak_mb": _torch_cuda_peak_mb(),
        "row_profiles": rows,
    }
    _write_profile_report(output_dir, profile)
    return {"output_dir": str(output_dir), "rows": rows, "availability": availability, "profile": profile}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_benchmark(args)
    print(f"Demo 3 tracking benchmark outputs written to {summary['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
