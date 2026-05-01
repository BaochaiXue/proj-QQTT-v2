from __future__ import annotations

import argparse
import csv
import json
from argparse import Namespace
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_process.depth_backends.ffs_defaults import (
    DEFAULT_FFS_MAX_DISP,
    DEFAULT_FFS_MODEL_PATH,
    DEFAULT_FFS_REPO,
    DEFAULT_FFS_SCALE,
    DEFAULT_FFS_VALID_ITERS,
)


def _parse_csv_items(value: str, *, item_type=str) -> list[Any]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if not items:
        raise ValueError(f"Expected a non-empty comma-separated value, got {value!r}.")
    return [item_type(item) for item in items]


def _experiment_id(mode: str, threshold: float) -> str:
    return f"mode_{mode}_threshold_{float(threshold):.2f}"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _summarize_depth_outputs(case_dir: Path, *, camera_count: int, frame_count: int) -> dict[str, float]:
    import numpy as np

    ratios: list[float] = []
    point_counts: list[float] = []
    for camera_idx in range(int(camera_count)):
        for frame_idx in range(int(frame_count)):
            depth_path = case_dir / "depth" / str(camera_idx) / f"{frame_idx}.npy"
            if not depth_path.is_file():
                depth_path = case_dir / "depth_ffs" / str(camera_idx) / f"{frame_idx}.npy"
            if not depth_path.is_file():
                continue
            depth = np.load(depth_path)
            nonzero = int(np.count_nonzero(depth))
            ratios.append(float(nonzero / max(1, depth.size)))
            point_counts.append(float(nonzero))
    if not ratios:
        return {
            "depth_nonzero_ratio_mean": 0.0,
            "point_count_mean": 0.0,
        }
    return {
        "depth_nonzero_ratio_mean": float(sum(ratios) / len(ratios)),
        "point_count_mean": float(sum(point_counts) / len(point_counts)),
    }


def _stats_summary_value(stats_summary: dict[str, Any], key: str, default: float) -> float:
    value = stats_summary.get(key)
    if value is None:
        return float(default)
    return float(value)


def _select_best(rows: list[dict[str, Any]], *, min_valid_ratio: float, max_low_confidence_reject_ratio: float) -> dict[str, Any]:
    if not rows:
        return {
            "best_by_hole_ratio": None,
            "best_by_valid_ratio": None,
            "best_balanced": None,
            "recommendation": None,
        }

    best_by_hole = min(rows, key=lambda row: float(row["hole_ratio_after_confidence_mean"]))
    best_by_valid = max(rows, key=lambda row: float(row["valid_ratio_after_confidence_mean"]))
    candidates = [
        row
        for row in rows
        if float(row["valid_ratio_after_confidence_mean"]) >= float(min_valid_ratio)
        and float(row["low_confidence_reject_ratio_mean"]) <= float(max_low_confidence_reject_ratio)
    ]
    if not candidates:
        candidates = [best_by_valid]

    def balanced_key(row: dict[str, Any]) -> tuple[float, float, float]:
        floating_score = row.get("floating_artifact_score")
        primary = float(floating_score) if floating_score not in (None, "") else float(row["hole_ratio_after_confidence_mean"])
        return (
            primary,
            float(row["low_confidence_reject_ratio_mean"]),
            -float(row["valid_ratio_after_confidence_mean"]),
        )

    best_balanced = min(candidates, key=balanced_key)
    reason = (
        "selected lowest hole ratio among configurations that preserve enough valid depth"
        if candidates and best_balanced is not best_by_valid
        else "no configuration met the balanced validity/rejection targets; selected highest valid ratio"
    )
    return {
        "best_by_hole_ratio": _experiment_id(str(best_by_hole["mode"]), float(best_by_hole["threshold"])),
        "best_by_valid_ratio": _experiment_id(str(best_by_valid["mode"]), float(best_by_valid["threshold"])),
        "best_balanced": _experiment_id(str(best_balanced["mode"]), float(best_balanced["threshold"])),
        "recommendation": {
            "mode": str(best_balanced["mode"]),
            "threshold": float(best_balanced["threshold"]),
            "reason": (
                f"{reason}; confidence thresholds trade fewer low-confidence floating artifacts "
                "against more holes."
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep FFS confidence filtering modes and thresholds on a recorded case.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base_path", type=Path, default=ROOT / "data_collect")
    parser.add_argument("--output_root", type=Path, default=ROOT / "result" / "ffs_confidence_filter_sweep")
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--ffs_repo", type=str, default=str(DEFAULT_FFS_REPO))
    parser.add_argument("--ffs_model_path", type=str, default=str(DEFAULT_FFS_MODEL_PATH))
    parser.add_argument("--ffs_scale", type=float, default=DEFAULT_FFS_SCALE)
    parser.add_argument("--ffs_valid_iters", type=int, default=DEFAULT_FFS_VALID_ITERS)
    parser.add_argument("--ffs_max_disp", type=int, default=DEFAULT_FFS_MAX_DISP)
    parser.add_argument("--modes", type=str, default="margin,max_softmax,entropy,variance")
    parser.add_argument("--thresholds", type=str, default="0.3,0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--depth_min_m", type=float, default=0.2)
    parser.add_argument("--depth_max_m", type=float, default=1.5)
    parser.add_argument("--min_valid_ratio", type=float, default=0.60)
    parser.add_argument("--max_low_confidence_reject_ratio", type=float, default=0.50)
    parser.add_argument("--write_debug", action="store_true")
    return parser


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    from data_process.record_data_align import align_case

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    modes = _parse_csv_items(args.modes, item_type=str)
    thresholds = _parse_csv_items(args.thresholds, item_type=float)
    valid_modes = {"margin", "max_softmax", "entropy", "variance"}
    invalid_modes = [mode for mode in modes if mode not in valid_modes]
    if invalid_modes:
        raise ValueError(f"Unsupported confidence modes: {invalid_modes}")

    try:
        import open3d  # noqa: F401

        open3d_available = True
    except Exception:
        open3d_available = False

    rows: list[dict[str, Any]] = []
    experiments: list[dict[str, Any]] = []
    for mode in modes:
        for threshold in thresholds:
            experiment_id = _experiment_id(mode, threshold)
            experiment_dir = output_root / experiment_id
            aligned_root = experiment_dir / "aligned_case"
            run_args = Namespace(
                base_path=Path(args.base_path).resolve(),
                case_name=str(args.case_name),
                output_path=aligned_root,
                start=int(args.start),
                end=int(args.end),
                fps=None,
                write_mp4=False,
                depth_backend="ffs",
                ffs_repo=str(args.ffs_repo),
                ffs_model_path=str(args.ffs_model_path),
                ffs_scale=float(args.ffs_scale),
                ffs_valid_iters=int(args.ffs_valid_iters),
                ffs_max_disp=int(args.ffs_max_disp),
                ffs_radius_outlier_filter=False,
                ffs_radius_outlier_radius_m=0.01,
                ffs_radius_outlier_nb_points=40,
                ffs_native_like_postprocess=False,
                ffs_confidence_mode=str(mode),
                ffs_confidence_threshold=float(threshold),
                ffs_confidence_depth_min_m=float(args.depth_min_m),
                ffs_confidence_depth_max_m=float(args.depth_max_m),
                write_ffs_confidence_debug=bool(args.write_debug),
                write_ffs_valid_mask_debug=bool(args.write_debug),
                write_ffs_float_m=False,
                fail_if_no_ir_stereo=True,
            )
            start_s = time.monotonic()
            aligned_metadata = align_case(run_args)
            runtime_sec = time.monotonic() - start_s
            case_dir = aligned_root / str(args.case_name)
            metadata_ext = _load_json(case_dir / "metadata_ext.json")
            confidence_summary = metadata_ext.get("ffs_confidence_filter", {}).get("stats_summary", {})
            frame_count = int(aligned_metadata["frame_num"])
            camera_count = len(aligned_metadata["serial_numbers"])
            depth_summary = _summarize_depth_outputs(case_dir, camera_count=camera_count, frame_count=frame_count)
            row = {
                "case_name": str(args.case_name),
                "mode": str(mode),
                "threshold": float(threshold),
                "frame_count": frame_count,
                "camera_count": camera_count,
                "valid_ratio_after_confidence_mean": _stats_summary_value(
                    confidence_summary,
                    "valid_ratio_after_confidence_mean",
                    depth_summary["depth_nonzero_ratio_mean"],
                ),
                "hole_ratio_after_confidence_mean": _stats_summary_value(
                    confidence_summary,
                    "hole_ratio_after_confidence_mean",
                    1.0 - depth_summary["depth_nonzero_ratio_mean"],
                ),
                "low_confidence_reject_ratio_mean": _stats_summary_value(
                    confidence_summary,
                    "low_confidence_reject_ratio_mean",
                    0.0,
                ),
                "point_count_mean": depth_summary["point_count_mean"],
                "depth_nonzero_ratio_mean": depth_summary["depth_nonzero_ratio_mean"],
                "runtime_sec": float(runtime_sec),
                "fps": float(frame_count / max(runtime_sec, 1e-9)),
                "radius_outlier_ratio_before": "",
                "radius_outlier_ratio_after": "",
                "floating_artifact_score": "",
            }
            rows.append(row)
            experiment_summary = {
                "experiment_id": experiment_id,
                "experiment_dir": str(experiment_dir.resolve()),
                "aligned_case_dir": str(case_dir.resolve()),
                "row": row,
                "confidence_filter": metadata_ext.get("ffs_confidence_filter", {}),
                "open3d_available": open3d_available,
                "outlier_metrics_enabled": False,
            }
            _write_json(experiment_dir / "stats.json", experiment_summary)
            experiments.append(experiment_summary)

    csv_path = output_root / "results.csv"
    fieldnames = [
        "case_name",
        "mode",
        "threshold",
        "frame_count",
        "camera_count",
        "valid_ratio_after_confidence_mean",
        "hole_ratio_after_confidence_mean",
        "low_confidence_reject_ratio_mean",
        "point_count_mean",
        "depth_nonzero_ratio_mean",
        "runtime_sec",
        "fps",
        "radius_outlier_ratio_before",
        "radius_outlier_ratio_after",
        "floating_artifact_score",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    selection = _select_best(
        rows,
        min_valid_ratio=float(args.min_valid_ratio),
        max_low_confidence_reject_ratio=float(args.max_low_confidence_reject_ratio),
    )
    summary = {
        "case_name": str(args.case_name),
        "base_path": str(Path(args.base_path).resolve()),
        "output_root": str(output_root),
        "results_csv": str(csv_path.resolve()),
        "modes": modes,
        "thresholds": thresholds,
        "depth_min_m": float(args.depth_min_m),
        "depth_max_m": float(args.depth_max_m),
        "min_valid_ratio": float(args.min_valid_ratio),
        "max_low_confidence_reject_ratio": float(args.max_low_confidence_reject_ratio),
        "open3d_available": open3d_available,
        "outlier_metrics_enabled": False,
        "experiments": experiments,
        **selection,
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_sweep(args)
    print(f"FFS confidence sweep results written to {summary['output_root']}")
    print(json.dumps(summary["recommendation"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
