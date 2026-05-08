from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_BATCH1_MODEL_DIR = Path(
    "/home/xinjie/proj-QQTT-v2/data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864"
)
DEFAULT_BATCH3_MODEL_DIR = (
    ROOT
    / "data"
    / "experiments"
    / "ffs_trt_4090_848x480_pad864_builderopt5_batch3"
    / "engines"
    / "model_20-30-48_iters_4_res_480x864_batch3"
)
DEFAULT_REPLAY_DIR = ROOT / "result" / "demo_v0_3_ir_triplet_100kits_848x480"
DEFAULT_FFS_REPO = Path("/home/xinjie/Fast-FoundationStereo")
SUMMARY_PATH = ROOT / "docs" / "generated" / "demo_v03_batch3_validate_100kit_4090.summary.json"


def summarize_values(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {key: 0.0 for key in ("avg", "min", "max", "p50", "p90", "p95", "p99")}
    return {
        "avg": float(np.mean(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def measured_window_indices(*, total_count: int, warmup_kits: int, measure_kits: int) -> list[int]:
    start = int(warmup_kits)
    stop = start + int(measure_kits)
    if start < 0 or int(measure_kits) <= 0:
        raise ValueError("warmup_kits must be non-negative and measure_kits must be positive")
    if stop > int(total_count):
        raise ValueError(f"Replay folder has {total_count} kits, need warmup+measure={stop}.")
    return list(range(start, stop))


def camera_order_diagonal_pass(diff_matrix: np.ndarray) -> bool:
    matrix = np.asarray(diff_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square diff matrix, got {matrix.shape}.")
    row_best = np.argmin(matrix, axis=1)
    col_best = np.argmin(matrix, axis=0)
    expected = np.arange(matrix.shape[0])
    return bool(np.array_equal(row_best, expected) and np.array_equal(col_best, expected))


@dataclass(frozen=True)
class ReplayFrame:
    left_path: Path
    right_path: Path


@dataclass(frozen=True)
class ReplayKit:
    kit_index: int
    frames: list[ReplayFrame]


@dataclass
class DiffAccumulator:
    max_samples: int = 1_000_000
    total: float = 0.0
    count: int = 0
    min_value: float = math.inf
    max_value: float = 0.0
    samples: list[np.ndarray] = field(default_factory=list)

    def update(self, values: np.ndarray) -> None:
        flat = np.asarray(values, dtype=np.float32).reshape(-1)
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            return
        self.total += float(np.sum(flat, dtype=np.float64))
        self.count += int(flat.size)
        self.min_value = min(self.min_value, float(np.min(flat)))
        self.max_value = max(self.max_value, float(np.max(flat)))
        sample_count = sum(int(item.size) for item in self.samples)
        remaining = max(0, int(self.max_samples) - sample_count)
        if remaining > 0:
            stride = max(1, int(math.ceil(flat.size / remaining)))
            self.samples.append(np.asarray(flat[::stride][:remaining], dtype=np.float32))

    def summary(self) -> dict[str, float]:
        if self.count <= 0:
            return {key: 0.0 for key in ("avg", "min", "max", "p50", "p90", "p95", "p99")}
        sample = np.concatenate(self.samples) if self.samples else np.array([0.0], dtype=np.float32)
        summary = summarize_values(sample)
        summary["avg"] = float(self.total / self.count)
        summary["min"] = float(self.min_value)
        summary["max"] = float(self.max_value)
        return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate RTX 4090 FFS TensorRT batch=3 engine against batch=1 sequential output.")
    parser.add_argument("--ffs-repo", type=Path, default=DEFAULT_FFS_REPO)
    parser.add_argument("--batch1-model-dir", type=Path, default=DEFAULT_BATCH1_MODEL_DIR)
    parser.add_argument("--batch3-model-dir", type=Path, default=DEFAULT_BATCH3_MODEL_DIR)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--warmup-kits", type=int, default=20)
    parser.add_argument("--measure-kits", type=int, default=100)
    parser.add_argument("--depth-scale-m-per-unit", type=float, default=0.001)
    parser.add_argument("--baseline-m", type=float, default=0.055)
    parser.add_argument("--max-diff-samples", type=int, default=1_000_000)
    parser.add_argument("--summary-json", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--debug", action="store_true")
    return parser


def _image_files(directory: Path) -> list[Path]:
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.npy")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(directory.glob(pattern))
    return sorted(path for path in files if path.is_file())


def discover_replay_kits(replay_dir: Path, *, camera_count: int = 3) -> list[ReplayKit]:
    replay_dir = Path(replay_dir)
    frames_by_camera: list[list[ReplayFrame]] = []
    for cam_idx in range(int(camera_count)):
        cam_dir = replay_dir / f"cam{cam_idx}"
        left_dir = cam_dir / "left"
        right_dir = cam_dir / "right"
        left_files = _image_files(left_dir)
        right_files = _image_files(right_dir)
        if not left_files or not right_files:
            raise FileNotFoundError(f"Missing replay left/right frames for cam{cam_idx}: {left_dir}, {right_dir}")
        count = min(len(left_files), len(right_files))
        frames_by_camera.append(
            [ReplayFrame(left_path=left_files[idx], right_path=right_files[idx]) for idx in range(count)]
        )
    kit_count = min(len(items) for items in frames_by_camera)
    return [
        ReplayKit(kit_index=idx, frames=[frames_by_camera[cam_idx][idx] for cam_idx in range(camera_count)])
        for idx in range(kit_count)
    ]


def load_ir_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        image = np.load(path)
    else:
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Failed to load replay image: {path}")
    image = np.asarray(image)
    if image.ndim == 3:
        image = image[..., 0]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image, dtype=np.uint8)


def default_k_ir(*, width: int, height: int) -> np.ndarray:
    fx = 600.0 * (float(width) / 848.0)
    fy = 600.0 * (float(height) / 480.0)
    return np.array([[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def make_batch_samples(kit: ReplayKit, *, baseline_m: float) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for frame in kit.frames:
        left = load_ir_image(frame.left_path)
        right = load_ir_image(frame.right_path)
        if left.shape != right.shape:
            raise ValueError(f"Left/right shape mismatch: {frame.left_path} {left.shape} vs {frame.right_path} {right.shape}")
        height, width = left.shape[:2]
        samples.append(
            {
                "left_image": left,
                "right_image": right,
                "K_ir_left": default_k_ir(width=width, height=height),
                "baseline_m": float(baseline_m),
            }
        )
    return samples


def depth_array(output: dict[str, Any]) -> np.ndarray:
    if "depth_ir_left_m" not in output:
        raise KeyError("FFS output missing depth_ir_left_m")
    return np.asarray(output["depth_ir_left_m"], dtype=np.float32)


def finite_abs_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    left = np.asarray(a, dtype=np.float32)
    right = np.asarray(b, dtype=np.float32)
    if left.shape != right.shape:
        raise ValueError(f"Depth shape mismatch: {left.shape} vs {right.shape}")
    finite = np.isfinite(left) & np.isfinite(right)
    return np.abs(left[finite] - right[finite]).astype(np.float32, copy=False)


def _runner(*, ffs_repo: Path, model_dir: Path):
    from data_process.depth_backends.fast_foundation_stereo import FastFoundationStereoTensorRTRunner

    return FastFoundationStereoTensorRTRunner(ffs_repo=ffs_repo, model_dir=model_dir)


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    replay_dir = Path(args.replay_dir)
    kits = discover_replay_kits(replay_dir)
    measure_indices = measured_window_indices(
        total_count=len(kits),
        warmup_kits=int(args.warmup_kits),
        measure_kits=int(args.measure_kits),
    )
    batch1 = _runner(ffs_repo=Path(args.ffs_repo), model_dir=Path(args.batch1_model_dir))
    batch3 = _runner(ffs_repo=Path(args.ffs_repo), model_dir=Path(args.batch3_model_dir))

    diff_accumulators = [DiffAccumulator(max_samples=int(args.max_diff_samples)) for _ in range(3)]
    nonzero_min = [10**18, 10**18, 10**18]
    order_matrix_sum = np.zeros((3, 3), dtype=np.float64)
    measured = 0

    for idx, kit_idx in enumerate(measure_indices):
        kit = kits[kit_idx]
        samples = make_batch_samples(kit, baseline_m=float(args.baseline_m))
        batch1_outputs = [
            batch1.run_pair(
                sample["left_image"],
                sample["right_image"],
                K_ir_left=sample["K_ir_left"],
                baseline_m=float(sample["baseline_m"]),
            )
            for sample in samples
        ]
        batch3_outputs = batch3.run_batch(samples)
        if len(batch3_outputs) != 3:
            raise RuntimeError(f"batch3 output count mismatch for kit {kit.kit_index}: {len(batch3_outputs)}")

        batch1_depths = [depth_array(output) for output in batch1_outputs]
        batch3_depths = [depth_array(output) for output in batch3_outputs]
        for cam_idx, depth in enumerate(batch3_depths):
            nonzero = int(np.count_nonzero(np.isfinite(depth) & (depth > 0.0)))
            nonzero_min[cam_idx] = min(nonzero_min[cam_idx], nonzero)
            diff = finite_abs_diff(batch3_depths[cam_idx], batch1_depths[cam_idx])
            diff_accumulators[cam_idx].update(diff)
        for row in range(3):
            for col in range(3):
                pair_diff = finite_abs_diff(batch3_depths[row], batch1_depths[col])
                order_matrix_sum[row, col] += float(np.mean(pair_diff)) if pair_diff.size else float("inf")
        measured += 1
        if bool(args.debug) and (idx + 1) % 10 == 0:
            print(f"[batch3-validate] measured={idx + 1}/{len(measure_indices)} kit={kit.kit_index}", flush=True)

    order_matrix = order_matrix_sum / max(1, measured)
    diff_summaries = [acc.summary() for acc in diff_accumulators]
    order_pass = camera_order_diagonal_pass(order_matrix)
    fail_reasons: list[str] = []
    if any(value <= 0 for value in nonzero_min):
        fail_reasons.append("one or more cameras produced zero nonzero depth pixels")
    if not order_pass:
        fail_reasons.append("camera order diagonal check failed")
    for cam_idx, summary in enumerate(diff_summaries):
        if summary["p50"] > 0.02:
            fail_reasons.append(f"cam{cam_idx} median_abs_diff_m {summary['p50']:.6f} > 0.02")
        if summary["p95"] > 0.10:
            fail_reasons.append(f"cam{cam_idx} p95_abs_diff_m {summary['p95']:.6f} > 0.10")

    result = {
        "validation": "pass" if not fail_reasons else "fail",
        "fail_reasons": fail_reasons,
        "replay_dir": str(replay_dir),
        "warmup_kits": int(args.warmup_kits),
        "measure_kits": int(args.measure_kits),
        "measured_kits": int(measured),
        "batch3_engine_load": "pass",
        "batch3_output_count": 3,
        "depth_nonzero_cam0_min": int(nonzero_min[0]),
        "depth_nonzero_cam1_min": int(nonzero_min[1]),
        "depth_nonzero_cam2_min": int(nonzero_min[2]),
        "cam0_abs_diff_m": diff_summaries[0],
        "cam1_abs_diff_m": diff_summaries[1],
        "cam2_abs_diff_m": diff_summaries[2],
        "camera_order_diff_matrix_mean_m": order_matrix.tolist(),
        "camera_order_check": "pass" if order_pass else "fail",
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_validation(args)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["validation"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
