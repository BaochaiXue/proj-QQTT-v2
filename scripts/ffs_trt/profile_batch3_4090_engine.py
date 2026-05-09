from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ffs_trt.validate_batch3_4090_engine import (  # noqa: E402
    DEFAULT_BATCH1_MODEL_DIR,
    DEFAULT_BATCH3_MODEL_DIR,
    DEFAULT_FFS_REPO,
    DEFAULT_REPLAY_DIR,
    SUMMARY_PATH as VALIDATION_SUMMARY_PATH,
    depth_array,
    discover_replay_kits,
    make_batch_samples,
    measured_window_indices,
    summarize_values,
)


SUMMARY_PATH = ROOT / "docs" / "generated" / "demo_v03_batch3_profile_100kit_4090.summary.json"
PER_KIT_PATH = ROOT / "docs" / "generated" / "demo_v03_batch3_profile_100kit_4090.per_kit.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile RTX 4090 FFS TensorRT batch=1 sequential vs batch=3 triplet.")
    parser.add_argument("--ffs-repo", type=Path, default=DEFAULT_FFS_REPO)
    parser.add_argument("--batch1-model-dir", type=Path, default=DEFAULT_BATCH1_MODEL_DIR)
    parser.add_argument("--batch3-model-dir", type=Path, default=DEFAULT_BATCH3_MODEL_DIR)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--warmup-kits", type=int, default=20)
    parser.add_argument("--measure-kits", type=int, default=100)
    parser.add_argument("--baseline-m", type=float, default=0.055)
    parser.add_argument("--summary-json", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--per-kit-jsonl", type=Path, default=PER_KIT_PATH)
    parser.add_argument("--validation-summary-json", type=Path, default=VALIDATION_SUMMARY_PATH)
    parser.add_argument("--batch1-only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


def _runner(*, ffs_repo: Path, model_dir: Path):
    from data_process.depth_backends.fast_foundation_stereo import FastFoundationStereoTensorRTRunner

    return FastFoundationStereoTensorRTRunner(ffs_repo=ffs_repo, model_dir=model_dir)


def _sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _load_validation_pass(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return str(data.get("validation", "")).lower() == "pass"


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    kits = discover_replay_kits(Path(args.replay_dir))
    measure_indices = measured_window_indices(
        total_count=len(kits),
        warmup_kits=int(args.warmup_kits),
        measure_kits=int(args.measure_kits),
    )
    warmup_indices = list(range(int(args.warmup_kits)))
    batch1 = _runner(ffs_repo=Path(args.ffs_repo), model_dir=Path(args.batch1_model_dir))
    batch3 = None if bool(args.batch1_only) else _runner(ffs_repo=Path(args.ffs_repo), model_dir=Path(args.batch3_model_dir))

    batch1_cam_ms = [[], [], []]
    batch1_triplet_ms: list[float] = []
    batch3_triplet_ms: list[float] = []
    batch3_nonzero = [[], [], []]
    per_kit_rows: list[dict[str, Any]] = []

    def run_batch1(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[float], float]:
        outputs: list[dict[str, Any]] = []
        cam_times: list[float] = []
        triplet_start = time.perf_counter()
        for sample in samples:
            _sync_cuda()
            start = time.perf_counter()
            output = batch1.run_pair(
                sample["left_image"],
                sample["right_image"],
                K_ir_left=sample["K_ir_left"],
                baseline_m=float(sample["baseline_m"]),
            )
            _sync_cuda()
            cam_times.append((time.perf_counter() - start) * 1000.0)
            outputs.append(output)
        _sync_cuda()
        triplet_ms = (time.perf_counter() - triplet_start) * 1000.0
        return outputs, cam_times, triplet_ms

    for kit_idx in warmup_indices:
        samples = make_batch_samples(kits[kit_idx], baseline_m=float(args.baseline_m))
        run_batch1(samples)
        if batch3 is not None:
            _sync_cuda()
            batch3.run_batch(samples)
            _sync_cuda()

    for ordinal, kit_idx in enumerate(measure_indices):
        kit = kits[kit_idx]
        samples = make_batch_samples(kit, baseline_m=float(args.baseline_m))
        _, cam_times, triplet_ms = run_batch1(samples)
        for cam_idx, value in enumerate(cam_times):
            batch1_cam_ms[cam_idx].append(float(value))
        batch1_triplet_ms.append(float(triplet_ms))

        row: dict[str, Any] = {
            "kit_index": int(kit.kit_index),
            "batch1_cam0_ms": float(cam_times[0]),
            "batch1_cam1_ms": float(cam_times[1]),
            "batch1_cam2_ms": float(cam_times[2]),
            "batch1_triplet_ms": float(triplet_ms),
        }
        if batch3 is not None:
            _sync_cuda()
            start = time.perf_counter()
            outputs = batch3.run_batch(samples)
            _sync_cuda()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            batch3_triplet_ms.append(float(elapsed_ms))
            row["batch3_triplet_ms"] = float(elapsed_ms)
            for cam_idx, output in enumerate(outputs):
                nonzero = int(np.count_nonzero(np.isfinite(depth_array(output)) & (depth_array(output) > 0.0)))
                batch3_nonzero[cam_idx].append(nonzero)
                row[f"batch3_cam{cam_idx}_depth_nonzero"] = nonzero
        per_kit_rows.append(row)
        if bool(args.debug) and (ordinal + 1) % 10 == 0:
            print(f"[batch3-profile] measured={ordinal + 1}/{len(measure_indices)} kit={kit.kit_index}", flush=True)

    summary: dict[str, Any] = {
        "warmup_kits": int(args.warmup_kits),
        "measure_kits": int(args.measure_kits),
        "warmup_included_in_stats": False,
        "measured_kits": int(len(measure_indices)),
        "batch1_cam0_ms": summarize_values(batch1_cam_ms[0]),
        "batch1_cam1_ms": summarize_values(batch1_cam_ms[1]),
        "batch1_cam2_ms": summarize_values(batch1_cam_ms[2]),
        "batch1_triplet_ms": summarize_values(batch1_triplet_ms),
        "batch1_only": bool(args.batch1_only),
    }
    if not bool(args.batch1_only):
        batch3_stats = summarize_values(batch3_triplet_ms)
        batch1_stats = summary["batch1_triplet_ms"]
        validation_passed = _load_validation_pass(Path(args.validation_summary_json))
        nonzero_ok = all(values and min(values) > 0 for values in batch3_nonzero)
        summary.update(
            {
                "batch3_triplet_ms": batch3_stats,
                "speedup_p50": float(batch1_stats["p50"] / batch3_stats["p50"]) if batch3_stats["p50"] > 0 else 0.0,
                "speedup_p90": float(batch1_stats["p90"] / batch3_stats["p90"]) if batch3_stats["p90"] > 0 else 0.0,
                "speedup_p99": float(batch1_stats["p99"] / batch3_stats["p99"]) if batch3_stats["p99"] > 0 else 0.0,
                "depth_nonzero_cam0_min": int(min(batch3_nonzero[0])) if batch3_nonzero[0] else 0,
                "depth_nonzero_cam1_min": int(min(batch3_nonzero[1])) if batch3_nonzero[1] else 0,
                "depth_nonzero_cam2_min": int(min(batch3_nonzero[2])) if batch3_nonzero[2] else 0,
                "validation_passed": bool(validation_passed),
                "batch3_profile_pass": bool(
                    validation_passed
                    and nonzero_ok
                    and batch3_stats["p50"] < batch1_stats["p50"]
                    and batch3_stats["p90"] < batch1_stats["p90"]
                ),
            }
        )

    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.per_kit_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.per_kit_jsonl, "w", encoding="utf-8") as handle:
        for row in per_kit_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_profile(args)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
