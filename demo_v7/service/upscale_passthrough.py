"""Upscale-stage passthrough: the mask-bbox crop WITHOUT SD x4 upscaling.

    (spawned by the prewarm pool when the operator disables 上采样)

CLI/lifecycle mirror of ``demo_v7.runtime.shape_prior.upscale`` (same flags, same
``StageProfileRun`` WAITING/GO/COMPLETED handshake, same profile-JSON field
set) so the untouched v6.2 warmup client can drive it interchangeably. The
output contract is the same file at the same path — ``high_resolution.png``
just holds the ORIGINAL-resolution crop (identical bbox math: mask bbox
square, x1.2 margin, PIL out-of-bounds padding) instead of the SD-upscaled
one. Downstream is dimension-agnostic: SAM3.1 segments whatever that file
is, and the generate backends resize their conditioning input anyway.

No model, no GPU: ``model_load_ms``/``inference_ms`` stay 0.0 and the
prewarmed process reaches WAITING immediately.
"""

import time


_MODULE_IMPORT_START_S = time.perf_counter()


import sys  # noqa: E402
from argparse import ArgumentParser  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

# Field-uniform with demo_v7.runtime.shape_prior.upscale for timeline consumers.
_ACTIVE_TIMING_FIELDS = (
    "module_import_ms",
    "model_load_ms",
    "input_crop_ms",
    "inference_ms",
    "output_write_ms",
)


def build_parser() -> ArgumentParser:
    """Build the command-line argument parser (upscale.py CLI mirror)."""
    parser = ArgumentParser()
    parser.add_argument("--img_path", type=str)
    parser.add_argument("--mask_path", type=str, default=None)
    parser.add_argument("--output_path", type=str)
    parser.add_argument(
        "--category",
        type=str,
        help="Accepted for upscale.py argv parity; no prompt is used.",
    )
    parser.add_argument("--profile-json", type=Path, default=None)
    parser.add_argument(
        "--wait-signal",
        dest="wait_signal",
        action="store_true",
        help="Signal WAITING (nothing to load), then block on stdin for GO.",
    )
    return parser


def crop_like_upscale(img_path: str, mask_path: str | None) -> Image.Image:
    """The exact crop upscale.py feeds SD: mask-bbox square, x1.2 margin."""
    low_res_img = Image.open(img_path).convert("RGB")
    if mask_path is None:
        return low_res_img
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    bbox = np.argwhere(mask > 0.8 * 255)
    bbox = (
        np.min(bbox[:, 1]),
        np.min(bbox[:, 0]),
        np.max(bbox[:, 1]),
        np.max(bbox[:, 0]),
    )
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1.2)
    bbox = (
        center[0] - size // 2,
        center[1] - size // 2,
        center[0] + size // 2,
        center[1] + size // 2,
    )
    return low_res_img.crop(bbox)  # type: ignore[arg-type]


def _elapsed_ms(start_s: float) -> float:
    """Local perf-counter delta in ms (timing.elapsed_ms mirror, pre-import)."""
    duration_ms = (time.perf_counter() - float(start_s)) * 1000.0
    if duration_ms < 0.0:
        raise ValueError(f"invalid timing duration: {duration_ms}")
    return float(duration_ms)


def _import_stage_profile_run():
    """Import the v6.2 timing contract (repo root via parent PYTHONPATH)."""
    try:
        from demo_v7.runtime.shape_prior.timing import StageProfileRun
    except ModuleNotFoundError:
        repo_root = str(Path(__file__).resolve().parents[2])
        if repo_root not in sys.path:
            sys.path.append(repo_root)
        from demo_v7.runtime.shape_prior.timing import StageProfileRun
    return StageProfileRun


def main(argv: list[str] | None = None) -> None:
    """Run the command-line entry point (upscale.py lifecycle mirror)."""
    module_import_ms = _elapsed_ms(_MODULE_IMPORT_START_S)
    args = build_parser().parse_args(argv)
    StageProfileRun = _import_stage_profile_run()

    run = StageProfileRun(
        stage="upscale",
        profile_json=args.profile_json,
        wait_signal=args.wait_signal,
        timing_ms=dict.fromkeys(
            (*_ACTIVE_TIMING_FIELDS, "go_wait_ms", "total_ms", "process_lifetime_ms"),
            0.0,
        ),
        active_fields=_ACTIVE_TIMING_FIELDS,
        process_started_s=_MODULE_IMPORT_START_S,
    )
    timing_ms = run.timing_ms
    timing_ms["module_import_ms"] = float(module_import_ms)

    if args.wait_signal:
        run.write_waiting()
        if not run.wait_for_go():
            return

    input_start_s = time.perf_counter()
    cropped = crop_like_upscale(args.img_path, args.mask_path)
    timing_ms["input_crop_ms"] = _elapsed_ms(input_start_s)

    output_start_s = time.perf_counter()
    cropped.save(args.output_path)
    timing_ms["output_write_ms"] = _elapsed_ms(output_start_s)
    run.write_completed()


if __name__ == "__main__":
    main()
