import time


_MODULE_IMPORT_START_S = time.perf_counter()


from argparse import ArgumentParser  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from diffusers import StableDiffusionUpscalePipeline  # noqa: E402
from PIL import Image  # noqa: E402

from demo_v6_2.shape_prior_timing import (  # noqa: E402
    STAGE_PROFILE_STATUS_COMPLETED,
    STAGE_PROFILE_STATUS_WAITING,
    elapsed_ms,
    write_stage_profile,
)
from demo_v6_2.utils import stage_prewarm  # noqa: E402

MODEL_ID = "stabilityai/stable-diffusion-x4-upscaler"
_ACTIVE_TIMING_FIELDS = (
    "module_import_ms",
    "model_load_ms",
    "input_crop_ms",
    "inference_ms",
    "output_write_ms",
)


def build_parser() -> ArgumentParser:
    """Build the command-line argument parser."""
    parser = ArgumentParser()
    parser.add_argument(
        "--img_path",
        type=str,
    )
    parser.add_argument("--mask_path", type=str, default=None)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--category", type=str)
    parser.add_argument("--profile-json", type=Path, default=None)
    parser.add_argument(
        "--wait-signal",
        dest="wait_signal",
        action="store_true",
        help="Load the model, then block on stdin for GO before upscaling.",
    )
    return parser


def load_pipeline() -> StableDiffusionUpscalePipeline:
    """Load the upscale model and scheduler onto the GPU."""
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    )
    return pipeline.to("cuda")


def _upscale_image_with_timing(
    pipeline: StableDiffusionUpscalePipeline,
    *,
    img_path: str,
    mask_path: str | None,
    output_path: str,
    category: str,
) -> dict[str, float]:
    """Run the unchanged upscale algorithm and return its active timings."""
    input_start_s = time.perf_counter()
    low_res_img = Image.open(img_path).convert("RGB")
    if mask_path is not None:
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
        low_res_img = low_res_img.crop(bbox)  # type: ignore
    input_crop_ms = elapsed_ms(input_start_s)

    prompt = f"Hand manipulates a {category}."

    inference_start_s = time.perf_counter()
    upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
    inference_ms = elapsed_ms(inference_start_s)

    output_start_s = time.perf_counter()
    upscaled_image.save(output_path)
    output_write_ms = elapsed_ms(output_start_s)
    return {
        "input_crop_ms": input_crop_ms,
        "inference_ms": inference_ms,
        "output_write_ms": output_write_ms,
    }


def upscale_image(
    pipeline: StableDiffusionUpscalePipeline,
    *,
    img_path: str,
    mask_path: str | None,
    output_path: str,
    category: str,
) -> None:
    """Crop to the mask bbox and upscale, mirroring the original stage."""
    _upscale_image_with_timing(
        pipeline,
        img_path=img_path,
        mask_path=mask_path,
        output_path=output_path,
        category=category,
    )


def _active_total_ms(timing_ms: dict[str, float]) -> float:
    """Return active stage time, intentionally excluding the GO idle wait."""
    return float(sum(timing_ms[field] for field in _ACTIVE_TIMING_FIELDS))


def main(argv: list[str] | None = None) -> None:
    """Run the command-line entry point."""
    module_import_ms = elapsed_ms(_MODULE_IMPORT_START_S)
    args = build_parser().parse_args(argv)

    execution_mode = "prewarmed" if args.wait_signal else "cold"
    timing_ms = {
        "module_import_ms": module_import_ms,
        "model_load_ms": 0.0,
        "input_crop_ms": 0.0,
        "inference_ms": 0.0,
        "output_write_ms": 0.0,
        "go_wait_ms": 0.0,
        "total_ms": 0.0,
        "process_lifetime_ms": 0.0,
    }

    model_load_start_s = time.perf_counter()
    pipeline = load_pipeline()
    timing_ms["model_load_ms"] = elapsed_ms(model_load_start_s)

    ready_wall_time_s: float | None = None
    if args.wait_signal:
        ready_wall_time_s = time.time()
        timing_ms["total_ms"] = _active_total_ms(timing_ms)
        timing_ms["process_lifetime_ms"] = elapsed_ms(_MODULE_IMPORT_START_S)
        write_stage_profile(
            args.profile_json,
            stage="upscale",
            status=STAGE_PROFILE_STATUS_WAITING,
            execution_mode=execution_mode,
            timing_ms=timing_ms,
            ready_wall_time_s=ready_wall_time_s,
        )
        go_wait_start_s = time.perf_counter()
        should_run = stage_prewarm.wait_for_go("upscale")
        timing_ms["go_wait_ms"] = elapsed_ms(go_wait_start_s)
        if not should_run:
            return

    timing_ms.update(
        _upscale_image_with_timing(
            pipeline,
            img_path=args.img_path,
            mask_path=args.mask_path,
            output_path=args.output_path,
            category=args.category,
        )
    )
    timing_ms["total_ms"] = _active_total_ms(timing_ms)
    timing_ms["process_lifetime_ms"] = elapsed_ms(_MODULE_IMPORT_START_S)
    write_stage_profile(
        args.profile_json,
        stage="upscale",
        status=STAGE_PROFILE_STATUS_COMPLETED,
        execution_mode=execution_mode,
        timing_ms=timing_ms,
        ready_wall_time_s=ready_wall_time_s,
    )


if __name__ == "__main__":
    main()
