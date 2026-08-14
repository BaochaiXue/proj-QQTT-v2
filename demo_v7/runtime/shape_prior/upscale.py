import time


_MODULE_IMPORT_START_S = time.perf_counter()


from argparse import ArgumentParser  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from diffusers import StableDiffusionUpscalePipeline  # noqa: E402
from PIL import Image  # noqa: E402

from demo_v7.runtime.shape_prior.timing import (  # noqa: E402
    StageProfileRun,
    elapsed_ms,
)

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


def main(argv: list[str] | None = None) -> None:
    """Run the command-line entry point."""
    module_import_ms = elapsed_ms(_MODULE_IMPORT_START_S)
    args = build_parser().parse_args(argv)

    run = StageProfileRun(
        stage="upscale",
        profile_json=args.profile_json,
        wait_signal=args.wait_signal,
        timing_ms={
            "module_import_ms": module_import_ms,
            "model_load_ms": 0.0,
            "input_crop_ms": 0.0,
            "inference_ms": 0.0,
            "output_write_ms": 0.0,
            "go_wait_ms": 0.0,
            "total_ms": 0.0,
            "process_lifetime_ms": 0.0,
        },
        active_fields=_ACTIVE_TIMING_FIELDS,
        process_started_s=_MODULE_IMPORT_START_S,
    )
    timing_ms = run.timing_ms

    model_load_start_s = time.perf_counter()
    pipeline = load_pipeline()
    timing_ms["model_load_ms"] = elapsed_ms(model_load_start_s)

    if args.wait_signal:
        run.write_waiting()
        if not run.wait_for_go():
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
    run.write_completed()


if __name__ == "__main__":
    main()
