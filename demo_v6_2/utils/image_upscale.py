from argparse import ArgumentParser

import cv2
import numpy as np
import torch
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image

from demo_v6_2.utils import stage_prewarm

MODEL_ID = "stabilityai/stable-diffusion-x4-upscaler"


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


def upscale_image(
    pipeline: StableDiffusionUpscalePipeline,
    *,
    img_path: str,
    mask_path: str | None,
    output_path: str,
    category: str,
) -> None:
    """Crop to the mask bbox and upscale, mirroring the original stage."""
    low_res_img = Image.open(img_path).convert("RGB")
    if mask_path is not None:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        bbox = np.argwhere(mask > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        low_res_img = low_res_img.crop(bbox)  # type: ignore

    prompt = f"Hand manipulates a {category}."

    upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
    upscaled_image.save(output_path)


def main(argv: list[str] | None = None) -> None:
    """Run the command-line entry point."""
    args = build_parser().parse_args(argv)
    pipeline = load_pipeline()
    if args.wait_signal and not stage_prewarm.wait_for_go("upscale"):
        return
    upscale_image(
        pipeline,
        img_path=args.img_path,
        mask_path=args.mask_path,
        output_path=args.output_path,
        category=args.category,
    )


if __name__ == "__main__":
    main()
