"""Stage: extract the TripoSplat input (RGBA) from a demo_v6_2 frame-0 case.

    python -m gaussian_align_demo.prepare_input \
        --case-dir outputs/shape_prior_case/shape_prior_frame0 --run-dir <run>

Writes <run>/input/: frame0_rgb.png, frame0_rgba.png (alpha = object mask),
object_mask.png, controller_mask.png, input_manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from gaussian_align_demo.case_loader import load_frame0_case


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)

    case = load_frame0_case(args.case_dir)
    input_dir = Path(args.run_dir) / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    rgba = case.object_rgba()
    Image.fromarray(case.rgb_u8).save(input_dir / "frame0_rgb.png")
    Image.fromarray(rgba).save(input_dir / "frame0_rgba.png")
    Image.fromarray(np.where(case.object_mask, 255, 0).astype(np.uint8)).save(
        input_dir / "object_mask.png"
    )
    Image.fromarray(np.where(case.controller_mask, 255, 0).astype(np.uint8)).save(
        input_dir / "controller_mask.png"
    )

    manifest = {
        "case_dir": str(Path(args.case_dir).resolve()),
        "object_name": case.object_name,
        "controller_name": case.controller_name,
        "width": case.width,
        "height": case.height,
        "object_mask_px": int(case.object_mask.sum()),
        "object_depth_valid_px": int((case.object_mask & case.depth_valid).sum()),
        "K": case.K.tolist(),
        "c2w": case.c2w.tolist(),
        "frame0_rgba_sha256": hashlib.sha256(
            (input_dir / "frame0_rgba.png").read_bytes()
        ).hexdigest(),
    }
    (input_dir / "input_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[input] {input_dir}: object '{case.object_name}' "
          f"{manifest['object_mask_px']} px ({manifest['object_depth_valid_px']} depth-valid)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
