from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert saved FFS disparity into compatible depth outputs.")
    parser.add_argument("--sample_dir", required=True)
    parser.add_argument("--ffs_out_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def save_quicklook(depth_m, output_path: Path) -> None:
    import cv2
    import numpy as np

    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        preview = np.zeros(depth.shape + (3,), dtype=np.uint8)
    else:
        normalized = np.zeros_like(depth, dtype=np.float32)
        min_depth = float(depth[valid].min())
        max_depth = float(depth[valid].max())
        if max_depth > min_depth:
            normalized[valid] = (depth[valid] - min_depth) / (max_depth - min_depth)
        preview = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        preview[~valid] = 0
    cv2.imwrite(str(output_path), preview)


def main() -> int:
    args = parse_args()

    import numpy as np

    from scripts.harness.ffs_geometry import (
        disparity_to_metric_depth,
        project_to_color,
        quantize_depth_with_invalid_zero,
        rasterize_nearest_depth,
        transform_points,
        unproject_ir_depth,
    )

    sample_dir = Path(args.sample_dir).resolve()
    ffs_out_dir = Path(args.ffs_out_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    disparity = np.load(ffs_out_dir / "disparity_raw.npy")
    run_metadata_path = ffs_out_dir / "run_metadata.json"
    run_metadata = {}
    if run_metadata_path.exists():
        run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))

    K_ir_left = np.asarray(run_metadata.get("K_ir_left_used", metadata["K_ir_left"]), dtype=np.float32)
    baseline_m = float(run_metadata.get("baseline_m", metadata["ir_baseline_m"]))
    fx_ir = float(K_ir_left[0, 0])

    depth_ir = disparity_to_metric_depth(disparity, fx_ir=fx_ir, baseline_m=baseline_m)
    np.save(out_dir / "depth_ir_left_float_m.npy", depth_ir)

    conversion_metadata: dict[str, object] = {
        "source_serial": metadata["serial"],
        "baseline_m": baseline_m,
        "fx_ir": fx_ir,
        "K_ir_left": K_ir_left.tolist(),
        "encoding_mode": "float_only",
        "depth_scale_m_per_unit": metadata.get("depth_scale_m_per_unit"),
        "invalid_value": 0,
        "notes": [
            "Stereo source is one D455 internal pair only.",
            "FFS output first lives in IR-left coordinates.",
            "Color-aligned output is produced only by explicit IR-left to color reprojection.",
            "Runtime intrinsics and runtime extrinsics are authoritative.",
        ],
    }

    quicklook_source = depth_ir
    if "K_color" in metadata and "T_ir_left_to_color" in metadata:
        points_ir, _ = unproject_ir_depth(depth_ir, K_ir_left)
        T_ir_left_to_color = np.asarray(metadata["T_ir_left_to_color"], dtype=np.float32)
        points_color = transform_points(points_ir, T_ir_left_to_color)
        uv_color, z_color = project_to_color(points_color, np.asarray(metadata["K_color"], dtype=np.float32))
        output_shape = (int(metadata["height"]), int(metadata["width"]))
        depth_color = rasterize_nearest_depth(uv_color, z_color, output_shape=output_shape)
        np.save(out_dir / "depth_color_aligned_float_m.npy", depth_color)
        quicklook_source = depth_color
        conversion_metadata["K_color"] = metadata["K_color"]
        conversion_metadata["T_ir_left_to_color"] = metadata["T_ir_left_to_color"]
        conversion_metadata["encoding_mode"] = "float_color_aligned"

        depth_scale = metadata.get("depth_scale_m_per_unit")
        if depth_scale is not None:
            depth_u16 = quantize_depth_with_invalid_zero(depth_color, float(depth_scale))
            np.save(out_dir / "depth_color_aligned_u16.npy", depth_u16)
            conversion_metadata["encoding_mode"] = "float_and_u16_color_aligned"

    save_quicklook(quicklook_source, out_dir / "quicklook_depth.png")
    (out_dir / "conversion_metadata.json").write_text(
        json.dumps(conversion_metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Saved converted FFS depth artifacts to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
