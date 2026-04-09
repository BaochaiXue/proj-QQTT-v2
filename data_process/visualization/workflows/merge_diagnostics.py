from __future__ import annotations

from ..types import RenderOutputSpec


def build_render_output_spec_models(
    *,
    geom_render_mode: str,
    render_both_modes: bool,
) -> list[RenderOutputSpec]:
    outputs = [
        RenderOutputSpec(
            name="geom",
            render_mode=str(geom_render_mode),
            video_name="orbit_compare_geom.mp4",
            gif_name="orbit_compare_geom.gif",
            sheet_name="turntable_keyframes_geom.png",
            frames_dir_name="frames_geom",
        )
    ]
    if render_both_modes:
        outputs.append(
            RenderOutputSpec(
                name="rgb",
                render_mode="color_by_rgb",
                video_name="orbit_compare_rgb.mp4",
                gif_name="orbit_compare_rgb.gif",
                sheet_name="turntable_keyframes_rgb.png",
                frames_dir_name="frames_rgb",
            )
        )
    outputs.extend(
        [
            RenderOutputSpec(
                name="support",
                render_mode="support_count",
                video_name="orbit_compare_support.mp4",
                gif_name="orbit_compare_support.gif",
                sheet_name="turntable_keyframes_support.png",
                frames_dir_name="frames_support",
            ),
            RenderOutputSpec(
                name="source",
                render_mode="source_attribution_alpha",
                video_name="orbit_compare_source.mp4",
                gif_name="orbit_compare_source.gif",
                sheet_name="turntable_keyframes_source.png",
                frames_dir_name="frames_source",
            ),
            RenderOutputSpec(
                name="mismatch",
                render_mode="mismatch_residual",
                video_name="orbit_compare_mismatch.mp4",
                gif_name="orbit_compare_mismatch.gif",
                sheet_name="turntable_keyframes_mismatch.png",
                frames_dir_name="frames_mismatch",
            ),
        ]
    )
    return outputs
