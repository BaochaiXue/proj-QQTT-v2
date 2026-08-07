"""CPU-only contract tests for the FORMAL gaussian display channel."""

from __future__ import annotations

import numpy as np

from demo_v7.service import gaussian_live


def test_gaussian_channel_uses_opaque_white_background(monkeypatch) -> None:
    """CH_GAUSSIAN must be an isolated white-canvas render, not RGB overlay."""
    captured: dict[str, object] = {}
    rendered_rgb = np.array(
        [
            [[12, 34, 56], [255, 255, 255]],
            [[90, 80, 70], [1, 2, 3]],
        ],
        dtype=np.uint8,
    )

    def fake_render_gaussians(
        tensors,
        *,
        viewmat,
        intrinsics,
        width,
        height,
        background,
        device,
    ):
        captured.update(
            background=background,
            width=width,
            height=height,
            device=device,
        )
        return rendered_rgb.copy(), np.ones((height, width), dtype=np.float32)

    monkeypatch.setattr(gaussian_live, "render_gaussians", fake_render_gaussians)

    # Bypass __init__: this test exercises only the display-channel contract,
    # so it must not require CUDA, a PLY file, torch, or gsplat.
    renderer = object.__new__(gaussian_live.GaussianLiveRenderer)
    renderer.failed = False
    renderer.device = "cpu"
    renderer._tensors = {"sentinel": object()}

    # Use conspicuous camera pixels: any accidental compositing would make the
    # result differ from the renderer's RGB output below.
    frame_bgr = np.full((2, 2, 3), 173, dtype=np.uint8)
    output = renderer.render_over(
        frame_bgr,
        viewmat=np.eye(4, dtype=np.float32),
        intrinsics=np.eye(3, dtype=np.float32),
    )

    assert captured == {
        "background": (1.0, 1.0, 1.0),
        "width": 2,
        "height": 2,
        "device": "cpu",
    }
    np.testing.assert_array_equal(output, rendered_rgb[..., ::-1])
    assert output.flags.c_contiguous
