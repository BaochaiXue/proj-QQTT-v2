"""Unit tests for the shape-prior backend selector (sam3d/trellis2/none).

Covers: the v7 clients' surgical argv swaps (every other stage byte-identical
to the v6.2 base class), the two deliberately DIFFERENT degenerate-face
filters, the crop-only upscale passthrough, and the orchestrator session's
argv mapping for backend none plus its fail-fast option normalization.
CPU-only; no GPU, no camera, no subprocess spawns (prewarm stays off).
"""

from __future__ import annotations

import numpy as np
import pytest

from demo_v7.service import backend_options
from demo_v7.service.trellis2_generate import _arap_safe_face_mask


def _client_kwargs(tmp_path) -> dict:
    """The unchanged ShapePriorLocalClient ctor kwargs every backend shares."""
    return dict(
        case_root=tmp_path / "case_root",
        cuda_visible_devices="0",
        object_prompt="sloth",
        controller_name="hand",
        object_id=None,
        cache_root=tmp_path / "mesh_cache",
        sam3d_root=None,
        sam3d_config=None,
        sam31_device="cuda",
    )


def _session(tmp_path, **kwargs):
    from demo_v7.orchestration.session import OrchestratorSession

    return OrchestratorSession(
        source="fake-live",
        fake_live_case="data_collect/fake",
        base_path=tmp_path / "run",
        **kwargs,
    )


class TestArapSafeFaceMask:
    """The generate-stage output invariant that keeps align's ARAP solvable."""

    def test_drops_zero_area_collinear_face(self) -> None:
        """Pins the area term of trellis2_generate.py:294
        (``return distinct & (areas > 1e-12)``).

        Defect: o_voxel's atlas export ships zero-area UV-seam slivers; their
        cotangent weights come out nan/inf and align's ARAP dies with
        'Failed to build solver' (commit 918c80b, the phase-2 runner work).
        Deleting the area term flips the second entry to True.
        """
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]], dtype=np.float64
        )
        faces = np.array([[0, 1, 3], [0, 1, 2]])
        mask = _arap_safe_face_mask(vertices, faces)
        assert mask.tolist() == [True, False]


class TestZeroExtentFaceMask:
    """final_mesh cleanup keeps every face with any extent at all."""

    @pytest.mark.parametrize(
        "vertices, faces, expected",
        [
            pytest.param(
                [[0, 0, 0], [1e-7, 0, 0], [0, 1e-7, 0], [1, 0, 0], [0, 1, 0]],
                [[0, 1, 2], [0, 3, 4]],
                [True, True],
                id="tiny-but-real-face-survives",
            ),
            pytest.param(
                # vertex 3 is collinear with 0-1: exactly zero area.
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0]],
                [[0, 1, 2], [0, 1, 3]],
                [True, False],
                id="exact-zero-area-dropped",
            ),
        ],
    )
    def test_keeps_tiny_but_nonzero_faces(self, vertices, faces, expected) -> None:
        """Pins sample_asap_safe.py:68 ``return distinct & (areas > 0.0)`` --
        the threshold that deliberately DIFFERS from its otherwise identical
        sibling at trellis2_generate.py:294 (``> 1e-12``).

        Defect: the two functions are copies apart from that constant, so the
        obvious 'dedupe into one helper' refactor silently unifies them and the
        final_mesh cleanup starts eating tiny-but-real faces (first row).
        Deleting the area term instead keeps the collinear face (second row).
        """
        from demo_v7.service.sample_asap_safe import _zero_extent_face_mask

        mask = _zero_extent_face_mask(
            np.array(vertices, dtype=np.float64), np.array(faces)
        )
        assert mask.tolist() == expected


class TestTrellis2StageCommands:
    """Every v7 backend is argv surgery on the v6.2 stage command table."""

    @pytest.mark.parametrize(
        "client, baseline, swapped, generate_replaced",
        [
            pytest.param(
                "Trellis2ShapePriorClient",
                None,  # baseline = the untouched v6.2 client
                {
                    "sample": "SAMPLE_ASAP_SAFE_RUNNER",
                    "align": "ALIGN_FAST_SAFE_RUNNER",
                },
                True,
                id="trellis2-vs-v62",
            ),
            pytest.param(
                "NoUpscaleShapePriorClient",
                None,
                {"upscale": "UPSCALE_PASSTHROUGH_RUNNER"},
                False,
                id="no-upscale-vs-v62",
            ),
            pytest.param(
                # The composed class must still reach the mixin: flipping the
                # base order at shape_prior_backends.py:131 to
                # (Trellis2ShapePriorClient, _NoUpscaleStageMixin) leaves the
                # MRO on the SD upscale stage and only this row notices.
                "NoUpscaleTrellis2ShapePriorClient",
                "Trellis2ShapePriorClient",
                {"upscale": "UPSCALE_PASSTHROUGH_RUNNER"},
                False,
                id="no-upscale-composed-with-trellis2",
            ),
        ],
    )
    def test_only_generate_sample_align_swapped(
        self, tmp_path, client, baseline, swapped, generate_replaced
    ) -> None:
        """Pins the wrapper surgery at shape_prior_backends.py:108
        (``*sample[3:]``), :119 (``*align[3:]``) and :59-61 (the upscale
        mixin's ``upscale[0]`` interpreter + ``*upscale[3:]`` tail).

        Defect: the swap replaces two argv tokens (``-m``, module) with one
        (the runner path), so the tail slice must be [3:] while the new argv
        is read at [2:]. An off-by-one leaves ``demo_v7.runtime.shape_prior.
        sample`` as a positional arg (or eats --base_path) and the prewarmed
        stage dies inside the pool with an opaque error. Every stage the
        backend does not claim must stay byte-identical to its baseline.
        """
        from demo_v7.runtime.shape_prior import warmup as shape_prior_warmup
        from demo_v7.service import shape_prior_backends

        kwargs = _client_kwargs(tmp_path)
        baseline_cls = (
            shape_prior_warmup.ShapePriorLocalClient
            if baseline is None
            else getattr(shape_prior_backends, baseline)
        )
        base_cmds = baseline_cls(**kwargs)._stage_commands()
        cmds = getattr(shape_prior_backends, client)(**kwargs)._stage_commands()

        assert set(cmds) == set(base_cmds)
        for stage, runner in swapped.items():
            # Same interpreter, same CLI tail; only the entry becomes the v7
            # wrapper (same GO protocol).
            assert cmds[stage][0] == base_cmds[stage][0]
            assert cmds[stage][1] == str(getattr(shape_prior_backends, runner))
            assert cmds[stage][2:] == base_cmds[stage][3:]
        untouched = set(base_cmds) - set(swapped)
        if generate_replaced:
            untouched.discard("generate")
        for stage in untouched:
            assert cmds[stage] == base_cmds[stage]

        if not generate_replaced:
            return
        # TRELLIS.2 replaces generate wholesale (own interpreter + runner).
        assert cmds["generate"] != base_cmds["generate"]
        argv = cmds["generate"]
        shape_dir = tmp_path / "case_root" / "shape_prior_frame0" / "shape"
        # RGBA-alpha input contract (trellis2_generate.py:302-307): the runner
        # needs masked_image.png, not the upscaled high_resolution.png.
        assert argv[argv.index("--img_path") + 1] == str(
            shape_dir / "masked_image.png"
        )
        # Schema-v1 profile lands exactly where the orchestrator reads it.
        assert argv[argv.index("--profile-json") + 1] == str(
            shape_dir / "timing" / "generate.json"
        )


class TestSessionBackendArgv:
    """OrchestratorSession maps backend none onto existing v6.2 switches."""

    @pytest.mark.parametrize(
        "backend, expected",
        [
            pytest.param(
                "none",
                {
                    "shape_prior_warmup": False,
                    "asap_augment": False,
                    "downstream_mode": "disabled",
                },
                id="none-skips-warmup-and-asap",
            ),
            pytest.param(
                None,  # default backend: the negative control
                {"shape_prior_warmup": True, "asap_augment": True},
                id="default-backend-keeps-v62-defaults",
            ),
        ],
    )
    def test_none_maps_to_v62_skip_flags(self, tmp_path, backend, expected) -> None:
        """Pins session.py:325
        ``argv.extend(['--no-shape-prior-warmup', '--no-asap-augment'])``.

        Defect: backend none has no mesh and ASAP hard-requires one -- if only
        the warmup flag is emitted the run reaches ASAP with nothing to
        deform. The default row is the negative control: the skip flags must
        NOT be emitted for a mesh-producing backend.
        """
        kwargs = {} if backend is None else {"shape_prior_backend": backend}
        session = _session(tmp_path, **kwargs)
        if backend is not None:
            assert session.shape_prior_backend == backend
        for name, value in expected.items():
            assert getattr(session._args, name) == value, name

    @pytest.mark.parametrize(
        "kwarg, value, message",
        [
            ("shape_prior_backend", "tre11is", "unknown shape-prior backend"),
            ("shape_prior_upscale", "maybe", "upscale toggle"),
        ],
    )
    def test_invalid_backend_raises(self, tmp_path, kwarg, value, message) -> None:
        """normalize_backend / normalize_upscale run inside the session ctor
        (session.py:247-251 and :258-261), before the strict v6.2 parse.

        Defect: this is the documented GUI-side fail-fast -- without it a
        typo'd selector from the CLI/config surfaces only as a camera-service
        stderr line and the operator sees a generic connect timeout.
        """
        with pytest.raises(ValueError, match=message):
            _session(tmp_path, **{kwarg: value})


class TestUpscaleToggle:
    """上采样 on/off: normalization, factory dispatch, CLI passthrough."""

    def test_normalize_upscale(self) -> None:
        """Pins the explicit _UPSCALE_TRUE/_UPSCALE_FALSE tables and the raise
        at backend_options.py:91-94.

        Defect: the value arrives as a STRING from both the CLI
        (--shape-prior-upscale off) and default.yaml, so the natural-looking
        ``bool(value)`` makes 'off' and '0' truthy -- the operator disables
        上采样 and gets the 15.2s SD stage anyway.
        """
        assert backend_options.normalize_upscale(None) is True
        assert backend_options.normalize_upscale(True) is True
        assert backend_options.normalize_upscale(False) is False
        assert backend_options.normalize_upscale("on") is True
        assert backend_options.normalize_upscale("OFF") is False
        assert backend_options.normalize_upscale("true") is True
        assert backend_options.normalize_upscale("0") is False
        with pytest.raises(ValueError, match="upscale toggle"):
            backend_options.normalize_upscale("maybe")

    def test_create_client_dispatch_upscale(self, tmp_path) -> None:
        """Pins the 2x2 backend x use_upscale branch table at
        shape_prior_backends.py:148-155 and the :156 raise for backend none.

        Defect: a copy-paste that ignores use_upscale on one branch silently
        runs the 15.2s SD upscale after the operator turned the GUI 上采样
        toggle off (or vice versa) -- no error, just a wrong-resolution
        warmup. ``type(...) is``, never isinstance: the NoUpscale* classes
        subclass the base client, so isinstance passes even when the toggle
        is ignored entirely.
        """
        from demo_v7.runtime.shape_prior import warmup as shape_prior_warmup
        from demo_v7.service.shape_prior_backends import (
            NoUpscaleShapePriorClient,
            NoUpscaleTrellis2ShapePriorClient,
            Trellis2ShapePriorClient,
            create_shape_prior_client,
        )

        kwargs = _client_kwargs(tmp_path)
        on_sam3d = create_shape_prior_client("sam3d", use_upscale=True, **kwargs)
        assert type(on_sam3d) is shape_prior_warmup.ShapePriorLocalClient
        off_sam3d = create_shape_prior_client("sam3d", use_upscale=False, **kwargs)
        assert type(off_sam3d) is NoUpscaleShapePriorClient
        on_tr2 = create_shape_prior_client("trellis2", use_upscale=True, **kwargs)
        assert type(on_tr2) is Trellis2ShapePriorClient
        off_tr2 = create_shape_prior_client("trellis2", use_upscale=False, **kwargs)
        assert type(off_tr2) is NoUpscaleTrellis2ShapePriorClient
        # backend none never reaches a client at all.
        with pytest.raises(ValueError, match="does not use a shape-prior client"):
            create_shape_prior_client("none", **kwargs)

    def test_passthrough_cli_crop_and_profile(self, tmp_path) -> None:
        """Pins upscale_passthrough.py:70-85: crop_like_upscale must reproduce
        the SD upscale stage's bbox math (upscale.py:72-88) bit for bit.

        Defect: this file is a hand-copied mirror; any drift (x/y swapped in
        the np.argwhere bbox at :71-74, the x1.2 margin dropped at :78)
        silently feeds SAM3.1 + generate a differently-framed crop, with no
        error anywhere. The zeroed model_load/inference timings are what tell
        the profile reader this stage did no SD work.
        """
        import json

        import cv2
        from PIL import Image

        from demo_v7.service import upscale_passthrough

        rng = np.random.default_rng(7)
        image = rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
        mask = np.zeros((48, 64), dtype=np.uint8)
        mask[10:30, 20:50] = 255  # bbox: x 20..49, y 10..29
        img_path = tmp_path / "color.png"
        mask_path = tmp_path / "mask.png"
        out_path = tmp_path / "high_resolution.png"
        profile_path = tmp_path / "upscale.json"
        Image.fromarray(image).save(img_path)
        cv2.imwrite(str(mask_path), mask)

        upscale_passthrough.main(
            [
                "--img_path", str(img_path),
                "--mask_path", str(mask_path),
                "--output_path", str(out_path),
                "--category", "sloth",
                "--profile-json", str(profile_path),
            ]
        )

        # Expected crop: upscale.py's exact bbox math (square, x1.2 margin).
        x0, y0, x1, y1 = 20, 10, 49, 29
        center = ((x0 + x1) / 2, (y0 + y1) / 2)
        size = int(max(x1 - x0, y1 - y0) * 1.2)
        box = (
            center[0] - size // 2,
            center[1] - size // 2,
            center[0] + size // 2,
            center[1] + size // 2,
        )
        expected = np.asarray(Image.fromarray(image).crop(box))
        produced = np.asarray(Image.open(out_path).convert("RGB"))
        assert produced.shape == expected.shape
        assert np.array_equal(produced, expected)

        profile = json.loads(profile_path.read_text())
        assert profile["stage"] == "upscale"
        assert profile["status"] == "completed"
        timing = profile["timing_ms"]
        for field in (
            "module_import_ms", "model_load_ms", "input_crop_ms",
            "inference_ms", "output_write_ms", "total_ms",
        ):
            assert field in timing
        assert timing["model_load_ms"] == 0.0
        assert timing["inference_ms"] == 0.0


class TestSessionUpscaleResolution:
    @pytest.mark.parametrize(
        "value, expected",
        [
            pytest.param(False, False, id="explicit-False"),
            pytest.param("off", False, id="string-off"),
        ],
    )
    def test_explicit_off(self, tmp_path, value, expected) -> None:
        """Pins session.py:258-261, specifically the ``if shape_prior_upscale
        is not None`` on :260 and the normalize_upscale call on :258.

        Defect: the falsy-argument trap -- rewriting :258-261 as
        ``shape_prior_upscale or session_cfg.get('shape_prior_upscale')``
        makes an explicit False fall through to default.yaml's ``true``, so
        the GUI toggle is ignored exactly when it is switched off (first row);
        using ``bool(...)`` instead of normalize_upscale turns the CLI/config
        string 'off' into True (second row).
        """
        session = _session(tmp_path, shape_prior_upscale=value)
        assert session.shape_prior_upscale is expected
