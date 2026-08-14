"""Unit tests for the shape-prior backend selector (sam3d/trellis2/none).

Covers: backend id normalization, the Trellis2 client's surgical generate
argv swap (every other stage byte-identical to the v6.2 base class), the
ARAP-safe face filter contract, the orchestrator session's argv mapping for
backend none, and the camera-service v7 flag. CPU-only; no GPU, no camera,
no subprocess spawns (prewarm stays off).
"""

from __future__ import annotations

import numpy as np
import pytest

from demo_v7.service import backend_options
from demo_v7.service.trellis2_generate import _arap_safe_face_mask


class TestBackendOptions:
    def test_normalize_defaults_to_trellis2(self) -> None:
        # Default flipped 2026-08-07 after the same-frame quality comparison
        # (TRELLIS.2: IoU 0.905 vs 0.852, candidates 2.3x closer to obs).
        assert backend_options.normalize_backend(None) == "trellis2"
        assert backend_options.normalize_backend("") == "trellis2"

    def test_normalize_accepts_known_ids_case_insensitive(self) -> None:
        assert backend_options.normalize_backend("TRELLIS2") == "trellis2"
        assert backend_options.normalize_backend(" none ") == "none"
        assert backend_options.normalize_backend("sam3d") == "sam3d"

    def test_normalize_rejects_unknown(self) -> None:
        with pytest.raises(ValueError, match="unknown shape-prior backend"):
            backend_options.normalize_backend("sam3d-v2")


class TestArapSafeFaceMask:
    """The generate-stage output invariant that keeps align's ARAP solvable."""

    def test_keeps_healthy_faces(self) -> None:
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64
        )
        faces = np.array([[0, 1, 2], [1, 3, 2]])
        assert _arap_safe_face_mask(vertices, faces).all()

    def test_drops_face_degenerate_under_exact_weld(self) -> None:
        # Vertex 3 duplicates vertex 1's position exactly (a UV-seam split):
        # face (0, 1, 3) collapses to two identical indices after the o3d
        # remove_duplicated_vertices weld align performs.
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.float64
        )
        faces = np.array([[0, 1, 2], [0, 1, 3]])
        mask = _arap_safe_face_mask(vertices, faces)
        assert mask.tolist() == [True, False]

    def test_drops_zero_area_collinear_face(self) -> None:
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]], dtype=np.float64
        )
        faces = np.array([[0, 1, 3], [0, 1, 2]])
        mask = _arap_safe_face_mask(vertices, faces)
        assert mask.tolist() == [True, False]


class TestZeroExtentFaceMask:
    """final_mesh cleanup keeps every face with any extent at all."""

    def test_keeps_tiny_but_nonzero_faces(self) -> None:
        from demo_v7.service.sample_asap_safe import _zero_extent_face_mask

        vertices = np.array(
            [[0, 0, 0], [1e-7, 0, 0], [0, 1e-7, 0], [1, 0, 0], [0, 1, 0]],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [0, 3, 4]])
        assert _zero_extent_face_mask(vertices, faces).all()

    def test_drops_weld_collapsed_and_exact_zero_area(self) -> None:
        from demo_v7.service.sample_asap_safe import _zero_extent_face_mask

        vertices = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 0, 0],  # exact duplicate of vertex 1 (weld collapse)
                [2, 0, 0],  # collinear with 0-1 (exact zero area)
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]])
        assert _zero_extent_face_mask(vertices, faces).tolist() == [
            True,
            False,
            False,
        ]


class TestTrellis2StageCommands:
    def _make_clients(self, tmp_path):
        from demo_v7.runtime.shape_prior import warmup as shape_prior_warmup
        from demo_v7.service.shape_prior_backends import Trellis2ShapePriorClient

        kwargs = dict(
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
        return (
            shape_prior_warmup.ShapePriorLocalClient(**kwargs),
            Trellis2ShapePriorClient(**kwargs),
        )

    def test_only_generate_sample_align_swapped(self, tmp_path) -> None:
        base, trellis = self._make_clients(tmp_path)
        base_cmds = base._stage_commands()
        trellis_cmds = trellis._stage_commands()
        assert set(base_cmds) == set(trellis_cmds)
        assert trellis_cmds["upscale"] == base_cmds["upscale"]
        assert trellis_cmds["generate"] != base_cmds["generate"]
        # sample/align keep the identical CLI tail; only the entry becomes
        # the v7 wrapper (same interpreter, same GO protocol): sample =
        # zero-extent cleanup, align = binned candidate rasterization.
        from demo_v7.service.shape_prior_backends import (
            ALIGN_FAST_SAFE_RUNNER,
            SAMPLE_ASAP_SAFE_RUNNER,
        )

        assert trellis_cmds["sample"][0] == base_cmds["sample"][0]
        assert trellis_cmds["sample"][1] == str(SAMPLE_ASAP_SAFE_RUNNER)
        assert trellis_cmds["sample"][2:] == base_cmds["sample"][3:]
        assert trellis_cmds["align"][0] == base_cmds["align"][0]
        assert trellis_cmds["align"][1] == str(ALIGN_FAST_SAFE_RUNNER)
        assert trellis_cmds["align"][2:] == base_cmds["align"][3:]

    def test_generate_argv_contract(self, tmp_path) -> None:
        from demo_v7.service.shape_prior_backends import TRELLIS2_RUNNER

        _base, trellis = self._make_clients(tmp_path)
        argv = trellis._stage_commands()["generate"]
        assert argv[0] == str(backend_options.TRELLIS2_PYTHON)
        assert argv[1] == str(TRELLIS2_RUNNER)
        shape_dir = tmp_path / "case_root" / "shape_prior_frame0" / "shape"
        assert argv[argv.index("--img_path") + 1] == str(
            shape_dir / "masked_image.png"
        )
        assert argv[argv.index("--output_dir") + 1] == str(shape_dir)
        assert argv[argv.index("--trellis2-repo") + 1] == str(
            backend_options.TRELLIS2_REPO
        )
        assert argv[argv.index("--seed") + 1] == "42"
        assert argv[argv.index("--profile-json") + 1] == str(
            shape_dir / "timing" / "generate.json"
        )

    def test_create_client_dispatch(self, tmp_path) -> None:
        from demo_v7.runtime.shape_prior import warmup as shape_prior_warmup
        from demo_v7.service.shape_prior_backends import (
            Trellis2ShapePriorClient,
            create_shape_prior_client,
        )

        kwargs = dict(
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
        assert isinstance(
            create_shape_prior_client("sam3d", **kwargs),
            shape_prior_warmup.ShapePriorLocalClient,
        )
        assert isinstance(
            create_shape_prior_client("trellis2", **kwargs),
            Trellis2ShapePriorClient,
        )
        with pytest.raises(ValueError, match="does not use a shape-prior client"):
            create_shape_prior_client("none", **kwargs)


class TestSessionBackendArgv:
    """OrchestratorSession maps backend none onto existing v6.2 switches."""

    def _session(self, tmp_path, **kwargs):
        from demo_v7.orchestration.session import OrchestratorSession

        return OrchestratorSession(
            source="fake-live",
            fake_live_case="data_collect/fake",
            base_path=tmp_path / "run",
            **kwargs,
        )

    def test_none_maps_to_v62_skip_flags(self, tmp_path) -> None:
        session = self._session(tmp_path, shape_prior_backend="none")
        assert session.shape_prior_backend == "none"
        assert session._args.shape_prior_warmup is False
        assert session._args.asap_augment is False
        assert session._args.downstream_mode == "disabled"

    def test_none_respects_explicit_downstream(self, tmp_path) -> None:
        session = self._session(
            tmp_path, shape_prior_backend="none", downstream_mode="disabled"
        )
        assert session._args.downstream_mode == "disabled"

    def test_default_backend_keeps_v62_defaults(self, tmp_path) -> None:
        session = self._session(tmp_path)
        assert session.shape_prior_backend == "trellis2"
        assert session._args.shape_prior_warmup is True
        assert session._args.asap_augment is True

    def test_trellis2_keeps_chain_enabled(self, tmp_path) -> None:
        session = self._session(tmp_path, shape_prior_backend="trellis2")
        assert session.shape_prior_backend == "trellis2"
        assert session._args.shape_prior_warmup is True

    def test_invalid_backend_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="unknown shape-prior backend"):
            self._session(tmp_path, shape_prior_backend="tre11is")


class TestCameraServiceFlag:
    def test_v7_parser_consumes_backend_flag(self) -> None:
        from demo_v7.service.camera_service import _build_v7_parser

        v7_args, rest = _build_v7_parser().parse_known_args(
            [
                "--socket-dir",
                "/tmp/x",
                "--shape-prior-backend",
                "trellis2",
                "--input-source",
                "fake-live",
            ]
        )
        assert v7_args.shape_prior_backend == "trellis2"
        assert "--shape-prior-backend" not in rest
        assert "--input-source" in rest


class TestUpscaleToggle:
    """上采样 on/off: normalization, argv swap surgery, CLI passthrough."""

    def _client_kwargs(self, tmp_path):
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

    def test_normalize_upscale(self) -> None:
        assert backend_options.normalize_upscale(None) is True
        assert backend_options.normalize_upscale(True) is True
        assert backend_options.normalize_upscale(False) is False
        assert backend_options.normalize_upscale("on") is True
        assert backend_options.normalize_upscale("OFF") is False
        assert backend_options.normalize_upscale("true") is True
        assert backend_options.normalize_upscale("0") is False
        with pytest.raises(ValueError, match="upscale toggle"):
            backend_options.normalize_upscale("maybe")

    def test_no_upscale_swaps_only_upscale_stage(self, tmp_path) -> None:
        from demo_v7.runtime.shape_prior import warmup as shape_prior_warmup
        from demo_v7.service.shape_prior_backends import (
            UPSCALE_PASSTHROUGH_RUNNER,
            NoUpscaleShapePriorClient,
        )

        kwargs = self._client_kwargs(tmp_path)
        base_cmds = shape_prior_warmup.ShapePriorLocalClient(
            **kwargs
        )._stage_commands()
        no_up_cmds = NoUpscaleShapePriorClient(**kwargs)._stage_commands()
        assert set(base_cmds) == set(no_up_cmds)
        for stage in ("generate", "align", "sample"):
            assert no_up_cmds[stage] == base_cmds[stage]
        # Same interpreter, same CLI tail; only the entry becomes the
        # crop-only passthrough runner.
        assert no_up_cmds["upscale"][0] == base_cmds["upscale"][0]
        assert no_up_cmds["upscale"][1] == str(UPSCALE_PASSTHROUGH_RUNNER)
        assert no_up_cmds["upscale"][2:] == base_cmds["upscale"][3:]

    def test_no_upscale_composes_with_trellis2(self, tmp_path) -> None:
        from demo_v7.service.shape_prior_backends import (
            UPSCALE_PASSTHROUGH_RUNNER,
            NoUpscaleTrellis2ShapePriorClient,
            Trellis2ShapePriorClient,
        )

        kwargs = self._client_kwargs(tmp_path)
        trellis_cmds = Trellis2ShapePriorClient(**kwargs)._stage_commands()
        no_up_cmds = NoUpscaleTrellis2ShapePriorClient(**kwargs)._stage_commands()
        for stage in ("generate", "align", "sample"):
            assert no_up_cmds[stage] == trellis_cmds[stage]
        assert no_up_cmds["upscale"][1] == str(UPSCALE_PASSTHROUGH_RUNNER)
        assert no_up_cmds["upscale"][2:] == trellis_cmds["upscale"][3:]

    def test_create_client_dispatch_upscale(self, tmp_path) -> None:
        from demo_v7.runtime.shape_prior import warmup as shape_prior_warmup
        from demo_v7.service.shape_prior_backends import (
            NoUpscaleShapePriorClient,
            NoUpscaleTrellis2ShapePriorClient,
            Trellis2ShapePriorClient,
            create_shape_prior_client,
        )

        kwargs = self._client_kwargs(tmp_path)
        on_sam3d = create_shape_prior_client("sam3d", use_upscale=True, **kwargs)
        assert type(on_sam3d) is shape_prior_warmup.ShapePriorLocalClient
        off_sam3d = create_shape_prior_client("sam3d", use_upscale=False, **kwargs)
        assert type(off_sam3d) is NoUpscaleShapePriorClient
        on_tr2 = create_shape_prior_client("trellis2", use_upscale=True, **kwargs)
        assert type(on_tr2) is Trellis2ShapePriorClient
        off_tr2 = create_shape_prior_client("trellis2", use_upscale=False, **kwargs)
        assert type(off_tr2) is NoUpscaleTrellis2ShapePriorClient

    def test_passthrough_cli_crop_and_profile(self, tmp_path) -> None:
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
    def _session(self, tmp_path, **kwargs):
        from demo_v7.orchestration.session import OrchestratorSession

        return OrchestratorSession(
            source="fake-live",
            fake_live_case="data_collect/fake",
            base_path=tmp_path / "run",
            **kwargs,
        )

    def test_default_is_on(self, tmp_path) -> None:
        assert self._session(tmp_path).shape_prior_upscale is True

    def test_explicit_off(self, tmp_path) -> None:
        session = self._session(tmp_path, shape_prior_upscale=False)
        assert session.shape_prior_upscale is False

    def test_string_off(self, tmp_path) -> None:
        session = self._session(tmp_path, shape_prior_upscale="off")
        assert session.shape_prior_upscale is False

    def test_invalid_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="upscale toggle"):
            self._session(tmp_path, shape_prior_upscale="maybe")


class TestCameraServiceUpscaleFlag:
    def test_v7_parser_consumes_upscale_flag(self) -> None:
        from demo_v7.service.camera_service import _build_v7_parser

        v7_args, rest = _build_v7_parser().parse_known_args(
            [
                "--socket-dir", "/tmp/x",
                "--shape-prior-upscale", "off",
                "--input-source", "fake-live",
            ]
        )
        assert v7_args.shape_prior_upscale == "off"
        assert "--shape-prior-upscale" not in rest
