"""Unit acceptance for the shape-prior canonical-mesh cache (v1).

Exercises the mesh_cache state machine, atomic publish/materialize, manifest
validation, and the ShapePriorLocalClient cache resolution + conditional
prewarm selection -- without running SAM3D/GPU. Canonical meshes are stand-in
trimesh boxes exported as GLB (a real, loadable mesh with vertices + faces).
"""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import trimesh

from demo_v6_2.shape_prior import mesh_cache
from demo_v6_2.shape_prior.mesh_cache import (
    CACHE_STATUS_DISABLED,
    CACHE_STATUS_HIT,
    CACHE_STATUS_MISS,
    ShapePriorMeshCache,
    ShapePriorMeshCacheError,
    normalize_object_id,
    sha256_file,
    validate_cache_root,
    validate_mesh_glb,
)


def _write_box_glb(path: Path) -> Path:
    """Write a small valid mesh GLB (a unit box) to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(str(path))
    return path


def _frame0_request():
    """Return a minimal valid frame-0 request for request-path tests."""
    from demo_v6_2.shape_prior.case import (  # noqa: PLC0415
        ShapePriorFrame0Request,
    )

    points_world_m = np.zeros((2, 2, 3), dtype=np.float32)
    points_world_m[..., 2] = 1.0
    return ShapePriorFrame0Request(
        seq=0,
        source_timestamp_s=0.0,
        input_source="fake-live",
        depth_backend="realsense",
        depth_source_internal="realsense",
        rgb_u8=np.zeros((2, 2, 3), dtype=np.uint8),
        object_mask=np.asarray([[True, False], [False, False]], dtype=bool),
        controller_mask=np.asarray([[False, True], [False, False]], dtype=bool),
        depth_color_m=np.ones((2, 2), dtype=np.float32),
        depth_valid_mask=np.ones((2, 2), dtype=bool),
        points_world_m=points_world_m,
        k_color=np.eye(3, dtype=np.float32),
        camera_to_world_c2w=np.eye(4, dtype=np.float32),
    )


class NormalizeObjectIdTests(unittest.TestCase):
    def test_null_disables_cache(self) -> None:
        self.assertIsNone(normalize_object_id(None))

    def test_valid_identity_passes(self) -> None:
        for good in ("sloth_plush_01_v1", "blue_sloth_02_v1", "obj-A.v3"):
            self.assertEqual(normalize_object_id(good), good)

    def test_rejects_sentinel_and_empty_strings(self) -> None:
        # The spec forbids treating "none"/"null"/"" as a real identity: only
        # YAML null disables the cache.
        for bad in ("", "   ", "none", "None", "NULL", "null"):
            with self.assertRaises(ShapePriorMeshCacheError):
                normalize_object_id(bad)

    def test_rejects_path_like_identities(self) -> None:
        for bad in (
            "a/b",
            "..",
            "../x",
            "/abs/path",
            "x\\y",
            "has space",
            "a\tb",
            ".hidden",
            "sloth:v1",
            "sloth*v1",
        ):
            with self.assertRaises(ShapePriorMeshCacheError):
                normalize_object_id(bad)

    def test_rejects_non_string_identity(self) -> None:
        for bad in (False, 0, 123, ["sloth_v1"]):
            with self.subTest(bad=bad):
                with self.assertRaises(ShapePriorMeshCacheError):
                    normalize_object_id(bad)


class CacheRootTests(unittest.TestCase):
    def test_rejects_root_under_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "outputs"
            inside = base / "cache"
            with self.assertRaises(ShapePriorMeshCacheError):
                validate_cache_root(inside, forbidden_root=base)
            with self.assertRaises(ShapePriorMeshCacheError):
                validate_cache_root(base, forbidden_root=base)

    def test_accepts_root_outside_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "outputs"
            outside = Path(tmp) / "persistent_cache"
            resolved = validate_cache_root(outside, forbidden_root=base)
            self.assertEqual(resolved, outside.resolve())

    def test_output_cleanup_preserves_external_cache(self) -> None:
        from demo_v6_2.orchestration.main_layout import (  # noqa: PLC0415
            prepare_realtime_output_for_new_run,
        )

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "outputs"
            cache_root = Path(tmp) / "persistent_cache"
            source = _write_box_glb(Path(tmp) / "source" / "object.glb")
            cache = ShapePriorMeshCache(
                object_id="sloth_v1",
                cache_root=cache_root,
            )
            cache.publish(
                source_glb=source,
                object_prompt_at_generation="sloth",
                generator_seed=42,
            )
            (base / "capture").mkdir(parents=True)
            (base / "capture" / "stale.txt").write_text("stale")

            prepare_realtime_output_for_new_run(
                base,
                legacy_case_prefix="demo_v6_2",
            )

            self.assertFalse((base / "capture").exists())
            self.assertEqual(cache.resolve().status, CACHE_STATUS_HIT)


class MeshValidationTests(unittest.TestCase):
    def test_valid_box_glb_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            glb = _write_box_glb(Path(tmp) / "object.glb")
            validate_mesh_glb(glb)  # must not raise

    def test_missing_and_empty_glb_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ShapePriorMeshCacheError):
                validate_mesh_glb(Path(tmp) / "absent.glb")
            empty = Path(tmp) / "empty.glb"
            empty.write_bytes(b"")
            with self.assertRaises(ShapePriorMeshCacheError):
                validate_mesh_glb(empty)

    def test_garbage_glb_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            junk = Path(tmp) / "junk.glb"
            junk.write_bytes(b"not a real glb file")
            with self.assertRaises(ShapePriorMeshCacheError):
                validate_mesh_glb(junk)


class CacheStateMachineTests(unittest.TestCase):
    def _cache(self, root: Path, object_id: str | None) -> ShapePriorMeshCache:
        return ShapePriorMeshCache(object_id=object_id, cache_root=root)

    def test_disabled_when_object_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            resolution = self._cache(Path(tmp) / "cache", None).resolve()
            self.assertEqual(resolution.status, CACHE_STATUS_DISABLED)
            self.assertFalse(resolution.enabled)
            self.assertFalse(resolution.hit)

    def test_miss_when_no_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cache"
            resolution = self._cache(root, "sloth_v1").resolve()
            self.assertEqual(resolution.status, CACHE_STATUS_MISS)
            self.assertTrue(resolution.enabled)
            self.assertFalse(resolution.hit)
            self.assertTrue((root / mesh_cache.SCHEMA_DIR_NAME).is_dir())

    def test_publish_then_hit_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cache"
            src = _write_box_glb(Path(tmp) / "gen" / "object.glb")
            cache = self._cache(root, "sloth_v1")
            manifest = cache.publish(
                source_glb=src,
                object_prompt_at_generation="sloth",
                generator_seed=42,
            )
            self.assertEqual(manifest["schema_version"], mesh_cache.SCHEMA_VERSION)
            self.assertEqual(manifest["object_id"], "sloth_v1")
            self.assertEqual(manifest["object_prompt_at_generation"], "sloth")
            self.assertEqual(manifest["asset_status"], "generated")
            self.assertEqual(manifest["mesh_file"], "object.glb")
            self.assertEqual(manifest["generator"], {"type": "sam3d", "seed": 42})
            self.assertEqual(manifest["mesh_sha256"], sha256_file(src))

            entry = root / mesh_cache.SCHEMA_DIR_NAME / "sloth_v1"
            self.assertTrue((entry / "object.glb").is_file())
            self.assertTrue((entry / "manifest.json").is_file())
            # No temp publish dirs left behind.
            self.assertEqual(
                [
                    p
                    for p in (root / mesh_cache.SCHEMA_DIR_NAME).iterdir()
                    if p.name.startswith(".tmp")
                ],
                [],
            )

            resolution = self._cache(root, "sloth_v1").resolve()
            self.assertEqual(resolution.status, CACHE_STATUS_HIT)
            self.assertTrue(resolution.hit)
            self.assertEqual(resolution.manifest["mesh_sha256"], sha256_file(src))

    def test_prompt_change_still_hits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cache"
            src = _write_box_glb(Path(tmp) / "gen" / "object.glb")
            self._cache(root, "sloth_v1").publish(
                source_glb=src, object_prompt_at_generation="sloth", generator_seed=42
            )
            # A later run with a different prompt but the same object id: the
            # cache key is the object id only, so it still hits and preserves the
            # generation prompt in the manifest.
            resolution = self._cache(root, "sloth_v1").resolve()
            self.assertEqual(resolution.status, CACHE_STATUS_HIT)
            self.assertEqual(
                resolution.manifest["object_prompt_at_generation"], "sloth"
            )

    def test_conflict_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cache"
            first = _write_box_glb(Path(tmp) / "gen1" / "object.glb")
            cache = self._cache(root, "sloth_v1")
            cache.publish(
                source_glb=first, object_prompt_at_generation="sloth", generator_seed=42
            )
            entry_sha = sha256_file(
                root / mesh_cache.SCHEMA_DIR_NAME / "sloth_v1" / "object.glb"
            )
            # A different mesh, same object id -> conflict, first entry intact.
            second = Path(tmp) / "gen2" / "object.glb"
            second.parent.mkdir(parents=True)
            trimesh.creation.box(extents=(2.0, 2.0, 2.0)).export(str(second))
            with self.assertRaises(ShapePriorMeshCacheError):
                self._cache(root, "sloth_v1").publish(
                    source_glb=second,
                    object_prompt_at_generation="sloth",
                    generator_seed=42,
                )
            self.assertEqual(
                sha256_file(
                    root / mesh_cache.SCHEMA_DIR_NAME / "sloth_v1" / "object.glb"
                ),
                entry_sha,
            )

    def test_corrupt_entries_raise_on_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cache"
            src = _write_box_glb(Path(tmp) / "gen" / "object.glb")
            self._cache(root, "sloth_v1").publish(
                source_glb=src, object_prompt_at_generation="sloth", generator_seed=42
            )
            entry = root / mesh_cache.SCHEMA_DIR_NAME / "sloth_v1"

            # (a) mesh bytes tampered -> hash mismatch.
            good_bytes = (entry / "object.glb").read_bytes()
            (entry / "object.glb").write_bytes(good_bytes + b"tamper")
            with self.assertRaises(ShapePriorMeshCacheError):
                self._cache(root, "sloth_v1").resolve()
            (entry / "object.glb").write_bytes(good_bytes)
            self.assertEqual(
                self._cache(root, "sloth_v1").resolve().status,
                CACHE_STATUS_HIT,
            )

            # A present entry with either required file missing is corrupt,
            # never a cache miss.
            manifest_path = entry / mesh_cache.MANIFEST_FILENAME
            manifest_bytes = manifest_path.read_bytes()
            manifest_path.unlink()
            with self.assertRaises(ShapePriorMeshCacheError):
                self._cache(root, "sloth_v1").resolve()
            manifest_path.write_bytes(manifest_bytes)

            (entry / "object.glb").unlink()
            with self.assertRaises(ShapePriorMeshCacheError):
                self._cache(root, "sloth_v1").resolve()

    def test_manifest_contract_corruption_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cache"
            src = _write_box_glb(Path(tmp) / "gen" / "object.glb")
            self._cache(root, "sloth_v1").publish(
                source_glb=src,
                object_prompt_at_generation="sloth",
                generator_seed=42,
            )
            manifest_path = (
                root
                / mesh_cache.SCHEMA_DIR_NAME
                / "sloth_v1"
                / mesh_cache.MANIFEST_FILENAME
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            corruptions = {
                "schema_type": dict(manifest, schema_version="1"),
                "prompt": dict(manifest, object_prompt_at_generation=""),
                "asset_status": dict(manifest, asset_status="verified"),
                "mesh_file": dict(manifest, mesh_file="other.glb"),
                "sha_format": dict(manifest, mesh_sha256="not-a-sha"),
                "created_at": dict(manifest, created_at_utc="yesterday"),
                "generator_type": {
                    **manifest,
                    "generator": {"type": "other", "seed": 42},
                },
                "generator_seed": {
                    **manifest,
                    "generator": {"type": "sam3d", "seed": True},
                },
            }
            for name, corrupted in corruptions.items():
                with self.subTest(name=name):
                    manifest_path.write_text(
                        json.dumps(corrupted),
                        encoding="utf-8",
                    )
                    with self.assertRaises(ShapePriorMeshCacheError):
                        self._cache(root, "sloth_v1").resolve()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(
                self._cache(root, "sloth_v1").resolve().status,
                CACHE_STATUS_HIT,
            )

    def test_materialize_is_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cache"
            src = _write_box_glb(Path(tmp) / "gen" / "object.glb")
            src_sha = sha256_file(src)
            cache = self._cache(root, "sloth_v1")
            cache.publish(
                source_glb=src, object_prompt_at_generation="sloth", generator_seed=42
            )
            resolution = self._cache(root, "sloth_v1").resolve()
            dest = Path(tmp) / "run_case" / "shape" / "object.glb"
            returned_sha = cache.materialize(resolution=resolution, dest_glb=dest)
            self.assertTrue(dest.is_file())
            self.assertEqual(sha256_file(dest), src_sha)
            self.assertEqual(returned_sha, src_sha)


class ClientCacheResolutionTests(unittest.TestCase):
    """ShapePriorLocalClient resolves cache + selects prewarm stages (no GPU)."""

    def _client(self, *, case_root, cache_root, object_id):
        from demo_v6_2.shape_prior.warmup import (  # noqa: PLC0415
            PREWARM_STAGE_ALIGN,
            PREWARM_STAGE_GENERATE,
            PREWARM_STAGE_SAMPLE,
            PREWARM_STAGE_UPSCALE,
            ShapePriorLocalClient,
        )

        client = ShapePriorLocalClient(
            case_root=case_root,
            object_prompt="sloth",
            controller_name="hand",
            object_id=object_id,
            cache_root=cache_root,
        )
        return client, {
            "upscale": PREWARM_STAGE_UPSCALE,
            "generate": PREWARM_STAGE_GENERATE,
            "align": PREWARM_STAGE_ALIGN,
            "sample": PREWARM_STAGE_SAMPLE,
        }

    def test_disabled_prewarms_full_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client, stages = self._client(
                case_root=Path(tmp) / "case",
                cache_root=Path(tmp) / "cache",
                object_id=None,
            )
            self.assertEqual(client.cache_resolution.status, CACHE_STATUS_DISABLED)
            self.assertTrue(client.reuse_sam31_model)
            self.assertEqual(
                set(client._prewarm_stages()),
                {
                    stages["upscale"],
                    stages["generate"],
                    stages["align"],
                    stages["sample"],
                },
            )

    def test_miss_prewarms_full_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client, stages = self._client(
                case_root=Path(tmp) / "case",
                cache_root=Path(tmp) / "cache",
                object_id="sloth_v1",
            )
            self.assertEqual(client.cache_resolution.status, CACHE_STATUS_MISS)
            self.assertTrue(client.reuse_sam31_model)
            self.assertEqual(
                set(client._prewarm_stages()),
                {
                    stages["upscale"],
                    stages["generate"],
                    stages["align"],
                    stages["sample"],
                },
            )

    def test_hit_prewarms_only_align_and_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cache"
            src = _write_box_glb(Path(tmp) / "gen" / "object.glb")
            ShapePriorMeshCache(object_id="sloth_v1", cache_root=root).publish(
                source_glb=src, object_prompt_at_generation="sloth", generator_seed=42
            )
            client, stages = self._client(
                case_root=Path(tmp) / "case",
                cache_root=root,
                object_id="sloth_v1",
            )
            self.assertEqual(client.cache_resolution.status, CACHE_STATUS_HIT)
            self.assertFalse(client.reuse_sam31_model)
            self.assertEqual(
                set(client._prewarm_stages()),
                {stages["align"], stages["sample"]},
            )

    def test_corrupt_entry_fails_client_construction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cache"
            src = _write_box_glb(Path(tmp) / "gen" / "object.glb")
            ShapePriorMeshCache(object_id="sloth_v1", cache_root=root).publish(
                source_glb=src, object_prompt_at_generation="sloth", generator_seed=42
            )
            entry = root / mesh_cache.SCHEMA_DIR_NAME / "sloth_v1"
            (entry / "object.glb").write_bytes(b"corrupted")
            with self.assertRaises(ShapePriorMeshCacheError):
                self._client(
                    case_root=Path(tmp) / "case", cache_root=root, object_id="sloth_v1"
                )

    def test_manager_reuses_initial_sam31_only_when_generation_is_needed(self) -> None:
        from demo_v6_2.shape_prior.warmup import (  # noqa: PLC0415
            ShapePriorWarmupManager,
        )

        def manager(client, *, enabled=True):
            return ShapePriorWarmupManager(
                enabled=enabled,
                client=client,
                input_source="fake-live",
                depth_backend_label="realsense",
                depth_source="realsense",
                profile_json=None,
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cache"
            miss_client, _ = self._client(
                case_root=Path(tmp) / "miss_case",
                cache_root=root,
                object_id="miss_v1",
            )
            source = _write_box_glb(Path(tmp) / "source" / "object.glb")
            ShapePriorMeshCache(object_id="hit_v1", cache_root=root).publish(
                source_glb=source,
                object_prompt_at_generation="sloth",
                generator_seed=42,
            )
            hit_client, _ = self._client(
                case_root=Path(tmp) / "hit_case",
                cache_root=root,
                object_id="hit_v1",
            )

            self.assertTrue(manager(miss_client).requires_sam31_reuse)
            self.assertFalse(manager(hit_client).requires_sam31_reuse)
            self.assertFalse(manager(miss_client, enabled=False).requires_sam31_reuse)


class ClientRequestPathTests(unittest.TestCase):
    """Exercise cache hit/miss/disabled request branches without GPU work."""

    @staticmethod
    def _client(*, case_root: Path, cache_root: Path, object_id: str | None):
        from demo_v6_2.shape_prior.warmup import (  # noqa: PLC0415
            ShapePriorLocalClient,
        )

        return ShapePriorLocalClient(
            case_root=case_root,
            object_prompt="sloth",
            controller_name="hand",
            object_id=object_id,
            cache_root=cache_root,
        )

    def _request_with_fake_stages(
        self,
        client,
        *,
        calls: list[str],
        segment_options: list[dict[str, object]],
        fail_stage: str | None = None,
    ):
        from demo_v6_2.shape_prior.warmup import (  # noqa: PLC0415
            PREWARM_STAGE_ALIGN,
            PREWARM_STAGE_GENERATE,
            PREWARM_STAGE_SAMPLE,
            PREWARM_STAGE_UPSCALE,
        )

        shape_dir = client.case_root / client.case_name / "shape"

        def run_subprocess_stage(
            stage,
            _command,
            *,
            env,
            prewarmed_stages,
            defer_reap=None,
        ):
            del env, prewarmed_stages, defer_reap
            calls.append(stage)
            if stage == fail_stage:
                raise RuntimeError(f"forced {stage} failure")
            if stage == PREWARM_STAGE_UPSCALE:
                output = shape_dir / "high_resolution.png"
                output.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(output)
            elif stage == PREWARM_STAGE_GENERATE:
                _write_box_glb(shape_dir / mesh_cache.MESH_FILENAME)
            elif stage == PREWARM_STAGE_ALIGN:
                _write_box_glb(shape_dir / "matching" / "final_mesh.glb")
            elif stage == PREWARM_STAGE_SAMPLE:
                from demo_v6_2.shape_prior import sample as sample_stage  # noqa: PLC0415

                sample_stage.write_shape_prior_candidates(
                    shape_dir / sample_stage.CANDIDATES_FILENAME,
                    raw_surface_points=np.asarray(
                        [[0.0, 0.0, 0.0]], dtype=np.float64
                    ),
                    raw_interior_points=np.asarray(
                        [[0.1, 0.1, 0.1]], dtype=np.float64
                    ),
                )
            else:  # pragma: no cover - an unexpected stage is a test failure
                raise AssertionError(f"unexpected subprocess stage: {stage}")
            return 0.1, {
                "execution_mode": "cold",
                "critical_path_ms": 0.1,
                "go_wall_time_s": None,
            }

        def completed_stage_details(stage, *, orchestration):
            return {
                "stage": stage,
                "execution_mode": orchestration["execution_mode"],
            }

        def segment_image(**kwargs):
            calls.append("segment_image")
            segment_options.append(dict(kwargs))
            if fail_stage == "segment_image":
                raise RuntimeError("forced segment_image failure")
            output = Path(kwargs["output_path"])
            Image.fromarray(np.zeros((2, 2, 4), dtype=np.uint8)).save(output)
            return output, {"execution_mode": "in_process"}

        with (
            patch.object(
                client,
                "_run_stage_maybe_prewarmed",
                side_effect=run_subprocess_stage,
            ),
            patch.object(
                client,
                "_completed_stage_details",
                side_effect=completed_stage_details,
            ),
            patch(
                "demo_v6_2.perception.sam31_image_segmentation."
                "segment_image_to_origin_rgba",
                side_effect=segment_image,
            ),
        ):
            return client.request_shape_prior(_frame0_request())

    def test_disabled_always_generates_without_cache_io(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "persistent_cache"
            client = self._client(
                case_root=root / "outputs" / "shape_prior_case",
                cache_root=cache_root,
                object_id=None,
            )
            calls: list[str] = []
            segment_options: list[dict[str, object]] = []
            result = self._request_with_fake_stages(
                client,
                calls=calls,
                segment_options=segment_options,
            )

            self.assertEqual(
                calls,
                ["upscale", "segment_image", "generate", "align", "sample"],
            )
            self.assertFalse(cache_root.exists())
            self.assertFalse(result.metadata["shape_prior_cache_enabled"])
            self.assertIsNone(result.metadata["shape_prior_cache_hit"])
            self.assertEqual(
                result.metadata["shape_prior_canonical_mesh_source"],
                "generated",
            )
            self.assertRegex(
                result.metadata["shape_prior_canonical_mesh_sha256"],
                r"^[0-9a-f]{64}$",
            )
            self.assertEqual(result.metadata["shape_prior_object_name"], "sloth")
            self.assertTrue(segment_options[0]["reuse_model"])

    def test_miss_generates_publishes_then_becomes_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "persistent_cache"
            client = self._client(
                case_root=root / "outputs" / "shape_prior_case",
                cache_root=cache_root,
                object_id="sloth_v1",
            )
            calls: list[str] = []
            result = self._request_with_fake_stages(
                client,
                calls=calls,
                segment_options=[],
            )

            self.assertEqual(
                calls,
                ["upscale", "segment_image", "generate", "align", "sample"],
            )
            self.assertFalse(result.metadata["shape_prior_cache_hit"])
            self.assertEqual(
                result.metadata["shape_prior_cache_status"],
                CACHE_STATUS_MISS,
            )
            self.assertEqual(
                ShapePriorMeshCache(
                    object_id="sloth_v1",
                    cache_root=cache_root,
                )
                .resolve()
                .status,
                CACHE_STATUS_HIT,
            )

    def test_hit_skips_reconstruction_but_runs_align_and_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "persistent_cache"
            source = _write_box_glb(root / "source" / "object.glb")
            source_sha = sha256_file(source)
            ShapePriorMeshCache(
                object_id="sloth_v1",
                cache_root=cache_root,
            ).publish(
                source_glb=source,
                object_prompt_at_generation="stuffed animal",
                generator_seed=42,
            )
            client = self._client(
                case_root=root / "outputs" / "shape_prior_case",
                cache_root=cache_root,
                object_id="sloth_v1",
            )
            calls: list[str] = []
            result = self._request_with_fake_stages(
                client,
                calls=calls,
                segment_options=[],
            )

            self.assertEqual(calls, ["align", "sample"])
            self.assertTrue(result.metadata["shape_prior_cache_hit"])
            self.assertEqual(
                result.metadata["shape_prior_canonical_mesh_source"],
                "cache",
            )
            self.assertEqual(
                result.metadata["shape_prior_canonical_mesh_sha256"],
                source_sha,
            )
            self.assertEqual(
                result.metadata["shape_prior_cache_mesh_sha256"],
                source_sha,
            )
            self.assertEqual(
                result.metadata["shape_prior_object_prompt_at_generation"],
                "stuffed animal",
            )
            run_mesh = (
                client.case_root / client.case_name / "shape" / mesh_cache.MESH_FILENAME
            )
            self.assertEqual(sha256_file(run_mesh), source_sha)

            generate_entry = next(
                entry
                for entry in result.metadata["shape_prior_timing"]["critical_path"]
                if entry["stage"] == "generate"
            )
            self.assertEqual(
                generate_entry["details"]["execution_mode"],
                "cache_hit_materialize",
            )

    def test_miss_publishes_before_alignment_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "persistent_cache"
            client = self._client(
                case_root=root / "outputs" / "shape_prior_case",
                cache_root=cache_root,
                object_id="sloth_v1",
            )
            calls: list[str] = []
            with self.assertRaisesRegex(RuntimeError, "forced align failure"):
                self._request_with_fake_stages(
                    client,
                    calls=calls,
                    segment_options=[],
                    fail_stage="align",
                )

            self.assertEqual(
                calls,
                ["upscale", "segment_image", "generate", "align"],
            )
            self.assertEqual(
                ShapePriorMeshCache(
                    object_id="sloth_v1",
                    cache_root=cache_root,
                )
                .resolve()
                .status,
                CACHE_STATUS_HIT,
            )

    def test_hit_alignment_failure_never_regenerates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "persistent_cache"
            source = _write_box_glb(root / "source" / "object.glb")
            ShapePriorMeshCache(
                object_id="sloth_v1",
                cache_root=cache_root,
            ).publish(
                source_glb=source,
                object_prompt_at_generation="sloth",
                generator_seed=42,
            )
            client = self._client(
                case_root=root / "outputs" / "shape_prior_case",
                cache_root=cache_root,
                object_id="sloth_v1",
            )
            calls: list[str] = []
            with self.assertRaisesRegex(RuntimeError, "forced align failure"):
                self._request_with_fake_stages(
                    client,
                    calls=calls,
                    segment_options=[],
                    fail_stage="align",
                )
            self.assertEqual(calls, ["align"])

    def test_generate_failure_does_not_create_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "persistent_cache"
            client = self._client(
                case_root=root / "outputs" / "shape_prior_case",
                cache_root=cache_root,
                object_id="sloth_v1",
            )
            calls: list[str] = []
            with self.assertRaisesRegex(RuntimeError, "forced generate failure"):
                self._request_with_fake_stages(
                    client,
                    calls=calls,
                    segment_options=[],
                    fail_stage="generate",
                )
            self.assertEqual(calls, ["upscale", "segment_image", "generate"])
            self.assertEqual(
                ShapePriorMeshCache(
                    object_id="sloth_v1",
                    cache_root=cache_root,
                )
                .resolve()
                .status,
                CACHE_STATUS_MISS,
            )


class ConfigPlumbingTests(unittest.TestCase):
    def test_orchestrator_cli_cache_overrides(self) -> None:
        from demo_v6_2 import main_cli  # noqa: PLC0415

        args = main_cli.build_parser().parse_args(
            [
                "--shape-prior-object",
                "sloth_v1",
                "--shape-prior-object-prompt",
                "sloth",
                "--shape-prior-cache-root",
                "/tmp/x",
            ]
        )
        self.assertEqual(args.shape_prior_object, "sloth_v1")
        self.assertEqual(args.shape_prior_object_prompt, "sloth")
        self.assertEqual(Path(args.shape_prior_cache_root), Path("/tmp/x"))

    def test_prompt_is_forwarded_when_shape_prior_warmup_is_disabled(self) -> None:
        from demo_v6_2 import main_cli  # noqa: PLC0415
        from demo_v6_2.main_subprocess import (  # noqa: PLC0415
            build_main_data_processing_command,
        )

        args = main_cli.build_parser().parse_args(
            [
                "--no-shape-prior-warmup",
                "--shape-prior-object-prompt",
                "blue sloth",
            ]
        )
        command = build_main_data_processing_command(
            args,
            capture_dir=Path("/tmp/cap"),
            profile_json=Path("/tmp/p.json"),
        )
        prompt_index = command.index("--shape-prior-object-prompt")
        self.assertEqual(command[prompt_index + 1], "blue sloth")
        self.assertNotIn("--shape-prior-object", command)
        self.assertNotIn("--shape-prior-cache-root", command)

    def test_orchestrator_rejects_enabled_cache_under_output(self) -> None:
        from demo_v6_2 import main_cli  # noqa: PLC0415
        from demo_v6_2.orchestration.run_config import (  # noqa: PLC0415
            OrchestratorRunConfig,
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "outputs"
            args = main_cli.build_parser().parse_args(
                [
                    "--base-path",
                    str(output_root),
                    "--shape-prior-object",
                    "sloth_v1",
                    "--shape-prior-cache-root",
                    str(output_root / "cache"),
                ]
            )
            with self.assertRaisesRegex(ValueError, "must not live under"):
                OrchestratorRunConfig.from_args(args)

    def test_orchestrator_rejects_string_null_identity(self) -> None:
        from demo_v6_2 import main_cli  # noqa: PLC0415
        from demo_v6_2.orchestration.run_config import (  # noqa: PLC0415
            OrchestratorRunConfig,
        )

        args = main_cli.build_parser().parse_args(["--shape-prior-object", "none"])
        with self.assertRaisesRegex(ValueError, "YAML null"):
            OrchestratorRunConfig.from_args(args)

    def test_subprocess_forwarding_omits_object_when_disabled(self) -> None:
        from demo_v6_2 import main_cli  # noqa: PLC0415
        from demo_v6_2.main_subprocess import (  # noqa: PLC0415
            build_main_data_processing_command,
        )

        base = main_cli.build_parser().parse_args(
            ["--shape-prior-warmup", "--shape-prior-controller-name", "hand"]
        )
        base.shape_prior_object = None
        cmd = build_main_data_processing_command(
            base,
            capture_dir=Path("/tmp/cap"),
            profile_json=Path("/tmp/p.json"),
        )
        self.assertIn("--shape-prior-object-prompt", cmd)
        self.assertIn("--shape-prior-cache-root", cmd)
        self.assertNotIn("--shape-prior-object", cmd)

        enabled = main_cli.build_parser().parse_args(
            [
                "--shape-prior-warmup",
                "--shape-prior-controller-name",
                "hand",
                "--shape-prior-object",
                "sloth_v1",
            ]
        )
        cmd = build_main_data_processing_command(
            enabled,
            capture_dir=Path("/tmp/cap"),
            profile_json=Path("/tmp/p.json"),
        )
        self.assertIn("--shape-prior-object", cmd)
        self.assertEqual(cmd[cmd.index("--shape-prior-object") + 1], "sloth_v1")

    def test_camera_cli_parses_cache_flags(self) -> None:
        from demo_v6_2.mdp import cli as mdp_cli  # noqa: PLC0415

        args = mdp_cli.build_parser().parse_args(
            [
                "--shape-prior-object",
                "sloth_v1",
                "--shape-prior-object-prompt",
                "sloth",
                "--shape-prior-cache-root",
                "/tmp/x",
            ]
        )
        self.assertEqual(args.shape_prior_object, "sloth_v1")
        self.assertEqual(args.shape_prior_object_prompt, "sloth")
        self.assertEqual(Path(args.shape_prior_cache_root), Path("/tmp/x"))

    def test_camera_cli_requires_prompt_for_shape_prior_warmup(self) -> None:
        from demo_v6_2.mdp import cli as mdp_cli  # noqa: PLC0415

        args = mdp_cli.build_parser().parse_args(
            [
                "--shape-prior-warmup",
                "--shape-prior-controller-name",
                "hand",
                "--shape-prior-object-prompt",
                "",
                "--track-mode",
                "controller-only",
            ]
        )
        with self.assertRaisesRegex(ValueError, "object-prompt"):
            mdp_cli.validate_and_normalize_args(args)


if __name__ == "__main__":
    unittest.main()
