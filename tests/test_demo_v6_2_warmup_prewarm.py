"""Unit acceptance for the warm-up critical-path overlaps (2026-07-19).

Covers, without GPU or model weights:
- stage_prewarm PRERENDER directive handling inside wait_for_go
- PrewarmWorkerPool.send_directive against live, popped, and dead workers
- ShapePriorLocalClient.send_align_prerender payload + cache gating
- ShapePriorWarmupManager.notify_frame0_geometry forwarding + error swallow
- mdp.warmup.Sam31PreloadThread success / failure / unstarted semantics

Numeric identity of the hoisted-SuperPoint and prerender paths is verified
separately by a GPU A/B run of the align stage on a fixed case (old vs new
outputs compared byte-for-byte).
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import trimesh

from demo_v6_2.utils import stage_prewarm


def _write_box_glb(path: Path) -> Path:
    """Write a small valid mesh GLB (a unit box) to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(str(path))
    return path


class WaitForGoDirectiveTests(unittest.TestCase):
    """wait_for_go keeps its GO/EXIT contract and adds PRERENDER handling."""

    def _wait(self, stdin_text: str, **kwargs):
        with patch.object(sys, "stdin", io.StringIO(stdin_text)):
            return stage_prewarm.wait_for_go("align", **kwargs)

    def test_go_runs_stage(self) -> None:
        self.assertTrue(self._wait("GO\n"))

    def test_exit_and_eof_skip_stage(self) -> None:
        self.assertFalse(self._wait("EXIT\n"))
        self.assertFalse(self._wait(""))

    def test_prerender_directive_reaches_handler_then_go(self) -> None:
        seen: list[str] = []
        result = self._wait(
            'PRERENDER {"width": 848}\nGO\n',
            on_directive=seen.append,
        )
        self.assertTrue(result)
        self.assertEqual(seen, ['{"width": 848}'])

    def test_prerender_without_handler_is_protocol_error(self) -> None:
        with self.assertRaises(ValueError):
            self._wait("PRERENDER {}\n")

    def test_unexpected_line_still_raises_with_handler(self) -> None:
        with self.assertRaises(ValueError):
            self._wait("BOGUS\n", on_directive=lambda payload: None)


class PoolSendDirectiveTests(unittest.TestCase):
    """send_directive writes to waiting workers and degrades to False."""

    _READER_SCRIPT = (
        "import sys\n"
        "while True:\n"
        "    line = sys.stdin.readline()\n"
        "    if not line or line.strip() in ('GO', 'EXIT'):\n"
        "        break\n"
    )

    def _pool_with_worker(self, script: str):
        from demo_v6_2.shape_prior.warmup import (  # noqa: PLC0415
            PREWARM_STAGE_ALIGN,
            PrewarmWorkerPool,
        )

        pool = PrewarmWorkerPool()
        pool.spawn(
            {PREWARM_STAGE_ALIGN: [sys.executable, "-u", "-c", script]},
            dict(os.environ),
            active_stages=(PREWARM_STAGE_ALIGN,),
        )
        return pool, PREWARM_STAGE_ALIGN

    def test_absent_worker_returns_false(self) -> None:
        from demo_v6_2.shape_prior.warmup import (  # noqa: PLC0415
            PREWARM_STAGE_ALIGN,
            PrewarmWorkerPool,
        )

        self.assertFalse(
            PrewarmWorkerPool().send_directive(PREWARM_STAGE_ALIGN, "PRERENDER {}")
        )

    def test_directive_then_go_and_popped_worker_refuses(self) -> None:
        pool, stage = self._pool_with_worker(self._READER_SCRIPT)
        try:
            self.assertTrue(pool.send_directive(stage, "PRERENDER {}"))
            result = pool.pop_and_go(stage)
            self.assertIsNotNone(result)
            self.assertFalse(pool.send_directive(stage, "PRERENDER {}"))
        finally:
            pool.close()

    def test_dead_worker_returns_false(self) -> None:
        pool, stage = self._pool_with_worker("pass\n")
        try:
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                with pool._lock:
                    worker = pool._workers[stage]
                if worker.poll() is not None:
                    break
                time.sleep(0.05)
            self.assertIsNotNone(worker.poll())
            self.assertFalse(pool.send_directive(stage, "PRERENDER {}"))
        finally:
            pool.close()


class ClientSendAlignPrerenderTests(unittest.TestCase):
    """send_align_prerender fires only on a cache hit, with a full payload."""

    def _client(self, *, case_root: Path, cache_root: Path, object_id):
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

    def test_cache_disabled_sends_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = self._client(
                case_root=Path(tmp) / "case",
                cache_root=Path(tmp) / "cache",
                object_id=None,
            )
            sent: list[tuple[str, str]] = []
            with patch.object(
                client._prewarm_pool,
                "send_directive",
                lambda stage, line: sent.append((stage, line)) or True,
            ):
                self.assertFalse(
                    client.send_align_prerender(width=848, height=480, fx_color=430.5)
                )
            self.assertEqual(sent, [])

    def test_cache_hit_sends_verified_payload(self) -> None:
        from demo_v6_2.shape_prior.mesh_cache import (  # noqa: PLC0415
            ShapePriorMeshCache,
            sha256_file,
        )
        from demo_v6_2.shape_prior.warmup import (  # noqa: PLC0415
            PREWARM_STAGE_ALIGN,
        )

        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp) / "cache"
            source_glb = _write_box_glb(Path(tmp) / "generated" / "object.glb")
            ShapePriorMeshCache(object_id="sloth", cache_root=cache_root).publish(
                source_glb=source_glb,
                object_prompt_at_generation="sloth",
                generator_seed=42,
            )
            client = self._client(
                case_root=Path(tmp) / "case",
                cache_root=cache_root,
                object_id="sloth",
            )
            self.assertTrue(client.cache_resolution.hit)
            sent: list[tuple[str, str]] = []
            with patch.object(
                client._prewarm_pool,
                "send_directive",
                lambda stage, line: sent.append((stage, line)) or True,
            ):
                self.assertTrue(
                    client.send_align_prerender(width=848, height=480, fx_color=430.5)
                )
            (stage, line), = sent
            self.assertEqual(stage, PREWARM_STAGE_ALIGN)
            self.assertTrue(line.startswith(stage_prewarm.PRERENDER_DIRECTIVE_PREFIX))
            payload = json.loads(
                line[len(stage_prewarm.PRERENDER_DIRECTIVE_PREFIX) :]
            )
            self.assertEqual(
                payload,
                {
                    "mesh_path": str(client.cache_resolution.mesh_path),
                    "mesh_sha256": sha256_file(client.cache_resolution.mesh_path),
                    "width": 848,
                    "height": 480,
                    "fx": 430.5,
                },
            )


class _FakeClient:
    """Just enough of ShapePriorLocalClient for manager-level tests."""

    def __init__(self, *, raise_on_send: bool = False) -> None:
        from demo_v6_2.shape_prior.mesh_cache import (  # noqa: PLC0415
            CACHE_STATUS_DISABLED,
            CacheResolution,
        )

        self.cache_resolution = CacheResolution(
            status=CACHE_STATUS_DISABLED,
            object_id=None,
            cache_root=None,
            entry_dir=None,
            mesh_path=None,
            manifest=None,
        )
        self.cache_root = Path("/nonexistent-cache-root")
        self.raise_on_send = raise_on_send
        self.sent: list[dict] = []

    def send_align_prerender(self, *, width, height, fx_color) -> bool:
        if self.raise_on_send:
            raise RuntimeError("boom")
        self.sent.append({"width": width, "height": height, "fx_color": fx_color})
        return True


class ManagerNotifyFrame0GeometryTests(unittest.TestCase):
    """notify_frame0_geometry forwards on enabled managers and never raises."""

    def _manager(self, *, enabled: bool, client):
        from demo_v6_2.shape_prior.warmup import (  # noqa: PLC0415
            ShapePriorWarmupManager,
        )

        return ShapePriorWarmupManager(
            enabled=enabled,
            client=client,
            input_source="fake-live",
            depth_backend_label="realsense",
            depth_source="realsense",
            profile_json=None,
        )

    def test_forwards_cast_geometry(self) -> None:
        client = _FakeClient()
        self._manager(enabled=True, client=client).notify_frame0_geometry(
            width=848, height=480, fx_color=430.25
        )
        self.assertEqual(
            client.sent,
            [{"width": 848, "height": 480, "fx_color": 430.25}],
        )

    def test_disabled_or_clientless_is_noop(self) -> None:
        client = _FakeClient()
        self._manager(enabled=False, client=client).notify_frame0_geometry(
            width=848, height=480, fx_color=430.25
        )
        self.assertEqual(client.sent, [])
        self._manager(enabled=True, client=None).notify_frame0_geometry(
            width=848, height=480, fx_color=430.25
        )

    def test_client_failure_is_swallowed(self) -> None:
        manager = self._manager(enabled=True, client=_FakeClient(raise_on_send=True))
        manager.notify_frame0_geometry(width=848, height=480, fx_color=430.25)


class Sam31PreloadThreadTests(unittest.TestCase):
    """Sam31PreloadThread loads in the background and re-raises at wait."""

    def test_wait_before_start_is_zero_noop(self) -> None:
        from demo_v6_2.mdp.warmup import Sam31PreloadThread  # noqa: PLC0415

        timings = Sam31PreloadThread(device="cuda").wait_for_model()
        self.assertEqual(timings, {"preload_ms": 0.0, "join_wait_ms": 0.0})

    def test_success_returns_timings_and_uses_device(self) -> None:
        from demo_v6_2.mdp.warmup import Sam31PreloadThread  # noqa: PLC0415

        devices: list[str] = []
        with patch(
            "demo_v6_2.perception.sam31_image_segmentation."
            "preload_sam31_image_runtime",
            lambda *, device: devices.append(device) or {},
        ):
            preload = Sam31PreloadThread(device="cuda:1")
            preload.start()
            timings = preload.wait_for_model()
        self.assertEqual(devices, ["cuda:1"])
        self.assertGreaterEqual(timings["preload_ms"], 0.0)
        self.assertGreaterEqual(timings["join_wait_ms"], 0.0)

    def test_failure_reraises_at_wait(self) -> None:
        from demo_v6_2.mdp.warmup import Sam31PreloadThread  # noqa: PLC0415

        def _boom(*, device):
            raise RuntimeError("checkpoint missing")

        with patch(
            "demo_v6_2.perception.sam31_image_segmentation."
            "preload_sam31_image_runtime",
            _boom,
        ):
            preload = Sam31PreloadThread(device="cuda")
            preload.start()
            with self.assertRaises(RuntimeError):
                preload.wait_for_model()

    def test_double_start_rejected(self) -> None:
        from demo_v6_2.mdp.warmup import Sam31PreloadThread  # noqa: PLC0415

        with patch(
            "demo_v6_2.perception.sam31_image_segmentation."
            "preload_sam31_image_runtime",
            lambda *, device: {},
        ):
            preload = Sam31PreloadThread(device="cuda")
            preload.start()
            preload.wait_for_model()
            with self.assertRaises(RuntimeError):
                preload.start()


class PerceptionPreloaderTests(unittest.TestCase):
    """Camera-free preload legs + the frame-0 readiness barrier."""

    @staticmethod
    def _mode(**overrides):
        from types import SimpleNamespace  # noqa: PLC0415

        base = {
            "object_tracking_enabled": False,
            "controller_tracking_enabled": False,
            "lossless_enabled": False,
            "tracker_enabled": False,
        }
        base.update(overrides)
        return SimpleNamespace(**base)

    @staticmethod
    def _args(track_mode: str = "controller-object"):
        from types import SimpleNamespace  # noqa: PLC0415

        return SimpleNamespace(track_mode=track_mode, device="cuda")

    def test_no_legs_when_nothing_consumes_frame0(self) -> None:
        from demo_v6_2.mdp.preload import PerceptionPreloader  # noqa: PLC0415

        preloader = PerceptionPreloader(
            args=self._args(track_mode="none"), mode=self._mode()
        )
        self.assertFalse(preloader.has_frame0_consumers)
        preloader.start()
        self.assertTrue(preloader.wait_frame0_consumers_ready(timeout=0.0))

    def test_barrier_waits_for_every_leg(self) -> None:
        import threading  # noqa: PLC0415

        from demo_v6_2.mdp import preload as preload_module  # noqa: PLC0415

        release = threading.Event()

        def slow_edgetam(_args):
            release.wait(timeout=5.0)
            return "edgetam-runtime"

        with (
            patch.object(preload_module, "load_edgetam_runtime", slow_edgetam),
            patch(
                "demo_v6_2.mdp.tracker.build_tracker_adapter",
                lambda args: "adapter",
            ),
        ):
            preloader = preload_module.PerceptionPreloader(
                args=self._args(),
                mode=self._mode(lossless_enabled=True, tracker_enabled=True),
            )
            preloader.start()
            self.assertTrue(preloader.has_frame0_consumers)
            self.assertFalse(preloader.wait_frame0_consumers_ready(timeout=0.05))
            release.set()
            # Legs done is not enough: the seg worker must also report ready
            # (joins + optional precompile), so a slow precompile can never
            # eat into the hold-still window.
            self.assertFalse(preloader.wait_frame0_consumers_ready(timeout=0.2))
            preloader.mark_seg_frame0_ready()
            self.assertTrue(preloader.wait_frame0_consumers_ready(timeout=5.0))
            self.assertEqual(preloader.join_edgetam(), "edgetam-runtime")
            self.assertEqual(preloader.join_tracker(), "adapter")

    def test_failed_leg_opens_barrier_and_reraises_at_join(self) -> None:
        from demo_v6_2.mdp import preload as preload_module  # noqa: PLC0415

        def boom(_args):
            raise RuntimeError("edgetam load failed")

        with patch.object(preload_module, "load_edgetam_runtime", boom):
            preloader = preload_module.PerceptionPreloader(
                args=self._args(), mode=self._mode()
            )
            preloader.start()
            # A failed leg opens the barrier WITHOUT the seg-ready mark (the
            # seg worker will re-raise at its join and never mark).
            self.assertTrue(preloader.wait_frame0_consumers_ready(timeout=5.0))
            with self.assertRaisesRegex(RuntimeError, "edgetam load failed"):
                preloader.join_edgetam()

    def test_sam31_leg_follows_tracking_flags(self) -> None:
        from demo_v6_2.mdp.preload import PerceptionPreloader  # noqa: PLC0415

        with patch(
            "demo_v6_2.perception.sam31_image_segmentation."
            "preload_sam31_image_runtime",
            lambda *, device: {},
        ):
            with patch(
                "demo_v6_2.mdp.preload.load_edgetam_runtime",
                lambda args: "rt",
            ):
                preloader = PerceptionPreloader(
                    args=self._args(),
                    mode=self._mode(object_tracking_enabled=True),
                )
                preloader.start()
                preloader.mark_seg_frame0_ready()
                self.assertTrue(preloader.wait_frame0_consumers_ready(timeout=5.0))
                timings = preloader.join_sam31()
                self.assertIsNotNone(timings)
        preloader_none = PerceptionPreloader(
            args=self._args(track_mode="none"), mode=self._mode()
        )
        self.assertIsNone(preloader_none.join_sam31())


class LiveFrame0BarrierTests(unittest.TestCase):
    """The live capture worker designates frame 0 only after the barrier."""

    def test_barrier_pumps_preview_then_frame0_is_next_camera_frame(self) -> None:
        import threading  # noqa: PLC0415
        from types import SimpleNamespace  # noqa: PLC0415
        from unittest import mock  # noqa: PLC0415

        import numpy as np  # noqa: PLC0415

        from demo_v6_2.mdp import capture as capture_module  # noqa: PLC0415
        from demo_v6_2.utils.concurrency import LatestSlot  # noqa: PLC0415

        class _ColorFrame:
            def get_data(self):
                return np.zeros((2, 2, 3), dtype=np.uint8)

        class _FrameSet:
            def get_color_frame(self):
                return _ColorFrame()

        stop_event = threading.Event()
        input_preview_slot: LatestSlot = LatestSlot()
        capture_slot = mock.Mock()
        lossless = mock.Mock()
        lossless.submit_frame.return_value = True
        lossless.first_pair_published.wait.return_value = True
        sampler = mock.Mock()
        sampler.pop_due.return_value = None
        sampler.start.side_effect = lambda *, first_publish_s: stop_event.set()
        preload = mock.Mock()
        preload.has_frame0_consumers = True
        # Two barrier misses -> two preview pump publishes -> barrier opens.
        preload.wait_frame0_consumers_ready.side_effect = [False, False, True]

        stage = capture_module.CaptureStage(
            args=SimpleNamespace(
                track_mode="controller-object",
                depth_source="none",
                write_input_rgb_timeline=False,
            ),
            mode=SimpleNamespace(
                fake_live_input=False,
                lossless_enabled=True,
                # Huge fps -> tiny pacing period so every pump tick publishes.
                lossless_input_fps=1_000_000.0,
            ),
            session=SimpleNamespace(
                camera_runtime=SimpleNamespace(
                    pipeline=SimpleNamespace(
                        wait_for_frames=mock.Mock(return_value=_FrameSet())
                    ),
                    align=None,
                    intrinsics=mock.sentinel.intrinsics,
                    depth_scale_m_per_unit=0.001,
                    k_ir_left=None,
                    t_ir_left_to_color=None,
                    k_color=np.eye(3, dtype=np.float32),
                    ir_baseline_m=0.0,
                ),
                headless_capture_writer=None,
            ),
            lossless=lossless,
            capture_slot=capture_slot,
            input_preview_slot=input_preview_slot,
            stage_stats=mock.Mock(),
            preload=preload,
            first_frame_segmented=SimpleNamespace(
                wait=mock.Mock(return_value=True)
            ),
            stop_event=stop_event,
            fatal=mock.Mock(),
        )

        with mock.patch.object(
            capture_module, "LiveLatestFrameSampler", return_value=sampler
        ):
            stage.run()

        # Pump frames are display-only: the pipeline saw nothing until the
        # barrier opened, then exactly one packet (frame 0, output seq 0).
        self.assertEqual(capture_slot.put.call_count, 1)
        frame0 = capture_slot.put.call_args[0][0]
        self.assertEqual(frame0.seq, 0)
        # Preview seq stays monotonic across pump frames + frame 0.
        latest = input_preview_slot.get_latest_after(-1)
        self.assertIsNotNone(latest)
        self.assertEqual(latest.seq, 3)


class PendingStageReapTests(unittest.TestCase):
    """pop_and_go_nowait returns at the snapshot; reap collects exit codes."""

    _TAIL_SCRIPT = (
        "import pathlib, sys, time\n"
        "sys.stdin.readline()\n"
        "pathlib.Path(sys.argv[1]).write_text('done')\n"
        "time.sleep(0.5)\n"
    )
    _FAIL_SCRIPT = "import sys\nsys.stdin.readline()\nsys.exit(3)\n"

    def _pool_with_worker(self, script: str, *extra_args: str):
        from demo_v6_2.shape_prior.warmup import (  # noqa: PLC0415
            PREWARM_STAGE_ALIGN,
            PrewarmWorkerPool,
        )

        pool = PrewarmWorkerPool()
        pool.spawn(
            {
                PREWARM_STAGE_ALIGN: [
                    sys.executable,
                    "-u",
                    "-c",
                    script,
                    *extra_args,
                ]
            },
            dict(os.environ),
            active_stages=(PREWARM_STAGE_ALIGN,),
        )
        return pool, PREWARM_STAGE_ALIGN

    def test_snapshot_returns_before_worker_exit_then_reap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "snapshot.json"
            pool, stage = self._pool_with_worker(self._TAIL_SCRIPT, str(marker))
            try:
                pending = pool.pop_and_go_nowait(stage)
                self.assertIsNotNone(pending)
                stage_ms = pending.wait_snapshot(marker.is_file)
                self.assertGreaterEqual(stage_ms, 0.0)
                # The 0.5s exit tail is still running when the snapshot lands.
                self.assertIsNone(pending.worker.poll())
                pending.reap()
                self.assertTrue(pending.reaped)
                self.assertEqual(pending.worker.returncode, 0)
            finally:
                pool.close()

    def test_nonzero_exit_raises_during_snapshot_wait(self) -> None:
        import subprocess  # noqa: PLC0415

        pool, stage = self._pool_with_worker(self._FAIL_SCRIPT)
        try:
            pending = pool.pop_and_go_nowait(stage)
            self.assertIsNotNone(pending)
            with self.assertRaises(subprocess.CalledProcessError):
                pending.wait_snapshot(lambda: False)
            self.assertTrue(pending.reaped)
        finally:
            pool.close()


class DeferredSam31ReleaseTests(unittest.TestCase):
    """The cache-hit SAM3.1 release comes off the frame-0 critical path."""

    def _run_bundle(self, *, defer_release: bool, fail: bool = False):
        from types import SimpleNamespace  # noqa: PLC0415

        import numpy as np  # noqa: PLC0415

        from demo_v6_2.mdp import warmup as mdp_warmup  # noqa: PLC0415

        color = np.zeros((4, 4, 3), dtype=np.uint8)
        mask = np.ones((4, 4), dtype=bool)
        calls: list[str] = []

        def fake_segmentation(**kwargs):
            if fail:
                raise RuntimeError("forced sam31 failure")
            return {
                "masks_by_label": {"sloth": [mask]},
                "timing_ms": {"inference_ms": 1.0},
            }

        with (
            patch(
                "demo_v6_2.perception.sam31_image_segmentation."
                "run_image_segmentation",
                fake_segmentation,
            ),
            patch(
                "demo_v6_2.perception.sam31_image_segmentation."
                "parse_text_prompts",
                lambda prompt: [prompt],
            ),
            patch.object(
                mdp_warmup,
                "release_sam31_runtime_resources",
                lambda device: calls.append("release") or 1.0,
            ),
            patch.object(
                mdp_warmup,
                "_reclaim_cuda_memory",
                lambda device, **kwargs: calls.append("trim") or 1.0,
            ),
        ):
            bundle, timing = None, None
            try:
                bundle, timing = mdp_warmup.run_sam31_first_frame_mask_bundle(
                    color,
                    SimpleNamespace(
                        device="cuda", shape_prior_object_prompt="sloth"
                    ),
                    SimpleNamespace(
                        object_tracking_enabled=True,
                        controller_tracking_enabled=False,
                    ),
                    reuse_sam31_runtime=False,
                    defer_release=defer_release,
                )
            except RuntimeError:
                if not fail:
                    raise
        return calls, timing

    def test_defer_release_skips_inline_release(self) -> None:
        calls, timing = self._run_bundle(defer_release=True)
        self.assertEqual(calls, [])
        self.assertEqual(timing.release_cleanup_ms, 0.0)

    def test_without_defer_release_is_inline(self) -> None:
        calls, _timing = self._run_bundle(defer_release=False)
        self.assertEqual(calls, ["release"])

    def test_failure_path_always_releases_inline(self) -> None:
        calls, _timing = self._run_bundle(defer_release=True, fail=True)
        self.assertEqual(calls, ["release"])


class CameraOpenPrerenderHintTests(unittest.TestCase):
    """The align prerender hint fires from camera metadata at source open."""

    def _fake_runtime(self, *, warmup_enabled: bool, k_color):
        from types import SimpleNamespace  # noqa: PLC0415
        from unittest import mock  # noqa: PLC0415

        return SimpleNamespace(
            args=SimpleNamespace(shape_prior_warmup=warmup_enabled),
            session=SimpleNamespace(
                width=848,
                height=480,
                camera_runtime=SimpleNamespace(k_color=k_color),
            ),
            shape_prior_manager=mock.Mock(),
        )

    def test_hint_uses_session_dimensions_and_runtime_fx(self) -> None:
        import numpy as np  # noqa: PLC0415

        from demo_v6_2.mdp.runtime import MainDataProcessingDemo  # noqa: PLC0415

        k_color = np.eye(3, dtype=np.float32)
        k_color[0, 0] = 430.25
        fake = self._fake_runtime(warmup_enabled=True, k_color=k_color)
        MainDataProcessingDemo._notify_frame0_geometry_from_camera(fake)
        fake.shape_prior_manager.notify_frame0_geometry.assert_called_once_with(
            width=848, height=480, fx_color=430.25
        )

    def test_hint_skipped_without_warmup_or_intrinsics(self) -> None:
        from demo_v6_2.mdp.runtime import MainDataProcessingDemo  # noqa: PLC0415

        disabled = self._fake_runtime(warmup_enabled=False, k_color=None)
        MainDataProcessingDemo._notify_frame0_geometry_from_camera(disabled)
        disabled.shape_prior_manager.notify_frame0_geometry.assert_not_called()
        no_intrinsics = self._fake_runtime(warmup_enabled=True, k_color=None)
        MainDataProcessingDemo._notify_frame0_geometry_from_camera(no_intrinsics)
        no_intrinsics.shape_prior_manager.notify_frame0_geometry.assert_not_called()


if __name__ == "__main__":
    unittest.main()
