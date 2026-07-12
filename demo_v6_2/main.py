#!/usr/bin/env python3
"""Demo v6.2 realtime orchestration entrypoint.

This runner owns process boundaries, GPU routing, and artifact publication. The
actual camera/tracker stack runs in ``demo_v6_2/main_data_processing.py``; SAM3D
shape-prior warm-up runs as local one-shot stages; the default side-by-side
visualizer starts as soon as capture starts. See ``demo_v6_2/PIPELINE.md`` for the
full stage-by-stage map (the 23 pipeline questions answered in code). All three
processes append lifecycle events to ``<base_path>/pipeline_status.jsonl`` so the
viewer can show what the pipeline is doing right now / whether warm-up failed.

Note: some emitted telemetry strings (``demo_version``/``case_prefix``/
``schema_version``) still read ``demo_v6_1`` so online chunk / final_data output
stays byte-identical to the frozen demo_v6_1 reference; the code itself is fully
self-contained under ``demo_v6_2``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Sequence


# Keep this repo at the front of the import path when the script is launched
# from another working directory. Removing the existing entry first avoids a
# duplicate path while preserving the "current checkout wins" import order.
# This must run before importing any ``demo_v6_2.*`` module so those imports
# resolve when this file is executed as a script (``python demo_v6_2/main.py``).
_BOOTSTRAP_REPO_ROOT_STR = str(Path(__file__).resolve().parents[1])
if _BOOTSTRAP_REPO_ROOT_STR in sys.path:
    sys.path.remove(_BOOTSTRAP_REPO_ROOT_STR)
sys.path.insert(0, _BOOTSTRAP_REPO_ROOT_STR)

from demo_v6_2.chunk_data_stream import (
    stream_chunk_data_from_headless_capture,
    write_chunk_data_from_headless_capture,
)
from demo_v6_2.phystwin_shen_launch import (
    PhystwinShenLaunch,
    PhystwinShenLaunchError,
    launch_phystwin_shen,
)
from demo_v6_2.pipeline_status import (
    STAGE_CHUNK_COMMITTED,
    STAGE_DOWNSTREAM_START,
    STAGE_FATAL,
    STAGE_RUN_FINISHED,
    STAGE_RUN_START,
    PipelineStatusWriter,
)
from demo_v6_2.main_config import (
    DEFAULT_SAM31_CHECKPOINT_PATH,
    REPO_ROOT,
    SAM31_CHECKPOINT_ENV,
)
from demo_v6_2.main_cli import build_parser
from demo_v6_2.main_options import (
    _load_optional_points,
    _python_command_prefix,
    demo_visualizer_enabled,
    phystwin_shen_enabled,
    resolve_camera_serials,
    resolve_camera_source_replay_fps,
    resolve_chunk_frame_count,
    resolve_downstream_mode,
    resolve_main_data_processing_cuda_visible_devices,
    resolve_phystwin_shen_settings,
    resolve_shape_prior_warmup_cuda_visible_devices,
    resolve_visualizer_cuda_visible_devices,
    resolve_visualizer_layout,
    resolve_write_input_rgb_timeline,
    visualizer_start_policy,
    visualizer_uses_side_by_side,
)
from demo_v6_2.main_layout import (
    prepare_realtime_output_for_new_run,
    resolve_online_dir,
    resolve_run_summary_path,
    resolve_shape_prior_case_root,
    resolve_shape_prior_points_npz,
    resolve_static_data_path,
)
from demo_v6_2.main_subprocess import (
    _contract,
    _default_capture_dir,
    _start_visualizer,
    _stop_process,
    build_main_data_processing_command,
    build_visualizer_command,
    validate_runtime_args,
)
from demo_v6_2.utils.runtime_summary import _runtime_chunk_summary


# ---------------------------------------------------------------------------
# Run summary and entrypoint
# ---------------------------------------------------------------------------


COLLECT_FINISH_BANNER = "##################\ncollect finish\n##################"


def _stop_phystwin_launch(launch: PhystwinShenLaunch) -> int | None:
    """Terminate the saved full-pipeline process group."""
    try:
        return _stop_process(
            launch.pipeline_process,
            process_group_id=launch.process_group_id,
        )
    finally:
        launch.finish_pipeline_output_relay()


def _wait_for_phystwin_launch(launch: PhystwinShenLaunch) -> int:
    """Wait for normal completion and guarantee that no descendants remain."""
    try:
        return_code = int(launch.pipeline_process.wait())
    except BaseException:
        _stop_phystwin_launch(launch)
        raise
    _stop_phystwin_launch(launch)
    if return_code != 0:
        raise PhystwinShenLaunchError(
            f"Phystwin_shen full pipeline failed with return code {return_code}"
        )
    return return_code


def main(argv: Sequence[str] | None = None) -> int:
    """Run Demo v6.1 offline conversion or live/fake-live orchestration."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    chunk_frame_count = resolve_chunk_frame_count(args)
    validate_runtime_args(args, chunk_frame_count=chunk_frame_count)

    if bool(args.dry_run):
        print(json.dumps(_contract(args), indent=2, sort_keys=True))
        return 0

    base_path = Path(args.base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    startup_output_cleanup = prepare_realtime_output_for_new_run(
        base_path,
        clear_capture=args.source_headless_capture is None,
        legacy_case_prefix=str(args.case_prefix),
    )
    if args.source_headless_capture is not None:
        # Offline conversion path: consume an existing capture directory and
        # write online/static final_data products without launching camera or
        # visualizer subprocesses.
        manifests = write_chunk_data_from_headless_capture(
            args.source_headless_capture,
            base_path=base_path,
            case_prefix=str(args.case_prefix),
            chunk_frame_count=chunk_frame_count,
            fps=int(round(float(args.replay_fps))),
            max_chunks=args.max_chunks,
            surface_points=_load_optional_points(args.surface_points_npy),
            interior_points=_load_optional_points(args.interior_points_npy),
            asap_augment=bool(args.asap_augment),
            asap_mesh_path=args.asap_mesh_path,
        )
        summary = {
            "demo_version": "demo_v6_1",
            "mode": "source-headless-capture",
            "source_headless_capture": str(args.source_headless_capture),
            "base_path": str(base_path),
            "case_prefix": str(args.case_prefix),
            "output_format": "online-primary-static-case",
            "online_dir": str(resolve_online_dir(args)),
            "static_data_path": str(resolve_static_data_path(args)),
            "shape_prior_case_root": str(resolve_shape_prior_case_root(args)),
            "shape_prior_points_npz": str(resolve_shape_prior_points_npz(args)),
            "startup_output_cleanup": startup_output_cleanup,
            "chunk_frame_count": int(chunk_frame_count),
            "max_chunks": args.max_chunks,
            "chunk_count": int(len(manifests)),
            "chunks": manifests,
        }
        summary.update(_runtime_chunk_summary(manifests))
        summary_path = resolve_run_summary_path(base_path)
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    # Live pipeline-status stream (design question 23): the orchestrator, the
    # camera process, and the SAM3D shape-prior stages all append lifecycle
    # events to <base_path>/pipeline_status.jsonl and the visualizer tails it to
    # show what the pipeline is doing right now / whether warm-up failed. Writing
    # is best-effort and never affects the published products.
    run_start_monotonic_s = time.monotonic()
    status = PipelineStatusWriter(base_path, "orchestrator")
    status.emit(
        STAGE_RUN_START,
        f"input={args.input_source} downstream={resolve_downstream_mode(args)}",
    )

    capture_dir = _default_capture_dir(args, base_path)
    capture_dir.mkdir(parents=True, exist_ok=True)
    profile_json = (
        Path(args.shape_prior_profile_json)
        if args.shape_prior_profile_json is not None
        else capture_dir / "shape_prior_profile.json"
    )
    main_data_processing_command = build_main_data_processing_command(
        args,
        capture_dir=capture_dir,
        profile_json=profile_json,
        chunk_frame_count=chunk_frame_count,
    )
    main_data_processing_env = os.environ.copy()
    if not main_data_processing_env.get(SAM31_CHECKPOINT_ENV):
        # A caller-provided checkpoint env var wins. Otherwise anchor the
        # configured (possibly relative) YAML path to the repo root so launches
        # from other working directories still find the vendored checkpoint.
        checkpoint_path = Path(DEFAULT_SAM31_CHECKPOINT_PATH).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = REPO_ROOT / checkpoint_path
        main_data_processing_env[SAM31_CHECKPOINT_ENV] = str(checkpoint_path)
    main_data_processing_cuda_visible_devices = (
        resolve_main_data_processing_cuda_visible_devices(args).strip()
    )
    if main_data_processing_cuda_visible_devices:
        main_data_processing_env["CUDA_VISIBLE_DEVICES"] = (
            main_data_processing_cuda_visible_devices
        )
    visualizer_process: subprocess.Popen[bytes] | None = None
    visualizer_started = False
    visualizer_started_manifest: dict[str, object] | None = None
    visualizer_start_wall_s: float | None = None
    visualizer_return_code: int | None = None
    visualizer_left_running = False
    phystwin_launch: PhystwinShenLaunch | None = None
    shape_prior_points_npz = resolve_shape_prior_points_npz(args)

    def _maybe_start_phystwin_shen() -> None:
        """Launch once at shape-prior readiness and enforce live health.

        The warmup completion artifact (points.npz) doubles as the
        "shape-prior GPU is free" signal: the SAM3D stage subprocesses have
        exited by the time it is written. train_online_warp.py then keeps
        waiting for the first committed chunk on its own. Without warmup there
        is nothing to wait for, so the launch happens on the first poll.
        """
        nonlocal phystwin_launch
        if not phystwin_shen_enabled(args):
            return
        if phystwin_launch is not None:
            phystwin_launch.assert_pipeline_output_relay_healthy()
            return_code = phystwin_launch.pipeline_process.poll()
            if return_code not in (None, 0):
                raise PhystwinShenLaunchError(
                    "Phystwin_shen full pipeline exited before the Demo stream "
                    f"finished (return code {return_code})"
                )
            return
        if bool(args.shape_prior_warmup):
            if not shape_prior_points_npz.is_file():
                return
            trigger = "shape_prior_points_ready"
        else:
            trigger = "warmup_disabled_immediate"
        phystwin_launch = launch_phystwin_shen(
            resolve_phystwin_shen_settings(args),
            python_prefix=_python_command_prefix(args.phystwin_shen_conda_env),
            log_dir=base_path / "phystwin_shen",
            trigger=trigger,
            wall_time_origin_s=run_start_monotonic_s,
        )
        print(
            "[demo_v6_1] phystwin_shen started "
            f"({trigger}); viewers: {phystwin_launch.settings.viewer_urls}"
        )
        status.emit(STAGE_DOWNSTREAM_START, f"phystwin_shen ({trigger})")

    def on_chunk_written(manifest: dict[str, object]) -> None:
        """Start downstream consumers exactly once when the first chunk commits."""
        nonlocal visualizer_process
        nonlocal visualizer_started
        nonlocal visualizer_started_manifest
        nonlocal visualizer_start_wall_s
        status.emit(
            STAGE_CHUNK_COMMITTED,
            f"chunk {manifest.get('chunk_index', manifest.get('chunk_name', '?'))}",
            chunk_index=manifest.get("chunk_index"),
            track_status=manifest.get("track_process_status"),
        )
        # Output-only viewing starts after the first committed chunk. The
        # side-by-side visualizer starts immediately after launch so warmup
        # RGB remains visible while the output side waits for chunks.
        if (
            demo_visualizer_enabled(args)
            and not visualizer_uses_side_by_side(args)
            and visualizer_process is None
        ):
            visualizer_process = _start_visualizer(args)
            visualizer_started = True
            visualizer_started_manifest = dict(manifest)
            visualizer_start_wall_s = time.monotonic()
        # Safety net: a chunk can only commit after the shape prior is ready.
        _maybe_start_phystwin_shen()

    surface_points = _load_optional_points(args.surface_points_npy)
    interior_points = _load_optional_points(args.interior_points_npy)
    manifests: list[dict[str, object]] = []
    main_data_processing: subprocess.Popen[bytes] | None = None
    main_data_processing_return_code: int | None = None
    camera_return_before_stop: int | None = None
    phystwin_pipeline_return_code: int | None = None
    camera_stopped = False
    try:
        main_data_processing = subprocess.Popen(
            main_data_processing_command,
            env=main_data_processing_env,
            start_new_session=True,
        )
        if demo_visualizer_enabled(args) and visualizer_uses_side_by_side(args):
            visualizer_process = _start_visualizer(args, capture_dir=capture_dir)
            visualizer_started = True
            visualizer_start_wall_s = time.monotonic()
        # The bridge tails frames.jsonl and publishes fixed-size chunks while
        # the camera subprocess is still running, so fake-live and live share the
        # same realtime chunking path.
        # Offline parity with data_process_track.py:L37-L378 and
        # data_process_sample.py:L250-L352. stream_chunk_data_from_headless_capture
        # materializes those outputs incrementally instead of after the
        # recording has finished.
        manifests = stream_chunk_data_from_headless_capture(
            capture_dir,
            base_path=base_path,
            case_prefix=str(args.case_prefix),
            chunk_frame_count=chunk_frame_count,
            fps=int(round(float(args.replay_fps))),
            max_chunks=args.max_chunks,
            capture_finished=lambda: main_data_processing.poll() is not None,
            require_shape_prior=bool(args.shape_prior_warmup),
            shape_prior_wait_timeout_s=float(args.shape_prior_chunk_wait_timeout_s),
            poll_interval_s=float(args.chunk_poll_interval_s),
            surface_points=surface_points,
            interior_points=interior_points,
            on_chunk_written=on_chunk_written,
            before_poll=_maybe_start_phystwin_shen,
            asap_augment=bool(args.asap_augment),
            asap_mesh_path=args.asap_mesh_path,
        )
        if args.max_chunks is not None and len(manifests) >= int(args.max_chunks):
            print(COLLECT_FINISH_BANNER, flush=True)
        camera_return_before_stop = main_data_processing.poll()
        main_data_processing_return_code = _stop_process(main_data_processing)
        camera_stopped = True
        if camera_return_before_stop not in (None, 0):
            raise RuntimeError(
                "main_data_processing exited before the Demo completed "
                f"(return code {camera_return_before_stop})"
            )
        expected_stop_codes = {
            0,
            -signal.SIGTERM,
            -signal.SIGKILL,
            128 + signal.SIGTERM,
            128 + signal.SIGKILL,
        }
        if (
            camera_return_before_stop is None
            and main_data_processing_return_code not in expected_stop_codes
        ):
            raise RuntimeError(
                "main_data_processing failed while the Demo was stopping it "
                f"(return code {main_data_processing_return_code})"
            )
        if args.max_chunks is not None and len(manifests) < int(args.max_chunks):
            raise RuntimeError(
                "Demo stream ended before the requested chunk target: "
                f"wrote {len(manifests)} of {int(args.max_chunks)}"
            )
        if phystwin_shen_enabled(args):
            if phystwin_launch is None:
                raise PhystwinShenLaunchError(
                    "Demo stream finished before Phystwin_shen was launched"
                )
            phystwin_pipeline_return_code = _wait_for_phystwin_launch(phystwin_launch)
    except BaseException as error:
        status.emit(STAGE_FATAL, str(error), ok=False)
        if phystwin_launch is not None:
            _stop_phystwin_launch(phystwin_launch)
        if visualizer_process is not None:
            _stop_process(visualizer_process)
        raise
    finally:
        if main_data_processing is not None and not camera_stopped:
            camera_return_before_stop = main_data_processing.poll()
            main_data_processing_return_code = _stop_process(main_data_processing)
        if visualizer_process is not None:
            visualizer_return_code = visualizer_process.poll()
            visualizer_left_running = visualizer_return_code is None
    runtime_summary = _runtime_chunk_summary(manifests)
    if args.max_chunks is not None and len(manifests) >= int(args.max_chunks):
        stop_reason = "max_chunks_reached"
    elif camera_return_before_stop == 0:
        stop_reason = "main_data_processing_completed"
    elif camera_return_before_stop is None:
        stop_reason = "main_data_processing_stopped_after_stream"
    else:
        stop_reason = "main_data_processing_exited_before_target"
    summary = {
        "demo_version": "demo_v6_1",
        "mode": (
            "full-fake-main-data-processing"
            if str(args.input_source) == "fake-live"
            else "full-live-main-data-processing"
        ),
        "main_data_processing_command": main_data_processing_command,
        "main_data_processing_cuda_visible_devices": (
            main_data_processing_cuda_visible_devices
        ),
        "camera_lossless_max_backlog_seconds": args.camera_lossless_max_backlog_seconds,
        "camera_headless_prepared_only": bool(args.camera_headless_prepared_only),
        "write_input_rgb_timeline": resolve_write_input_rgb_timeline(args),
        "camera_serials": resolve_camera_serials(args),
        "camera_source_replay_fps": resolve_camera_source_replay_fps(args),
        "camera_source_replay_fps_override": (
            None
            if args.camera_source_replay_fps is None
            else float(args.camera_source_replay_fps)
        ),
        "shape_prior_warmup_cuda_visible_devices": (
            resolve_shape_prior_warmup_cuda_visible_devices(args)
        ),
        "shape_prior_controller_name": str(args.shape_prior_controller_name),
        "shape_prior_sam3d_root": (
            None
            if args.shape_prior_sam3d_root is None
            else str(args.shape_prior_sam3d_root)
        ),
        "shape_prior_config": (
            None if args.shape_prior_config is None else str(args.shape_prior_config)
        ),
        "main_data_processing_return_code": main_data_processing_return_code,
        "main_data_processing_return_code_before_stop": camera_return_before_stop,
        "main_data_processing_stop_reason": stop_reason,
        "main_data_processing_capture_dir": str(capture_dir),
        "base_path": str(base_path),
        "case_prefix": str(args.case_prefix),
        "output_format": "online-primary-static-case",
        "online_dir": str(resolve_online_dir(args)),
        "static_data_path": str(resolve_static_data_path(args)),
        "shape_prior_case_root": str(resolve_shape_prior_case_root(args)),
        "shape_prior_points_npz": str(resolve_shape_prior_points_npz(args)),
        "startup_output_cleanup": startup_output_cleanup,
        "chunk_frame_count": int(chunk_frame_count),
        "chunk_poll_interval_s": float(args.chunk_poll_interval_s),
        "max_chunks": args.max_chunks,
        "chunk_count": int(len(manifests)),
        "chunks": manifests,
        "external_shape_prior_points": bool(
            surface_points is not None or interior_points is not None
        ),
        "downstream_mode": resolve_downstream_mode(args),
        "visualizer_layout": resolve_visualizer_layout(args),
        "visualizer_started": visualizer_started,
        "visualizer_start_policy": visualizer_start_policy(args),
        "visualizer_capture_dir": str(capture_dir)
        if visualizer_uses_side_by_side(args)
        else None,
        "visualizer_started_from_chunk": visualizer_started_manifest,
        "visualizer_start_wall_s": visualizer_start_wall_s,
        "visualizer_command": build_visualizer_command(
            args,
            capture_dir=capture_dir if visualizer_uses_side_by_side(args) else None,
        ),
        "visualizer_cuda_visible_devices": resolve_visualizer_cuda_visible_devices(
            args
        ),
        "visualizer_fps": float(args.visualizer_playback_fps),
        "visualizer_object_color_mode": str(args.visualizer_object_color_mode),
        "visualizer_return_code": visualizer_return_code,
        "visualizer_left_running": visualizer_left_running,
        "phystwin_shen_started": phystwin_launch is not None,
        "phystwin_shen_pipeline_return_code": phystwin_pipeline_return_code,
    }
    if phystwin_launch is not None:
        summary.update(phystwin_launch.summary())
    summary.update(runtime_summary)
    summary_path = resolve_run_summary_path(base_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    run_ok = visualizer_return_code in (0, None)
    status.emit(
        STAGE_RUN_FINISHED if run_ok else STAGE_FATAL,
        stop_reason,
        ok=run_ok,
        chunk_count=int(len(manifests)),
    )
    # The live orchestration only reaches this point after the external full
    # pipeline has completed successfully. A camera deliberately stopped after
    # max_chunks may report SIGTERM; an observed pre-stop camera failure was
    # already raised above.
    if demo_visualizer_enabled(args) and not visualizer_started:
        return 1
    if visualizer_return_code not in (0, None):
        return int(visualizer_return_code)
    if phystwin_shen_enabled(args):
        if phystwin_launch is None:
            return 1
        if phystwin_pipeline_return_code != 0:
            return int(phystwin_pipeline_return_code or 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
