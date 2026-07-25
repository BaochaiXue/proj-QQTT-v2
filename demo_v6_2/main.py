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
import signal
import subprocess
import sys
import time
from pathlib import Path
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

from demo_v6_2.streaming.session import ChunkStreamSession
from demo_v6_2.main_cli import build_parser
from demo_v6_2.orchestration.main_config import (
    DEFAULT_SAM31_CHECKPOINT_PATH,
    REPO_ROOT,
    SAM31_CHECKPOINT_ENV,
)
from demo_v6_2.orchestration.main_layout import (
    prepare_realtime_output_for_new_run,
    resolve_run_summary_path,
    resolve_shape_prior_points_npz,
)
from demo_v6_2.orchestration.run_config import (
    OrchestratorRunConfig,
    dry_run_contract,
    static_run_contract,
)
from demo_v6_2.main_options import load_optional_points, python_command_prefix
from demo_v6_2.main_subprocess import (
    build_main_data_processing_command,
    build_visualizer_command,
    default_capture_dir,
    start_visualizer,
    stop_process,
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
from demo_v6_2.utils.runtime_summary import runtime_chunk_summary

# ---------------------------------------------------------------------------
# Run summary and entrypoint
# ---------------------------------------------------------------------------


COLLECT_FINISH_BANNER = "##################\ncollect finish\n##################"


def _stop_phystwin_launch(launch: PhystwinShenLaunch) -> int | None:
    """Terminate the saved full-pipeline process group."""
    try:
        return stop_process(
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
    """Run Demo v6.2 live/fake-live orchestration."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    # check parameters
    config = OrchestratorRunConfig.from_args(args)
    if bool(args.dry_run):
        print(json.dumps(dry_run_contract(args, config), indent=2, sort_keys=True))
        return 0

    base_path = Path(args.base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    startup_output_cleanup = prepare_realtime_output_for_new_run(
        base_path,
        legacy_case_prefix=str(args.case_prefix),
    )

    # Live pipeline-status stream (design question 25): the orchestrator, the
    # camera process, and the SAM3D shape-prior stages all append lifecycle
    # events to <base_path>/pipeline_status.jsonl and the visualizer tails it to
    # show what the pipeline is doing right now / whether warm-up failed. Writing
    # is best-effort and never affects the published products.
    run_start_monotonic_s = time.monotonic()
    status = PipelineStatusWriter(base_path, "orchestrator")
    status.emit(
        STAGE_RUN_START,
        f"input={args.input_source} downstream={config.downstream_mode}",
    )  # time record strat

    capture_dir = default_capture_dir(args, base_path)
    capture_dir.mkdir(parents=True, exist_ok=True)
    profile_json = (
        Path(args.shape_prior_profile_json)
        if args.shape_prior_profile_json is not None
        else capture_dir / "shape_prior_profile.json"
    )  # profile files
    main_data_processing_command = build_main_data_processing_command(
        args,
        capture_dir=capture_dir,
        profile_json=profile_json,
    )  # generate the command text
    main_data_processing_env = os.environ.copy()
    from demo_v6_2.perception.sam31_image_segmentation import (  # noqa: PLC0415
        QQTT_SAM31_CHECKPOINT_ENV,
    )

    assert SAM31_CHECKPOINT_ENV == QQTT_SAM31_CHECKPOINT_ENV, (
        "sam31 checkpoint env var name diverged: config/default.yaml "
        "paths.sam31_checkpoint_env vs "
        "demo_v6_2/perception/sam31_image_segmentation.py QQTT_SAM31_CHECKPOINT_ENV"
    )
    # get SAM3.1 checkpoint path
    if not main_data_processing_env.get(SAM31_CHECKPOINT_ENV):
        # A caller-provided checkpoint env var wins. Otherwise anchor the
        # configured (possibly relative) YAML path to the repo root so launches
        # from other working directories still find the vendored checkpoint.
        checkpoint_path = Path(DEFAULT_SAM31_CHECKPOINT_PATH).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = REPO_ROOT / checkpoint_path
        main_data_processing_env[SAM31_CHECKPOINT_ENV] = str(checkpoint_path)
    main_data_processing_env["CUDA_VISIBLE_DEVICES"] = (
        config.main_data_processing_cuda_visible_devices
    )
    visualizer_process: subprocess.Popen[bytes] | None = None
    visualizer_started = False
    visualizer_started_manifest: dict[str, object] | None = None
    visualizer_start_wall_s: float | None = None
    visualizer_return_code: int | None = None
    visualizer_left_running = False
    phystwin_launch: PhystwinShenLaunch | None = None
    shape_prior_points_npz = resolve_shape_prior_points_npz(args)

    # start Phystwin
    def _ensure_phystwin_shen_running() -> None:
        """Launch once at final structure readiness and enforce live health.

        points.npz now means "the chunk-0 unified origin sampling is done":
        it is written by the chunk session right after chunk 0 commits, with
        the final surface/interior points (the shape-prior heavy stages
        exited earlier, at candidates-ready). Without warmup there is nothing
        to wait for, so the launch happens on the first poll.
        """
        nonlocal phystwin_launch
        if not config.phystwin_shen_enabled:
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
        # launch
        phystwin_launch = launch_phystwin_shen(
            config.phystwin_shen_settings,
            python_prefix=python_command_prefix(args.phystwin_shen_conda_env),
            log_dir=base_path / "phystwin_shen",
            trigger=trigger,
            wall_time_origin_s=run_start_monotonic_s,
        )
        print(
            "[demo_v6_1] phystwin_shen started "
            f"({trigger}); viewers: {phystwin_launch.settings.viewer_urls}"
        )
        status.emit(STAGE_DOWNSTREAM_START, f"phystwin_shen ({trigger})")

    # post processing of chunks
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
        # The window output-only viewer starts after the first committed
        # chunk; the web frontend and the side-by-side window start
        # immediately (visualizer_start_policy).
        if (
            config.visualizer_start_policy == "after_first_committed_online_chunk"
            and visualizer_process is None
        ):
            visualizer_process = start_visualizer(args)
            visualizer_started = True
            visualizer_started_manifest = dict(manifest)
            visualizer_start_wall_s = time.monotonic()
        # Safety net: a chunk can only commit after the shape prior is ready.
        _ensure_phystwin_shen_running()

    surface_points = load_optional_points(args.surface_points_npy)
    interior_points = load_optional_points(args.interior_points_npy)
    manifests: list[dict[str, object]] = []
    main_data_processing: subprocess.Popen[bytes] | None = None
    main_data_processing_return_code: int | None = None
    main_data_processing_return_code_before_stop: int | None = None
    phystwin_pipeline_return_code: int | None = None
    main_data_processing_stopped = False
    try:
        main_data_processing = subprocess.Popen(
            main_data_processing_command,
            env=main_data_processing_env,
            start_new_session=True,
        )
        if config.visualizer_start_policy == "immediate_after_camera_start":
            visualizer_process = start_visualizer(args, capture_dir=capture_dir)
            visualizer_started = True
            visualizer_start_wall_s = time.monotonic()
        # The bridge tails frames.jsonl and publishes fixed-size chunks while
        # the camera subprocess is still running, so fake-live and live share the
        # same realtime chunking path.
        # Offline parity with data_process_track.py:L37-L378 and
        # data_process_sample.py:L250-L352. ChunkStreamSession materializes
        # those outputs incrementally instead of after the recording has
        # finished.
        chunk_stream = ChunkStreamSession(
            capture_dir,
            base_path=base_path,
            case_prefix=str(args.case_prefix),
            chunk_size=config.chunk_frame_count,
            volume_sample_size_m=config.volume_sample_size_m,
            fps=int(round(float(args.replay_fps))),
            max_chunks=args.max_chunks,
            capture_finished=lambda: main_data_processing.poll() is not None,
            require_shape_prior=bool(args.shape_prior_warmup),
            points_npz=shape_prior_points_npz,
            shape_prior_wait_timeout_s=float(args.shape_prior_chunk_wait_timeout_s),
            poll_interval_s=float(args.chunk_poll_interval_s),
            surface_points=surface_points,
            interior_points=interior_points,
            on_chunk_written=on_chunk_written,
            before_poll=_ensure_phystwin_shen_running,
            asap_augment=bool(args.asap_augment),
            asap_mesh_path=args.asap_mesh_path,
        )
        manifests = chunk_stream.run()
        if args.max_chunks is not None and len(manifests) >= int(args.max_chunks):
            print(COLLECT_FINISH_BANNER, flush=True)
        main_data_processing_return_code_before_stop = main_data_processing.poll()
        main_data_processing_return_code = stop_process(main_data_processing)
        main_data_processing_stopped = True
        if main_data_processing_return_code_before_stop not in (None, 0):
            raise RuntimeError(
                "main_data_processing exited before the Demo completed "
                f"(return code {main_data_processing_return_code_before_stop})"
            )
        expected_stop_codes = {
            0,
            -signal.SIGTERM,
            -signal.SIGKILL,
            128 + signal.SIGTERM,
            128 + signal.SIGKILL,
        }
        if (
            main_data_processing_return_code_before_stop is None
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
        if config.phystwin_shen_enabled:
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
            stop_process(visualizer_process)
        raise
    finally:
        if main_data_processing is not None and not main_data_processing_stopped:
            main_data_processing_return_code_before_stop = main_data_processing.poll()
            main_data_processing_return_code = stop_process(main_data_processing)
        if visualizer_process is not None:
            visualizer_return_code = visualizer_process.poll()
            visualizer_left_running = visualizer_return_code is None
    runtime_summary = runtime_chunk_summary(manifests)
    if args.max_chunks is not None and len(manifests) >= int(args.max_chunks):
        stop_reason = "max_chunks_reached"
    elif main_data_processing_return_code_before_stop == 0:
        stop_reason = "main_data_processing_completed"
    elif main_data_processing_return_code_before_stop is None:
        stop_reason = "main_data_processing_stopped_after_stream"
    else:
        stop_reason = "main_data_processing_exited_before_target"
    summary = {
        **static_run_contract(args, config),
        "mode": (
            "full-fake-main-data-processing"
            if str(args.input_source) == "fake-live"
            else "full-live-main-data-processing"
        ),
        "main_data_processing_command": main_data_processing_command,
        "main_data_processing_return_code": main_data_processing_return_code,
        "main_data_processing_return_code_before_stop": (
            main_data_processing_return_code_before_stop
        ),
        "main_data_processing_stop_reason": stop_reason,
        "startup_output_cleanup": startup_output_cleanup,
        "chunk_count": int(len(manifests)),
        "chunks": manifests,
        "external_shape_prior_points": bool(
            surface_points is not None or interior_points is not None
        ),
        "visualizer_started": visualizer_started,
        "visualizer_capture_dir": str(capture_dir) if config.side_by_side else None,
        "visualizer_started_from_chunk": visualizer_started_manifest,
        "visualizer_start_wall_s": visualizer_start_wall_s,
        "visualizer_command": build_visualizer_command(
            args,
            capture_dir=capture_dir if config.side_by_side else None,
        ),
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
    if config.demo_visualizer_enabled and not visualizer_started:
        return 1
    if visualizer_return_code not in (0, None):
        return int(visualizer_return_code)
    if config.phystwin_shen_enabled:
        if phystwin_launch is None:
            return 1
        if phystwin_pipeline_return_code != 0:
            return int(phystwin_pipeline_return_code or 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
