"""Headless camera worker for the visual calibration dialog.

Runs as a QProcess child of ``CalibrationDialog``: owns the RealSense for
its whole lifetime (an in-process reopen of a stopped RealSense reliably
never delivers a first frame — process death is the only full release, and
it also matches the repo rule that GUIs never open cameras in-process),
streams per-frame strict-estimation results as JSON lines on stdout, and
saves the CLI-identical calibration file set when the parent sends CAPTURE.

Threading contract (hard-won): ``CameraSystem`` is fork-based
multiprocessing (``SingleRealsense(mp.Process)`` + ``SharedMemoryManager``);
forking a multi-threaded parent deadlocks the children probabilistically,
so NO Python thread may exist until the camera is fully up. Camera open
therefore runs threadless, with ``camera_start_timeout_s`` turning a busy
or wedged device into a raised error instead of an infinite hang; only
after ``camera_ready`` does the stdin-command thread start.

Protocol (line-oriented):
    stdout -> {"type":"status","stage":"starting"|"camera_ready"}
              {"type":"frame","seq":N,"ok":bool,"message":str,
               "corner_count":int,"corner_fraction":f,
               "reprojection_error_px":f,"preview_path":str}
              {"type":"saved","output":str}
              {"type":"fatal","message":str}
    stdin  <- "CAPTURE\n" save the latest ok frame and exit 0
              "QUIT\n"    exit 0 without saving
Parent death is detected as stdin EOF; both exit paths fall back to a
SIGKILL of the worker's own process group (camera children included — an
orphaned camera child starves every later open of the device).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from demo_v7.gui.calibrate_core import (  # noqa: E402
    CALIBRATE_FPS,
    CALIBRATE_HEIGHT,
    CALIBRATE_WIDTH,
    estimate_frame,
    save_estimate,
)

_CAMERA_START_TIMEOUT_S = 45.0


def _emit(payload: dict) -> None:
    print(json.dumps(payload), flush=True)


def _kill_own_process_group_soon(delay_s: float) -> None:
    """Fallback for a main thread stuck in a camera call that never returns.

    SIGKILL to our whole process group (``main`` makes us its leader) takes
    the fork-children down with us. The graceful path — stop the camera,
    return — is always tried first; this thread simply dies with the
    process when that succeeds.
    """
    import signal  # noqa: PLC0415

    def _force() -> None:
        time.sleep(delay_s)
        try:
            os.killpg(os.getpgid(0), signal.SIGKILL)
        except OSError:
            os._exit(1)

    threading.Thread(target=_force, daemon=True).start()


class _StdinCommands:
    """Latest command from the parent, read on a daemon thread.

    Must only be constructed AFTER the camera is up — see the module
    docstring's threading contract.
    """

    def __init__(self) -> None:
        self.capture = threading.Event()
        self.quit = threading.Event()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in sys.stdin:
            command = line.strip().upper()
            if command == "CAPTURE":
                self.capture.set()
            elif command == "QUIT":
                self.quit.set()
                _kill_own_process_group_soon(15.0)  # graceful first
                return
        # Parent closed stdin (crashed / was killed): nobody is reading our
        # events any more — release the camera and go.
        self.quit.set()
        _kill_own_process_group_soon(15.0)


def _open_camera(serial: str | None):
    """Open + prove streaming; one retry with a settle for USB hiccups.

    Threadless by contract; ``camera_start_timeout_s`` bounds a busy or
    wedged device to a raised error instead of an infinite constructor
    hang.
    """
    from demo_v7.runtime.orchestration.main_config import (
        DEFAULT_CAMERA_COLOR_EXPOSURE,
        DEFAULT_CAMERA_COLOR_GAIN,
    )
    from qqtt.env import CameraSystem

    last_error: Exception | None = None
    for attempt in range(2):
        camera_system = None
        try:
            camera_system = CameraSystem(
                WH=[CALIBRATE_WIDTH, CALIBRATE_HEIGHT],
                fps=CALIBRATE_FPS,
                num_cam=1,
                serial_numbers=[serial] if serial else None,
                capture_mode="color",
                # Match the RUNTIME camera settings, not qqtt's capture
                # defaults (owner report 2026-08-07: the preview rendered
                # salmon-pink and blown out). Exposure/gain come from the
                # same config the camera service is launched with (45/45
                # vs qqtt's 70/60 — today's lab lighting overexposes at
                # the latter, clipping the board to pink-white), and white
                # balance is AUTO (None) exactly like the service, which
                # never touches WB — qqtt's pinned 3800K is what caused
                # the cast. Calibration should see the runtime's colors.
                exposure=DEFAULT_CAMERA_COLOR_EXPOSURE,
                gain=DEFAULT_CAMERA_COLOR_GAIN,
                white_balance=None,
                enable_keyboard_listener=False,
                camera_start_timeout_s=_CAMERA_START_TIMEOUT_S,
            )
            camera_system.get_observation()  # first-frame gate
            return camera_system
        except Exception as exc:  # noqa: BLE001 — retry once, then raise
            last_error = exc
            if camera_system is not None:
                try:
                    camera_system.stop(wait=True)
                except Exception:  # noqa: BLE001
                    pass
            if attempt == 0:
                time.sleep(3.0)
    assert last_error is not None
    raise last_error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--serial", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, default=None)
    parser.add_argument("--preview-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    # Own process group: camera fork-children join it, so the forced-exit
    # fallbacks can SIGKILL them together with us instead of stranding them
    # on the device. (Never kill the GUI's group.)
    try:
        os.setpgrp()
    except OSError:
        pass

    import cv2

    from qqtt.env.camera.calibration_boards import (
        DEFAULT_CALIBRATION_BOARD,
        get_calibration_board_config,
    )

    board_config = get_calibration_board_config(DEFAULT_CALIBRATION_BOARD)
    _emit({"type": "status", "stage": "starting"})
    args.preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = args.preview_dir / "preview.jpg"
    preview_tmp = args.preview_dir / ".preview.tmp.jpg"

    camera_system = None
    try:
        # Threadless until here — see the module docstring.
        camera_system = _open_camera(args.serial)
        _emit({"type": "status", "stage": "camera_ready"})
        commands = _StdinCommands()  # fork done; threads are safe now
        camera_matrix = camera_system.realsense.get_intrinsics()[0]
        import numpy as np

        stream_metadata = list(getattr(camera_system, "stream_metadata", []))
        camera_metadata = stream_metadata[0] if stream_metadata else {}
        coeffs = camera_metadata.get("color_distortion_coeffs")
        dist_coeffs = None
        if coeffs is not None:
            coeffs_array = np.asarray(coeffs, dtype=np.float64).reshape(-1, 1)
            dist_coeffs = coeffs_array if coeffs_array.size else None
        serials = [
            str(s) for s in getattr(camera_system, "serial_numbers", [])
        ] or ([args.serial] if args.serial else ["cam0"])

        seq = 0
        latest_ok = None
        while not commands.quit.is_set():
            loop_started_s = time.monotonic()
            obs = camera_system.get_observation()
            estimate = estimate_frame(
                obs[0]["color"],
                board_config=board_config,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
            )
            estimate.distortion_model = camera_metadata.get(
                "color_distortion_model"
            )
            estimate.serial_numbers = serials
            if estimate.ok:
                latest_ok = estimate
            cv2.imwrite(
                str(preview_tmp),
                estimate.display_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), 85],
            )
            os.replace(preview_tmp, preview_path)  # readers never see a torn file
            _emit(
                {
                    "type": "frame",
                    "seq": seq,
                    "ok": estimate.ok,
                    "message": estimate.message[-300:],
                    "corner_count": estimate.corner_count,
                    "corner_fraction": estimate.corner_fraction,
                    "reprojection_error_px": estimate.reprojection_error_px,
                    "preview_path": str(preview_path),
                }
            )
            seq += 1
            if commands.capture.is_set():
                commands.capture.clear()
                if latest_ok is None:
                    _emit({"type": "frame", "seq": seq, "ok": False,
                           "message": "no accepted frame to capture yet",
                           "corner_count": 0, "corner_fraction": 0.0,
                           "reprojection_error_px": 0.0,
                           "preview_path": str(preview_path)})
                    continue
                save_estimate(
                    latest_ok,
                    board_config=board_config,
                    output_path=args.output,
                    diagnostic_path=args.diagnostic,
                )
                _emit({"type": "saved", "output": str(args.output)})
                return 0
            # get_observation returns the latest buffered frame without
            # pacing — cap the preview loop at ~10 Hz.
            elapsed_s = time.monotonic() - loop_started_s
            if elapsed_s < 0.1:
                time.sleep(0.1 - elapsed_s)
        return 0
    except Exception as exc:  # noqa: BLE001 — one fatal line, nonzero exit
        _emit({"type": "fatal", "message": (str(exc) or repr(exc))[-300:]})
        return 1
    finally:
        if camera_system is not None:
            try:
                camera_system.stop(wait=True)
            except Exception:  # noqa: BLE001 — release must not raise
                pass


if __name__ == "__main__":
    raise SystemExit(main())
