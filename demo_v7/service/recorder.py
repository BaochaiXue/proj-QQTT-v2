"""Fake-live case recorder: tee live FramePackets into a data_collect case.

The output directory is a valid ``data_collect/<case>`` recording — the exact
file set demo_v6_2's fake-live reader (``RecordedRgbdFrameSource``) consumes:

    color/0/<step>.png            uint8 BGR via cv2.imwrite
    depth/0/<step>.npy            uint16, native depth units, 0 = invalid
    ir_left/0/<step>.png          (only when the packets carry IR frames)
    ir_right/0/<step>.png
    metadata.json                 qqtt_recording_v2 keys the reader parses:
                                  streams_present, recording{"0":{step:ts}},
                                  K_color(+intrinsics dup), WH, fps,
                                  depth_scale_m_per_unit, serial_numbers,
                                  and the IR calibration keys when present.

Camera→world extrinsics deliberately are NOT part of the contract (replay
takes them from ``--table-calibrate`` like any live run); the repo-root
``calibrate.pkl`` + sidecar are still copied in at close for the offline
export scripts, mirroring ``record_data.py``.

Threading: ``submit()`` is called on the acquisition/publish path and must
NEVER block — it stamps the capture-side timestamp, does a ``put_nowait``
into a bounded queue and counts drops (the reader paces by per-frame
timestamps, so a dropped frame is just a small time gap, same as the
original recorder's dropped-frame gaps). Timestamps come from the producer,
not the disk worker, so a backlog drain cannot compress replay pacing. One
daemon worker thread does all disk IO and rewrites ``metadata.json`` every
``_META_FLUSH_EVERY`` frames, so a killed process degrades to a truncated
but replayable case instead of losing the recording. FramePacket arrays are
owned by the packet (the acquisition loop copies per frame), so no
defensive copies here.
"""

from __future__ import annotations

import json
import queue
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from demo_v7.runtime.mdp.packets import FramePacket

# ~1.7 MB per RGB-D frame reference; 128 bounds worst-case retention while a
# slow disk catches up. PNG+npy writes measure ~10-15 ms/frame on this box —
# comfortably under the 33 ms frame budget, so drops mean real disk trouble.
_QUEUE_MAX_FRAMES = 128
_CLOSE_DRAIN_TIMEOUT_S = 30.0
_META_FLUSH_EVERY = 100
_SCHEMA_VERSION = "qqtt_recording_v2"


class FakeLiveCaseRecorder:
    """Write every submitted FramePacket into a fake-live-replayable case."""

    def __init__(self, case_dir: Path) -> None:
        self.case_dir = Path(case_dir)
        if self.case_dir.exists():
            if not self.case_dir.is_dir():
                raise FileExistsError(
                    f"record dir is an existing file: {self.case_dir}"
                )
            if any(self.case_dir.iterdir()):
                raise FileExistsError(
                    f"record dir is not empty: {self.case_dir} — refusing "
                    "to mix recordings"
                )
        (self.case_dir / "color" / "0").mkdir(parents=True, exist_ok=True)
        (self.case_dir / "depth" / "0").mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[tuple[FramePacket, float] | None] = queue.Queue(
            maxsize=_QUEUE_MAX_FRAMES
        )
        self._timestamps: dict[str, float] = {}
        self._step = 0
        self.dropped = 0
        self.written = 0
        self._error_repr: str | None = None
        self._closed = False
        self._meta_lock = threading.Lock()
        self._first_packet: FramePacket | None = None
        self._has_ir = False
        self.serial: str = "demo-v7-live"
        self._worker = threading.Thread(
            target=self._run, name="v7-case-recorder", daemon=True
        )
        self._worker.start()

    # ------------------------------------------------------------------
    # Producer side (acquisition/publish path — never blocks)
    # ------------------------------------------------------------------
    def submit(self, packet: FramePacket) -> None:
        """Enqueue one packet for recording; drops (and counts) on backlog."""
        if self._closed or self._error_repr is not None:
            return
        if packet.color_bgr is None or packet.depth_u16 is None:
            # Color-only preview stubs (warmup gate pump) are not recordable
            # RGB-D steps; the reader requires depth for every step.
            return
        try:
            self._queue.put_nowait((packet, time.time()))
        except queue.Full:
            self.dropped += 1

    # ------------------------------------------------------------------
    # Worker side
    # ------------------------------------------------------------------
    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                self._write_frame(*item)
            except BaseException as exc:  # noqa: BLE001 — latch, keep demo alive
                if self._error_repr is None:
                    self._error_repr = repr(exc)
                    print(
                        f"[recorder] write failed, recording stops: {exc!r}",
                        flush=True,
                    )
            finally:
                # Drop the packet reference so a latched error doesn't pin
                # the arrays for the recorder's lifetime.
                item = None

    def _write_frame(self, packet: FramePacket, captured_at_s: float) -> None:
        if self._error_repr is not None:
            return
        if self._first_packet is None:
            with self._meta_lock:
                self._first_packet = packet
                self._has_ir = (
                    packet.ir_left_u8 is not None
                    and packet.ir_right_u8 is not None
                )
                if self._has_ir:
                    (self.case_dir / "ir_left" / "0").mkdir(
                        parents=True, exist_ok=True
                    )
                    (self.case_dir / "ir_right" / "0").mkdir(
                        parents=True, exist_ok=True
                    )
        step = str(self._step)
        ok = cv2.imwrite(
            str(self.case_dir / "color" / "0" / f"{step}.png"), packet.color_bgr
        )
        if not ok:
            raise IOError(f"cv2.imwrite failed for color step {step}")
        np.save(
            self.case_dir / "depth" / "0" / f"{step}.npy",
            np.ascontiguousarray(packet.depth_u16, dtype=np.uint16),
        )
        if self._has_ir and packet.ir_left_u8 is not None and packet.ir_right_u8 is not None:
            cv2.imwrite(
                str(self.case_dir / "ir_left" / "0" / f"{step}.png"),
                packet.ir_left_u8,
            )
            cv2.imwrite(
                str(self.case_dir / "ir_right" / "0" / f"{step}.png"),
                packet.ir_right_u8,
            )
        self._timestamps[step] = captured_at_s
        self._step += 1
        self.written += 1
        # Crash consistency: keep metadata.json fresh enough that a killed
        # process (SIGTERM skips finally blocks) leaves a truncated but
        # replayable case rather than losing the whole recording.
        if self.written % _META_FLUSH_EVERY == 0:
            self._write_metadata()

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------
    def close(self) -> dict:
        """Drain, write metadata.json, return an honest summary dict."""
        if self._closed:
            return self._summary()
        self._closed = True
        try:
            self._queue.put(None, timeout=_CLOSE_DRAIN_TIMEOUT_S)
        except queue.Full:
            pass  # worker is stuck/dead; join below is bounded either way
        self._worker.join(timeout=_CLOSE_DRAIN_TIMEOUT_S)
        if self.written > 0 and self._first_packet is not None:
            self._write_metadata()
            self._copy_repo_calibration()
        else:
            self._remove_empty_scaffolding()
        print(
            f"[recorder] closed: {self.written} frames -> {self.case_dir} "
            f"(dropped {self.dropped}"
            + (f", error {self._error_repr}" if self._error_repr else "")
            + ")",
            flush=True,
        )
        return self._summary()

    def _remove_empty_scaffolding(self) -> None:
        """A 0-frame run must not poison the target dir for the next run."""
        for sub in ("color/0", "color", "depth/0", "depth",
                    "ir_left/0", "ir_left", "ir_right/0", "ir_right"):
            try:
                (self.case_dir / sub).rmdir()  # rmdir refuses non-empty dirs
            except OSError:
                pass
        try:
            self.case_dir.rmdir()
        except OSError:
            pass

    def _summary(self) -> dict:
        return {
            "case_dir": str(self.case_dir),
            "frames_written": int(self.written),
            "frames_dropped": int(self.dropped),
            "error": self._error_repr,
        }

    def _measured_fps(self) -> float:
        ts = sorted(self._timestamps.values())
        if len(ts) < 2 or ts[-1] <= ts[0]:
            return 30.0
        return round((len(ts) - 1) / (ts[-1] - ts[0]), 3)

    def _write_metadata(self) -> None:
        packet = self._first_packet
        assert packet is not None
        height, width = packet.color_bgr.shape[:2]
        streams = ["color", "depth"] + (
            ["ir_left", "ir_right"] if self._has_ir else []
        )
        k_color = np.asarray(packet.k_color, dtype=float).reshape(3, 3).tolist()
        metadata: dict = {
            "schema_version": _SCHEMA_VERSION,
            "recorded_by": "demo_v7.service.recorder",
            "capture_mode": "demo_v7_live_tee",
            "serial_numbers": [str(self.serial)],
            "logical_camera_names": ["cam0"],
            "streams_present": streams,
            "fps": self._measured_fps(),
            "WH": [int(width), int(height)],
            "K_color": [k_color],
            "intrinsics": [k_color],
            "depth_scale_m_per_unit": [float(packet.depth_scale_m_per_unit)],
            "depth_encoding": "uint16_meters_scaled_invalid_zero",
            "alignment_target": "color",
            "depth_coordinate_frame": "color",
            "recording": {"0": dict(sorted(
                self._timestamps.items(), key=lambda kv: int(kv[0])
            ))},
        }
        if self._has_ir:
            if packet.k_ir_left is not None:
                k_ir = np.asarray(packet.k_ir_left, dtype=float).reshape(3, 3)
                metadata["K_ir_left"] = [k_ir.tolist()]
                metadata["K_ir_right"] = [k_ir.tolist()]
            if packet.t_ir_left_to_color is not None:
                metadata["T_ir_left_to_color"] = [
                    np.asarray(packet.t_ir_left_to_color, dtype=float)
                    .reshape(4, 4)
                    .tolist()
                ]
            if packet.ir_baseline_m is not None:
                metadata["ir_baseline_m"] = [float(packet.ir_baseline_m)]
        # Atomic replace: the periodic mid-run flush must never leave a
        # half-written metadata.json for the reader to choke on.
        tmp_path = self.case_dir / ".metadata.json.tmp"
        with open(tmp_path, "w") as handle:
            json.dump(metadata, handle, indent=1)
        tmp_path.replace(self.case_dir / "metadata.json")

    def _copy_repo_calibration(self) -> None:
        """Mirror record_data.py: carry the repo calibration for exporters.

        The fake-live reader ignores these (extrinsics come from
        ``--table-calibrate`` at replay time); copying keeps the case usable
        by the offline export scripts. table_calibrate.pkl is also
        snapshotted: it is the c2w the runtime actually applied while this
        case was recorded, so a later replay can be checked against it if
        the camera gets recalibrated in between. Best-effort only.
        """
        repo_root = Path(__file__).resolve().parents[2]
        for name in (
            "calibrate.pkl",
            "calibrate_metadata.json",
            "table_calibrate.pkl",
            "table_calibrate_metadata.json",
        ):
            src = repo_root / name
            if src.is_file():
                try:
                    shutil.copyfile(src, self.case_dir / name)
                except OSError as exc:
                    print(f"[recorder] calibration copy skipped: {exc}", flush=True)
