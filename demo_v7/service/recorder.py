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
NEVER block — it does a ``put_nowait`` into a bounded queue and counts drops
(the reader paces by per-frame timestamps, so a dropped frame is just a
small time gap, same as the original recorder's dropped-frame gaps). One
daemon worker thread does all disk IO. FramePacket arrays are owned by the
packet (the acquisition loop copies per frame), so no defensive copies here.
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

from demo_v6_2.mdp.packets import FramePacket

# ~1.7 MB per RGB-D frame reference; 128 bounds worst-case retention while a
# slow disk catches up. PNG+npy writes measure ~10-15 ms/frame on this box —
# comfortably under the 33 ms frame budget, so drops mean real disk trouble.
_QUEUE_MAX_FRAMES = 128
_CLOSE_DRAIN_TIMEOUT_S = 30.0
_SCHEMA_VERSION = "qqtt_recording_v2"


class FakeLiveCaseRecorder:
    """Write every submitted FramePacket into a fake-live-replayable case."""

    def __init__(self, case_dir: Path) -> None:
        self.case_dir = Path(case_dir)
        if self.case_dir.exists() and any(self.case_dir.iterdir()):
            raise FileExistsError(
                f"record dir is not empty: {self.case_dir} — refusing to mix "
                "recordings"
            )
        (self.case_dir / "color" / "0").mkdir(parents=True, exist_ok=True)
        (self.case_dir / "depth" / "0").mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[FramePacket | None] = queue.Queue(
            maxsize=_QUEUE_MAX_FRAMES
        )
        self._timestamps: dict[str, float] = {}
        self._step = 0
        self.dropped = 0
        self.written = 0
        self._error: BaseException | None = None
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
        if self._closed or self._error is not None:
            return
        if packet.color_bgr is None or packet.depth_u16 is None:
            # Color-only preview stubs (warmup gate pump) are not recordable
            # RGB-D steps; the reader requires depth for every step.
            return
        try:
            self._queue.put_nowait(packet)
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
                self._write_frame(item)
            except BaseException as exc:  # noqa: BLE001 — latch, keep demo alive
                if self._error is None:
                    self._error = exc
                    print(
                        f"[recorder] write failed, recording stops: {exc!r}",
                        flush=True,
                    )

    def _write_frame(self, packet: FramePacket) -> None:
        if self._error is not None:
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
        self._timestamps[step] = time.time()
        self._step += 1
        self.written += 1

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
        print(
            f"[recorder] closed: {self.written} frames -> {self.case_dir} "
            f"(dropped {self.dropped}"
            + (f", error {self._error!r}" if self._error else "")
            + ")",
            flush=True,
        )
        return self._summary()

    def _summary(self) -> dict:
        return {
            "case_dir": str(self.case_dir),
            "frames_written": int(self.written),
            "frames_dropped": int(self.dropped),
            "error": repr(self._error) if self._error else None,
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
        with open(self.case_dir / "metadata.json", "w") as handle:
            json.dump(metadata, handle, indent=1)

    def _copy_repo_calibration(self) -> None:
        """Mirror record_data.py: carry the repo calibration for exporters.

        The fake-live reader ignores these (extrinsics come from
        ``--table-calibrate`` at replay time); copying keeps the case usable
        by the offline export scripts. Best-effort only.
        """
        repo_root = Path(__file__).resolve().parents[2]
        for name in ("calibrate.pkl", "calibrate_metadata.json"):
            src = repo_root / name
            if src.is_file():
                try:
                    shutil.copyfile(src, self.case_dir / name)
                except OSError as exc:
                    print(f"[recorder] calibration copy skipped: {exc}", flush=True)
