"""demo_v7 GUI <-> camera-service wire protocol (single source of truth).

Two Unix-domain sockets under the run's socket dir:

- ``control.sock`` — JSON lines, bidirectional. GUI sends command objects,
  the service replies with event objects (plus unsolicited events). Every
  message is one JSON object per line, UTF-8.
- ``frames.sock`` — length-prefixed binary frames, service -> GUI only.
  Header = one JSON line (channel, seq, meta) then ``payload_len`` bytes of
  JPEG. The GUI keeps only the newest frame per channel (latest-wins);
  the service caps encode rate per channel.

The service owns the sockets (binds/listens); the GUI connects. Exactly one
GUI client at a time. Both sides import THIS module — no other place may
define command/event/channel names.
"""

from __future__ import annotations

from dataclasses import dataclass

PROTOCOL_VERSION = 1

CONTROL_SOCKET_NAME = "control.sock"
FRAMES_SOCKET_NAME = "frames.sock"

# ---------------------------------------------------------------------------
# Service states (event: state_changed, field: state)
# ---------------------------------------------------------------------------
STATE_STARTING = "starting"  # process up, source/preloads initializing
STATE_PREVIEW = "preview"  # live frames flowing; frame-0 not captured
STATE_FRAME0_PENDING = "frame0_pending"  # candidate frozen, awaiting confirm
STATE_WARMUP = "warmup"  # frame-0 pipeline running (no tracking)
STATE_REVIEW = "review"  # warmup done; artifacts ready for inspection
STATE_REPOSITION = "reposition"  # mask overlay preview; awaiting start
STATE_FORMAL = "formal"  # full v6.2 lossless pipeline running
STATE_FINISHED = "finished"  # formal ended (stop / replay exhausted)
STATE_FATAL = "fatal"  # unrecoverable error; see event: error

# ---------------------------------------------------------------------------
# Commands (GUI -> service). {"cmd": <name>, ...fields}
# Service replies {"event": "ack", "cmd": <name>, "ok": bool, "error": str?}
# before any state event the command causes.
# ---------------------------------------------------------------------------
CMD_HELLO = "hello"  # {} -> ack carries {"version", "state", "source_kind"}
CMD_CAPTURE_FRAME0 = "capture_frame0"  # PREVIEW -> FRAME0_PENDING
CMD_RETAKE_FRAME0 = "retake_frame0"  # FRAME0_PENDING -> PREVIEW
CMD_CONFIRM_FRAME0 = "confirm_frame0"  # FRAME0_PENDING -> WARMUP
CMD_ENTER_REVIEW = "enter_review"  # WARMUP(done) -> REVIEW (no-op if auto)
CMD_BEGIN_REPOSITION = "begin_reposition"  # REVIEW -> REPOSITION
CMD_START_FORMAL = "start_formal"  # REPOSITION -> FORMAL
CMD_STOP_FORMAL = "stop_formal"  # FORMAL -> FINISHED
CMD_SHUTDOWN = "shutdown"  # any -> exit(0)
CMD_REGEN_GAUSSIAN = "regen_gaussian"  # REVIEW: {"seed"?} re-roll the splats

# ---------------------------------------------------------------------------
# Events (service -> GUI). {"event": <name>, ...fields}
# ---------------------------------------------------------------------------
EVT_ACK = "ack"
EVT_STATE = "state_changed"  # {"state", "detail"?}
EVT_PROGRESS = "progress"  # {"stage", "detail", "ok": bool, "elapsed_ms"?}
#   stage strings mirror demo_v6_2 pipeline_status stages plus:
#   "preload", "sam31_masks", "shape_prior_submit", "shape_prior_ready".
EVT_ARTIFACTS = "artifacts_ready"
#   {"kind": one of ARTIFACT_KINDS, "paths": {name: abs_path}} — emitted as
#   review material lands on disk (masks pngs, turntable mp4, overlays...).
EVT_ERROR = "error"  # {"where", "message"} (service stays up unless fatal)
EVT_REPLAY_EXHAUSTED = "replay_exhausted"  # fake source ran out (FORMAL only)
EVT_FORMAL_STATS = "formal_stats"  # periodic {"seq","fps":{...},"latency_ms"}

ARTIFACT_KIND_FRAME0 = "frame0"  # frame0 rgb png + depth npy preview png
ARTIFACT_KIND_MASKS = "masks"  # object/hand_a/hand_b pngs + overlay png
ARTIFACT_KIND_SHAPE_PRIOR = "shape_prior"  # turntable mp4, mesh glb, renders
ARTIFACT_KIND_ALIGNMENT = "alignment"  # aligned overlay png(s)
ARTIFACT_KIND_GAUSSIAN = "gaussian"  # triposplat turntable/overlay/ply paths

# ---------------------------------------------------------------------------
# Frame channels (frames.sock header field: channel)
# ---------------------------------------------------------------------------
CH_RGB = "rgb"  # live color (all states)
CH_DEPTH = "depth"  # live depth colormap (all states)
CH_OVERLAY = "overlay"  # REPOSITION: rgb + 50% mask tint (service-side)
CH_COMPOSITE = "composite"  # FORMAL: v6.2 live-viewer pair composite
CH_GAUSSIAN = "gaussian"  # FORMAL: tracked-motion gaussian over live rgb
FRAME_CHANNELS = (CH_RGB, CH_DEPTH, CH_OVERLAY, CH_COMPOSITE, CH_GAUSSIAN)

# Per-channel encode caps (service side), Hz.
CHANNEL_MAX_HZ = {
    CH_RGB: 20.0,
    CH_DEPTH: 10.0,
    CH_OVERLAY: 20.0,
    CH_COMPOSITE: 15.0,
    CH_GAUSSIAN: 10.0,
}
JPEG_QUALITY = 85


@dataclass(frozen=True)
class FrameHeader:
    """Parsed frames.sock header line."""

    channel: str
    seq: int
    payload_len: int
    width: int
    height: int
    # perf_counter timestamp on the service side (drift-free deltas only).
    t_service_s: float

    def to_json_obj(self) -> dict:
        return {
            "channel": self.channel,
            "seq": int(self.seq),
            "payload_len": int(self.payload_len),
            "width": int(self.width),
            "height": int(self.height),
            "t_service_s": float(self.t_service_s),
        }

    @staticmethod
    def from_json_obj(obj: dict) -> "FrameHeader":
        return FrameHeader(
            channel=str(obj["channel"]),
            seq=int(obj["seq"]),
            payload_len=int(obj["payload_len"]),
            width=int(obj["width"]),
            height=int(obj["height"]),
            t_service_s=float(obj["t_service_s"]),
        )
