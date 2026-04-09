from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = next(
    (p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents] if (p / ".git").exists()),
    Path(__file__).resolve().parents[3],
)
DEFAULT_PROBE_RESULTS_JSON = (PROJECT_ROOT / "docs" / "generated" / "d455_stream_probe_results.json").resolve()
DEFAULT_PROBE_RESULTS_MD = (PROJECT_ROOT / "docs" / "generated" / "d455_stream_probe_results.md").resolve()

CAPTURE_PREFLIGHT_POLICY = {
    "rgbd": {
        "probe_stream_set": None,
        "unsupported_behavior": "allow",
        "operator_status": "supported",
        "policy_label": "supported_without_probe",
    },
    "stereo_ir": {
        "probe_stream_set": "rgb_ir_pair",
        "unsupported_behavior": "warn",
        "operator_status": "experimental_warning",
        "policy_label": "warn_if_probe_fails",
    },
    "both_eval": {
        "probe_stream_set": "rgbd_ir_pair",
        "unsupported_behavior": "block",
        "operator_status": "blocked",
        "policy_label": "block_if_probe_fails",
    },
}


@dataclass(slots=True)
class CapturePreflightDecision:
    capture_mode: str
    serials: list[str]
    width: int
    height: int
    fps: int
    emitter: str
    topology_type: str | None
    stream_set: str | None
    probe_support: bool | None
    policy_label: str
    unsupported_behavior: str
    operator_status: str
    allowed_to_record: bool
    requires_probe: bool
    reason: str
    probe_results_json: str
    probe_results_md: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "capture_mode": self.capture_mode,
            "serials": list(self.serials),
            "width": int(self.width),
            "height": int(self.height),
            "fps": int(self.fps),
            "emitter": self.emitter,
            "topology_type": self.topology_type,
            "stream_set": self.stream_set,
            "probe_support": self.probe_support,
            "policy_label": self.policy_label,
            "unsupported_behavior": self.unsupported_behavior,
            "operator_status": self.operator_status,
            "allowed_to_record": bool(self.allowed_to_record),
            "requires_probe": bool(self.requires_probe),
            "reason": self.reason,
            "probe_results_json": self.probe_results_json,
            "probe_results_md": self.probe_results_md,
        }


def _topology_type_for_serials(serials: list[str] | None) -> str | None:
    if serials is None:
        return None
    if len(serials) == 1:
        return "single"
    if len(serials) == 3:
        return "three_camera"
    return None


def lookup_probe_support(
    *,
    capture_mode: str,
    serials: list[str],
    width: int,
    height: int,
    fps: int,
    emitter: str,
    probe_results_path: str | Path = DEFAULT_PROBE_RESULTS_JSON,
) -> bool | None:
    probe_path = Path(probe_results_path).resolve()
    if not probe_path.exists():
        return None
    data = json.loads(probe_path.read_text(encoding="utf-8"))
    topology_type = _topology_type_for_serials(serials)
    stream_set = CAPTURE_PREFLIGHT_POLICY.get(capture_mode, {}).get("probe_stream_set")
    if topology_type is None or stream_set is None:
        return None

    for case in data.get("cases", []):
        if (
            case.get("topology_type") == topology_type
            and case.get("stream_set") == stream_set
            and case.get("serials") == serials
            and case.get("width") == int(width)
            and case.get("height") == int(height)
            and case.get("fps") == int(fps)
            and case.get("emitter_request") == emitter
        ):
            return bool(case.get("success"))
    return None


def evaluate_capture_preflight(
    *,
    capture_mode: str,
    serials: list[str] | None,
    width: int,
    height: int,
    fps: int,
    emitter: str,
    probe_results_path: str | Path = DEFAULT_PROBE_RESULTS_JSON,
    probe_results_md_path: str | Path = DEFAULT_PROBE_RESULTS_MD,
) -> CapturePreflightDecision:
    if capture_mode not in CAPTURE_PREFLIGHT_POLICY:
        raise ValueError(f"Unsupported capture_mode: {capture_mode}")

    policy = CAPTURE_PREFLIGHT_POLICY[capture_mode]
    serial_list = [] if serials is None else list(serials)
    topology_type = _topology_type_for_serials(serial_list) if serials is not None else None
    stream_set = policy["probe_stream_set"]
    requires_probe = stream_set is not None

    if not requires_probe:
        return CapturePreflightDecision(
            capture_mode=capture_mode,
            serials=serial_list,
            width=width,
            height=height,
            fps=fps,
            emitter=emitter,
            topology_type=topology_type,
            stream_set=stream_set,
            probe_support=None,
            policy_label=str(policy["policy_label"]),
            unsupported_behavior=str(policy["unsupported_behavior"]),
            operator_status=str(policy["operator_status"]),
            allowed_to_record=True,
            requires_probe=False,
            reason="rgbd capture does not require the D455 IR-pair stream probe gate.",
            probe_results_json=str(Path(probe_results_path).resolve()),
            probe_results_md=str(Path(probe_results_md_path).resolve()),
        )

    if serials is None:
        return CapturePreflightDecision(
            capture_mode=capture_mode,
            serials=[],
            width=width,
            height=height,
            fps=fps,
            emitter=emitter,
            topology_type=None,
            stream_set=stream_set,
            probe_support=None,
            policy_label=str(policy["policy_label"]),
            unsupported_behavior=str(policy["unsupported_behavior"]),
            operator_status="pending_serial_resolution",
            allowed_to_record=True,
            requires_probe=True,
            reason="Serials are not resolved yet; final probe policy will be applied after camera enumeration.",
            probe_results_json=str(Path(probe_results_path).resolve()),
            probe_results_md=str(Path(probe_results_md_path).resolve()),
        )

    probe_support = lookup_probe_support(
        capture_mode=capture_mode,
        serials=serial_list,
        width=width,
        height=height,
        fps=fps,
        emitter=emitter,
        probe_results_path=probe_results_path,
    )

    if probe_support is True:
        return CapturePreflightDecision(
            capture_mode=capture_mode,
            serials=serial_list,
            width=width,
            height=height,
            fps=fps,
            emitter=emitter,
            topology_type=topology_type,
            stream_set=stream_set,
            probe_support=True,
            policy_label=str(policy["policy_label"]),
            unsupported_behavior=str(policy["unsupported_behavior"]),
            operator_status="supported",
            allowed_to_record=True,
            requires_probe=True,
            reason="The latest D455 stream probe marked this capture profile as supported.",
            probe_results_json=str(Path(probe_results_path).resolve()),
            probe_results_md=str(Path(probe_results_md_path).resolve()),
        )

    if probe_support is False:
        unsupported_behavior = str(policy["unsupported_behavior"])
        if unsupported_behavior == "block":
            operator_status = "blocked"
            allowed_to_record = False
            reason = "The latest D455 stream probe marked this profile unsupported, and current repo policy blocks it."
        else:
            operator_status = "experimental_warning"
            allowed_to_record = True
            reason = "The latest D455 stream probe marked this profile unstable, but current repo policy still allows recording with a warning."
        return CapturePreflightDecision(
            capture_mode=capture_mode,
            serials=serial_list,
            width=width,
            height=height,
            fps=fps,
            emitter=emitter,
            topology_type=topology_type,
            stream_set=stream_set,
            probe_support=False,
            policy_label=str(policy["policy_label"]),
            unsupported_behavior=unsupported_behavior,
            operator_status=operator_status,
            allowed_to_record=allowed_to_record,
            requires_probe=True,
            reason=reason,
            probe_results_json=str(Path(probe_results_path).resolve()),
            probe_results_md=str(Path(probe_results_md_path).resolve()),
        )

    return CapturePreflightDecision(
        capture_mode=capture_mode,
        serials=serial_list,
        width=width,
        height=height,
        fps=fps,
        emitter=emitter,
        topology_type=topology_type,
        stream_set=stream_set,
        probe_support=None,
        policy_label=str(policy["policy_label"]),
        unsupported_behavior=str(policy["unsupported_behavior"]),
        operator_status="probe_unknown",
        allowed_to_record=True,
        requires_probe=True,
        reason="No matching probe result was found for this exact profile; recording may proceed according to current repo policy, but the operator should treat it as unverified.",
        probe_results_json=str(Path(probe_results_path).resolve()),
        probe_results_md=str(Path(probe_results_md_path).resolve()),
    )


def format_capture_preflight_summary(decision: CapturePreflightDecision) -> str:
    serial_label = ",".join(decision.serials) if decision.serials else "<pending>"
    return "\n".join(
        [
            "[record] preflight summary",
            f"  capture_mode: {decision.capture_mode}",
            f"  serials: {serial_label}",
            f"  profile: {decision.width}x{decision.height}@{decision.fps}, emitter={decision.emitter}",
            f"  topology: {decision.topology_type or 'pending'}",
            f"  stream_set: {decision.stream_set or 'n/a'}",
            f"  probe_support: {decision.probe_support}",
            f"  policy: {decision.policy_label}",
            f"  operator_status: {decision.operator_status}",
            f"  allowed_to_record: {decision.allowed_to_record}",
            f"  reason: {decision.reason}",
            f"  probe_json: {decision.probe_results_json}",
        ]
    )
