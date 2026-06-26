"""Legacy compatibility wrapper for :mod:`demo_v5.realtime_data_process_track`."""
from __future__ import annotations

from demo_v5.realtime_data_process_track import *  # noqa: F401,F403
from demo_v5 import realtime_data_process_track as _canonical


_controller_anchor_manifest_fields = _canonical._controller_track_manifest_fields
_controller_quality_invalid = _canonical._track_process_invalid
_controller_quality_online_publish_skip_reason = _canonical._track_process_online_publish_skip_reason
_object_anchor_manifest_fields = _canonical._object_track_manifest_fields


__all__ = [
    "stream_chunks_from_headless_capture",
    "write_chunks_from_headless_capture",
]
