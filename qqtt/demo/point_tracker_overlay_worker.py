from __future__ import annotations

from qqtt.demo.cotracker3_overlay_worker import (
    COTRACKER_UPDATE_MODE_AUTO,
    COTRACKER_UPDATE_MODE_BATCH,
    COTRACKER_UPDATE_MODE_SERIAL,
    COTRACKER_UPDATE_MODES,
    CoTracker3OverlayThread,
    CoTracker3OverlayWorker,
    LatestTrackingInputSlot,
    LatestTrackingOverlaySlot,
    OVERLAY_DISPLAY_SCOPE_CONTROLLER,
    OVERLAY_DISPLAY_SCOPE_OBJECT,
    OVERLAY_DISPLAY_SCOPE_UNION,
    OVERLAY_DISPLAY_SCOPES,
    TrackingOverlayInputPacket,
    TrackingOverlayPacket,
)


PointTrackerOverlayWorker = CoTracker3OverlayWorker
PointTrackerOverlayThread = CoTracker3OverlayThread


__all__ = [
    "COTRACKER_UPDATE_MODE_AUTO",
    "COTRACKER_UPDATE_MODE_BATCH",
    "COTRACKER_UPDATE_MODE_SERIAL",
    "COTRACKER_UPDATE_MODES",
    "LatestTrackingInputSlot",
    "LatestTrackingOverlaySlot",
    "OVERLAY_DISPLAY_SCOPE_CONTROLLER",
    "OVERLAY_DISPLAY_SCOPE_OBJECT",
    "OVERLAY_DISPLAY_SCOPE_UNION",
    "OVERLAY_DISPLAY_SCOPES",
    "PointTrackerOverlayThread",
    "PointTrackerOverlayWorker",
    "TrackingOverlayInputPacket",
    "TrackingOverlayPacket",
]
