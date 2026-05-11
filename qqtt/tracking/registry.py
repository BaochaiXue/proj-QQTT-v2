from __future__ import annotations

from typing import Any

from .base import BackendAvailability, TrackingBackend
from .backends.cotracker3_online import CoTracker3OnlineBackend
from .backends.external_probe import LocoTrackBackend, TapirBackend, TapNextBackend
from .backends.fake import FakeTrackingBackend
from .backends.nvofa import NvofaBackend
from .backends.unavailable import UnavailableBackend
from .backends.vpi_lk import VpiLkBackend


BACKEND_NAMES = ["cotracker3_online", "nvofa", "tapnext", "locotrack", "tapir", "vpi_lk"]

PLANNED_BACKEND_REASONS: dict[str, str] = {
    "nvofa": "NVIDIA Optical Flow SDK Python binding not found or not implemented in this repo.",
    "tapnext": "tapnext backend not implemented or dependency missing.",
    "locotrack": "locotrack backend not implemented or dependency missing.",
    "tapir": "tapir backend not implemented or dependency missing.",
    "vpi_lk": "NVIDIA VPI Python binding not found or VPI LK backend not implemented.",
}


def available_backend_names() -> tuple[str, ...]:
    return tuple(BACKEND_NAMES)


def create_backend(name: str, **kwargs: Any) -> TrackingBackend:
    backend_name = str(name).strip().lower()
    if backend_name == "cotracker3_online":
        return CoTracker3OnlineBackend(**kwargs)
    if backend_name == "fake":
        return FakeTrackingBackend()
    if backend_name == "nvofa":
        return NvofaBackend(**kwargs)
    if backend_name == "vpi_lk":
        return VpiLkBackend(**kwargs)
    if backend_name == "tapnext":
        return TapNextBackend(**kwargs)
    if backend_name == "locotrack":
        return LocoTrackBackend(**kwargs)
    if backend_name == "tapir":
        return TapirBackend(**kwargs)
    if backend_name in PLANNED_BACKEND_REASONS:
        return UnavailableBackend(backend_name, PLANNED_BACKEND_REASONS[backend_name])
    raise KeyError(f"Unknown tracking backend: {name}")


def check_backend_availability(names: list[str] | tuple[str, ...] | None = None) -> dict[str, BackendAvailability]:
    backend_names = available_backend_names() if names is None else tuple(names)
    availability: dict[str, BackendAvailability] = {}
    for name in backend_names:
        try:
            backend = create_backend(name)
            availability[str(name)] = backend.availability()
        except Exception as exc:
            availability[str(name)] = BackendAvailability(str(name), False, str(exc))
    return availability
