from __future__ import annotations

from dataclasses import dataclass, field, replace
import threading
from typing import Any


BATCH_BUNDLE_POLICY_SAME_BUNDLE_LATEST_WINS = "same-bundle-latest-wins"
BATCH_BUNDLE_POLICY_STRICT_SOURCE = "strict-source"
BATCH_BUNDLE_POLICY_LATEST_REUSE_DEBUG = "latest-reuse-debug"
BATCH_BUNDLE_POLICIES = (
    BATCH_BUNDLE_POLICY_SAME_BUNDLE_LATEST_WINS,
    BATCH_BUNDLE_POLICY_STRICT_SOURCE,
    BATCH_BUNDLE_POLICY_LATEST_REUSE_DEBUG,
)


@dataclass(frozen=True)
class BundleProvenance:
    bundle_group_id: int
    rgb_group_id: int | None = None
    depth_group_id: int | None = None
    mask_group_id: int | None = None
    query_group_id: int | None = None
    tracker_result_group_id: int | None = None
    pcd_group_id: int | None = None
    surface_anchor_group_id: int | None = None
    render_group_id: int | None = None

    def asdict(self) -> dict[str, int | None | bool]:
        values = {
            "bundle_group_id": int(self.bundle_group_id),
            "rgb_group_id": self.rgb_group_id,
            "depth_group_id": self.depth_group_id,
            "mask_group_id": self.mask_group_id,
            "query_group_id": self.query_group_id,
            "tracker_result_group_id": self.tracker_result_group_id,
            "pcd_group_id": self.pcd_group_id,
            "surface_anchor_group_id": self.surface_anchor_group_id,
            "render_group_id": self.render_group_id,
        }
        expected = int(self.bundle_group_id)
        present = [value for value in values.values() if value is not None and isinstance(value, int)]
        values["same_bundle_rendered"] = bool(present and all(int(value) == expected for value in present))
        return values


@dataclass(frozen=True)
class Batch3FrameBundle:
    group_id: int
    created_perf_s: float = 0.0
    timestamp_s: float = 0.0
    depth_group: Any | None = None
    masks: dict[int, Any] = field(default_factory=dict)
    pending_fusion_bundle: Any | None = None
    precomputed_render_packet: Any | None = None
    tracker_input_published: bool = False
    tracker_result: Any | None = None
    protected: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)

    def with_update(self, **updates: Any) -> "Batch3FrameBundle":
        return replace(self, **updates)


@dataclass(frozen=True)
class BundleStoreMatch:
    bundle: Batch3FrameBundle | None
    match_mode: str
    pending_ids_before: tuple[int, ...]
    used_nearest: bool = False


class BundleStore:
    """Bounded exact-group store for rendered same-bundle tracker pipelines."""

    def __init__(self, *, max_groups: int = 128) -> None:
        self.max_groups = max(1, int(max_groups))
        self._bundles: dict[int, Batch3FrameBundle] = {}
        self._protected_group_ids: set[int] = set()
        self._lock = threading.Lock()
        self.created_count = 0
        self.pending_fusion_attached_count = 0
        self.precomputed_render_attached_count = 0
        self.tracker_input_protected_count = 0
        self.exact_match_count = 0
        self.nearest_match_count = 0
        self.missing_exact_count = 0
        self.evicted_count = 0
        self.drop_through_count = 0
        self.protected_eviction_avoided_count = 0

    def upsert(self, group_id: int, **updates: Any) -> Batch3FrameBundle:
        gid = int(group_id)
        with self._lock:
            bundle = self._bundles.get(gid)
            if bundle is None:
                bundle = Batch3FrameBundle(group_id=gid)
                self.created_count += 1
            bundle = bundle.with_update(**updates)
            if gid in self._protected_group_ids:
                bundle = bundle.with_update(protected=True)
            self._bundles[gid] = bundle
            self._prune_locked()
            return bundle

    def attach_pending_fusion_bundle(self, bundle: Any) -> Batch3FrameBundle:
        gid = int(bundle.group_id)
        updated = self.upsert(
            gid,
            created_perf_s=float(getattr(bundle, "created_perf_s", 0.0) or 0.0),
            timestamp_s=float(getattr(getattr(bundle, "depth_group", None), "timestamp_s", 0.0) or 0.0),
            depth_group=getattr(bundle, "depth_group", None),
            masks=dict(getattr(bundle, "masks", {}) or {}),
            pending_fusion_bundle=bundle,
        )
        with self._lock:
            self.pending_fusion_attached_count += 1
        return updated

    def attach_precomputed_render_packet(self, packet: Any) -> Batch3FrameBundle:
        gid = int(packet.group_id)
        with self._lock:
            self.precomputed_render_attached_count += 1
        return self.upsert(gid, precomputed_render_packet=packet)

    def protect(self, group_id: int) -> None:
        gid = int(group_id)
        with self._lock:
            self._protected_group_ids.add(gid)
            bundle = self._bundles.get(gid)
            if bundle is not None:
                self._bundles[gid] = bundle.with_update(protected=True, tracker_input_published=True)
            self.tracker_input_protected_count += 1

    def unprotect(self, group_id: int) -> None:
        gid = int(group_id)
        with self._lock:
            self._protected_group_ids.discard(gid)
            bundle = self._bundles.get(gid)
            if bundle is not None:
                self._bundles[gid] = bundle.with_update(protected=False)
            self._prune_locked()

    def get(self, group_id: int) -> Batch3FrameBundle | None:
        with self._lock:
            return self._bundles.get(int(group_id))

    def take_for_tracker_result(self, group_id: int, *, allow_nearest: bool = False) -> BundleStoreMatch:
        gid = int(group_id)
        with self._lock:
            pending_ids = tuple(sorted(int(item) for item in self._renderable_ids_locked()))
            bundle = self._bundles.get(gid)
            if bundle is not None and self._is_renderable_locked(bundle):
                self.exact_match_count += 1
                return BundleStoreMatch(bundle=bundle, match_mode="exact", pending_ids_before=pending_ids)
            if allow_nearest and pending_ids:
                nearest_gid = min(
                    pending_ids,
                    key=lambda candidate: (abs(int(candidate) - gid), 0 if int(candidate) >= gid else 1, int(candidate)),
                )
                self.nearest_match_count += 1
                return BundleStoreMatch(
                    bundle=self._bundles.get(nearest_gid),
                    match_mode="nearest",
                    pending_ids_before=pending_ids,
                    used_nearest=True,
                )
            self.missing_exact_count += 1
            return BundleStoreMatch(bundle=None, match_mode="missing-exact", pending_ids_before=pending_ids)

    def drop_through(self, group_id: int) -> None:
        gid = int(group_id)
        with self._lock:
            stale_ids = [
                key
                for key in self._bundles
                if int(key) <= gid and int(key) not in self._protected_group_ids
            ]
            for key in stale_ids:
                self._bundles.pop(key, None)
                self.drop_through_count += 1

    def group_ids(self) -> set[int]:
        with self._lock:
            return {int(item) for item in self._bundles}

    def renderable_group_ids(self) -> set[int]:
        with self._lock:
            return set(self._renderable_ids_locked())

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            renderable = self._renderable_ids_locked()
            return {
                "bundle_store_groups": int(len(self._bundles)),
                "bundle_store_renderable_groups": int(len(renderable)),
                "bundle_store_oldest_group_id": int(min(self._bundles)) if self._bundles else None,
                "bundle_store_newest_group_id": int(max(self._bundles)) if self._bundles else None,
                "bundle_store_protected_groups": int(len(self._protected_group_ids)),
                "bundle_store_created_count": int(self.created_count),
                "bundle_store_pending_fusion_attached_count": int(self.pending_fusion_attached_count),
                "bundle_store_precomputed_render_attached_count": int(self.precomputed_render_attached_count),
                "bundle_store_tracker_input_protected_count": int(self.tracker_input_protected_count),
                "bundle_store_exact_match_count": int(self.exact_match_count),
                "bundle_store_nearest_match_count": int(self.nearest_match_count),
                "bundle_store_missing_exact_count": int(self.missing_exact_count),
                "bundle_store_evicted_count": int(self.evicted_count),
                "bundle_store_drop_through_count": int(self.drop_through_count),
                "bundle_store_protected_eviction_avoided_count": int(
                    self.protected_eviction_avoided_count
                ),
            }

    def _is_renderable_locked(self, bundle: Batch3FrameBundle) -> bool:
        return bundle.pending_fusion_bundle is not None or bundle.precomputed_render_packet is not None

    def _renderable_ids_locked(self) -> list[int]:
        return [int(group_id) for group_id, bundle in self._bundles.items() if self._is_renderable_locked(bundle)]

    def _prune_locked(self) -> None:
        while len(self._bundles) > self.max_groups:
            unprotected = [group_id for group_id in self._bundles if group_id not in self._protected_group_ids]
            if not unprotected:
                self.protected_eviction_avoided_count += 1
                break
            oldest = min(unprotected)
            self._bundles.pop(oldest, None)
            self.evicted_count += 1


__all__ = [
    "BATCH_BUNDLE_POLICIES",
    "BATCH_BUNDLE_POLICY_LATEST_REUSE_DEBUG",
    "BATCH_BUNDLE_POLICY_SAME_BUNDLE_LATEST_WINS",
    "BATCH_BUNDLE_POLICY_STRICT_SOURCE",
    "Batch3FrameBundle",
    "BundleProvenance",
    "BundleStore",
    "BundleStoreMatch",
]
