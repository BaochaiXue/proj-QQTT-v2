import numpy as np
from demo_v5.contracts import DEMO_V5_SCHEMA_NAME, DEMO_V5_SCHEMA_VERSION, DemoV5SessionTopology, hash_topology
from demo_v5.controller_selection import select_controller_query_ids
from demo_v5.object_sampling import select_object_query_ids
from demo_v5.topology_warmup import prepare_warmup


def build_session_topology(warmup_frames, *, surface_points, interior_points, fps=5.0, coordinate_frame="table_world_z0", controller_count=30, object_voxel_m=0.005, minimum_shape_surface=700, minimum_shape_interior=1000):
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    surface, interior, queries, semantics, points_t, valid_t, object_candidates, controller_candidates = prepare_warmup(warmup_frames, surface_points, interior_points, minimum_shape_surface, minimum_shape_interior)
    object_ids = select_object_query_ids(object_candidates, points_t[0], surface, interior, object_voxel_m)
    controller_ids = select_controller_query_ids(points_t, valid_t, controller_candidates, controller_count)
    rest = np.ascontiguousarray(points_t[0], dtype=np.float32)
    selected_rest = np.concatenate([rest[object_ids], rest[controller_ids]])
    if not np.isfinite(selected_rest).all() or np.any(np.linalg.norm(selected_rest, axis=1) <= 1e-9):
        raise ValueError("selected topology has nonfinite or zero points")
    metadata = {"schema_name": DEMO_V5_SCHEMA_NAME, "schema_version": DEMO_V5_SCHEMA_VERSION, "fps": fps, "coordinate_frame": coordinate_frame, "object_voxel_m": object_voxel_m, "controller_count": controller_count, "warmup_frame_count": len(warmup_frames)}
    topology_hash = hash_topology((np.round(queries, 4).astype(np.float32), semantics, object_ids, controller_ids, np.round(selected_rest, 6).astype(np.float32)), metadata)
    return DemoV5SessionTopology(fps=fps, frame_dt_s=1.0 / fps, warmup_frame_count=len(warmup_frames), coordinate_frame=coordinate_frame, query_points_yx=np.ascontiguousarray(queries), query_semantics=np.ascontiguousarray(semantics), query_rest_points=rest, object_candidate_query_ids=np.ascontiguousarray(object_candidates), controller_candidate_query_ids=np.ascontiguousarray(controller_candidates), object_query_ids=np.ascontiguousarray(object_ids), controller_query_ids=np.ascontiguousarray(controller_ids), topology_hash=topology_hash)
