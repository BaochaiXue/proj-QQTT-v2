import numpy as np
from demo_v5.contracts import as_points
from demo_v5.knn_recovery import recover_selected
from demo_v5.motion_filter import motion_valid_for_points
from demo_v5.tracking_samples import frame_direct_samples


class SessionProjector:
    def __init__(self, topology, *, surface_points, interior_points, recovery_k=8, recovery_radius_m=0.08):
        self.topology = topology
        self.surface_points = as_points(surface_points)
        self.interior_points = as_points(interior_points)
        self.recovery_k = int(recovery_k)
        self.recovery_radius_m = float(recovery_radius_m)
        self.last_object = self.last_controller = self.last_colors = None

    def project(self, frames, *, lookahead_frame=None):
        source = list(frames) + ([] if lookahead_frame is None else [lookahead_frame])
        if not frames:
            raise ValueError("project requires frames")
        names = ("object_points", "object_colors", "object_visibilities", "object_recovered", "object_recovery_confidence", "controller_points", "controller_observed", "controller_recovered", "controller_recovery_confidence")
        out = {name: [] for name in names}
        for frame in source:
            queries = np.asarray(frame.query_points_yx, dtype=np.float32).reshape(-1, 2)
            if queries.shape != self.topology.query_points_yx.shape or not np.allclose(queries, self.topology.query_points_yx, atol=1e-4, rtol=0.0):
                raise ValueError("query identity changed")
            points, colors, valid = frame_direct_samples(frame, self.topology.query_semantics)
            obj_ids, ctrl_ids = self.topology.object_query_ids, self.topology.controller_query_ids
            obj, obj_color, obj_rec, obj_conf = recover_selected(self.topology.query_rest_points, points, colors, valid, obj_ids, self.topology.object_candidate_query_ids, self.last_object, self.last_colors, self.recovery_k, self.recovery_radius_m)
            ctrl, _, ctrl_rec, ctrl_conf = recover_selected(self.topology.query_rest_points, points, colors, valid, ctrl_ids, self.topology.controller_candidate_query_ids, self.last_controller, None, self.recovery_k, self.recovery_radius_m)
            values = (obj, obj_color, valid[obj_ids], obj_rec, obj_conf, ctrl, valid[ctrl_ids], ctrl_rec, ctrl_conf)
            for name, value in zip(names, values):
                out[name].append(value)
            self.last_object, self.last_controller, self.last_colors = obj.copy(), ctrl.copy(), obj_color.copy()
        motion, _ = motion_valid_for_points(np.stack(out["object_points"]), np.stack(out["object_visibilities"]))
        count = len(frames)
        result = {name: np.ascontiguousarray(np.stack(value)[:count]) for name, value in out.items()}
        result["object_motions_valid"] = np.ascontiguousarray(motion[:count])
        result["surface_points"], result["interior_points"] = self.surface_points, self.interior_points
        if result["controller_points"].shape[1:] != (30, 3):
            raise RuntimeError("expected 30 controller anchors")
        return result
