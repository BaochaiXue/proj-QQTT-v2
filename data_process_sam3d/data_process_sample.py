# Optionally do the shape completion for the object points (including both suface and interior points)
# Do the volume sampling for the object points, prioritize the original object points, then surface points, then interior points

import numpy as np
import open3d as o3d
import pickle
import matplotlib.pyplot as plt
import trimesh
import cv2
from utils.align_util import as_mesh
from argparse import ArgumentParser
from scipy.spatial import cKDTree

try:
    from data_process_sam3d.shape_prior_sampling import (
        ShapePriorBatchSelector,
        effective_shape_prior_max_dist,
    )
except ModuleNotFoundError:
    from shape_prior_sampling import (
        ShapePriorBatchSelector,
        effective_shape_prior_max_dist,
    )

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
parser.add_argument("--shape_prior", action="store_true", default=False)
parser.add_argument("--num_surface_points", type=int, default=1024)
parser.add_argument("--volume_sample_size", type=float, default=0.005)
parser.add_argument(
    "--shape_prior_max_dist",
    type=float,
    default=0.05,
    help=(
        "Filter sampled shape-prior points that are too far from observed object points "
        "(meters; set <=0 to disable; positive values are capped at 0.035 m)."
    ),
)
parser.add_argument(
    "--ground-policy",
    "--ground_policy",
    choices=["preserve", "clamp-positive-z"],
    default="preserve",
    help=(
        "How to handle object point z values before volume sampling. "
        "preserve keeps the input coordinate frame unchanged; clamp-positive-z "
        "keeps legacy behavior by clamping z values above --ground-z to --ground-z."
    ),
)
parser.add_argument(
    "--ground-z",
    "--ground_z",
    type=float,
    default=0.0,
    help="Ground plane z used only by --ground-policy clamp-positive-z.",
)
parser.add_argument("--target_surface_points", type=int, default=700)
parser.add_argument("--target_interior_points", type=int, default=1000)
parser.add_argument(
    "--skip_visualization",
    action="store_true",
    default=False,
    help="Skip final turntable videos while still writing final_data.pkl.",
)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name

# Used to judge if using the shape prior
SHAPE_PRIOR = args.shape_prior
num_surface_points = args.num_surface_points
volume_sample_size = args.volume_sample_size
shape_prior_max_dist = args.shape_prior_max_dist
ground_policy = args.ground_policy
ground_z = args.ground_z
target_surface_points = args.target_surface_points
target_interior_points = args.target_interior_points
skip_visualization = args.skip_visualization


def filter_points_by_nn_distance(
    points: np.ndarray, reference_points: np.ndarray, max_dist: float
) -> np.ndarray:
    if max_dist <= 0 or points.size == 0 or reference_points.size == 0:
        return points
    tree = cKDTree(reference_points)
    distances, _ = tree.query(points, k=1)
    return points[distances <= max_dist]


def apply_ground_policy(points: np.ndarray) -> np.ndarray:
    """Apply the configured ground policy without changing coordinate frames by default."""
    if ground_policy == "preserve":
        return points
    if ground_policy == "clamp-positive-z":
        points = np.asarray(points).copy()
        points[points[..., 2] > ground_z, 2] = ground_z
        return points
    raise ValueError(f"Unsupported ground policy: {ground_policy}")


def point_grid_index(
    point: np.ndarray, min_bound: np.ndarray, grid_size: float | None = None
) -> tuple[int, int, int]:
    size = volume_sample_size if grid_size is None else grid_size
    return tuple(np.floor((point - min_bound) / size).astype(int))


def dedupe_points(
    points: np.ndarray,
    min_bound: np.ndarray,
    occupied: set[tuple[int, int, int]] | None = None,
    limit: int | None = None,
    grid_size: float | None = None,
) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    seen: set[tuple[int, int, int]] = set() if occupied is None else set(occupied)
    selected = []
    for point in points:
        grid_index = point_grid_index(point, min_bound, grid_size=grid_size)
        if grid_index in seen:
            continue
        seen.add(grid_index)
        selected.append(point)
        if limit is not None and len(selected) >= limit:
            break
    if not selected:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(selected)


def sort_by_reference_distance(points: np.ndarray, reference_points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    tree = cKDTree(reference_points)
    distances, _ = tree.query(points, k=1)
    return points[np.argsort(distances)]


def voxel_interior_candidates(
    trimesh_mesh: trimesh.Trimesh,
    reference_points: np.ndarray,
    max_dist: float,
) -> np.ndarray:
    bounds = trimesh_mesh.bounds
    spacing = max(volume_sample_size, 1e-4)
    axes = [
        np.arange(bounds[0, axis] + spacing * 0.5, bounds[1, axis], spacing)
        for axis in range(3)
    ]
    if any(len(axis) == 0 for axis in axes):
        return np.zeros((0, 3), dtype=np.float32)
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    if grid.shape[0] > 250000:
        step = int(np.ceil(grid.shape[0] / 250000))
        grid = grid[::step]

    try:
        scene = o3d.t.geometry.RaycastingScene()
        vertices = o3d.core.Tensor(np.asarray(trimesh_mesh.vertices), dtype=o3d.core.Dtype.Float32)
        triangles = o3d.core.Tensor(np.asarray(trimesh_mesh.faces), dtype=o3d.core.Dtype.UInt32)
        scene.add_triangles(vertices, triangles)
        signed = scene.compute_signed_distance(
            o3d.core.Tensor(grid.astype(np.float32), dtype=o3d.core.Dtype.Float32)
        ).numpy()
        interior = grid[signed < 0]
    except Exception:
        try:
            interior = grid[trimesh_mesh.contains(grid)]
        except Exception:
            interior = np.zeros((0, 3), dtype=np.float32)

    return interior


def sample_sam3d_prior_points(
    trimesh_mesh: trimesh.Trimesh,
    reference_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(42)
    min_bound = np.min(reference_points, axis=0)
    prior_grid_size = max(volume_sample_size * 0.4, 1e-4)
    max_dist = effective_shape_prior_max_dist(shape_prior_max_dist)
    reference_tree = cKDTree(reference_points)

    surface_selector = ShapePriorBatchSelector(
        reference_points=reference_points,
        min_bound=min_bound,
        grid_size=prior_grid_size,
        max_dist=max_dist,
        reference_tree=reference_tree,
    )
    surface_points = np.zeros((0, 3), dtype=np.float32)
    for count in [max(num_surface_points, 4096), 10000, 50000, 200000]:
        sampled, _ = trimesh.sample.sample_surface(trimesh_mesh, count)
        surface_selector.add_batch(sampled, limit=target_surface_points)
        surface_points = surface_selector.points()
        if len(surface_points) >= target_surface_points:
            break
    if len(surface_points) < target_surface_points:
        for _ in range(2):
            sampled, _ = trimesh.sample.sample_surface(trimesh_mesh, 200000)
            surface_selector.add_batch(sampled, limit=target_surface_points)
            surface_points = surface_selector.points()
            if len(surface_points) >= target_surface_points:
                break

    interior_selector = ShapePriorBatchSelector(
        reference_points=reference_points,
        min_bound=min_bound,
        grid_size=prior_grid_size,
        max_dist=max_dist,
        reference_tree=reference_tree,
    )
    interior_points = np.zeros((0, 3), dtype=np.float32)
    for count in [10000, 50000, 200000]:
        try:
            sampled = trimesh.sample.volume_mesh(trimesh_mesh, count)
        except Exception:
            sampled = np.zeros((0, 3), dtype=np.float32)
        interior_selector.add_batch(sampled, limit=target_interior_points)
        interior_points = interior_selector.points()
        if len(interior_points) >= target_interior_points:
            break

    if len(interior_points) < target_interior_points:
        fallback = voxel_interior_candidates(
            trimesh_mesh, reference_points, max_dist
        )
        if fallback.size:
            interior_selector.add_batch(fallback, limit=target_interior_points)
            interior_points = interior_selector.points()

    return surface_points, interior_points


def getSphereMesh(center, radius=0.1, color=[0, 0, 0]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius).translate(center)
    sphere.paint_uniform_color(color)
    return sphere


def process_unique_points(track_data):
    object_points = track_data["object_points"]
    object_colors = track_data["object_colors"]
    object_visibilities = track_data["object_visibilities"]
    object_motions_valid = track_data["object_motions_valid"]
    controller_points = track_data["controller_points"]

    # Get the unique index in the object points
    first_object_points = object_points[0]
    unique_idx = np.unique(first_object_points, axis=0, return_index=True)[1]
    object_points = object_points[:, unique_idx, :]
    object_colors = object_colors[:, unique_idx, :]
    object_visibilities = object_visibilities[:, unique_idx]
    object_motions_valid = object_motions_valid[:, unique_idx]

    object_points = apply_ground_policy(object_points)

    if SHAPE_PRIOR:
        shape_mesh_path = f"{base_path}/{case_name}/shape/matching/final_mesh.glb"
        trimesh_mesh = trimesh.load(shape_mesh_path, force="mesh")
        trimesh_mesh = as_mesh(trimesh_mesh)
        surface_points, interior_points = sample_sam3d_prior_points(
            trimesh_mesh, object_points[0]
        )

    if SHAPE_PRIOR:
        all_points = np.concatenate(
            [surface_points, interior_points, object_points[0]], axis=0
        )
    else:
        all_points = object_points[0]
    # Do the volume sampling for the object points, prioritize the original object points, then surface points, then interior points
    min_bound = np.min(all_points, axis=0)
    index = []
    grid_flag = {}
    for i in range(object_points.shape[1]):
        grid_index = tuple(
            np.floor((object_points[0, i] - min_bound) / volume_sample_size).astype(int)
        )
        if grid_index not in grid_flag:
            grid_flag[grid_index] = 1
            index.append(i)
    if SHAPE_PRIOR:
        final_surface_points = surface_points[:target_surface_points]
        final_interior_points = interior_points[:target_interior_points]
        all_points = np.concatenate(
            [final_surface_points, final_interior_points, object_points[0][index]],
            axis=0,
        )
    else:
        all_points = object_points[0][index]

    if not skip_visualization:
        # Render the final pcd with interior filling as a turntable video
        all_pcd = o3d.geometry.PointCloud()
        all_pcd.points = o3d.utility.Vector3dVector(all_points)
        coorindate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        dummy_frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        height, width, _ = dummy_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            f"{base_path}/{case_name}/final_pcd.mp4", fourcc, 30, (width, height)
        )
        if not video_writer.isOpened():
            raise RuntimeError(
                f"Failed to open VideoWriter for {base_path}/{case_name}/final_pcd.mp4"
            )

        vis.add_geometry(all_pcd)
        # vis.add_geometry(coorindate)
        view_control = vis.get_view_control()
        for j in range(360):
            view_control.rotate(10, 0)
            vis.poll_events()
            vis.update_renderer()
            frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            frame = (frame * 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)
        video_writer.release()
        vis.destroy_window()

    track_data.pop("object_points")
    track_data.pop("object_colors")
    track_data.pop("object_visibilities")
    track_data.pop("object_motions_valid")
    track_data["object_points"] = object_points[:, index, :]
    track_data["object_colors"] = object_colors[:, index, :]
    track_data["object_visibilities"] = object_visibilities[:, index]
    track_data["object_motions_valid"] = object_motions_valid[:, index]
    if SHAPE_PRIOR:
        track_data["surface_points"] = np.array(final_surface_points)
        track_data["interior_points"] = np.array(final_interior_points)
    else:
        track_data["surface_points"] = np.zeros((0, 3))
        track_data["interior_points"] = np.zeros((0, 3))
    track_data["shape_prior_ground_policy"] = ground_policy
    track_data["shape_prior_ground_z"] = float(ground_z)

    return track_data


def visualize_track(track_data):
    object_points = track_data["object_points"]
    object_colors = track_data["object_colors"]
    object_visibilities = track_data["object_visibilities"]
    object_motions_valid = track_data["object_motions_valid"]
    controller_points = track_data["controller_points"]

    frame_num = object_points.shape[0]

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    dummy_frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    height, width, _ = dummy_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        f"{base_path}/{case_name}/final_data.mp4", fourcc, 30, (width, height)
    )
    if not video_writer.isOpened():
        raise RuntimeError(
            f"Failed to open VideoWriter for {base_path}/{case_name}/final_data.mp4"
        )

    controller_meshes = []
    prev_center = []

    y_min, y_max = np.min(object_points[0, :, 1]), np.max(object_points[0, :, 1])
    y_normalized = (object_points[0, :, 1] - y_min) / (y_max - y_min)
    rainbow_colors = plt.cm.rainbow(y_normalized)[:, :3]

    for i in range(frame_num):
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(
            object_points[i, np.where(object_visibilities[i])[0], :]
        )
        # object_pcd.colors = o3d.utility.Vector3dVector(
        #     object_colors[i, np.where(object_motions_valid[i])[0], :]
        # )
        object_pcd.colors = o3d.utility.Vector3dVector(
            rainbow_colors[np.where(object_visibilities[i])[0]]
        )

        if i == 0:
            render_object_pcd = object_pcd
            vis.add_geometry(render_object_pcd)
            # Use sphere mesh for each controller point
            for j in range(controller_points.shape[1]):
                origin = controller_points[i, j]
                origin_color = [1, 0, 0]
                controller_meshes.append(
                    getSphereMesh(origin, color=origin_color, radius=0.01)
                )
                vis.add_geometry(controller_meshes[-1])
                prev_center.append(origin)
            # Adjust the viewpoint
            view_control = vis.get_view_control()
            view_control.set_front([1, 0, -2])
            view_control.set_up([0, 0, -1])
            view_control.set_zoom(1)
        else:
            render_object_pcd.points = o3d.utility.Vector3dVector(object_pcd.points)
            render_object_pcd.colors = o3d.utility.Vector3dVector(object_pcd.colors)
            vis.update_geometry(render_object_pcd)
            for j in range(controller_points.shape[1]):
                origin = controller_points[i, j]
                controller_meshes[j].translate(origin - prev_center[j])
                vis.update_geometry(controller_meshes[j])
                prev_center[j] = origin
            vis.poll_events()
            vis.update_renderer()

        frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        frame = (frame * 255).astype(np.uint8)
        # Convert RGB to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
    video_writer.release()


if __name__ == "__main__":
    with open(f"{base_path}/{case_name}/track_process_data.pkl", "rb") as f:
        track_data = pickle.load(f)

    track_data = process_unique_points(track_data)

    with open(f"{base_path}/{case_name}/final_data.pkl", "wb") as f:
        pickle.dump(track_data, f)

    if not skip_visualization:
        visualize_track(track_data)
