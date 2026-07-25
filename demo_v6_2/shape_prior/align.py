# Single-camera port of data_process_origin/align.py (original PhysTwin shape
# alignment). The function bodies are kept structurally parallel to that
# origin file on purpose -- keep them diffable. Demo-specific deltas are
# limited to: package-local imports, VIS defaulting to False, the
# processed-mask camera_count validation, stage timing, and the pre-GO
# candidate prerender (mesh-cache hit; verified at GO, cold fallback).
import json
import os
import pickle
import time
from argparse import ArgumentParser, Namespace

_MODULE_IMPORT_STARTED_S = time.perf_counter()

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402
from demo_v6_2.shape_prior.match_pairs import (  # noqa: E402
    get_matching_model,
    image_pair_matching,
    prepare_candidate_features,
)
from demo_v6_2.shape_prior.mesh_cache import sha256_file  # noqa: E402
from demo_v6_2.shape_prior.timing import (  # noqa: E402
    StageProfileRun,
    elapsed_ms,
)
from scipy.optimize import minimize  # noqa: E402
from scipy.spatial import KDTree  # noqa: E402
from demo_v6_2.utils.align_util import (  # noqa: E402
    as_mesh,
    plot_image_with_points,
    plot_mesh_with_points,
    project_2d_to_3d,
    render_image,
    render_multi_images,
    select_point,
)

_MODULE_IMPORT_MS = elapsed_ms(_MODULE_IMPORT_STARTED_S)

VIS = False
parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
    default="",
)
parser.add_argument("--case_name", type=str, required=True, default="")
parser.add_argument(
    "--controller_name",
    type=str,
    required=True,
    default="",
)
parser.add_argument(
    "--wait-signal",
    dest="wait_signal",
    action="store_true",
    help="Preload CUDA + SuperGlue, then block on stdin for GO before aligning.",
)
parser.add_argument(
    "--profile-json",
    type=str,
    default=None,
    help="Optional JSON path for detailed align-stage timing.",
)
# Import-safe placeholder so the module can be imported without CLI args;
# main() re-parses argv and rebinds these globals.
args = Namespace(
    base_path="",
    case_name="",
    controller_name="",
    wait_signal=False,
    profile_json=None,
)

base_path = args.base_path
case_name = args.case_name
CONTROLLER_NAME = args.controller_name
output_dir = f"{base_path}/{case_name}/shape/matching"


def existDir(dir_path):
    """Return the exist dir."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# Pre-GO candidate prerender (mesh-cache hit only). The orchestrator sends a
# PRERENDER directive with frame-0 geometry while SAM3.1 still runs; the
# waiting worker renders the 8x4 candidate views and their SuperPoint features
# ahead of GO. main() verifies mesh sha + width/height/fov against the run's
# real inputs before using it, so a stale or wrong prerender degrades to the
# cold render instead of changing the output.
_PRERENDER_STATE = None


def _prerender_candidates(payload):
    """Render pose candidates + SuperPoint features from a PRERENDER payload."""
    mesh_path = str(payload["mesh_path"])
    width = int(payload["width"])
    height = int(payload["height"])
    fx = float(payload["fx"])
    # Same fov/radius math as main() + pose_selection_render_superglue; the
    # cached mesh bytes equal the materialized case mesh (sha verified at GO).
    fov = 2 * np.arctan(width / (2 * fx))
    mesh = as_mesh(trimesh.load_mesh(mesh_path, force="mesh"))
    bounding_box = mesh.bounds
    max_dimension = np.linalg.norm(bounding_box[1] - bounding_box[0])
    radius = 2 * (max_dimension / 2) / np.tan(fov / 2)
    colors, depths, camera_poses, camera_intrinsics = render_multi_images(
        mesh_path,
        width,
        height,
        fov,
        radius=radius,
        num_samples=8,
        num_ups=4,
        device="cuda",
    )
    grays = [cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) for color in colors]
    matching = get_matching_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    candidate_features = [
        prepare_candidate_features(matching, gray, device) for gray in grays
    ]
    return {
        "mesh_sha256": str(payload["mesh_sha256"]),
        "width": width,
        "height": height,
        "fov": float(fov),
        "colors": colors,
        "depths": depths,
        "camera_poses": camera_poses,
        "camera_intrinsics": camera_intrinsics,
        "grays": grays,
        "candidate_features": candidate_features,
    }


def _verify_prerender(state, *, mesh_path, raw_img, fov):
    """Return the prerender state when it matches this run's inputs, else None."""
    if state is None:
        return None
    mismatches = []
    if (
        int(raw_img.shape[1]) != state["width"]
        or int(raw_img.shape[0]) != state["height"]
    ):
        mismatches.append("geometry")
    if float(fov) != state["fov"]:
        mismatches.append("fov")
    if sha256_file(mesh_path) != state["mesh_sha256"]:
        mismatches.append("mesh")
    if mismatches:
        print(
            "[prewarm] align: discarding prerender "
            f"(mismatch: {', '.join(mismatches)})",
            flush=True,
        )
        return None
    return state


def pose_selection_render_superglue(
    raw_img,
    fov,
    mesh_path,
    mesh,
    crop_img,
    output_dir,
    timing_ms=None,
    prerender=None,
):
    # Calculate suitable rendering radius
    """Return the pose selection render superglue."""
    render_started_s = time.perf_counter()
    if prerender is not None:
        # Verified prerender: identical mesh bytes + geometry, so these are
        # the exact renders/features the cold branch below would produce.
        colors = prerender["colors"]
        depths = prerender["depths"]
        camera_poses = prerender["camera_poses"]
        camera_intrinsics = prerender["camera_intrinsics"]
        grays = prerender["grays"]
        candidate_features = prerender["candidate_features"]
        print(
            f"[prewarm] align: reusing {len(colors)} prerendered pose candidates",
            flush=True,
        )
    else:
        bounding_box = mesh.bounds
        max_dimension = np.linalg.norm(bounding_box[1] - bounding_box[0])
        radius = 2 * (max_dimension / 2) / np.tan(fov / 2)

        # Render multimle images and feature matching
        colors, depths, camera_poses, camera_intrinsics = render_multi_images(
            mesh_path,
            raw_img.shape[1],
            raw_img.shape[0],
            fov,
            radius=radius,
            num_samples=8,
            num_ups=4,
            device="cuda",
        )
        grays = [cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) for color in colors]
        candidate_features = None
    render_candidates_ms = elapsed_ms(render_started_s)
    # Use superglue to match the features
    match_started_s = time.perf_counter()
    best_idx, match_result = image_pair_matching(
        grays, crop_img, output_dir, viz_best=True,
        candidate_features=candidate_features,
    )
    superglue_match_ms = elapsed_ms(match_started_s)
    print("matched point number", np.sum(match_result["matches"] > -1))

    best_color = colors[best_idx]
    best_depth = depths[best_idx]
    best_pose = camera_poses[best_idx].cpu().numpy()
    if timing_ms is not None:
        timing_ms["render_candidates_ms"] = render_candidates_ms
        timing_ms["superglue_match_ms"] = superglue_match_ms
    return best_color, best_depth, best_pose, match_result, camera_intrinsics


def registration_pnp(mesh_matching_points, raw_matching_points, intrinsic):
    # Solve the PNP and verify the reprojection error
    """Return the registration pnp."""
    success, rvec, tvec = cv2.solvePnP(
        np.float32(mesh_matching_points),
        np.float32(raw_matching_points),
        np.float32(intrinsic),
        distCoeffs=np.zeros(4, dtype=np.float32),
        flags=cv2.SOLVEPNP_EPNP,
    )
    assert success, "solvePnP failed"
    projected_points, _ = cv2.projectPoints(
        np.float32(mesh_matching_points),
        rvec,
        tvec,
        intrinsic,
        np.zeros(4, dtype=np.float32),
    )
    error = np.linalg.norm(
        np.float32(raw_matching_points) - projected_points.reshape(-1, 2), axis=1
    ).mean()
    print(f"Reprojection Error: {error}")
    if error > 50:
        print(f"solvePnP failed for this case {case_name}.$$$$$$$$$$$$$$$$$$$$$$$$$$")

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    mesh2raw_camera = np.eye(4, dtype=np.float32)
    mesh2raw_camera[:3, :3] = rotation_matrix
    mesh2raw_camera[:3, 3] = tvec.squeeze()

    return mesh2raw_camera


def registration_scale(mesh_matching_points_cam, matching_points_cam):
    # After PNP, optimize the scale in the camera coordinate
    """Return the registration scale."""

    def objective(scale, mesh_points, pcd_points):
        """Return the objective."""
        transformed_points = scale * mesh_points
        loss = np.sum(np.sum((transformed_points - pcd_points) ** 2, axis=1))
        return loss

    initial_scale = 1
    result = minimize(
        objective,
        initial_scale,
        args=(mesh_matching_points_cam, matching_points_cam),
        method="L-BFGS-B",
    )
    optimal_scale = result.x[0]
    print("Rescale:", optimal_scale)
    return optimal_scale


def deform_ARAP(initial_mesh_world, mesh_matching_points_world, matching_points):
    # Do the ARAP deformation based on the matching keypoints
    """Return the deform a r a p."""
    mesh_vertices = np.asarray(initial_mesh_world.vertices)
    kdtree = KDTree(mesh_vertices)
    _, mesh_points_indices = kdtree.query(mesh_matching_points_world)
    mesh_points_indices = np.asarray(mesh_points_indices, dtype=np.int32)
    deform_mesh = initial_mesh_world.deform_as_rigid_as_possible(
        o3d.utility.IntVector(mesh_points_indices),
        o3d.utility.Vector3dVector(matching_points),
        max_iter=1,
    )
    return deform_mesh, mesh_points_indices


def get_matching_ray_registration(
    mesh_world, obs_points_world, mesh, trimesh_indices, c2w, w2c
):
    # Get the matching indices and targets based on the viewpoint
    """Return the get matching ray registration."""
    obs_points_cam = np.dot(
        w2c,
        np.hstack((obs_points_world, np.ones((obs_points_world.shape[0], 1)))).T,
    ).T
    obs_points_cam = obs_points_cam[:, :3]
    vertices_cam = np.dot(
        w2c,
        np.hstack(
            (
                np.asarray(mesh_world.vertices),
                np.ones((np.asarray(mesh_world.vertices).shape[0], 1)),
            )
        ).T,
    ).T
    vertices_cam = vertices_cam[:, :3]

    obs_kd = KDTree(obs_points_cam)

    new_indices = []
    new_targets = []
    # trimesh used to do the ray-casting test
    mesh.vertices = np.asarray(vertices_cam)[trimesh_indices]
    # One batched occlusion query instead of one intersects_location call per
    # vertex. The trimesh backend resolves every ray independently (per-ray
    # rtree candidates, per-ray closest-hit argmin under multiple_hits=False),
    # so each per-vertex decision below is identical to the original
    # one-ray-at-a-time loop; only the Python call overhead is removed.
    ray_directions = np.array(
        [vertex / np.linalg.norm(vertex) for vertex in vertices_cam]
    )
    locations, index_rays, _ = mesh.ray.intersects_location(
        ray_origins=np.zeros((len(vertices_cam), 3)),
        ray_directions=ray_directions,
        multiple_hits=False,
    )
    first_hit = {}
    for location, ray_index in zip(locations, index_rays):
        first_hit.setdefault(int(ray_index), location)
    for index, vertex in enumerate(vertices_cam):
        ignore_flag = False

        first_intersection = first_hit.get(index)
        if first_intersection is not None:
            vertex_distance = np.linalg.norm(vertex)
            intersection_distance = np.linalg.norm(first_intersection)
            if intersection_distance < vertex_distance - 1e-4:
                # If the intersection point is not the vertex, it means the vertex is not visible from the camera viewpoint
                ignore_flag = True

        if ignore_flag:
            continue
        else:
            # Select the closest point to the ray of the observation points as the matching point
            indices = obs_kd.query_ball_point(vertex, 0.02)
            line_distances = line_point_distance(vertex, obs_points_cam[indices])
            # Get the closest point
            if len(line_distances) > 0:
                closest_index = np.argmin(line_distances)
                target = np.dot(
                    c2w, np.hstack((obs_points_cam[indices][closest_index], 1))
                )
                new_indices.append(index)
                new_targets.append(target[:3])

    new_indices = np.asarray(new_indices)
    new_targets = np.asarray(new_targets)

    return new_indices, new_targets


def deform_ARAP_ray_registration(
    deform_kp_mesh_world,
    obs_points_world,
    mesh,
    trimesh_indices,
    c2ws,
    w2cs,
    mesh_points_indices,
    matching_points,
):
    """Return the deform a r a p ray registration."""
    final_indices = []
    final_targets = []
    for index, target in zip(mesh_points_indices, matching_points):
        if index not in final_indices:
            final_indices.append(index)
            final_targets.append(target)

    for c2w, w2c in zip(c2ws, w2cs):
        new_indices, new_targets = get_matching_ray_registration(
            deform_kp_mesh_world, obs_points_world, mesh, trimesh_indices, c2w, w2c
        )
        for index, target in zip(new_indices, new_targets):
            if index not in final_indices:
                final_indices.append(index)
                final_targets.append(target)

    # Also need to adjust the positions to make sure they are above the table
    indices = np.where(np.asarray(deform_kp_mesh_world.vertices)[:, 2] > 0)[0]
    for index in indices:
        if index not in final_indices:
            final_indices.append(index)
            target = np.asarray(deform_kp_mesh_world.vertices)[index].copy()
            target[2] = 0
            final_targets.append(target)
        else:
            target = final_targets[final_indices.index(index)]
            if target[2] > 0:
                target[2] = 0
                final_targets[final_indices.index(index)] = target

    final_mesh_world = deform_kp_mesh_world.deform_as_rigid_as_possible(
        o3d.utility.IntVector(final_indices),
        o3d.utility.Vector3dVector(final_targets),
        max_iter=1,
    )
    return final_mesh_world


def align_full_vendor_compatible(
    initial_mesh_world,
    mesh_matching_points_world,
    matching_points,
    obs_points,
    mesh,
    trimesh_indices,
    c2ws,
    w2cs,
    timing_ms=None,
):
    # ARAP based on the keypoints
    """Return the align full vendor compatible."""
    keypoint_started_s = time.perf_counter()
    deform_kp_mesh_world, mesh_points_indices = deform_ARAP(
        initial_mesh_world, mesh_matching_points_world, matching_points
    )
    arap_keypoint_ms = elapsed_ms(keypoint_started_s)

    # Do the ARAP based on both the ray-casting matching and the keypoints
    # Identify the vertex which blocks or blocked by the observation, then match them with the observation points on the ray
    ray_started_s = time.perf_counter()
    final_mesh_world = deform_ARAP_ray_registration(
        deform_kp_mesh_world,
        obs_points,
        mesh,
        trimesh_indices,
        c2ws,
        w2cs,
        mesh_points_indices,
        matching_points,
    )
    if timing_ms is not None:
        timing_ms["arap_keypoint_ms"] = arap_keypoint_ms
        timing_ms["arap_ray_ms"] = elapsed_ms(ray_started_s)
    return final_mesh_world


def line_point_distance(p, points):
    # Compute the distance between points and the line between p and [0, 0, 0]
    """Return the line point distance."""
    p = p / np.linalg.norm(p)
    points_to_origin = points
    cross_product = np.linalg.norm(np.cross(points_to_origin, p), axis=1)
    return cross_product / np.linalg.norm(p)


def _prewarm_models():
    """Initialize the CUDA context and load SuperGlue weights ahead of GO.

    ``pose_selection_render_superglue`` -> ``image_pair_matching`` reuses the
    cached model via ``get_matching_model`` defaults, so this only front-loads
    checkpoint I/O and CUDA init; the compute path stays byte-identical.
    """
    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")
    from demo_v6_2.shape_prior.match_pairs import (  # noqa: PLC0415
        get_matching_model,
    )

    get_matching_model()


def main(argv=None):
    """Run the command-line entry point."""
    global args, base_path, case_name, CONTROLLER_NAME, output_dir

    args = parser.parse_args(argv)
    base_path = args.base_path
    case_name = args.case_name
    CONTROLLER_NAME = args.controller_name
    output_dir = f"{base_path}/{case_name}/shape/matching"
    run = StageProfileRun(
        stage="align",
        profile_json=args.profile_json,
        wait_signal=args.wait_signal,
        timing_ms={
            "module_import_ms": _MODULE_IMPORT_MS,
            "model_prewarm_ms": 0.0,
            "prerender_ms": 0.0,
            "input_load_ms": 0.0,
            "render_candidates_ms": 0.0,
            "superglue_match_ms": 0.0,
            "pnp_scale_ms": 0.0,
            "observation_prepare_ms": 0.0,
            "arap_keypoint_ms": 0.0,
            "arap_ray_ms": 0.0,
            "mesh_export_ms": 0.0,
            "go_wait_ms": 0.0,
            "post_go_ms": 0.0,
            "total_ms": 0.0,
            "process_lifetime_ms": 0.0,
        },
        active_fields=(
            "module_import_ms",
            "model_prewarm_ms",
            "input_load_ms",
            "render_candidates_ms",
            "superglue_match_ms",
            "pnp_scale_ms",
            "observation_prepare_ms",
            "arap_keypoint_ms",
            "arap_ray_ms",
            "mesh_export_ms",
        ),
        process_started_s=_MODULE_IMPORT_STARTED_S,
    )
    timing_ms = run.timing_ms

    if args.wait_signal:
        prewarm_started_s = time.perf_counter()
        _prewarm_models()
        timing_ms["model_prewarm_ms"] = elapsed_ms(prewarm_started_s)
        run.write_waiting()

        def _handle_prerender_directive(payload_text):
            # Prerender is a pre-GO optimization: any failure logs and falls
            # back to the cold render at GO instead of killing the worker.
            global _PRERENDER_STATE
            directive_started_s = time.perf_counter()
            try:
                _PRERENDER_STATE = _prerender_candidates(json.loads(payload_text))
                print(
                    "[prewarm] align: prerendered "
                    f"{len(_PRERENDER_STATE['colors'])} pose candidates pre-GO",
                    flush=True,
                )
            except Exception as exc:
                _PRERENDER_STATE = None
                print(
                    "[prewarm] align: prerender failed, will render cold: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            timing_ms["prerender_ms"] = elapsed_ms(directive_started_s)

        if not run.wait_for_go(on_directive=_handle_prerender_directive):
            return

    stage_started_s = time.perf_counter()
    input_load_started_s = time.perf_counter()
    existDir(output_dir)

    cam_idx = 0
    img_path = f"{base_path}/{case_name}/color/{cam_idx}/0.png"
    mesh_path = f"{base_path}/{case_name}/shape/object.glb"
    # Get the mask index of the object
    with open(f"{base_path}/{case_name}/mask/mask_info_{cam_idx}.json", "r") as f:
        data = json.load(f)
    obj_idx = None
    for key, value in data.items():
        if value != CONTROLLER_NAME:
            if obj_idx is not None:
                raise ValueError("More than one object detected.")
            obj_idx = int(key)
    mask_img_path = f"{base_path}/{case_name}/mask/{cam_idx}/{obj_idx}/0.png"
    # Load the metadata
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    intrinsic = np.array(data["intrinsics"])[cam_idx]

    # Load the c2w for the camera
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
        c2w = c2ws[cam_idx]
        w2c = np.linalg.inv(c2w)
        w2cs = [np.linalg.inv(c2w) for c2w in c2ws]

    # Load the shape prior
    mesh = trimesh.load_mesh(mesh_path, force="mesh")
    mesh = as_mesh(mesh)

    # Load and process the image to get a cropped version for easy superglue
    raw_img = cv2.imread(img_path)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    # Get mask bounding box, larger than the original bounding box
    mask_img = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)

    # Calculate camera parameters
    fov = 2 * np.arctan(raw_img.shape[1] / (2 * intrinsic[0, 0]))
    timing_ms["input_load_ms"] = elapsed_ms(input_load_started_s)

    if not os.path.exists(f"{output_dir}/best_match.pkl"):
        # 2D feature Matching to get the best pose of the object
        crop_started_s = time.perf_counter()
        bbox = np.argwhere(mask_img > 0.8 * 255)
        bbox = (
            np.min(bbox[:, 1]),
            np.min(bbox[:, 0]),
            np.max(bbox[:, 1]),
            np.max(bbox[:, 0]),
        )
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = (
            int(center[0] - size // 2),
            int(center[1] - size // 2),
            int(center[0] + size // 2),
            int(center[1] + size // 2),
        )
        # Make sure the bounding box is within the image
        bbox = (
            max(0, bbox[0]),
            max(0, bbox[1]),
            min(raw_img.shape[1], bbox[2]),
            min(raw_img.shape[0], bbox[3]),
        )
        # Get the masked cropped image used for superglue
        crop_img = raw_img.copy()
        mask_bool = mask_img > 0
        crop_img[~mask_bool] = 0
        crop_img = crop_img[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
        timing_ms["input_load_ms"] += elapsed_ms(crop_started_s)

        # Render the object and match the features. A verified prerender
        # (mesh sha + geometry + fov all matching) skips the candidate render
        # and per-candidate SuperPoint; any mismatch renders cold.
        prerender = _verify_prerender(
            _PRERENDER_STATE,
            mesh_path=mesh_path,
            raw_img=raw_img,
            fov=fov,
        )
        best_color, best_depth, best_pose, match_result, camera_intrinsics = (
            pose_selection_render_superglue(
                raw_img,
                fov,
                mesh_path,
                mesh,
                crop_img,
                output_dir=output_dir,
                timing_ms=timing_ms,
                prerender=prerender,
            )
        )
        with open(f"{output_dir}/best_match.pkl", "wb") as f:
            pickle.dump(
                [
                    best_color,
                    best_depth,
                    best_pose,
                    match_result,
                    camera_intrinsics,
                    bbox,
                ],
                f,
            )
    else:
        cached_match_started_s = time.perf_counter()
        with open(f"{output_dir}/best_match.pkl", "rb") as f:
            best_color, best_depth, best_pose, match_result, camera_intrinsics, bbox = (
                pickle.load(f)
            )
        timing_ms["input_load_ms"] += elapsed_ms(cached_match_started_s)

    # Process to get the matching points on the mesh and on the image
    # Get the projected 3D matching points on the mesh
    observation_started_s = time.perf_counter()
    valid_matches = match_result["matches"] > -1
    render_matching_points = match_result["keypoints0"][valid_matches]
    mesh_matching_points, valid_mask = project_2d_to_3d(
        render_matching_points, best_depth, camera_intrinsics, best_pose
    )
    render_matching_points = render_matching_points[valid_mask]
    # Get the matching points on the raw image
    raw_matching_points_box = match_result["keypoints1"][
        match_result["matches"][valid_matches]
    ]
    raw_matching_points_box = raw_matching_points_box[valid_mask]
    raw_matching_points = raw_matching_points_box + np.array([bbox[0], bbox[1]])
    timing_ms["observation_prepare_ms"] += elapsed_ms(observation_started_s)

    if VIS:
        # Do visualization for the matching
        plot_mesh_with_points(
            mesh,
            mesh_matching_points,
            f"{output_dir}/mesh_matching.png",
        )
        plot_image_with_points(
            best_depth,
            render_matching_points,
            f"{output_dir}/render_matching.png",
        )
        plot_image_with_points(
            raw_img,
            raw_matching_points,
            f"{output_dir}/raw_matching.png",
        )

    # Do PnP optimization to optimize the rotation between the 3D mesh keypoints and the 2D image keypoints
    pnp_started_s = time.perf_counter()
    mesh2raw_camera = registration_pnp(
        mesh_matching_points, raw_matching_points, intrinsic
    )
    timing_ms["pnp_scale_ms"] += elapsed_ms(pnp_started_s)

    if VIS:
        pnp_camera_pose = np.eye(4, dtype=np.float32)
        pnp_camera_pose[:3, :3] = np.linalg.inv(mesh2raw_camera[:3, :3])
        pnp_camera_pose[3, :3] = mesh2raw_camera[:3, 3]
        pnp_camera_pose[:, :2] = -pnp_camera_pose[:, :2]
        color, depth = render_image(
            mesh_path, pnp_camera_pose, raw_img.shape[1], raw_img.shape[0], fov, "cuda"
        )
        vis_mask = depth > 0
        color[0][~vis_mask] = raw_img[~vis_mask]
        plt.imsave(f"{output_dir}/pnp_results.png", color[0])

    # Transform the mesh into the real world coordinate
    observation_started_s = time.perf_counter()
    mesh_matching_points_cam = np.dot(
        mesh2raw_camera,
        np.hstack(
            (mesh_matching_points, np.ones((mesh_matching_points.shape[0], 1)))
        ).T,
    ).T
    mesh_matching_points_cam = mesh_matching_points_cam[:, :3]

    # Load the pcd in world coordinate of raw image matching points
    obs_points = []
    obs_colors = []
    pcd_path = f"{base_path}/{case_name}/pcd/0.npz"
    mask_path = f"{base_path}/{case_name}/mask/processed_masks.pkl"
    data = np.load(pcd_path)
    with open(mask_path, "rb") as f:
        processed_masks = pickle.load(f)
    camera_count = int(np.asarray(data["points"]).shape[0])
    if camera_count != len(processed_masks[0]):
        raise ValueError("pcd camera count does not match processed mask count")
    # Every camera count runs the original PhysTwin alignment flow:
    # keypoint ARAP, then ray-casting ARAP registration with the above-table
    # clamp. Single-camera warmup has no rigid-prior bypass.
    for i in range(camera_count):
        points = data["points"][i]
        colors = data["colors"][i]
        mask = processed_masks[0][i]["object"]
        obs_points.append(points[mask])
        obs_colors.append(colors[mask])
        if i == 0:
            first_points = points
            first_mask = mask

    obs_points = np.vstack(obs_points)
    obs_colors = np.vstack(obs_colors)

    # Find the cloest points for the raw_matching_points
    new_match, matching_points = select_point(
        first_points, raw_matching_points, first_mask
    )
    matching_points_cam = np.dot(
        w2c, np.hstack((matching_points, np.ones((matching_points.shape[0], 1)))).T
    ).T
    matching_points_cam = matching_points_cam[:, :3]

    if VIS:
        # Draw the raw_matching_points and new matching points on the masked
        vis_img = raw_img.copy()
        vis_img[~first_mask] = 0
        plot_image_with_points(
            vis_img,
            raw_matching_points,
            f"{output_dir}/raw_matching_valid.png",
            new_match,
        )

    timing_ms["observation_prepare_ms"] += elapsed_ms(observation_started_s)

    # Use the matching points in the camera coordinate to optimize the scame between the mesh and the observation
    scale_started_s = time.perf_counter()
    optimal_scale = registration_scale(mesh_matching_points_cam, matching_points_cam)
    timing_ms["pnp_scale_ms"] += elapsed_ms(scale_started_s)

    # Compute the rigid transformation from the original mesh to the final world coordinate
    observation_started_s = time.perf_counter()
    scale_matrix = np.eye(4) * optimal_scale
    scale_matrix[3, 3] = 1
    mesh2world = np.dot(c2w, np.dot(scale_matrix, mesh2raw_camera))

    mesh_matching_points_world = np.dot(
        mesh2world,
        np.hstack(
            (mesh_matching_points, np.ones((mesh_matching_points.shape[0], 1)))
        ).T,
    ).T
    mesh_matching_points_world = mesh_matching_points_world[:, :3]

    # Do the ARAP based on the matching keypoints
    # Convert the mesh to open3d to use the ARAP function
    initial_mesh_world = o3d.geometry.TriangleMesh()
    initial_mesh_world.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    initial_mesh_world.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    # Need to remove the duplicated vertices to enable open3d, however, the duplicated points are important in trimesh for texture
    initial_mesh_world = initial_mesh_world.remove_duplicated_vertices()
    # Get the index from original vertices to the mesh vertices, mapping between trimesh and open3d
    kdtree = KDTree(initial_mesh_world.vertices)
    _, trimesh_indices = kdtree.query(np.asarray(mesh.vertices))
    trimesh_indices = np.asarray(trimesh_indices, dtype=np.int32)
    initial_mesh_world.transform(mesh2world)
    timing_ms["observation_prepare_ms"] += elapsed_ms(observation_started_s)

    final_mesh_world = align_full_vendor_compatible(
        initial_mesh_world,
        mesh_matching_points_world,
        matching_points,
        obs_points,
        mesh,
        trimesh_indices,
        c2ws,
        w2cs,
        timing_ms=timing_ms,
    )

    if VIS:
        final_mesh_world.compute_vertex_normals()

        # Visualize the partial observation and the mesh
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obs_points)
        pcd.colors = o3d.utility.Vector3dVector(obs_colors)

        # Render the final stuffs as a turntable video
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        dummy_frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        height, width, _ = dummy_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        video_writer = cv2.VideoWriter(
            f"{output_dir}/final_matching.mp4", fourcc, 30, (width, height)
        )
        # final_mesh_world.compute_vertex_normals()
        # final_mesh_world.translate([0, 0, 0.2])
        # mesh_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(final_mesh_world)
        # o3d.visualization.draw_geometries([pcd, final_mesh_world], window_name="Matching")
        vis.add_geometry(pcd)
        vis.add_geometry(final_mesh_world)
        # vis.add_geometry(coordinate)
        view_control = vis.get_view_control()

        for j in range(360):
            view_control.rotate(10, 0)
            vis.poll_events()
            vis.update_renderer()
            frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            frame = (frame * 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)
        vis.destroy_window()

    mesh_export_started_s = time.perf_counter()
    mesh.vertices = np.asarray(final_mesh_world.vertices)[trimesh_indices]
    mesh.export(f"{output_dir}/final_mesh.glb")
    timing_ms["mesh_export_ms"] = elapsed_ms(mesh_export_started_s)
    timing_ms["post_go_ms"] = elapsed_ms(stage_started_s)
    run.write_completed()


if __name__ == "__main__":
    main()
