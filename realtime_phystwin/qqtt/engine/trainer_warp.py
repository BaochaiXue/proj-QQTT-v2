from qqtt.data import RealData, SimpleData
from qqtt.utils import logger, visualize_pc, cfg
from qqtt.model.diff_simulator import (
    SpringMassSystemWarp,
)
from qqtt.model.diff_simulator.spring_mass_warp_batched import (
    SpringMassSystemWarp as SpringMassSystemWarpBatched,
)
import open3d as o3d
import numpy as np
import torch
import wandb
import os
import shutil
from tqdm import tqdm
import warp as wp
from scipy.spatial import KDTree
import pickle
import cv2
from pynput import keyboard
import pyrender
import trimesh
import matplotlib.pyplot as plt

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.scene.cameras import Camera
from gaussian_splatting.gaussian_renderer import render as render_gaussian
from gaussian_splatting.dynamic_utils import (
    interpolate_motions_speedup,
    knn_weights,
    knn_weights_sparse,
    get_topk_indices,
    calc_weights_vals_from_indices,
)
from gaussian_splatting.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from gs_render import (
    remove_gaussians_with_low_opacity,
    remove_gaussians_with_point_mesh_distance,
)
from gaussian_splatting.rotation_utils import quaternion_multiply, matrix_to_quaternion

from sklearn.cluster import KMeans
import copy
import time
import threading
import time
import csv
import json


class InvPhyTrainerWarp:
    def __init__(
        self,
        data_path,
        base_dir,
        train_frame=None,
        mask_path=None,
        velocity_path=None,
        pure_inference_mode=False,
        device="cuda:0",
        dataset_override=None,
        batch_mode=False,
        batch_size=1,
        segment_len=10,
        segment_stride=10,
        batch_vis_per_instance=False,
        batch_vis_interval=50,
        batch_vis_num_instances=1,
        batch_vis_num_groups=1,
        rollout_prefix_switch=False,
        rollout_switch_start_iter=50,
        rollout_switch_ramp_iters=100,
        rollout_replace_thresh=0.03,
        rollout_baseline_iters=5,
        rollout_baseline_ratio=0.8,
        rollout_check_len=5,
        rollout_switch_log_interval=10,
        batch_loss_weighting=False,
        batch_loss_weight_min=0.5,
        batch_loss_weight_max=2.0,
        batch_loss_weight_eps=1e-8,
        batch_loss_weight_log_interval=10,
    ):
        cfg.data_path = data_path
        cfg.base_dir = base_dir
        cfg.device = device
        cfg.run_name = base_dir.split("/")[-1]
        cfg.train_frame = train_frame

        self.batch_mode = batch_mode
        self.batch_size = batch_size
        self.segment_len = segment_len
        self.segment_stride = segment_stride
        self.batch_simulator = None
        self.batch_size_loaded = None
        self.batch_segment_len_loaded = None
        self.batch_vis_per_instance = bool(batch_vis_per_instance)
        self.batch_vis_interval = max(1, int(batch_vis_interval))
        self.batch_vis_num_instances = int(batch_vis_num_instances)
        self.batch_vis_num_groups = max(1, int(batch_vis_num_groups))
        self.rollout_prefix_switch = bool(rollout_prefix_switch)
        self.rollout_switch_start_iter = int(rollout_switch_start_iter)
        self.rollout_switch_ramp_iters = max(1, int(rollout_switch_ramp_iters))
        self.rollout_replace_thresh = float(rollout_replace_thresh)
        self.rollout_baseline_iters = max(0, int(rollout_baseline_iters))
        self.rollout_baseline_ratio = float(rollout_baseline_ratio)
        self.rollout_check_len = max(0, int(rollout_check_len))
        self.rollout_switch_log_interval = max(1, int(rollout_switch_log_interval))
        self.batch_loss_weighting = bool(batch_loss_weighting)
        self.batch_loss_weight_min = float(batch_loss_weight_min)
        self.batch_loss_weight_max = float(batch_loss_weight_max)
        if self.batch_loss_weight_max < self.batch_loss_weight_min:
            raise ValueError("batch_loss_weight_max must be >= batch_loss_weight_min")
        self.batch_loss_weight_eps = float(batch_loss_weight_eps)
        self.batch_loss_weight_log_interval = max(1, int(batch_loss_weight_log_interval))
        self.window_loss_weights_by_global_id = None
        self.prev_rollout_pos_cache = None
        self.prev_rollout_vel_cache = None
        self.rollout_overlap_error_history = {}
        self._last_rollout_switch_count = None
        self.rest_debug_log_count = 0
        self.nonfinite_debug_log_count = 0

        self.init_masks = None
        self.init_velocities = None
        # Load the data
        if cfg.data_type == "real":
            self.dataset = (
                dataset_override
                if dataset_override is not None
                else RealData(visualize=False, save_gt=False)
            )
            self._load_real_dataset_attributes()
        elif cfg.data_type == "synthetic":
            self.dataset = SimpleData(visualize=False)
            self.use_asap_train_points = False
            self.object_points = self.dataset.data
            self.train_object_points = self.dataset.data
            self.object_colors = None
            self.object_visibilities = None
            self.object_motions_valid = None
            self.train_object_visibilities = None
            self.train_object_motions_valid = None
            self.controller_points = None
            self.asap_object_points_filled = None
            self.asap_surface_points = None
            self.asap_interior_points = None
            self.structure_points = self.dataset.data[0]
            self.num_original_points = None
            self.num_surface_points = None
            self.num_all_points = len(self.dataset.data[0])
            # Prepare for the multiple object case
            if mask_path is not None:
                mask = np.load(mask_path)
                self.init_masks = torch.tensor(
                    mask, dtype=torch.float32, device=cfg.device
                )
            if velocity_path is not None:
                velocity = np.load(velocity_path)
                self.init_velocities = torch.tensor(
                    velocity, dtype=torch.float32, device=cfg.device
                )
        else:
            raise ValueError(f"Data type {cfg.data_type} not supported")

        # Initialize the vertices, springs, rest lengths and masses
        if self.controller_points is None:
            firt_frame_controller_points = None
        else:
            firt_frame_controller_points = self.controller_points[0]
        (
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            self.num_object_springs,
        ) = self._init_start(
            self.structure_points,
            firt_frame_controller_points,
            object_radius=cfg.object_radius,
            object_max_neighbours=cfg.object_max_neighbours,
            controller_radius=cfg.controller_radius,
            controller_max_neighbours=cfg.controller_max_neighbours,
            mask=self.init_masks,
        )
        self._log_controller_spring_stats()

        self.simulator = SpringMassSystemWarp(
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            dt=cfg.dt,
            num_substeps=cfg.num_substeps,
            spring_Y=cfg.init_spring_Y,
            collide_elas=cfg.collide_elas,
            collide_fric=cfg.collide_fric,
            dashpot_damping=cfg.dashpot_damping,
            drag_damping=cfg.drag_damping,
            collide_object_elas=cfg.collide_object_elas,
            collide_object_fric=cfg.collide_object_fric,
            init_masks=self.init_masks,
            collision_dist=cfg.collision_dist,
            init_velocities=self.init_velocities,
            num_object_points=self.num_all_points,
            num_surface_points=self.num_surface_points,
            num_original_points=self.num_original_points,
            controller_points=self.controller_points,
            reverse_z=cfg.reverse_z,
            spring_Y_min=cfg.spring_Y_min,
            spring_Y_max=cfg.spring_Y_max,
            gt_object_points=self.train_object_points,
            gt_object_visibilities=self.train_object_visibilities,
            gt_object_motions_valid=self.train_object_motions_valid,
            self_collision=cfg.self_collision,
        )

        if not pure_inference_mode:
            self.optimizer = torch.optim.Adam(
                [
                    wp.to_torch(self.simulator.wp_spring_Y),
                    wp.to_torch(self.simulator.wp_collide_elas),
                    wp.to_torch(self.simulator.wp_collide_fric),
                    wp.to_torch(self.simulator.wp_collide_object_elas),
                    wp.to_torch(self.simulator.wp_collide_object_fric),
                ],
                lr=cfg.base_lr,
                betas=(0.9, 0.99),
            )

            if "debug" not in cfg.run_name:
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="final_pipeline",
                    name=cfg.run_name,
                    config=cfg.to_dict(),
                )
            else:
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="Debug",
                    name=cfg.run_name,
                    config=cfg.to_dict(),
                )
            if not os.path.exists(f"{cfg.base_dir}/train"):
                # Create directory if it doesn't exist
                os.makedirs(f"{cfg.base_dir}/train")

    def _load_real_dataset_attributes(self):
        # Keep these aliases in one place so online buffers can refresh them
        # after new chunks are appended.
        self.object_points = self.dataset.object_points
        self.train_object_points = self.dataset.object_points
        self.object_colors = self.dataset.object_colors
        self.object_visibilities = self.dataset.object_visibilities
        self.object_motions_valid = self.dataset.object_motions_valid
        self.train_object_visibilities = self.object_visibilities
        self.train_object_motions_valid = self.object_motions_valid
        self.controller_points = self.dataset.controller_points
        self.asap_object_points_filled = self.dataset.asap_object_points_filled
        self.asap_surface_points = self.dataset.asap_surface_points
        self.asap_interior_points = self.dataset.asap_interior_points
        self.structure_points = self.dataset.structure_points
        self.num_original_points = self.dataset.num_original_points
        self.num_surface_points = self.dataset.num_surface_points
        self.num_all_points = self.dataset.num_all_points
        self.source_frame_indices = getattr(self.dataset, "source_frame_indices", None)

    def refresh_real_data_from_dataset(self):
        if cfg.data_type != "real":
            return
        self._load_real_dataset_attributes()
        if hasattr(self, "simulator"):
            self.simulator.gt_object_points = self.train_object_points
            self.simulator.gt_object_visibilities = self.train_object_visibilities
            self.simulator.gt_object_motions_valid = self.train_object_motions_valid
            self.simulator.controller_points = self.controller_points

    def _log_controller_spring_stats(self):
        if self.controller_points is None:
            logger.info("[Controller-Springs]: no controller points")
            return

        num_ctrl = int(self.controller_points.shape[1])
        ctrl_springs = self.init_springs[self.num_object_springs :]
        logger.info(
            "[Controller-Springs]: "
            f"object_springs={self.num_object_springs}, "
            f"controller_springs={int(ctrl_springs.shape[0])}, "
            f"controller_points={num_ctrl}"
        )
        if ctrl_springs.shape[0] == 0:
            logger.warning("[Controller-Springs]: no controller-object springs found")
            return

        ctrl_idx = (ctrl_springs[:, 0] - self.num_all_points).long()
        valid = torch.logical_and(ctrl_idx >= 0, ctrl_idx < num_ctrl)
        if not bool(valid.all()):
            invalid_count = int((~valid).sum().item())
            logger.warning(
                f"[Controller-Springs]: found {invalid_count} springs whose first "
                "endpoint is not a valid controller point"
            )
            ctrl_idx = ctrl_idx[valid]

        degree = torch.bincount(ctrl_idx, minlength=num_ctrl).float()
        logger.info(
            "[Controller-Springs]: degree per controller point "
            f"min={degree.min().item():.0f}, max={degree.max().item():.0f}, "
            f"mean={degree.mean().item():.2f}, zero_count={(degree == 0).sum().item():.0f}"
        )

        if self.controller_points.shape[0] > 1:
            ctrl_disp = torch.linalg.norm(
                self.controller_points[1:] - self.controller_points[:-1], dim=-1
            )
            mean_motion = ctrl_disp.mean(dim=0)
            max_motion = ctrl_disp.max(dim=0).values
            logger.info(
                "[Controller-Motion]: per point displacement "
                f"mean(min/mean/max)="
                f"{mean_motion.min().item():.5f}/"
                f"{mean_motion.mean().item():.5f}/"
                f"{mean_motion.max().item():.5f}, "
                f"max(min/mean/max)="
                f"{max_motion.min().item():.5f}/"
                f"{max_motion.mean().item():.5f}/"
                f"{max_motion.max().item():.5f}"
            )

        if num_ctrl < 2:
            return

        try:
            first_ctrl = self.controller_points[0].detach().cpu().numpy()
            labels = KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(
                first_ctrl
            )
        except Exception as exc:
            logger.warning(f"[Controller-Springs]: kmeans split failed: {exc}")
            return

        for cluster_id in range(2):
            mask_np = labels == cluster_id
            if not np.any(mask_np):
                continue
            mask = torch.from_numpy(mask_np).to(device=degree.device)
            cluster_degree = degree[mask]
            center = first_ctrl[mask_np].mean(axis=0)
            msg = (
                f"[Controller-Cluster {cluster_id}]: points={int(mask.sum().item())}, "
                f"center=({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}), "
                f"degree_sum={cluster_degree.sum().item():.0f}, "
                f"degree_mean={cluster_degree.mean().item():.2f}, "
                f"degree_min={cluster_degree.min().item():.0f}, "
                f"degree_max={cluster_degree.max().item():.0f}"
            )
            if self.controller_points.shape[0] > 1:
                cluster_motion = mean_motion[mask]
                cluster_max_motion = max_motion[mask]
                msg += (
                    f", motion_mean={cluster_motion.mean().item():.5f}, "
                    f"motion_max={cluster_max_motion.max().item():.5f}"
                )
            logger.info(msg)

    def _init_start(
        self,
        object_points,
        controller_points,
        object_radius=0.02,
        object_max_neighbours=30,
        controller_radius=0.04,
        controller_max_neighbours=50,
        mask=None,
    ):
        object_points = object_points.cpu().numpy()
        if controller_points is not None:
            controller_points = controller_points.cpu().numpy()
        if mask is None:
            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(object_points)
            pcd_tree = o3d.geometry.KDTreeFlann(object_pcd)

            # Connect the springs of the objects first
            points = np.asarray(object_pcd.points)
            spring_flags = np.zeros((len(points), len(points)))
            springs = []
            rest_lengths = []
            for i in range(len(points)):
                [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                    points[i], object_radius, object_max_neighbours
                )
                idx = idx[1:]
                for j in idx:
                    rest_length = np.linalg.norm(points[i] - points[j])
                    if (
                        spring_flags[i, j] == 0
                        and spring_flags[j, i] == 0
                        and rest_length > 1e-4
                    ):
                        spring_flags[i, j] = 1
                        spring_flags[j, i] = 1
                        springs.append([i, j])
                        rest_lengths.append(np.linalg.norm(points[i] - points[j]))

            num_object_springs = len(springs)

            if controller_points is not None:
                # Connect the springs between the controller points and the object points
                num_object_points = len(points)
                points = np.concatenate([points, controller_points], axis=0)
                for i in range(len(controller_points)):
                    [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                        controller_points[i],
                        controller_radius,
                        controller_max_neighbours,
                    )
                    for j in idx:
                        springs.append([num_object_points + i, j])
                        rest_lengths.append(
                            np.linalg.norm(controller_points[i] - points[j])
                        )

            springs = np.array(springs)
            rest_lengths = np.array(rest_lengths)
            masses = np.ones(len(points))
            return (
                torch.tensor(points, dtype=torch.float32, device=cfg.device),
                torch.tensor(springs, dtype=torch.int32, device=cfg.device),
                torch.tensor(rest_lengths, dtype=torch.float32, device=cfg.device),
                torch.tensor(masses, dtype=torch.float32, device=cfg.device),
                num_object_springs,
            )
        else:
            mask = mask.cpu().numpy()
            # Get the unique value in masks
            unique_values = np.unique(mask)
            vertices = []
            springs = []
            rest_lengths = []
            index = 0
            # Loop different objects to connect the springs separately
            for value in unique_values:
                temp_points = object_points[mask == value]
                temp_pcd = o3d.geometry.PointCloud()
                temp_pcd.points = o3d.utility.Vector3dVector(temp_points)
                temp_tree = o3d.geometry.KDTreeFlann(temp_pcd)
                temp_spring_flags = np.zeros((len(temp_points), len(temp_points)))
                temp_springs = []
                temp_rest_lengths = []
                for i in range(len(temp_points)):
                    [k, idx, _] = temp_tree.search_hybrid_vector_3d(
                        temp_points[i], object_radius, object_max_neighbours
                    )
                    idx = idx[1:]
                    for j in idx:
                        rest_length = np.linalg.norm(temp_points[i] - temp_points[j])
                        if (
                            temp_spring_flags[i, j] == 0
                            and temp_spring_flags[j, i] == 0
                            and rest_length > 1e-4
                        ):
                            temp_spring_flags[i, j] = 1
                            temp_spring_flags[j, i] = 1
                            temp_springs.append([i + index, j + index])
                            temp_rest_lengths.append(rest_length)
                vertices += temp_points.tolist()
                springs += temp_springs
                rest_lengths += temp_rest_lengths
                index += len(temp_points)

            num_object_springs = len(springs)

            vertices = np.array(vertices)
            springs = np.array(springs)
            rest_lengths = np.array(rest_lengths)
            masses = np.ones(len(vertices))

            return (
                torch.tensor(vertices, dtype=torch.float32, device=cfg.device),
                torch.tensor(springs, dtype=torch.int32, device=cfg.device),
                torch.tensor(rest_lengths, dtype=torch.float32, device=cfg.device),
                torch.tensor(masses, dtype=torch.float32, device=cfg.device),
                num_object_springs,
            )

    def _tile_time_tensor(self, tensor, batch_size):
        if tensor is None:
            return None
        assert tensor.dim() >= 2, "Expected tensor with shape [T, N, ...]"
        repeats = [1, batch_size] + [1] * (tensor.dim() - 1)
        view_shape = [tensor.shape[0], batch_size * tensor.shape[1], *tensor.shape[2:]]
        return tensor.unsqueeze(1).repeat(*repeats).reshape(*view_shape).contiguous()

    def _slice_time_with_padding(self, tensor, start_idx, length):
        start_idx = int(start_idx)
        length = int(length)
        end_idx = start_idx + length
        sliced = tensor[start_idx : min(end_idx, tensor.shape[0])]
        if sliced.shape[0] == length:
            return sliced.contiguous()
        if sliced.shape[0] == 0:
            raise ValueError(
                f"Cannot slice tensor from start_idx={start_idx}; tensor has "
                f"{tensor.shape[0]} frames"
            )
        pad_count = length - sliced.shape[0]
        pad_shape = [pad_count, *sliced.shape[1:]]
        pad = sliced[-1:].expand(*pad_shape)
        return torch.cat([sliced, pad], dim=0).contiguous()

    def _compute_segment_start_indices(self, total_frames):
        seg_len = int(self.segment_len)
        seg_stride = int(self.segment_stride)
        if seg_len <= 1:
            raise ValueError("segment_len must be >= 2 for rollout training")
        if seg_stride <= 0:
            raise ValueError("segment_stride must be positive")
        if total_frames < seg_len:
            raise ValueError(
                f"total_frames ({total_frames}) must be >= segment_len ({seg_len})"
            )
        return list(range(0, total_frames - seg_len + 1, seg_stride))

    def _get_trainable_total_frames(self):
        if cfg.train_frame is None:
            total_frames = int(self.train_object_points.shape[0])
        else:
            total_frames = int(min(cfg.train_frame, self.train_object_points.shape[0]))
        if self.controller_points is not None:
            total_frames = min(total_frames, int(self.controller_points.shape[0]))
        if self.train_object_visibilities is not None:
            total_frames = min(total_frames, int(self.train_object_visibilities.shape[0]))
        if self.train_object_motions_valid is not None:
            total_frames = min(total_frames, int(self.train_object_motions_valid.shape[0]))
        if self.asap_surface_points is not None:
            total_frames = min(total_frames, int(self.asap_surface_points.shape[0]))
        if self.asap_interior_points is not None:
            total_frames = min(total_frames, int(self.asap_interior_points.shape[0]))
        return total_frames

    def _estimate_segment_init_velocity(self, start_idx, object_vertices_start):
        start_idx = int(start_idx)
        prev_idx = max(0, start_idx - 4)
        steps = start_idx - prev_idx
        velocities = torch.zeros_like(object_vertices_start)
        if steps <= 0:
            return velocities

        frame_dt = float(cfg.dt) * float(cfg.num_substeps)
        num_gt_points = int(self.num_original_points)
        num_surface_extra = int(self.num_surface_points) - num_gt_points

        start_pos = self._get_reset_object_points(start_idx)
        prev_pos = self._get_reset_object_points(prev_idx)
        object_velocity = (start_pos - prev_pos) / (steps * frame_dt)
        object_velocity = torch.nan_to_num(
            object_velocity, nan=0.0, posinf=0.0, neginf=0.0
        )
        velocities[:num_gt_points] = object_velocity

        if num_surface_extra > 0:
            velocities[num_gt_points : num_gt_points + num_surface_extra] = (
                self.asap_surface_points[start_idx] - self.asap_surface_points[prev_idx]
            ) / (steps * frame_dt)

        if self.asap_interior_points is not None and self.asap_interior_points.shape[1] > 0:
            velocities[self.num_surface_points : self.num_all_points] = (
                self.asap_interior_points[start_idx]
                - self.asap_interior_points[prev_idx]
            ) / (steps * frame_dt)

        return velocities

    def _get_valid_object_mask(self, frame_idx):
        num_gt_points = int(self.num_original_points)
        valid = torch.isfinite(
            self.train_object_points[frame_idx, :num_gt_points]
        ).all(dim=-1)
        if self.train_object_visibilities is not None:
            valid = valid & self.train_object_visibilities[frame_idx, :num_gt_points]
        if self.train_object_motions_valid is not None:
            valid = valid & self.train_object_motions_valid[frame_idx, :num_gt_points]
        return valid

    def _get_reset_object_points(self, frame_idx):
        num_gt_points = int(self.num_original_points)
        reset_points = self.train_object_points[frame_idx, :num_gt_points].clone()
        reset_points = torch.nan_to_num(reset_points, nan=0.0, posinf=0.0, neginf=0.0)
        return reset_points

    def _compute_segment_rest_lengths(self, object_vertices, controller_vertices=None):
        if controller_vertices is not None:
            self._log_controller_rest_ratio(object_vertices, controller_vertices)
            vertices = torch.cat([object_vertices, controller_vertices], dim=0)
        else:
            vertices = object_vertices

        springs = self.init_springs.long()
        p0 = vertices[springs[:, 0]]
        p1 = vertices[springs[:, 1]]
        return torch.clamp(torch.linalg.norm(p0 - p1, dim=-1), min=1e-4).contiguous()

    def _log_controller_rest_ratio(self, object_vertices, controller_vertices):
        if self.rest_debug_log_count >= 20:
            return
        ctrl_springs = self.init_springs[self.num_object_springs :].long()
        if ctrl_springs.numel() == 0:
            return

        vertices = torch.cat([object_vertices, controller_vertices], dim=0)
        p0 = vertices[ctrl_springs[:, 0]]
        p1 = vertices[ctrl_springs[:, 1]]
        current_dist = torch.linalg.norm(p0 - p1, dim=-1)
        base_rest = torch.clamp(
            self.init_rest_lengths[self.num_object_springs :], min=1e-8
        )
        ratio = current_dist / base_rest
        finite = torch.isfinite(ratio)
        should_log = (
            self.rest_debug_log_count < 3
            or not bool(finite.all())
            or ratio[finite].max().item() > 5.0
            or ratio[finite].min().item() < 0.2
        )
        if not should_log:
            return

        if bool(finite.any()):
            finite_ratio = ratio[finite]
            finite_dist = current_dist[finite]
            logger.warning(
                "[Train-Batch-Rest]: controller/object rest ratio "
                f"min/mean/max="
                f"{finite_ratio.min().item():.4f}/"
                f"{finite_ratio.mean().item():.4f}/"
                f"{finite_ratio.max().item():.4f}, "
                f"distance min/mean/max="
                f"{finite_dist.min().item():.6f}/"
                f"{finite_dist.mean().item():.6f}/"
                f"{finite_dist.max().item():.6f}, "
                f"nonfinite={(~finite).sum().item()}"
            )
        else:
            logger.warning("[Train-Batch-Rest]: all controller/object rest ratios are nonfinite")
        self.rest_debug_log_count += 1

    def _compute_batched_segment_rest_lengths(
        self, object_vertices_batched, controller_vertices_batched=None
    ):
        rest_lengths = []
        for batch_idx in range(object_vertices_batched.shape[0]):
            controller_vertices = (
                controller_vertices_batched[batch_idx]
                if controller_vertices_batched is not None
                else None
            )
            rest_lengths.append(
                self._compute_segment_rest_lengths(
                    object_vertices_batched[batch_idx], controller_vertices
                )
            )
        return torch.cat(rest_lengths, dim=0).contiguous()

    def _visible_chamfer_error(self, pred_vertices, frame_idx):
        if cfg.data_type != "real":
            return None

        num_surface_points = int(self.num_surface_points)
        gt_points = self.train_object_points[frame_idx]
        if self.train_object_visibilities is not None:
            mask = self.train_object_visibilities[frame_idx]
        else:
            mask = torch.isfinite(gt_points).all(dim=-1)
        gt_visible = gt_points[mask]
        if gt_visible.shape[0] == 0:
            return None

        pred_surface = pred_vertices[:num_surface_points]
        finite_pred = torch.isfinite(pred_surface).all(dim=-1)
        pred_surface = pred_surface[finite_pred]
        if pred_surface.shape[0] == 0:
            return None

        distances = torch.cdist(
            gt_visible.unsqueeze(0),
            pred_surface.unsqueeze(0),
            p=1,
        )
        return distances.min(dim=2).values.mean()

    def _overlap_chamfer_error(self, window_idx, segment_starts, segment_len):
        if (
            self.prev_rollout_pos_cache is None
            or window_idx <= 0
            or window_idx >= len(segment_starts)
        ):
            return None

        prev_idx = window_idx - 1
        prev_start = int(segment_starts[prev_idx])
        cur_start = int(segment_starts[window_idx])
        overlap_start = cur_start
        prev_cache_len = int(self.prev_rollout_pos_cache.shape[1])
        overlap_end = min(
            prev_start + prev_cache_len,
            cur_start + int(segment_len),
        )
        total_frames = self._get_trainable_total_frames()
        overlap_end = min(overlap_end, total_frames)

        errors = []
        for frame_idx in range(overlap_start, overlap_end):
            local_prev = frame_idx - prev_start
            if local_prev < 0 or local_prev >= self.prev_rollout_pos_cache.shape[1]:
                continue
            pred_vertices = self.prev_rollout_pos_cache[prev_idx, local_prev]
            error = self._visible_chamfer_error(pred_vertices, frame_idx)
            if error is not None and bool(torch.isfinite(error)):
                errors.append(error)

        if len(errors) == 0:
            return None
        return torch.stack(errors).mean()

    def _record_rollout_baseline_errors(self, segment_starts, segment_len):
        if (
            not self.rollout_prefix_switch
            or self.rollout_baseline_iters <= 0
            or self.prev_rollout_pos_cache is None
        ):
            return

        for window_idx in range(1, len(segment_starts)):
            history = self.rollout_overlap_error_history.setdefault(window_idx, [])
            if len(history) >= self.rollout_baseline_iters:
                continue

            error = self._overlap_chamfer_error(window_idx, segment_starts, segment_len)
            if error is None or not bool(torch.isfinite(error)):
                continue

            history.append(float(error.detach().cpu().item()))

    def _rollout_threshold_for_window(self, window_idx):
        if self.rollout_baseline_iters <= 0:
            return {
                "threshold": self.rollout_replace_thresh,
                "baseline": None,
                "count": 0,
                "ready": True,
                "mode": "fixed",
            }

        history = self.rollout_overlap_error_history.get(window_idx, [])
        count = len(history)
        if count < self.rollout_baseline_iters:
            return {
                "threshold": None,
                "baseline": None,
                "count": count,
                "ready": False,
                "mode": "baseline",
            }

        baseline = float(np.median(np.asarray(history, dtype=np.float64)))
        return {
            "threshold": baseline * self.rollout_baseline_ratio,
            "baseline": baseline,
            "count": count,
            "ready": True,
            "mode": "baseline",
        }

    def _compute_rollout_switch_info(self, iteration, segment_starts, segment_len):
        num_windows = len(segment_starts)
        if not self.rollout_prefix_switch:
            return {
                "num_windows": num_windows,
                "num_forced_windows": 0,
                "base_prefix": 0,
                "num_rollout_windows": 0,
                "close_windows": [],
                "stop_window": None,
                "stop_error": None,
                "stop_threshold": None,
                "stop_baseline": None,
                "stop_baseline_count": None,
                "stop_reason": None,
            }

        self._record_rollout_baseline_errors(segment_starts, segment_len)

        progress = (int(iteration) - self.rollout_switch_start_iter) / float(
            self.rollout_switch_ramp_iters
        )
        progress = max(0.0, min(1.0, progress))
        num_forced = int(progress * num_windows)
        num_forced = max(0, min(num_windows, num_forced))

        # Window 0 is the frame-0 base state. It does not need previous-window cache,
        # but it anchors prefix extension to window 1.
        base_prefix = 1 if num_windows > 0 else 0
        num_rollout = max(num_forced, base_prefix)
        num_rollout = min(num_rollout, num_windows)

        close_windows = []
        stop_window = None
        stop_error = None
        stop_threshold = None
        stop_baseline = None
        stop_baseline_count = None
        stop_reason = None
        if self.prev_rollout_pos_cache is None:
            stop_window = int(num_rollout) if num_rollout < num_windows else None
            stop_reason = "no previous rollout cache"
        while num_rollout < num_windows and self.prev_rollout_pos_cache is not None:
            error = self._overlap_chamfer_error(
                num_rollout, segment_starts, segment_len
            )
            if error is None:
                stop_window = int(num_rollout)
                stop_reason = "no valid overlap chamfer"
                break

            error_value = float(error.detach().cpu().item())
            threshold_info = self._rollout_threshold_for_window(num_rollout)
            threshold = threshold_info["threshold"]
            if threshold is None:
                stop_window = int(num_rollout)
                stop_error = error_value
                stop_baseline_count = threshold_info["count"]
                stop_reason = (
                    "baseline collecting "
                    f"{threshold_info['count']}/{self.rollout_baseline_iters}"
                )
                break

            if error_value < threshold:
                close_windows.append(
                    {
                        "window_idx": int(num_rollout),
                        "start_idx": int(segment_starts[num_rollout]),
                        "error": error_value,
                        "threshold": float(threshold),
                        "baseline": threshold_info["baseline"],
                        "baseline_count": threshold_info["count"],
                        "threshold_mode": threshold_info["mode"],
                    }
                )
                num_rollout += 1
            else:
                stop_window = int(num_rollout)
                stop_error = error_value
                stop_threshold = float(threshold)
                stop_baseline = threshold_info["baseline"]
                stop_baseline_count = threshold_info["count"]
                stop_reason = "over threshold"
                break

        return {
            "num_windows": num_windows,
            "num_forced_windows": num_forced,
            "base_prefix": base_prefix,
            "num_rollout_windows": num_rollout,
            "close_windows": close_windows,
            "stop_window": stop_window,
            "stop_error": stop_error,
            "stop_threshold": stop_threshold,
            "stop_baseline": stop_baseline,
            "stop_baseline_count": stop_baseline_count,
            "stop_reason": stop_reason,
        }

    def _log_rollout_switch_info(self, iteration, switch_info):
        if not self.rollout_prefix_switch:
            return

        rollout_count = int(switch_info["num_rollout_windows"])
        should_log = (
            iteration % self.rollout_switch_log_interval == 0
            or self._last_rollout_switch_count != rollout_count
        )
        if not should_log:
            return

        forced = int(switch_info["num_forced_windows"])
        base_prefix = int(switch_info["base_prefix"])
        close_extra = max(0, rollout_count - max(forced, base_prefix))
        logger.info(
            "[Train-Batched-Switch]: "
            f"iter={iteration}, rollout_windows={rollout_count}/"
            f"{switch_info['num_windows']}, forced={forced}, "
            f"base_prefix={base_prefix}, close_extra={close_extra}, "
            f"baseline_iters={self.rollout_baseline_iters}, "
            f"baseline_ratio={self.rollout_baseline_ratio:.3f}"
        )

        for close_info in switch_info["close_windows"]:
            baseline = close_info["baseline"]
            baseline_text = "none" if baseline is None else f"{baseline:.6f}"
            logger.info(
                "[Train-Batched-Switch]: close "
                f"window={close_info['window_idx']}, "
                f"start={close_info['start_idx']}, "
                f"overlap_chamfer={close_info['error']:.6f}, "
                f"baseline={baseline_text}, "
                f"baseline_count={close_info['baseline_count']}, "
                f"thresh={close_info['threshold']:.6f}, "
                f"mode={close_info['threshold_mode']}"
            )

        if switch_info["stop_window"] is not None:
            stop_msg = (
                f"[Train-Batched-Switch]: stop window={switch_info['stop_window']}"
            )
            if switch_info["stop_error"] is not None:
                stop_msg += (
                    f", overlap_chamfer={switch_info['stop_error']:.6f}"
                )
            if switch_info.get("stop_threshold") is not None:
                stop_msg += f", thresh={switch_info['stop_threshold']:.6f}"
            if switch_info.get("stop_baseline") is not None:
                stop_msg += f", baseline={switch_info['stop_baseline']:.6f}"
            if switch_info.get("stop_baseline_count") is not None:
                stop_msg += f", baseline_count={switch_info['stop_baseline_count']}"
            if switch_info.get("stop_reason") is not None:
                stop_msg += f", reason={switch_info['stop_reason']}"
            logger.info(stop_msg)

        self._last_rollout_switch_count = rollout_count

    def _init_window_loss_weights(self, num_windows):
        if not self.batch_loss_weighting:
            self.window_loss_weights_by_global_id = None
            return
        self.window_loss_weights_by_global_id = np.ones(
            int(num_windows), dtype=np.float32
        )

    def _get_batch_loss_weights(self, batch_data):
        batch_size = int(batch_data["batch_size"])
        if (
            not self.batch_loss_weighting
            or self.window_loss_weights_by_global_id is None
        ):
            return torch.ones(batch_size, dtype=torch.float32, device=cfg.device)

        offset = int(batch_data["global_window_offset"])
        weights = np.ones(batch_size, dtype=np.float32)
        for inst_local_idx in range(batch_size):
            global_window_id = offset + inst_local_idx
            if global_window_id < len(self.window_loss_weights_by_global_id):
                weights[inst_local_idx] = self.window_loss_weights_by_global_id[
                    global_window_id
                ]

        return torch.tensor(weights, dtype=torch.float32, device=cfg.device)

    def _update_window_loss_weights(
        self, iteration, segment_batches, per_inst_total, per_inst_steps
    ):
        if not self.batch_loss_weighting:
            return

        num_windows = sum(len(batch_data["starts"]) for batch_data in segment_batches)
        errors = np.full(num_windows, np.nan, dtype=np.float64)
        for group_idx, batch_data in enumerate(segment_batches):
            offset = int(batch_data["global_window_offset"])
            for inst_local_idx, _start_idx in enumerate(batch_data["starts"]):
                steps = int(per_inst_steps[group_idx, inst_local_idx])
                if steps <= 0:
                    continue
                global_window_id = offset + inst_local_idx
                errors[global_window_id] = (
                    per_inst_total[group_idx, inst_local_idx] / float(steps)
                )

        valid = np.isfinite(errors) & (errors > 0.0)
        if not bool(valid.any()):
            logger.warning(
                "[Train-Batched-Weight]: no valid per-window errors; keep previous weights"
            )
            return

        mean_error = float(errors[valid].mean())
        if mean_error <= self.batch_loss_weight_eps:
            self.window_loss_weights_by_global_id = np.ones(
                num_windows, dtype=np.float32
            )
            return

        weights = np.ones(num_windows, dtype=np.float64)
        weights[valid] = errors[valid] / mean_error
        weights[valid] = np.clip(
            weights[valid], self.batch_loss_weight_min, self.batch_loss_weight_max
        )
        mean_weight = float(weights[valid].mean())
        if mean_weight > self.batch_loss_weight_eps:
            weights[valid] = weights[valid] / mean_weight
            weights[valid] = np.clip(
                weights[valid], self.batch_loss_weight_min, self.batch_loss_weight_max
            )

        self.window_loss_weights_by_global_id = weights.astype(np.float32)

        should_log = (
            iteration % self.batch_loss_weight_log_interval == 0
            or iteration == 0
        )
        if should_log:
            valid_weights = self.window_loss_weights_by_global_id[valid]
            logger.info(
                "[Train-Batched-Weight]: "
                f"iter={iteration}, error min/mean/max="
                f"{errors[valid].min():.6e}/{mean_error:.6e}/"
                f"{errors[valid].max():.6e}, weight min/mean/max="
                f"{valid_weights.min():.4f}/{valid_weights.mean():.4f}/"
                f"{valid_weights.max():.4f}, clamp="
                f"{self.batch_loss_weight_min:.3f}-{self.batch_loss_weight_max:.3f}"
            )
            if num_windows <= 20:
                logger.info(
                    "[Train-Batched-Weight]: "
                    f"iter={iteration}, weights="
                    f"{self.window_loss_weights_by_global_id.tolist()}"
                )

    def _prepare_timing_csv(self, start_epoch):
        timing_dir = f"{cfg.base_dir}/train/timing"
        os.makedirs(timing_dir, exist_ok=True)
        timing_csv_path = f"{timing_dir}/train_timing.csv"
        if start_epoch < 0 and os.path.exists(timing_csv_path):
            os.remove(timing_csv_path)
        if not os.path.exists(timing_csv_path):
            with open(timing_csv_path, "w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(
                    [
                        "case_name",
                        "mode",
                        "iteration",
                        "train_iter_sec",
                        "full_rollout_eval_sec",
                        "total_iter_sec",
                        "num_groups",
                        "batch_size",
                        "segment_len",
                        "segment_stride",
                        "num_train_steps",
                    ]
                )
        return timing_csv_path

    def _append_timing_row(
        self,
        timing_csv_path,
        mode,
        iteration,
        train_iter_sec,
        full_rollout_eval_sec,
        total_iter_sec,
        num_groups,
        batch_size,
        segment_len,
        segment_stride,
        num_train_steps,
    ):
        with open(timing_csv_path, "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    cfg.run_name,
                    mode,
                    int(iteration),
                    float(train_iter_sec),
                    float(full_rollout_eval_sec),
                    float(total_iter_sec),
                    int(num_groups),
                    int(batch_size),
                    int(segment_len),
                    int(segment_stride),
                    int(num_train_steps),
                ]
            )

    def _build_segment_batch_tensors(
        self,
        start_indices,
        global_window_offset=0,
        num_rollout_windows=0,
    ):
        if len(start_indices) == 0:
            raise ValueError("start_indices cannot be empty")

        B = len(start_indices)
        seg_len = int(self.segment_len)
        rollout_cache_len = seg_len + int(self.rollout_check_len)
        device = cfg.device
        num_gt_points = (
            int(self.num_original_points)
            if self.num_original_points is not None
            else int(self.train_object_points.shape[1])
        )
        num_surface_extra = int(self.num_surface_points) - num_gt_points

        obj_segments = [
            self.train_object_points[s : s + seg_len].contiguous() for s in start_indices
        ]
        obj_stack = torch.stack(obj_segments, dim=0)  # [B, T, N_orig, 3]
        batched_object_points = (
            obj_stack.permute(1, 0, 2, 3)
            .contiguous()
            .reshape(seg_len, B * num_gt_points, 3)
        )

        batched_controller_points = None
        if self.controller_points is not None:
            ctrl_segments = [
                self._slice_time_with_padding(
                    self.controller_points, s, rollout_cache_len
                )
                for s in start_indices
            ]
            ctrl_stack = torch.stack(ctrl_segments, dim=0)  # [B, T, N_ctrl, 3]
            batched_controller_points = (
                ctrl_stack.permute(1, 0, 2, 3)
                .contiguous()
                .reshape(rollout_cache_len, B * ctrl_stack.shape[2], 3)
            )

        batched_object_visibilities = None
        if self.train_object_visibilities is not None:
            vis_segments = [
                self.train_object_visibilities[s : s + seg_len].contiguous()
                for s in start_indices
            ]
            vis_stack = torch.stack(vis_segments, dim=0)  # [B, T, N_orig]
            batched_object_visibilities = (
                vis_stack.permute(1, 0, 2)
                .contiguous()
                .reshape(seg_len, B * num_gt_points)
            )

        batched_object_motions_valid = None
        if self.train_object_motions_valid is not None:
            mot_segments = [
                self.train_object_motions_valid[s : s + seg_len].contiguous()
                for s in start_indices
            ]
            mot_stack = torch.stack(mot_segments, dim=0)  # [B, T, N_orig]
            batched_object_motions_valid = (
                mot_stack.permute(1, 0, 2)
                .contiguous()
                .reshape(seg_len, B * num_gt_points)
            )

        init_object_vertices = []
        init_object_velocities = []
        init_rest_lengths = []
        rollout_mode_mask = []
        cache_offset = int(self.segment_stride)
        # A merged prefix should behave like one continuous rollout: x/v are taken
        # from the previous rollout cache, while rest lengths stay fixed to the
        # prefix root instead of being recomputed at each window boundary.
        rollout_chain_rest_lengths = self.init_rest_lengths.detach().clone()
        for batch_idx, start_idx in enumerate(start_indices):
            start_idx = int(start_idx)
            global_window_id = int(global_window_offset) + batch_idx
            rollout_mode = (
                self.rollout_prefix_switch
                and global_window_id < int(num_rollout_windows)
            )
            use_rollout_cache = (
                rollout_mode
                and global_window_id > 0
                and self.prev_rollout_pos_cache is not None
                and self.prev_rollout_vel_cache is not None
            )

            if use_rollout_cache:
                if cache_offset >= self.prev_rollout_pos_cache.shape[1]:
                    raise ValueError(
                        "segment_stride must be smaller than cached rollout length when "
                        "rollout prefix switching is enabled"
                    )
                vertices = self.prev_rollout_pos_cache[
                    global_window_id - 1, cache_offset
                ].detach().clone()
                velocities = self.prev_rollout_vel_cache[
                    global_window_id - 1, cache_offset
                ].detach().clone()
                rest_lengths = rollout_chain_rest_lengths.clone()
            else:
                vertices = self.init_vertices[: self.num_all_points].clone()
                vertices[:num_gt_points] = self._get_reset_object_points(start_idx)
                if num_surface_extra > 0:
                    vertices[
                        num_gt_points : num_gt_points + num_surface_extra
                    ] = self.asap_surface_points[start_idx]
                if (
                    self.asap_interior_points is not None
                    and self.asap_interior_points.shape[1] > 0
                ):
                    vertices[self.num_surface_points : self.num_all_points] = (
                        self.asap_interior_points[start_idx]
                    )
                velocities = self._estimate_segment_init_velocity(start_idx, vertices)
                if rollout_mode and global_window_id == 0:
                    rest_lengths = rollout_chain_rest_lengths.clone()
                else:
                    controller_vertices = (
                        self.controller_points[start_idx]
                        if self.controller_points is not None
                        else None
                    )
                    rest_lengths = self._compute_segment_rest_lengths(
                        vertices, controller_vertices
                    )

            if vertices.shape[0] != self.num_all_points:
                raise ValueError(
                    f"Expected {self.num_all_points} rollout vertices, "
                    f"got {vertices.shape[0]}"
                )
            init_object_vertices.append(vertices)
            init_object_velocities.append(velocities)
            init_rest_lengths.append(rest_lengths)
            rollout_mode_mask.append(bool(rollout_mode))

        init_object_vertices_by_batch = torch.stack(init_object_vertices, dim=0)
        init_object_velocities_by_batch = torch.stack(init_object_velocities, dim=0)
        init_object_vertices_batched = init_object_vertices_by_batch.reshape(
            B * self.num_all_points, 3
        )
        init_object_velocities_batched = init_object_velocities_by_batch.reshape(
            B * self.num_all_points, 3
        )

        init_rest_lengths_batched = torch.cat(init_rest_lengths, dim=0).contiguous()

        if batched_controller_points is not None:
            ctrl_start = batched_controller_points[0]
            init_vertices_batched = torch.cat(
                [init_object_vertices_batched, ctrl_start], dim=0
            ).contiguous()
        else:
            init_vertices_batched = init_object_vertices_batched.contiguous()

        return {
            "batch_size": B,
            "segment_len": seg_len,
            "rollout_cache_len": rollout_cache_len,
            "starts": start_indices,
            "gt_object_points": batched_object_points.contiguous(),
            "controller_points": (
                batched_controller_points.contiguous()
                if batched_controller_points is not None
                else None
            ),
            "gt_object_visibilities": (
                batched_object_visibilities.contiguous()
                if batched_object_visibilities is not None
                else None
            ),
            "gt_object_motions_valid": (
                batched_object_motions_valid.contiguous()
                if batched_object_motions_valid is not None
                else None
            ),
            "init_vertices_batched": init_vertices_batched,
            "init_object_vertices_batched": init_object_vertices_batched,
            "init_object_vertices_by_batch": init_object_vertices_by_batch,
            "init_object_velocities_by_batch": init_object_velocities_by_batch,
            "init_velocities_batched": init_object_velocities_batched,
            "init_rest_lengths_batched": init_rest_lengths_batched,
            "rollout_mode_mask": rollout_mode_mask,
            "global_window_offset": int(global_window_offset),
        }

    def _build_single_segment_tensors(self, start_idx, segment_len):
        total_frames = self._get_trainable_total_frames()
        if start_idx < 0:
            raise ValueError(f"start_idx must be non-negative, got {start_idx}")
        if start_idx + segment_len > total_frames:
            raise ValueError(
                f"Segment [{start_idx}, {start_idx + segment_len}) exceeds "
                f"available frames {total_frames}"
            )

        num_gt_points = (
            int(self.num_original_points)
            if self.num_original_points is not None
            else int(self.train_object_points.shape[1])
        )
        num_surface_extra = int(self.num_surface_points) - num_gt_points
        object_points = self.train_object_points[
            start_idx : start_idx + segment_len
        ].contiguous()
        controller_points = (
            self.controller_points[start_idx : start_idx + segment_len].contiguous()
            if self.controller_points is not None
            else None
        )

        init_object_vertices = self.init_vertices[: self.num_all_points].clone()
        init_object_vertices[:num_gt_points] = self._get_reset_object_points(start_idx)
        if num_surface_extra > 0:
            init_object_vertices[
                num_gt_points : num_gt_points + num_surface_extra
            ] = self.asap_surface_points[start_idx]
        if self.asap_interior_points is not None and self.asap_interior_points.shape[1] > 0:
            init_object_vertices[self.num_surface_points : self.num_all_points] = (
                self.asap_interior_points[start_idx]
            )

        init_object_velocities = self._estimate_segment_init_velocity(
            start_idx, init_object_vertices
        )
        init_rest_lengths = self._compute_segment_rest_lengths(
            init_object_vertices,
            controller_points[0] if controller_points is not None else None,
        )

        return {
            "object_points": object_points,
            "controller_points": controller_points,
            "init_object_vertices": init_object_vertices.contiguous(),
            "init_object_velocities": init_object_velocities.contiguous(),
            "init_rest_lengths": init_rest_lengths,
        }

    def _build_single_segment_from_batch_data(self, batch_data, inst_idx):
        B = int(batch_data["batch_size"])
        inst_idx = int(inst_idx)
        start_idx = int(batch_data["starts"][inst_idx])
        segment_len = int(batch_data["segment_len"])
        rollout_cache_len = int(batch_data.get("rollout_cache_len", segment_len))

        controller_points = None
        if batch_data["controller_points"] is not None:
            num_ctrl = self.controller_points.shape[1]
            controller_points = (
                batch_data["controller_points"]
                .reshape(rollout_cache_len, B, num_ctrl, 3)[:segment_len, inst_idx]
                .contiguous()
            )

        init_rest_lengths = batch_data["init_rest_lengths_batched"].reshape(
            B, self.init_springs.shape[0]
        )[inst_idx]

        return {
            "object_points": self.train_object_points[
                start_idx : start_idx + segment_len
            ].contiguous(),
            "controller_points": controller_points,
            "init_object_vertices": batch_data["init_object_vertices_by_batch"][
                inst_idx
            ].contiguous(),
            "init_object_velocities": batch_data["init_object_velocities_by_batch"][
                inst_idx
            ].contiguous(),
            "init_rest_lengths": init_rest_lengths.contiguous(),
        }

    def _visualize_segment(self, start_idx, segment_len, video_path, segment=None):
        logger.info(
            f"[Visualize-Batched]: rollout segment start={start_idx}, len={segment_len}"
        )
        if segment is None:
            segment = self._build_single_segment_tensors(start_idx, segment_len)

        simulator_controller_points = self.simulator.controller_points
        simulator_rest_lengths = wp.to_torch(
            self.simulator.wp_rest_lengths, requires_grad=False
        ).detach().clone()
        self.simulator.controller_points = segment["controller_points"]
        try:
            self.simulator.set_rest_lengths(segment["init_rest_lengths"])
            self.simulator.set_init_state(
                wp.from_torch(
                    segment["init_object_vertices"], dtype=wp.vec3, requires_grad=False
                ),
                wp.from_torch(
                    segment["init_object_velocities"], dtype=wp.vec3, requires_grad=False
                ),
                pure_inference=True,
            )
            vertices = [
                wp.to_torch(self.simulator.wp_states[0].wp_x, requires_grad=False).cpu()
            ]

            with wp.ScopedTimer("simulate_segment"):
                for frame_idx in range(1, segment_len):
                    if self.simulator.controller_points is not None:
                        self.simulator.set_controller_target(
                            frame_idx, pure_inference=True
                        )
                    if self.simulator.object_collision_flag:
                        self.simulator.update_collision_graph()

                    if cfg.use_graph:
                        wp.capture_launch(self.simulator.forward_graph)
                    else:
                        self.simulator.step()

                    x = wp.to_torch(
                        self.simulator.wp_states[-1].wp_x, requires_grad=False
                    )
                    vertices.append(x.cpu())
                    self.simulator.set_init_state(
                        self.simulator.wp_states[-1].wp_x,
                        self.simulator.wp_states[-1].wp_v,
                        pure_inference=True,
                    )
        finally:
            self.simulator.set_rest_lengths(simulator_rest_lengths)
            self.simulator.controller_points = simulator_controller_points

        object_colors = None
        if self.object_colors is not None:
            end_idx = start_idx + segment_len
            if self.object_colors.shape[0] >= end_idx:
                object_colors = self.object_colors[start_idx:end_idx]
            else:
                object_colors = self.object_colors[:segment_len]

        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        vertices = torch.stack(vertices, dim=0)
        visualize_pc(
            vertices[:, : self.num_all_points, :],
            object_colors=object_colors,
            controller_points=segment["controller_points"],
            visualize=False,
            save_video=True,
            save_path=video_path,
            frame_start_idx=int(start_idx),
        )

    def _visualize_batch_instances(self, segment_batches, expected_segment_len, iteration):
        if len(segment_batches) == 0:
            return

        if self.batch_vis_num_instances == -1:
            num_instances = self.batch_size
        else:
            num_instances = max(1, min(self.batch_size, self.batch_vis_num_instances))
        num_groups = min(len(segment_batches), self.batch_vis_num_groups)

        if iteration < 0:
            round_idx = 0
        else:
            round_idx = int(iteration // self.batch_vis_interval)
        group_offset = (round_idx * num_groups) % len(segment_batches)
        selected_group_indices = [
            (group_offset + idx) % len(segment_batches) for idx in range(num_groups)
        ]

        if iteration < 0:
            save_dir = f"{cfg.base_dir}/train/batch_instances/init"
        else:
            save_dir = f"{cfg.base_dir}/train/batch_instances/iter_{iteration}"
        os.makedirs(save_dir, exist_ok=True)

        for group_idx in selected_group_indices:
            starts = segment_batches[group_idx]["starts"]
            for inst_idx, start_idx in enumerate(starts[:num_instances]):
                video_path = f"{save_dir}/inst_{inst_idx:02d}_start_{int(start_idx):04d}.mp4"
                segment = self._build_single_segment_from_batch_data(
                    segment_batches[group_idx], inst_idx
                )
                self._visualize_segment(
                    start_idx=int(start_idx),
                    segment_len=expected_segment_len,
                    video_path=video_path,
                    segment=segment,
                )

    def _append_full_rollout_loss_rows(
        self, iteration, total_frames, first_batch_data, frame_loss_rows
    ):
        if first_batch_data is None or len(first_batch_data["starts"]) == 0:
            return

        start_idx = int(first_batch_data["starts"][0])
        if start_idx != 0:
            logger.warning(
                "[Train-Batched-FullRollout]: skip full rollout loss logging because "
                f"the first window starts at {start_idx}, not 0"
            )
            return

        self._sync_single_simulator_from_batch()

        simulator_controller_points = self.simulator.controller_points
        simulator_rest_lengths = wp.to_torch(
            self.simulator.wp_rest_lengths, requires_grad=False
        ).detach().clone()

        batch_size = int(first_batch_data["batch_size"])
        init_rest_lengths = first_batch_data["init_rest_lengths_batched"].reshape(
            batch_size, self.init_springs.shape[0]
        )[0]

        controller_points = None
        if self.controller_points is not None:
            controller_points = self.controller_points[:total_frames].contiguous()

        try:
            self.simulator.controller_points = controller_points
            self.simulator.set_rest_lengths(init_rest_lengths.contiguous())
            self.simulator.set_init_state(
                wp.from_torch(
                    first_batch_data["init_object_vertices_by_batch"][0].contiguous(),
                    dtype=wp.vec3,
                    requires_grad=False,
                ),
                wp.from_torch(
                    first_batch_data["init_object_velocities_by_batch"][0].contiguous(),
                    dtype=wp.vec3,
                    requires_grad=False,
                ),
                pure_inference=True,
            )
            self.simulator.clear_loss()

            for frame_idx in range(1, int(total_frames)):
                self.simulator.set_controller_target(frame_idx)
                if self.simulator.object_collision_flag:
                    self.simulator.update_collision_graph()

                self.simulator.step()
                if cfg.data_type == "real":
                    self.simulator.calculate_loss()
                    chamfer_loss = wp.to_torch(
                        self.simulator.chamfer_loss, requires_grad=False
                    )
                    track_loss = wp.to_torch(
                        self.simulator.track_loss, requires_grad=False
                    )
                    acc_loss = wp.to_torch(
                        self.simulator.acc_loss, requires_grad=False
                    )
                    chamfer_value = float(chamfer_loss.item())
                    track_value = float(track_loss.item())
                    acc_value = float(acc_loss.item())
                else:
                    self.simulator.calculate_simple_loss()
                    chamfer_value = 0.0
                    track_value = 0.0
                    acc_value = 0.0

                loss = wp.to_torch(self.simulator.loss, requires_grad=False)
                frame_loss_rows.append(
                    [
                        cfg.run_name,
                        int(iteration),
                        -1,
                        -1,
                        -1,
                        0,
                        int(frame_idx),
                        int(frame_idx),
                        float(loss.item()),
                        chamfer_value,
                        track_value,
                        acc_value,
                    ]
                )

                self.simulator.clear_loss()
                self.simulator.set_init_state(
                    self.simulator.wp_states[-1].wp_x,
                    self.simulator.wp_states[-1].wp_v,
                    pure_inference=True,
                )
        finally:
            self.simulator.clear_loss()
            self.simulator.set_rest_lengths(simulator_rest_lengths)
            self.simulator.controller_points = simulator_controller_points

    def _load_segment_batch_into_sim(self, sim, batch_data):
        sim.gt_object_points = batch_data["gt_object_points"]
        sim.controller_points = batch_data["controller_points"]
        if "init_rest_lengths_batched" in batch_data:
            sim.set_rest_lengths(batch_data["init_rest_lengths_batched"])
        if cfg.data_type == "real":
            sim.gt_object_visibilities = batch_data["gt_object_visibilities"].int()
            sim.gt_object_motions_valid = batch_data["gt_object_motions_valid"].int()

    def _build_batched_simulator_for_train(self, batch_data=None):
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        B = self.batch_size if batch_data is None else int(batch_data["batch_size"])
        num_ctrl_single = (
            int(self.controller_points.shape[1]) if self.controller_points is not None else 0
        )

        if batch_data is None:
            # Batched GT / controller trajectories in flattened layout: [T, B*N, ...]
            batched_object_points = self._tile_time_tensor(self.train_object_points, B)
            batched_controller_points = (
                self._tile_time_tensor(self.controller_points, B)
                if self.controller_points is not None
                else None
            )
            batched_object_visibilities = (
                self._tile_time_tensor(self.train_object_visibilities, B)
                if self.train_object_visibilities is not None
                else None
            )
            batched_object_motions_valid = (
                self._tile_time_tensor(self.train_object_motions_valid, B)
                if self.train_object_motions_valid is not None
                else None
            )

            # Batched initial vertices: [B*object_points, B*controller_points]
            obj_init_single = self.init_vertices[: self.num_all_points]
            obj_init_batched = (
                obj_init_single.unsqueeze(0)
                .repeat(B, 1, 1)
                .reshape(B * self.num_all_points, 3)
            )
            if num_ctrl_single > 0:
                ctrl_init_single = self.init_vertices[
                    self.num_all_points : self.num_all_points + num_ctrl_single
                ]
                ctrl_init_batched = (
                    ctrl_init_single.unsqueeze(0)
                    .repeat(B, 1, 1)
                    .reshape(B * num_ctrl_single, 3)
                )
                init_vertices_batched = torch.cat(
                    [obj_init_batched, ctrl_init_batched], dim=0
                )
            else:
                init_vertices_batched = obj_init_batched

            batched_init_velocities = None
            if self.init_velocities is not None:
                batched_init_velocities = (
                    self.init_velocities[: self.num_all_points]
                    .unsqueeze(0)
                    .repeat(B, 1, 1)
                    .reshape(B * self.num_all_points, 3)
                    .contiguous()
                )
        else:
            batched_object_points = batch_data["gt_object_points"]
            batched_controller_points = batch_data["controller_points"]
            batched_object_visibilities = batch_data["gt_object_visibilities"]
            batched_object_motions_valid = batch_data["gt_object_motions_valid"]
            init_vertices_batched = batch_data["init_vertices_batched"]
            batched_init_velocities = batch_data["init_velocities_batched"]
            init_rest_lengths = batch_data["init_rest_lengths_batched"]

        if batch_data is None:
            init_rest_lengths = self.init_rest_lengths

        self.batch_simulator = SpringMassSystemWarpBatched(
            init_vertices_batched,
            self.init_springs,
            init_rest_lengths,
            self.init_masses[: self.num_all_points],
            dt=cfg.dt,
            num_substeps=cfg.num_substeps,
            spring_Y=cfg.init_spring_Y,
            collide_elas=cfg.collide_elas,
            collide_fric=cfg.collide_fric,
            dashpot_damping=cfg.dashpot_damping,
            drag_damping=cfg.drag_damping,
            collide_object_elas=cfg.collide_object_elas,
            collide_object_fric=cfg.collide_object_fric,
            init_masks=self.init_masks,
            collision_dist=cfg.collision_dist,
            init_velocities=batched_init_velocities,
            batch_size=B,
            num_object_points_single=self.num_all_points,
            num_control_points_single=num_ctrl_single,
            num_original_points_single=self.num_original_points,
            num_surface_points_single=self.num_surface_points,
            num_object_points=B * self.num_all_points,
            num_surface_points=(
                B * self.num_surface_points if self.num_surface_points is not None else None
            ),
            num_original_points=(
                B * self.num_original_points if self.num_original_points is not None else None
            ),
            controller_points=batched_controller_points,
            reverse_z=cfg.reverse_z,
            spring_Y_min=cfg.spring_Y_min,
            spring_Y_max=cfg.spring_Y_max,
            gt_object_points=batched_object_points,
            gt_object_visibilities=batched_object_visibilities,
            gt_object_motions_valid=batched_object_motions_valid,
            self_collision=cfg.self_collision,
        )
        self.batch_size_loaded = B
        self.batch_segment_len_loaded = (
            int(batched_object_points.shape[0]) if batched_object_points is not None else None
        )

    def _sync_single_simulator_from_batch(self):
        if self.batch_simulator is None:
            return

        spring_Y = wp.to_torch(self.batch_simulator.wp_spring_Y, requires_grad=False).detach().clone()
        collide_elas = wp.to_torch(self.batch_simulator.wp_collide_elas, requires_grad=False).detach().clone()
        collide_fric = wp.to_torch(self.batch_simulator.wp_collide_fric, requires_grad=False).detach().clone()
        collide_object_elas = wp.to_torch(
            self.batch_simulator.wp_collide_object_elas, requires_grad=False
        ).detach().clone()
        collide_object_fric = wp.to_torch(
            self.batch_simulator.wp_collide_object_fric, requires_grad=False
        ).detach().clone()

        self.simulator.set_spring_Y(spring_Y)
        self.simulator.set_collide(collide_elas, collide_fric)
        self.simulator.set_collide_object(collide_object_elas, collide_object_fric)

    def train(self, start_epoch=-1):
        if self.batch_mode:
            return self.train_batched(start_epoch=start_epoch)

        # Render the initial visualization
        video_path = f"{cfg.base_dir}/train/init.mp4"
        self.visualize_sim(save_only=True, video_path=video_path)

        timing_csv_path = self._prepare_timing_csv(start_epoch)
        best_loss = None
        best_epoch = None
        # Train the model with the physical simulator
        for i in range(start_epoch + 1, cfg.iterations):
            iter_wall_start = time.perf_counter()
            full_rollout_eval_sec = 0.0
            total_loss = 0.0
            if cfg.data_type == "real":
                total_chamfer_loss = 0.0
                total_track_loss = 0.0
            train_iter_start = time.perf_counter()
            self.simulator.set_init_state(
                self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
            )
            with wp.ScopedTimer("backward"):
                for j in tqdm(range(1, cfg.train_frame)):
                    self.simulator.set_controller_target(j)
                    if self.simulator.object_collision_flag:
                        self.simulator.update_collision_graph()

                    if cfg.use_graph:
                        wp.capture_launch(self.simulator.graph)
                    else:
                        if cfg.data_type == "real":
                            with self.simulator.tape:
                                self.simulator.step()
                                self.simulator.calculate_loss()
                            self.simulator.tape.backward(self.simulator.loss)
                        else:
                            with self.simulator.tape:
                                self.simulator.step()
                                self.simulator.calculate_simple_loss()
                            self.simulator.tape.backward(self.simulator.loss)

                    self.optimizer.step()

                    if cfg.data_type == "real":
                        chamfer_loss = wp.to_torch(
                            self.simulator.chamfer_loss, requires_grad=False
                        )
                        track_loss = wp.to_torch(
                            self.simulator.track_loss, requires_grad=False
                        )
                        total_chamfer_loss += chamfer_loss.item()
                        total_track_loss += track_loss.item()

                    loss = wp.to_torch(self.simulator.loss, requires_grad=False)
                    total_loss += loss.item()

                    if cfg.use_graph:
                        # Only need to clear the gradient, the tape is created in the graph
                        self.simulator.tape.zero()
                    else:
                        # Need to reset the compute graph and clear the gradient
                        self.simulator.tape.reset()
                    self.simulator.clear_loss()
                    # Set the intial state for the next step
                    self.simulator.set_init_state(
                        self.simulator.wp_states[-1].wp_x,
                        self.simulator.wp_states[-1].wp_v,
                    )
            train_iter_sec = time.perf_counter() - train_iter_start

            total_loss /= cfg.train_frame - 1
            if cfg.data_type == "real":
                total_chamfer_loss /= cfg.train_frame - 1
                total_track_loss /= cfg.train_frame - 1
            wandb.log(
                {
                    "loss": total_loss,
                    "chamfer_loss": (
                        total_chamfer_loss if cfg.data_type == "real" else 0
                    ),
                    "track_loss": total_track_loss if cfg.data_type == "real" else 0,
                    "collide_else": wp.to_torch(
                        self.simulator.wp_collide_elas, requires_grad=False
                    ).item(),
                    "collide_fric": wp.to_torch(
                        self.simulator.wp_collide_fric, requires_grad=False
                    ).item(),
                    "collide_object_elas": wp.to_torch(
                        self.simulator.wp_collide_object_elas, requires_grad=False
                    ).item(),
                    "collide_object_fric": wp.to_torch(
                        self.simulator.wp_collide_object_fric, requires_grad=False
                    ).item(),
                },
                step=i,
            )

            logger.info(
                f"[Train]: Case: {cfg.run_name}, Iteration: {i}, Loss: {total_loss}"
            )

            if i % cfg.vis_interval == 0 or i == cfg.iterations - 1:
                video_path = f"{cfg.base_dir}/train/sim_iter{i}.mp4"
                self.visualize_sim(save_only=True, video_path=video_path)
                wandb.log(
                    {
                        "video": wandb.Video(
                            video_path,
                            format="mp4",
                            fps=cfg.FPS,
                        ),
                    },
                    step=i,
                )
                # Save the parameters
                cur_model = {
                    "epoch": i,
                    "num_object_springs": self.num_object_springs,
                    "spring_Y": torch.exp(
                        wp.to_torch(self.simulator.wp_spring_Y, requires_grad=False)
                    ),
                    "collide_elas": wp.to_torch(
                        self.simulator.wp_collide_elas, requires_grad=False
                    ),
                    "collide_fric": wp.to_torch(
                        self.simulator.wp_collide_fric, requires_grad=False
                    ),
                    "collide_object_elas": wp.to_torch(
                        self.simulator.wp_collide_object_elas, requires_grad=False
                    ),
                    "collide_object_fric": wp.to_torch(
                        self.simulator.wp_collide_object_fric, requires_grad=False
                    ),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                }
                if best_loss == None or total_loss < best_loss:
                    # Remove old best model file if it exists
                    if best_loss is not None:
                        old_best_model_path = (
                            f"{cfg.base_dir}/train/best_{best_epoch}.pth"
                        )
                        if os.path.exists(old_best_model_path):
                            os.remove(old_best_model_path)

                    # Update best loss and best epoch
                    best_loss = total_loss
                    best_epoch = i

                    # Save new best model
                    best_model_path = f"{cfg.base_dir}/train/best_{best_epoch}.pth"
                    torch.save(cur_model, best_model_path)
                    logger.info(
                        f"Latest best model saved: epoch {best_epoch} with loss {best_loss}"
                    )

                torch.save(cur_model, f"{cfg.base_dir}/train/iter_{i}.pth")
                logger.info(
                    f"[Visualize]: Visualize the simulation at iteration {i} and save the model"
                )

            total_iter_sec = time.perf_counter() - iter_wall_start
            self._append_timing_row(
                timing_csv_path=timing_csv_path,
                mode="single",
                iteration=i,
                train_iter_sec=train_iter_sec,
                full_rollout_eval_sec=full_rollout_eval_sec,
                total_iter_sec=total_iter_sec,
                num_groups=1,
                batch_size=1,
                segment_len=cfg.train_frame,
                segment_stride=cfg.train_frame,
                num_train_steps=cfg.train_frame - 1,
            )
            logger.info(
                "[Train-Timing]: "
                f"Case: {cfg.run_name}, Iteration: {i}, "
                f"train_iter_sec={train_iter_sec:.3f}, "
                f"full_rollout_eval_sec={full_rollout_eval_sec:.3f}, "
                f"total_iter_sec={total_iter_sec:.3f}"
            )

        wandb.finish()

    def train_batched(self, start_epoch=-1):
        if self.batch_size <= 1:
            logger.warning(
                "[Train-Batched]: batch_size <= 1, fallback to single training loop."
            )
            prev_batch_mode = self.batch_mode
            self.batch_mode = False
            try:
                return self.train(start_epoch=start_epoch)
            finally:
                self.batch_mode = prev_batch_mode

        total_frames = self._get_trainable_total_frames()

        segment_starts = self._compute_segment_start_indices(total_frames)
        if len(segment_starts) < self.batch_size:
            raise ValueError(
                f"Need at least batch_size ({self.batch_size}) segments, "
                f"but only got {len(segment_starts)}"
            )

        remainder = len(segment_starts) % self.batch_size
        if remainder != 0:
            logger.warning(
                "[Train-Batched]: segment count is not divisible by batch_size, "
                f"dropping last {remainder} segments."
            )
            segment_starts = segment_starts[: len(segment_starts) - remainder]

        grouped_starts = [
            segment_starts[k : k + self.batch_size]
            for k in range(0, len(segment_starts), self.batch_size)
        ]
        self._init_window_loss_weights(len(segment_starts))

        def build_segment_batches(num_rollout_windows=0):
            batches = []
            global_window_offset = 0
            for starts in grouped_starts:
                batches.append(
                    self._build_segment_batch_tensors(
                        starts,
                        global_window_offset=global_window_offset,
                        num_rollout_windows=num_rollout_windows,
                    )
                )
                global_window_offset += len(starts)
            return batches

        segment_batches = build_segment_batches()
        if len(segment_batches) == 0:
            raise ValueError("No valid segment batches were built for batched training")
        logger.info(
            "[Train-Batched]: Prepared segments before training loop - "
            f"train_frame_cfg={cfg.train_frame}, train_frame_effective={total_frames}, "
            f"slices={len(segment_starts)}, groups={len(grouped_starts)}, "
            f"batch_size={self.batch_size}, segment_len={self.segment_len}, "
            f"segment_stride={self.segment_stride}, "
            f"rollout_check_len={self.rollout_check_len}"
        )
        if self.batch_loss_weighting:
            logger.info(
                "[Train-Batched-Weight]: enabled, using previous-iteration "
                "per-window total_loss as error, "
                f"clamp={self.batch_loss_weight_min:.3f}-"
                f"{self.batch_loss_weight_max:.3f}"
            )

        # Record per-instance loss progression across iterations.
        per_instance_loss_dir = f"{cfg.base_dir}/train/per_instance_loss"
        os.makedirs(per_instance_loss_dir, exist_ok=True)
        per_instance_loss_csv_path = (
            f"{per_instance_loss_dir}/batched_instance_loss.csv"
        )
        if start_epoch < 0 and os.path.exists(per_instance_loss_csv_path):
            os.remove(per_instance_loss_csv_path)
        if not os.path.exists(per_instance_loss_csv_path):
            with open(per_instance_loss_csv_path, "w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(
                    [
                        "iteration",
                        "group_idx",
                        "inst_local_idx",
                        "start_idx",
                        "num_steps",
                        "total_loss",
                        "chamfer_loss",
                        "track_loss",
                        "acc_loss",
                    ]
                )

        # Record per-window, per-frame loss for plotting overlapping window rollouts.
        per_frame_loss_dir = f"{cfg.base_dir}/train/per_frame_loss"
        os.makedirs(per_frame_loss_dir, exist_ok=True)
        per_frame_loss_csv_path = f"{per_frame_loss_dir}/batched_frame_loss.csv"
        if start_epoch < 0 and os.path.exists(per_frame_loss_csv_path):
            os.remove(per_frame_loss_csv_path)
        if not os.path.exists(per_frame_loss_csv_path):
            with open(per_frame_loss_csv_path, "w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(
                    [
                        "case_name",
                        "iteration",
                        "group_idx",
                        "inst_local_idx",
                        "global_window_id",
                        "window_start",
                        "local_frame",
                        "global_frame",
                        "total_loss",
                        "chamfer_loss",
                        "track_loss",
                        "acc_loss",
                    ]
                )

        timing_csv_path = self._prepare_timing_csv(start_epoch)

        expected_segment_len = int(segment_batches[0]["segment_len"])
        rollout_cache_len = int(
            segment_batches[0].get("rollout_cache_len", expected_segment_len)
        )
        if (
            self.batch_simulator is None
            or self.batch_size_loaded != self.batch_size
            or self.batch_segment_len_loaded != expected_segment_len
        ):
            logger.info(
                "[Train-Batched]: build batched simulator with "
                f"batch_size={self.batch_size}, segment_len={expected_segment_len}, "
                f"rollout_cache_len={rollout_cache_len}"
            )
            self._build_batched_simulator_for_train(batch_data=segment_batches[0])

        sim = self.batch_simulator
        self.optimizer = torch.optim.Adam(
            [
                wp.to_torch(sim.wp_spring_Y),
                wp.to_torch(sim.wp_collide_elas),
                wp.to_torch(sim.wp_collide_fric),
                wp.to_torch(sim.wp_collide_object_elas),
                wp.to_torch(sim.wp_collide_object_fric),
            ],
            lr=cfg.base_lr,
            betas=(0.9, 0.99),
        )

        best_loss = None
        best_epoch = None

        if start_epoch < 0:
            self._sync_single_simulator_from_batch()
            video_path = f"{cfg.base_dir}/train/init.mp4"
            logger.info("[Train-Batched]: Save initial full rollout video")
            self.visualize_sim(save_only=True, video_path=video_path)

            if self.batch_vis_per_instance:
                logger.info("[Train-Batched]: Save initial per-instance rollout videos")
                self._visualize_batch_instances(
                    segment_batches=segment_batches,
                    expected_segment_len=expected_segment_len,
                    iteration=-1,
                )

        for i in range(start_epoch + 1, cfg.iterations):
            iter_wall_start = time.perf_counter()
            if self.rollout_prefix_switch:
                switch_info = self._compute_rollout_switch_info(
                    i, segment_starts, expected_segment_len
                )
                self._log_rollout_switch_info(i, switch_info)
                segment_batches = build_segment_batches(
                    num_rollout_windows=switch_info["num_rollout_windows"]
                )

            frame_loss_rows = []
            full_rollout_start = time.perf_counter()
            self._append_full_rollout_loss_rows(
                iteration=i,
                total_frames=total_frames,
                first_batch_data=segment_batches[0],
                frame_loss_rows=frame_loss_rows,
            )
            full_rollout_eval_sec = time.perf_counter() - full_rollout_start

            total_loss = 0.0
            total_steps = 0
            if cfg.data_type == "real":
                total_chamfer_loss = 0.0
                total_track_loss = 0.0

            num_groups = len(segment_batches)
            per_inst_total = np.zeros((num_groups, self.batch_size), dtype=np.float64)
            per_inst_steps = np.zeros((num_groups, self.batch_size), dtype=np.int32)
            if cfg.data_type == "real":
                per_inst_chamfer = np.zeros(
                    (num_groups, self.batch_size), dtype=np.float64
                )
                per_inst_track = np.zeros(
                    (num_groups, self.batch_size), dtype=np.float64
                )
                per_inst_acc = np.zeros((num_groups, self.batch_size), dtype=np.float64)

            iter_pos_cache_groups = []
            iter_vel_cache_groups = []
            train_iter_start = time.perf_counter()
            with wp.ScopedTimer("backward_batched"):
                for batch_group_idx, batch_data in enumerate(tqdm(segment_batches)):
                    sim.set_loss_weights(self._get_batch_loss_weights(batch_data))
                    self._load_segment_batch_into_sim(sim, batch_data)
                    sim.set_init_state(
                        wp.from_torch(
                            batch_data["init_object_vertices_batched"],
                            dtype=wp.vec3,
                            requires_grad=False,
                        ),
                        wp.from_torch(
                            batch_data["init_velocities_batched"],
                            dtype=wp.vec3,
                            requires_grad=False,
                        ),
                    )
                    group_pos_steps = [
                        batch_data["init_object_vertices_by_batch"].detach().clone()
                    ]
                    group_vel_steps = [
                        batch_data["init_object_velocities_by_batch"].detach().clone()
                    ]

                    for j in range(1, expected_segment_len):
                        sim.set_controller_target(j)
                        if sim.object_collision_flag:
                            sim.update_collision_graph()

                        if cfg.use_graph:
                            wp.capture_launch(sim.graph)
                        else:
                            if cfg.data_type == "real":
                                with sim.tape:
                                    sim.step()
                                    sim.calculate_loss()
                                sim.tape.backward(sim.loss)
                            else:
                                with sim.tape:
                                    sim.step()
                                    sim.calculate_simple_loss()
                                sim.tape.backward(sim.loss)

                        self.optimizer.step()

                        if cfg.data_type == "real":
                            chamfer_loss = wp.to_torch(
                                sim.chamfer_loss, requires_grad=False
                            )
                            track_loss = wp.to_torch(
                                sim.track_loss, requires_grad=False
                            )
                            total_chamfer_loss += chamfer_loss.item()
                            total_track_loss += track_loss.item()

                        loss = wp.to_torch(sim.loss, requires_grad=False)
                        if not bool(torch.isfinite(loss).all()):
                            if self.nonfinite_debug_log_count < 20:
                                loss_per_batch = wp.to_torch(
                                    sim.loss_per_batch, requires_grad=False
                                )
                                logger.error(
                                    "[Train-Batch-NaN]: non-finite loss detected "
                                    f"starts={batch_data['starts']}, frame_step={j}, "
                                    f"loss={loss.detach().cpu().numpy().tolist()}, "
                                    f"loss_per_batch={loss_per_batch.detach().cpu().numpy().tolist()}"
                                )
                                x = wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False)
                                finite_x = torch.isfinite(x).all(dim=-1)
                                if bool(finite_x.any()):
                                    x_finite = x[finite_x]
                                    logger.error(
                                        "[Train-Batch-NaN]: state bbox "
                                        f"min={x_finite.min(dim=0).values.detach().cpu().numpy().tolist()}, "
                                        f"max={x_finite.max(dim=0).values.detach().cpu().numpy().tolist()}, "
                                        f"nonfinite_points={(~finite_x).sum().item()}"
                                    )
                                else:
                                    logger.error("[Train-Batch-NaN]: all state points are nonfinite")
                                self.nonfinite_debug_log_count += 1
                        total_loss += loss.item()
                        total_steps += 1

                        loss_per_batch = (
                            wp.to_torch(sim.loss_per_batch, requires_grad=False)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        if cfg.data_type == "real":
                            chamfer_per_batch = (
                                wp.to_torch(
                                    sim.chamfer_loss_per_batch, requires_grad=False
                                )
                                .detach()
                                .cpu()
                                .numpy()
                            )
                            track_per_batch = (
                                wp.to_torch(sim.track_loss_per_batch, requires_grad=False)
                                .detach()
                                .cpu()
                                .numpy()
                            )
                            acc_per_batch = (
                                wp.to_torch(sim.acc_loss_per_batch, requires_grad=False)
                                .detach()
                                .cpu()
                                .numpy()
                            )

                        for inst_local_idx, _start_idx in enumerate(batch_data["starts"]):
                            window_start = int(_start_idx)
                            global_window_id = (
                                int(batch_data["global_window_offset"])
                                + int(inst_local_idx)
                            )
                            if cfg.data_type == "real":
                                chamfer_value = float(chamfer_per_batch[inst_local_idx])
                                track_value = float(track_per_batch[inst_local_idx])
                                acc_value = float(acc_per_batch[inst_local_idx])
                            else:
                                chamfer_value = 0.0
                                track_value = 0.0
                                acc_value = 0.0
                            frame_loss_rows.append(
                                [
                                    cfg.run_name,
                                    i + 1,
                                    batch_group_idx,
                                    inst_local_idx,
                                    global_window_id,
                                    window_start,
                                    j,
                                    window_start + j,
                                    float(loss_per_batch[inst_local_idx]),
                                    chamfer_value,
                                    track_value,
                                    acc_value,
                                ]
                            )
                            per_inst_total[batch_group_idx, inst_local_idx] += float(
                                loss_per_batch[inst_local_idx]
                            )
                            per_inst_steps[batch_group_idx, inst_local_idx] += 1
                            if cfg.data_type == "real":
                                per_inst_chamfer[batch_group_idx, inst_local_idx] += float(
                                    chamfer_per_batch[inst_local_idx]
                                )
                                per_inst_track[batch_group_idx, inst_local_idx] += float(
                                    track_per_batch[inst_local_idx]
                                )
                                per_inst_acc[batch_group_idx, inst_local_idx] += float(
                                    acc_per_batch[inst_local_idx]
                                )

                        if cfg.use_graph:
                            sim.tape.zero()
                        else:
                            sim.tape.reset()
                        sim.clear_loss()
                        x_cache = wp.to_torch(
                            sim.wp_states[-1].wp_x, requires_grad=False
                        )
                        v_cache = wp.to_torch(
                            sim.wp_states[-1].wp_v, requires_grad=False
                        )
                        group_pos_steps.append(
                            x_cache.reshape(
                                batch_data["batch_size"], self.num_all_points, 3
                            )
                            .detach()
                            .clone()
                        )
                        group_vel_steps.append(
                            v_cache.reshape(
                                batch_data["batch_size"], self.num_all_points, 3
                            )
                            .detach()
                            .clone()
                        )
                        sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v)

                    group_rollout_cache_len = int(
                        batch_data.get("rollout_cache_len", expected_segment_len)
                    )
                    for j in range(expected_segment_len, group_rollout_cache_len):
                        sim.set_controller_target(j, pure_inference=True)
                        if sim.object_collision_flag:
                            sim.update_collision_graph()

                        if cfg.use_graph:
                            wp.capture_launch(sim.forward_graph)
                        else:
                            sim.step()

                        x_cache = wp.to_torch(
                            sim.wp_states[-1].wp_x, requires_grad=False
                        )
                        v_cache = wp.to_torch(
                            sim.wp_states[-1].wp_v, requires_grad=False
                        )
                        group_pos_steps.append(
                            x_cache.reshape(
                                batch_data["batch_size"], self.num_all_points, 3
                            )
                            .detach()
                            .clone()
                        )
                        group_vel_steps.append(
                            v_cache.reshape(
                                batch_data["batch_size"], self.num_all_points, 3
                            )
                            .detach()
                            .clone()
                        )
                        sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v)

                    iter_pos_cache_groups.append(
                        torch.stack(group_pos_steps, dim=1).detach()
                    )
                    iter_vel_cache_groups.append(
                        torch.stack(group_vel_steps, dim=1).detach()
                    )
            train_iter_sec = time.perf_counter() - train_iter_start

            if total_steps == 0:
                raise RuntimeError("No training steps were executed in train_batched()")
            if len(iter_pos_cache_groups) > 0:
                self.prev_rollout_pos_cache = torch.cat(
                    iter_pos_cache_groups, dim=0
                ).detach()
                self.prev_rollout_vel_cache = torch.cat(
                    iter_vel_cache_groups, dim=0
                ).detach()
            total_loss /= total_steps
            if cfg.data_type == "real":
                total_chamfer_loss /= total_steps
                total_track_loss /= total_steps
            self._update_window_loss_weights(
                i, segment_batches, per_inst_total, per_inst_steps
            )

            with open(per_instance_loss_csv_path, "a", newline="") as csv_file:
                writer = csv.writer(csv_file)
                for group_idx, batch_data in enumerate(segment_batches):
                    for inst_local_idx, start_idx in enumerate(batch_data["starts"]):
                        steps = int(per_inst_steps[group_idx, inst_local_idx])
                        if steps <= 0:
                            continue
                        total_avg = per_inst_total[group_idx, inst_local_idx] / float(steps)
                        if cfg.data_type == "real":
                            chamfer_avg = per_inst_chamfer[group_idx, inst_local_idx] / float(
                                steps
                            )
                            track_avg = per_inst_track[group_idx, inst_local_idx] / float(
                                steps
                            )
                            acc_avg = per_inst_acc[group_idx, inst_local_idx] / float(steps)
                        else:
                            chamfer_avg = 0.0
                            track_avg = 0.0
                            acc_avg = 0.0
                        writer.writerow(
                            [
                                i + 1,
                                group_idx,
                                inst_local_idx,
                                int(start_idx),
                                steps,
                                total_avg,
                                chamfer_avg,
                                track_avg,
                                acc_avg,
                            ]
                        )

            if len(frame_loss_rows) > 0:
                with open(per_frame_loss_csv_path, "a", newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerows(frame_loss_rows)

            wandb.log(
                {
                    "loss": total_loss,
                    "chamfer_loss": total_chamfer_loss if cfg.data_type == "real" else 0,
                    "track_loss": total_track_loss if cfg.data_type == "real" else 0,
                    "collide_else": wp.to_torch(sim.wp_collide_elas, requires_grad=False).item(),
                    "collide_fric": wp.to_torch(sim.wp_collide_fric, requires_grad=False).item(),
                    "collide_object_elas": wp.to_torch(
                        sim.wp_collide_object_elas, requires_grad=False
                    ).item(),
                    "collide_object_fric": wp.to_torch(
                        sim.wp_collide_object_fric, requires_grad=False
                    ).item(),
                    "batch_size": self.batch_size,
                    "segment_len": expected_segment_len,
                    "segment_stride": int(self.segment_stride),
                    "segment_batches": len(segment_batches),
                },
                step=i,
            )

            logger.info(
                f"[Train-Batched]: Case: {cfg.run_name}, Iteration: {i}, "
                f"Loss: {total_loss}, Batch: {self.batch_size}"
            )

            need_global_vis = (i % cfg.vis_interval == 0) or (i == cfg.iterations - 1)
            need_instance_vis = self.batch_vis_per_instance and (
                (i % self.batch_vis_interval == 0) or (i == cfg.iterations - 1)
            )

            if need_global_vis or need_instance_vis:
                self._sync_single_simulator_from_batch()

            if need_global_vis:
                video_path = f"{cfg.base_dir}/train/sim_iter{i}.mp4"
                self.visualize_sim(save_only=True, video_path=video_path)
                wandb.log(
                    {
                        "video": wandb.Video(
                            video_path,
                            format="mp4",
                            fps=cfg.FPS,
                        ),
                    },
                    step=i,
                )

                cur_model = {
                    "epoch": i,
                    "num_object_springs": self.num_object_springs,
                    "spring_Y": torch.exp(wp.to_torch(sim.wp_spring_Y, requires_grad=False)),
                    "collide_elas": wp.to_torch(sim.wp_collide_elas, requires_grad=False),
                    "collide_fric": wp.to_torch(sim.wp_collide_fric, requires_grad=False),
                    "collide_object_elas": wp.to_torch(
                        sim.wp_collide_object_elas, requires_grad=False
                    ),
                    "collide_object_fric": wp.to_torch(
                        sim.wp_collide_object_fric, requires_grad=False
                    ),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                }

                if best_loss is None or total_loss < best_loss:
                    if best_loss is not None:
                        old_best_model_path = f"{cfg.base_dir}/train/best_{best_epoch}.pth"
                        if os.path.exists(old_best_model_path):
                            os.remove(old_best_model_path)
                    best_loss = total_loss
                    best_epoch = i
                    best_model_path = f"{cfg.base_dir}/train/best_{best_epoch}.pth"
                    torch.save(cur_model, best_model_path)
                    logger.info(
                        f"[Train-Batched]: Latest best model saved: epoch {best_epoch}, "
                        f"loss {best_loss}"
                    )

                torch.save(cur_model, f"{cfg.base_dir}/train/iter_{i}.pth")

            if need_instance_vis:
                self._visualize_batch_instances(
                    segment_batches=segment_batches,
                    expected_segment_len=expected_segment_len,
                    iteration=i,
                )

            total_iter_sec = time.perf_counter() - iter_wall_start
            self._append_timing_row(
                timing_csv_path=timing_csv_path,
                mode="batched",
                iteration=i,
                train_iter_sec=train_iter_sec,
                full_rollout_eval_sec=full_rollout_eval_sec,
                total_iter_sec=total_iter_sec,
                num_groups=len(segment_batches),
                batch_size=self.batch_size,
                segment_len=expected_segment_len,
                segment_stride=self.segment_stride,
                num_train_steps=total_steps,
            )
            logger.info(
                "[Train-Batched-Timing]: "
                f"Case: {cfg.run_name}, Iteration: {i}, "
                f"train_iter_sec={train_iter_sec:.3f}, "
                f"full_rollout_eval_sec={full_rollout_eval_sec:.3f}, "
                f"total_iter_sec={total_iter_sec:.3f}, "
                f"steps={total_steps}"
            )

        final_frame_loss_rows = []
        self._append_full_rollout_loss_rows(
            iteration=cfg.iterations,
            total_frames=total_frames,
            first_batch_data=segment_batches[0],
            frame_loss_rows=final_frame_loss_rows,
        )
        if len(final_frame_loss_rows) > 0:
            with open(per_frame_loss_csv_path, "a", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerows(final_frame_loss_rows)

        wandb.finish()

    def _atomic_save_npz(self, output_path, **arrays):
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        tmp_path = output_path + ".tmp"
        with open(tmp_path, "wb") as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, output_path)

    def _atomic_save_json(self, output_path, data):
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        tmp_path = output_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, output_path)

    def _write_realtime_manifest(
        self,
        realtime_vis_dir,
        iteration,
        arrays,
    ):
        iterations_dir = os.path.join(realtime_vis_dir, "iterations")
        iteration_files = []
        if os.path.isdir(iterations_dir):
            iteration_files = sorted(
                int(path[len("iter_") : -len(".npz")])
                for path in os.listdir(iterations_dir)
                if path.startswith("iter_") and path.endswith(".npz")
            )

        manifest = {
            "case_name": cfg.run_name,
            "latest_iteration": int(iteration),
            "iterations": iteration_files,
            "latest_file": "latest_window.npz",
            "iterations_dir": "iterations",
            "first_seen_dir": "first_seen",
            "fps": int(cfg.FPS),
            "image_width": int(cfg.WH[0]) if cfg.WH is not None else None,
            "image_height": int(cfg.WH[1]) if cfg.WH is not None else None,
            "window_starts": arrays["window_starts"].astype(int).tolist(),
            "segment_len": int(arrays["segment_len"]),
            "num_original_points": int(arrays["num_original_points"]),
            "num_surface_points": int(arrays["num_surface_points"]),
            "num_all_points": int(arrays["num_all_points"]),
            "timestamp": float(arrays["timestamp"]),
        }
        self._atomic_save_json(
            os.path.join(realtime_vis_dir, "manifest.json"),
            manifest,
        )

    def _export_realtime_windows(
        self,
        realtime_vis_dir,
        iteration,
        window_starts,
        batch_data,
        pred_windows,
        write_latest=True,
        keep_iteration_history=True,
    ):
        window_starts = np.asarray(window_starts, dtype=np.int64)
        pred_points = np.stack(pred_windows, axis=1).astype(np.float32)
        online_frame_indices = window_starts[:, None] + np.arange(
            pred_points.shape[1], dtype=np.int64
        )[None, :]
        frame_indices = online_frame_indices
        if self.source_frame_indices is not None:
            source_frame_indices = self.source_frame_indices
            if torch.is_tensor(source_frame_indices):
                source_frame_indices = (
                    source_frame_indices.detach().cpu().numpy().astype(np.int64)
                )
            else:
                source_frame_indices = np.asarray(source_frame_indices, dtype=np.int64)
            max_online_idx = int(online_frame_indices.max())
            if source_frame_indices.shape[0] <= max_online_idx:
                raise ValueError(
                    "source_frame_indices is shorter than realtime export indices: "
                    f"len={source_frame_indices.shape[0]}, max_idx={max_online_idx}"
                )
            frame_indices = source_frame_indices[online_frame_indices]

        gt_object_points = np.stack(
            [
                self.train_object_points[
                    int(start_idx) : int(start_idx) + pred_points.shape[1]
                ]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
                for start_idx in window_starts
            ],
            axis=0,
        )
        object_colors = np.empty((0, 0, 0, 3), dtype=np.float32)
        if self.object_colors is not None:
            object_colors = np.stack(
                [
                    self.object_colors[
                        int(start_idx) : int(start_idx) + pred_points.shape[1]
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                    for start_idx in window_starts
                ],
                axis=0,
            )

        object_visibilities = np.empty((0, 0, 0), dtype=np.bool_)
        if self.train_object_visibilities is not None:
            object_visibilities = np.stack(
                [
                    self.train_object_visibilities[
                        int(start_idx) : int(start_idx) + pred_points.shape[1]
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.bool_)
                    for start_idx in window_starts
                ],
                axis=0,
            )

        controller_points = np.empty(
            (pred_points.shape[0], pred_points.shape[1], 0, 3), dtype=np.float32
        )
        if batch_data["controller_points"] is not None:
            num_ctrl = int(self.controller_points.shape[1])
            ctrl = batch_data["controller_points"]
            controller_points = (
                ctrl.reshape(ctrl.shape[0], int(batch_data["batch_size"]), num_ctrl, 3)[
                    : pred_points.shape[1], : pred_points.shape[0]
                ]
                .permute(1, 0, 2, 3)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        latest_path = os.path.join(realtime_vis_dir, "latest_window.npz")
        arrays = {
            "iteration": np.array(int(iteration), dtype=np.int64),
            "window_starts": window_starts,
            "frame_indices": frame_indices,
            "online_frame_indices": online_frame_indices,
            "pred_points": pred_points,
            "gt_object_points": gt_object_points,
            "object_colors": object_colors,
            "object_visibilities": object_visibilities,
            "controller_points": controller_points,
            "num_original_points": np.array(
                int(self.num_original_points), dtype=np.int64
            ),
            "num_surface_points": np.array(
                int(self.num_surface_points), dtype=np.int64
            ),
            "num_all_points": np.array(int(self.num_all_points), dtype=np.int64),
            "batch_size": np.array(int(batch_data["batch_size"]), dtype=np.int64),
            "real_window_count": np.array(
                int(pred_points.shape[0]), dtype=np.int64
            ),
            "segment_len": np.array(int(pred_points.shape[1]), dtype=np.int64),
            "timestamp": np.array(float(time.time()), dtype=np.float64),
        }

        first_seen_dir = os.path.join(realtime_vis_dir, "first_seen")
        for window_idx, start in enumerate(window_starts):
            first_seen_path = os.path.join(
                first_seen_dir, f"window_{int(start):06d}.npz"
            )
            if os.path.exists(first_seen_path):
                continue
            self._atomic_save_npz(
                first_seen_path,
                iteration=np.array(int(iteration), dtype=np.int64),
                first_iteration=np.array(int(iteration), dtype=np.int64),
                window_start=np.array(int(start), dtype=np.int64),
                frame_indices=frame_indices[window_idx],
                online_frame_indices=online_frame_indices[window_idx],
                pred_points=pred_points[window_idx],
                gt_object_points=gt_object_points[window_idx],
                object_colors=(
                    object_colors[window_idx]
                    if object_colors.shape[0] == pred_points.shape[0]
                    else object_colors
                ),
                object_visibilities=(
                    object_visibilities[window_idx]
                    if object_visibilities.shape[0] == pred_points.shape[0]
                    else object_visibilities
                ),
                controller_points=controller_points[window_idx],
                num_original_points=arrays["num_original_points"],
                num_surface_points=arrays["num_surface_points"],
                num_all_points=arrays["num_all_points"],
                segment_len=arrays["segment_len"],
                timestamp=arrays["timestamp"],
            )
            logger.info(
                "[Train-Online-Realtime]: saved first-seen snapshot "
                f"window_start={int(start)}, iteration={int(iteration)}"
            )

        if write_latest:
            if keep_iteration_history:
                iteration_path = os.path.join(
                    realtime_vis_dir,
                    "iterations",
                    f"iter_{int(iteration):06d}.npz",
                )
                self._atomic_save_npz(iteration_path, **arrays)
            self._atomic_save_npz(latest_path, **arrays)
            self._write_realtime_manifest(
                realtime_vis_dir=realtime_vis_dir,
                iteration=iteration,
                arrays=arrays,
            )

    def train_online_batched(
        self,
        online_reader,
        online_buffer,
        start_epoch=-1,
        poll_sec=1.0,
        recent_window_count=8,
        checkpoint_interval=None,
        stop_when_finished=False,
        save_video=False,
        realtime_vis_dir=None,
        realtime_vis_every=1,
        realtime_keep_iterations=True,
        sample_recent=True,
    ):
        if cfg.data_type != "real":
            raise ValueError("train_online_batched currently supports real data only")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if checkpoint_interval is None:
            checkpoint_interval = int(cfg.vis_interval)
        checkpoint_interval = int(checkpoint_interval)
        recent_window_count = max(int(recent_window_count), int(self.batch_size))
        realtime_vis_every = max(1, int(realtime_vis_every))
        poll_sec = max(0.0, float(poll_sec))
        if realtime_vis_dir is not None:
            os.makedirs(realtime_vis_dir, exist_ok=True)
            latest_path = os.path.join(realtime_vis_dir, "latest_window.npz")
            first_seen_dir = os.path.join(realtime_vis_dir, "first_seen")
            iterations_dir = os.path.join(realtime_vis_dir, "iterations")
            manifest_path = os.path.join(realtime_vis_dir, "manifest.json")
            if start_epoch < 0:
                if os.path.exists(latest_path):
                    os.remove(latest_path)
                if os.path.exists(manifest_path):
                    os.remove(manifest_path)
                if os.path.isdir(first_seen_dir):
                    shutil.rmtree(first_seen_dir)
                if os.path.isdir(iterations_dir):
                    shutil.rmtree(iterations_dir)

        timing_csv_path = self._prepare_timing_csv(start_epoch)
        best_loss = None
        best_epoch = None
        rng = np.random.default_rng(42)
        i = start_epoch + 1

        logger.info(
            "[Train-Online]: start online batched training, "
            f"batch_size={self.batch_size}, segment_len={self.segment_len}, "
            f"segment_stride={self.segment_stride}, recent_window_count={recent_window_count}"
        )

        while i < cfg.iterations:
            iter_wall_start = time.perf_counter()

            new_chunks = online_reader.load_new_chunks()
            if len(new_chunks) > 0:
                online_buffer.append_chunks(new_chunks)
                online_buffer.sync_to_device(cfg.device)
                self.refresh_real_data_from_dataset()
                logger.info(
                    "[Train-Online]: refreshed online data, "
                    f"frames={self._get_trainable_total_frames()}, "
                    f"last_chunk={online_reader.last_loaded_chunk}"
                )

            total_frames = self._get_trainable_total_frames()
            if total_frames < int(self.segment_len):
                if online_reader.is_finished:
                    raise RuntimeError(
                        "Online stream finished before enough frames were available "
                        f"for segment_len={self.segment_len}"
                    )
                logger.info(
                    "[Train-Online]: waiting for frames, "
                    f"available={total_frames}, need={self.segment_len}"
                )
                time.sleep(poll_sec)
                continue

            segment_starts = self._compute_segment_start_indices(total_frames)
            if len(segment_starts) == 0:
                if online_reader.is_finished:
                    raise RuntimeError(
                        "Online stream finished before any trainable windows were available"
                    )
                logger.info(
                    "[Train-Online]: waiting for windows, "
                    f"available={len(segment_starts)}, need=1"
                )
                time.sleep(poll_sec)
                continue

            recent_starts = segment_starts[-recent_window_count:]
            real_batch_size = min(int(self.batch_size), len(recent_starts))
            if sample_recent and len(recent_starts) > real_batch_size:
                selected = rng.choice(
                    np.asarray(recent_starts, dtype=np.int64),
                    size=int(real_batch_size),
                    replace=False,
                )
                real_start_indices = sorted(int(v) for v in selected.tolist())
            else:
                real_start_indices = recent_starts[-int(real_batch_size) :]

            if len(real_start_indices) == 0:
                raise RuntimeError("No real online windows selected for training")
            padded_start_indices = list(real_start_indices)
            while len(padded_start_indices) < int(self.batch_size):
                padded_start_indices.append(real_start_indices[-1])
            loss_weights = torch.zeros(
                int(self.batch_size), dtype=torch.float32, device=cfg.device
            )
            loss_weights[:real_batch_size] = 1.0

            batch_data = self._build_segment_batch_tensors(padded_start_indices)
            expected_segment_len = int(batch_data["segment_len"])
            actual_batch_size = int(batch_data["batch_size"])
            if actual_batch_size != int(self.batch_size):
                raise RuntimeError(
                    f"Expected padded online batch size {self.batch_size}, "
                    f"got {actual_batch_size}"
                )

            if (
                self.batch_simulator is None
                or self.batch_size_loaded != self.batch_size
                or self.batch_segment_len_loaded != expected_segment_len
            ):
                logger.info(
                    "[Train-Online]: build batched simulator with "
                    f"batch_size={self.batch_size}, "
                    f"segment_len={expected_segment_len}"
                )
                self._build_batched_simulator_for_train(batch_data=batch_data)
                sim = self.batch_simulator
                self.optimizer = torch.optim.Adam(
                    [
                        wp.to_torch(sim.wp_spring_Y),
                        wp.to_torch(sim.wp_collide_elas),
                        wp.to_torch(sim.wp_collide_fric),
                        wp.to_torch(sim.wp_collide_object_elas),
                        wp.to_torch(sim.wp_collide_object_fric),
                    ],
                    lr=cfg.base_lr,
                    betas=(0.9, 0.99),
                )

            sim = self.batch_simulator
            train_iter_start = time.perf_counter()
            total_loss = 0.0
            total_steps = 0
            total_chamfer_loss = 0.0
            total_track_loss = 0.0
            export_realtime = realtime_vis_dir is not None and (
                i % realtime_vis_every == 0 or i == cfg.iterations - 1
            )
            first_seen_needed = False
            if realtime_vis_dir is not None:
                first_seen_dir = os.path.join(realtime_vis_dir, "first_seen")
                first_seen_needed = any(
                    not os.path.exists(
                        os.path.join(
                            first_seen_dir, f"window_{int(start):06d}.npz"
                        )
                    )
                    for start in real_start_indices
                )
            capture_realtime = export_realtime or first_seen_needed
            realtime_window_starts = None
            realtime_pred_windows = None
            if capture_realtime:
                realtime_window_starts = np.asarray(real_start_indices, dtype=np.int64)
                realtime_pred_windows = [
                    batch_data["init_object_vertices_by_batch"][
                        : int(real_batch_size)
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                ]

            with wp.ScopedTimer("backward_online_batched"):
                sim.set_loss_weights(loss_weights)
                self._load_segment_batch_into_sim(sim, batch_data)
                sim.set_init_state(
                    wp.from_torch(
                        batch_data["init_object_vertices_batched"],
                        dtype=wp.vec3,
                        requires_grad=False,
                    ),
                    wp.from_torch(
                        batch_data["init_velocities_batched"],
                        dtype=wp.vec3,
                        requires_grad=False,
                    ),
                )

                for j in range(1, expected_segment_len):
                    sim.set_controller_target(j)
                    if sim.object_collision_flag:
                        sim.update_collision_graph()

                    if cfg.use_graph:
                        wp.capture_launch(sim.graph)
                    else:
                        with sim.tape:
                            sim.step()
                            sim.calculate_loss()
                        sim.tape.backward(sim.loss)

                    self.optimizer.step()

                    chamfer_loss = wp.to_torch(
                        sim.chamfer_loss, requires_grad=False
                    )
                    track_loss = wp.to_torch(sim.track_loss, requires_grad=False)
                    loss = wp.to_torch(sim.loss, requires_grad=False)
                    total_chamfer_loss += chamfer_loss.item()
                    total_track_loss += track_loss.item()
                    total_loss += loss.item()
                    total_steps += 1
                    if capture_realtime:
                        state_x = wp.to_torch(
                            sim.wp_states[-1].wp_x, requires_grad=False
                        )
                        state_x = state_x.reshape(
                            int(self.batch_size), int(self.num_all_points), 3
                        )
                        realtime_pred_windows.append(
                            state_x[: int(real_batch_size)]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )

                    if cfg.use_graph:
                        sim.tape.zero()
                    else:
                        sim.tape.reset()
                    sim.clear_loss()
                    sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v)

            train_iter_sec = time.perf_counter() - train_iter_start
            if total_steps == 0:
                raise RuntimeError("No online training steps were executed")

            total_loss /= total_steps
            total_chamfer_loss /= total_steps
            total_track_loss /= total_steps
            if capture_realtime:
                self._export_realtime_windows(
                    realtime_vis_dir=realtime_vis_dir,
                    iteration=i,
                    window_starts=realtime_window_starts,
                    batch_data=batch_data,
                    pred_windows=realtime_pred_windows,
                    write_latest=export_realtime,
                    keep_iteration_history=realtime_keep_iterations,
                )

            wandb.log(
                {
                    "loss": total_loss,
                    "chamfer_loss": total_chamfer_loss,
                    "track_loss": total_track_loss,
                    "online_frames": total_frames,
                    "online_windows": len(segment_starts),
                    "online_real_batch_size": real_batch_size,
                    "online_target_batch_size": self.batch_size,
                    "online_padded_batch_lanes": self.batch_size - real_batch_size,
                    "online_last_chunk": online_reader.last_loaded_chunk,
                    "collide_else": wp.to_torch(
                        sim.wp_collide_elas, requires_grad=False
                    ).item(),
                    "collide_fric": wp.to_torch(
                        sim.wp_collide_fric, requires_grad=False
                    ).item(),
                    "collide_object_elas": wp.to_torch(
                        sim.wp_collide_object_elas, requires_grad=False
                    ).item(),
                    "collide_object_fric": wp.to_torch(
                        sim.wp_collide_object_fric, requires_grad=False
                    ).item(),
                },
                step=i,
            )

            logger.info(
                f"[Train-Online]: Case: {cfg.run_name}, Iteration: {i}, "
                f"Loss: {total_loss}, frames={total_frames}, "
                f"batch={real_batch_size}/{self.batch_size}, "
                f"starts={real_start_indices}, padded_starts={padded_start_indices}"
            )

            should_save = checkpoint_interval > 0 and (
                i % checkpoint_interval == 0 or i == cfg.iterations - 1
            )
            if should_save:
                self._sync_single_simulator_from_batch()
                if save_video:
                    video_path = f"{cfg.base_dir}/train/online_sim_iter{i}.mp4"
                    self.visualize_sim(save_only=True, video_path=video_path)
                    wandb.log(
                        {
                            "video": wandb.Video(
                                video_path,
                                format="mp4",
                                fps=cfg.FPS,
                            ),
                        },
                        step=i,
                    )

                cur_model = {
                    "epoch": i,
                    "num_object_springs": self.num_object_springs,
                    "spring_Y": torch.exp(
                        wp.to_torch(sim.wp_spring_Y, requires_grad=False)
                    ),
                    "collide_elas": wp.to_torch(
                        sim.wp_collide_elas, requires_grad=False
                    ),
                    "collide_fric": wp.to_torch(
                        sim.wp_collide_fric, requires_grad=False
                    ),
                    "collide_object_elas": wp.to_torch(
                        sim.wp_collide_object_elas, requires_grad=False
                    ),
                    "collide_object_fric": wp.to_torch(
                        sim.wp_collide_object_fric, requires_grad=False
                    ),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "online_frames": total_frames,
                    "online_last_chunk": online_reader.last_loaded_chunk,
                }

                if best_loss is None or total_loss < best_loss:
                    if best_loss is not None:
                        old_best_model_path = (
                            f"{cfg.base_dir}/train/best_{best_epoch}.pth"
                        )
                        if os.path.exists(old_best_model_path):
                            os.remove(old_best_model_path)
                    best_loss = total_loss
                    best_epoch = i
                    best_model_path = f"{cfg.base_dir}/train/best_{best_epoch}.pth"
                    torch.save(cur_model, best_model_path)
                    logger.info(
                        f"[Train-Online]: latest best model saved: epoch={best_epoch}, "
                        f"loss={best_loss}"
                    )

                torch.save(cur_model, f"{cfg.base_dir}/train/iter_{i}.pth")

            total_iter_sec = time.perf_counter() - iter_wall_start
            self._append_timing_row(
                timing_csv_path=timing_csv_path,
                mode="online_batched",
                iteration=i,
                train_iter_sec=train_iter_sec,
                full_rollout_eval_sec=0.0,
                total_iter_sec=total_iter_sec,
                num_groups=1,
                batch_size=self.batch_size,
                segment_len=expected_segment_len,
                segment_stride=self.segment_stride,
                num_train_steps=total_steps,
            )

            i += 1
            if stop_when_finished and online_reader.is_finished:
                logger.info("[Train-Online]: stream finished; stopping training")
                break

        wandb.finish()

    def test(self, model_path=None):
        if model_path is not None:
            # Load the model
            logger.info(f"Load model from {model_path}")
            checkpoint = torch.load(model_path, map_location=cfg.device)

            spring_Y = checkpoint["spring_Y"]
            collide_elas = checkpoint["collide_elas"]
            collide_fric = checkpoint["collide_fric"]
            collide_object_elas = checkpoint["collide_object_elas"]
            collide_object_fric = checkpoint["collide_object_fric"]
            num_object_springs = checkpoint["num_object_springs"]

            assert (
                len(spring_Y) == self.simulator.n_springs
            ), "Check if the loaded checkpoint match the config file to connect the springs"

            self.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
            self.simulator.set_collide(
                collide_elas.detach().clone(), collide_fric.detach().clone()
            )
            self.simulator.set_collide_object(
                collide_object_elas.detach().clone(),
                collide_object_fric.detach().clone(),
            )

        # Render the initial visualization
        video_path = f"{cfg.base_dir}/inference.mp4"
        save_path = f"{cfg.base_dir}/inference.pkl"
        self.visualize_sim(
            save_only=True,
            video_path=video_path,
            save_trajectory=True,
            save_path=save_path,
        )

    def visualize_sim(
        self, save_only=True, video_path=None, save_trajectory=False, save_path=None
    ):
        logger.info("Visualizing the simulation")
        # Visualize the whole simulation using current set of parameters in the physical simulator
        frame_len = self.dataset.frame_len
        self.simulator.set_init_state(
            self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
        )
        vertices = [
            wp.to_torch(self.simulator.wp_states[0].wp_x, requires_grad=False).cpu()
        ]

        with wp.ScopedTimer("simulate"):
            for i in tqdm(range(1, frame_len)):
                if cfg.data_type == "real":
                    self.simulator.set_controller_target(i, pure_inference=True)
                if self.simulator.object_collision_flag:
                    self.simulator.update_collision_graph()

                if cfg.use_graph:
                    wp.capture_launch(self.simulator.forward_graph)
                else:
                    self.simulator.step()
                x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
                vertices.append(x.cpu())
                # Set the intial state for the next step
                self.simulator.set_init_state(
                    self.simulator.wp_states[-1].wp_x,
                    self.simulator.wp_states[-1].wp_v,
                )

        vertices = torch.stack(vertices, dim=0)

        if save_trajectory:
            logger.info(f"Save the trajectory to {save_path}")
            vertices_to_save = vertices.cpu().numpy()
            with open(save_path, "wb") as f:
                pickle.dump(vertices_to_save, f)

        if not save_only:
            visualize_pc(
                vertices[:, : self.num_all_points, :],
                self.object_colors,
                self.controller_points,
                visualize=True,
            )
        else:
            assert video_path is not None, "Please provide the video path to save"
            visualize_pc(
                vertices[:, : self.num_all_points, :],
                self.object_colors,
                self.controller_points,
                visualize=False,
                save_video=True,
                save_path=video_path,
            )

    def on_press(self, key):
        try:
            self.pressed_keys.add(key.char)
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            self.pressed_keys.remove(key.char)
        except (KeyError, AttributeError):
            try:
                self.pressed_keys.remove(str(key))
            except KeyError:
                pass

    def get_target_change(self):
        target_change = np.zeros((self.n_ctrl_parts, 3))
        for key in self.pressed_keys:
            if key in self.key_mappings:
                idx, change = self.key_mappings[key]
                target_change[idx] += change
        return target_change

    def init_control_ui(self):

        height = cfg.WH[1]
        width = cfg.WH[0]

        self.arrow_size = 30

        self.arrow_empty_orig = cv2.imread(
            "./assets/arrow_empty.png", cv2.IMREAD_UNCHANGED
        )[:, :, [2, 1, 0, 3]]
        self.arrow_1_orig = cv2.imread("./assets/arrow_1.png", cv2.IMREAD_UNCHANGED)[
            :, :, [2, 1, 0, 3]
        ]
        self.arrow_2_orig = cv2.imread("./assets/arrow_2.png", cv2.IMREAD_UNCHANGED)[
            :, :, [2, 1, 0, 3]
        ]

        spacing = self.arrow_size + 5

        self.bottom_margin = 25  # Margin from bottom of screen
        bottom_y = height - self.bottom_margin
        top_y = height - self.bottom_margin - spacing

        self.edge_buffer = self.bottom_margin
        set1_margin_x = self.edge_buffer  # Add buffer from left edge
        set2_margin_x = width - self.edge_buffer

        self.arrow_positions_set1 = {
            "q": (set1_margin_x + spacing * 3, top_y),  # Up
            "w": (set1_margin_x + spacing, top_y),  # Forward
            "a": (set1_margin_x, bottom_y),  # Left
            "s": (set1_margin_x + spacing, bottom_y),  # Backward
            "d": (set1_margin_x + spacing * 2, bottom_y),  # Right
            "e": (set1_margin_x + spacing * 3, bottom_y),  # Down
        }

        self.arrow_positions_set2 = {
            "u": (set2_margin_x - spacing * 3, top_y),  # Up
            "i": (set2_margin_x - spacing * 1, top_y),  # Forward
            "j": (set2_margin_x - spacing * 2, bottom_y),  # Left
            "k": (set2_margin_x - spacing * 1, bottom_y),  # Backward
            "l": (set2_margin_x, bottom_y),  # Right
            "o": (set2_margin_x - spacing * 3, bottom_y),  # Down
        }

        self.interm_size = 512
        self.rotations = {
            "w": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 0, 1
            ),  # Forward
            "a": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 90, 1
            ),  # Left
            "s": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 180, 1
            ),  # Backward
            "d": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 270, 1
            ),  # Right
            "q": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 0, 1
            ),  # Up
            "e": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 180, 1
            ),  # Down
            "i": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 0, 1
            ),  # Forward
            "j": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 90, 1
            ),  # Left
            "k": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 180, 1
            ),  # Backward
            "l": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 270, 1
            ),  # Right
            "u": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 0, 1
            ),  # Up
            "o": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 180, 1
            ),  # Down
        }

        self.hand_left = cv2.imread("./assets/Picture2.png", cv2.IMREAD_UNCHANGED)[
            :, :, [2, 1, 0, 3]
        ]
        self.hand_right = cv2.imread("./assets/Picture1.png", cv2.IMREAD_UNCHANGED)[
            :, :, [2, 1, 0, 3]
        ]

        self.hand_left_pos = torch.tensor([0.0, 0.0, 0.0], device=cfg.device)
        self.hand_right_pos = torch.tensor([0.0, 0.0, 0.0], device=cfg.device)

        # pre-compute all rotated arrows to avoid aliasing
        self.arrow_rotated_filled = {}
        self.arrow_rotated_empty = {}
        for key in self.arrow_positions_set1:
            self.arrow_rotated_filled[key] = cv2.resize(
                self._rotate_arrow(
                    cv2.resize(
                        self.arrow_1_orig,
                        (self.interm_size, self.interm_size),
                        interpolation=cv2.INTER_AREA,
                    ),
                    key,
                ),
                (self.arrow_size, self.arrow_size),
                interpolation=cv2.INTER_AREA,
            )
            self.arrow_rotated_empty[key] = cv2.resize(
                self._rotate_arrow(
                    cv2.resize(
                        self.arrow_empty_orig,
                        (self.interm_size, self.interm_size),
                        interpolation=cv2.INTER_AREA,
                    ),
                    key,
                ),
                (self.arrow_size, self.arrow_size),
                interpolation=cv2.INTER_AREA,
            )
        for key in self.arrow_positions_set2:
            self.arrow_rotated_filled[key] = cv2.resize(
                self._rotate_arrow(
                    cv2.resize(
                        self.arrow_2_orig,
                        (self.interm_size, self.interm_size),
                        interpolation=cv2.INTER_AREA,
                    ),
                    key,
                ),
                (self.arrow_size, self.arrow_size),
                interpolation=cv2.INTER_AREA,
            )
            self.arrow_rotated_empty[key] = cv2.resize(
                self._rotate_arrow(
                    cv2.resize(
                        self.arrow_empty_orig,
                        (self.interm_size, self.interm_size),
                        interpolation=cv2.INTER_AREA,
                    ),
                    key,
                ),
                (self.arrow_size, self.arrow_size),
                interpolation=cv2.INTER_AREA,
            )

    def _rotate_arrow(self, arrow, key):
        rotation_matrix = self.rotations[key]
        rotated = cv2.warpAffine(
            arrow,
            rotation_matrix,
            (self.interm_size, self.interm_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_TRANSPARENT,
        )
        return rotated

    def _overlay_arrow(self, background, arrow, position, key, filled=True):
        x, y = position

        if filled:
            rotated_arrow = self.arrow_rotated_filled[key].copy()
        else:
            rotated_arrow = self.arrow_rotated_empty[key].copy()

        h, w = rotated_arrow.shape[:2]

        roi_x = max(0, x - w // 2)
        roi_y = max(0, y - h // 2)
        roi_w = min(w, background.shape[1] - roi_x)
        roi_h = min(h, background.shape[0] - roi_y)

        arrow_x = max(0, w // 2 - x)
        arrow_y = max(0, h // 2 - y)

        roi = background[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]

        arrow_roi = rotated_arrow[arrow_y : arrow_y + roi_h, arrow_x : arrow_x + roi_w]

        alpha = arrow_roi[:, :, 3] / 255.0

        for c in range(3):  # Apply for RGB channels
            roi[:, :, c] = roi[:, :, c] * (1 - alpha) + arrow_roi[:, :, c] * alpha

        background[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w] = roi

        return background

    def _overlay_hand_at_position(
        self, frame, target_points, x_axis, hand_size, hand_icon, align="center"
    ):
        result = frame.copy()

        mean_pos = target_points.cpu().numpy().mean(axis=0)

        pixel_mean = self.projection @ np.append(mean_pos, 1)
        pixel_mean = pixel_mean[:2] / pixel_mean[2]

        pos_1 = np.append(mean_pos + hand_size * x_axis, 1)
        pixel_1 = self.projection @ pos_1
        pixel_1 = pixel_1[:2] / pixel_1[2]

        pos_2 = np.append(mean_pos - hand_size * x_axis, 1)
        pixel_2 = self.projection @ pos_2
        pixel_2 = pixel_2[:2] / pixel_2[2]

        icon_size = int(np.linalg.norm(pixel_1[:2] - pixel_2[:2]) / 2)
        icon_size = max(1, min(icon_size, 100))

        resized_icon = cv2.resize(hand_icon, (icon_size, icon_size))
        h, w = resized_icon.shape[:2]
        x, y = int(pixel_mean[0]), int(pixel_mean[1])

        if align == "top-left":
            roi_x = int(max(0, x - w * 0.15))
            roi_y = int(max(0, y - h * 0.1))
        if align == "top-right":
            roi_x = int(max(0, x - w + w * 0.15))
            roi_y = int(max(0, y - h * 0.1))
        if align == "center":
            roi_x = int(max(0, x - w // 2))
            roi_y = int(max(0, y - h // 2))
        roi_w = min(w, result.shape[1] - roi_x)
        roi_h = min(h, result.shape[0] - roi_y)

        if roi_w <= 0 or roi_h <= 0:
            return result

        icon_x = max(0, w // 2 - x)
        icon_y = max(0, h // 2 - y)

        roi = result[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
        icon_roi = resized_icon[icon_y : icon_y + roi_h, icon_x : icon_x + roi_w]

        if icon_roi.size == 0 or roi.shape[:2] != icon_roi.shape[:2]:
            return result

        if icon_roi.shape[2] == 4:
            alpha = icon_roi[:, :, 3] / 255.0
            for c in range(3):
                roi[:, :, c] = roi[:, :, c] * (1 - alpha) + icon_roi[:, :, c] * alpha
            result[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w] = roi
        else:
            result[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w] = icon_roi[:, :, :3]

        return result

    def _overlay_hand_icons(self, frame):
        if self.n_ctrl_parts not in [1, 2]:
            raise ValueError("Only support 1 or 2 control parts")

        result = frame.copy()

        c2w = np.linalg.inv(self.w2c)
        x_axis = c2w[:3, 0]
        self.projection = self.intrinsic @ self.w2c[:3, :]
        hand_size = 0.1  # size in physical space (in meters)

        if self.n_ctrl_parts == 1:
            current_target = self.hand_left_pos.unsqueeze(0)
            # align = 'top-right'
            align = "center"
            result = self._overlay_hand_at_position(
                result, current_target, x_axis, hand_size, self.hand_left, align
            )
        else:
            for i in range(2):
                current_target = (
                    self.hand_left_pos.unsqueeze(0)
                    if i == 0
                    else self.hand_right_pos.unsqueeze(0)
                )
                # align = 'top-right' if i == 0 else 'top-left'
                align = "center"
                hand_icon = self.hand_left if i == 0 else self.hand_right
                result = self._overlay_hand_at_position(
                    result, current_target, x_axis, hand_size, hand_icon, align
                )

        return result

    def update_frame(self, frame, pressed_keys):
        result = frame.copy()

        result = self._overlay_hand_icons(result)

        # overlay an transparent white mask on the bottom left and bottom right corners with width trans_width, and height trans_height
        trans_width = 160
        trans_height = 120
        overlay = result.copy()

        bottom_left_pt1 = (0, cfg.WH[1] - trans_height)
        bottom_left_pt2 = (trans_width, cfg.WH[1])
        cv2.rectangle(overlay, bottom_left_pt1, bottom_left_pt2, (255, 255, 255), -1)

        if self.n_ctrl_parts == 2:
            bottom_right_pt1 = (cfg.WH[0] - trans_width, cfg.WH[1] - trans_height)
            bottom_right_pt2 = (cfg.WH[0], cfg.WH[1])
            cv2.rectangle(
                overlay, bottom_right_pt1, bottom_right_pt2, (255, 255, 255), -1
            )

        alpha = 0.6
        cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)

        # Draw all buttons for Set 1 (left side)
        for key, pos in self.arrow_positions_set1.items():
            if key in pressed_keys:
                result = self._overlay_arrow(result, None, pos, key, filled=True)
            else:
                result = self._overlay_arrow(result, None, pos, key, filled=False)

        # Draw all buttons for Set 2 (right side)
        if self.n_ctrl_parts == 2:
            for key, pos in self.arrow_positions_set2.items():
                if key in pressed_keys:
                    result = self._overlay_arrow(result, None, pos, key, filled=True)
                else:
                    result = self._overlay_arrow(result, None, pos, key, filled=False)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        control1_x = self.edge_buffer  # hard coded for now
        control2_x = cfg.WH[0] - self.edge_buffer - 113  # hard coded for now
        text_y = (
            cfg.WH[1] - self.arrow_size * 2 - self.bottom_margin - 10
        )  # hard coded for now
        cv2.putText(
            result,
            "Left Hand",
            (control1_x, text_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
        )
        if self.n_ctrl_parts == 2:
            cv2.putText(
                result,
                "Right Hand",
                (control2_x, text_y),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
            )

        return result

    def _find_closest_point(self, target_points):
        """Find the closest structure point to any of the target points."""
        dist_matrix = torch.sum(
            (target_points.unsqueeze(1) - self.structure_points.unsqueeze(0)) ** 2,
            dim=2,
        )
        min_dist_per_ctrl_pts, min_indices = torch.min(dist_matrix, dim=1)
        min_idx = min_indices[torch.argmin(min_dist_per_ctrl_pts)]
        return self.structure_points[min_idx].unsqueeze(0)

    def interactive_playground(
        self, model_path, gs_path, n_ctrl_parts=1, inv_ctrl=False, virtual_key_input=False
    ):
        # Load the model
        logger.info(f"Load model from {model_path}")
        checkpoint = torch.load(model_path, map_location=cfg.device)

        spring_Y = checkpoint["spring_Y"]
        collide_elas = checkpoint["collide_elas"]
        collide_fric = checkpoint["collide_fric"]
        collide_object_elas = checkpoint["collide_object_elas"]
        collide_object_fric = checkpoint["collide_object_fric"]
        num_object_springs = checkpoint["num_object_springs"]

        assert (
            len(spring_Y) == self.simulator.n_springs
        ), "Check if the loaded checkpoint match the config file to connect the springs"

        self.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
        self.simulator.set_collide(
            collide_elas.detach().clone(), collide_fric.detach().clone()
        )
        self.simulator.set_collide_object(
            collide_object_elas.detach().clone(),
            collide_object_fric.detach().clone(),
        )

        ###########################################################################

        logger.info("Party Time Start!!!!")
        self.simulator.set_init_state(
            self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
        )
        prev_x = wp.to_torch(
            self.simulator.wp_states[0].wp_x, requires_grad=False
        ).clone()

        vis_cam_idx = 0
        FPS = cfg.FPS
        width, height = cfg.WH
        intrinsic = cfg.intrinsics[vis_cam_idx]
        w2c = cfg.w2cs[vis_cam_idx]

        current_target = self.simulator.controller_points[0]
        prev_target = current_target

        vis_controller_points = current_target.cpu().numpy()

        gaussians = GaussianModel(sh_degree=3)
        gaussians.load_ply(gs_path)
        gaussians = remove_gaussians_with_low_opacity(gaussians, 0.1)
        gaussians.isotropic = True
        current_pos = gaussians.get_xyz
        current_rot = gaussians.get_rotation
        use_white_background = True  # set to True for white background
        bg_color = [1, 1, 1] if use_white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        view = self._create_gs_view(w2c, intrinsic, height, width)
        prev_x = None
        relations = None
        weights = None
        image_path = cfg.bg_img_path
        overlay = cv2.imread(image_path)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        overlay = torch.tensor(overlay, dtype=torch.float32, device=cfg.device)

        if n_ctrl_parts > 1:
            kmeans = KMeans(n_clusters=n_ctrl_parts, random_state=0, n_init=10)
            cluster_labels = kmeans.fit_predict(vis_controller_points)
            N = vis_controller_points.shape[0]
            masks_ctrl_pts = []
            for i in range(n_ctrl_parts):
                mask = cluster_labels == i
                masks_ctrl_pts.append(torch.from_numpy(mask))
            # project the center of the cluster to the object to the image space, those on the left will be mask 1
            center1 = np.mean(vis_controller_points[masks_ctrl_pts[0]], axis=0)
            center2 = np.mean(vis_controller_points[masks_ctrl_pts[1]], axis=0)
            center1 = np.concatenate([center1, [1]])
            center2 = np.concatenate([center2, [1]])
            proj_mat = intrinsic @ w2c[:3, :]
            center1 = proj_mat @ center1
            center2 = proj_mat @ center2
            center1 = center1 / center1[-1]
            center2 = center2 / center2[-1]
            if center1[0] > center2[0]:
                print("Switching the control parts")
                masks_ctrl_pts = [masks_ctrl_pts[1], masks_ctrl_pts[0]]
        else:
            masks_ctrl_pts = None
        self.n_ctrl_parts = n_ctrl_parts
        self.mask_ctrl_pts = masks_ctrl_pts
        self.scale_factors = 1.0
        assert n_ctrl_parts <= 2, "Only support 1 or 2 control parts"
        print("UI Controls:")
        print("- Set 1: WASD (XY movement), QE (Z movement)")
        print("- Set 2: IJKL (XY movement), UO (Z movement)")
        self.inv_ctrl = -1.0 if inv_ctrl else 1.0
        self.key_mappings = {
            # Set 1 controls
            "w": (0, np.array([0.005, 0, 0]) * self.inv_ctrl),
            "s": (0, np.array([-0.005, 0, 0]) * self.inv_ctrl),
            "a": (0, np.array([0, -0.005, 0]) * self.inv_ctrl),
            "d": (0, np.array([0, 0.005, 0]) * self.inv_ctrl),
            "e": (0, np.array([0, 0, 0.005])),
            "q": (0, np.array([0, 0, -0.005])),
            # Set 2 controls
            "i": (1, np.array([0.005, 0, 0]) * self.inv_ctrl),
            "k": (1, np.array([-0.005, 0, 0]) * self.inv_ctrl),
            "j": (1, np.array([0, -0.005, 0]) * self.inv_ctrl),
            "l": (1, np.array([0, 0.005, 0]) * self.inv_ctrl),
            "o": (1, np.array([0, 0, 0.005])),
            "u": (1, np.array([0, 0, -0.005])),
        }
        self.pressed_keys = set()
        self.w2c = w2c
        self.intrinsic = intrinsic
        self.init_control_ui()
        if n_ctrl_parts > 1:
            hand_positions = []
            for i in range(2):
                target_points = torch.from_numpy(
                    vis_controller_points[self.mask_ctrl_pts[i]]
                ).to("cuda")
                hand_positions.append(self._find_closest_point(target_points))
            self.hand_left_pos, self.hand_right_pos = hand_positions
        else:
            target_points = torch.from_numpy(vis_controller_points).to("cuda")
            self.hand_left_pos = self._find_closest_point(target_points)

        if virtual_key_input:
            # Initialize keyboard tracking variables
            self.virtual_keys = {}     # Dictionary to track virtual keys with timestamps
            self.virtual_key_duration = 0.03  # Virtual key press duration in seconds
        
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        self.target_change = np.zeros((n_ctrl_parts, 3))

        ############## Temporary timer ##############
        import time

        class Timer:
            def __init__(self, name):
                self.name = name
                self.elapsed = 0
                self.start_time = None
                self.cuda_start_event = None
                self.cuda_end_event = None
                self.use_cuda = torch.cuda.is_available()

            def start(self):
                if self.use_cuda:
                    torch.cuda.synchronize()
                    self.cuda_start_event = torch.cuda.Event(enable_timing=True)
                    self.cuda_end_event = torch.cuda.Event(enable_timing=True)
                    self.cuda_start_event.record()
                self.start_time = time.time()

            def stop(self):
                if self.use_cuda:
                    self.cuda_end_event.record()
                    torch.cuda.synchronize()
                    self.elapsed = (
                        self.cuda_start_event.elapsed_time(self.cuda_end_event) / 1000
                    )  # convert ms to seconds
                else:
                    self.elapsed = time.time() - self.start_time
                return self.elapsed

            def reset(self):
                self.elapsed = 0
                self.start_time = None
                self.cuda_start_event = None
                self.cuda_end_event = None

        sim_timer = Timer("Simulator")
        render_timer = Timer("Rendering")
        frame_timer = Timer("Frame Compositing")
        interp_timer = Timer("Full Motion Interpolation")
        total_timer = Timer("Total Loop")
        knn_weights_timer = Timer("KNN Weights")
        motion_interp_timer = Timer("Motion Interpolation")

        # Performance stats
        fps_history = []
        component_times = {
            "simulator": [],
            "rendering": [],
            "frame_compositing": [],
            "full_motion_interpolation": [],
            "total": [],
            "knn_weights": [],
            "motion_interp": [],
        }

        # Number of frames to average over for stats
        STATS_WINDOW = 10
        frame_count = 0

        ############## End Temporary timer ##############

        while True:

            total_timer.start()

            # 1. Simulator step

            sim_timer.start()

            self.simulator.set_controller_interactive(prev_target, current_target)
            if self.simulator.object_collision_flag:
                self.simulator.update_collision_graph()
            wp.capture_launch(self.simulator.forward_graph)
            x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
            # Set the intial state for the next step
            self.simulator.set_init_state(
                self.simulator.wp_states[-1].wp_x,
                self.simulator.wp_states[-1].wp_v,
            )

            sim_time = sim_timer.stop()
            component_times["simulator"].append(sim_time)

            torch.cuda.synchronize()

            # 2. Frame initialization and setup

            frame_timer.start()

            frame = overlay.clone()

            frame_setup_time = (
                frame_timer.stop()
            )  # We'll accumulate times for frame compositing

            torch.cuda.synchronize()

            # 3. Rendering
            render_timer.start()

            # render with gaussians and paste the image on top of the frame
            results = render_gaussian(view, gaussians, None, background)
            rendering = results["render"]  # (4, H, W)
            image = rendering.permute(1, 2, 0).detach()

            render_time = render_timer.stop()
            component_times["rendering"].append(render_time)

            torch.cuda.synchronize()

            # Continue frame compositing
            frame_timer.start()

            image = image.clamp(0, 1)
            if use_white_background:
                image_mask = torch.logical_and(
                    (image != 1.0).any(dim=2), image[:, :, 3] > 100 / 255
                )
            else:
                image_mask = torch.logical_and(
                    (image != 0.0).any(dim=2), image[:, :, 3] > 100 / 255
                )
            image[..., 3].masked_fill_(~image_mask, 0.0)

            alpha = image[..., 3:4]
            rgb = image[..., :3] * 255
            frame = alpha * rgb + (1 - alpha) * frame
            frame = frame.cpu().numpy()
            image_mask = image_mask.cpu().numpy()
            frame = frame.astype(np.uint8)

            frame = self.update_frame(frame, self.pressed_keys)

            # Add shadows
            final_shadow = get_simple_shadow(
                x, intrinsic, w2c, width, height, image_mask, light_point=[0, 0, -3]
            )
            frame[final_shadow] = (frame[final_shadow] * 0.95).astype(np.uint8)
            final_shadow = get_simple_shadow(
                x, intrinsic, w2c, width, height, image_mask, light_point=[1, 0.5, -2]
            )
            frame[final_shadow] = (frame[final_shadow] * 0.97).astype(np.uint8)
            final_shadow = get_simple_shadow(
                x, intrinsic, w2c, width, height, image_mask, light_point=[-3, -0.5, -5]
            )
            frame[final_shadow] = (frame[final_shadow] * 0.98).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cv2.imshow("Interactive Playground", frame)
            key = cv2.waitKey(1)

            if virtual_key_input:
                # Handle virtual keyboard input through OpenCV window
                if key != -1:
                    key_char = chr(key & 0xFF).lower()
                    if key_char in self.key_mappings:
                        # Store virtual key with timestamp - refresh timestamp if already pressed
                        self.virtual_keys[key_char] = time.time()
                        self.pressed_keys.add(key_char)
                    elif key == 27:  # ESC key to exit
                        break
                
                # Process all keyboard inputs (both physical and virtual)
                # For virtual keys, check if they're still active based on timestamp
                current_time = time.time()
                keys_to_remove = []
                for k, press_time in self.virtual_keys.items():
                    if current_time - press_time > self.virtual_key_duration:
                        keys_to_remove.append(k)
                
                # Remove expired virtual keys
                for k in keys_to_remove:
                    if k in self.pressed_keys:
                        self.pressed_keys.discard(k)
                    if k in self.virtual_keys:
                        del self.virtual_keys[k]
            
            frame_comp_time = (
                frame_timer.stop() + frame_setup_time
            )  # Total frame compositing time
            component_times["frame_compositing"].append(frame_comp_time)

            torch.cuda.synchronize()

            if prev_x is not None:
                with torch.no_grad():

                    prev_particle_pos = prev_x
                    cur_particle_pos = x

                    if relations is None:
                        relations = get_topk_indices(
                            prev_x, K=16
                        )  # only computed in the first iteration

                    if weights is None:
                        weights, weights_indices = knn_weights_sparse(
                            prev_particle_pos, current_pos, K=16
                        )  # only computed in the first iteration

                    interp_timer.start()

                    weights = calc_weights_vals_from_indices(
                        prev_particle_pos, current_pos, weights_indices
                    )

                    current_pos, current_rot, _ = interpolate_motions_speedup(
                        bones=prev_particle_pos,
                        motions=cur_particle_pos - prev_particle_pos,
                        relations=relations,
                        weights=weights,
                        weights_indices=weights_indices,
                        xyz=current_pos,
                        quat=current_rot,
                    )

                    # update gaussians with the new positions and rotations
                    gaussians._xyz = current_pos
                    gaussians._rotation = current_rot

                interp_time = interp_timer.stop()
                component_times["full_motion_interpolation"].append(interp_time)

            torch.cuda.synchronize()

            prev_x = x.clone()

            prev_target = current_target
            target_change = self.get_target_change()
            if masks_ctrl_pts is not None:
                for i in range(n_ctrl_parts):
                    if masks_ctrl_pts[i].sum() > 0:
                        current_target[masks_ctrl_pts[i]] += torch.tensor(
                            target_change[i], dtype=torch.float32, device=cfg.device
                        )
                        if i == 0:
                            self.hand_left_pos += torch.tensor(
                                target_change[i], dtype=torch.float32, device=cfg.device
                            )
                        if i == 1:
                            self.hand_right_pos += torch.tensor(
                                target_change[i], dtype=torch.float32, device=cfg.device
                            )
            else:
                current_target += torch.tensor(
                    target_change, dtype=torch.float32, device=cfg.device
                )
                self.hand_left_pos += torch.tensor(
                    target_change, dtype=torch.float32, device=cfg.device
                )

            ############### Temporary timer ###############
            # Total loop time
            total_time = total_timer.stop()
            component_times["total"].append(total_time)

            # Calculate FPS
            fps = 1.0 / total_time
            fps_history.append(fps)

            # Display performance stats periodically
            frame_count += 1
            if frame_count % 10 == 0:
                # Limit stats to last STATS_WINDOW frames
                if len(fps_history) > STATS_WINDOW:
                    fps_history = fps_history[-STATS_WINDOW:]
                    for key in component_times:
                        component_times[key] = component_times[key][-STATS_WINDOW:]

                avg_fps = np.mean(fps_history)
                print(
                    f"\n--- Performance Stats (avg over last {len(fps_history)} frames) ---"
                )
                print(f"FPS: {avg_fps:.2f}")

                # Calculate percentages for pie chart
                total_avg = np.mean(component_times["total"])
                print(f"Total Frame Time: {total_avg*1000:.2f} ms")

                # Display individual component times
                for key in [
                    "simulator",
                    "rendering",
                    "frame_compositing",
                    "full_motion_interpolation",
                    "knn_weights",
                    "motion_interp",
                ]:
                    avg_time = np.mean(component_times[key])
                    percentage = (avg_time / total_avg) * 100
                    print(
                        f"{key.capitalize()}: {avg_time*1000:.2f} ms ({percentage:.1f}%)"
                    )

        listener.stop()

    def _transform_gs(self, gaussians, M, majority_scale=1):

        new_gaussians = copy.copy(gaussians)

        new_xyz = gaussians.get_xyz.clone()
        ones = torch.ones(
            (new_xyz.shape[0], 1), device=new_xyz.device, dtype=new_xyz.dtype
        )
        new_xyz = torch.cat((new_xyz, ones), dim=1)
        print("inside:", new_xyz.max(), new_xyz.min())
        new_xyz = new_xyz @ M.T
        print("outside:", new_xyz.max(), new_xyz.min())

        new_rotation = gaussians.get_rotation.clone()
        new_rotation = quaternion_multiply(
            matrix_to_quaternion(M[:3, :3]), new_rotation
        )

        new_scales = gaussians._scaling.clone()
        new_scales += torch.log(
            torch.tensor(
                majority_scale, device=new_scales.device, dtype=new_scales.dtype
            )
        )

        new_gaussians._xyz = new_xyz[:, :3]
        new_gaussians._rotation = new_rotation
        new_gaussians._scaling = new_scales

        return new_gaussians

    def _create_gs_view(self, w2c, intrinsic, height, width):
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]
        K = torch.tensor(intrinsic, dtype=torch.float32, device="cuda")
        focal_length_x = K[0, 0]
        focal_length_y = K[1, 1]
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)
        view = Camera(
            (width, height),
            colmap_id="0000",
            R=R,
            T=T,
            FoVx=FovX,
            FoVy=FovY,
            depth_params=None,
            image=None,
            invdepthmap=None,
            image_name="0000",
            uid="0000",
            data_device="cuda",
            train_test_exp=None,
            is_test_dataset=None,
            is_test_view=None,
            K=K,
            normal=None,
            depth=None,
            occ_mask=None,
        )
        return view

    def visualize_force(self, model_path, gs_path, n_ctrl_parts=2, force_scale=30000):
        # Load the model
        logger.info(f"Load model from {model_path}")
        checkpoint = torch.load(model_path, map_location=cfg.device)

        spring_Y = checkpoint["spring_Y"]
        collide_elas = checkpoint["collide_elas"]
        collide_fric = checkpoint["collide_fric"]
        collide_object_elas = checkpoint["collide_object_elas"]
        collide_object_fric = checkpoint["collide_object_fric"]
        num_object_springs = checkpoint["num_object_springs"]

        assert (
            len(spring_Y) == self.simulator.n_springs
        ), "Check if the loaded checkpoint match the config file to connect the springs"

        self.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
        self.simulator.set_collide(
            collide_elas.detach().clone(), collide_fric.detach().clone()
        )
        self.simulator.set_collide_object(
            collide_object_elas.detach().clone(),
            collide_object_fric.detach().clone(),
        )

        video_path = f"{cfg.base_dir}/force_visualization.mp4"

        vis_cam_idx = 0
        FPS = cfg.FPS
        width, height = cfg.WH
        intrinsic = cfg.intrinsics[vis_cam_idx]
        w2c = cfg.w2cs[vis_cam_idx]

        gaussians = GaussianModel(sh_degree=3)
        gaussians.load_ply(gs_path)
        gaussians = remove_gaussians_with_low_opacity(gaussians, 0.1)
        gaussians.isotropic = True
        current_pos = gaussians.get_xyz
        current_rot = gaussians.get_rotation
        use_white_background = True  # set to True for white background
        bg_color = [1, 1, 1] if use_white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=cfg.device)
        view = self._create_gs_view(w2c, intrinsic, height, width)
        prev_x = None
        relations = None
        weights = None

        # Get the controller points index
        first_frame_controller_points = self.simulator.controller_points[0]
        force_indexes = []
        if n_ctrl_parts == 1:
            force_indexes.append(
                torch.arange(first_frame_controller_points.shape[0], device=cfg.device)
            )
        else:
            # Use kmeans to find the two set of controller points
            kmeans = KMeans(n_clusters=n_ctrl_parts, random_state=0, n_init=10)
            cluster_labels = kmeans.fit_predict(
                first_frame_controller_points.cpu().numpy()
            )
            for i in range(n_ctrl_parts):
                force_indexes.append(
                    torch.tensor(np.where(cluster_labels == i)[0], device=cfg.device)
                )

        # Preprocess to get all the springs for different set of control points
        control_springs = self.init_springs[num_object_springs:]

        # Judge the springs whose left point is in the force_indexes
        force_springs = []
        force_object_points = []
        force_rest_lengths = []
        force_spring_Y = []

        for i in range(n_ctrl_parts):
            force_springs.append([])
            force_rest_lengths.append([])
            force_spring_Y.append([])
            force_object_points.append([])
            for j in range(len(control_springs)):
                if (control_springs[j][0] - self.num_all_points) in force_indexes[i]:
                    force_springs[i].append(control_springs[j])
                    force_rest_lengths[i].append(
                        self.init_rest_lengths[j + num_object_springs]
                    )
                    force_spring_Y[i].append(spring_Y[j + num_object_springs])
                    force_object_points[i].append(control_springs[j][1])
            force_springs[i] = torch.vstack(force_springs[i])
            force_springs[i][:, 0] -= self.num_all_points
            force_rest_lengths[i] = torch.tensor(
                force_rest_lengths[i], device=cfg.device
            )
            force_spring_Y[i] = torch.tensor(force_spring_Y[i], device=cfg.device)
            force_object_points[i] = torch.tensor(
                force_object_points[i], device=cfg.device
            )

        # Start to visualize the stuffs
        logger.info("Visualizing the simulation")
        # Visualize the whole simulation using current set of parameters in the physical simulator
        frame_len = self.dataset.frame_len
        self.simulator.set_init_state(
            self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
        )
        prev_x = wp.to_torch(
            self.simulator.wp_states[0].wp_x, requires_grad=False
        ).clone()

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=width, height=height)
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Codec for .mp4 file format
        video_writer = cv2.VideoWriter(video_path, fourcc, FPS, (width, height))

        frame_path = f"{cfg.overlay_path}/{vis_cam_idx}/0.png"
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = render_gaussian(view, gaussians, None, background)
        rendering = results["render"]  # (4, H, W)
        image = rendering.permute(1, 2, 0).detach().cpu().numpy()

        image = image.clip(0, 1)
        if use_white_background:
            image_mask = np.logical_and(
                (image != 1.0).any(axis=2), image[:, :, 3] > 100 / 255
            )
        else:
            image_mask = np.logical_and(
                (image != 0.0).any(axis=2), image[:, :, 3] > 100 / 255
            )
        image[~image_mask, 3] = 0

        alpha = image[..., 3:4]
        rgb = image[..., :3] * 255
        frame = alpha * rgb + (1 - alpha) * frame
        frame = frame.astype(np.uint8)

        force_arrow_meshes = []
        for j in range(n_ctrl_parts):
            # Calculate the center of the force_object_points
            force_center = (
                torch.mean(prev_x[force_object_points[j]], dim=0).cpu().numpy()
            )
            # Calculate the force vector
            force_vector = (
                self.get_force_vector(
                    prev_x,
                    force_springs[j],
                    force_rest_lengths[j],
                    force_spring_Y[j],
                    self.num_all_points,
                    self.simulator.controller_points[0],
                )
                .cpu()
                .numpy()
            )
            # Create arrow mesh in open3d
            if not (force_vector == 0).all():
                arrow_mesh = getArrowMesh(
                    origin=force_center,
                    end=force_center + force_vector / force_scale,
                    color=[1, 0, 0],
                )
                force_arrow_meshes.append(arrow_mesh)
                vis.add_geometry(force_arrow_meshes[j])
        # Adjust the viewpoint
        view_control = vis.get_view_control()
        camera_params = o3d.camera.PinholeCameraParameters()
        intrinsic_parameter = o3d.camera.PinholeCameraIntrinsic(
            width, height, intrinsic
        )
        camera_params.intrinsic = intrinsic_parameter
        camera_params.extrinsic = w2c
        view_control.convert_from_pinhole_camera_parameters(
            camera_params, allow_arbitrary=True
        )

        force_image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        force_image = (force_image * 255).astype(np.uint8)
        force_vis_mask = np.all(force_image == [255, 255, 255], axis=-1)
        frame[~force_vis_mask] = force_image[~force_vis_mask]

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # cv2.imshow("Interactive Playground", frame)
        # cv2.waitKey(0)
        video_writer.write(frame)

        for i in tqdm(range(1, frame_len)):
            if cfg.data_type == "real":
                self.simulator.set_controller_target(i, pure_inference=True)
            if self.simulator.object_collision_flag:
                self.simulator.update_collision_graph()

            wp.capture_launch(self.simulator.forward_graph)
            x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
            # Set the intial state for the next step
            self.simulator.set_init_state(
                self.simulator.wp_states[-1].wp_x,
                self.simulator.wp_states[-1].wp_v,
            )

            torch.cuda.synchronize()

            with torch.no_grad():
                # Do LBS on the gaussian kernels
                prev_particle_pos = prev_x
                cur_particle_pos = x
                if relations is None:
                    relations = get_topk_indices(
                        prev_x, K=16
                    )  # only computed in the first iteration

                if weights is None:
                    weights, weights_indices = knn_weights_sparse(
                        prev_particle_pos, current_pos, K=16
                    )  # only computed in the first iteration

                weights = calc_weights_vals_from_indices(
                    prev_particle_pos, current_pos, weights_indices
                )

                current_pos, current_rot, _ = interpolate_motions_speedup(
                    bones=prev_particle_pos,
                    motions=cur_particle_pos - prev_particle_pos,
                    relations=relations,
                    weights=weights,
                    weights_indices=weights_indices,
                    xyz=current_pos,
                    quat=current_rot,
                )

                # update gaussians with the new positions and rotations
                gaussians._xyz = current_pos
                gaussians._rotation = current_rot

            prev_x = x.clone()

            frame_path = f"{cfg.overlay_path}/{vis_cam_idx}/{i}.png"
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = render_gaussian(view, gaussians, None, background)
            rendering = results["render"]  # (4, H, W)
            image = rendering.permute(1, 2, 0).detach().cpu().numpy()

            image = image.clip(0, 1)
            if use_white_background:
                image_mask = np.logical_and(
                    (image != 1.0).any(axis=2), image[:, :, 3] > 100 / 255
                )
            else:
                image_mask = np.logical_and(
                    (image != 0.0).any(axis=2), image[:, :, 3] > 100 / 255
                )
            image[~image_mask, 3] = 0

            alpha = image[..., 3:4]
            rgb = image[..., :3] * 255
            frame = alpha * rgb + (1 - alpha) * frame
            frame = frame.astype(np.uint8)

            for arrow_mesh in force_arrow_meshes:
                vis.remove_geometry(arrow_mesh)

            force_arrow_meshes = []
            for j in range(n_ctrl_parts):
                # Calculate the center of the force_object_points
                force_center = (
                    torch.mean(x[force_object_points[j]], dim=0).cpu().numpy()
                )
                # Calculate the force vector
                force_vector = (
                    self.get_force_vector(
                        x,
                        force_springs[j],
                        force_rest_lengths[j],
                        force_spring_Y[j],
                        self.num_all_points,
                        self.simulator.controller_points[i],
                    )
                    .cpu()
                    .numpy()
                )
                if not (force_vector == 0).all():
                    # Create arrow mesh in open3d
                    arrow_mesh = getArrowMesh(
                        origin=force_center,
                        end=force_center + force_vector / force_scale,
                        color=[1, 0, 0],
                    )
                force_arrow_meshes.append(arrow_mesh)
                vis.add_geometry(force_arrow_meshes[j])

            view_control = vis.get_view_control()
            camera_params = o3d.camera.PinholeCameraParameters()
            intrinsic_parameter = o3d.camera.PinholeCameraIntrinsic(
                width, height, intrinsic
            )
            camera_params.intrinsic = intrinsic_parameter
            camera_params.extrinsic = w2c
            view_control.convert_from_pinhole_camera_parameters(
                camera_params, allow_arbitrary=True
            )

            vis.poll_events()
            vis.update_renderer()

            force_image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            force_image = (force_image * 255).astype(np.uint8)
            force_vis_mask = np.all(force_image == [255, 255, 255], axis=-1)
            frame[~force_vis_mask] = force_image[~force_vis_mask]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)

            # cv2.imshow("Interactive Playground", frame)
            # cv2.waitKey(0)
        vis.destroy_window()
        video_writer.release()

    def get_force_vector(
        self, x, springs, rest_lengths, spring_Y, num_object_points, controller_points
    ):
        with torch.no_grad():
            # Calculate the force of the springs
            x1 = controller_points[springs[:, 0]]
            x2 = x[springs[:, 1]]

            dis = x2 - x1
            dis_len = torch.norm(dis, dim=1)

            d = dis / torch.clamp(dis_len, min=1e-6)[:, None]
            spring_forces = (
                torch.clamp(spring_Y, min=cfg.spring_Y_min, max=cfg.spring_Y_max)[
                    :, None
                ]
                * (dis_len / rest_lengths - 1.0)[:, None]
                * d
            )

            total_force = -spring_forces.sum(dim=0)
        return total_force

    def visualize_material(self, model_path, gs_path, relative_material=True):
        # Load the model
        logger.info(f"Load model from {model_path}")
        checkpoint = torch.load(model_path, map_location=cfg.device)

        spring_Y = checkpoint["spring_Y"]
        collide_elas = checkpoint["collide_elas"]
        collide_fric = checkpoint["collide_fric"]
        collide_object_elas = checkpoint["collide_object_elas"]
        collide_object_fric = checkpoint["collide_object_fric"]
        num_object_springs = checkpoint["num_object_springs"]

        assert (
            len(spring_Y) == self.simulator.n_springs
        ), "Check if the loaded checkpoint match the config file to connect the springs"

        self.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
        self.simulator.set_collide(
            collide_elas.detach().clone(), collide_fric.detach().clone()
        )
        self.simulator.set_collide_object(
            collide_object_elas.detach().clone(),
            collide_object_fric.detach().clone(),
        )

        video_path = f"{cfg.base_dir}/material_visualization.mp4"

        vis_cam_idx = 0
        FPS = cfg.FPS
        width, height = cfg.WH
        intrinsic = cfg.intrinsics[vis_cam_idx]
        w2c = cfg.w2cs[vis_cam_idx]

        gaussians = GaussianModel(sh_degree=3)
        gaussians.load_ply(gs_path)
        gaussians = remove_gaussians_with_low_opacity(gaussians, 0.1)
        gaussians.isotropic = True
        current_pos = gaussians.get_xyz
        current_rot = gaussians.get_rotation
        use_white_background = True  # set to True for white background
        bg_color = [1, 1, 1] if use_white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=cfg.device)
        view = self._create_gs_view(w2c, intrinsic, height, width)
        prev_x = None
        relations = None
        weights = None

        # Start to visualize the stuffs
        logger.info("Visualizing the simulation")
        # Visualize the whole simulation using current set of parameters in the physical simulator
        frame_len = self.dataset.frame_len
        self.simulator.set_init_state(
            self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
        )
        prev_x = wp.to_torch(
            self.simulator.wp_states[0].wp_x, requires_grad=False
        ).clone()

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=width, height=height)
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Codec for .mp4 file format
        video_writer = cv2.VideoWriter(video_path, fourcc, FPS, (width, height))

        frame_path = f"{cfg.overlay_path}/{vis_cam_idx}/0.png"
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = render_gaussian(view, gaussians, None, background)
        rendering = results["render"]  # (4, H, W)
        image = rendering.permute(1, 2, 0).detach().cpu().numpy()

        image = image.clip(0, 1)
        if use_white_background:
            image_mask = np.logical_and(
                (image != 1.0).any(axis=2), image[:, :, 3] > 100 / 255
            )
        else:
            image_mask = np.logical_and(
                (image != 0.0).any(axis=2), image[:, :, 3] > 100 / 255
            )
        image[~image_mask, 3] = 0

        alpha = image[..., 3:4]
        rgb = image[..., :3] * 255
        frame = alpha * rgb + (1 - alpha) * frame
        frame = frame.astype(np.uint8)

        # Add the material visualization
        object_springs = self.init_springs[:num_object_springs]
        material_field = torch.zeros((self.num_all_points, 3), device=cfg.device)
        count_field = torch.zeros(
            self.num_all_points, dtype=torch.int32, device=cfg.device
        )
        clamp_object_spring_Y = torch.clamp(
            spring_Y[:num_object_springs], min=cfg.spring_Y_min, max=cfg.spring_Y_max
        )
        object_rest_lengths = self.init_rest_lengths[:num_object_springs]

        # idx1 = object_springs[:, 0]
        # idx2 = object_springs[:, 1]
        # x1 = prev_x[idx1]
        # x2 = prev_x[idx2]
        # dis = x2 - x1
        # dis_len = torch.norm(dis, dim=1)
        # d = dis / torch.clamp(dis_len, min=1e-6)[:, None]
        # # import pdb
        # # pdb.set_trace()
        # material_field.index_add_(
        #     0,
        #     idx1,
        #     clamp_object_spring_Y[:, None] / object_rest_lengths[:, None] * d,
        # )
        # material_field.index_add_(
        #     0,
        #     idx2,
        #     clamp_object_spring_Y[:, None] / object_rest_lengths[:, None] * d,
        # )
        # material_field = torch.norm(material_field, dim=1)
        # import pdb
        # pdb.set_trace()
        # count_field.index_add_(
        #     0, idx1, torch.ones_like(idx1, dtype=torch.int32, device=cfg.device)
        # )
        # count_field.index_add_(
        #     0, idx2, torch.ones_like(idx2, dtype=torch.int32, device=cfg.device)
        # )
        # material_field /= count_field
        # if relative_material:
        #     material_field_normalized = (material_field - material_field.min()) / (
        #         material_field.max() - material_field.min()
        #     )
        # else:
        #     material_field_normalized = (material_field - cfg.spring_Y_min) / (
        #         cfg.spring_Y_max - cfg.spring_Y_min
        #     )
        # rainbow_colors = plt.cm.rainbow(material_field_normalized.cpu().numpy())[:, :3]

        stiffness_map = compute_effective_stiffness(
            points=prev_x,
            springs=object_springs,
            Y=clamp_object_spring_Y,
            rest_lengths=object_rest_lengths,
            device=cfg.device,
        )
        normed = (stiffness_map - stiffness_map.min()) / (
            stiffness_map.max() - stiffness_map.min()
        )
        rainbow_colors = plt.cm.rainbow(normed.cpu().numpy())[:, :3]

        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(prev_x.cpu().numpy())
        object_pcd.colors = o3d.utility.Vector3dVector(rainbow_colors)
        vis.add_geometry(object_pcd)

        # Adjust the viewpoint
        view_control = vis.get_view_control()
        camera_params = o3d.camera.PinholeCameraParameters()
        intrinsic_parameter = o3d.camera.PinholeCameraIntrinsic(
            width, height, intrinsic
        )
        camera_params.intrinsic = intrinsic_parameter
        camera_params.extrinsic = w2c
        view_control.convert_from_pinhole_camera_parameters(
            camera_params, allow_arbitrary=True
        )

        material_image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        material_image = (material_image * 255).astype(np.uint8)
        material_vis_mask = np.all(material_image == [255, 255, 255], axis=-1)
        frame[~material_vis_mask] = material_image[~material_vis_mask]

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Interactive Playground", frame)
        cv2.waitKey(1)
        video_writer.write(frame)

        for i in tqdm(range(1, frame_len)):
            if cfg.data_type == "real":
                self.simulator.set_controller_target(i, pure_inference=True)
            if self.simulator.object_collision_flag:
                self.simulator.update_collision_graph()

            wp.capture_launch(self.simulator.forward_graph)
            x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
            # Set the intial state for the next step
            self.simulator.set_init_state(
                self.simulator.wp_states[-1].wp_x,
                self.simulator.wp_states[-1].wp_v,
            )

            torch.cuda.synchronize()

            with torch.no_grad():
                # Do LBS on the gaussian kernels
                prev_particle_pos = prev_x
                cur_particle_pos = x
                if relations is None:
                    relations = get_topk_indices(
                        prev_x, K=16
                    )  # only computed in the first iteration

                if weights is None:
                    weights, weights_indices = knn_weights_sparse(
                        prev_particle_pos, current_pos, K=16
                    )  # only computed in the first iteration

                weights = calc_weights_vals_from_indices(
                    prev_particle_pos, current_pos, weights_indices
                )

                current_pos, current_rot, _ = interpolate_motions_speedup(
                    bones=prev_particle_pos,
                    motions=cur_particle_pos - prev_particle_pos,
                    relations=relations,
                    weights=weights,
                    weights_indices=weights_indices,
                    xyz=current_pos,
                    quat=current_rot,
                )

                # update gaussians with the new positions and rotations
                gaussians._xyz = current_pos
                gaussians._rotation = current_rot

            prev_x = x.clone()

            frame_path = f"{cfg.overlay_path}/{vis_cam_idx}/{i}.png"
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = render_gaussian(view, gaussians, None, background)
            rendering = results["render"]  # (4, H, W)
            image = rendering.permute(1, 2, 0).detach().cpu().numpy()

            image = image.clip(0, 1)
            if use_white_background:
                image_mask = np.logical_and(
                    (image != 1.0).any(axis=2), image[:, :, 3] > 100 / 255
                )
            else:
                image_mask = np.logical_and(
                    (image != 0.0).any(axis=2), image[:, :, 3] > 100 / 255
                )
            image[~image_mask, 3] = 0

            alpha = image[..., 3:4]
            rgb = image[..., :3] * 255
            frame = alpha * rgb + (1 - alpha) * frame
            frame = frame.astype(np.uint8)

            # Update the object pcd
            object_pcd.points = o3d.utility.Vector3dVector(prev_x.cpu().numpy())
            vis.update_geometry(object_pcd)

            vis.poll_events()
            vis.update_renderer()

            force_image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            force_image = (force_image * 255).astype(np.uint8)
            force_vis_mask = np.all(force_image == [255, 255, 255], axis=-1)
            frame[~force_vis_mask] = force_image[~force_vis_mask]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)

            cv2.imshow("Interactive Playground", frame)
            cv2.waitKey(1)
        vis.destroy_window()
        video_writer.release()


def get_simple_shadow(
    points,
    intrinsic,
    w2c,
    width,
    height,
    image_mask,
    kernel_size=7,
    light_point=[0, 0, -3],
):
    points = points.cpu().numpy()

    t = -points[:, 2] / light_point[2]
    points_on_table = points + t[:, None] * light_point

    points_homogeneous = np.hstack(
        [points_on_table, np.ones((points_on_table.shape[0], 1))]
    )  # Convert to homogeneous coordinates
    points_camera = (w2c @ points_homogeneous.T).T

    points_pixels = (intrinsic @ points_camera[:, :3].T).T
    points_pixels /= points_pixels[:, 2:3]
    pixel_coords = points_pixels[:, :2]

    valid_mask = (
        (pixel_coords[:, 0] >= 0)
        & (pixel_coords[:, 0] < width)
        & (pixel_coords[:, 1] >= 0)
        & (pixel_coords[:, 1] < height)
    )

    valid_pixel_coords = pixel_coords[valid_mask]
    valid_pixel_coords = valid_pixel_coords.astype(int)

    shadow_image = np.zeros((height, width), dtype=np.uint8)
    shadow_image[valid_pixel_coords[:, 1], valid_pixel_coords[:, 0]] = 255

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    kernel_1 = np.ones((3, 3), np.uint(8))
    dilated_shadow = cv2.dilate(shadow_image, kernel, iterations=1)
    dilated_shadow = cv2.dilate(dilated_shadow, kernel_1, iterations=1)
    final_shadow = cv2.erode(dilated_shadow, kernel, iterations=1)

    final_shadow[image_mask] = 0
    final_shadow = final_shadow == 255
    return final_shadow


# Borrow ideas and codes from H. Sánchez's answer
# https://stackoverflow.com/questions/59026581/create-arrows-in-open3d
def getArrowMesh(origin=[0, 0, 0], end=None, color=[0, 0, 0]):
    vec_Arr = np.array(end) - np.array(origin)
    vec_len = np.linalg.norm(vec_Arr)
    mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
        cone_height=0.05 * vec_len,
        cone_radius=0.002,
        cylinder_height=0.2 * vec_len,
        cylinder_radius=0.003,
    )
    mesh_arrow.paint_uniform_color(color)
    rot_mat = _caculate_align_mat(vec_Arr / vec_len)
    mesh_arrow.rotate(rot_mat, center=np.array([0, 0, 0]))
    mesh_arrow.translate(np.array(origin))
    return mesh_arrow


def _get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array(
        [
            [0, -pVec_Arr[2], pVec_Arr[1]],
            [pVec_Arr[2], 0, -pVec_Arr[0]],
            [-pVec_Arr[1], pVec_Arr[0], 0],
        ]
    )
    return qCross_prod_mat


def _caculate_align_mat(pVec_Arr):
    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr / scale
    # must ensure pVec_Arr is also a unit vec.
    z_unit_Arr = np.array([0, 0, 1])
    z_mat = _get_cross_prod_mat(z_unit_Arr)

    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = _get_cross_prod_mat(z_c_vec)
    if np.dot(z_unit_Arr, pVec_Arr) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, pVec_Arr) == 1:
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = (
            np.eye(3, 3)
            + z_c_vec_mat
            + np.matmul(z_c_vec_mat, z_c_vec_mat) / (1 + np.dot(z_unit_Arr, pVec_Arr))
        )
    qTrans_Mat *= scale
    return qTrans_Mat


def construct_stiffness_matrix_sparse(
    springs, positions, spring_Y, rest_lengths, num_points, device
):
    # springs: (N_springs, 2)
    # positions: (N_points, 3)
    # spring_Y: (N_springs,)
    # rest_lengths: (N_springs,)

    i = springs[:, 0]
    j = springs[:, 1]

    x_i = positions[i]  # (N, 3)
    x_j = positions[j]
    d = x_j - x_i  # (N, 3)
    d_norm = torch.norm(d, dim=1, keepdim=True) + 1e-8
    d_hat = d / d_norm  # (N, 3)

    coeff = spring_Y / rest_lengths  # (N,)
    k_blocks = coeff[:, None, None] * (
        d_hat[:, :, None] @ d_hat[:, None, :]
    )  # (N, 3, 3)

    indices = []
    values = []

    for shift_i, shift_j, sign in [(0, 0, 1), (0, 1, -1), (1, 0, -1), (1, 1, 1)]:
        node_i = springs[:, shift_i]
        node_j = springs[:, shift_j]

        for a in range(3):
            for b in range(3):
                row_idx = 3 * node_i + a
                col_idx = 3 * node_j + b
                val = sign * k_blocks[:, a, b]
                indices.append(torch.stack([row_idx, col_idx], dim=0))  # (2, N)
                values.append(val)

    indices = torch.cat(indices, dim=1)  # (2, total_nonzero)
    values = torch.cat(values, dim=0)  # (total_nonzero,)
    size = (3 * num_points, 3 * num_points)
    K_sparse = torch.sparse_coo_tensor(indices, values, size, device=device).coalesce()
    return K_sparse


def compute_effective_stiffness(points, springs, Y, rest_lengths, device):
    """
    Compute effective stiffness for each point based on stiffness matrix diagonal blocks.
    Return: (N_points,) tensor of Frobenius norm of 3x3 diagonal blocks in stiffness matrix.
    """
    num_points = points.shape[0]
    K_sparse = construct_stiffness_matrix_sparse(
        springs=springs,
        positions=points,
        spring_Y=Y,
        rest_lengths=rest_lengths,
        num_points=num_points,
        device=device,
    )

    K_dense = K_sparse.to_dense()
    stiffness_map = torch.zeros(num_points, device=device)
    for i in range(num_points):
        block = K_dense[3 * i : 3 * i + 3, 3 * i : 3 * i + 3]
        stiffness_map[i] = torch.norm(block, p="fro")
    return stiffness_map
