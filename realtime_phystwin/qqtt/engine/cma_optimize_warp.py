from qqtt.data import RealData, SimpleData
from qqtt.utils import logger, visualize_pc, cfg
from qqtt.model.diff_simulator import SpringMassSystemWarp
from qqtt.model.diff_simulator.spring_mass_warp_batched import (
    SpringMassSystemWarp as SpringMassSystemWarpBatched,
)
import open3d as o3d
import numpy as np
import torch
from tqdm import tqdm
import warp as wp
import cma
import pickle
import os
import csv
import time
import json
import shutil


class OptimizerCMA:
    def __init__(
        self,
        data_path,
        base_dir,
        train_frame,
        mask_path=None,
        velocity_path=None,
        device="cuda:0",
        batch_mode=False,
        batch_size=1,
        segment_len=10,
        segment_stride=10,
        batch_debug_interval=5,
        dataset_override=None,
    ):
        cfg.data_path = data_path
        cfg.base_dir = base_dir
        cfg.device = device
        cfg.run_name = base_dir.split("/")[-1]
        cfg.train_frame = train_frame
        self.batch_mode = bool(batch_mode)
        self.batch_size = int(batch_size)
        self.segment_len = int(segment_len)
        self.segment_stride = int(segment_stride)
        self.batch_debug_interval = int(batch_debug_interval)
        self.rest_debug_log_count = 0
        self.nonfinite_debug_log_count = 0
        logger.info(
            "[CMA]: "
            f"batch_mode={self.batch_mode}, batch_size={self.batch_size}, "
            f"segment_len={self.segment_len}, segment_stride={self.segment_stride}, "
            f"batch_debug_interval={self.batch_debug_interval}"
        )

        if not os.path.exists(f"{cfg.base_dir}/optimizeCMA"):
            # Create directory if it doesn't exist
            os.makedirs(f"{cfg.base_dir}/optimizeCMA")

        self.init_masks = None
        self.init_velocities = None
        # Load the data
        if cfg.data_type == "real":
            self.dataset = (
                RealData(visualize=False)
                if dataset_override is None
                else dataset_override
            )
            self._load_real_dataset_attributes()
        elif cfg.data_type == "synthetic":
            self.dataset = SimpleData(visualize=False)
            self.object_points = self.dataset.data
            self.object_colors = None
            self.object_visibilities = None
            self.object_motions_valid = None
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

    def _load_real_dataset_attributes(self):
        self.object_points = self.dataset.object_points
        self.object_colors = self.dataset.object_colors
        self.object_visibilities = self.dataset.object_visibilities
        self.object_motions_valid = self.dataset.object_motions_valid
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
        if cfg.data_type == "real":
            self._load_real_dataset_attributes()

    def _compute_segment_start_indices(self, total_frames):
        seg_len = int(self.segment_len)
        seg_stride = int(self.segment_stride)
        if seg_len <= 1:
            raise ValueError("segment_len must be >= 2 for batched CMA")
        if seg_stride <= 0:
            raise ValueError("segment_stride must be positive")
        if total_frames < seg_len:
            raise ValueError(
                f"total_frames ({total_frames}) must be >= segment_len ({seg_len})"
            )
        return list(range(0, total_frames - seg_len + 1, seg_stride))

    def _get_trainable_total_frames(self):
        if cfg.train_frame is None:
            total_frames = int(self.object_points.shape[0])
        else:
            total_frames = int(min(cfg.train_frame, self.object_points.shape[0]))
        if self.controller_points is not None:
            total_frames = min(total_frames, int(self.controller_points.shape[0]))
        if self.object_visibilities is not None:
            total_frames = min(total_frames, int(self.object_visibilities.shape[0]))
        if self.object_motions_valid is not None:
            total_frames = min(total_frames, int(self.object_motions_valid.shape[0]))
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
            surface_velocity = (
                self.asap_surface_points[start_idx] - self.asap_surface_points[prev_idx]
            ) / (steps * frame_dt)
            velocities[
                num_gt_points : num_gt_points + num_surface_extra
            ] = surface_velocity

        if self.asap_interior_points is not None and self.asap_interior_points.shape[1] > 0:
            interior_velocity = (
                self.asap_interior_points[start_idx]
                - self.asap_interior_points[prev_idx]
            ) / (steps * frame_dt)
            velocities[self.num_surface_points : self.num_all_points] = interior_velocity

        return velocities

    def _get_valid_object_mask(self, frame_idx):
        num_gt_points = int(self.num_original_points)
        valid = torch.isfinite(self.object_points[frame_idx, :num_gt_points]).all(dim=-1)
        if self.object_visibilities is not None:
            valid = valid & self.object_visibilities[frame_idx, :num_gt_points]
        if self.object_motions_valid is not None:
            valid = valid & self.object_motions_valid[frame_idx, :num_gt_points]
        return valid

    def _get_reset_object_points(self, frame_idx):
        num_gt_points = int(self.num_original_points)
        reset_points = self.object_points[frame_idx, :num_gt_points].clone()
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
                "[CMA-Batch-Rest]: controller/object rest ratio "
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
            logger.warning("[CMA-Batch-Rest]: all controller/object rest ratios are nonfinite")
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

    def _build_segment_batch_tensors(self, start_indices):
        if len(start_indices) == 0:
            raise ValueError("start_indices cannot be empty")

        B = len(start_indices)
        seg_len = int(self.segment_len)
        num_gt_points = int(self.num_original_points)
        num_surface_extra = int(self.num_surface_points) - num_gt_points

        obj_segments = [
            self.object_points[s : s + seg_len].contiguous() for s in start_indices
        ]
        obj_stack = torch.stack(obj_segments, dim=0)
        batched_object_points = (
            obj_stack.permute(1, 0, 2, 3)
            .contiguous()
            .reshape(seg_len, B * num_gt_points, 3)
        )

        batched_controller_points = None
        if self.controller_points is not None:
            ctrl_segments = [
                self.controller_points[s : s + seg_len].contiguous()
                for s in start_indices
            ]
            ctrl_stack = torch.stack(ctrl_segments, dim=0)
            batched_controller_points = (
                ctrl_stack.permute(1, 0, 2, 3)
                .contiguous()
                .reshape(seg_len, B * ctrl_stack.shape[2], 3)
            )

        batched_object_visibilities = None
        if self.object_visibilities is not None:
            vis_segments = [
                self.object_visibilities[s : s + seg_len].contiguous()
                for s in start_indices
            ]
            vis_stack = torch.stack(vis_segments, dim=0)
            batched_object_visibilities = (
                vis_stack.permute(1, 0, 2)
                .contiguous()
                .reshape(seg_len, B * num_gt_points)
            )

        batched_object_motions_valid = None
        if self.object_motions_valid is not None:
            mot_segments = [
                self.object_motions_valid[s : s + seg_len].contiguous()
                for s in start_indices
            ]
            mot_stack = torch.stack(mot_segments, dim=0)
            batched_object_motions_valid = (
                mot_stack.permute(1, 0, 2)
                .contiguous()
                .reshape(seg_len, B * num_gt_points)
            )

        init_object_vertices = []
        init_object_velocities = []
        for start_idx in start_indices:
            vertices = self.init_vertices[: self.num_all_points].clone()
            vertices[:num_gt_points] = self._get_reset_object_points(start_idx)
            if num_surface_extra > 0:
                vertices[
                    num_gt_points : num_gt_points + num_surface_extra
                ] = self.asap_surface_points[start_idx]
            if self.asap_interior_points is not None and self.asap_interior_points.shape[1] > 0:
                vertices[self.num_surface_points : self.num_all_points] = (
                    self.asap_interior_points[start_idx]
                )
            init_object_vertices.append(vertices)
            init_object_velocities.append(
                self._estimate_segment_init_velocity(start_idx, vertices)
            )

        init_object_vertices_by_batch = torch.stack(init_object_vertices, dim=0)
        init_object_velocities_by_batch = torch.stack(init_object_velocities, dim=0)
        init_object_vertices_batched = init_object_vertices_by_batch.reshape(
            B * self.num_all_points, 3
        )
        init_object_velocities_batched = init_object_velocities_by_batch.reshape(
            B * self.num_all_points, 3
        )

        if batched_controller_points is not None:
            ctrl_start = batched_controller_points[0]
            ctrl_start_by_batch = ctrl_start.reshape(B, -1, 3)
            init_rest_lengths_batched = self._compute_batched_segment_rest_lengths(
                init_object_vertices_by_batch, ctrl_start_by_batch
            )
            init_vertices_batched = torch.cat(
                [init_object_vertices_batched, ctrl_start], dim=0
            ).contiguous()
        else:
            init_rest_lengths_batched = self._compute_batched_segment_rest_lengths(
                init_object_vertices_by_batch
            )
            init_vertices_batched = init_object_vertices_batched.contiguous()

        return {
            "batch_size": B,
            "segment_len": seg_len,
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
            "init_velocities_batched": init_object_velocities_batched,
            "init_rest_lengths_batched": init_rest_lengths_batched,
        }

    def _save_batch_instance_videos(self, debug_tag, starts, vertices, controller_points):
        debug_dir = f"{cfg.base_dir}/optimizeCMA/batch_instances/{debug_tag}"
        os.makedirs(debug_dir, exist_ok=True)
        B = len(starts)
        for batch_idx, start_idx in enumerate(starts):
            lo = batch_idx * self.num_all_points
            hi = (batch_idx + 1) * self.num_all_points
            segment_vertices = vertices[:, lo:hi, :]
            segment_colors = (
                self.object_colors[start_idx : start_idx + vertices.shape[0]]
                if self.object_colors is not None
                else None
            )
            segment_controller_points = None
            if controller_points is not None:
                num_ctrl = int(self.controller_points.shape[1])
                ctrl_lo = batch_idx * num_ctrl
                ctrl_hi = (batch_idx + 1) * num_ctrl
                segment_controller_points = controller_points[:, ctrl_lo:ctrl_hi, :]

            visualize_pc(
                segment_vertices,
                segment_colors,
                segment_controller_points,
                visualize=False,
                save_video=True,
                save_path=(
                    f"{debug_dir}/instance_{batch_idx:03d}_"
                    f"start_{int(start_idx):05d}.mp4"
                ),
                frame_start_idx=int(start_idx),
            )

    def _save_batch_loss_csv(self, debug_tag, loss_rows):
        debug_dir = f"{cfg.base_dir}/optimizeCMA/batch_instances/{debug_tag}"
        os.makedirs(debug_dir, exist_ok=True)
        csv_path = f"{debug_dir}/loss_per_instance.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "instance",
                    "start_frame",
                    "mean_loss",
                    "num_steps",
                ],
            )
            writer.writeheader()
            writer.writerows(loss_rows)

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

    def normalize(self, value, min, max):
        assert min < max, "The minimum value should be less than the maximum value"
        return (value - min) / (max - min)

    def denormalize(self, value, min, max):
        assert min < max, "The minimum value should be less than the maximum value"
        return value * (max - min) + min

    def _initial_cma_vector(self):
        return [
            self.normalize(cfg.init_spring_Y, cfg.spring_Y_min, cfg.spring_Y_max),
            self.normalize(cfg.object_radius, 0.01, 0.05),
            self.normalize(cfg.object_max_neighbours, 10, 50),
            self.normalize(cfg.controller_radius, 0.01, 0.08),
            self.normalize(cfg.controller_max_neighbours, 10, 80),
            cfg.collide_elas,
            self.normalize(cfg.collide_fric, 0, 2),
            cfg.collide_object_elas,
            self.normalize(cfg.collide_object_fric, 0, 2),
            self.normalize(cfg.collision_dist, 0.01, 0.05),
            self.normalize(cfg.drag_damping, 0, 20),
            self.normalize(cfg.dashpot_damping, 0, 200),
        ]

    def _decode_cma_parameters(self, x):
        return {
            "global_spring_Y": self.denormalize(
                x[0], cfg.spring_Y_min, cfg.spring_Y_max
            ),
            "object_radius": self.denormalize(x[1], 0.01, 0.05),
            "object_max_neighbours": int(self.denormalize(x[2], 10, 50)),
            "controller_radius": self.denormalize(x[3], 0.01, 0.08),
            "controller_max_neighbours": int(self.denormalize(x[4], 10, 80)),
            "collide_elas": x[5],
            "collide_fric": self.denormalize(x[6], 0, 2),
            "collide_object_elas": x[7],
            "collide_object_fric": self.denormalize(x[8], 0, 2),
            "collision_dist": self.denormalize(x[9], 0.01, 0.05),
            "drag_damping": self.denormalize(x[10], 0, 20),
            "dashpot_damping": self.denormalize(x[11], 0, 200),
        }

    def _save_optimal_parameters(self, x):
        optimal_results = self._decode_cma_parameters(x)
        with open(f"{cfg.base_dir}/optimal_params.pkl", "wb") as f:
            pickle.dump(optimal_results, f)
        return optimal_results

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

    def _write_realtime_manifest(self, realtime_vis_dir, iteration, arrays):
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
        pred_points,
        keep_iteration_history=True,
    ):
        window_starts = np.asarray(window_starts, dtype=np.int64)
        pred_points = np.asarray(pred_points, dtype=np.float32)
        if pred_points.ndim != 4:
            raise ValueError(f"Expected pred_points [W,T,N,3], got {pred_points.shape}")

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
                self.object_points[int(start) : int(start) + pred_points.shape[1]]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
                for start in window_starts
            ],
            axis=0,
        )

        object_colors = np.empty((0, 0, 0, 3), dtype=np.float32)
        if self.object_colors is not None:
            object_colors = np.stack(
                [
                    self.object_colors[int(start) : int(start) + pred_points.shape[1]]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                    for start in window_starts
                ],
                axis=0,
            )

        object_visibilities = np.empty((0, 0, 0), dtype=np.bool_)
        if self.object_visibilities is not None:
            object_visibilities = np.stack(
                [
                    self.object_visibilities[
                        int(start) : int(start) + pred_points.shape[1]
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.bool_)
                    for start in window_starts
                ],
                axis=0,
            )

        controller_points = np.empty(
            (pred_points.shape[0], pred_points.shape[1], 0, 3), dtype=np.float32
        )
        if self.controller_points is not None:
            controller_points = np.stack(
                [
                    self.controller_points[
                        int(start) : int(start) + pred_points.shape[1]
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                    for start in window_starts
                ],
                axis=0,
            )

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
            "batch_size": np.array(int(pred_points.shape[0]), dtype=np.int64),
            "real_window_count": np.array(int(pred_points.shape[0]), dtype=np.int64),
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

        if keep_iteration_history:
            iteration_path = os.path.join(
                realtime_vis_dir,
                "iterations",
                f"iter_{int(iteration):06d}.npz",
            )
            self._atomic_save_npz(iteration_path, **arrays)
        self._atomic_save_npz(
            os.path.join(realtime_vis_dir, "latest_window.npz"), **arrays
        )
        self._write_realtime_manifest(realtime_vis_dir, iteration, arrays)

    def optimize_online_batched(
        self,
        online_reader,
        online_buffer,
        max_iter=10,
        poll_sec=1.0,
        recent_window_count=8,
        sample_recent=True,
        seed=42,
        realtime_vis_dir=None,
        realtime_vis_every=1,
        realtime_keep_iterations=True,
    ):
        if cfg.data_type != "real":
            raise ValueError("Online CMA currently supports real data only")
        if not self.batch_mode:
            raise ValueError("Online CMA requires batch_mode=True")

        max_iter = int(max_iter)
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        recent_window_count = max(int(recent_window_count), int(self.batch_size))
        poll_sec = max(0.0, float(poll_sec))
        realtime_vis_every = max(1, int(realtime_vis_every))
        rng = np.random.default_rng(int(seed))

        if realtime_vis_dir is not None:
            os.makedirs(realtime_vis_dir, exist_ok=True)
            for path in (
                os.path.join(realtime_vis_dir, "latest_window.npz"),
                os.path.join(realtime_vis_dir, "manifest.json"),
            ):
                if os.path.exists(path):
                    os.remove(path)
            for dirname in ("first_seen", "iterations"):
                path = os.path.join(realtime_vis_dir, dirname)
                if os.path.isdir(path):
                    shutil.rmtree(path)

        x_init = self._initial_cma_vector()
        es = cma.CMAEvolutionStrategy(
            x_init,
            1 / 6,
            {"bounds": [0.0, 1.0], "seed": int(seed)},
        )
        final_x = np.asarray(x_init, dtype=np.float64)
        final_loss = None

        logger.info(
            "[CMA-Online]: start, "
            f"iterations={max_iter}, batch_size={self.batch_size}, "
            f"segment_len={self.segment_len}, "
            f"segment_stride={self.segment_stride}, "
            f"recent_window_count={recent_window_count}"
        )

        iteration = 0
        while iteration < max_iter:
            new_chunks = online_reader.load_new_chunks()
            if len(new_chunks) > 0:
                online_buffer.append_chunks(new_chunks)
                online_buffer.sync_to_device(cfg.device)
                self.refresh_real_data_from_dataset()
                logger.info(
                    "[CMA-Online]: refreshed data, "
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
                    "[CMA-Online]: waiting for frames, "
                    f"available={total_frames}, need={self.segment_len}"
                )
                time.sleep(poll_sec)
                continue

            segment_starts = self._compute_segment_start_indices(total_frames)
            recent_starts = segment_starts[-recent_window_count:]
            selected_count = min(int(self.batch_size), len(recent_starts))
            if sample_recent and len(recent_starts) > selected_count:
                selected = rng.choice(
                    np.asarray(recent_starts, dtype=np.int64),
                    size=selected_count,
                    replace=False,
                )
                selected_starts = sorted(int(v) for v in selected.tolist())
            else:
                selected_starts = recent_starts[-selected_count:]

            solutions = es.ask()
            losses = []
            iter_start = time.perf_counter()
            for solution in solutions:
                loss = self.error_func_batched(
                    solution,
                    start_indices=selected_starts,
                )
                losses.append(float(loss) if np.isfinite(loss) else 1e12)
            es.tell(solutions, losses)

            best_idx = int(np.argmin(losses))
            final_x = np.asarray(solutions[best_idx], dtype=np.float64)
            final_loss = float(losses[best_idx])
            if realtime_vis_dir is not None and (
                iteration % realtime_vis_every == 0 or iteration == max_iter - 1
            ):
                rollout = self.error_func_batched(
                    final_x,
                    start_indices=selected_starts,
                    return_rollout=True,
                )
                self._export_realtime_windows(
                    realtime_vis_dir=realtime_vis_dir,
                    iteration=iteration,
                    window_starts=rollout["window_starts"],
                    pred_points=rollout["pred_points"],
                    keep_iteration_history=realtime_keep_iterations,
                )
                logger.info(
                    "[CMA-Online-Realtime]: exported best candidate "
                    f"iteration={iteration}, loss={rollout['loss']}"
                )
            logger.info(
                "[CMA-Online]: "
                f"iteration={iteration}, loss={final_loss}, "
                f"frames={total_frames}, starts={selected_starts}, "
                f"elapsed={time.perf_counter() - iter_start:.3f}s"
            )
            iteration += 1

        optimal_results = self._save_optimal_parameters(final_x)
        logger.info(
            "[CMA-Online]: saved optimal parameters to "
            f"{cfg.base_dir}/optimal_params.pkl, final_loss={final_loss}"
        )
        return optimal_results

    def optimize(self, max_iter=100):
        # Initialize the parameters
        init_global_spring_Y = self.normalize(
            cfg.init_spring_Y, cfg.spring_Y_min, cfg.spring_Y_max
        )
        init_object_radius = self.normalize(cfg.object_radius, 0.01, 0.05)
        init_object_max_neighbours = self.normalize(cfg.object_max_neighbours, 10, 50)
        init_controller_radius = self.normalize(cfg.controller_radius, 0.01, 0.08)
        init_controller_max_neighbours = self.normalize(
            cfg.controller_max_neighbours, 10, 80
        )
        init_collide_elas = cfg.collide_elas
        init_collide_fric = self.normalize(cfg.collide_fric, 0, 2)
        init_collide_object_elas = cfg.collide_object_elas
        init_collide_object_fric = self.normalize(cfg.collide_object_fric, 0, 2)
        init_collision_dist = self.normalize(cfg.collision_dist, 0.01, 0.05)
        init_drag_damping = self.normalize(cfg.drag_damping, 0, 20)
        init_dashpot_damping = self.normalize(cfg.dashpot_damping, 0, 200)

        x_init = [
            init_global_spring_Y,
            init_object_radius,
            init_object_max_neighbours,
            init_controller_radius,
            init_controller_max_neighbours,
            init_collide_elas,
            init_collide_fric,
            init_collide_object_elas,
            init_collide_object_fric,
            init_collision_dist,
            init_drag_damping,
            init_dashpot_damping,
        ]

        self.error_func(
            x_init, visualize=True, video_path=f"{cfg.base_dir}/optimizeCMA/init.mp4"
        )
        if self.batch_mode:
            logger.info("[CMA-Batch]: saving init per-instance videos and losses")
            self.error_func_batched(x_init, debug_tag="init")

        std = 1 / 6
        es = cma.CMAEvolutionStrategy(x_init, std, {"bounds": [0.0, 1.0], "seed": 42})
        for iteration in range(max_iter):
            solutions = es.ask()
            losses = [self.error_func(x) for x in solutions]
            es.tell(solutions, losses)
            es.disp()
            current_iter = iteration + 1
            if (
                self.batch_mode
                and self.batch_debug_interval > 0
                and current_iter % self.batch_debug_interval == 0
            ):
                best_idx = int(np.argmin(losses))
                debug_tag = f"iter_{current_iter:04d}"
                logger.info(
                    "[CMA-Batch]: "
                    f"saving per-instance videos and losses for {debug_tag}"
                )
                self.error_func_batched(solutions[best_idx], debug_tag=debug_tag)

        # Get the results
        res = es.result
        optimal_x = np.array(res[0]).astype(np.float32)
        optimal_error = res[1]
        logger.info(f"Optimal x: {optimal_x}, Optimal error: {optimal_error}")

        final_global_spring_Y = self.denormalize(
            optimal_x[0], cfg.spring_Y_min, cfg.spring_Y_max
        )
        final_object_radius = self.denormalize(optimal_x[1], 0.01, 0.05)
        final_object_max_neighbours = int(self.denormalize(optimal_x[2], 10, 50))
        final_controller_radius = self.denormalize(optimal_x[3], 0.01, 0.08)
        final_controller_max_neighbours = int(self.denormalize(optimal_x[4], 10, 80))
        final_collide_elas = optimal_x[5]
        final_collide_fric = self.denormalize(optimal_x[6], 0, 2)
        final_collide_object_elas = optimal_x[7]
        final_collide_object_fric = self.denormalize(optimal_x[8], 0, 2)
        final_collision_dist = self.denormalize(optimal_x[9], 0.01, 0.05)
        final_drag_damping = self.denormalize(optimal_x[10], 0, 20)
        final_dashpot_damping = self.denormalize(optimal_x[11], 0, 200)

        self.error_func(
            optimal_x,
            visualize=True,
            video_path=f"{cfg.base_dir}/optimizeCMA/optimal.mp4",
        )
        if self.batch_mode:
            logger.info("[CMA-Batch]: saving optimal per-instance videos and losses")
            self.error_func_batched(optimal_x, debug_tag="optimal")

        optimal_results = {}
        optimal_results["global_spring_Y"] = final_global_spring_Y
        optimal_results["object_radius"] = final_object_radius
        optimal_results["object_max_neighbours"] = final_object_max_neighbours
        optimal_results["controller_radius"] = final_controller_radius
        optimal_results["controller_max_neighbours"] = final_controller_max_neighbours
        optimal_results["collide_elas"] = final_collide_elas
        optimal_results["collide_fric"] = final_collide_fric
        optimal_results["collide_object_elas"] = final_collide_object_elas
        optimal_results["collide_object_fric"] = final_collide_object_fric
        optimal_results["collision_dist"] = final_collision_dist
        optimal_results["drag_damping"] = final_drag_damping
        optimal_results["dashpot_damping"] = final_dashpot_damping

        # Save out all the initialized parameters
        with open(f"{cfg.base_dir}/optimal_params.pkl", "wb") as f:
            pickle.dump(optimal_results, f)

    def error_func_batched(
        self,
        parameters,
        debug_tag=None,
        start_indices=None,
        return_rollout=False,
    ):
        global_spring_Y = self.denormalize(
            parameters[0], cfg.spring_Y_min, cfg.spring_Y_max
        )
        object_radius = self.denormalize(parameters[1], 0.01, 0.05)
        object_max_neighbours = int(self.denormalize(parameters[2], 10, 50))
        controller_radius = self.denormalize(parameters[3], 0.01, 0.08)
        controller_max_neighbours = int(self.denormalize(parameters[4], 10, 80))
        collide_elas = parameters[5]
        collide_fric = self.denormalize(parameters[6], 0, 2)
        collide_object_elas = parameters[7]
        collide_object_fric = self.denormalize(parameters[8], 0, 2)
        collision_dist = self.denormalize(parameters[9], 0.01, 0.05)
        drag_damping = self.denormalize(parameters[10], 0, 20)
        dashpot_damping = self.denormalize(parameters[11], 0, 200)

        if self.controller_points is None:
            first_frame_controller_points = None
        else:
            first_frame_controller_points = self.controller_points[0]
        (
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            self.num_object_springs,
        ) = self._init_start(
            self.structure_points,
            first_frame_controller_points,
            object_radius=object_radius,
            object_max_neighbours=object_max_neighbours,
            controller_radius=controller_radius,
            controller_max_neighbours=controller_max_neighbours,
            mask=self.init_masks,
        )

        total_frames = self._get_trainable_total_frames()
        if start_indices is None:
            segment_starts = self._compute_segment_start_indices(total_frames)
        else:
            segment_starts = [int(start) for start in start_indices]
            if len(segment_starts) == 0:
                raise ValueError("start_indices cannot be empty")
            max_required_frame = max(segment_starts) + int(self.segment_len)
            if max_required_frame > total_frames:
                raise ValueError(
                    "Selected CMA window exceeds available online frames: "
                    f"required={max_required_frame}, available={total_frames}"
                )
        batch_size = max(1, min(int(self.batch_size), len(segment_starts)))
        grouped_starts = [
            segment_starts[k : k + batch_size]
            for k in range(0, len(segment_starts), batch_size)
        ]

        total_loss = 0.0
        total_steps = 0
        loss_rows = []
        global_instance_idx = 0
        rollout_starts = []
        rollout_pred_groups = []

        for starts in grouped_starts:
            batch_data = self._build_segment_batch_tensors(starts)
            B = int(batch_data["batch_size"])
            collect_debug = debug_tag is not None
            collect_vertices = collect_debug or bool(return_rollout)
            num_ctrl_single = (
                int(self.controller_points.shape[1])
                if self.controller_points is not None
                else 0
            )

            simulator = SpringMassSystemWarpBatched(
                batch_data["init_vertices_batched"],
                self.init_springs,
                batch_data["init_rest_lengths_batched"],
                self.init_masses[: self.num_all_points],
                dt=cfg.dt,
                num_substeps=cfg.num_substeps,
                spring_Y=global_spring_Y,
                collide_elas=collide_elas,
                collide_fric=collide_fric,
                dashpot_damping=dashpot_damping,
                drag_damping=drag_damping,
                collide_object_elas=collide_object_elas,
                collide_object_fric=collide_object_fric,
                init_masks=self.init_masks,
                collision_dist=collision_dist,
                init_velocities=batch_data["init_velocities_batched"],
                batch_size=B,
                num_object_points_single=self.num_all_points,
                num_control_points_single=num_ctrl_single,
                num_original_points_single=self.num_original_points,
                num_surface_points_single=self.num_surface_points,
                num_object_points=B * self.num_all_points,
                num_surface_points=(
                    B * self.num_surface_points
                    if self.num_surface_points is not None
                    else None
                ),
                num_original_points=(
                    B * self.num_original_points
                    if self.num_original_points is not None
                    else None
                ),
                controller_points=batch_data["controller_points"],
                reverse_z=cfg.reverse_z,
                spring_Y_min=cfg.spring_Y_min,
                spring_Y_max=cfg.spring_Y_max,
                gt_object_points=batch_data["gt_object_points"],
                gt_object_visibilities=batch_data["gt_object_visibilities"],
                gt_object_motions_valid=batch_data["gt_object_motions_valid"],
                self_collision=cfg.self_collision,
                disable_backward=True,
            )

            simulator.set_init_state(
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

            if cfg.data_type == "real":
                simulator.set_acc_count(False)

            if collect_vertices:
                vertices = [
                    wp.to_torch(
                        simulator.wp_states[0].wp_x, requires_grad=False
                    ).cpu()
                ]
            if collect_debug:
                instance_loss_sums = torch.zeros(
                    B, dtype=torch.float32, device=cfg.device
                )
                instance_loss_counts = torch.zeros(
                    B, dtype=torch.float32, device=cfg.device
                )

            for j in range(1, int(batch_data["segment_len"])):
                simulator.set_controller_target(j)
                if simulator.object_collision_flag:
                    simulator.update_collision_graph()

                if cfg.use_graph:
                    wp.capture_launch(simulator.graph)
                else:
                    if cfg.data_type == "real":
                        with simulator.tape:
                            simulator.step()
                            simulator.calculate_loss()
                    else:
                        with simulator.tape:
                            simulator.step()
                            simulator.calculate_simple_loss()

                if cfg.data_type == "real":
                    if wp.to_torch(simulator.acc_count, requires_grad=False)[0] == 0:
                        simulator.set_acc_count(True)
                    simulator.update_acc()

                loss = wp.to_torch(simulator.loss, requires_grad=False)
                if not bool(torch.isfinite(loss).all()):
                    if self.nonfinite_debug_log_count < 20:
                        loss_per_batch = wp.to_torch(
                            simulator.loss_per_batch, requires_grad=False
                        )
                        logger.error(
                            "[CMA-Batch-NaN]: non-finite loss detected "
                            f"starts={starts}, frame_step={j}, "
                            f"loss={loss.detach().cpu().numpy().tolist()}, "
                            f"loss_per_batch={loss_per_batch.detach().cpu().numpy().tolist()}"
                        )
                        x = wp.to_torch(
                            simulator.wp_states[-1].wp_x, requires_grad=False
                        )
                        finite_x = torch.isfinite(x).all(dim=-1)
                        if bool(finite_x.any()):
                            x_finite = x[finite_x]
                            logger.error(
                                "[CMA-Batch-NaN]: state bbox "
                                f"min={x_finite.min(dim=0).values.detach().cpu().numpy().tolist()}, "
                                f"max={x_finite.max(dim=0).values.detach().cpu().numpy().tolist()}, "
                                f"nonfinite_points={(~finite_x).sum().item()}"
                            )
                        else:
                            logger.error("[CMA-Batch-NaN]: all state points are nonfinite")
                        self.nonfinite_debug_log_count += 1
                total_loss += loss.item()
                total_steps += 1
                if collect_debug:
                    loss_per_batch = wp.to_torch(
                        simulator.loss_per_batch, requires_grad=False
                    )
                    instance_loss_sums += loss_per_batch
                    instance_loss_counts += 1.0
                if collect_vertices:
                    vertices.append(
                        wp.to_torch(
                            simulator.wp_states[-1].wp_x, requires_grad=False
                        ).cpu()
                    )

                simulator.clear_loss()
                simulator.set_init_state(
                    simulator.wp_states[-1].wp_x,
                    simulator.wp_states[-1].wp_v,
                )

            if collect_vertices:
                vertices = torch.stack(vertices, dim=0)
            if return_rollout:
                pred = vertices[:, : B * self.num_all_points, :].reshape(
                    vertices.shape[0], B, self.num_all_points, 3
                )
                pred = pred.permute(1, 0, 2, 3).contiguous().numpy().astype(np.float32)
                rollout_starts.extend(int(start) for start in starts)
                rollout_pred_groups.append(pred)

            if collect_debug:
                self._save_batch_instance_videos(
                    debug_tag,
                    starts,
                    vertices[:, : B * self.num_all_points, :],
                    batch_data["controller_points"],
                )
                mean_losses = (
                    instance_loss_sums
                    / torch.clamp(instance_loss_counts, min=1.0)
                ).detach().cpu()
                counts = instance_loss_counts.detach().cpu()
                for local_idx, start_idx in enumerate(starts):
                    loss_rows.append(
                        {
                            "instance": global_instance_idx + local_idx,
                            "start_frame": int(start_idx),
                            "mean_loss": float(mean_losses[local_idx].item()),
                            "num_steps": int(counts[local_idx].item()),
                        }
                    )
                global_instance_idx += B

        if total_steps == 0:
            raise RuntimeError("No batched CMA optimization steps were executed")
        if debug_tag is not None:
            self._save_batch_loss_csv(debug_tag, loss_rows)
        mean_loss = total_loss / total_steps
        if return_rollout:
            if len(rollout_pred_groups) == 0:
                raise RuntimeError("return_rollout=True but no rollout was collected")
            return {
                "loss": float(mean_loss),
                "window_starts": np.asarray(rollout_starts, dtype=np.int64),
                "pred_points": np.concatenate(rollout_pred_groups, axis=0),
            }
        return mean_loss

    def error_func(self, parameters, visualize=False, video_path=None):
        if self.batch_mode and not visualize:
            return self.error_func_batched(parameters)

        global_spring_Y = self.denormalize(
            parameters[0], cfg.spring_Y_min, cfg.spring_Y_max
        )
        object_radius = self.denormalize(parameters[1], 0.01, 0.05)
        object_max_neighbours = int(self.denormalize(parameters[2], 10, 50))
        controller_radius = self.denormalize(parameters[3], 0.01, 0.08)
        controller_max_neighbours = int(self.denormalize(parameters[4], 10, 80))
        collide_elas = parameters[5]
        collide_fric = self.denormalize(parameters[6], 0, 2)
        collide_object_elas = parameters[7]
        collide_object_fric = self.denormalize(parameters[8], 0, 2)
        collision_dist = self.denormalize(parameters[9], 0.01, 0.05)
        drag_damping = self.denormalize(parameters[10], 0, 20)
        dashpot_damping = self.denormalize(parameters[11], 0, 200)

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
            object_radius=object_radius,
            object_max_neighbours=object_max_neighbours,
            controller_radius=controller_radius,
            controller_max_neighbours=controller_max_neighbours,
            mask=self.init_masks,
        )

        self.simulator = SpringMassSystemWarp(
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            dt=cfg.dt,
            num_substeps=cfg.num_substeps,
            spring_Y=global_spring_Y,
            collide_elas=collide_elas,
            collide_fric=collide_fric,
            dashpot_damping=dashpot_damping,
            drag_damping=drag_damping,
            collide_object_elas=collide_object_elas,
            collide_object_fric=collide_object_fric,
            init_masks=self.init_masks,
            collision_dist=collision_dist,
            init_velocities=self.init_velocities,
            num_object_points=self.num_all_points,
            num_surface_points=self.num_surface_points,
            num_original_points=self.num_original_points,
            controller_points=self.controller_points,
            reverse_z=cfg.reverse_z,
            spring_Y_min=cfg.spring_Y_min,
            spring_Y_max=cfg.spring_Y_max,
            gt_object_points=self.object_points,
            gt_object_visibilities=self.object_visibilities,
            gt_object_motions_valid=self.object_motions_valid,
            self_collision=cfg.self_collision,
            disable_backward=True,
        )

        self.simulator.set_init_state(
            self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
        )

        if visualize == True:
            vertices = [
                wp.to_torch(self.simulator.wp_states[0].wp_x, requires_grad=False).cpu()
            ]

        if cfg.data_type == "real":
            self.simulator.set_acc_count(False)

        total_loss = 0.0
        if not visualize:
            # Only optimize on the train frames
            max_frame = cfg.train_frame
        else:
            max_frame = self.dataset.frame_len

        for j in range(1, max_frame):
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
                else:
                    with self.simulator.tape:
                        self.simulator.step()
                        self.simulator.calculate_simple_loss()

            if visualize == True:
                x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
                vertices.append(x.cpu())

            if cfg.data_type == "real":
                if wp.to_torch(self.simulator.acc_count, requires_grad=False)[0] == 0:
                    self.simulator.set_acc_count(True)

                # Update the prev_acc used to calculate the acceleration loss
                self.simulator.update_acc()

            loss = wp.to_torch(self.simulator.loss, requires_grad=False)
            total_loss += loss.item()

            self.simulator.clear_loss()
            # Set the intial state for the next step
            self.simulator.set_init_state(
                self.simulator.wp_states[-1].wp_x,
                self.simulator.wp_states[-1].wp_v,
            )

        total_loss /= cfg.train_frame - 1

        if visualize == True:
            vertices = torch.stack(vertices, dim=0)
            visualize_pc(
                vertices[:, : self.num_all_points, :],
                self.object_colors,
                self.controller_points,
                visualize=False,
                save_video=True,
                save_path=video_path,
            )

        return total_loss
