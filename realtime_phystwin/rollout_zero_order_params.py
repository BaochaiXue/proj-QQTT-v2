import json
import os
import pickle
import shutil
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

InvPhyTrainerWarp = None
OnlineChunkReader = None
OnlineFrameBuffer = None
cfg = None
logger = None
visualize_pc = None
wp = None


def import_runtime_modules():
    global InvPhyTrainerWarp
    global OnlineChunkReader
    global OnlineFrameBuffer
    global cfg
    global logger
    global visualize_pc
    global wp

    import warp as wp_module
    from qqtt import InvPhyTrainerWarp as TrainerWarp
    from qqtt.data import OnlineChunkReader as ChunkReader
    from qqtt.data import OnlineFrameBuffer as FrameBuffer
    from qqtt.utils import cfg as runtime_cfg
    from qqtt.utils import logger as runtime_logger
    from qqtt.utils import visualize_pc as runtime_visualize_pc

    wp = wp_module
    InvPhyTrainerWarp = TrainerWarp
    OnlineChunkReader = ChunkReader
    OnlineFrameBuffer = FrameBuffer
    cfg = runtime_cfg
    logger = runtime_logger
    visualize_pc = runtime_visualize_pc


def load_case_config(case_name):
    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")


def load_camera_config(base_path, case_name):
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)

    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        metadata = json.load(f)
    cfg.intrinsics = np.array(metadata["intrinsics"])
    cfg.WH = metadata["WH"]
    cfg.overlay_path = f"{base_path}/{case_name}/color"


def apply_zero_order_params(params_path):
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    if not isinstance(params, dict):
        raise TypeError(f"Expected dict in {params_path}, got {type(params)}")

    cfg_params = dict(params)
    if "global_spring_Y" in cfg_params:
        cfg_params["init_spring_Y"] = cfg_params.pop("global_spring_Y")
    cfg.update_from_dict(cfg_params)
    return params


def load_online_buffer(online_dir, static_data_path, device):
    reader = OnlineChunkReader(online_dir=online_dir)
    manifest = reader.read_manifest()
    if manifest is None:
        raise FileNotFoundError(f"Cannot read online manifest: {reader.manifest_path}")

    buffer = OnlineFrameBuffer(static_data_path=static_data_path, device=device)
    chunks = reader.load_new_chunks()
    if len(chunks) == 0:
        raise RuntimeError(f"No committed chunks found under {online_dir}")
    buffer.append_chunks(chunks)
    buffer.sync_to_device(device)
    return buffer


def rollout_vertices(trainer, max_frames=None):
    frame_len = int(trainer.dataset.frame_len)
    if max_frames is not None:
        frame_len = min(frame_len, int(max_frames))
    if frame_len < 1:
        raise ValueError("rollout needs at least one frame")

    simulator = trainer.simulator
    simulator.set_init_state(
        simulator.wp_init_vertices,
        simulator.wp_init_velocities,
        pure_inference=True,
    )
    vertices = [
        wp.to_torch(simulator.wp_states[0].wp_x, requires_grad=False).detach().cpu()
    ]

    with wp.ScopedTimer("zero_order_rollout"):
        for frame_idx in tqdm(range(1, frame_len), desc="rollout"):
            if cfg.data_type == "real":
                simulator.set_controller_target(frame_idx, pure_inference=True)
            if simulator.object_collision_flag:
                simulator.update_collision_graph()

            if cfg.use_graph:
                wp.capture_launch(simulator.forward_graph)
            else:
                simulator.step()

            x = wp.to_torch(simulator.wp_states[-1].wp_x, requires_grad=False)
            vertices.append(x.detach().cpu())
            simulator.set_init_state(
                simulator.wp_states[-1].wp_x,
                simulator.wp_states[-1].wp_v,
                pure_inference=True,
            )

    return torch.stack(vertices, dim=0).numpy()


def tensor_to_numpy(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def compute_metrics(trainer, vertices):
    num_frames = int(vertices.shape[0])
    num_original = int(trainer.num_original_points)
    pred = vertices[:, :num_original, :]
    gt = tensor_to_numpy(trainer.train_object_points)[:num_frames, :num_original, :]
    dist = np.linalg.norm(pred - gt, axis=-1)

    mask = np.ones(dist.shape, dtype=bool)
    vis = tensor_to_numpy(getattr(trainer, "train_object_visibilities", None))
    if vis is not None:
        mask &= vis[:num_frames, :num_original].astype(bool)
    valid = tensor_to_numpy(getattr(trainer, "train_object_motions_valid", None))
    if valid is not None:
        mask &= valid[:num_frames, :num_original].astype(bool)

    masked = np.where(mask, dist, np.nan)
    per_frame = np.nanmean(masked, axis=1)
    finite = masked[np.isfinite(masked)]
    if finite.size == 0:
        mean_error = None
        median_error = None
        max_error = None
    else:
        mean_error = float(np.mean(finite))
        median_error = float(np.median(finite))
        max_error = float(np.max(finite))

    return {
        "num_frames": num_frames,
        "num_original_points": num_original,
        "mean_error": mean_error,
        "median_error": median_error,
        "max_error": max_error,
        "first_frame_mean_error": (
            None if not np.isfinite(per_frame[0]) else float(per_frame[0])
        ),
        "last_frame_mean_error": (
            None if not np.isfinite(per_frame[-1]) else float(per_frame[-1])
        ),
        "per_frame_mean_error": [
            None if not np.isfinite(value) else float(value) for value in per_frame
        ],
    }


def atomic_save_npz(output_path, **arrays):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, output_path)


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_save_json(output_path, data):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, default=json_default)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, output_path)


def build_window_starts(num_frames, segment_len, segment_stride):
    segment_len = min(int(segment_len), int(num_frames))
    segment_stride = max(1, int(segment_stride))
    if segment_len < 1:
        raise ValueError("segment_len must be positive")
    starts = list(range(0, num_frames - segment_len + 1, segment_stride))
    if len(starts) == 0:
        starts = [0]
    final_start = num_frames - segment_len
    if final_start not in starts:
        starts.append(final_start)
    return np.array(sorted(set(starts)), dtype=np.int64), segment_len


def export_realtime_rollout(
    trainer,
    vertices,
    realtime_dir,
    segment_len,
    segment_stride,
    iteration_label=0,
    overwrite=False,
):
    realtime_dir = Path(realtime_dir)
    if realtime_dir.exists() and overwrite:
        shutil.rmtree(realtime_dir)
    if realtime_dir.exists() and (realtime_dir / "manifest.json").exists():
        raise FileExistsError(
            f"{realtime_dir} already has a manifest; pass --overwrite to replace it"
        )

    pred_all = vertices[:, : int(trainer.num_all_points), :].astype(np.float32)
    starts, segment_len = build_window_starts(
        pred_all.shape[0],
        segment_len=segment_len,
        segment_stride=segment_stride,
    )
    local_indices = np.arange(segment_len, dtype=np.int64)
    online_frame_indices = starts[:, None] + local_indices[None, :]

    source_frame_indices = tensor_to_numpy(getattr(trainer, "source_frame_indices", None))
    if source_frame_indices is None:
        frame_indices = online_frame_indices
    else:
        source_frame_indices = source_frame_indices.astype(np.int64)
        frame_indices = source_frame_indices[online_frame_indices]

    pred_points = np.stack(
        [pred_all[int(start) : int(start) + segment_len] for start in starts],
        axis=0,
    ).astype(np.float32)
    gt = tensor_to_numpy(trainer.train_object_points).astype(np.float32)
    gt_object_points = np.stack(
        [gt[int(start) : int(start) + segment_len] for start in starts],
        axis=0,
    ).astype(np.float32)

    object_colors = np.empty((0, 0, 0, 3), dtype=np.float32)
    colors = tensor_to_numpy(getattr(trainer, "object_colors", None))
    if colors is not None:
        object_colors = np.stack(
            [colors[int(start) : int(start) + segment_len] for start in starts],
            axis=0,
        ).astype(np.float32)

    object_visibilities = np.empty((0, 0, 0), dtype=np.bool_)
    vis = tensor_to_numpy(getattr(trainer, "train_object_visibilities", None))
    if vis is not None:
        object_visibilities = np.stack(
            [vis[int(start) : int(start) + segment_len] for start in starts],
            axis=0,
        ).astype(np.bool_)

    controller_points = np.empty(
        (pred_points.shape[0], pred_points.shape[1], 0, 3), dtype=np.float32
    )
    ctrl = tensor_to_numpy(getattr(trainer, "controller_points", None))
    if ctrl is not None:
        controller_points = np.stack(
            [ctrl[int(start) : int(start) + segment_len] for start in starts],
            axis=0,
        ).astype(np.float32)

    arrays = {
        "iteration": np.array(int(iteration_label), dtype=np.int64),
        "window_starts": starts,
        "frame_indices": frame_indices,
        "online_frame_indices": online_frame_indices,
        "pred_points": pred_points,
        "gt_object_points": gt_object_points,
        "object_colors": object_colors,
        "object_visibilities": object_visibilities,
        "controller_points": controller_points,
        "num_original_points": np.array(int(trainer.num_original_points), dtype=np.int64),
        "num_surface_points": np.array(int(trainer.num_surface_points), dtype=np.int64),
        "num_all_points": np.array(int(trainer.num_all_points), dtype=np.int64),
        "batch_size": np.array(int(pred_points.shape[0]), dtype=np.int64),
        "real_window_count": np.array(int(pred_points.shape[0]), dtype=np.int64),
        "segment_len": np.array(int(segment_len), dtype=np.int64),
        "timestamp": np.array(float(time.time()), dtype=np.float64),
    }

    first_seen_dir = realtime_dir / "first_seen"
    for window_idx, start in enumerate(starts):
        atomic_save_npz(
            str(first_seen_dir / f"window_{int(start):06d}.npz"),
            iteration=np.array(int(iteration_label), dtype=np.int64),
            first_iteration=np.array(int(iteration_label), dtype=np.int64),
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

    atomic_save_npz(
        str(realtime_dir / "iterations" / f"iter_{int(iteration_label):06d}.npz"),
        **arrays,
    )
    atomic_save_npz(str(realtime_dir / "latest_window.npz"), **arrays)
    manifest = {
        "case_name": cfg.run_name,
        "latest_iteration": int(iteration_label),
        "iterations": [int(iteration_label)],
        "latest_file": "latest_window.npz",
        "iterations_dir": "iterations",
        "first_seen_dir": "first_seen",
        "fps": int(cfg.FPS),
        "image_width": int(cfg.WH[0]) if cfg.WH is not None else None,
        "image_height": int(cfg.WH[1]) if cfg.WH is not None else None,
        "window_starts": starts.astype(int).tolist(),
        "segment_len": int(segment_len),
        "num_original_points": int(trainer.num_original_points),
        "num_surface_points": int(trainer.num_surface_points),
        "num_all_points": int(trainer.num_all_points),
        "timestamp": float(arrays["timestamp"]),
    }
    atomic_save_json(str(realtime_dir / "manifest.json"), manifest)
    return manifest


def main():
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, default="data/different_types")
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--experiments_dir", type=str, default="experiments_online_cma")
    parser.add_argument("--params_path", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--vis_cam_idx", type=int, default=0)
    parser.add_argument("--use_online_data", action="store_true")
    parser.add_argument("--online_base_path", type=str, default="online_data")
    parser.add_argument("--online_dir", type=str, default=None)
    parser.add_argument("--static_data_path", type=str, default=None)
    parser.add_argument("--realtime_vis", action="store_true")
    parser.add_argument("--realtime_vis_dir", type=str, default=None)
    parser.add_argument("--segment_len", type=int, default=32)
    parser.add_argument("--segment_stride", type=int, default=30)
    parser.add_argument("--iteration_label", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import_runtime_modules()

    case_name = args.case_name
    base_dir = os.path.join(args.experiments_dir, case_name)
    params_path = args.params_path or os.path.join(base_dir, "optimal_params.pkl")
    out_dir = args.out_dir or os.path.join(base_dir, "zero_order_rollout")
    static_data_path = args.static_data_path or os.path.join(
        args.base_path, case_name, "final_data.pkl"
    )

    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Cannot find zero-order params: {params_path}")
    if os.path.exists(out_dir) and not args.overwrite:
        raise FileExistsError(f"{out_dir} exists; pass --overwrite to replace files")
    os.makedirs(out_dir, exist_ok=True)

    load_case_config(case_name)
    cfg.device = args.device
    load_camera_config(args.base_path, case_name)
    optimal_params = apply_zero_order_params(params_path)

    dataset_override = None
    data_source = "static"
    if args.use_online_data:
        online_dir = args.online_dir or os.path.join(args.online_base_path, case_name)
        dataset_override = load_online_buffer(
            online_dir=online_dir,
            static_data_path=static_data_path,
            device=args.device,
        )
        data_source = str(online_dir)

    logger.set_log_file(path=out_dir, name="zero_order_rollout_log")
    trainer = InvPhyTrainerWarp(
        data_path=static_data_path,
        base_dir=out_dir,
        pure_inference_mode=True,
        device=args.device,
        dataset_override=dataset_override,
    )

    vertices = rollout_vertices(trainer, max_frames=args.max_frames)
    rollout_path = os.path.join(out_dir, "rollout.pkl")
    with open(rollout_path, "wb") as f:
        pickle.dump(vertices, f)

    metrics = compute_metrics(trainer, vertices)
    metrics.update(
        {
            "case_name": case_name,
            "params_path": params_path,
            "data_source": data_source,
            "fps": int(cfg.FPS),
            "dt": float(cfg.dt),
            "num_substeps": int(cfg.num_substeps),
            "optimal_params": optimal_params,
            "rollout_path": rollout_path,
        }
    )
    metrics_path = os.path.join(out_dir, "rollout_metrics.json")
    atomic_save_json(metrics_path, metrics)

    npz_path = os.path.join(out_dir, "rollout.npz")
    atomic_save_npz(
        npz_path,
        vertices=vertices.astype(np.float32),
        gt_object_points=tensor_to_numpy(trainer.train_object_points)[
            : vertices.shape[0]
        ].astype(np.float32),
        controller_points=tensor_to_numpy(trainer.controller_points)[
            : vertices.shape[0]
        ].astype(np.float32),
        source_frame_indices=(
            tensor_to_numpy(getattr(trainer, "source_frame_indices", None))[
                : vertices.shape[0]
            ].astype(np.int64)
            if getattr(trainer, "source_frame_indices", None) is not None
            else np.arange(vertices.shape[0], dtype=np.int64)
        ),
    )

    if args.save_video:
        video_path = os.path.join(out_dir, "rollout.mp4")
        visualize_pc(
            vertices[:, : int(trainer.num_all_points), :],
            trainer.object_colors[: vertices.shape[0]],
            trainer.controller_points[: vertices.shape[0]],
            visualize=False,
            save_video=True,
            save_path=video_path,
            vis_cam_idx=args.vis_cam_idx,
        )
        metrics["video_path"] = video_path
        atomic_save_json(metrics_path, metrics)

    if args.realtime_vis:
        realtime_dir = args.realtime_vis_dir or os.path.join(out_dir, "realtime")
        manifest = export_realtime_rollout(
            trainer=trainer,
            vertices=vertices,
            realtime_dir=realtime_dir,
            segment_len=args.segment_len,
            segment_stride=args.segment_stride,
            iteration_label=args.iteration_label,
            overwrite=args.overwrite,
        )
        metrics["realtime_vis_dir"] = realtime_dir
        metrics["realtime_window_starts"] = manifest["window_starts"]
        atomic_save_json(metrics_path, metrics)

    print(f"[OK] rollout saved: {rollout_path}")
    print(f"[OK] metrics saved: {metrics_path}")
    if args.realtime_vis:
        print(f"[OK] realtime export: {metrics['realtime_vis_dir']}")
    if metrics["mean_error"] is not None:
        print(
            "[METRIC] "
            f"mean={metrics['mean_error']:.6f}, "
            f"median={metrics['median_error']:.6f}, "
            f"last={metrics['last_frame_mean_error']:.6f}"
        )


if __name__ == "__main__":
    main()
