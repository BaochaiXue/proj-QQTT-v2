import glob
import json
import os
import pickle
import re
from argparse import ArgumentParser

import numpy as np
import torch
import warp as wp
from tqdm import tqdm

from qqtt import InvPhyTrainerWarp
from qqtt.utils import cfg, logger


def checkpoint_sort_key(path):
    name = os.path.basename(path)
    match = re.search(r"_(\d+)\.pth$", name)
    iter_idx = int(match.group(1)) if match else -1
    is_best = 1 if name.startswith("best_") else 0
    return iter_idx, is_best, name


def checkpoint_output_name(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    match = re.search(r"^(iter|best)_(\d+)$", stem)
    if match:
        return f"{match.group(1)}_{int(match.group(2)):04d}.pkl"
    return f"{stem}.pkl"


def torch_load_checkpoint(model_path):
    try:
        return torch.load(model_path, map_location=cfg.device, weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=cfg.device)


def load_checkpoint_into_simulator(trainer, model_path):
    logger.info(f"[Inference-By-Checkpoint]: load {model_path}")
    checkpoint = torch_load_checkpoint(model_path)

    spring_Y = checkpoint["spring_Y"]
    collide_elas = checkpoint["collide_elas"]
    collide_fric = checkpoint["collide_fric"]
    collide_object_elas = checkpoint["collide_object_elas"]
    collide_object_fric = checkpoint["collide_object_fric"]

    if len(spring_Y) != trainer.simulator.n_springs:
        raise AssertionError(
            "Check if the loaded checkpoint matches the config file to connect the springs"
        )

    trainer.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
    trainer.simulator.set_collide(
        collide_elas.detach().clone(), collide_fric.detach().clone()
    )
    trainer.simulator.set_collide_object(
        collide_object_elas.detach().clone(),
        collide_object_fric.detach().clone(),
    )


def rollout_vertices(trainer):
    frame_len = trainer.dataset.frame_len
    simulator = trainer.simulator
    simulator.set_init_state(
        simulator.wp_init_vertices,
        simulator.wp_init_velocities,
        pure_inference=True,
    )
    vertices = [
        wp.to_torch(simulator.wp_states[0].wp_x, requires_grad=False).detach().cpu()
    ]

    with wp.ScopedTimer("simulate_checkpoint"):
        for frame_idx in tqdm(range(1, frame_len)):
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


def build_trainer(base_path, case_name, experiments_dir):
    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    base_dir = os.path.join(experiments_dir, case_name)

    with open(os.path.join(base_path, case_name, "calibrate.pkl"), "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)

    with open(os.path.join(base_path, case_name, "metadata.json"), "r") as f:
        metadata = json.load(f)
    cfg.intrinsics = np.array(metadata["intrinsics"])
    cfg.WH = metadata["WH"]
    cfg.overlay_path = os.path.join(base_path, case_name, "color")

    logger.set_log_file(path=base_dir, name="inference_by_checkpoint_log")
    return InvPhyTrainerWarp(
        data_path=os.path.join(base_path, case_name, "final_data.pkl"),
        base_dir=base_dir,
        pure_inference_mode=True,
    )


def find_cases(experiments_dir, case_name=None):
    if case_name is not None:
        return [case_name]

    train_dirs = sorted(glob.glob(os.path.join(experiments_dir, "*", "train")))
    cases = []
    for train_dir in train_dirs:
        case = os.path.basename(os.path.dirname(train_dir))
        if glob.glob(os.path.join(train_dir, "*.pth")):
            cases.append(case)
    return cases


def run_case(args, case_name):
    train_dir = os.path.join(args.experiments_dir, case_name, "train")
    checkpoint_paths = sorted(
        glob.glob(os.path.join(train_dir, args.checkpoint_pattern)),
        key=checkpoint_sort_key,
    )
    if len(checkpoint_paths) == 0:
        print(
            f"[SKIP] {case_name}: no checkpoints matched "
            f"{args.checkpoint_pattern} under {train_dir}"
        )
        return

    if args.out_dir is None:
        out_dir = os.path.join(args.experiments_dir, case_name, "inference_by_iter")
    else:
        out_dir = os.path.join(args.out_dir, case_name)
    os.makedirs(out_dir, exist_ok=True)

    print(
        f"[CASE] {case_name}: {len(checkpoint_paths)} checkpoints, "
        f"output={out_dir}"
    )
    trainer = build_trainer(args.base_path, case_name, args.experiments_dir)

    for checkpoint_path in checkpoint_paths:
        out_path = os.path.join(out_dir, checkpoint_output_name(checkpoint_path))
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[SKIP] exists: {out_path}")
            continue

        load_checkpoint_into_simulator(trainer, checkpoint_path)
        vertices = rollout_vertices(trainer)
        with open(out_path, "wb") as f:
            pickle.dump(vertices, f)
        print(f"[OK] saved {out_path}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, default="data/different_types")
    parser.add_argument("--experiments_dir", type=str, default="experiments")
    parser.add_argument("--case_name", type=str, default=None)
    parser.add_argument("--checkpoint_pattern", type=str, default="*.pth")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cases = find_cases(args.experiments_dir, args.case_name)
    if len(cases) == 0:
        print(f"[WARN] no checkpoint cases found under {args.experiments_dir}")
        return

    for case_name in cases:
        run_case(args, case_name)


if __name__ == "__main__":
    main()
