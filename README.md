# proj-QQTT

## Overview

`proj-QQTT` builds digital twins for deformable objects such as cloth, rope, and soft packages.

This repository is not only a reconstruction project and not only a simulator. The main goal is to connect:

1. real multi-view RGB-D observations,
2. inverse physics over a differentiable spring-mass model,
3. Gaussian Splatting for realistic appearance rendering,
4. evaluation and interactive control.

In practice, the project takes a real manipulation sequence, converts it into object and controller trajectories in a shared 3D frame, fits physical parameters to explain the observed motion, and then uses the recovered dynamics to drive a Gaussian Splatting appearance model.

## End-To-End Pipeline

The full workflow is:

1. Record synchronized RGB-D manipulation data with 3 RealSense cameras.
2. Calibrate all cameras into a shared world frame.
3. Segment the object and controller, run dense tracking, and lift observations into 3D.
4. Build `final_data.pkl`, which contains the object trajectory, controller trajectory, visibility masks, and structural points used by the simulator.
5. Run a two-stage inverse-physics pipeline:
   stage 1: coarse global parameter search with CMA-ES
   stage 2: differentiable training with Warp
6. Export the case as a Gaussian Splatting dataset and train a static appearance model.
7. Drive the Gaussian model with predicted dynamics to render dynamic videos.
8. Evaluate geometry, tracking, and rendering quality, or run interactive demos / teleoperation.

## Repository Map

- `qqtt/env/`: camera capture and calibration, including multi-camera RealSense support.
- `qqtt/data/`: dataset loaders for real and synthetic trajectories.
- `qqtt/model/diff_simulator/`: differentiable spring-mass simulators implemented with Warp.
- `qqtt/engine/`: inverse-physics optimization and training logic.
- `data_process/`: segmentation, tracking, 3D lifting, alignment, and final data generation.
- `gaussian_splatting/`: modified Gaussian Splatting code used for appearance modeling and rendering.
- `configs/`: physics and training defaults for real, cloth, and synthetic settings.
- top-level scripts: entry points for collection, processing, training, rendering, evaluation, and interaction.

## Core Components

### 1. Camera And Data Collection

The data entry point is the camera subsystem under `qqtt/env/camera/`.

- `qqtt/env/camera/camera_system.py` manages 3 RealSense cameras, synchronization, recording, and ChArUco calibration.
- `qqtt/env/camera/realsense/` contains the low-level multi-process RealSense capture stack.
- `cameras_calibrate.py` runs calibration and saves `calibrate.pkl`.
- `record_data.py` records color, depth, and metadata for a manipulation sequence.

This part produces the raw multi-view RGB-D observations used by the whole project.

### 2. Data Processing

`process_data.py` is the main post-processing entry point. It orchestrates:

- video segmentation,
- optional shape-prior generation,
- dense tracking,
- RGB-D to world-coordinate point cloud lifting,
- mask cleanup,
- 3D trajectory construction,
- final sampling into `final_data.pkl`.

For real data, the important final fields are:

- `object_points`
- `object_visibilities`
- `object_motions_valid`
- `controller_points`
- `surface_points`
- `interior_points`

These are loaded by `qqtt/data/real_data.py` and directly consumed by the inverse-physics trainer.

### 3. Inverse Physics

The physical core of the project is a differentiable spring-mass simulator built with Warp.

- `qqtt/model/diff_simulator/spring_mass_warp.py` defines the main simulator.
- `qqtt/model/diff_simulator/spring_mass_warp_accelerate.py` provides an accelerated rollout version.
- `qqtt/engine/cma_optimize_warp.py` performs coarse parameter search with CMA-ES.
- `qqtt/engine/trainer_warp.py` performs differentiable training and inference.

The physical loss is based on 3D geometric consistency rather than image reconstruction alone. The trainer matches simulated motion to observed object trajectories using Chamfer-style and tracking losses.

### 4. Gaussian Splatting

The appearance branch models how the object should look.

- `export_gaussian_data.py` packages a case into a Gaussian Splatting dataset.
- `gs_train.py` trains the static Gaussian model.
- `gs_render.py` renders static results.
- `gs_render_dynamics.py` uses the predicted controller / object motion to animate the Gaussian model.

This is where physics and appearance meet:

- inverse physics predicts how the object moves,
- Gaussian Splatting renders how that moving object should look.

### 5. Evaluation And Interaction

The repository includes tools for:

- geometric / tracking evaluation: `evaluate_prediction.py`
- rendering evaluation: scripts under `gaussian_splatting/custom_eval*.py`
- result visualization: `visualize_prediction.py`, `visualize_render_results.py`, `visualize_SG_results.py`
- interactive manipulation demos: `interactive_playground.py`
- robot-facing rollout and teleoperation: `physics_dynamics_module.py`, `interactive_robot_teleop.py`

## Environment Setup

```bash
# if default CUDA is not 12.1
# export PATH={YOUR_DIR}/cuda/cuda-12.1/bin:$PATH
# export LD_LIBRARY_PATH={YOUR_DIR}/cuda/cuda-12.1/lib64:$LD_LIBRARY_PATH

conda create -y -n qqtt python=3.10
conda activate qqtt
bash ./env_install/env_install.sh
bash ./env_install/download_pretrained_models.sh

# download pretrained gaussians from:
# https://drive.google.com/file/d/1ffNOGlPPJ20XKjxfR-sSl6K6zodpaRHK/view?usp=sharing
# and place it under the gaussian_splatting folder
```

## Data Collection

Set the footswitch button:

```bash
footswitch -1 -k space -2 -k space -3 -k space
```

Record a sequence. The controller should stay attached to the same region of the object throughout the sequence.

```bash
python cameras_calibrate.py
python record_data.py
```

Raw recordings are saved under `data_collect/<timestamp>/` and typically contain:

- `color/<camera_id>/<step>.png`
- `depth/<camera_id>/<step>.npy`
- `metadata.json`
- `calibrate.pkl`

## Processing Raw Recordings

If you start from raw collected data, first align and trim the sequence:

```bash
python data_process/record_data_align.py --case_name [] --start [] --end []
```

After this step, the processed case should include color frames and videos, depth, `calibrate.pkl`, and `metadata.json`.

Then run the main processing pipeline:

```bash
python process_data.py --base_path [] --case_name [] --category []
```

If the case uses a shape prior, add:

```bash
python process_data.py --base_path [] --case_name [] --category [] --shape_prior
```

Important outputs include:

- `mask/`
- `pcd/*.npz`
- `track_process_data.pkl`
- optional `shape/`
- `final_data.pkl`
- `split.json`

## Inverse-Physics Training

The inverse-physics pipeline is two-stage.

### Stage 1: CMA-ES Parameter Search

This stage searches global parameters such as spring stiffness initialization, neighborhood radii, damping, and collision settings.

```bash
python optimize_cma.py --base_path [] --case_name [] --train_frame []
```

Output:

- `experiments_optimization/<case_name>/optimal_params.pkl`

### Stage 2: Differentiable Training

This stage loads the optimized global parameters and trains the differentiable physical model.

```bash
python train_warp.py --base_path [] --case_name [] --train_frame []
```

Outputs typically include:

- `experiments/<case_name>/train/best_*.pth`
- training videos
- logs

### Inference

```bash
python inference_warp.py --base_path [] --case_name []
```

This produces rollout results used by evaluation and dynamic rendering.

For batched physics-only runs across many cases, see:

- `script_optimize.py`
- `script_train.py`
- `script_inference.py`
- `final_pipeline.sh`

## Gaussian Splatting Pipeline

First export the processed cases into the Gaussian dataset format:

```bash
python export_gaussian_data.py
```

Then train and render the static Gaussian appearance models:

```bash
bash gs_run.sh
```

To render dynamic Gaussian results driven by the inferred motion:

```bash
bash gs_run_simulate.sh
```

The dynamic rendering step reads motion predictions from `experiments/<case_name>/inference.pkl`.

## Evaluation

Useful evaluation and visualization scripts include:

- `evaluate_prediction.py`: Chamfer and tracking errors for predicted trajectories.
- `evaluate_outdomain.py`: transfer / out-of-domain evaluation.
- `visualize_prediction.py`: overlay and inspect predicted 3D dynamics.
- `visualize_render_results.py`: inspect rendered videos.
- `prepare_results.py`: aggregate experiment outputs.

## What This Repository Is Really Doing

In one sentence:

This project learns a controllable digital twin of a deformable object from real multi-view manipulation data, then renders that digital twin with Gaussian Splatting.

That means the repository is simultaneously about:

- data collection in the real world,
- 3D trajectory construction,
- inverse physics,
- realistic appearance modeling,
- rollout, evaluation, and interaction.
