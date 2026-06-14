from .realsense import MultiRealsense, SingleRealsense
from .recording_metadata import build_recording_metadata as build_recording_metadata_payload
from .calibration_metadata import build_calibration_metadata, write_calibration_metadata
from .calibration_boards import (
    charuco_board_config_to_metadata,
    create_charuco_board,
    get_charuco_chessboard_corners,
    get_calibration_board_config,
)
from .defaults import (
    DEFAULT_COLOR_EXPOSURE_OVERRIDES,
    DEFAULT_COLOR_GAIN_OVERRIDES,
    DEFAULT_EXPOSURE,
    DEFAULT_FPS,
    DEFAULT_GAIN,
    DEFAULT_HEIGHT,
    DEFAULT_NUM_CAM,
    DEFAULT_WHITE_BALANCE,
    DEFAULT_WIDTH,
    resolve_per_camera_control_values,
)
from multiprocessing.managers import SharedMemoryManager
import numpy as np
import time
import cv2
import json
import os
import pickle
from pathlib import Path
from typing import Optional, Any

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

DEFAULT_EXPOSURE_OVERRIDES = DEFAULT_COLOR_EXPOSURE_OVERRIDES
DEFAULT_GAIN_OVERRIDES = DEFAULT_COLOR_GAIN_OVERRIDES
DEFAULT_CAMERA_START_TIMEOUT_S = 30.0

CAPTURE_MODE_CONFIGS = {
    "color": {
        "enable_color": True,
        "enable_depth": False,
        "enable_ir_left": False,
        "enable_ir_right": False,
        "process_depth": False,
        "streams_present": ["color"],
    },
    "rgbd": {
        "enable_color": True,
        "enable_depth": True,
        "enable_ir_left": False,
        "enable_ir_right": False,
        "process_depth": True,
        "streams_present": ["color", "depth"],
    },
    "stereo_ir": {
        "enable_color": True,
        "enable_depth": False,
        "enable_ir_left": True,
        "enable_ir_right": True,
        "process_depth": False,
        "streams_present": ["color", "ir_left", "ir_right"],
    },
    "both_eval": {
        "enable_color": True,
        "enable_depth": True,
        "enable_ir_left": True,
        "enable_ir_right": True,
        "process_depth": True,
        "streams_present": ["color", "depth", "ir_left", "ir_right"],
    },
}

CALIBRATION_WORLD_FRAME_OPENCV_BOARD_NATIVE = "opencv-board-native"
CALIBRATION_WORLD_FRAME_ROBOPIL_RX180 = "robopil-rx180"
CALIBRATION_WORLD_FRAME_CHOICES = {
    CALIBRATION_WORLD_FRAME_OPENCV_BOARD_NATIVE,
    CALIBRATION_WORLD_FRAME_ROBOPIL_RX180,
}


def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def _dist_coeffs_from_metadata(metadata: dict[str, Any], key: str):
    coeffs = metadata.get(key)
    if coeffs is None:
        return None
    coeffs_array = np.asarray(coeffs, dtype=np.float64).reshape(-1, 1)
    if coeffs_array.size == 0:
        return None
    return coeffs_array


def _dist_coeffs_to_metadata(coeffs) -> list[float] | None:
    if coeffs is None:
        return None
    return [float(value) for value in np.asarray(coeffs, dtype=np.float64).reshape(-1)]


def _apply_calibration_world_frame(R_board2cam: np.ndarray, convention: str) -> np.ndarray:
    if convention == CALIBRATION_WORLD_FRAME_OPENCV_BOARD_NATIVE:
        return R_board2cam
    if convention == CALIBRATION_WORLD_FRAME_ROBOPIL_RX180:
        rx180 = np.diag([1.0, -1.0, -1.0])
        return R_board2cam @ rx180
    raise ValueError(
        f"Unsupported calibration world frame {convention!r}. "
        f"Choices: {sorted(CALIBRATION_WORLD_FRAME_CHOICES)}"
    )


def _rotation_angle_deg(R: np.ndarray) -> float:
    trace_value = float(np.trace(R))
    cos_angle = max(-1.0, min(1.0, (trace_value - 1.0) / 2.0))
    return float(np.degrees(np.arccos(cos_angle)))


def _compute_pose_stability(c2w_samples: list[list[np.ndarray]]) -> dict[str, Any] | None:
    if len(c2w_samples) < 2 or not c2w_samples or len(c2w_samples[0]) < 2:
        return None
    num_cam = len(c2w_samples[0])
    result: dict[str, Any] = {
        "sample_count": len(c2w_samples),
        "relative_to_camera_index": 0,
        "per_camera": {},
    }
    for camera_idx in range(1, num_cam):
        relatives = [
            np.linalg.inv(sample[0]) @ sample[camera_idx]
            for sample in c2w_samples
        ]
        translations = np.asarray([rel[:3, 3] for rel in relatives], dtype=np.float64)
        base_R = relatives[0][:3, :3]
        rotation_angles = [
            _rotation_angle_deg(base_R.T @ rel[:3, :3])
            for rel in relatives[1:]
        ]
        result["per_camera"][f"cam{camera_idx}"] = {
            "translation_std_m": translations.std(axis=0).tolist(),
            "translation_norm_std_m": float(np.linalg.norm(translations, axis=1).std()),
            "rotation_angle_max_deg": max(rotation_angles) if rotation_angles else 0.0,
        }
    return result


class CameraSystem:
    def __init__(
        self,
        WH=(DEFAULT_WIDTH, DEFAULT_HEIGHT),
        fps=DEFAULT_FPS,
        num_cam=DEFAULT_NUM_CAM,
        serial_numbers=None,
        capture_mode="rgbd",
        emitter="auto",
        exposure=DEFAULT_EXPOSURE,
        gain=DEFAULT_GAIN,
        white_balance=DEFAULT_WHITE_BALANCE,
        exposure_overrides=None,
        gain_overrides=None,
        calibration_reference_serials=None,
        enable_keyboard_listener=True,
        camera_start_timeout_s=DEFAULT_CAMERA_START_TIMEOUT_S,
    ):
        self.WH = WH
        self.fps = fps
        self.listener: Optional[Any] = None
        self._keyboard = None
        self._stopped = False

        if capture_mode not in CAPTURE_MODE_CONFIGS:
            raise ValueError(f"Unsupported capture_mode: {capture_mode}")
        if emitter not in {"on", "off", "auto"}:
            raise ValueError(f"Unsupported emitter: {emitter}")

        connected_serials = SingleRealsense.get_connected_devices_serial()
        self.connected_serial_numbers = list(connected_serials)
        if serial_numbers is not None:
            serial_numbers = list(serial_numbers)
            if len(serial_numbers) != len(set(serial_numbers)):
                raise ValueError(f"Requested serials contain duplicates: {serial_numbers}")
            missing = [serial for serial in serial_numbers if serial not in connected_serials]
            if missing:
                raise AssertionError(f"Requested serials not connected: {missing}")
            self.serial_numbers = serial_numbers
        else:
            if len(connected_serials) < num_cam:
                raise AssertionError(f"Only {len(connected_serials)} cameras are connected.")
            self.serial_numbers = connected_serials[:num_cam]
        if calibration_reference_serials is not None:
            calibration_reference_serials = list(calibration_reference_serials)
            if len(calibration_reference_serials) != len(set(calibration_reference_serials)):
                raise ValueError(f"Calibration reference serials contain duplicates: {calibration_reference_serials}")
            missing_from_calibration = [
                serial for serial in self.serial_numbers if serial not in calibration_reference_serials
            ]
            if missing_from_calibration:
                raise ValueError(
                    "Calibration reference serials do not cover selected cameras. "
                    f"missing={missing_from_calibration}, calibration_reference_serials={calibration_reference_serials}"
                )
            self.calibration_reference_serials = calibration_reference_serials
        else:
            self.calibration_reference_serials = list(self.connected_serial_numbers)
        self.num_cam = len(self.serial_numbers)
        self.capture_mode = capture_mode
        self.capture_config = CAPTURE_MODE_CONFIGS[capture_mode]
        self.streams_present = list(self.capture_config["streams_present"])
        self.emitter = emitter
        self.white_balance = white_balance

        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()

        self.realsense = MultiRealsense(
            serial_numbers=self.serial_numbers,
            shm_manager=self.shm_manager,
            resolution=(self.WH[0], self.WH[1]),
            capture_fps=self.fps,
            enable_color=self.capture_config["enable_color"],
            enable_depth=self.capture_config["enable_depth"],
            process_depth=self.capture_config["process_depth"],
            enable_ir_left=self.capture_config["enable_ir_left"],
            enable_ir_right=self.capture_config["enable_ir_right"],
            emitter=emitter,
            verbose=False,
        )
        # Some camera settings
        if exposure_overrides is None:
            exposure_overrides = DEFAULT_EXPOSURE_OVERRIDES
        if gain_overrides is None:
            gain_overrides = DEFAULT_GAIN_OVERRIDES
        exposure_values = resolve_per_camera_control_values(
            exposure,
            overrides=exposure_overrides,
            serial_numbers=self.serial_numbers,
            label="exposure",
        )
        gain_values = resolve_per_camera_control_values(
            gain,
            overrides=gain_overrides,
            serial_numbers=self.serial_numbers,
            label="gain",
        )
        self.exposure = exposure_values
        self.gain = gain_values
        self.realsense.set_exposure(exposure=exposure_values, gain=gain_values)
        self.realsense.set_white_balance(white_balance)

        try:
            self.realsense.start(start_timeout_s=camera_start_timeout_s)
            time.sleep(3)
            self.stream_metadata = self.realsense.get_stream_metadata()
        except Exception:
            self.stop(wait=True)
            raise
        self.recording = False
        self.end = False
        if enable_keyboard_listener:
            try:
                from pynput import keyboard as pynput_keyboard
            except ImportError as e:
                raise ImportError(
                    "pynput is required when enable_keyboard_listener=True. "
                    "Set enable_keyboard_listener=False for calibration-only usage."
                ) from e
            self._keyboard = pynput_keyboard
            self.listener = self._keyboard.Listener(on_press=self.on_press)
            self.listener.start()
        print("Camera system is ready.")

    def stop(self, wait=True):
        if self._stopped:
            return
        self._stopped = True
        self.end = True
        listener = getattr(self, "listener", None)
        if listener is not None:
            try:
                listener.stop()
            except Exception:
                pass
            self.listener = None
        realsense = getattr(self, "realsense", None)
        if realsense is not None:
            try:
                realsense.stop(wait=wait)
            except Exception:
                pass
        shm_manager = getattr(self, "shm_manager", None)
        if shm_manager is not None:
            try:
                shm_manager.shutdown()
            except Exception:
                pass
            self.shm_manager = None

    def get_observation(self):
        # Used to get the latest observations from all cameras
        data = self._get_sync_frame()
        # TODO: Process the data when needed
        return data

    def _get_sync_frame(self, k=4):
        assert self.realsense.is_ready

        # Get the latest k frames from all cameras, and picked the latest synchronized frames
        last_realsense_data = self.realsense.get(k=k)
        timestamp_list = [
            last_realsense_data[camera_idx]["timestamp"][-1]
            for camera_idx in range(self.num_cam)
        ]
        last_timestamp = np.min(timestamp_list)

        data = {}
        for camera_idx in range(self.num_cam):
            value = last_realsense_data[camera_idx]
            this_timestamps = value["timestamp"]
            min_diff = 10
            best_idx = None
            for i, this_timestamp in enumerate(this_timestamps):
                diff = np.abs(this_timestamp - last_timestamp)
                if diff < min_diff:
                    min_diff = diff
                    best_idx = i
            # remap key, step_idx is different, timestamp can be the same when some frames are lost
            data[camera_idx] = {}
            for key in self.streams_present:
                if key in value:
                    data[camera_idx][key] = value[key][best_idx]
            data[camera_idx]["timestamp"] = value["timestamp"][best_idx]
            data[camera_idx]["step_idx"] = value["step_idx"][best_idx]

        return data

    def on_press(self, key):
        if self._keyboard is None:
            return
        try:
            if key == self._keyboard.Key.space:
                if self.recording == False:
                    self.recording = True
                    print("Start recording")
                else:
                    self.recording = False
                    self.end = True
        except AttributeError:
            pass

    def record(self, output_path, max_frames=None):
        output_path = str(output_path)
        exist_dir(output_path)

        for stream_name in self.streams_present:
            exist_dir(f"{output_path}/{stream_name}")
            for i in range(self.num_cam):
                exist_dir(f"{output_path}/{stream_name}/{i}")

        metadata = self.build_recording_metadata()
        for i in range(self.num_cam):
            metadata["recording"][i] = {}

        if max_frames is not None:
            self.recording = True

        last_step_idxs = [-1] * self.num_cam
        frame_counts = [0] * self.num_cam
        progress_interval_s = 1.0
        stall_timeout_s = 15.0 if max_frames is not None else None
        last_progress_time = time.time()
        last_progress_time_by_camera = [last_progress_time] * self.num_cam
        last_log_time = last_progress_time

        try:
            while not self.end:
                if not self.recording:
                    time.sleep(0.01)
                    continue

                last_realsense_data = self.realsense.get()
                timestamps = [
                    last_realsense_data[i]["timestamp"].item()
                    for i in range(self.num_cam)
                ]
                step_idxs = [
                    last_realsense_data[i]["step_idx"].item()
                    for i in range(self.num_cam)
                ]

                any_progress = False
                if not all(
                    [step_idxs[i] == last_step_idxs[i] for i in range(self.num_cam)]
                ):
                    for i in range(self.num_cam):
                        if last_step_idxs[i] != step_idxs[i]:
                            time_stamp = timestamps[i]
                            step_idx = step_idxs[i]
                            metadata["recording"][i][step_idx] = time_stamp
                            for stream_name in self.streams_present:
                                if stream_name not in last_realsense_data[i]:
                                    continue
                                stream_value = last_realsense_data[i][stream_name]
                                if stream_name == "depth":
                                    np.save(f"{output_path}/{stream_name}/{i}/{step_idx}.npy", stream_value)
                                else:
                                    cv2.imwrite(f"{output_path}/{stream_name}/{i}/{step_idx}.png", stream_value)
                            last_step_idxs[i] = step_idx
                            frame_counts[i] = len(metadata["recording"][i])
                            last_progress_time_by_camera[i] = time.time()
                            any_progress = True

                now = time.time()
                if any_progress:
                    last_progress_time = now

                if now - last_log_time >= progress_interval_s:
                    print(
                        f"[record] counts={frame_counts} "
                        f"steps={last_step_idxs} "
                        f"target={max_frames}",
                        flush=True,
                    )
                    last_log_time = now

                if max_frames is not None and min(frame_counts) >= int(max_frames):
                    self.end = True

                if stall_timeout_s is not None and not self.end:
                    lagging_camera_idxs = [
                        i
                        for i in range(self.num_cam)
                        if frame_counts[i] < int(max_frames)
                        and (now - last_progress_time_by_camera[i]) >= stall_timeout_s
                    ]
                    if lagging_camera_idxs:
                        lagging_serials = [
                            self.serial_numbers[i] if i < len(self.serial_numbers) else f"cam{i}"
                            for i in lagging_camera_idxs
                        ]
                        raise RuntimeError(
                            "Recording partially stalled before every camera reached the requested "
                            "frame target. "
                            f"lagging_camera_idxs={lagging_camera_idxs}, "
                            f"lagging_serials={lagging_serials}, "
                            f"counts={frame_counts}, steps={last_step_idxs}"
                        )

                if (
                    stall_timeout_s is not None
                    and not self.end
                    and (now - last_progress_time) >= stall_timeout_s
                ):
                    raise RuntimeError(
                        "Recording stalled before every camera reached the requested "
                        f"frame target. counts={frame_counts}, steps={last_step_idxs}"
                    )

            print("End recording")
            with open(f"{output_path}/metadata.json", "w") as f:
                json.dump(metadata, f)
        finally:
            if self.listener is not None:
                self.listener.stop()
            self.realsense.stop()

    def build_recording_metadata(self):
        return build_recording_metadata_payload(
            serial_numbers=self.serial_numbers,
            calibration_reference_serials=self.calibration_reference_serials,
            capture_mode=self.capture_mode,
            streams_present=self.streams_present,
            fps=self.fps,
            WH=self.WH,
            emitter_request=self.emitter,
            stream_metadata=self.stream_metadata,
        )

    def calibrate(
        self,
        visualize=True,
        board_config=None,
        world_frame_convention=CALIBRATION_WORLD_FRAME_OPENCV_BOARD_NATIVE,
        calibration_samples=1,
    ):
        if world_frame_convention not in CALIBRATION_WORLD_FRAME_CHOICES:
            raise ValueError(
                f"Unsupported calibration world frame {world_frame_convention!r}. "
                f"Choices: {sorted(CALIBRATION_WORLD_FRAME_CHOICES)}"
            )
        calibration_samples = int(calibration_samples)
        if calibration_samples < 1:
            raise ValueError("calibration_samples must be >= 1.")
        # Initialize the calibration board information.
        board_config = get_calibration_board_config(board_config)
        dictionary, board = create_charuco_board(board_config)
        # Get the intrinsic information from the realsense camera
        intrinsics = self.realsense.get_intrinsics()
        color_dist_coeffs = [
            _dist_coeffs_from_metadata(metadata, "color_distortion_coeffs")
            for metadata in self.stream_metadata
        ]
        color_dist_models = [
            metadata.get("color_distortion_model")
            for metadata in self.stream_metadata
        ]
        error_threshold = 0.3
        min_charuco_corners = max(
            11,
            min(35, int(0.4 * board_config.chessboard_corner_count)),
        )
        print(
            "[Calibrate] Board profile: "
            f"{board_config.name} "
            f"({board_config.squares_x}x{board_config.squares_y}, "
            f"square={board_config.square_length_mm:.1f}mm, "
            f"marker={board_config.marker_length_mm:.1f}mm, "
            f"dictionary={board_config.dictionary_name}, "
            f"min_corners={min_charuco_corners}, "
            f"error_threshold={error_threshold:.3f}, "
            f"world_frame={world_frame_convention}, "
            f"samples={calibration_samples})"
        )
        print(
            "[Calibrate] Color distortion models: "
            + ", ".join([str(model) for model in color_dist_models])
        )
        if board_config.deprecated:
            print(
                "[Calibrate] WARNING: this calibration board profile is deprecated. "
                "Use the Calib.io 12x9 default for new calibrations."
            )

        attempt_idx = 0
        accepted_samples: list[dict[str, Any]] = []
        while len(accepted_samples) < calibration_samples:
            attempt_idx += 1
            obs = self.get_observation()
            colors = [obs[i]["color"] for i in range(self.num_cam)]
            print(f"[Calibrate] Attempt {attempt_idx}")

            c2ws = []
            per_camera_errors = []
            per_camera_corner_counts = []
            sample_failed = False
            for i in range(self.num_cam):
                intrinsic = intrinsics[i]
                dist_coeffs = color_dist_coeffs[i] if i < len(color_dist_coeffs) else None
                calibration_img = colors[i]
                serial = (
                    self.serial_numbers[i]
                    if i < len(self.serial_numbers)
                    else f"cam{i}"
                )
                cam_tag = f"[Cam {i} | {serial}]"
                # cv2.imshow("cablibration", calibration_img)
                # cv2.waitKey(0)

                corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
                    image=calibration_img,
                    dictionary=dictionary,
                    parameters=None,
                )
                if ids is None or len(corners) == 0:
                    sample_failed = True
                    print(
                        f"{cam_tag} No ArUco markers detected. "
                        "Please adjust the board and try again."
                    )
                    break
                retval, charuco_corners, charuco_ids = (
                    cv2.aruco.interpolateCornersCharuco(
                        markerCorners=corners,
                        markerIds=ids,
                        image=calibration_img,
                        board=board,
                        cameraMatrix=intrinsic,
                        distCoeffs=dist_coeffs,
                    )
                )
                if (
                    charuco_corners is None
                    or charuco_ids is None
                    or len(charuco_corners) == 0
                ):
                    sample_failed = True
                    print(
                        f"{cam_tag} No ChArUco corners detected. "
                        "Please adjust the board and try again."
                    )
                    break
                # cv2.imshow("cablibration", calibration_img)

                print(f"{cam_tag} Number of corners: {len(charuco_corners)}")

                rvec = None
                tvec = None
                retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    charuco_corners,
                    charuco_ids,
                    board,
                    intrinsic,
                    dist_coeffs,
                    rvec=rvec,
                    tvec=tvec,
                )
                if (not retval) or (rvec is None) or (tvec is None):
                    sample_failed = True
                    print("Failed to estimate ChArUco pose. Please try again.")
                    break

                # Reproject the points to calculate the error
                charuco_id_values = charuco_ids.reshape(-1)
                reprojected_points, _ = cv2.projectPoints(
                    get_charuco_chessboard_corners(board)[charuco_id_values, :],
                    rvec,
                    tvec,
                    intrinsic,
                    dist_coeffs,
                )
                # Reshape for easier handling
                reprojected_points = reprojected_points.reshape(-1, 2)
                charuco_corners = charuco_corners.reshape(-1, 2)
                per_camera_corner_counts.append(int(len(charuco_corners)))
                # Calculate the error
                error = np.sqrt(
                    np.sum((reprojected_points - charuco_corners) ** 2, axis=1)
                ).mean()
                per_camera_errors.append(float(error))

                print(f"{cam_tag} Reprojection Error: {error:.6f}")
                if (
                    error > error_threshold
                    or len(charuco_corners) < min_charuco_corners
                ):
                    sample_failed = True
                    print(
                        f"{cam_tag} Reprojection check failed "
                        f"(error={error:.6f}, corners={len(charuco_corners)}). "
                        "Please try again."
                    )
                    break
                R_board2cam = cv2.Rodrigues(rvec)[0]
                R_board2cam = _apply_calibration_world_frame(
                    R_board2cam,
                    world_frame_convention,
                )
                t_board2cam = tvec[:, 0]
                w2c = np.eye(4)
                w2c[:3, :3] = R_board2cam
                w2c[:3, 3] = t_board2cam
                c2ws.append(np.linalg.inv(w2c))

                if visualize:
                    calibration_vis = calibration_img.copy()
                    cv2.aruco.drawDetectedMarkers(calibration_vis, corners, ids)
                    cv2.aruco.drawDetectedCornersCharuco(
                        image=calibration_vis,
                        charucoCorners=charuco_corners.reshape(-1, 1, 2),
                        charucoIds=charuco_id_values.reshape(-1, 1),
                    )
                    draw_rvec, _ = cv2.Rodrigues(R_board2cam)
                    cv2.drawFrameAxes(
                        calibration_vis,
                        intrinsic,
                        dist_coeffs,
                        draw_rvec,
                        tvec,
                        0.1,
                    )
                    cv2.imshow("cablibration", calibration_vis)
                    cv2.waitKey(1)

            if (not sample_failed) and len(per_camera_errors) == self.num_cam:
                errors_np = np.asarray(per_camera_errors, dtype=np.float64)
                accepted_samples.append(
                    {
                        "c2ws": c2ws,
                        "per_camera_errors": per_camera_errors,
                        "per_camera_corner_counts": per_camera_corner_counts,
                        "mean_error": float(errors_np.mean()),
                    }
                )
                print(
                    f"[Calibrate] Accepted sample {len(accepted_samples)}/"
                    f"{calibration_samples}: "
                    + ", ".join([f"{e:.6f}" for e in errors_np.tolist()])
                )
                print(
                    f"[Calibrate] Error summary: mean={errors_np.mean():.6f}, "
                    f"max={errors_np.max():.6f}"
                )

        sample_mean_errors = [
            float(sample["mean_error"])
            for sample in accepted_samples
        ]
        selected_sample_index = int(np.argmin(sample_mean_errors))
        selected_sample = accepted_samples[selected_sample_index]
        c2ws = selected_sample["c2ws"]
        per_camera_errors = selected_sample["per_camera_errors"]
        per_camera_corner_counts = selected_sample["per_camera_corner_counts"]
        pose_stability = _compute_pose_stability(
            [sample["c2ws"] for sample in accepted_samples]
        )
        print(
            f"[Calibrate] Selected sample {selected_sample_index} "
            f"with mean reprojection error {sample_mean_errors[selected_sample_index]:.6f}"
        )

        calibrate_path = Path("calibrate.pkl")
        with calibrate_path.open("wb") as f:
            pickle.dump(c2ws, f)
        sidecar_path = write_calibration_metadata(
            calibrate_path,
            build_calibration_metadata(
                serial_numbers=self.serial_numbers,
                WH=self.WH,
                fps=self.fps,
                transform_count=len(c2ws),
                per_camera_reprojection_error=per_camera_errors,
                calibration_board=charuco_board_config_to_metadata(board_config),
                world_frame_convention=world_frame_convention,
                distortion_used=all(item is not None for item in color_dist_coeffs),
                distortion_model_by_camera=color_dist_models,
                distortion_coeffs_by_camera=[
                    _dist_coeffs_to_metadata(item) for item in color_dist_coeffs
                ],
                per_camera_corner_count=per_camera_corner_counts,
                per_camera_pose_stability=pose_stability,
                calibration_samples_requested=calibration_samples,
                calibration_samples_used=len(accepted_samples),
                selected_sample_index=selected_sample_index,
                sample_mean_reprojection_error=sample_mean_errors,
            ),
        )
        print(f"[Calibrate] Wrote {calibrate_path} and {sidecar_path}")

        if self.listener is not None:
            self.listener.stop()
        self.realsense.stop()
