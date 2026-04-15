from .realsense import MultiRealsense, SingleRealsense
from .recording_metadata import build_recording_metadata as build_recording_metadata_payload
from .defaults import (
    DEFAULT_EXPOSURE,
    DEFAULT_FPS,
    DEFAULT_GAIN,
    DEFAULT_HEIGHT,
    DEFAULT_NUM_CAM,
    DEFAULT_WHITE_BALANCE,
    DEFAULT_WIDTH,
)
from multiprocessing.managers import SharedMemoryManager
import numpy as np
import time
import cv2
import json
import os
import pickle
from typing import Optional, Any

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

DEFAULT_EXPOSURE_OVERRIDES = {
    "239222303506": 156,
    "239222300781": 156,
}

CAPTURE_MODE_CONFIGS = {
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


def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


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
        enable_keyboard_listener=True,
    ):
        self.WH = WH
        self.fps = fps
        self.listener: Optional[Any] = None
        self._keyboard = None

        if capture_mode not in CAPTURE_MODE_CONFIGS:
            raise ValueError(f"Unsupported capture_mode: {capture_mode}")
        if emitter not in {"on", "off", "auto"}:
            raise ValueError(f"Unsupported emitter: {emitter}")

        connected_serials = SingleRealsense.get_connected_devices_serial()
        self.connected_serial_numbers = list(connected_serials)
        if serial_numbers is not None:
            missing = [serial for serial in serial_numbers if serial not in connected_serials]
            if missing:
                raise AssertionError(f"Requested serials not connected: {missing}")
            self.serial_numbers = list(serial_numbers)
        else:
            if len(connected_serials) < num_cam:
                raise AssertionError(f"Only {len(connected_serials)} cameras are connected.")
            self.serial_numbers = connected_serials[:num_cam]
        self.num_cam = len(self.serial_numbers)
        self.capture_mode = capture_mode
        self.capture_config = CAPTURE_MODE_CONFIGS[capture_mode]
        self.streams_present = list(self.capture_config["streams_present"])
        self.emitter = emitter
        self.exposure = exposure
        self.gain = gain
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
        if isinstance(exposure, (int, float)):
            exposure_values = [
                float(exposure_overrides.get(sn, exposure))
                for sn in self.serial_numbers
            ]
        else:
            exposure_values = exposure
        self.realsense.set_exposure(exposure=exposure_values, gain=gain)
        self.realsense.set_white_balance(white_balance)

        self.realsense.start()
        time.sleep(3)
        self.stream_metadata = self.realsense.get_stream_metadata()
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

    def get_observation(self):
        # Used to get the latest observations from all cameras
        data = self._get_sync_frame()
        # TODO: Process the data when needed
        return data

    def _get_sync_frame(self, k=4):
        assert self.realsense.is_ready

        # Get the latest k frames from all cameras, and picked the latest synchronized frames
        last_realsense_data = self.realsense.get(k=k)
        timestamp_list = [x["timestamp"][-1] for x in last_realsense_data.values()]
        last_timestamp = np.min(timestamp_list)

        data = {}
        for camera_idx, value in last_realsense_data.items():
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
            calibration_reference_serials=self.connected_serial_numbers,
            capture_mode=self.capture_mode,
            streams_present=self.streams_present,
            fps=self.fps,
            WH=self.WH,
            emitter_request=self.emitter,
            stream_metadata=self.stream_metadata,
        )

    def calibrate(self, visualize=True):
        # Initialize the calibration board information
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        board = cv2.aruco.CharucoBoard(
            (4, 5),
            squareLength=0.05,
            markerLength=0.037,
            dictionary=dictionary,
        )
        # Get the intrinsic information from the realsense camera
        intrinsics = self.realsense.get_intrinsics()
        error_threshold = 0.19
        min_charuco_corners = 11

        flag = True
        attempt_idx = 0
        while flag:
            attempt_idx += 1
            flag = False
            obs = self.get_observation()
            colors = [obs[i]["color"] for i in range(self.num_cam)]
            print(f"[Calibrate] Attempt {attempt_idx}")

            c2ws = []
            per_camera_errors = []
            for i in range(self.num_cam):
                intrinsic = intrinsics[i]
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
                    flag = True
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
                    )
                )
                if (
                    charuco_corners is None
                    or charuco_ids is None
                    or len(charuco_corners) == 0
                ):
                    flag = True
                    print(
                        f"{cam_tag} No ChArUco corners detected. "
                        "Please adjust the board and try again."
                    )
                    break
                # cv2.imshow("cablibration", calibration_img)

                print(f"{cam_tag} Number of corners: {len(charuco_corners)}")
                if visualize:
                    cv2.aruco.drawDetectedCornersCharuco(
                        image=calibration_img,
                        charucoCorners=charuco_corners,
                        charucoIds=charuco_ids,
                    )
                    cv2.imshow("cablibration", calibration_img)
                    cv2.waitKey(1)

                rvec = None
                tvec = None
                retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    charuco_corners,
                    charuco_ids,
                    board,
                    intrinsic,
                    None,
                    rvec=rvec,
                    tvec=tvec,
                )
                if (not retval) or (rvec is None) or (tvec is None):
                    flag = True
                    print("Failed to estimate ChArUco pose. Please try again.")
                    break

                # Reproject the points to calculate the error
                reprojected_points, _ = cv2.projectPoints(
                    board.getChessboardCorners()[charuco_ids, :],
                    rvec,
                    tvec,
                    intrinsic,
                    None,
                )
                # Reshape for easier handling
                reprojected_points = reprojected_points.reshape(-1, 2)
                charuco_corners = charuco_corners.reshape(-1, 2)
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
                    flag = True
                    print(
                        f"{cam_tag} Reprojection check failed "
                        f"(error={error:.6f}, corners={len(charuco_corners)}). "
                        "Please try again."
                    )
                    break
                R_board2cam = cv2.Rodrigues(rvec)[0]
                t_board2cam = tvec[:, 0]
                w2c = np.eye(4)
                w2c[:3, :3] = R_board2cam
                w2c[:3, 3] = t_board2cam
                c2ws.append(np.linalg.inv(w2c))

            if (not flag) and len(per_camera_errors) == self.num_cam:
                errors_np = np.asarray(per_camera_errors, dtype=np.float64)
                print(
                    "[Calibrate] Per-camera reprojection errors accepted: "
                    + ", ".join([f"{e:.6f}" for e in errors_np.tolist()])
                )
                print(
                    f"[Calibrate] Error summary: mean={errors_np.mean():.6f}, "
                    f"max={errors_np.max():.6f}"
                )

        with open("calibrate.pkl", "wb") as f:
            pickle.dump(c2ws, f)

        if self.listener is not None:
            self.listener.stop()
        self.realsense.stop()
