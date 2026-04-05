from .realsense import MultiRealsense, SingleRealsense
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


def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


class CameraSystem:
    def __init__(
        self,
        WH=(DEFAULT_WIDTH, DEFAULT_HEIGHT),
        fps=DEFAULT_FPS,
        num_cam=DEFAULT_NUM_CAM,
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

        self.serial_numbers = SingleRealsense.get_connected_devices_serial()
        self.num_cam = len(self.serial_numbers)
        assert self.num_cam == num_cam, f"Only {self.num_cam} cameras are connected."

        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()

        self.realsense = MultiRealsense(
            serial_numbers=self.serial_numbers,
            shm_manager=self.shm_manager,
            resolution=(self.WH[0], self.WH[1]),
            capture_fps=self.fps,
            enable_color=True,
            enable_depth=True,
            process_depth=True,
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
            data[camera_idx]["color"] = value["color"][best_idx]
            data[camera_idx]["depth"] = value["depth"][best_idx]
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

    def record(self, output_path):
        exist_dir(output_path)
        exist_dir(f"{output_path}/color")
        exist_dir(f"{output_path}/depth")

        metadata = {}
        intrinsics = self.realsense.get_intrinsics()
        metadata["intrinsics"] = intrinsics.tolist()
        metadata["serial_numbers"] = self.serial_numbers
        metadata["fps"] = self.fps
        metadata["WH"] = self.WH
        metadata["recording"] = {}
        for i in range(self.num_cam):
            metadata["recording"][i] = {}
            exist_dir(f"{output_path}/color/{i}")
            exist_dir(f"{output_path}/depth/{i}")

        # Set the max time for recording
        last_step_idxs = [-1] * self.num_cam
        while not self.end:
            if self.recording:
                last_realsense_data = self.realsense.get()
                timestamps = [
                    last_realsense_data[i]["timestamp"].item()
                    for i in range(self.num_cam)
                ]
                step_idxs = [
                    last_realsense_data[i]["step_idx"].item()
                    for i in range(self.num_cam)
                ]

                if not all(
                    [step_idxs[i] == last_step_idxs[i] for i in range(self.num_cam)]
                ):
                    for i in range(self.num_cam):
                        if last_step_idxs[i] != step_idxs[i]:
                            # Record the the step for this camera
                            time_stamp = timestamps[i]
                            step_idx = step_idxs[i]
                            color = last_realsense_data[i]["color"]
                            depth = last_realsense_data[i]["depth"]

                            metadata["recording"][i][step_idx] = time_stamp
                            cv2.imwrite(
                                f"{output_path}/color/{i}/{step_idx}.png", color
                            )
                            np.save(f"{output_path}/depth/{i}/{step_idx}.npy", depth)

        print("End recording")
        if self.listener is not None:
            self.listener.stop()
        with open(f"{output_path}/metadata.json", "w") as f:
            json.dump(metadata, f)

        self.realsense.stop()

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
