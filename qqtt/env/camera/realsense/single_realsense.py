# Description: MultiRealsense class for multiple RealSense cameras, based on code from Diffusion Policy

from typing import Optional, Callable, Dict
import os
import enum
import time
import json
import numpy as np
import pyrealsense2 as rs
import multiprocessing as mp
import cv2
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager

from .utils import get_accumulate_timestamp_idxs
from .shared_memory.shared_ndarray import SharedNDArray
from .shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from .shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty


class Command(enum.Enum):
    SET_COLOR_OPTION = 0
    SET_DEPTH_OPTION = 1
    START_RECORDING = 2
    STOP_RECORDING = 3
    RESTART_PUT = 4


def intrinsics_to_matrix(intrinsics) -> list[list[float]]:
    return [
        [float(intrinsics.fx), 0.0, float(intrinsics.ppx)],
        [0.0, float(intrinsics.fy), float(intrinsics.ppy)],
        [0.0, 0.0, 1.0],
    ]


def extrinsics_to_matrix(extrinsics) -> list[list[float]]:
    rotation = list(map(float, extrinsics.rotation))
    translation = list(map(float, extrinsics.translation))
    return [
        [rotation[0], rotation[1], rotation[2], translation[0]],
        [rotation[3], rotation[4], rotation[5], translation[1]],
        [rotation[6], rotation[7], rotation[8], translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def translation_norm(extrinsics) -> float:
    tx, ty, tz = map(float, extrinsics.translation)
    return float((tx * tx + ty * ty + tz * tz) ** 0.5)


class SingleRealsense(mp.Process):
    MAX_PATH_LENGTH = 4096  # linux path has a limit of 4096 bytes

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        serial_number,
        resolution=(1280, 720),
        capture_fps=30,
        put_fps=None,
        put_downsample=True,
        enable_color=True,
        enable_depth=False,
        process_depth=False,
        enable_ir_left=False,
        enable_ir_right=False,
        emitter="auto",
        get_max_k=30,
        advanced_mode_config=None,
        transform: Optional[Callable[[Dict], Dict]] = None,
        vis_transform: Optional[Callable[[Dict], Dict]] = None,
        is_master=False,
        verbose=False,
    ):
        super().__init__()

        if put_fps is None:
            put_fps = capture_fps

        # create ring buffer
        resolution = tuple(resolution)
        shape = resolution[::-1]
        examples = dict()
        if enable_color:
            examples["color"] = np.empty(shape=shape + (3,), dtype=np.uint8)
        if enable_depth:
            examples["depth"] = np.empty(shape=shape, dtype=np.uint16)
        if enable_ir_left:
            examples["ir_left"] = np.empty(shape=shape, dtype=np.uint8)
        if enable_ir_right:
            examples["ir_right"] = np.empty(shape=shape, dtype=np.uint8)
        examples["camera_capture_timestamp"] = 0.0
        examples["camera_receive_timestamp"] = 0.0
        examples["timestamp"] = 0.0
        examples["step_idx"] = 0

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if transform is None else transform(dict(examples)),
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=put_fps,
        )

        # create command queue
        examples = {
            "cmd": Command.SET_COLOR_OPTION.value,
            "option_enum": rs.option.exposure.value,
            "option_value": 0.0,
            "put_start_time": 0.0,
        }

        command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager, examples=examples, buffer_size=128
        )

        # create shared array for intrinsics
        intrinsics_array = SharedNDArray.create_from_shape(
            mem_mgr=shm_manager, shape=(7,), dtype=np.float64
        )
        intrinsics_array.get()[:] = 0

        # copied variables
        self.serial_number = serial_number
        self.resolution = resolution
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.enable_ir_left = enable_ir_left
        self.enable_ir_right = enable_ir_right
        self.emitter = emitter
        self.advanced_mode_config = advanced_mode_config
        self.transform = transform
        self.vis_transform = vis_transform
        self.process_depth = process_depth
        self.is_master = is_master
        self.verbose = verbose
        self.put_start_time = None

        # shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        self.command_queue = command_queue
        self.intrinsics_array = intrinsics_array
        self.metadata_queue = mp.Queue(maxsize=1)
        self.metadata_cache = None

    @staticmethod
    def get_connected_devices_serial():
        serials = list()
        for d in rs.context().devices:
            if d.get_info(rs.camera_info.name).lower() != "platform camera":
                serial = d.get_info(rs.camera_info.serial_number)
                product_line = d.get_info(rs.camera_info.product_line)
                if product_line == "D400":
                    # only works with D400 series
                    serials.append(serial)
        serials = sorted(serials)
        return serials

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= user API ===========
    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time
        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self):
        self.ready_event.wait()

    def end_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)

    # ========= user API ===========
    def set_color_option(self, option: rs.option, value: float):
        self.command_queue.put(
            {
                "cmd": Command.SET_COLOR_OPTION.value,
                "option_enum": option.value,
                "option_value": value,
            }
        )

    def set_exposure(self, exposure=None, gain=None):
        """
        exposure: (1, 10000) 100us unit. (0.1 ms, 1/10000s)
        gain: (0, 128)
        """

        if exposure is None and gain is None:
            # auto exposure
            self.set_color_option(rs.option.enable_auto_exposure, 1.0)
        else:
            # manual exposure
            self.set_color_option(rs.option.enable_auto_exposure, 0.0)
            if exposure is not None:
                self.set_color_option(rs.option.exposure, exposure)
            if gain is not None:
                self.set_color_option(rs.option.gain, gain)

    def set_white_balance(self, white_balance=None):
        if white_balance is None:
            self.set_color_option(rs.option.enable_auto_white_balance, 1.0)
        else:
            self.set_color_option(rs.option.enable_auto_white_balance, 0.0)
            self.set_color_option(rs.option.white_balance, white_balance)

    def get_intrinsics(self):
        assert self.ready_event.is_set()
        metadata = self.get_stream_metadata()
        if metadata.get("K_color") is not None:
            return np.asarray(metadata["K_color"], dtype=np.float64)
        fx, fy, ppx, ppy = self.intrinsics_array.get()[:4]
        mat = np.eye(3)
        mat[0, 0] = fx
        mat[1, 1] = fy
        mat[0, 2] = ppx
        mat[1, 2] = ppy
        return mat

    def get_depth_scale(self):
        assert self.ready_event.is_set()
        metadata = self.get_stream_metadata()
        if metadata.get("depth_scale_m_per_unit") is not None:
            return float(metadata["depth_scale_m_per_unit"])
        scale = self.intrinsics_array.get()[-1]
        return scale

    def get_stream_metadata(self):
        assert self.ready_event.is_set()
        if self.metadata_cache is None:
            self.metadata_cache = self.metadata_queue.get()
        return self.metadata_cache

    def depth_process(self, depth_frame):
        depth_to_disparity = rs.disparity_transform(True)
        disparity_to_depth = rs.disparity_transform(False)

        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 5)
        spatial.set_option(rs.option.filter_smooth_alpha, 0.75)
        spatial.set_option(rs.option.filter_smooth_delta, 1)
        spatial.set_option(rs.option.holes_fill, 1)

        temporal = rs.temporal_filter()
        temporal.set_option(rs.option.filter_smooth_alpha, 0.75)
        temporal.set_option(rs.option.filter_smooth_delta, 1)

        filtered_depth = depth_to_disparity.process(depth_frame)
        filtered_depth = spatial.process(filtered_depth)
        filtered_depth = temporal.process(filtered_depth)
        filtered_depth = disparity_to_depth.process(filtered_depth)
        return filtered_depth

    def restart_put(self, start_time):
        self.command_queue.put(
            {"cmd": Command.RESTART_PUT.value, "put_start_time": start_time}
        )

    # ========= interval API ===========
    def run(self):
        # limit threads
        threadpool_limits(1)
        cv2.setNumThreads(1)
        w, h = self.resolution
        fps = self.capture_fps
        align = rs.align(rs.stream.color)
        # Enable the streams from all the intel realsense devices
        rs_config = rs.config()
        if self.enable_color:
            rs_config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        if self.enable_depth:
            rs_config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        if self.enable_ir_left:
            rs_config.enable_stream(rs.stream.infrared, 1, w, h, rs.format.y8, fps)
        if self.enable_ir_right:
            rs_config.enable_stream(rs.stream.infrared, 2, w, h, rs.format.y8, fps)

        def init_device():
            rs_config.enable_device(self.serial_number)

            # start pipeline
            pipeline = rs.pipeline()
            pipeline_profile = pipeline.start(rs_config)
            self.pipeline = pipeline
            self.pipeline_profile = pipeline_profile

            # report global time
            # https://github.com/IntelRealSense/librealsense/pull/3909
            d = self.pipeline_profile.get_device().first_color_sensor()
            d.set_option(rs.option.global_time_enabled, 1)

            # setup advanced mode
            if self.advanced_mode_config is not None:
                json_text = json.dumps(self.advanced_mode_config)
                device = self.pipeline_profile.get_device()
                advanced_mode = rs.rs400_advanced_mode(device)
                advanced_mode.load_json(json_text)

            device = self.pipeline_profile.get_device()
            depth_sensor = device.first_depth_sensor()
            if self.emitter != "auto" and depth_sensor.supports(
                rs.option.emitter_enabled
            ):
                depth_sensor.set_option(
                    rs.option.emitter_enabled, 1.0 if self.emitter == "on" else 0.0
                )

            stream_metadata = {
                "serial": self.serial_number,
                "model_name": device.get_info(rs.camera_info.name),
                "product_line": device.get_info(rs.camera_info.product_line),
                "firmware_version": device.get_info(rs.camera_info.firmware_version),
                "usb_type_descriptor": device.get_info(
                    rs.camera_info.usb_type_descriptor
                ),
                "physical_port": device.get_info(rs.camera_info.physical_port),
                "resolution": [w, h],
                "fps": fps,
                "streams_present": [],
                "K_color": None,
                "K_ir_left": None,
                "K_ir_right": None,
                "T_ir_left_to_right": None,
                "T_ir_left_to_color": None,
                "ir_baseline_m": None,
                "depth_scale_m_per_unit": None,
                "depth_encoding": (
                    "uint16_meters_scaled_invalid_zero" if self.enable_depth else None
                ),
                "alignment_target": (
                    "color" if self.enable_depth and self.enable_color else None
                ),
                "depth_coordinate_frame": (
                    "color" if self.enable_depth and self.enable_color else None
                ),
                "emitter_request": self.emitter,
                "emitter_actual": None,
                "exposure": None,
                "gain": None,
                "white_balance": None,
            }

            if self.enable_color:
                color_stream = self.pipeline_profile.get_stream(
                    rs.stream.color
                ).as_video_stream_profile()
                intr = color_stream.get_intrinsics()
                order = ["fx", "fy", "ppx", "ppy", "height", "width"]
                for i, name in enumerate(order):
                    self.intrinsics_array.get()[i] = getattr(intr, name)
                stream_metadata["K_color"] = intrinsics_to_matrix(intr)
                stream_metadata["streams_present"].append("color")
                color_sensor = device.first_color_sensor()
                try:
                    stream_metadata["exposure"] = float(
                        color_sensor.get_option(rs.option.exposure)
                    )
                except Exception:
                    pass
                try:
                    stream_metadata["gain"] = float(
                        color_sensor.get_option(rs.option.gain)
                    )
                except Exception:
                    pass
                try:
                    stream_metadata["white_balance"] = float(
                        color_sensor.get_option(rs.option.white_balance)
                    )
                except Exception:
                    pass

            depth_scale = depth_sensor.get_depth_scale()
            self.intrinsics_array.get()[-1] = depth_scale
            stream_metadata["depth_scale_m_per_unit"] = float(depth_scale)
            if self.enable_depth:
                stream_metadata["streams_present"].append("depth")

            if depth_sensor.supports(rs.option.emitter_enabled):
                try:
                    stream_metadata["emitter_actual"] = float(
                        depth_sensor.get_option(rs.option.emitter_enabled)
                    )
                except Exception:
                    pass

            ir_left_profile = None
            if self.enable_ir_left:
                ir_left_profile = self.pipeline_profile.get_stream(
                    rs.stream.infrared, 1
                ).as_video_stream_profile()
                stream_metadata["K_ir_left"] = intrinsics_to_matrix(
                    ir_left_profile.get_intrinsics()
                )
                stream_metadata["streams_present"].append("ir_left")
            if self.enable_ir_right:
                ir_right_profile = self.pipeline_profile.get_stream(
                    rs.stream.infrared, 2
                ).as_video_stream_profile()
                stream_metadata["K_ir_right"] = intrinsics_to_matrix(
                    ir_right_profile.get_intrinsics()
                )
                stream_metadata["streams_present"].append("ir_right")
                if ir_left_profile is not None:
                    ir_left_to_right = ir_left_profile.get_extrinsics_to(
                        ir_right_profile
                    )
                    stream_metadata["T_ir_left_to_right"] = extrinsics_to_matrix(
                        ir_left_to_right
                    )
                    stream_metadata["ir_baseline_m"] = translation_norm(
                        ir_left_to_right
                    )
            if ir_left_profile is not None and self.enable_color:
                color_profile = self.pipeline_profile.get_stream(
                    rs.stream.color
                ).as_video_stream_profile()
                ir_left_to_color = ir_left_profile.get_extrinsics_to(color_profile)
                stream_metadata["T_ir_left_to_color"] = extrinsics_to_matrix(
                    ir_left_to_color
                )

            try:
                self.metadata_queue.put_nowait(stream_metadata)
            except Exception:
                pass

            # one-time setup (intrinsics etc, ignore for now)
            if self.verbose:
                print(f"[SingleRealsense {self.serial_number}] Main loop started.")

        try:
            init_device()
            # put frequency regulation
            put_idx = None
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = time.time()

            iter_idx = 0
            t_start = time.time()
            while not self.stop_event.is_set():
                # wait for frames to come in
                frameset = None
                while frameset is None:
                    try:
                        frameset = self.pipeline.wait_for_frames()
                    except RuntimeError as e:
                        print(
                            f"[SingleRealsense {self.serial_number}] Error: {e}. Ready state: {self.ready_event.is_set()}, Restarting device."
                        )
                        device = self.pipeline.get_active_profile().get_device()
                        device.hardware_reset()
                        self.pipeline.stop()
                        init_device()
                        continue
                receive_time = time.time()
                raw_frameset = frameset
                aligned_frameset = (
                    align.process(raw_frameset)
                    if (self.enable_color and self.enable_depth)
                    else raw_frameset
                )

                self.ring_buffer.ready_for_get = receive_time - put_start_time >= 0

                # grab data
                if self.verbose:
                    grad_start_time = time.time()
                data = dict()
                data["camera_receive_timestamp"] = receive_time
                # realsense report in ms
                data["camera_capture_timestamp"] = raw_frameset.get_timestamp() / 1000
                if self.enable_color:
                    # print(time.time())
                    color_frame = aligned_frameset.get_color_frame()
                    data["color"] = np.asarray(color_frame.get_data())
                    t = color_frame.get_timestamp() / 1000
                    data["camera_capture_timestamp"] = t
                    # print('device', time.time() - t)
                    # print(color_frame.get_frame_timestamp_domain())
                if self.enable_depth:
                    depth_frame = aligned_frameset.get_depth_frame()
                    if self.process_depth:
                        data["depth"] = self.depth_process(depth_frame).get_data()
                    else:
                        data["depth"] = np.asarray(depth_frame.get_data())
                if self.enable_ir_left:
                    data["ir_left"] = np.asarray(
                        raw_frameset.get_infrared_frame(1).get_data()
                    )
                if self.enable_ir_right:
                    data["ir_right"] = np.asarray(
                        raw_frameset.get_infrared_frame(2).get_data()
                    )
                if self.verbose:
                    print(
                        f"[SingleRealsense {self.serial_number}] Grab data time {time.time() - grad_start_time}"
                    )

                # apply transform
                if self.verbose:
                    transform_start_time = time.time()
                put_data = data
                if self.transform is not None:
                    put_data = self.transform(dict(data))
                if self.verbose:
                    print(
                        f"[SingleRealsense {self.serial_number}] Transform time {time.time() - transform_start_time}"
                    )

                if self.verbose:
                    put_data_start_time = time.time()
                if self.put_downsample:
                    # put frequency regulation
                    # print(self.serial_number, put_start_time, put_idx, len(global_idxs))
                    local_idxs, global_idxs, put_idx = get_accumulate_timestamp_idxs(
                        timestamps=[receive_time],
                        start_time=put_start_time,
                        dt=1 / self.put_fps,
                        # this is non in first iteration
                        # and then replaced with a concrete number
                        next_global_idx=put_idx,
                        # continue to pump frames even if not started.
                        # start_time is simply used to align timestamps.
                        allow_negative=True,
                    )
                    for step_idx in global_idxs:
                        put_data["step_idx"] = step_idx
                        # put_data['timestamp'] = put_start_time + step_idx / self.put_fps
                        put_data["timestamp"] = receive_time
                        # print(step_idx, data['timestamp'])
                        self.ring_buffer.put(
                            put_data, wait=False, serial_number=self.serial_number
                        )
                else:
                    step_idx = int((receive_time - put_start_time) * self.put_fps)
                    print(step_idx, receive_time)
                    put_data["step_idx"] = step_idx
                    put_data["timestamp"] = receive_time
                    self.ring_buffer.put(
                        put_data, wait=False, serial_number=self.serial_number
                    )
                if self.verbose:
                    print(
                        f"[SingleRealsense {self.serial_number}] Put data time {time.time() - put_data_start_time}",
                        end=" ",
                    )
                    print(
                        f"with downsample for {len(global_idxs)}x"
                        if self.put_downsample and len(global_idxs) > 1
                        else ""
                    )

                # signal ready
                if iter_idx == 0:
                    self.ready_event.set()

                # perf
                t_end = time.time()
                duration = t_end - t_start
                frequency = np.round(1 / duration, 1)
                t_start = t_end
                if self.verbose:
                    print(f"[SingleRealsense {self.serial_number}] FPS {frequency}")

                # fetch command from queue
                try:
                    commands = self.command_queue.get_all()
                    n_cmd = len(commands["cmd"])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command["cmd"]
                    if cmd == Command.SET_COLOR_OPTION.value:
                        sensor = self.pipeline_profile.get_device().first_color_sensor()
                        option = rs.option(command["option_enum"])
                        value = float(command["option_value"])
                        sensor.set_option(option, value)
                        # print('auto', sensor.get_option(rs.option.enable_auto_exposure))
                        # print('exposure', sensor.get_option(rs.option.exposure))
                        # print('gain', sensor.get_option(rs.option.gain))
                    elif cmd == Command.SET_DEPTH_OPTION.value:
                        sensor = self.pipeline_profile.get_device().first_depth_sensor()
                        option = rs.option(command["option_enum"])
                        value = float(command["option_value"])
                        sensor.set_option(option, value)
                    elif cmd == Command.RESTART_PUT.value:
                        put_idx = None
                        put_start_time = command["put_start_time"]

                iter_idx += 1
        finally:
            rs_config.disable_all_streams()
            self.ready_event.set()

        if self.verbose:
            print(f"[SingleRealsense {self.serial_number}] Exiting worker process.")
