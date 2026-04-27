# Description: MultiRealsense class for multiple RealSense cameras, based on code from Diffusion Policy

from typing import List, Optional, Union, Dict, Callable
import numbers
import time
from multiprocessing.managers import SharedMemoryManager
import numpy as np
import pyrealsense2 as rs
from .single_realsense import SingleRealsense


class MultiRealsense:
    def __init__(
        self,
        serial_numbers: Optional[List[str]] = None,
        shm_manager: Optional[SharedMemoryManager] = None,
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
        advanced_mode_config: Optional[Union[dict, List[dict]]] = None,
        transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]] = None,
        vis_transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]] = None,
        verbose=False,
    ):
        if serial_numbers is None:
            serial_numbers = SingleRealsense.get_connected_devices_serial()
        serial_numbers = list(serial_numbers)
        if len(serial_numbers) != len(set(serial_numbers)):
            raise ValueError(f"Duplicate RealSense serial numbers are not supported: {serial_numbers}")
        n_cameras = len(serial_numbers)
        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()

        advanced_mode_config = repeat_to_list(advanced_mode_config, n_cameras, dict)
        transform = repeat_to_list(transform, n_cameras, Callable)
        vis_transform = repeat_to_list(vis_transform, n_cameras, Callable)

        cameras = dict()
        for i, serial in enumerate(serial_numbers):
            cameras[serial] = SingleRealsense(
                shm_manager=shm_manager,
                serial_number=serial,
                resolution=resolution,
                capture_fps=capture_fps,
                put_fps=put_fps,
                put_downsample=put_downsample,
                enable_color=enable_color,
                enable_depth=enable_depth,
                process_depth=process_depth,
                enable_ir_left=enable_ir_left,
                enable_ir_right=enable_ir_right,
                emitter=emitter,
                get_max_k=get_max_k,
                advanced_mode_config=advanced_mode_config[i],
                transform=transform[i],
                vis_transform=vis_transform[i],
                is_master=(i == 0),
                verbose=verbose,
            )

        self.cameras = cameras
        self.serial_numbers = serial_numbers
        self.shm_manager = shm_manager
        self.resolution = resolution
        self.capture_fps = capture_fps

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @property
    def n_cameras(self):
        return len(self.cameras)

    def _cameras_in_logical_order(self):
        for serial_number in self.serial_numbers:
            yield self.cameras[serial_number]

    @property
    def is_ready(self):
        is_ready = True
        for camera in self._cameras_in_logical_order():
            if not camera.is_ready:
                is_ready = False
        return is_ready

    def start(self, wait=True, put_start_time=None):
        if put_start_time is None:
            put_start_time = time.time()
        for camera in self._cameras_in_logical_order():
            camera.start(wait=False, put_start_time=put_start_time)

        if wait:
            self.start_wait()

    def stop(self, wait=True):
        for camera in self._cameras_in_logical_order():
            camera.stop(wait=False)

        if wait:
            self.stop_wait()

    def start_wait(self):
        for camera in self._cameras_in_logical_order():
            print("processing camera {}".format(camera.serial_number))
            camera.start_wait()

    def stop_wait(self):
        for camera in self._cameras_in_logical_order():
            camera.join()

    def get(self, k=None, index=None, out=None) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Return order T,H,W,C
        {
            0: {
                'rgb': (T,H,W,C),
                'timestamp': (T,)
            },
            1: ...
        }
        """
        if index is not None:
            this_out = None
            this_out = self.cameras[self.serial_numbers[index]].get(k=k, out=this_out)
            return this_out
        if out is None:
            out = dict()
        for i, camera in enumerate(self._cameras_in_logical_order()):
            this_out = None
            if i in out:
                this_out = out[i]
            this_out = camera.get(k=k, out=this_out)
            out[i] = this_out
        return out

    def set_color_option(self, option, value):
        n_camera = len(self.cameras)
        value = repeat_to_list(value, n_camera, numbers.Number)
        for i, camera in enumerate(self._cameras_in_logical_order()):
            camera.set_color_option(option, value[i])

    def set_exposure(self, exposure=None, gain=None):
        """150nit. (0.1 ms, 1/10000s)
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
        return np.array([c.get_intrinsics() for c in self._cameras_in_logical_order()])

    def get_depth_scale(self):
        return np.array([c.get_depth_scale() for c in self._cameras_in_logical_order()])

    def get_stream_metadata(self):
        return [c.get_stream_metadata() for c in self._cameras_in_logical_order()]

    def restart_put(self, start_time):
        for camera in self._cameras_in_logical_order():
            camera.restart_put(start_time)


def repeat_to_list(x, n: int, cls):
    if x is None:
        x = [None] * n
    if isinstance(x, cls):
        x = [x] * n
    assert len(x) == n
    return x
