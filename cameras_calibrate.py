from qqtt.env import CameraSystem

if __name__ == "__main__":
    camera_system = CameraSystem(
        WH=[1280, 720],
        fps=5,
        num_cam=3,
        enable_keyboard_listener=False,
    )
    camera_system.calibrate()
