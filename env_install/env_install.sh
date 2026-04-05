#!/usr/bin/env bash
set -euo pipefail

# Camera-only installation for preview, calibration, recording, and alignment.
# Assumes a Python 3.10 conda environment is already active.

conda install -y numpy==1.26.4 opencv ffmpeg
pip install pyrealsense2 atomics pynput threadpoolctl

echo "Camera-only environment installation complete."
