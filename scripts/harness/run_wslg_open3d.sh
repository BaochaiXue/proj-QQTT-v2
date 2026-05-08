#!/usr/bin/env bash
set -euo pipefail

# WSLg/Open3D GUI repair shim.
#
# Open3D's GUI/Filament path needs a working OpenGL context. On WSLg, Mesa can
# accidentally try Zink/Vulkan and fail before Open3D creates a window. Keep this
# wrapper small and explicit so normal non-WSL runs are unaffected.

unset VK_ICD_FILENAMES
unset __GLX_VENDOR_LIBRARY_NAME
unset __EGL_VENDOR_LIBRARY_FILENAMES
unset LD_PRELOAD

# XWayland is more stable than Wayland/EGL for this Open3D GUI path on WSLg.
# On this rig Open3D/Filament creates a window with WAYLAND_DISPLAY empty, while
# fully unsetting it can still hit the EGL driver-selection crash.
export WAYLAND_DISPLAY=""
export EGL_PLATFORM="${EGL_PLATFORM:-x11}"

# WSLg hardware OpenGL path: Mesa Gallium d3d12.
export GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}"
export MESA_LOADER_DRIVER_OVERRIDE="${MESA_LOADER_DRIVER_OVERRIDE:-d3d12}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-0}"

# Open3D/Filament can hang or crash during Python process teardown on this WSLg
# path. The realtime harness observes this and exits after stopping its camera
# pipeline instead of letting Open3D destructors run.
export QQTT_WSLG_OPEN3D_FAST_EXIT="${QQTT_WSLG_OPEN3D_FAST_EXIT:-1}"

# Override to Intel/AMD or unset in the caller if needed.
export MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-NVIDIA}"

exec "$@"
