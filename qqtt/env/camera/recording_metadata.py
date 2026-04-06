from __future__ import annotations


def build_recording_metadata(
    *,
    serial_numbers,
    calibration_reference_serials,
    capture_mode,
    streams_present,
    fps,
    WH,
    emitter_request,
    stream_metadata,
):
    k_color = [m.get("K_color") for m in stream_metadata]
    depth_scales = [m.get("depth_scale_m_per_unit") for m in stream_metadata]
    has_any_depth_scale = any(scale is not None for scale in depth_scales)
    return {
        "schema_version": "qqtt_recording_v2",
        "serial_numbers": list(serial_numbers),
        "calibration_reference_serials": list(calibration_reference_serials),
        "logical_camera_names": [f"cam{i}" for i in range(len(serial_numbers))],
        "capture_mode": capture_mode,
        "streams_present": list(streams_present),
        "camera_model_per_camera": [m.get("model_name") for m in stream_metadata],
        "product_line_per_camera": [m.get("product_line") for m in stream_metadata],
        "fps": fps,
        "WH": list(WH),
        "intrinsics": k_color,
        "K_color": k_color,
        "K_ir_left": [m.get("K_ir_left") for m in stream_metadata],
        "K_ir_right": [m.get("K_ir_right") for m in stream_metadata],
        "T_ir_left_to_right": [m.get("T_ir_left_to_right") for m in stream_metadata],
        "T_ir_left_to_color": [m.get("T_ir_left_to_color") for m in stream_metadata],
        "ir_baseline_m": [m.get("ir_baseline_m") for m in stream_metadata],
        "depth_scale_m_per_unit": depth_scales,
        "depth_encoding": "uint16_meters_scaled_invalid_zero" if has_any_depth_scale else None,
        "alignment_target": "color" if "color" in streams_present else None,
        "depth_coordinate_frame": "color" if "depth" in streams_present else None,
        "emitter_request": emitter_request,
        "emitter_actual": [m.get("emitter_actual") for m in stream_metadata],
        "exposure_per_camera": [m.get("exposure") for m in stream_metadata],
        "gain_per_camera": [m.get("gain") for m in stream_metadata],
        "white_balance_per_camera": [m.get("white_balance") for m in stream_metadata],
        "recording": {},
    }
