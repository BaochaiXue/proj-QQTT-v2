from __future__ import annotations

import math

import cv2
import numpy as np


def overlay_panel_label(
    image: np.ndarray,
    *,
    label: str,
    text_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    canvas = np.asarray(image, dtype=np.uint8).copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, 32), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        label,
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        text_color,
        2,
        cv2.LINE_AA,
    )
    return canvas


def compose_grid_2x3(
    *,
    title: str,
    column_headers: list[str],
    row_headers: list[str],
    native_images: list[np.ndarray],
    ffs_images: list[np.ndarray],
) -> np.ndarray:
    if len(native_images) != 3 or len(ffs_images) != 3:
        raise ValueError("grid_2x3 layout requires exactly 3 native images and 3 ffs images.")
    if len(column_headers) != 3 or len(row_headers) != 2:
        raise ValueError("grid_2x3 layout requires 3 column headers and 2 row headers.")

    panel_h, panel_w = native_images[0].shape[:2]
    row_label_w = 170
    title_h = 42
    header_h = 38
    body = np.zeros((panel_h * 2, row_label_w + panel_w * 3, 3), dtype=np.uint8)

    for row_idx, row_images in enumerate((native_images, ffs_images)):
        y0 = row_idx * panel_h
        body[y0:y0 + panel_h, :row_label_w] = (16, 16, 16)
        header = row_headers[row_idx]
        text_size = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0]
        text_x = max(10, (row_label_w - text_size[0]) // 2)
        text_y = y0 + (panel_h + text_size[1]) // 2
        cv2.putText(body, header, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        for col_idx, image in enumerate(row_images):
            x0 = row_label_w + col_idx * panel_w
            body[y0:y0 + panel_h, x0:x0 + panel_w] = image

    header_bar = np.zeros((header_h, body.shape[1], 3), dtype=np.uint8)
    header_bar[:, :row_label_w] = (24, 24, 24)
    for col_idx, header in enumerate(column_headers):
        x0 = row_label_w + col_idx * panel_w
        header_bar[:, x0:x0 + panel_w] = (24, 24, 24)
        text_size = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
        text_x = x0 + max(8, (panel_w - text_size[0]) // 2)
        cv2.putText(header_bar, header, (text_x, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    title_bar = np.zeros((title_h, body.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([title_bar, header_bar, body])


def draw_text_box(
    image: np.ndarray,
    *,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float = 0.58,
    thickness: int = 2,
) -> None:
    text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x0 = int(origin[0])
    y0 = int(origin[1])
    top_left = (max(0, x0 - 4), max(0, y0 - text_size[1] - 6))
    bottom_right = (min(image.shape[1] - 1, x0 + text_size[0] + 4), min(image.shape[0] - 1, y0 + baseline + 4))
    cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1)
    cv2.putText(image, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def overlay_scalar_colorbar(
    image: np.ndarray,
    *,
    label: str,
    min_text: str,
    max_text: str,
    colormap: int,
) -> np.ndarray:
    canvas = np.asarray(image, dtype=np.uint8).copy()
    if canvas.shape[0] < 48 or canvas.shape[1] < 64:
        return canvas
    bar_h = max(36, min(180, canvas.shape[0] - 60))
    bar_w = max(10, min(20, canvas.shape[1] // 12))
    x0 = max(6, canvas.shape[1] - bar_w - 8)
    y0 = max(24, min(44, canvas.shape[0] - bar_h - 20))
    bg_left = max(0, x0 - 44)
    bg_top = max(0, y0 - 18)
    bg_right = min(canvas.shape[1] - 1, x0 + bar_w + 10)
    bg_bottom = min(canvas.shape[0] - 1, y0 + bar_h + 34)
    cv2.rectangle(canvas, (bg_left, bg_top), (bg_right, bg_bottom), (10, 10, 10), -1)
    gradient = np.linspace(255, 0, bar_h, dtype=np.uint8).reshape(-1, 1)
    gradient = np.repeat(gradient, bar_w, axis=1)
    bar = cv2.applyColorMap(gradient, colormap)
    canvas[y0:y0 + bar_h, x0:x0 + bar_w] = bar
    cv2.rectangle(canvas, (x0, y0), (x0 + bar_w, y0 + bar_h), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, label, (x0 - 38, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(canvas, max_text, (x0 - 38, y0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(canvas, min_text, (x0 - 38, y0 + bar_h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
    return canvas


def fit_image_to_canvas(
    image: np.ndarray,
    *,
    canvas_size: tuple[int, int],
    background_bgr: tuple[int, int, int] = (18, 18, 22),
) -> np.ndarray:
    target_w, target_h = [max(1, int(item)) for item in canvas_size]
    source = np.asarray(image, dtype=np.uint8)
    src_h, src_w = source.shape[:2]
    if src_h <= 0 or src_w <= 0:
        return np.full((target_h, target_w, 3), background_bgr, dtype=np.uint8)
    scale = min(float(target_w) / float(src_w), float(target_h) / float(src_h))
    fit_w = max(1, int(round(src_w * scale)))
    fit_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(source, (fit_w, fit_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), background_bgr, dtype=np.uint8)
    x0 = (target_w - fit_w) // 2
    y0 = (target_h - fit_h) // 2
    canvas[y0:y0 + fit_h, x0:x0 + fit_w] = resized
    return canvas


def overlay_large_panel_label(
    image: np.ndarray,
    *,
    label: str,
    accent_bgr: tuple[int, int, int],
) -> np.ndarray:
    canvas = np.asarray(image, dtype=np.uint8).copy()
    strip_h = 48
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, strip_h), (18, 18, 20), -1)
    cv2.rectangle(canvas, (0, 0), (18, strip_h), accent_bgr, -1)
    cv2.putText(canvas, label, (30, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.92, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def compose_hero_compare(
    *,
    title_lines: list[str],
    native_image: np.ndarray,
    ffs_image: np.ndarray,
    overview_inset: np.ndarray | None = None,
    warning_text: str | None = None,
    background_bgr: tuple[int, int, int] = (14, 16, 20),
) -> np.ndarray:
    gap = 18
    padding = 18
    native_labeled = overlay_large_panel_label(native_image, label="Native", accent_bgr=(80, 180, 255))
    ffs_labeled = overlay_large_panel_label(ffs_image, label="FFS", accent_bgr=(120, 220, 120))
    panel_h, panel_w = native_labeled.shape[:2]
    body_w = padding * 2 + panel_w * 2 + gap
    body_h = padding * 2 + panel_h

    inset_w = 0
    inset_h = 0
    inset = None
    if overview_inset is not None:
        inset = fit_image_to_canvas(np.asarray(overview_inset, dtype=np.uint8), canvas_size=(260, 170), background_bgr=background_bgr)
        inset_h, inset_w = inset.shape[:2]
    title_h = max(92, inset_h + 20)
    canvas = np.full((title_h + body_h, body_w, 3), background_bgr, dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (body_w - 1, title_h - 1), (10, 12, 16), -1)
    cv2.putText(
        canvas,
        title_lines[0] if title_lines else "",
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.96,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if len(title_lines) > 1:
        cv2.putText(
            canvas,
            title_lines[1],
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (220, 224, 230),
            1,
            cv2.LINE_AA,
        )
    if warning_text:
        text_size, baseline = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 1)
        box_w = text_size[0] + 20
        box_h = text_size[1] + baseline + 14
        box_x = max(20, body_w - box_w - 20 - inset_w)
        box_y = 18
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (58, 56, 118), -1)
        cv2.putText(
            canvas,
            warning_text,
            (box_x + 10, box_y + box_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if inset is not None:
        inset_x = body_w - inset_w - 18
        inset_y = max(10, (title_h - inset_h) // 2)
        canvas[inset_y:inset_y + inset_h, inset_x:inset_x + inset_w] = inset
        cv2.rectangle(canvas, (inset_x - 1, inset_y - 1), (inset_x + inset_w, inset_y + inset_h), (235, 235, 235), 1, cv2.LINE_AA)

    body_y = title_h + padding
    native_x = padding
    ffs_x = padding + panel_w + gap
    canvas[body_y:body_y + panel_h, native_x:native_x + panel_w] = native_labeled
    canvas[body_y:body_y + panel_h, ffs_x:ffs_x + panel_w] = ffs_labeled
    cv2.rectangle(canvas, (native_x - 1, body_y - 1), (native_x + panel_w, body_y + panel_h), (50, 54, 62), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (ffs_x - 1, body_y - 1), (ffs_x + panel_w, body_y + panel_h), (50, 54, 62), 1, cv2.LINE_AA)
    return canvas


def compose_side_by_side_large(
    *,
    title_lines: list[str],
    native_image: np.ndarray,
    ffs_image: np.ndarray,
    overview_inset: np.ndarray,
    warning_text: str | None = None,
) -> np.ndarray:
    native_labeled = overlay_large_panel_label(native_image, label="Native", accent_bgr=(80, 180, 255))
    ffs_labeled = overlay_large_panel_label(ffs_image, label="FFS", accent_bgr=(120, 220, 120))
    main_body = np.hstack([native_labeled, ffs_labeled])

    title_h = 86
    title_bar = np.zeros((title_h, main_body.shape[1], 3), dtype=np.uint8)
    title_bar[:] = (12, 14, 18)
    for line_idx, line in enumerate(title_lines[:2]):
        y = 28 + line_idx * 26
        cv2.putText(
            title_bar,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.76 if line_idx == 0 else 0.62,
            (255, 255, 255),
            2 if line_idx == 0 else 1,
            cv2.LINE_AA,
        )
    if warning_text:
        cv2.rectangle(title_bar, (title_bar.shape[1] - 430, 14), (title_bar.shape[1] - 16, 44), (48, 48, 120), -1)
        cv2.putText(title_bar, warning_text, (title_bar.shape[1] - 418, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    inset = np.asarray(overview_inset, dtype=np.uint8)
    overview_h, overview_w = inset.shape[:2]
    footer_h = max(overview_h + 32, 520)
    footer = np.zeros((footer_h, main_body.shape[1], 3), dtype=np.uint8)
    footer[:] = (16, 18, 22)
    cv2.putText(footer, "Overview", (22, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.02, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(footer, "Real camera frusta, ROI crop, supported arc, and current orbit camera", (22, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(footer, "The orbit path is identical for Native, FFS, and support render.", (22, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(footer, "Camera labels show the original calibrated viewpoints used to define the supported viewing arc.", (22, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (220, 220, 220), 2, cv2.LINE_AA)
    max_overview_w = max(260, footer.shape[1] - 420)
    max_overview_h = footer_h - 36
    scale = min(1.0, float(max_overview_w) / max(1, overview_w), float(max_overview_h) / max(1, overview_h))
    if scale < 1.0:
        overview_w = max(160, int(round(overview_w * scale)))
        overview_h = max(120, int(round(overview_h * scale)))
        inset = cv2.resize(inset, (overview_w, overview_h), interpolation=cv2.INTER_AREA)
    x0 = footer.shape[1] - overview_w - 28
    y0 = max(18, (footer_h - overview_h) // 2)
    footer[y0:y0 + overview_h, x0:x0 + overview_w] = inset
    cv2.rectangle(footer, (x0 - 1, y0 - 1), (x0 + overview_w, y0 + overview_h), (255, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([title_bar, main_body, footer])


def compose_turntable_board(
    *,
    title_lines: list[str],
    column_headers: list[str],
    row_headers: list[str],
    native_images: list[np.ndarray],
    ffs_images: list[np.ndarray],
    overview_inset: np.ndarray | None = None,
) -> np.ndarray:
    if len(native_images) != len(ffs_images):
        raise ValueError("native_images and ffs_images must have the same length.")
    if len(row_headers) != 2:
        raise ValueError("turntable board requires exactly 2 row headers.")
    if len(column_headers) != len(native_images):
        raise ValueError("column_headers must match image column count.")

    panel_h, panel_w = native_images[0].shape[:2]
    num_cols = len(native_images)
    row_label_w = 170
    header_h = 40
    body = np.zeros((panel_h * 2, row_label_w + panel_w * num_cols, 3), dtype=np.uint8)

    for row_idx, row_images in enumerate((native_images, ffs_images)):
        y0 = row_idx * panel_h
        body[y0:y0 + panel_h, :row_label_w] = (20, 20, 20)
        header = row_headers[row_idx]
        text_size = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, 0.92, 2)[0]
        text_x = max(10, (row_label_w - text_size[0]) // 2)
        text_y = y0 + (panel_h + text_size[1]) // 2
        cv2.putText(body, header, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.92, (255, 255, 255), 2, cv2.LINE_AA)
        for col_idx, image in enumerate(row_images):
            x0 = row_label_w + col_idx * panel_w
            body[y0:y0 + panel_h, x0:x0 + panel_w] = image

    header_bar = np.zeros((header_h, body.shape[1], 3), dtype=np.uint8)
    header_bar[:, :row_label_w] = (26, 26, 26)
    for col_idx, header in enumerate(column_headers):
        x0 = row_label_w + col_idx * panel_w
        header_bar[:, x0:x0 + panel_w] = (26, 26, 26)
        text_size = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, 0.80, 2)[0]
        text_x = x0 + max(8, (panel_w - text_size[0]) // 2)
        cv2.putText(header_bar, header, (text_x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 255, 255), 2, cv2.LINE_AA)

    inset_h = 0 if overview_inset is None else int(overview_inset.shape[0])
    title_h = max(74, inset_h + 16)
    title_bar = np.zeros((title_h, body.shape[1], 3), dtype=np.uint8)
    for line_idx, line in enumerate(title_lines[:2]):
        y = 26 + line_idx * 24
        cv2.putText(title_bar, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68 if line_idx == 0 else 0.58, (255, 255, 255), 2 if line_idx == 0 else 1, cv2.LINE_AA)

    if overview_inset is not None:
        inset = np.asarray(overview_inset, dtype=np.uint8)
        inset_h, inset_w = inset.shape[:2]
        x0 = title_bar.shape[1] - inset_w - 12
        y0 = max(8, (title_bar.shape[0] - inset_h) // 2)
        title_bar[y0:y0 + inset_h, x0:x0 + inset_w] = inset
        cv2.rectangle(title_bar, (x0 - 1, y0 - 1), (x0 + inset_w, y0 + inset_h), (255, 255, 255), 1, cv2.LINE_AA)

    return np.vstack([title_bar, header_bar, body])


def compose_keyframe_sheet(
    boards: list[np.ndarray],
    *,
    max_width: int = 4600,
    max_height: int = 4200,
    padding: int = 18,
) -> np.ndarray:
    if not boards:
        raise ValueError("compose_keyframe_sheet requires at least one board.")
    board_h, board_w = boards[0].shape[:2]
    cols = 1 if len(boards) == 1 else 2
    rows = int(math.ceil(len(boards) / cols))
    scale = min(
        1.0,
        float(max_width - padding * (cols + 1)) / max(1.0, cols * board_w),
        float(max_height - padding * (rows + 1)) / max(1.0, rows * board_h),
    )
    tile_w = max(320, int(round(board_w * scale)))
    tile_h = max(240, int(round(board_h * scale)))
    canvas = np.zeros((padding * (rows + 1) + tile_h * rows, padding * (cols + 1) + tile_w * cols, 3), dtype=np.uint8)
    canvas[:] = (10, 10, 10)
    for idx, board in enumerate(boards):
        row = idx // cols
        col = idx % cols
        x0 = padding + col * (tile_w + padding)
        y0 = padding + row * (tile_h + padding)
        resized = cv2.resize(board, (tile_w, tile_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
        canvas[y0:y0 + tile_h, x0:x0 + tile_w] = resized
    return canvas


def compose_depth_review_board(
    *,
    title_lines: list[str],
    metric_lines: list[str],
    rows: list[list[np.ndarray]],
    background_bgr: tuple[int, int, int] = (16, 18, 22),
    padding: int = 16,
    row_gap: int = 16,
    col_gap: int = 16,
) -> np.ndarray:
    if not rows or not rows[0]:
        raise ValueError("compose_depth_review_board requires at least one image row.")

    max_cols = max(len(row) for row in rows)
    col_widths = [0] * max_cols
    row_heights = [0] * len(rows)
    for row_idx, row in enumerate(rows):
        for col_idx, image in enumerate(row):
            row_heights[row_idx] = max(row_heights[row_idx], int(image.shape[0]))
            col_widths[col_idx] = max(col_widths[col_idx], int(image.shape[1]))

    body_w = padding * 2 + sum(col_widths) + col_gap * max(0, max_cols - 1)
    body_h = padding * 2 + sum(row_heights) + row_gap * max(0, len(rows) - 1)
    title_h = 84
    metrics_h = 52 if metric_lines else 0
    canvas = np.full((title_h + metrics_h + body_h, body_w, 3), background_bgr, dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (body_w - 1, title_h - 1), (10, 12, 16), -1)
    cv2.rectangle(canvas, (0, title_h), (body_w - 1, title_h + metrics_h - 1), (20, 24, 30), -1)
    for line_idx, line in enumerate(title_lines[:2]):
        cv2.putText(
            canvas,
            line,
            (18, 30 + line_idx * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78 if line_idx == 0 else 0.56,
            (255, 255, 255),
            2 if line_idx == 0 else 1,
            cv2.LINE_AA,
        )
    for metric_idx, line in enumerate(metric_lines[:2]):
        cv2.putText(
            canvas,
            line,
            (18, title_h + 20 + metric_idx * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (225, 225, 225),
            1,
            cv2.LINE_AA,
        )

    y = title_h + metrics_h + padding
    for row_idx, row in enumerate(rows):
        x = padding
        for col_idx, image in enumerate(row):
            col_w = col_widths[col_idx]
            row_h = row_heights[row_idx]
            fitted = fit_image_to_canvas(image, canvas_size=(col_w, row_h), background_bgr=background_bgr)
            canvas[y:y + row_h, x:x + col_w] = fitted
            cv2.rectangle(canvas, (x - 1, y - 1), (x + col_w, y + row_h), (52, 56, 64), 1, cv2.LINE_AA)
            x += col_w + col_gap
        y += row_heights[row_idx] + row_gap

    return canvas
