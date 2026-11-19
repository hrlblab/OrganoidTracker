"""One identity for a cyst wherever its mask is shown: a colour by cyst id and a ``C<id>`` label.

The window's result view, the exported overlay videos and the report's key image all use this
module, so that a row of ``cyst_summary.csv`` can be matched to an object in a picture. Nothing
here measures anything; colours and labels are presentation only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

# The palette of the exported videos since the first release (RGB); ids above 20 repeat it.
_PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 165, 0),
    (128, 0, 128),
    (255, 192, 203),
    (165, 42, 42),
    (144, 238, 144),
    (135, 206, 235),
    (221, 160, 221),
    (240, 230, 140),
    (255, 99, 71),
    (64, 224, 208),
    (238, 130, 238),
    (255, 182, 193),
    (152, 251, 152),
    (245, 222, 179),
)
BACKGROUND_COLOR = (128, 128, 128)


def cyst_color(identifier: int) -> tuple[int, int, int]:
    """The RGB colour of a cyst id; id 0 and negative ids are the background grey."""
    identifier = int(identifier)
    if identifier < 1:
        return BACKGROUND_COLOR
    return _PALETTE[(identifier - 1) % len(_PALETTE)]


def color_table(identifiers: Iterable[int]) -> dict[int, tuple[int, int, int]]:
    """``{id: colour}`` for the given ids plus the background entry 0, as the video writer expects."""
    table: dict[int, tuple[int, int, int]] = {0: BACKGROUND_COLOR}
    for identifier in identifiers:
        table.setdefault(identifier, cyst_color(identifier))
    return table


def cyst_label(identifier: int) -> str:
    return f"C{int(identifier)}"


def mask_box(mask: Any) -> tuple[int, int, int, int] | None:
    """``(x, y, width, height)`` of a mask's bounding box in its own pixels; ``None`` for an empty mask."""
    pixels = np.asarray(mask)
    if pixels.ndim > 2:
        pixels = pixels.squeeze()
    rows = np.flatnonzero(pixels.any(axis=1))
    if rows.size == 0:
        return None
    # Only the occupied rows are scanned for columns: a full second pass is slow on 4096-pixel frames.
    columns = np.flatnonzero(pixels[rows[0] : rows[-1] + 1].any(axis=0))
    return int(columns[0]), int(rows[0]), int(columns[-1] - columns[0] + 1), int(rows[-1] - rows[0] + 1)


def text_scale(width: int) -> float:
    """An OpenCV font scale that stays legible from 128-pixel test frames to 4096-pixel lab frames."""
    return max(0.45, min(4.0, width / 1300))


def draw_text(image: np.ndarray, text: str, x: int, y: int, scale: float, rgb: tuple[int, int, int]) -> None:
    """Text with a dark halo, its baseline at ``(x, y)``, drawn into an RGB uint8 image in place.

    The halo is the dilated text itself, so it fits whatever glyphs the OpenCV build draws; only
    the text's own rectangle is touched, which keeps 4096-pixel frames cheap.
    """
    import cv2

    thickness = max(1, round(scale * 2))
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    margin = thickness + 3
    height, width = image.shape[:2]
    left, top = max(0, x - margin), max(0, y - text_height - margin)
    right, bottom = min(width, x + text_width + margin), min(height, y + baseline + margin)
    if right <= left or bottom <= top:
        return
    layer = np.zeros((bottom - top, right - left), dtype=np.uint8)
    cv2.putText(layer, text, (x - left, y - top), font, scale, 255, thickness, cv2.LINE_AA)
    size = 2 * max(1, thickness) + 1
    halo = cv2.dilate(layer, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    region = image[top:bottom, left:right].astype(np.float32)
    region *= 1 - (halo.astype(np.float32) / 255)[..., None] * 0.85
    ink = (layer.astype(np.float32) / 255)[..., None]
    blended = region * (1 - ink) + np.array(rgb, dtype=np.float32) * ink
    image[top:bottom, left:right] = blended.clip(0, 255).astype(np.uint8)


def draw_cyst_labels(image: np.ndarray, boxes: Mapping[int, tuple[int, int, int, int]]) -> np.ndarray:
    """Write ``C<id>`` at the top-left corner of each mask's box (above it where there is room).

    ``boxes`` are in the image's own pixels. The image is modified in place and returned.
    """
    import cv2

    height, width = image.shape[:2]
    scale = text_scale(width)
    thickness = max(1, round(scale * 2))
    for identifier in sorted(boxes):
        x, y, _box_width, _box_height = boxes[identifier]
        text = cyst_label(identifier)
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        left = max(1, min(x, width - text_width - 2))
        base = y - baseline - 1
        if base - text_height < 1:  # no room above the box: write just inside its top edge
            base = min(height - baseline - 1, y + text_height + 2)
        draw_text(image, text, left, base, scale, (255, 255, 255))
    return image


def draw_frame_caption(image: np.ndarray, text: str) -> np.ndarray:
    """Outlined ``text`` in the top-left corner, without hiding the picture behind a strip.

    ``text`` may carry a second part after ``" | "``; it is dropped where the frame is too narrow.
    The image is modified in place and returned.
    """
    import cv2

    width = image.shape[1]
    scale = text_scale(width)
    thickness = max(1, round(scale * 2))
    font = cv2.FONT_HERSHEY_SIMPLEX
    if cv2.getTextSize(text, font, scale, thickness)[0][0] > width - 8:
        text = text.split(" | ")[0]
    text_height = cv2.getTextSize(text, font, scale, thickness)[0][1]
    draw_text(image, text, 4, text_height + 5, scale, (255, 255, 255))
    return image


def render_key_image(
    frame: np.ndarray | None,
    masks: Mapping[int, Any],
    *,
    caption: str | None = None,
    alpha: float = 0.4,
    max_width: int = 1600,
) -> np.ndarray:
    """One labelled overlay of ``masks`` (``{cyst id: mask}``) on ``frame``, for the report's key.

    Without a frame (a report exported while the source video is unavailable) the masks are
    drawn on black. The result is RGB uint8, at most ``max_width`` pixels wide.
    """
    import cv2

    arrays = {int(identifier): np.asarray(mask).squeeze().astype(bool) for identifier, mask in masks.items()}
    if frame is None:
        if not arrays:
            raise ValueError("a key image needs a frame or at least one mask")
        shape = next(iter(arrays.values())).shape
        image = np.zeros((shape[0], shape[1], 3), dtype=np.float32)
    else:
        image = np.asarray(frame, dtype=np.float32).copy()
    height, width = image.shape[:2]
    boxes: dict[int, tuple[int, int, int, int]] = {}
    for identifier in sorted(arrays):
        mask = arrays[identifier]
        if mask.shape != (height, width):
            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        box = mask_box(mask)
        if box is None:
            continue
        boxes[identifier] = box
        image[mask] = image[mask] * (1 - alpha) + np.array(cyst_color(identifier), dtype=np.float32) * alpha
    output = np.ascontiguousarray(image.clip(0, 255).astype(np.uint8))
    if width > max_width:
        factor = max_width / width
        resized = cv2.resize(output, (max_width, max(1, round(height * factor))), interpolation=cv2.INTER_AREA)
        output = np.ascontiguousarray(resized, dtype=np.uint8)
        boxes = {
            identifier: (round(x * factor), round(y * factor), round(w * factor), round(h * factor))
            for identifier, (x, y, w, h) in boxes.items()
        }
    draw_cyst_labels(output, boxes)
    if caption:
        draw_frame_caption(output, caption)
    return output
