"""Shared chronological frame selection, independent of a model or GUI toolkit."""

import cv2
import numpy as np


def unique_frame_indices(frames: list[np.ndarray], threshold: float) -> list[int]:
    """Keep frames differing from their decoded predecessor by the grayscale MAD threshold."""
    if not frames:
        return []
    keep = [0]
    previous = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY).astype(np.int16)
    for index in range(1, len(frames)):
        current = cv2.cvtColor(frames[index], cv2.COLOR_RGB2GRAY).astype(np.int16)
        if float(np.abs(current - previous).mean()) >= threshold:
            keep.append(index)
        previous = current
    return keep


def selected_frame_indices(count: int, indices=None, expected_count=None) -> list[int]:
    """Validate an acquisition mapping; image similarity never defines time points."""
    if expected_count is not None and count != expected_count:
        raise ValueError(
            f"Confirmed selection expects {expected_count} decoded frames, found {count}; confirm the source again"
        )
    if indices is None:
        return list(range(count))
    values = list(indices)
    if not values or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < count for i in values):
        raise ValueError(f"Selected source frame indices must be integers between 0 and {count - 1}")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("Selected source frame indices must increase strictly")
    return values
