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
