"""Synthetic fixtures shared by the tests. No lab data is used anywhere in the suite."""

from __future__ import annotations

import pathlib

import numpy as np


class DiscVideo:
    """A bright disc moving along the diagonal, one position per frame, on a dark background."""

    def __init__(self, n_frames=8, size=512, radius=30, start=80, step=50):
        self.n_frames, self.size, self.radius = n_frames, size, radius
        self.centers = [(start + step * k, start + step * k) for k in range(n_frames)]

    def frame(self, k: int) -> np.ndarray:
        import cv2

        img = np.full((self.size, self.size, 3), 40, np.uint8)
        cv2.circle(img, self.centers[k], self.radius, (235, 235, 235), -1)
        return img

    def frames(self):
        return [self.frame(k) for k in range(self.n_frames)]

    def disc_mask(self, k: int) -> np.ndarray:
        yy, xx = np.mgrid[0 : self.size, 0 : self.size]
        cx, cy = self.centers[k]
        return (xx - cx) ** 2 + (yy - cy) ** 2 <= self.radius**2

    def box(self, k: int, pad: int = 8):
        cx, cy = self.centers[k]
        r = self.radius + pad
        return (cx - r, cy - r, cx + r, cy + r)

    def closest_frame(self, image: np.ndarray) -> int:
        """Which synthetic frame an (RGB) image matches best, by mean absolute difference."""
        diffs = [np.abs(image.astype(np.int16) - self.frame(k).astype(np.int16)).mean() for k in range(self.n_frames)]
        return int(np.argmin(diffs))

    def write(self, path: pathlib.Path, repeat_pattern=None) -> pathlib.Path:
        """Write frames as mp4; ``repeat_pattern`` lists frame indices to emit (duplicates allowed)."""
        import cv2

        order = list(range(self.n_frames)) if repeat_pattern is None else list(repeat_pattern)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (self.size, self.size))
        assert writer.isOpened(), "OpenCV cannot write mp4v video"
        for k in order:
            writer.write(self.frame(k))  # grayscale content, so BGR/RGB order does not matter
        writer.release()
        return path


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def read_video(path) -> list:
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames
