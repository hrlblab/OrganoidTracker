"""Shared fixtures. Tests never touch lab data; everything here is synthetic."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TINY_CHECKPOINT = ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"


def pytest_configure(config):
    config.addinivalue_line("markers", "model: needs the SAM 2.1 tiny checkpoint and runs the real model")
    config.addinivalue_line("markers", "gui: needs a display for Tk")


@pytest.fixture(scope="session")
def tiny_checkpoint() -> pathlib.Path:
    if not TINY_CHECKPOINT.exists():
        pytest.skip("checkpoints/sam2.1_hiera_tiny.pt is missing; run `bash checkpoints/download_ckpts.sh`")
    return TINY_CHECKPOINT


@pytest.fixture(scope="session")
def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class DiscVideo:
    """A synthetic video of a bright disc moving along the diagonal, one position per frame."""

    def __init__(self, n_frames=8, size=256, radius=18, start=40, step=24):
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

    def write(self, path: pathlib.Path, repeat_pattern=None) -> pathlib.Path:
        """Write the frames as mp4. ``repeat_pattern`` lists frame indices to emit (allows duplicates)."""
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


@pytest.fixture(scope="session")
def disc() -> DiscVideo:
    return DiscVideo()


@pytest.fixture(scope="session")
def disc_video_path(disc, tmp_path_factory) -> pathlib.Path:
    return disc.write(tmp_path_factory.mktemp("video") / "disc.mp4")
