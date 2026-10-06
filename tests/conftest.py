"""Shared fixtures. Tests never touch lab data; everything here is synthetic."""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "tests")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from helpers import DiscVideo  # noqa: E402

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


@pytest.fixture(scope="session")
def disc() -> DiscVideo:
    return DiscVideo()


@pytest.fixture(scope="session")
def disc_video_path(disc, tmp_path_factory) -> pathlib.Path:
    return disc.write(tmp_path_factory.mktemp("video") / "disc.mp4")
