"""Shared fixtures. Tests never touch lab data; everything here is synthetic."""

from __future__ import annotations

import pathlib

import pytest
from helpers import DiscVideo

from organoidtracker.paths import checkpoints_dir

TINY_CHECKPOINT = checkpoints_dir() / "sam2.1_hiera_tiny.pt"


@pytest.fixture(scope="session")
def tiny_checkpoint() -> pathlib.Path:
    if not TINY_CHECKPOINT.exists():
        pytest.skip(f"{TINY_CHECKPOINT} is missing; run `bash checkpoints/download_ckpts.sh`")
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
