"""Output videos must show frame k with the masks of frame k, whatever the tracking direction."""

import numpy as np
import pytest

from helpers import read_video
from organoidtracker.core.masks import PackedMask
from organoidtracker.io.video_output import VideoOutputGenerator

N, H, W, SQ = 6, 64, 96, 16


def square_mask(k):
    mask = np.zeros((H, W), bool)
    x = 4 + 14 * k
    mask[24:24 + SQ, x:x + SQ] = True
    return mask


def square_slice(k):
    x = 4 + 14 * k
    return (slice(24 + 3, 24 + SQ - 3), slice(x + 3, x + SQ - 3))


def frames():
    return [np.full((H, W, 3), 40, np.uint8) for _ in range(N)]


def packed_segments():
    return {k: {1: PackedMask(square_mask(k))} for k in range(N)}


def legacy_logit_segments():
    torch = pytest.importorskip("torch")
    return {k: {1: torch.where(torch.from_numpy(square_mask(k))[None], 4.0, -4.0)} for k in range(N)}


def assert_identity_mapping(video_path):
    out = read_video(video_path)
    assert len(out) == N
    for k in range(N):
        own = out[k][square_slice(k)]
        assert own[..., 0].mean() > 90, f"frame {k}: no red overlay at its own square"
        mirror = N - 1 - k
        if mirror != k:
            other = out[k][square_slice(mirror)]
            assert other[..., 0].mean() < 70, f"frame {k}: overlay found at the mirrored square (reversal bug)"


def test_process_single_mask_accepts_packed_and_logit_masks():
    gen = VideoOutputGenerator()
    mask = square_mask(2)
    assert np.array_equal(gen._process_single_mask(PackedMask(mask), (H, W)), mask.astype(np.uint8))
    torch = pytest.importorskip("torch")
    logits = torch.where(torch.from_numpy(mask)[None], 3.0, -3.0)
    assert np.array_equal(gen._process_single_mask(logits, (H, W)), mask.astype(np.uint8))


def test_optimized_videos_use_chronological_indices(tmp_path):
    gen = VideoOutputGenerator()
    created = gen.create_optimized_multi_object_videos(frames(), packed_segments(), tmp_path, fps=2.0, alpha=0.6, quality_scale=1.0)
    assert set(created) == {"overlay", "mask", "side_by_side"} and all(created.values())
    assert_identity_mapping(created["overlay"])
    mask_frames = read_video(created["mask"])
    for k in range(N):
        assert mask_frames[k][square_slice(k)][..., 0].mean() > 150


def test_single_video_path_uses_chronological_indices(tmp_path):
    gen = VideoOutputGenerator()
    path = gen.create_multi_object_video(frames(), legacy_logit_segments(), tmp_path / "overlay.mp4", fps=2.0, video_type="overlay", alpha=0.6)
    assert path is not None
    assert_identity_mapping(path)


def test_optimized_videos_create_a_missing_output_directory(tmp_path):
    gen = VideoOutputGenerator()
    target = tmp_path / "nested" / "does-not-exist-yet"
    created = gen.create_optimized_multi_object_videos(frames(), packed_segments(), target, fps=2.0, quality_scale=1.0)
    assert all(created.values()) and target.is_dir()
    assert len(read_video(created["overlay"])) == N
