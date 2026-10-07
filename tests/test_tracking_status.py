"""A failure in the middle of propagation must be reported, not hidden behind a success message."""

import numpy as np
import pytest
import torch

from src.core.sam2_tracker import SAM2Tracker


class FakePredictor:
    def __init__(self, fail_at=None):
        self.fail_at = fail_at

    def propagate_in_video(self, inference_state, start_frame_idx=None, reverse=False):
        order = range(start_frame_idx, -1, -1) if reverse else range(start_frame_idx, 3)
        for frame_idx in order:
            if frame_idx == self.fail_at:
                raise RuntimeError("synthetic failure")
            logits = torch.full((1, 1, 32, 32), -5.0)
            logits[0, 0, 8:24, 8:24] = 5.0
            yield frame_idx, [1], logits


def make_offline_tracker(monkeypatch, fail_at):
    import config

    monkeypatch.setattr(config, "SAM2_IMPROVED_TRACKING", False)
    tracker = SAM2Tracker(model_config="sam2_hiera_t", checkpoint_path="/nonexistent.pt", device="cpu")
    tracker.predictor = FakePredictor(fail_at=fail_at)
    tracker.inference_state = {"obj_id_to_idx": {}, "output_dict_per_obj": {}}
    tracker.video_frames = [np.zeros((32, 32, 3), np.uint8) for _ in range(3)]
    tracker.frame_map = [0, 1, 2]
    tracker.annotation_frame_index = 2
    tracker.prompts = {1: [{"type": "bbox", "frame_idx": 2, "x1": 8, "y1": 8, "x2": 24, "y2": 24}]}
    return tracker


def test_complete_run(monkeypatch):
    tracker = make_offline_tracker(monkeypatch, fail_at=None)
    result = tracker.run_tracking()
    assert result.status == "completed" and result.frames_done == 3 and sorted(result) == [0, 1, 2]
    assert result[0][1].area == 16 * 16


def test_mid_run_failure_is_reported_as_partial(monkeypatch):
    tracker = make_offline_tracker(monkeypatch, fail_at=0)
    result = tracker.run_tracking()
    assert result.status == "partial" and result.frames_done == 2
    assert sorted(result) == [1, 2]
    assert "synthetic failure" in result.error
    assert "partial" in result.summary()


def test_failure_before_any_frame_raises(monkeypatch):
    tracker = make_offline_tracker(monkeypatch, fail_at=2)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        tracker.run_tracking()
