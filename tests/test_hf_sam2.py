"""The optional adapter on real weights and synthetic input; never downloads in tests."""

import numpy as np
import pytest

pytest.importorskip("transformers", reason="install the hf extra for the optional backend tests")

from helpers import DiscVideo, iou  # noqa: E402
from organoidtracker.core.hf_sam2_tracker import HfSam2Adapter  # noqa: E402
from organoidtracker.paths import checkpoints_dir  # noqa: E402
from organoidtracker.services.run_manifest import mask_digests  # noqa: E402


def test_missing_local_snapshot_is_refused_without_download(tmp_path, caplog):
    tracker = HfSam2Adapter(checkpoint_path=tmp_path / "absent", model_config="sam2_hiera_t", device="cpu")
    assert not tracker.load_model()
    assert "expected a local Hugging Face" in caplog.text


@pytest.fixture(scope="module")
def hf_tracker(device):
    snapshot = checkpoints_dir() / "hf" / "sam2.1-hiera-tiny"
    if not (snapshot / "model.safetensors").is_file():
        pytest.skip("local Hugging Face SAM 2.1 tiny snapshot is missing")
    tracker = HfSam2Adapter(checkpoint_path=str(snapshot), model_config="sam2_hiera_t", device=device)
    assert tracker.load_model()
    return tracker


@pytest.mark.model
def test_reverse_multi_object_prompts_cancel_and_repeat(hf_tracker, disc, disc_video_path):
    tracker = hf_tracker
    tracker.enable_reverse_tracking = True
    info = tracker.load_video(str(disc_video_path))
    assert info["annotation_frame_index"] == disc.n_frames - 1
    assert tracker.add_bbox_prompt(*disc.box(disc.n_frames - 1), obj_id=7)
    assert tracker.add_bbox_prompt(10, 400, 80, 480, obj_id=19)
    full = tracker.run_tracking()
    assert full.status == "completed" and full.tracked_frames == list(reversed(range(disc.n_frames)))
    assert set(full.presence[disc.n_frames - 1]) == {7, 19}
    for frame in range(disc.n_frames):
        assert iou(full[frame][7].numpy(), disc.disc_mask(frame)) > 0.8
    stopped = []
    cancelled = tracker.run_tracking(lambda *args: stopped.append(args[0]), lambda: len(stopped) == 2)
    assert cancelled.is_cancelled and cancelled.frames_done == 2
    again = tracker.run_tracking()
    assert mask_digests(again) == mask_digests(full)
    assert tracker.clear_prompts(19)
    assert tracker.get_active_objects() == [7]
    one = tracker.run_tracking()
    assert one.object_ids() == [7]
    assert tracker.provenance()["backend"] == "sam2-transformers"
    assert len(tracker.provenance()["snapshot_files"]["model.safetensors"]) == 64


@pytest.mark.model
def test_forward_control_and_single_frame_boundary(hf_tracker, disc, disc_video_path, tmp_path):
    tracker = hf_tracker
    tracker.enable_reverse_tracking = False
    info = tracker.load_video(str(disc_video_path))
    assert info["annotation_frame_index"] == 0
    assert tracker.add_bbox_prompt(*disc.box(0), obj_id=1)
    result = tracker.run_tracking()
    assert result.tracked_frames == list(range(disc.n_frames))
    for frame in range(disc.n_frames):
        assert iou(result[frame][1].numpy(), disc.disc_mask(frame)) > 0.8
    single = DiscVideo(n_frames=1)
    video = single.write(tmp_path / "single.mp4")
    tracker.enable_reverse_tracking = True
    tracker.load_video(str(video))
    tracker.add_bbox_prompt(*single.box(0), obj_id=1)
    result = tracker.run_tracking()
    assert result.status == "completed" and result.tracked_frames == [0]
    assert np.array_equal(result[0][1].numpy(), tracker.run_tracking()[0][1].numpy())
