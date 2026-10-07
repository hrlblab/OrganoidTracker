"""The frame contract, checked against the real SAM 2.1 tiny model on a synthetic video."""

import pytest

from helpers import iou

pytestmark = pytest.mark.model


def make_tracker(checkpoint, device, reverse):
    from organoidtracker.core.sam2_tracker import SAM2Tracker

    tracker = SAM2Tracker(model_config="sam2_hiera_t", checkpoint_path=str(checkpoint), device=device,
                          enable_reverse_tracking=reverse)
    assert tracker.load_model()
    return tracker


@pytest.fixture(scope="module")
def reverse_tracker(tiny_checkpoint, device):
    return make_tracker(tiny_checkpoint, device, reverse=True)


def test_reverse_mode_prompts_the_last_frame_and_tracks_backwards(reverse_tracker, disc, disc_video_path):
    tracker = reverse_tracker
    info = tracker.load_video(str(disc_video_path))
    n = disc.n_frames
    assert info["num_frames"] == n and info["duplicate_frames_removed"] == 0
    assert info["direction"] == "reverse" and info["annotation_frame_index"] == n - 1

    # the frame shown for annotation is the last chronological frame
    assert disc.closest_frame(tracker.get_first_frame()) == n - 1

    # a box drawn on the displayed frame lands on chronological frame n-1 inside the predictor
    assert tracker.add_bbox_prompt(*disc.box(n - 1), obj_id=1)
    assert sorted(tracker.inference_state["point_inputs_per_obj"][0].keys()) == [n - 1]
    assert tracker.prompts[1][0]["frame_idx"] == n - 1

    result = tracker.run_tracking()
    assert result.status == "completed" and result.frames_done == n
    assert sorted(result.keys()) == list(range(n))
    assert result.annotation_frame == n - 1 and result.direction == "reverse"
    for k in range(n):
        assert iou(result[k][1].numpy(), disc.disc_mask(k)) > 0.8, f"frame {k} mask does not match its own disc"
        assert 1 in result.presence[k]
    # masks are packed on the CPU, not float tensors on the GPU
    assert type(result[0][1]).__name__ == "PackedMask"


def test_forward_mode_control(tiny_checkpoint, device, disc, disc_video_path):
    tracker = make_tracker(tiny_checkpoint, device, reverse=False)
    info = tracker.load_video(str(disc_video_path))
    assert info["direction"] == "forward" and info["annotation_frame_index"] == 0
    assert disc.closest_frame(tracker.get_first_frame()) == 0
    assert tracker.add_bbox_prompt(*disc.box(0), obj_id=1)
    result = tracker.run_tracking()
    assert sorted(result.keys()) == list(range(disc.n_frames)) and result.status == "completed"
    for k in range(disc.n_frames):
        assert iou(result[k][1].numpy(), disc.disc_mask(k)) > 0.8


def test_duplicated_frames_collapse_to_unique_time_points(reverse_tracker, disc, tmp_path):
    pattern = [0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7]  # 12 decoded frames, 8 unique
    path = disc.write(tmp_path / "dup.mp4", repeat_pattern=pattern)
    info = reverse_tracker.load_video(str(path))
    assert info["decoded_frames"] == 12 and info["num_frames"] == 8 and info["duplicate_frames_removed"] == 4
    assert info["frame_map"] == [0, 2, 3, 5, 6, 8, 9, 11]
    assert disc.closest_frame(reverse_tracker.get_first_frame()) == 7

    assert reverse_tracker.add_bbox_prompt(*disc.box(7), obj_id=1)
    result = reverse_tracker.run_tracking()
    assert sorted(result.keys()) == list(range(8)) and result.frame_map == [0, 2, 3, 5, 6, 8, 9, 11]
    for k in range(8):
        assert iou(result[k][1].numpy(), disc.disc_mask(k)) > 0.8


def test_clearing_one_object_keeps_the_other(reverse_tracker, disc, disc_video_path):
    tracker = reverse_tracker
    tracker.load_video(str(disc_video_path))
    n = disc.n_frames
    assert tracker.add_bbox_prompt(*disc.box(n - 1), obj_id=1)
    # a second object: a box on the dark background corner
    assert tracker.add_bbox_prompt(10, 400, 80, 480, obj_id=2)
    assert tracker.get_prompt_count() == 2
    assert tracker.clear_prompts(obj_id=2)
    assert tracker.get_prompt_count() == 1 and 2 not in tracker.prompts
    assert sorted(tracker.inference_state["obj_ids"]) == [1]
    result = tracker.run_tracking()
    assert result.object_ids() == [1]
    assert iou(result[n - 1][1].numpy(), disc.disc_mask(n - 1)) > 0.8
