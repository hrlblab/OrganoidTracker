"""Acquisition observations are selected explicitly, never inferred from small pixel changes."""

from dataclasses import replace

import numpy as np
import pytest

from organoidtracker.services.pipeline import run_session
from organoidtracker.services.session import SessionError, VideoReference
from test_services import make_session
from test_services import small_disc as small_disc
from test_services import small_video as small_video


def test_low_change_observations_are_not_discarded(tmp_path, monkeypatch):
    from organoidtracker.core import video_decode
    from organoidtracker.core.sam2_tracker import SAM2Tracker

    frames = [np.zeros((64, 64, 3), np.uint8) for _ in range(3)]
    frames[1][8:10, 8:10] = 200
    frames[2][8:11, 8:11] = 200
    monkeypatch.setattr(video_decode, "decode_frames", lambda *a: (frames, 10.0, 3))
    path = tmp_path / "fixture.mp4"
    path.write_bytes(b"decoder fixture")

    class Predictor:
        def init_state_from_frames(self, frames, **kwargs):
            return {"frame_count": len(frames)}

    tracker = SAM2Tracker(device="cpu")
    tracker.is_loaded, tracker.predictor = True, Predictor()
    info = tracker.load_video(str(path))
    assert info["frame_map"] == [0, 1, 2]
    assert info["annotation_frame_index"] == 2


def test_unconfirmed_session_is_rejected_before_output(small_disc, small_video, tmp_path):
    session = make_session(small_disc, small_video)
    session = replace(session, video=VideoReference(session.video.path, session.video.sha256))
    with pytest.raises(SessionError, match="[Cc]onfirm.*frame|frame.*confirm"):
        run_session(session, tmp_path / "unconfirmed", videos=False)
    assert not (tmp_path / "unconfirmed").exists()


@pytest.mark.parametrize(
    "indices,count", [([], 8), ([0, 0], 8), ([1, 0], 8), ([8], 8), ([True], 8), ([0.5], 8), ([0], None)]
)
def test_invalid_selection_is_rejected(small_video, indices, count):
    with pytest.raises(SessionError):
        VideoReference(small_video, frame_indices=indices, decoded_frames=count)


def test_selection_round_trips_and_rejects_changed_source_count(small_disc, small_video, tmp_path):
    from organoidtracker.services.saved_results import load_saved_result
    from organoidtracker.services.session import Timing, load_session, session_from_document
    from test_services import run

    session = make_session(small_disc, small_video)
    session = replace(
        session, video=replace(session.video, frame_indices=(0, 2, 4, 7)), timing=Timing(frame_times_days=(1, 2, 4, 7))
    )
    assert session_from_document(session.to_document()).video == session.video
    result = run(session, tmp_path / "run", small_disc)
    saved = load_saved_result(result.output_dir)
    assert saved.video.frame_map == (0, 2, 4, 7)
    assert saved.session.video == session.video
    assert load_session(result.output_dir / "prompts.json").video == session.video
    assert saved.session.timing.frame_times_days == (1, 2, 4, 7)
    wrong = replace(session, video=replace(session.video, decoded_frames=9))
    with pytest.raises(ValueError, match="expects 9 decoded frames"):
        run(wrong, tmp_path / "wrong", small_disc)
