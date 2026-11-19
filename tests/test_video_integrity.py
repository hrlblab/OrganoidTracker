"""Indexed video loss must fail before a decoded prefix becomes the experiment's time axis."""

import av
import numpy as np
import pytest

from organoidtracker.services.session import SessionError
from organoidtracker.services.video_frames import decode_video


@pytest.fixture
def indexed_pair(tmp_path):
    full = tmp_path / "full.mp4"
    with av.open(str(full), "w", options={"movflags": "+faststart"}) as container:
        stream = container.add_stream("mpeg4", rate=8)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
        for index in range(8):
            pixels = np.full((64, 64, 3), 20 + index * 25, np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(full)) as container:
        assert container.streams.video[0].frames == 8
        positions = [packet.pos for packet in container.demux(video=0) if packet.size]
    truncated = tmp_path / "truncated.mp4"
    truncated.write_bytes(full.read_bytes()[: positions[5]])
    return full, truncated


def test_indexed_truncation_is_rejected_by_preview_decoder(indexed_pair):
    full, truncated = indexed_pair
    assert len(decode_video(full)[0]) == 8
    with pytest.raises(SessionError, match="8.*5|5.*8"):
        decode_video(truncated)


def test_indexed_truncation_is_rejected_before_predictor_initialization(indexed_pair):
    from organoidtracker.core.sam2_tracker import SAM2Tracker

    class NoInference:
        def init_state_from_frames(self, *args, **kwargs):
            pytest.fail("invalid video reached inference")

    tracker = SAM2Tracker(device="cpu")
    tracker.is_loaded, tracker.predictor = True, NoInference()
    with pytest.raises(ValueError, match="8.*5|5.*8"):
        tracker.load_video(str(indexed_pair[1]))


def test_unknown_index_does_not_use_duration_estimate(indexed_pair, monkeypatch):
    from organoidtracker.core import video_decode

    monkeypatch.setattr(video_decode, "indexed_frame_count", lambda path: 0)
    assert len(decode_video(indexed_pair[0])[0]) == 8


def test_intentional_prefix_is_not_truncation(indexed_pair):
    from organoidtracker.core.video_decode import decode_frames

    assert len(decode_frames(indexed_pair[0], max_frames=3)[0]) == 3
