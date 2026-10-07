from src.core.tracking_result import TrackingResult


def test_behaves_like_the_legacy_dict():
    result = TrackingResult({0: {1: "m0"}, 2: {1: "m2", 3: "x"}}, status="completed", frames_total=3, frames_done=3)
    assert dict(result) == {0: {1: "m0"}, 2: {1: "m2", 3: "x"}}
    assert len(result) == 2 and 2 in result and list(result.keys()) == [0, 2]
    assert result.object_ids() == [1, 3]
    assert result.is_complete and not result.is_partial


def test_partial_summary_mentions_error():
    result = TrackingResult({6: {1: "m"}}, status=TrackingResult.PARTIAL, frames_total=7, frames_done=1,
                            error="RuntimeError: boom", direction="reverse", annotation_frame=6, frame_map=[0, 2, 3])
    assert result.is_partial
    assert result.summary() == "partial: 1/7 frames, 1 objects, reverse (RuntimeError: boom)"
    assert result.frame_map == [0, 2, 3] and result.annotation_frame == 6
