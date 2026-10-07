"""The headless path through the services with a fake backend: complete and interrupted runs, tracked
frames without accepted masks, missing observations, mixed organoid populations, non-daily and explicit
timestamps, CSV round trips. Everything the exports say must come from what was tracked."""

import csv
import hashlib
import json
from pathlib import Path

import pytest

from helpers import DiscVideo, FakeTracker, expected_growth_per_day
from organoidtracker.analysis.csv_import import experiment_from_csv
from organoidtracker.services.annotations import AnnotationError
from organoidtracker.services.export_service import ExportError
from organoidtracker.services.pipeline import EXIT_PARTIAL, run_session
from organoidtracker.services.prompt_record import PROMPT_RECORD_SCHEMA, build_prompt_record
from organoidtracker.services.session import (
    Calibration,
    Session,
    SessionError,
    Timing,
    TrackingSpec,
    VideoReference,
    session_from_document,
)
from organoidtracker.services.tracking_service import TrackingError, TrackingService

N = 8


@pytest.fixture(scope="module")
def small_disc():
    return DiscVideo(n_frames=N, size=128, radius=10, start=20, step=12)


@pytest.fixture(scope="module")
def small_video(small_disc, tmp_path_factory):
    return small_disc.write(tmp_path_factory.mktemp("video") / "disc.mp4")


def annotations(disc, *, extra_empty=False, extra_untracked=False, two_objects=False):
    cx, cy = disc.centers[N - 1]
    cysts = [{"cyst_id": 1, "bbox": list(disc.box(N - 1))}]
    if two_objects:
        cysts.append({"cyst_id": 2, "bbox": [cx - 15, cy - 15, cx + 20, cy + 15]})
    organoids = [{"organoid_id": 1, "point": [cx - 30, cy - 30], "cysts": cysts}]
    if extra_empty:
        organoids.append({"organoid_id": 2, "point": [5, 5], "cysts": []})
    if extra_untracked:
        organoids.append({"organoid_id": 3, "point": [100, 100], "cysts": [{"cyst_id": 9, "bbox": [90, 90, 120, 120]}]})
    return organoids


def make_session(disc, video, *, timing=None, calibration=1.0, **kwargs) -> Session:
    return session_from_document(
        {
            "schema": "organoidtracker.session/1",
            "video": {"path": str(video), "sha256": hashlib.sha256(video.read_bytes()).hexdigest()},
            "tracking": {"model_config": "sam2_hiera_t", "device": "cpu"},
            "calibration": {"um_per_pixel": calibration},
            "timing": timing or {"time_lapse_days": 7.0},
            "organoids": annotations(disc, **kwargs),
        }
    )


def factory(disc, **fake_kwargs):
    return lambda spec: TrackingService(FakeTracker(disc, enable_reverse_tracking=spec.reverse, **fake_kwargs))


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def run(session, out, disc, *, videos=False, **fake_kwargs):
    return run_session(session, out, videos=videos, tracking_service_factory=factory(disc, **fake_kwargs))


def test_complete_run_produces_every_export_and_a_truthful_manifest(small_disc, small_video, tmp_path):
    session = make_session(small_disc, small_video)
    outcome = run(session, tmp_path / "run", small_disc, videos=True)
    assert outcome.complete and outcome.status == "completed" and outcome.exit_code == 0
    out = outcome.output_dir
    for name in (
        "session.json",
        "prompts.json",
        "run_manifest.json",
        "raw_cyst_data.csv",
        "cyst_summary.csv",
        "organoid_summary.csv",
        "analysis_summary.json",
        "organoid_analysis_report.pdf",
        "visualizations/f_lasagna_plot.png",
        "videos/multi_object_overlay.mp4",
        "videos/multi_object_mask.mp4",
        "videos/multi_object_side_by_side.mp4",
    ):
        assert (out / name).is_file(), name

    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["schema"] == "organoidtracker.run/1" and manifest["status"] == "completed" and manifest["complete"]
    assert manifest["results_version"] == 2
    tracking = manifest["tracking"]
    assert (
        tracking["frames_total"] == N
        and tracking["frames_done"] == N
        and tracking["tracked_frames"] == list(range(N - 1, -1, -1))  # visited in reverse order
    )
    assert (
        tracking["direction"] == "reverse" and tracking["annotation_frame"] == N - 1 and tracking["object_ids"] == [1]
    )
    assert tracking["frame_map"] == list(range(N)) and tracking["error"] is None
    assert sorted(int(k) for k in manifest["masks"]) == list(range(N))
    for objects in manifest["masks"].values():
        digest = objects["1"]
        assert digest["shape"] == [128, 128] and digest["area"] > 0 and len(digest["sha256"]) == 64
    assert manifest["video"]["sha256"] == session.video.sha256 and manifest["video"]["unique_frames"] == N
    assert manifest["session"]["video"]["sha256"] == session.video.sha256
    assert manifest["video_export"] == {"fps": 5.0, "alpha": 0.4, "quality": "original", "quality_scale": 1.0}
    assert manifest["provenance"]["backend"] == "fake" and "SAM2_MIN_MASK_AREA" in manifest["settings"]
    assert {"python", "torch", "opencv", "numpy"} <= set(manifest["environment"])
    # every listed file exists with the recorded content; the manifest itself is not listed
    assert "run_manifest.json" not in manifest["files"]
    for relative, info in manifest["files"].items():
        assert hashlib.sha256((out / relative).read_bytes()).hexdigest() == info["sha256"], relative
    assert "raw_cyst_data.csv" in manifest["files"] and "videos/multi_object_overlay.mp4" in manifest["files"]

    summary = json.loads((out / "analysis_summary.json").read_text())
    assert summary["success"] and summary["complete"] and summary["tracking"]["status"] == "completed"
    assert summary["experiment_info"]["total_frames"] == N and summary["experiment_info"]["total_cysts"] == 1
    rows = read_csv(out / "raw_cyst_data.csv")
    assert [int(r["Frame"]) for r in rows] == list(range(N))
    assert [float(r["Time_Days"]) for r in rows] == [1.0 + k for k in range(N)]  # 7 days over 8 frames, day 1 first

    record = json.loads((out / "prompts.json").read_text())
    assert record["schema"] == PROMPT_RECORD_SCHEMA and record["results_version"] == 2
    assert record["prompts"]["1"][0]["frame_idx"] == N - 1 and record["analysis_inputs"]["time_lapse_days"] == 7.0
    assert record["organoids"][0]["cysts"][0]["cyst_id"] == 1


def test_interrupted_run_exports_are_explicitly_partial(small_disc, small_video, tmp_path):
    session = make_session(small_disc, small_video)
    outcome = run(session, tmp_path / "run", small_disc, videos=True, partial_after=3)
    assert outcome.status == "partial" and not outcome.complete and outcome.exit_code == EXIT_PARTIAL == 3
    out = outcome.output_dir
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["status"] == "partial" and not manifest["complete"]
    assert manifest["tracking"]["frames_done"] == 3 and manifest["tracking"]["frames_total"] == N
    assert (
        manifest["tracking"]["tracked_frames"] == [7, 6, 5]
        and "synthetic interruption" in manifest["tracking"]["error"]
    )
    assert manifest["analysis"]["observed_frames"] == [5, 6, 7]
    assert any(w.startswith("Tracking partial: 3 of 8 frames") for w in manifest["analysis"]["warnings"])
    summary = json.loads((out / "analysis_summary.json").read_text())
    assert summary["success"] and not summary["complete"] and summary["tracking"]["status"] == "partial"
    assert summary["experiment_info"]["total_frames"] == N  # the time axis is the video's, not the covered frames
    rows = read_csv(out / "raw_cyst_data.csv")
    assert sorted(int(r["Frame"]) for r in rows) == [5, 6, 7]
    assert (out / "organoid_analysis_report.pdf").is_file() and (out / "videos/multi_object_overlay.mp4").is_file()
    # a partial run still re-plots with its gaps
    reloaded = experiment_from_csv(out / "raw_cyst_data.csv")
    assert reloaded.total_frames == N and reloaded.observed_frames == [5, 6, 7]


def test_tracked_frame_without_accepted_masks_keeps_the_time_axis(small_disc, small_video, tmp_path):
    session = make_session(small_disc, small_video)
    outcome = run(session, tmp_path / "run", small_disc, frames_without_masks={0})
    assert outcome.complete
    manifest = json.loads(outcome.manifest_path.read_text())
    assert manifest["tracking"]["frames_with_masks"] == N - 1 and manifest["tracking"]["frames_total"] == N
    assert sorted(manifest["tracking"]["tracked_frames"]) == list(range(N)) and "0" not in manifest["masks"]
    rows = read_csv(outcome.output_dir / "raw_cyst_data.csv")
    assert [int(r["Frame"]) for r in rows] == list(range(1, N))
    assert [float(r["Time_Days"]) for r in rows] == [2.0 + k for k in range(N - 1)]  # frame 1 is still day 2
    assert manifest["analysis"]["observed_frames"] == list(range(N))  # frame 0 was tracked: a real zero, not a gap


def test_missing_observation_and_untracked_cyst_are_reported_not_filled(small_disc, small_video, tmp_path):
    session = make_session(small_disc, small_video, two_objects=True, extra_untracked=True)
    outcome = run(session, tmp_path / "run", small_disc, drop={(3, 2)}, never_track={9})
    assert outcome.complete
    manifest = json.loads(outcome.manifest_path.read_text())
    assert manifest["tracking"]["object_ids"] == [1, 2]
    assert "2" not in manifest["masks"]["3"] and "1" in manifest["masks"]["3"]
    assert manifest["analysis"]["untracked_cysts"] == [9] and manifest["analysis"]["total_cysts"] == 2
    assert any("Annotated cysts without tracked masks: [9]" in w for w in manifest["analysis"]["warnings"])
    rows = read_csv(outcome.output_dir / "cyst_summary.csv")
    tracked = {int(r["Cyst_ID"]): int(r["Frames_Tracked"]) for r in rows}
    assert tracked == {1: N, 2: N - 1}
    raw = read_csv(outcome.output_dir / "raw_cyst_data.csv")
    assert sorted(int(r["Frame"]) for r in raw if r["Cyst_ID"] == "2") == [k for k in range(N) if k != 3]


def test_mixed_population_survives_exports_and_the_csv_round_trip(small_disc, small_video, tmp_path):
    session = make_session(small_disc, small_video, extra_empty=True)
    outcome = run(session, tmp_path / "run", small_disc)
    out = outcome.output_dir
    manifest = json.loads(outcome.manifest_path.read_text())
    assert manifest["analysis"]["total_organoids"] == 2 and manifest["analysis"]["organoids_without_cysts"] == [2]
    organoid_rows = read_csv(out / "organoid_summary.csv")
    assert [(r["Organoid_ID"], r["Total_Cysts"]) for r in organoid_rows] == [("1", "1"), ("2", "0")]
    summary = json.loads((out / "analysis_summary.json").read_text())
    assert (
        summary["experiment_info"]["total_organoids"] == 2 and summary["quality_metrics"]["organoids_with_cysts"] == 1
    )

    reloaded = experiment_from_csv(out / "raw_cyst_data.csv")
    assert reloaded.get_total_organoid_count() == 2 and not reloaded.organoids[2].cysts
    assert reloaded.get_percentage_organoids_with_cysts_at_frame(0) == 50.0
    assert reloaded.frame_timestamps == manifest["analysis"]["frame_timestamps"]
    cyst = reloaded.organoids[1].cysts[1]
    growth = float(read_csv(out / "cyst_summary.csv")[0]["Growth_Rate_um2_per_day"])
    assert reloaded.growth_rate_per_day(cyst) == pytest.approx(growth, abs=1e-3)


@pytest.mark.parametrize(
    ("timing", "expected_times"),
    [
        ({"time_lapse_days": 14.0}, [1.0 + 2 * k for k in range(N)]),  # two days per frame
        ({"frame_times_days": [0, 1, 3, 4, 7, 8, 10, 11]}, [0.0, 1.0, 3.0, 4.0, 7.0, 8.0, 10.0, 11.0]),
    ],
)
def test_non_daily_and_explicit_timestamps_drive_the_growth_rate(
    small_disc, small_video, tmp_path, timing, expected_times
):
    session = make_session(small_disc, small_video, timing=timing, calibration=1.6934)
    outcome = run(session, tmp_path / "run", small_disc, grow=3)
    out = outcome.output_dir
    manifest = json.loads(outcome.manifest_path.read_text())
    assert manifest["analysis"]["frame_timestamps"] == expected_times
    rows = read_csv(out / "raw_cyst_data.csv")
    assert [float(r["Time_Days"]) for r in rows] == expected_times
    areas = {int(frame): objects["1"]["area"] for frame, objects in manifest["masks"].items()}
    expected = expected_growth_per_day(areas, expected_times, 1.6934)
    assert float(read_csv(out / "cyst_summary.csv")[0]["Growth_Rate_um2_per_day"]) == pytest.approx(expected, abs=1e-3)
    assert float(read_csv(out / "organoid_summary.csv")[0]["Growth_Rate_um2_per_day"]) == pytest.approx(
        (areas[N - 1] - areas[0]) * 1.6934**2 / (expected_times[-1] - expected_times[0]), abs=1e-3
    )
    summary = json.loads((out / "analysis_summary.json").read_text())
    assert summary["experiment_info"]["time_lapse_days"] == pytest.approx(expected_times[-1] - expected_times[0])
    # areas in the raw table are the mask areas times the squared conversion factor
    assert float(rows[0]["Area_um2"]) == pytest.approx(areas[0] * 1.6934**2, abs=0.01)


def test_explicit_timestamps_must_match_the_frame_count(small_disc, small_video, tmp_path):
    session = make_session(small_disc, small_video, timing={"frame_times_days": [1, 2, 3]})
    with pytest.raises(SessionError, match="3 values but the video has 8"):
        run(session, tmp_path / "run", small_disc)


def test_video_checks_happen_before_the_model_is_loaded(small_disc, small_video, tmp_path):
    session = make_session(small_disc, small_video)
    missing = session.with_video(tmp_path / "gone.mp4")
    with pytest.raises(SessionError, match="video not found"):
        run(missing, tmp_path / "run", small_disc)
    other = tmp_path / "other.mp4"
    other.write_bytes(small_video.read_bytes() + b"\0")
    with pytest.raises(SessionError, match="different video"):
        run(session.with_video(other), tmp_path / "run", small_disc)
    assert not (tmp_path / "run" / "session.json").exists()


def test_relocated_video_with_the_same_content_runs(small_disc, small_video, tmp_path):
    session = make_session(small_disc, small_video)
    moved = tmp_path / "moved.mp4"
    moved.write_bytes(small_video.read_bytes())
    outcome = run(session.with_video(moved), tmp_path / "run", small_disc)
    assert outcome.complete and json.loads((outcome.output_dir / "session.json").read_text())["video"]["path"] == str(
        moved.resolve()
    )


def test_a_previous_run_is_not_overwritten_silently(small_disc, small_video, tmp_path):
    session = make_session(small_disc, small_video)
    run(session, tmp_path / "run", small_disc)
    with pytest.raises(ExportError, match="already holds a run"):
        run(session, tmp_path / "run", small_disc)
    outcome = run_session(
        session, tmp_path / "run", videos=False, overwrite=True, tracking_service_factory=factory(small_disc)
    )
    assert outcome.complete


def test_sessions_without_cysts_or_outside_boxes_are_refused(small_disc, small_video, tmp_path):
    document = make_session(small_disc, small_video).to_document()
    document["organoids"] = [{"organoid_id": 1, "point": [1, 1], "cysts": []}]
    with pytest.raises(AnnotationError, match="at least one cyst box"):
        run(session_from_document(document), tmp_path / "run", small_disc)
    document["organoids"] = [{"organoid_id": 1, "point": [1, 1], "cysts": [{"cyst_id": 1, "bbox": [0, 0, 200, 200]}]}]
    with pytest.raises(AnnotationError, match="exceeds the frame size 128x128"):
        run(session_from_document(document), tmp_path / "run", small_disc)


def test_backend_failures_are_tracking_errors(small_disc, small_video, tmp_path):
    session = make_session(small_disc, small_video)
    with pytest.raises(TrackingError, match="did not load"):
        run(session, tmp_path / "a", small_disc, fail_load=True)
    with pytest.raises(TrackingError, match="rejected the box"):
        run(session, tmp_path / "b", small_disc, reject_prompts=True)
    with pytest.raises(TrackingError, match="tracking failed"):
        run(session, tmp_path / "c", small_disc, partial_after=0)  # no mask at all


def test_prompt_record_round_trips_into_a_session(small_disc, small_video):
    tracker = FakeTracker(small_disc)
    assert tracker.load_model()
    tracker.load_video(str(small_video))
    organoid_data = {1: {"point": (10, 10), "cysts": [{"cyst_id": 1, "bbox": tuple(small_disc.box(N - 1))}]}}
    assert tracker.add_bbox_prompt(*small_disc.box(N - 1), obj_id=1)
    record = build_prompt_record(tracker, str(small_video), organoid_data, 6.0, 1.6934)
    assert record["schema"] == PROMPT_RECORD_SCHEMA and record["video"]["sha256"] == tracker.video_sha256
    assert record["tracking"]["annotation_frame_index"] == N - 1 and record["prompts"]["1"][0]["frame_idx"] == N - 1
    session = session_from_document(record)
    assert session.video.path == Path(str(small_video)) and session.video.sha256 == tracker.video_sha256
    assert session.timing == Timing(time_lapse_days=6.0) and session.calibration == Calibration(1.6934)
    assert session.tracking == TrackingSpec(
        model_config="sam2_hiera_t", checkpoint_family="2.1", device="cpu", checkpoint_path=Path("/nonexistent/fake.pt")
    )
    assert session.annotations.cyst_ids() == [1]
    assert session.video == VideoReference(Path(str(small_video)), tracker.video_sha256)
