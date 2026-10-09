"""Cancellation: a cooperative request honoured between frames, with four guarantees checked through the services,
the pipeline, the command line and the saved result: a request before the first mask ends the run with nothing to
export; a request after frames with masks keeps everything they produced, labelled cancelled everywhere; a run after
a cancellation completes like an uninterrupted one; a request when nothing runs is a no-op."""

import csv
import json
import signal
import threading

import pytest

from helpers import DiscVideo, FakeTracker
from organoidtracker import cli
from organoidtracker.services.pipeline import EXIT_CANCELLED, RunCancelled, export_saved_result, run_session
from organoidtracker.services.run_manifest import mask_digests
from organoidtracker.services.saved_results import load_saved_result
from organoidtracker.services.session import session_from_document
from organoidtracker.services.tracking_service import TrackingService

N = 8


@pytest.fixture(scope="module")
def small_disc():
    return DiscVideo(n_frames=N, size=128, radius=10, start=20, step=12)


@pytest.fixture(scope="module")
def small_video(small_disc, tmp_path_factory):
    return small_disc.write(tmp_path_factory.mktemp("video") / "disc.mp4")


def make_session(disc, video, calibration=1.6934, times=(0, 1, 3, 6, 10, 15, 21, 28)):
    import hashlib

    cx, cy = disc.centers[N - 1]
    return session_from_document(
        {
            "schema": "organoidtracker.session/1",
            "video": {"path": str(video), "sha256": hashlib.sha256(video.read_bytes()).hexdigest()},
            "tracking": {"model_config": "sam2_hiera_t", "device": "cpu"},
            "calibration": {"um_per_pixel": calibration},
            "timing": {"frame_times_days": list(times)},
            "organoids": [
                {
                    "organoid_id": 1,
                    "point": [cx - 30, cy - 30],
                    "cysts": [{"cyst_id": 1, "bbox": list(disc.box(N - 1))}],
                },
                {"organoid_id": 2, "point": [5, 5], "cysts": []},
            ],
        }
    )


def service_for(disc, **fake_kwargs):
    service = TrackingService(FakeTracker(disc, **fake_kwargs))
    service.load_model()
    return service


def cancel_at(service, frame_number):
    """A progress callback that requests cancellation when ``frame_number`` frames are done."""

    def progress(current, total, message):
        if current == frame_number:
            assert service.cancel()

    return progress


# ------------------------------------------------------------------------------ the service
def test_a_request_before_the_first_frame_ends_the_run_without_masks(small_disc, small_video):
    service = service_for(small_disc)
    service.open_video(small_video)
    service.annotate(make_session(small_disc, small_video).annotations)
    assert not service.cancel()  # nothing runs: a no-op
    result = service.run(should_stop=lambda: True)
    assert result.is_cancelled and result.frames_done == 0 and not result and result.tracked_frames == []
    assert service.state == "cancelled" and result.error is None
    # the next run on the same service completes
    again = service.run()
    assert again.is_complete and again.frames_done == N and service.state == "completed"


def test_a_request_mid_run_keeps_the_tracked_frames_and_the_next_run_completes(small_disc, small_video):
    service = service_for(small_disc)
    service.open_video(small_video)
    service.annotate(make_session(small_disc, small_video).annotations)
    result = service.run(cancel_at(service, 3))
    assert result.is_cancelled and result.frames_done == 3 and result.tracked_frames == [7, 6, 5]
    assert sorted(result) == [5, 6, 7] and result.error is None and service.state == "cancelled"
    assert not service.cancel_requested or service.state == "cancelled"
    full = service.run()
    assert full.is_complete and full.frames_done == N and service.state == "completed"
    reference = service_for(small_disc)
    reference.open_video(small_video)
    reference.annotate(make_session(small_disc, small_video).annotations)
    assert mask_digests(full) == mask_digests(reference.run())  # same masks as an uninterrupted run


def test_a_request_during_the_last_frame_is_not_a_cancellation(small_disc, small_video):
    service = service_for(small_disc)
    service.open_video(small_video)
    service.annotate(make_session(small_disc, small_video).annotations)
    result = service.run(cancel_at(service, N))  # requested when the last frame is done: nothing left to stop
    assert result.is_complete and result.frames_done == N and service.state == "completed"


def test_a_cancelled_run_whose_frames_kept_no_mask_has_nothing_to_export(small_disc, small_video, tmp_path):
    session = make_session(small_disc, small_video)

    def factory(spec):
        return service_for(small_disc, enable_reverse_tracking=spec.reverse, frames_without_masks=set(range(N)))

    cancel = threading.Event()

    def progress(phase, current, total, message):
        if phase == "tracking" and current == 2:
            cancel.set()

    with pytest.raises(RunCancelled, match="cancelled after 2 of 8 frames, no mask kept"):
        run_session(
            session, tmp_path / "run", videos=False, tracking_service_factory=factory, progress=progress, cancel=cancel
        )
    out = tmp_path / "run"
    assert sorted(p.name for p in out.iterdir()) == ["prompts.json", "session.json"]  # nothing else written


# ------------------------------------------------------------------------------ the pipeline and the saved result
def test_a_cancelled_run_is_saved_analyzed_and_exported_as_cancelled(small_disc, small_video, tmp_path):
    session = make_session(small_disc, small_video)
    cancel = threading.Event()

    def progress(phase, current, total, message):
        if phase == "tracking" and current == 3:
            cancel.set()

    outcome = run_session(
        session,
        tmp_path / "run",
        videos=True,
        tracking_service_factory=lambda spec: service_for(small_disc, enable_reverse_tracking=spec.reverse),
        progress=progress,
        cancel=cancel,
    )
    assert outcome.status == "cancelled" and not outcome.complete and outcome.exit_code == EXIT_CANCELLED == 4
    out = outcome.output_dir
    manifest = json.loads(outcome.manifest_path.read_text())
    assert manifest["status"] == "cancelled" and not manifest["complete"]
    assert manifest["tracking"]["frames_done"] == 3 and manifest["tracking"]["tracked_frames"] == [7, 6, 5]
    assert manifest["tracking"]["error"] is None and manifest["analysis"]["observed_frames"] == [5, 6, 7]
    assert any(w.startswith("Tracking cancelled: 3 of 8 frames") for w in manifest["analysis"]["warnings"])
    summary = json.loads((out / "analysis_summary.json").read_text())
    assert summary["success"] and not summary["complete"] and summary["tracking"]["status"] == "cancelled"
    assert (out / "organoid_analysis_report.pdf").is_file() and (out / "videos/multi_object_overlay.mp4").is_file()
    with open(out / "raw_cyst_data.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert sorted(int(r["Frame"]) for r in rows) == [5, 6, 7]
    assert sorted(float(r["Time_Days"]) for r in rows) == [15.0, 21.0, 28.0]  # the explicit axis survives
    saved = load_saved_result(out)
    assert saved.result.is_cancelled and saved.result.frames_done == 3
    assert mask_digests(saved.result) == manifest["masks"] == mask_digests(outcome.result)
    again = export_saved_result(saved, tmp_path / "again", videos=True)
    assert again.status == "cancelled" and again.exit_code == 4
    for name in ("raw_cyst_data.csv", "cyst_summary.csv", "organoid_summary.csv"):
        assert (again.output_dir / name).read_bytes() == (out / name).read_bytes(), name
    assert json.loads(again.manifest_path.read_text())["status"] == "cancelled"


def test_a_saved_cancelled_run_must_not_claim_every_frame(small_disc, small_video, tmp_path):
    session = make_session(small_disc, small_video)
    cancel = threading.Event()
    outcome = run_session(
        session,
        tmp_path / "run",
        videos=False,
        tracking_service_factory=lambda spec: service_for(small_disc, enable_reverse_tracking=spec.reverse),
        progress=lambda phase, current, total, message: cancel.set() if current == 2 else None,
        cancel=cancel,
    )
    results = outcome.output_dir / "results.json"
    data = json.loads(results.read_text())
    assert data["tracking"]["status"] == "cancelled"
    data["tracking"]["frames_done"] = N
    data["tracking"]["tracked_frames"] = list(range(N - 1, -1, -1))
    results.write_text(json.dumps(data))
    from organoidtracker.services.saved_results import SavedResultError

    with pytest.raises(SavedResultError, match="a cancelled run cannot have tracked every frame"):
        load_saved_result(outcome.output_dir)


# ------------------------------------------------------------------------------ the command line
def test_ctrl_c_cancels_the_command_line_run_and_exits_4(tmp_path, small_disc, small_video, monkeypatch):
    """The first SIGINT turns into a cancel request honoured after the frame in progress; exit status 4."""
    monkeypatch.setattr(
        TrackingService,
        "create",
        classmethod(
            lambda cls, spec, registry=None: cls(FakeTracker(small_disc, enable_reverse_tracking=spec.reverse))
        ),
    )
    session_path = tmp_path / "session.json"
    session_path.write_text(json.dumps(make_session(small_disc, small_video).to_document()))
    original = FakeTracker.run_tracking

    def run_tracking_raising_sigint_at_frame_two(self, progress_callback=None, should_stop=None):
        def progress(current, total, message):
            if current == 2:
                signal.raise_signal(signal.SIGINT)  # delivered to this (the main) thread at once
            if progress_callback:
                progress_callback(current, total, message)

        return original(self, progress, should_stop)

    monkeypatch.setattr(FakeTracker, "run_tracking", run_tracking_raising_sigint_at_frame_two)
    out = tmp_path / "run"
    code = cli.main(["run", "--session", str(session_path), "--out", str(out), "--no-videos", "--log-level", "ERROR"])
    assert code == 4
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["status"] == "cancelled" and manifest["tracking"]["frames_done"] == 2
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler  # the handler was restored
    # the saved run exports again as cancelled
    assert (
        cli.main(["export", "--run", str(out), "--out", str(tmp_path / "again"), "--no-videos", "--log-level", "ERROR"])
        == 4
    )


def test_ctrl_c_before_the_first_frame_exits_4_with_nothing_exported(tmp_path, small_disc, small_video, monkeypatch):
    monkeypatch.setattr(
        TrackingService,
        "create",
        classmethod(
            lambda cls, spec, registry=None: cls(FakeTracker(small_disc, enable_reverse_tracking=spec.reverse))
        ),
    )
    session_path = tmp_path / "session.json"
    session_path.write_text(json.dumps(make_session(small_disc, small_video).to_document()))
    original = FakeTracker.run_tracking

    def run_tracking_interrupted_before_the_loop(self, progress_callback=None, should_stop=None):
        signal.raise_signal(signal.SIGINT)
        return original(self, progress_callback, should_stop)

    monkeypatch.setattr(FakeTracker, "run_tracking", run_tracking_interrupted_before_the_loop)
    out = tmp_path / "run"
    assert (
        cli.main(["run", "--session", str(session_path), "--out", str(out), "--no-videos", "--log-level", "ERROR"]) == 4
    )
    assert sorted(p.name for p in out.iterdir()) == ["organoidtracker.log", "prompts.json", "session.json"]


def test_a_second_ctrl_c_aborts_at_once(tmp_path, small_disc, small_video, monkeypatch):
    monkeypatch.setattr(
        TrackingService,
        "create",
        classmethod(
            lambda cls, spec, registry=None: cls(FakeTracker(small_disc, enable_reverse_tracking=spec.reverse))
        ),
    )
    session_path = tmp_path / "session.json"
    session_path.write_text(json.dumps(make_session(small_disc, small_video).to_document()))
    original = FakeTracker.run_tracking

    def run_tracking_interrupted_twice(self, progress_callback=None, should_stop=None):
        signal.raise_signal(signal.SIGINT)
        signal.raise_signal(signal.SIGINT)
        return original(self, progress_callback, should_stop)

    monkeypatch.setattr(FakeTracker, "run_tracking", run_tracking_interrupted_twice)
    with pytest.raises(KeyboardInterrupt):
        cli.main(
            [
                "run",
                "--session",
                str(session_path),
                "--out",
                str(tmp_path / "run"),
                "--no-videos",
                "--log-level",
                "ERROR",
            ]
        )
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    assert not (tmp_path / "run" / "run_manifest.json").exists()
