"""The Tk window drives the user's workflow through the services and produces exactly what a headless run of the
same session produces (marker: gui; a display is needed, no model: the fake backend stands in).

Covered: load model, load video, organoid clicks and cyst boxes, a reverted cyst that must not be tracked, an
organoid without cysts that must stay in the population, tracking, videos, the report; a partial run that the
log reports as partial; an analysis export that cannot be completed, reported as an error rather than success.
Windows runs one Tk interpreter per process (the smoke test holds it), so these tests run on Linux only.
"""

import csv
import hashlib
import json
import logging
import shlex
import sys
import tkinter
from pathlib import Path

import pytest

from helpers import DiscVideo, FakeTracker, TkDriver
from organoidtracker.services.pipeline import run_session
from organoidtracker.services.run_manifest import mask_digests
from organoidtracker.services.session import session_from_document
from organoidtracker.services.tracking_service import TrackingService
from test_gui_smoke import environment_skip_reason

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(
        sys.platform == "win32", reason="one Tk interpreter per process on Windows; the smoke test holds it"
    ),
]

N = 8


@pytest.fixture(scope="module")
def small_disc():
    return DiscVideo(n_frames=N, size=128, radius=10, start=20, step=12)


@pytest.fixture(scope="module")
def small_video(small_disc, tmp_path_factory):
    return small_disc.write(tmp_path_factory.mktemp("video") / "disc.mp4")


@pytest.fixture
def app(monkeypatch, small_disc, request, tmp_path):
    """A window whose backend is the fake tracker; ``request.param`` holds the fake's keyword arguments.

    It runs in the test's directory: the window writes prompt records and saved runs below ``data/`` there.
    """
    from organoidtracker.gui_tk.main_window import VideoTrackerApp

    monkeypatch.chdir(tmp_path)
    fake_kwargs = getattr(request, "param", {})
    monkeypatch.setattr(
        TrackingService,
        "create",
        classmethod(
            lambda cls, spec, registry=None: cls(
                FakeTracker(small_disc, enable_reverse_tracking=spec.reverse, **fake_kwargs)
            )
        ),
    )
    try:
        window = VideoTrackerApp()
    except tkinter.TclError as error:
        reason = environment_skip_reason(error)
        if reason is None:
            raise
        pytest.skip(reason)
    yield window
    window.root.destroy()


def log_of(app) -> str:
    return app.output_info_text.get("1.0", "end")


def saved_runs(app) -> list[Path]:
    """The run directories the window saved automatically (below its run root, in the test's directory)."""
    root = Path(app.run_root).resolve()
    return sorted(root.iterdir()) if root.is_dir() else []


def saved_run(app, run_id) -> Path:
    """The one directory the window saved run ``run_id`` into."""
    found = [path for path in saved_runs(app) if path.name.endswith(f"_{run_id}")]
    assert len(found) == 1, (run_id, saved_runs(app))
    return found[0]


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def workflow(
    app,
    monkeypatch,
    video,
    out,
    *,
    boxes,
    time_lapse=7.0,
    conversion=1.0,
    revert_last_cyst=False,
    extra_empty_point=None,
):
    """Drive: load model, load video, annotate (optionally revert the last cyst, add an empty organoid), track, videos, report."""
    from organoidtracker.gui_tk import main_window

    videos_dir, report_dir = out / "videos", out / "report"
    videos_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)

    def load_model():
        app.device_var.set("cpu")
        app.model_config_var.set("sam2_hiera_s")
        app.load_selected_model()

    def load_video():
        monkeypatch.setattr(main_window.filedialog, "askopenfilename", lambda *a, **k: str(video))
        app.load_video()

    def annotate_and_track():
        for point, cysts in boxes:
            app.on_canvas_click(*point)
            for box in cysts:
                app.on_canvas_bbox(*box)
        if revert_last_cyst:
            app.revert_last_action()
        if extra_empty_point is not None:
            app.on_canvas_click(*extra_empty_point)
        app.time_lapse_var.set(time_lapse)
        app.conversion_factor_var.set(conversion)
        app.start_tracking()

    def generate_videos():
        monkeypatch.setattr(main_window.filedialog, "askdirectory", lambda *a, **k: str(videos_dir))
        app.generate_videos()

    def generate_report():
        monkeypatch.setattr(main_window.filedialog, "askdirectory", lambda *a, **k: str(report_dir))
        app.generate_analysis_report()

    def tracked():
        return (
            not app.tracking_in_progress
            and app.video_segments is not None
            and str(app.track_btn["state"]) == "normal"
            and not app.autosaves_in_progress
        )

    def videos_written():
        return str(app.generate_btn["state"]) == "normal" and "Video generation completed" in log_of(app)

    def report_done():
        return str(app.analysis_btn["state"]) == "normal" and (
            "analysis report generated successfully" in log_of(app) or "ANALYSIS FAILED" in log_of(app)
        )

    (
        TkDriver(app.root)
        .step(
            "model loaded",
            load_model,
            lambda: app.current_model is not None and str(app.load_model_btn["state"]) == "normal",
            30,
        )
        .step(
            "video loaded",
            load_video,
            lambda: app.current_video_path is not None and str(app.track_btn["state"]) == "normal",
            30,
        )
        .step("tracked", annotate_and_track, tracked, 60)
        .step("videos written", generate_videos, videos_written, 60)
        .step("report done", generate_report, report_done, 120)
        .run()
    )
    return videos_dir, report_dir


def equivalent_session(video, organoids, time_lapse=7.0, conversion=1.0):
    return session_from_document(
        {
            "schema": "organoidtracker.session/1",
            "video": {"path": str(video), "sha256": hashlib.sha256(video.read_bytes()).hexdigest()},
            "tracking": {"model_config": "sam2_hiera_s", "device": "cpu"},
            "calibration": {"um_per_pixel": conversion},
            "timing": {"time_lapse_days": time_lapse},
            "organoids": organoids,
        }
    )


def test_window_matches_the_headless_run_and_honours_reverts_and_empty_organoids(
    app, monkeypatch, small_disc, small_video, tmp_path
):
    cx, cy = small_disc.centers[N - 1]
    box1 = tuple(int(v) for v in small_disc.box(N - 1))
    box2 = (cx - 15, cy - 15, cx + 20, cy + 15)
    videos_dir, report_dir = workflow(
        app,
        monkeypatch,
        small_video,
        tmp_path / "gui",
        boxes=[((cx - 30, cy - 30), [box1]), ((40, 40), [box2])],
        revert_last_cyst=True,  # cyst 2 removed; its organoid 2 is auto-removed with it
        extra_empty_point=(5, 5),  # a new organoid 2 without cysts
        time_lapse=14.0,
        conversion=1.6934,
    )
    assert app.video_segments.object_ids() == [1], "the reverted cyst must not be tracked"
    assert sorted(app.organoid_data) == [1, 2] and app.organoid_data[2]["cysts"] == []
    assert "✅ Tracking completed" in log_of(app) and "❌" not in log_of(app)
    for name in ("multi_object_overlay.mp4", "multi_object_mask.mp4", "multi_object_side_by_side.mp4"):
        assert (videos_dir / name).is_file()
    assert (report_dir / "organoid_analysis_report.pdf").is_file()

    # the same session headlessly: byte-identical tables, identical masks
    session = equivalent_session(
        small_video,
        [
            {"organoid_id": 1, "point": [cx - 30, cy - 30], "cysts": [{"cyst_id": 1, "bbox": list(box1)}]},
            {"organoid_id": 2, "point": [5, 5], "cysts": []},
        ],
        time_lapse=14.0,
        conversion=1.6934,
    )
    outcome = run_session(
        session,
        tmp_path / "cli",
        videos=True,
        tracking_service_factory=lambda spec: TrackingService(
            FakeTracker(small_disc, enable_reverse_tracking=spec.reverse)
        ),
    )
    for name in ("raw_cyst_data.csv", "cyst_summary.csv", "organoid_summary.csv"):
        assert (report_dir / name).read_bytes() == (outcome.output_dir / name).read_bytes(), name
    assert mask_digests(app.video_segments) == mask_digests(outcome.result)
    for name in ("multi_object_overlay.mp4", "multi_object_mask.mp4", "multi_object_side_by_side.mp4"):
        assert (videos_dir / name).stat().st_size == (outcome.output_dir / "videos" / name).stat().st_size, name
    gui_summary = json.loads((report_dir / "analysis_summary.json").read_text())
    cli_summary = json.loads((outcome.output_dir / "analysis_summary.json").read_text())
    for key in ("experiment_info", "growth_statistics", "quality_metrics", "tracking", "complete"):
        assert gui_summary[key] == cli_summary[key], key
    assert gui_summary["experiment_info"]["total_organoids"] == 2
    rows = read_csv(report_dir / "organoid_summary.csv")
    assert [(r["Organoid_ID"], r["Total_Cysts"]) for r in rows] == [("1", "1"), ("2", "0")]


@pytest.mark.parametrize("app", [{"partial_after": 3}], indirect=True)
def test_window_reports_a_partial_run_as_partial(app, monkeypatch, small_disc, small_video, tmp_path):
    cx, cy = small_disc.centers[N - 1]
    _videos_dir, report_dir = workflow(
        app,
        monkeypatch,
        small_video,
        tmp_path / "gui",
        boxes=[((cx - 30, cy - 30), [tuple(int(v) for v in small_disc.box(N - 1))])],
    )
    assert app.video_segments.status == "partial"
    log = log_of(app)
    assert "⚠️ Tracking stopped early" in log and "treat exports as partial" in log
    assert "PARTIAL TRACKING RUN: 3 of 8 frames" in log
    summary = json.loads((report_dir / "analysis_summary.json").read_text())
    assert summary["success"] and not summary["complete"] and summary["tracking"]["status"] == "partial"


def test_window_reports_an_incomplete_report_as_a_failure(app, monkeypatch, small_disc, small_video, tmp_path):
    from organoidtracker.analysis.organoid_report_generator import OrganoidAnalysisReportGenerator

    monkeypatch.setattr(OrganoidAnalysisReportGenerator, "_generate_enhanced_pdf_report", lambda self, *a, **k: None)
    cx, cy = small_disc.centers[N - 1]
    _videos_dir, report_dir = workflow(
        app,
        monkeypatch,
        small_video,
        tmp_path / "gui",
        boxes=[((cx - 30, cy - 30), [tuple(int(v) for v in small_disc.box(N - 1))])],
    )
    log = log_of(app)
    assert "ORGANOID ANALYSIS FAILED" in log and "organoid_analysis_report.pdf" in log
    assert "report generated successfully" not in log
    assert json.loads((report_dir / "analysis_summary.json").read_text())["success"] is False


def test_reloading_the_model_discards_stale_results_and_recovers(app, monkeypatch, small_disc, small_video, tmp_path):
    """Load, annotate, track; load a model again: results and export controls of the old backend go away, a stale
    export request is refused without getting stuck, and the workflow runs again on the new backend."""
    from organoidtracker.gui_tk import main_window

    cx, cy = small_disc.centers[N - 1]
    box = tuple(int(v) for v in small_disc.box(N - 1))
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    state: dict = {}

    def load_model():
        app.device_var.set("cpu")
        app.model_config_var.set("sam2_hiera_s")
        app.load_selected_model()

    def model_loaded():
        return app.current_model is not None and str(app.load_model_btn["state"]) == "normal"

    def load_video():
        monkeypatch.setattr(main_window.filedialog, "askopenfilename", lambda *a, **k: str(small_video))
        app.load_video()

    def video_loaded():
        return app.current_video_path is not None and str(app.track_btn["state"]) == "normal"

    def annotate_and_track():
        app.on_canvas_click(cx - 30, cy - 30)
        app.on_canvas_bbox(*box)
        app.start_tracking()

    def tracked():
        return (
            not app.tracking_in_progress and app.video_segments is not None and str(app.track_btn["state"]) == "normal"
        )

    def reload_model():
        state["first_service"] = app.tracking
        state["first_results"] = app.video_segments
        assert str(app.generate_btn["state"]) == "normal" and str(app.analysis_btn["state"]) == "normal"
        app.load_selected_model()

    def reloaded():
        return (
            app.tracking is not None
            and app.tracking is not state["first_service"]
            and str(app.load_model_btn["state"]) == "normal"
        )

    def stale_export_requests():
        # everything downstream of the old backend is gone and disabled
        assert app.video_segments is None and app.current_video_path is None and not app.organoid_data
        assert not app.active_object_ids and app.current_organoid_id is None
        for button in (app.track_btn, app.generate_btn, app.analysis_btn, app.clear_prompts_btn, app.revert_btn):
            assert str(button["state"]) == "disabled", button
        assert "discarded" in log_of(app)
        # a stale request (as if the buttons had still been enabled) is refused and nothing gets stuck
        monkeypatch.setattr(main_window.filedialog, "askdirectory", lambda *a, **k: str(videos_dir))
        app.generate_videos()
        app.generate_analysis_report()
        assert "No tracking results available" in app.status_label["text"]
        assert "Generating videos" not in app.status_label["text"]
        # the race between the swap and a click: old results still present, new backend without a video
        app.video_segments = state["first_results"]
        app.generate_videos()
        assert "Load a video and run tracking first" in app.status_label["text"]
        assert str(app.generate_btn["state"]) == "disabled" and not list(videos_dir.iterdir())
        app.video_segments = None

    def second_run():
        load_video()

    def second_tracked():
        return tracked() and app.video_segments is not state["first_results"]

    def generate_videos():
        monkeypatch.setattr(main_window.filedialog, "askdirectory", lambda *a, **k: str(videos_dir))
        app.generate_videos()

    (
        TkDriver(app.root)
        .step("model loaded", load_model, model_loaded, 30)
        .step("video loaded", load_video, video_loaded, 30)
        .step("tracked", annotate_and_track, tracked, 60)
        .step("model reloaded", reload_model, reloaded, 30)
        .step("stale requests refused", stale_export_requests, None, 5)
        .step("video loaded again", second_run, video_loaded, 30)
        .step("tracked again", annotate_and_track, second_tracked, 60)
        .step("videos written", generate_videos, lambda: "Video generation completed" in log_of(app), 60)
        .run()
    )
    assert app.video_segments.object_ids() == [1]
    assert all((videos_dir / f"multi_object_{t}.mp4").is_file() for t in ("overlay", "mask", "side_by_side"))
    assert str(app.generate_btn["state"]) == "normal" and "❌ Video generation failed" not in log_of(app)


def test_reload_completing_inside_the_folder_chooser_is_refused(app, monkeypatch, small_disc, small_video, tmp_path):
    """The folder chooser is modal and runs the event loop, so a model reload can complete while it is open and
    discard the results the readiness check had just approved. Both export paths must notice when the chooser
    returns, before changing any control, and refuse the request without getting stuck."""
    from organoidtracker.gui_tk import main_window

    cx, cy = small_disc.centers[N - 1]
    box = tuple(int(v) for v in small_disc.box(N - 1))
    out = tmp_path / "out"
    out.mkdir()

    def load_model():
        app.device_var.set("cpu")
        app.model_config_var.set("sam2_hiera_s")
        app.load_selected_model()

    def model_loaded():
        return app.current_model is not None and str(app.load_model_btn["state"]) == "normal"

    def load_video():
        monkeypatch.setattr(main_window.filedialog, "askopenfilename", lambda *a, **k: str(small_video))
        app.load_video()

    def video_loaded():
        return app.current_video_path is not None and str(app.track_btn["state"]) == "normal"

    def annotate_and_track():
        app.on_canvas_click(cx - 30, cy - 30)
        app.on_canvas_bbox(*box)
        app.start_tracking()

    def tracked():
        return (
            not app.tracking_in_progress and app.video_segments is not None and str(app.track_btn["state"]) == "normal"
        )

    def chooser_during_which_a_reload_completes(*args, **kwargs):
        # what the main loop does while the real chooser is open: it drains the queue and installs the new backend
        app.on_model_loaded_success(TrackingService(FakeTracker(small_disc)), 0.0)
        return str(out)

    def interleaved(request):
        def run():
            assert str(app.generate_btn["state"]) == "normal" and str(app.analysis_btn["state"]) == "normal"
            monkeypatch.setattr(main_window.filedialog, "askdirectory", chooser_during_which_a_reload_completes)
            request()
            # refused after the chooser: no control changed by the request itself, nothing written
            assert "Generating" not in app.status_label["text"]
            assert "No tracking results available" in app.status_label["text"]
            assert app.video_segments is None and not list(out.iterdir())

        return run

    (
        TkDriver(app.root)
        .step("model loaded", load_model, model_loaded, 30)
        .step("video loaded", load_video, video_loaded, 30)
        .step("tracked", annotate_and_track, tracked, 60)
        .step("videos refused after the chooser", interleaved(app.generate_videos), None, 5)
        .step("video loaded again", load_video, video_loaded, 30)
        .step("tracked again", annotate_and_track, tracked, 60)
        .step("report refused after the chooser", interleaved(app.generate_analysis_report), None, 5)
        .step("video loaded once more", load_video, video_loaded, 30)
        .step("tracked once more", annotate_and_track, tracked, 60)
        .step("save refused after the chooser", interleaved(app.save_results), None, 5)
        .step("video loaded a fourth time", load_video, video_loaded, 30)
        .step("tracked a fourth time", annotate_and_track, tracked, 60)
        .run()
    )
    # the window is still fully usable after both refusals
    assert str(app.generate_btn["state"]) == "normal" and str(app.analysis_btn["state"]) == "normal"
    assert app.video_segments.object_ids() == [1]


def test_window_saves_results_that_export_identically_without_the_tracker(
    app, monkeypatch, small_disc, small_video, tmp_path
):
    """Track in the window, save the results, export them again headlessly: the same masks, tables and videos."""
    from organoidtracker.gui_tk import main_window
    from organoidtracker.services.pipeline import export_saved_result
    from organoidtracker.services.saved_results import load_saved_result

    cx, cy = small_disc.centers[N - 1]
    box1 = tuple(int(v) for v in small_disc.box(N - 1))
    videos_dir, report_dir = workflow(
        app,
        monkeypatch,
        small_video,
        tmp_path / "gui",
        boxes=[((cx - 30, cy - 30), [box1])],
        extra_empty_point=(5, 5),
        time_lapse=14.0,
        conversion=1.6934,
    )
    save_dir = tmp_path / "saved"
    answers = {"replace": False}

    def save():
        monkeypatch.setattr(main_window.filedialog, "askdirectory", lambda *a, **k: str(save_dir))
        monkeypatch.setattr(main_window.messagebox, "askyesno", lambda *a, **k: answers["replace"])
        app.save_results()

    def saved():
        return str(app.save_results_btn["state"]) == "normal" and "Results saved" in log_of(app)

    def save_again_cancelled():
        save()  # the directory now holds a saved run and the question is answered "no"
        assert "Saving cancelled" in app.status_label["text"]

    def save_again_replacing():
        answers["replace"] = True
        save()

    (
        TkDriver(app.root)
        .step("results saved", save, saved, 60)
        .step("second save cancelled", save_again_cancelled, None, 5)
        .step("second save replaces", save_again_replacing, lambda: log_of(app).count("Results saved") == 2, 60)
        .run()
    )
    assert sorted(p.name for p in save_dir.iterdir() if not p.name.startswith("masks-")) == [
        "prompts.json",
        "results.json",
        "session.json",
    ]
    assert len(list(save_dir.glob("masks-*.npz"))) == 1
    reloaded = load_saved_result(save_dir)
    assert mask_digests(reloaded.result) == mask_digests(app.video_segments)
    assert reloaded.session.annotations.organoid_data() == {
        1: {"point": (cx - 30, cy - 30), "cysts": [{"cyst_id": 1, "bbox": box1}]},
        2: {"point": (5, 5), "cysts": []},
    }
    assert reloaded.session.timing.time_lapse_days == 14.0 and reloaded.session.calibration.um_per_pixel == 1.6934
    assert reloaded.run_id == app.tracking_run_id and reloaded.provenance["backend"] == "fake"

    again = export_saved_result(reloaded, tmp_path / "export", videos=True)
    assert again.exit_code == 0
    for name in ("raw_cyst_data.csv", "cyst_summary.csv", "organoid_summary.csv"):
        assert (again.output_dir / name).read_bytes() == (report_dir / name).read_bytes(), name
    for name in ("multi_object_overlay.mp4", "multi_object_mask.mp4", "multi_object_side_by_side.mp4"):
        assert (again.output_dir / "videos" / name).stat().st_size == (videos_dir / name).stat().st_size, name
    assert "💾 Results saved" in log_of(app) and "❌" not in log_of(app)


def test_window_opens_a_saved_run_without_a_model_and_exports_it(app, monkeypatch, small_disc, small_video, tmp_path):
    """A run saved by the command line opens in a window without any model; its videos and report come out identical,
    with the run's explicit irregular time axis; loading a video through a model then sets the reopened run aside."""
    from organoidtracker.gui_tk import main_window
    from organoidtracker.services.session import session_from_document

    cx, cy = small_disc.centers[N - 1]
    box1 = tuple(int(v) for v in small_disc.box(N - 1))
    times = [0, 1, 3, 6, 10, 15, 21, 28]
    session = session_from_document(
        {
            "schema": "organoidtracker.session/1",
            "video": {"path": str(small_video), "sha256": hashlib.sha256(small_video.read_bytes()).hexdigest()},
            "tracking": {"model_config": "sam2_hiera_s", "device": "cpu"},
            "calibration": {"um_per_pixel": 1.6934},
            "timing": {"frame_times_days": times},
            "organoids": [
                {"organoid_id": 1, "point": [cx - 30, cy - 30], "cysts": [{"cyst_id": 1, "bbox": list(box1)}]},
                {"organoid_id": 2, "point": [5, 5], "cysts": []},
            ],
        }
    )
    run = run_session(
        session,
        tmp_path / "run",
        videos=True,
        tracking_service_factory=lambda spec: TrackingService(
            FakeTracker(small_disc, enable_reverse_tracking=spec.reverse, partial_after=5, grow=3)
        ),
    )
    assert run.status == "partial" and app.current_model is None  # no model in this window

    videos_dir, report_dir = tmp_path / "videos", tmp_path / "report"
    videos_dir.mkdir()
    report_dir.mkdir()

    def open_results():
        monkeypatch.setattr(main_window.filedialog, "askdirectory", lambda *a, **k: str(run.output_dir))
        app.open_results()

    def opened():
        return app.opened is not None and str(app.open_results_btn["state"]) == "normal"

    def check_state():
        assert app.video_segments is run.saved_result.result or mask_digests(app.video_segments) == mask_digests(
            run.result
        )
        assert app.organoid_data == session.annotations.organoid_data()
        assert app.time_lapse_var.get() == 28.0 and app.conversion_factor_var.get() == 1.6934
        assert str(app.track_btn["state"]) == "disabled" and str(app.save_results_btn["state"]) == "disabled"
        assert str(app.generate_btn["state"]) == "normal" and str(app.analysis_btn["state"]) == "normal"
        assert "reopened run is partial" in log_of(app) and "explicit frame times" in log_of(app)
        app.on_canvas_click(5, 5)  # inert: the annotations shown are the run's
        app.on_canvas_bbox(1, 1, 10, 10)
        assert app.organoid_data == session.annotations.organoid_data()
        assert "not editable" in app.status_label["text"]

    def generate_videos():
        monkeypatch.setattr(main_window.filedialog, "askdirectory", lambda *a, **k: str(videos_dir))
        app.generate_videos()

    def generate_report():
        monkeypatch.setattr(main_window.filedialog, "askdirectory", lambda *a, **k: str(report_dir))
        app.generate_analysis_report()

    def report_done():
        return str(app.analysis_btn["state"]) == "normal" and (
            "analysis report generated successfully" in log_of(app) or "ANALYSIS FAILED" in log_of(app)
        )

    def load_model():
        app.device_var.set("cpu")
        app.model_config_var.set("sam2_hiera_s")
        app.load_selected_model()

    def load_video():
        monkeypatch.setattr(main_window.filedialog, "askopenfilename", lambda *a, **k: str(small_video))
        app.load_video()

    (
        TkDriver(app.root)
        .step("results opened", open_results, opened, 30)
        .step("reopened state", check_state, None, 5)
        .step("videos written", generate_videos, lambda: "Video generation completed" in log_of(app), 60)
        .step("report done", generate_report, report_done, 120)
        .step("model loaded", load_model, lambda: app.current_model is not None, 30)
        .step("set aside", lambda: None, lambda: app.opened is not None, 5)  # a first model keeps the reopened run
        .step("video loaded", load_video, lambda: app.current_video_path == str(small_video), 30)
        .run()
    )
    for name in ("raw_cyst_data.csv", "cyst_summary.csv", "organoid_summary.csv"):
        assert (report_dir / name).read_bytes() == (run.output_dir / name).read_bytes(), name
    rows = read_csv(report_dir / "raw_cyst_data.csv")
    assert sorted({float(r["Time_Days"]) for r in rows}) == [6.0, 10.0, 15.0, 21.0, 28.0]  # the explicit axis
    summary = json.loads((report_dir / "analysis_summary.json").read_text())
    assert summary["tracking"]["status"] == "partial" and summary["experiment_info"]["total_organoids"] == 2
    for name in ("multi_object_overlay.mp4", "multi_object_mask.mp4", "multi_object_side_by_side.mp4"):
        assert (videos_dir / name).stat().st_size == (run.output_dir / "videos" / name).stat().st_size, name
    # the video loaded through the model replaced the reopened run
    assert app.opened is None and app.video_segments is None and not app.organoid_data
    assert "set aside" in log_of(app) and str(app.generate_btn["state"]) == "disabled"
    assert str(app.track_btn["state"]) == "normal"
    assert saved_runs(app) == []  # opening a saved run saves nothing


def _track_in_window(app, monkeypatch, video, box, point=None):
    """Driver steps that load the fake model, the video, one organoid with one cyst box, and track."""
    from organoidtracker.gui_tk import main_window

    def load_model():
        app.device_var.set("cpu")
        app.model_config_var.set("sam2_hiera_s")
        app.load_selected_model()

    def load_video():
        monkeypatch.setattr(main_window.filedialog, "askopenfilename", lambda *a, **k: str(video))
        app.load_video()

    def annotate_and_track():
        app.on_canvas_click(*(point or (5, 5)))
        app.on_canvas_bbox(*box)
        app.start_tracking()

    def tracked():
        return (
            not app.tracking_in_progress
            and app.video_segments is not None
            and str(app.track_btn["state"]) == "normal"
            and not app.autosaves_in_progress
        )

    return (
        TkDriver(app.root)
        .step("model loaded", load_model, lambda: app.current_model is not None, 30)
        .step("video loaded", load_video, lambda: app.current_video_path == str(video), 30)
        .step("tracked", annotate_and_track, tracked, 60)
    )


def test_a_failed_save_over_a_saved_run_keeps_it(app, monkeypatch, small_disc, small_video, tmp_path):
    """Review finding: Save Results cleared the previous saved run before writing; a failed write left nothing."""
    from organoidtracker.gui_tk import main_window
    from organoidtracker.services import saved_results
    from organoidtracker.services.saved_results import load_saved_result

    box = tuple(int(v) for v in small_disc.box(N - 1))
    cx, cy = small_disc.centers[N - 1]
    previous = run_session(
        equivalent_session(
            small_video, [{"organoid_id": 1, "point": [1, 1], "cysts": [{"cyst_id": 1, "bbox": list(box)}]}]
        ),
        tmp_path / "dest",
        videos=False,
        tracking_service_factory=lambda spec: TrackingService(
            FakeTracker(small_disc, enable_reverse_tracking=spec.reverse, grow=5)
        ),
    )
    dest = previous.output_dir
    before = {p.name: p.read_bytes() for p in dest.iterdir() if p.is_file()}

    def save():
        monkeypatch.setattr(main_window.filedialog, "askdirectory", lambda *a, **k: str(dest))
        monkeypatch.setattr(main_window.messagebox, "askyesno", lambda *a, **k: True)
        app.save_results()

    def failing_save():
        monkeypatch.setattr(
            saved_results,
            "_write_npz",
            lambda handle, arrays: (_ for _ in ()).throw(OSError("injected disk-full error")),
        )
        save()

    def failed():
        return (
            "Saving the results failed" in app.status_label["text"] and str(app.save_results_btn["state"]) == "normal"
        )

    def check_previous_intact():
        assert "injected disk-full error" in app.status_label["text"]
        assert {p.name: p.read_bytes() for p in dest.iterdir() if p.is_file()} == before  # manifest included
        assert mask_digests(load_saved_result(dest).result) == mask_digests(previous.result)
        monkeypatch.setattr(saved_results, "_write_npz", original_write_npz)

    import os

    original_replace = os.replace

    def failing_publication():
        # the review's reproduction: every file is written, the rename of session.json fails while publishing
        def fail_on_session(src, dst):
            if Path(dst).name == "session.json" and str(src).endswith(".tmp"):
                raise OSError("injected failure while publishing session.json")
            return original_replace(src, dst)

        monkeypatch.setattr(saved_results.os, "replace", fail_on_session)
        save()

    def publication_failed():
        return "publishing session.json" in app.status_label["text"] and str(app.save_results_btn["state"]) == "normal"

    def check_previous_intact_again():
        assert {p.name: p.read_bytes() for p in dest.iterdir() if p.is_file()} == before  # nothing mixed, nothing lost
        assert mask_digests(load_saved_result(dest).result) == mask_digests(previous.result)
        assert not list(dest.glob(".*"))
        monkeypatch.setattr(saved_results.os, "replace", original_replace)

    original_write_npz = saved_results._write_npz
    (
        _track_in_window(app, monkeypatch, small_video, box, point=(cx - 30, cy - 30))
        .step("failed save", failing_save, failed, 30)
        .step("previous run intact", check_previous_intact, None, 5)
        .step("failed publication", failing_publication, publication_failed, 30)
        .step("previous run intact again", check_previous_intact_again, None, 5)
        .step("replacing save", save, lambda: "Results saved" in app.status_label["text"], 30)
        .run()
    )
    assert not (dest / "run_manifest.json").exists()
    assert mask_digests(load_saved_result(dest).result) == mask_digests(app.video_segments)
    assert (dest / "raw_cyst_data.csv").read_bytes() == before["raw_cyst_data.csv"]  # exports left alone


def test_loading_another_video_invalidates_the_results(app, monkeypatch, small_disc, small_video, tmp_path):
    """Review finding: after loading another video through the same model, the old masks could be saved under the
    new video's identity (and exported over its frames). The results now belong to the video they were tracked on."""
    from organoidtracker.gui_tk import main_window
    from organoidtracker.services.saved_results import load_saved_result

    other_disc = DiscVideo(n_frames=N, size=128, radius=10, start=40, step=8)
    other_video = other_disc.write(tmp_path / "other.mp4")
    box = tuple(int(v) for v in small_disc.box(N - 1))
    cx, cy = small_disc.centers[N - 1]
    dest = tmp_path / "stale"
    state = {}

    def load_other_video():
        state["first_run_id"] = app.tracking_run_id
        monkeypatch.setattr(main_window.filedialog, "askopenfilename", lambda *a, **k: str(other_video))
        app.load_video()

    def other_loaded():
        return app.current_video_path == str(other_video) and str(app.track_btn["state"]) == "normal"

    def stale_requests_refused():
        assert app.video_segments is None and app.result_video is None
        for button in (app.generate_btn, app.analysis_btn, app.save_results_btn):
            assert str(button["state"]) == "disabled", button
        assert "previous tracking results were discarded" in log_of(app)
        monkeypatch.setattr(main_window.filedialog, "askdirectory", lambda *a, **k: str(dest))
        app.save_results()
        app.generate_videos()
        app.generate_analysis_report()
        assert "No tracking results available" in app.status_label["text"]
        assert not dest.exists()

    def track_other():
        app.on_canvas_click(5, 5)
        app.on_canvas_bbox(*tuple(int(v) for v in other_disc.box(N - 1)))
        app.start_tracking()

    def tracked_other():
        return not app.tracking_in_progress and app.video_segments is not None and app.result_video is not None

    def save_other():
        monkeypatch.setattr(main_window.filedialog, "askdirectory", lambda *a, **k: str(dest))
        app.save_results()

    (
        _track_in_window(app, monkeypatch, small_video, box, point=(cx - 30, cy - 30))
        .step("other video loaded", load_other_video, other_loaded, 30)
        .step("stale requests refused", stale_requests_refused, None, 5)
        .step("other video tracked", track_other, tracked_other, 60)
        .step("saved", save_other, lambda: "Results saved" in app.status_label["text"], 30)
        .run()
    )
    saved = load_saved_result(dest)
    assert saved.video.sha256 == hashlib.sha256(other_video.read_bytes()).hexdigest()
    assert mask_digests(saved.result) == mask_digests(app.video_segments)  # the fake's masks do not depend on the video
    assert saved.run_id == app.tracking_run_id and saved.run_id != state["first_run_id"]
    assert len(saved.session.annotations.organoids) == 1


def test_a_run_opened_inside_a_chooser_is_refused_and_the_next_request_uses_its_inputs(
    app, monkeypatch, small_disc, small_video, tmp_path
):
    """Review finding: run B installed while the report chooser was open was analyzed with run A's calibration and
    duration. A request whose results changed inside the dialog is refused; the next one uses B's inputs."""
    from dataclasses import replace

    from organoidtracker.gui_tk import main_window
    from organoidtracker.services.session import Calibration, Timing
    from organoidtracker.services.video_frames import load_video_frames

    box = tuple(int(v) for v in small_disc.box(N - 1))
    session_a = equivalent_session(
        small_video, [{"organoid_id": 1, "point": [1, 1], "cysts": [{"cyst_id": 1, "bbox": list(box)}]}]
    )
    run_a = run_session(
        session_a,
        tmp_path / "a",
        videos=False,
        tracking_service_factory=lambda spec: TrackingService(
            FakeTracker(small_disc, enable_reverse_tracking=spec.reverse)
        ),
    )
    saved_a = run_a.saved_result
    saved_b = replace(
        saved_a,
        run_id="run-b",
        session=replace(saved_a.session, calibration=Calibration(2.0), timing=Timing(time_lapse_days=14.0)),
    )
    frames = load_video_frames(small_video, saved_a.video)
    out = tmp_path / "report"
    out.mkdir()

    def chooser_during_which_run_b_opens(*args, **kwargs):
        app.on_results_opened(saved_b, frames, None, 0.0)  # what the main loop does while the real chooser is open
        return str(out)

    def report_with_swap():
        assert app.conversion_factor_var.get() == 1.0 and app.time_lapse_var.get() == 7.0
        monkeypatch.setattr(main_window.filedialog, "askdirectory", chooser_during_which_run_b_opens)
        app.generate_analysis_report()
        assert "results changed while the dialog was open" in app.status_label["text"]
        assert not list(out.iterdir())
        assert app.conversion_factor_var.get() == 2.0 and app.time_lapse_var.get() == 14.0  # B is shown now

    def videos_with_swap():
        monkeypatch.setattr(main_window.filedialog, "askdirectory", chooser_during_which_run_b_opens)
        app.generate_videos()
        assert "results changed while the dialog was open" in app.status_label["text"]
        assert not list(out.iterdir())

    def plain_report():
        monkeypatch.setattr(main_window.filedialog, "askdirectory", lambda *a, **k: str(out))
        app.generate_analysis_report()

    def report_done():
        return str(app.analysis_btn["state"]) == "normal" and "analysis report generated successfully" in log_of(app)

    (
        TkDriver(app.root)
        .step("run A opened", lambda: app.on_results_opened(saved_a, frames, None, 0.0), None, 5)
        .step("report refused after the swap", report_with_swap, None, 5)
        .step("videos refused after the swap", videos_with_swap, None, 5)
        .step("report of B", plain_report, report_done, 120)
        .run()
    )
    summary = json.loads((out / "analysis_summary.json").read_text())
    assert summary["experiment_info"]["conversion_factor_um_per_pixel"] == 2.0
    assert summary["experiment_info"]["time_lapse_days"] == 14.0


def _prepare_tracking(app, monkeypatch, video, box, point):
    """Driver steps that load the fake model and the video and place one organoid with one cyst box."""
    from organoidtracker.gui_tk import main_window

    def load_model():
        app.device_var.set("cpu")
        app.model_config_var.set("sam2_hiera_s")
        app.load_selected_model()

    def load_video():
        monkeypatch.setattr(main_window.filedialog, "askopenfilename", lambda *a, **k: str(video))
        app.load_video()

    def annotate():
        app.on_canvas_click(*point)
        app.on_canvas_bbox(*box)

    return (
        TkDriver(app.root)
        .step("model loaded", load_model, lambda: app.current_model is not None, 30)
        .step("video loaded", load_video, lambda: app.current_video_path == str(video), 30)
        .step("annotated", annotate, None, 5)
    )


@pytest.mark.parametrize("app", [{"frame_delay_s": 0.25}], indirect=True)
def test_cancel_from_the_progress_dialog_keeps_the_tracked_frames(app, monkeypatch, small_disc, small_video, tmp_path):
    """The dialog's Cancel button stops the run after the frame in progress; the tracked frames stay exportable and
    saveable as a cancelled run; tracking again on the same video completes with the uninterrupted masks."""
    from organoidtracker.gui_tk import main_window
    from organoidtracker.services.saved_results import load_saved_result

    cx, cy = small_disc.centers[N - 1]
    box = tuple(int(v) for v in small_disc.box(N - 1))
    save_dir = tmp_path / "saved"
    state = {}

    def dialog_showing_progress():
        dialog = app.tracking_dialog
        return dialog is not None and dialog.dialog is not None and dialog.progress_var.get() >= 100 * 2 / N

    def press_cancel():
        state["dialog"] = app.tracking_dialog
        app.tracking_dialog.request_cancel()  # the button's command
        assert (
            "Cancelling" in app.status_label["text"] and str(app.tracking_dialog.cancel_button["state"]) == "disabled"
        )

    def cancelled():
        return not app.tracking_in_progress and "cancelled" in app.status_label["text"]

    def check_cancelled_state():
        result = app.video_segments
        assert result is not None and result.status == "cancelled" and 2 <= result.frames_done < N
        assert result.tracked_frames == list(range(N - 1, N - 1 - result.frames_done, -1))
        assert state["dialog"].dialog is None  # closed
        for button in (app.track_btn, app.generate_btn, app.analysis_btn, app.save_results_btn):
            assert str(button["state"]) == "normal", button
        log = log_of(app)
        assert "⏹ Cancel requested" in log and "⏹ Tracking cancelled after" in log and "treat exports as partial" in log
        state["cancelled_digests"] = mask_digests(result)
        state["run_id"] = app.tracking_run_id

    def save():
        monkeypatch.setattr(main_window.filedialog, "askdirectory", lambda *a, **k: str(save_dir))
        app.save_results()

    def track_again():
        app.start_tracking()

    def completed():
        return not app.tracking_in_progress and app.video_segments is not None and app.video_segments.is_complete

    (
        _prepare_tracking(app, monkeypatch, small_video, box, (cx - 30, cy - 30))
        .step("tracking started", app.start_tracking, dialog_showing_progress, 30)
        .step("cancel pressed", press_cancel, cancelled, 30)
        .step("cancelled state", check_cancelled_state, None, 5)
        .step("saved", save, lambda: "Results saved" in app.status_label["text"], 30)
        .step("tracked again", track_again, completed, 60)
        .run()
    )
    saved = load_saved_result(save_dir)
    assert saved.result.status == "cancelled" and mask_digests(saved.result) == state["cancelled_digests"]
    assert saved.run_id == state["run_id"]
    # the second run is a new run with every frame; its masks equal an uninterrupted run's (the fake is deterministic)
    assert app.tracking_run_id != state["run_id"] and app.video_segments.frames_done == N
    full = run_session(
        equivalent_session(
            small_video, [{"organoid_id": 1, "point": [cx - 30, cy - 30], "cysts": [{"cyst_id": 1, "bbox": list(box)}]}]
        ),
        tmp_path / "full",
        videos=False,
        tracking_service_factory=lambda spec: TrackingService(
            FakeTracker(small_disc, enable_reverse_tracking=spec.reverse)
        ),
    )
    assert mask_digests(app.video_segments) == mask_digests(full.result)
    assert "✅ Tracking completed" in log_of(app)


@pytest.mark.parametrize("app", [{"start_delay_s": 0.5, "frame_delay_s": 0.05}], indirect=True)
def test_cancel_before_the_first_frame_leaves_no_results_and_tracking_again_works(
    app, monkeypatch, small_disc, small_video
):
    cx, cy = small_disc.centers[N - 1]
    box = tuple(int(v) for v in small_disc.box(N - 1))

    def start_and_cancel_at_once():
        app.start_tracking()
        assert app.tracking_in_progress
        app.cancel_tracking()  # before the backend's first check (it pauses 0.5 s before it)
        assert "Cancelling" in app.status_label["text"]

    def cancelled():
        return not app.tracking_in_progress and "cancelled" in app.status_label["text"]

    def check_state():
        assert app.video_segments is None and app.result_video is None
        assert "before any mask was kept" in log_of(app)
        for button in (app.generate_btn, app.analysis_btn, app.save_results_btn):
            assert str(button["state"]) == "disabled", button
        assert str(app.track_btn["state"]) == "normal"
        app.cancel_tracking()  # nothing runs: a no-op
        assert "No tracking is running" in app.status_label["text"]
        assert app.autosaves_in_progress == set() and "saved automatically" not in log_of(app)
        assert saved_runs(app) == []  # no mask was kept, so nothing was saved

    def completed():
        return not app.tracking_in_progress and app.video_segments is not None and app.video_segments.is_complete

    (
        _prepare_tracking(app, monkeypatch, small_video, box, (cx - 30, cy - 30))
        .step("started and cancelled", start_and_cancel_at_once, cancelled, 30)
        .step("no results", check_state, None, 5)
        .step("tracked again", app.start_tracking, completed, 60)
        .run()
    )
    assert app.video_segments.frames_done == N and str(app.save_results_btn["state"]) == "normal"
    TkDriver(app.root).step("saved", None, lambda: not app.autosaves_in_progress, 30).run()
    assert saved_runs(app) == [saved_run(app, app.tracking_run_id)]  # only the run that kept masks


def test_stale_tracking_callbacks_are_ignored(app, monkeypatch, small_disc, small_video, tmp_path):
    """A completion or error posted by a run that is no longer the current one changes nothing and closes its dialog."""
    from organoidtracker.core.tracking_result import TrackingResult

    cx, cy = small_disc.centers[N - 1]
    box = tuple(int(v) for v in small_disc.box(N - 1))

    class ClosableDialog:
        closed = False

        def close(self):
            self.closed = True

    def stale_callbacks():
        current = app.video_segments
        digests = mask_digests(current)
        stale_result = TrackingResult({0: {1: current[0][1]}}, status="completed", frames_total=N, frames_done=N)
        dialog = ClosableDialog()
        app.on_tracking_complete_success("old-run", app.tracking, dialog, stale_result, 0.0)
        assert dialog.closed and app.video_segments is current and mask_digests(app.video_segments) == digests
        other_service = TrackingService(FakeTracker(small_disc))
        dialog = ClosableDialog()
        app.on_tracking_complete_error(app.tracking_run_id, other_service, dialog, "late failure", 0.0)
        assert dialog.closed and app.video_segments is current and str(app.track_btn["state"]) == "normal"
        assert "Error during tracking" not in app.status_label["text"]
        app._on_tracking_progress("old-run", dialog, 50, "late progress")  # nothing to update, no error
        assert log_of(app).count("Ignored a stale tracking") == 2
        assert app.autosaves_in_progress == set()  # a save would have been registered synchronously
        assert log_of(app).count("saved automatically") == 1  # the run's own save, nothing else
        assert saved_runs(app) == [saved_run(app, app.tracking_run_id)]

    (
        _track_in_window(app, monkeypatch, small_video, box, point=(cx - 30, cy - 30))
        .step("stale callbacks ignored", stale_callbacks, None, 5)
        .run()
    )
    assert str(app.generate_btn["state"]) == "normal" and app.video_segments.is_complete


@pytest.mark.parametrize("app", [{"frame_delay_s": 0.15}], indirect=True)
def test_a_reload_finishing_before_a_cancelled_runs_callback_closes_its_dialog(
    app, monkeypatch, small_disc, small_video
):
    """Review finding at 51c1cdf: start a model reload, track and cancel on the old model, let the reload finish before
    the tracking callback arrives. The stale callback was ignored but its dialog stayed open and held the grab."""
    import threading

    reload_release = threading.Event()
    result_ready = threading.Event()
    callback_release = threading.Event()
    state = {}
    cx, cy = small_disc.centers[N - 1]
    box = tuple(int(v) for v in small_disc.box(N - 1))

    def begin_reload():
        old = app.tracking
        new = TrackingService(FakeTracker(small_disc))
        state.update(old=old, new=new)
        real_load = new.load_model
        real_run = old.run

        def held_load():
            assert reload_release.wait(15)
            real_load()

        def held_result(*args, **kwargs):
            result = real_run(*args, **kwargs)
            state["result"] = result
            result_ready.set()
            assert callback_release.wait(15)
            return result

        monkeypatch.setattr(new, "load_model", held_load)
        monkeypatch.setattr(old, "run", held_result)
        monkeypatch.setattr(TrackingService, "create", classmethod(lambda cls, spec, registry=None: new))
        app.load_selected_model()

    def tracking_has_frames():
        return app.tracking_dialog is not None and app.tracking_dialog.progress_var.get() >= 25

    def cancel():
        state["dialog"] = app.tracking_dialog
        app.tracking_dialog.request_cancel()

    def reload_installed():
        return app.tracking is state["new"]

    def check_dialog_closed_by_the_reload():
        # the reload discarded the old backend's run: its dialog is gone before the stale callback even arrives
        assert state["dialog"].dialog is None and app.tracking_dialog is None
        assert not app.tracking_in_progress and app.video_segments is None
        assert app.root.grab_current() is None

    def check_stale_callback():
        assert state["result"].is_cancelled and app.tracking is state["new"]
        assert state["dialog"].dialog is None and app.tracking_dialog is None and app.video_segments is None
        assert str(app.track_btn["state"]) == "disabled"  # no video on the new backend yet

    try:
        (
            _prepare_tracking(app, monkeypatch, small_video, box, (cx - 30, cy - 30))
            .step("reload pending", begin_reload, None, 5)
            .step("tracking has frames", app.start_tracking, tracking_has_frames, 10)
            .step("cancelled result pending delivery", cancel, result_ready.is_set, 10)
            .step("reload installed", reload_release.set, reload_installed, 10)
            .step("dialog closed by the reload", check_dialog_closed_by_the_reload, None, 5)
            .step(
                "stale completion delivered",
                callback_release.set,
                lambda: "Ignored a stale tracking completion" in log_of(app),
                10,
            )
            .step("still closed", check_stale_callback, None, 5)
            .run()
        )
    finally:
        reload_release.set()
        callback_release.set()


def test_a_stale_callback_closes_a_dialog_that_is_still_the_current_reference(
    app, monkeypatch, small_disc, small_video
):
    """The narrower case of the same finding: a stale callback whose dialog is still the window's reference."""
    from organoidtracker.core.tracking_result import TrackingResult
    from organoidtracker.gui_tk.progress_dialog import ProgressDialog

    cx, cy = small_disc.centers[N - 1]
    box = tuple(int(v) for v in small_disc.box(N - 1))

    def stale_with_current_dialog():
        dialog = ProgressDialog(app.root, "stale")
        dialog.show()
        app.tracking_dialog = dialog  # as left behind by a run whose state was discarded elsewhere
        current = app.video_segments
        stale = TrackingResult({0: {1: current[0][1]}}, status="cancelled", frames_total=N, frames_done=1)
        app.on_tracking_complete_success("old-run", app.tracking, dialog, stale, 0.0)
        assert dialog.dialog is None and app.tracking_dialog is None and app.video_segments is current
        assert app.root.grab_current() is None

    (
        _track_in_window(app, monkeypatch, small_video, box, point=(cx - 30, cy - 30))
        .step("stale callback with the current dialog", stale_with_current_dialog, None, 5)
        .run()
    )


# ------------------------------------------------------------------------------------ automatic save
def test_every_tracking_run_is_saved_automatically_and_exports_like_the_window(
    app, monkeypatch, small_disc, small_video, tmp_path
):
    """The run is on disk right after tracking, without a dialog; reopened and exported without a model it gives the
    window's own tables and videos; tracking again saves a second run and leaves the first as it was."""
    from organoidtracker import cli
    from organoidtracker.services.saved_results import load_saved_result

    cx, cy = small_disc.centers[N - 1]
    box1 = tuple(int(v) for v in small_disc.box(N - 1))
    videos_dir, report_dir = workflow(
        app,
        monkeypatch,
        small_video,
        tmp_path / "gui",
        boxes=[((cx - 30, cy - 30), [box1])],
        extra_empty_point=(5, 5),
        time_lapse=14.0,
        conversion=1.6934,
    )
    first_id = app.tracking_run_id
    run_dir = saved_run(app, first_id)
    assert run_dir.parent == tmp_path / "data" / "output_videos" / "runs" and run_dir.name.startswith("disc_")
    assert sorted(p.name for p in run_dir.iterdir() if not p.name.startswith("masks-")) == [
        "prompts.json",
        "results.json",
        "session.json",
    ]
    assert len(list(run_dir.glob("masks-*.npz"))) == 1
    saved = load_saved_result(run_dir)
    assert saved.run_id == first_id and saved.result.status == "completed" and saved.result.frames_done == N
    assert mask_digests(saved.result) == mask_digests(app.video_segments)
    assert saved.session.annotations.organoid_data() == {
        1: {"point": (cx - 30, cy - 30), "cysts": [{"cyst_id": 1, "bbox": box1}]},
        2: {"point": (5, 5), "cysts": []},
    }
    assert saved.session.timing.time_lapse_days == 14.0 and saved.session.calibration.um_per_pixel == 1.6934
    assert saved.video.sha256 == hashlib.sha256(small_video.read_bytes()).hexdigest()
    assert saved.provenance["backend"] == "fake"
    assert f"Run {first_id} saved automatically" in log_of(app) and "❌" not in log_of(app)
    assert len(list((tmp_path / "data" / "output_videos" / "prompts").glob(f"*_{first_id}.json"))) == 1

    # the public consumer: the command line exports the saved run without a model, as the window exported it
    out = tmp_path / "exported"
    assert cli.main(["export", "--run", str(run_dir), "--out", str(out), "--log-level", "WARNING"]) == 0
    for name in ("raw_cyst_data.csv", "cyst_summary.csv", "organoid_summary.csv"):
        assert (out / name).read_bytes() == (report_dir / name).read_bytes(), name
    for name in ("multi_object_overlay.mp4", "multi_object_mask.mp4", "multi_object_side_by_side.mp4"):
        assert (out / "videos" / name).stat().st_size == (videos_dir / name).stat().st_size, name
    # the command the window prints, as printed (review finding: it lacked --out and exited 2)
    (hint,) = [line for line in log_of(app).splitlines() if f'--run "{run_dir}"' in line]
    command = shlex.split(hint.split("without a model: ", 1)[1])
    assert command[:2] == ["organoidtracker", "export"] and command[-1] == "DIR"
    assert cli.main([*command[1:-1], str(tmp_path / "as-printed"), "--log-level", "WARNING"]) == 0
    assert (tmp_path / "as-printed" / "cyst_summary.csv").read_bytes() == (report_dir / "cyst_summary.csv").read_bytes()

    first_files = {p.name: p.read_bytes() for p in run_dir.iterdir()}

    def tracked_again():
        return not app.tracking_in_progress and app.tracking_run_id != first_id and not app.autosaves_in_progress

    TkDriver(app.root).step("tracked again", app.start_tracking, tracked_again, 60).run()
    second = saved_run(app, app.tracking_run_id)
    assert saved_runs(app) == sorted([run_dir, second])
    assert load_saved_result(second).run_id == app.tracking_run_id
    assert {p.name: p.read_bytes() for p in run_dir.iterdir()} == first_files


@pytest.mark.parametrize("app", [{"partial_after": 3}], indirect=True)
def test_a_partial_run_is_saved_as_partial(app, monkeypatch, small_disc, small_video, tmp_path):
    from organoidtracker import cli
    from organoidtracker.services.saved_results import load_saved_result

    cx, cy = small_disc.centers[N - 1]
    box = tuple(int(v) for v in small_disc.box(N - 1))
    _track_in_window(app, monkeypatch, small_video, box, point=(cx - 30, cy - 30)).run()
    saved = load_saved_result(saved_run(app, app.tracking_run_id))
    assert saved.result.status == "partial" and saved.result.frames_done == 3
    assert saved.result.tracked_frames == [N - 1, N - 2, N - 3]
    assert mask_digests(saved.result) == mask_digests(app.video_segments)
    out = tmp_path / "exported"
    assert cli.main(["export", "--run", str(saved.directory), "--out", str(out), "--log-level", "ERROR"]) == 3


@pytest.mark.parametrize("app", [{"frame_delay_s": 0.2}], indirect=True)
def test_a_cancelled_run_is_saved_as_cancelled(app, monkeypatch, small_disc, small_video, tmp_path):
    from organoidtracker import cli
    from organoidtracker.services.saved_results import load_saved_result

    cx, cy = small_disc.centers[N - 1]
    box = tuple(int(v) for v in small_disc.box(N - 1))

    def two_frames_done():
        dialog = app.tracking_dialog
        return dialog is not None and dialog.dialog is not None and dialog.progress_var.get() >= 100 * 2 / N

    def cancelled_and_saved():
        return (
            not app.tracking_in_progress and "cancelled" in app.status_label["text"] and not app.autosaves_in_progress
        )

    (
        _prepare_tracking(app, monkeypatch, small_video, box, (cx - 30, cy - 30))
        .step("tracking started", app.start_tracking, two_frames_done, 30)
        .step("cancelled", app.cancel_tracking, cancelled_and_saved, 30)
        .run()
    )
    result = app.video_segments
    assert result.status == "cancelled" and 2 <= result.frames_done < N
    saved = load_saved_result(saved_run(app, app.tracking_run_id))
    assert saved.result.status == "cancelled" and saved.result.frames_done == result.frames_done
    assert mask_digests(saved.result) == mask_digests(result)
    out = tmp_path / "exported"
    assert cli.main(["export", "--run", str(saved.directory), "--out", str(out), "--log-level", "ERROR"]) == 4


@pytest.mark.parametrize("failure", ["mask file write", "run root is a file"])
def test_a_failed_automatic_save_leaves_no_run_and_keeps_the_results(
    app, monkeypatch, small_disc, small_video, tmp_path, failure
):
    """The results stay shown and exportable, a warning names the error, nothing half-written remains, the status is
    the tracking's; Save Results then saves the run."""
    from organoidtracker.gui_tk import main_window
    from organoidtracker.services import saved_results
    from organoidtracker.services.saved_results import load_saved_result

    cx, cy = small_disc.centers[N - 1]
    box = tuple(int(v) for v in small_disc.box(N - 1))
    root = Path(app.run_root).resolve()
    original_write_npz = saved_results._write_npz
    app.log_handler.setLevel(logging.ERROR)  # review finding: the warning went only through the panel's handler
    if failure == "mask file write":
        monkeypatch.setattr(
            saved_results,
            "_write_npz",
            lambda handle, arrays: (_ for _ in ()).throw(OSError("injected disk-full error")),
        )
        expected = "injected disk-full error"
    else:
        root.parent.mkdir(parents=True)
        root.write_text("a file where the run directories go\n")
        expected = "cannot create the run directory"
    save_dir = tmp_path / "saved"

    def check_results_kept():
        assert app.video_segments is not None and app.video_segments.is_complete
        for button in (app.track_btn, app.generate_btn, app.analysis_btn, app.save_results_btn):
            assert str(button["state"]) == "normal", button
        assert "Tracking completed!" in app.status_label["text"]
        if failure == "mask file write":
            assert root.is_dir() and saved_runs(app) == []  # the new directory was removed again
            assert not list(root.rglob("*"))
        else:
            assert root.read_text() == "a file where the run directories go\n"
        monkeypatch.setattr(saved_results, "_write_npz", original_write_npz)

    def save():
        monkeypatch.setattr(main_window.filedialog, "askdirectory", lambda *a, **k: str(save_dir))
        app.save_results()

    (
        _track_in_window(app, monkeypatch, small_video, box, point=(cx - 30, cy - 30))
        .step("warning shown", None, lambda: "was not saved automatically" in log_of(app), 10)
        .step("results kept", check_results_kept, None, 5)
        .step("saved by hand", save, lambda: "Results saved" in app.status_label["text"], 30)
        .run()
    )
    assert expected in log_of(app) and "Use Save Results... to save it." in log_of(app)
    assert mask_digests(load_saved_result(save_dir).result) == mask_digests(app.video_segments)


@pytest.mark.parametrize("conversion", [0.0, "not a number"])
def test_invalid_analysis_inputs_skip_the_automatic_save(
    app, monkeypatch, small_disc, small_video, tmp_path, conversion
):
    """Nothing is saved and a warning says why; once the input is corrected, Save Results saves the run."""
    from organoidtracker.gui_tk import main_window
    from organoidtracker.services.saved_results import load_saved_result

    cx, cy = small_disc.centers[N - 1]
    box = tuple(int(v) for v in small_disc.box(N - 1))
    save_dir = tmp_path / "saved"
    app.conversion_factor_var.set(conversion)

    def correct_and_save():
        assert saved_runs(app) == [] and str(app.save_results_btn["state"]) == "normal"
        assert log_of(app).count("was not saved automatically") == 1
        monkeypatch.setattr(main_window.filedialog, "askdirectory", lambda *a, **k: str(save_dir))
        app.save_results()  # still invalid: refused before anything is written
        assert "Cannot save:" in app.status_label["text"] and "❌ Cannot save the results" in log_of(app)
        assert not save_dir.exists()
        app.conversion_factor_var.set(1.6934)
        monkeypatch.setattr(main_window.filedialog, "askdirectory", lambda *a, **k: str(save_dir))
        app.save_results()

    (
        _track_in_window(app, monkeypatch, small_video, box, point=(cx - 30, cy - 30))
        .step("warning shown", None, lambda: "was not saved automatically" in log_of(app), 10)
        .step("saved after the correction", correct_and_save, lambda: "Results saved" in app.status_label["text"], 30)
        .run()
    )
    reason = (
        "um_per_pixel: expected a positive number" if conversion == 0.0 else "check the time lapse and the conversion"
    )
    assert reason in log_of(app)
    saved = load_saved_result(save_dir)
    assert saved.session.calibration.um_per_pixel == 1.6934 and saved.run_id == app.tracking_run_id
    assert saved_runs(app) == []


def test_the_automatic_save_writes_its_own_run_while_the_window_moves_on(
    app, monkeypatch, small_disc, small_video, tmp_path
):
    """The save is held in its worker while another video is loaded; released, it writes the first run (its video and
    masks) and changes nothing in the window. Its thread is not a daemon, so closing the window lets it finish."""
    import threading

    from organoidtracker.gui_tk import main_window
    from organoidtracker.services import export_service
    from organoidtracker.services.saved_results import load_saved_result

    other_video = DiscVideo(n_frames=N, size=128, radius=14, start=30, step=10).write(tmp_path / "other.mp4")
    cx, cy = small_disc.centers[N - 1]
    box = tuple(int(v) for v in small_disc.box(N - 1))
    entered, release = threading.Event(), threading.Event()
    original_stage = export_service.stage_saved_result

    def held_stage(*args, **kwargs):
        entered.set()
        release.wait(30)
        return original_stage(*args, **kwargs)

    monkeypatch.setattr(export_service, "stage_saved_result", held_stage)
    first = {}

    def tracked_and_held():
        return not app.tracking_in_progress and app.video_segments is not None and entered.is_set()

    def check_held_and_load_another_video():
        first["run_id"] = app.tracking_run_id
        first["digests"] = mask_digests(app.video_segments)
        assert app.autosaves_in_progress == {first["run_id"]}
        (thread,) = [t for t in threading.enumerate() if t.name == f"autosave-{first['run_id']}"]
        assert thread.daemon is False
        monkeypatch.setattr(main_window.filedialog, "askopenfilename", lambda *a, **k: str(other_video))
        app.load_video()

    def other_video_loaded():
        return app.current_video_path == str(other_video) and str(app.track_btn["state"]) == "normal"

    try:
        (
            _prepare_tracking(app, monkeypatch, small_video, box, (cx - 30, cy - 30))
            .step("tracked, save held", app.start_tracking, tracked_and_held, 60)
            .step("another video loaded", check_held_and_load_another_video, other_video_loaded, 30)
            .step("save released", release.set, lambda: not app.autosaves_in_progress, 30)
            .run()
        )
    finally:
        release.set()
    saved = load_saved_result(saved_run(app, first["run_id"]))
    assert saved.run_id == first["run_id"] and mask_digests(saved.result) == first["digests"]
    assert saved.video.sha256 == hashlib.sha256(small_video.read_bytes()).hexdigest()
    assert Path(saved.session.video.path).resolve() == small_video.resolve()
    record = json.loads((saved.directory / "prompts.json").read_text())
    assert Path(record["video"]["path"]).resolve() == small_video.resolve()
    assert record["video"]["sha256"] == saved.video.sha256 and [o["organoid_id"] for o in record["organoids"]] == [1]
    assert list(record["prompts"]) == ["1"] and len(record["prompts"]["1"]) == 1
    assert app.video_segments is None and app.current_video_path == str(other_video)
    assert "Video loaded!" in app.status_label["text"]  # the save's completion did not touch the status
    assert f"Run {first['run_id']} saved automatically" in log_of(app)


def test_tracking_is_refused_while_a_video_loads(app, monkeypatch, small_disc, small_video, tmp_path):
    """Review finding: Track stayed enabled while a video loaded, so a run could share the backend with the loader and
    be saved under the new video. Track is now disabled and refused until the video has loaded."""
    import threading

    from organoidtracker.gui_tk import main_window

    other_video = DiscVideo(n_frames=N, size=128, radius=14, start=30, step=10).write(tmp_path / "other.mp4")
    cx, cy = small_disc.centers[N - 1]
    box = tuple(int(v) for v in small_disc.box(N - 1))
    release = threading.Event()
    original_open = TrackingService.open_video

    def held_open(service, path):
        if Path(path) == other_video:
            release.wait(30)
        return original_open(service, path)

    monkeypatch.setattr(TrackingService, "open_video", held_open)
    first = {}

    def load_another_video_and_track():
        first["run_id"] = app.tracking_run_id
        monkeypatch.setattr(main_window.filedialog, "askopenfilename", lambda *a, **k: str(other_video))
        app.load_video()
        assert app.video_loading and str(app.track_btn["state"]) == "disabled"
        app.start_tracking()  # what the button would do if it were pressed
        assert not app.tracking_in_progress and "A video is loading" in app.status_label["text"]
        release.set()

    def other_video_loaded():
        return app.current_video_path == str(other_video) and not app.video_loading

    try:
        (
            _track_in_window(app, monkeypatch, small_video, box, point=(cx - 30, cy - 30))
            .step("track refused while loading", load_another_video_and_track, other_video_loaded, 30)
            .run()
        )
    finally:
        release.set()
    assert app.tracking_run_id is None and app.video_segments is None
    assert saved_runs(app) == [saved_run(app, first["run_id"])] and app.autosaves_in_progress == set()
    assert "Cannot start tracking - a video is loading" in log_of(app)


@pytest.mark.parametrize("app", [{"frame_delay_s": 0.2}], indirect=True)
def test_a_run_whose_video_was_replaced_while_it_ran_is_discarded(app, monkeypatch, small_disc, small_video):
    """Defense in depth for the same finding: a completion whose backend holds another video by then is neither
    installed nor saved."""
    import dataclasses

    cx, cy = small_disc.centers[N - 1]
    box = tuple(int(v) for v in small_disc.box(N - 1))

    def one_frame_done():
        dialog = app.tracking_dialog
        return dialog is not None and dialog.dialog is not None and dialog.progress_var.get() >= 100 / N

    def replace_the_video():  # what a concurrent load would have done to the backend
        app.tracking.video = dataclasses.replace(app.tracking.video)

    (
        _prepare_tracking(app, monkeypatch, small_video, box, (cx - 30, cy - 30))
        .step("tracking", app.start_tracking, one_frame_done, 30)
        .step("finished", replace_the_video, lambda: not app.tracking_in_progress, 30)
        .run()
    )
    assert app.video_segments is None and app.result_video is None and app.autosaves_in_progress == set()
    assert "the video changed while it was tracked" in log_of(app) and saved_runs(app) == []
    assert str(app.save_results_btn["state"]) == "disabled" and str(app.track_btn["state"]) == "normal"


def test_the_saved_run_follows_the_inputs_of_its_analysis_report(app, monkeypatch, small_disc, small_video, tmp_path):
    """Review finding: the analysis inputs are usually set after tracking, so the saved run held those of the end of
    tracking and exporting it disagreed with the window's report. The report now rewrites the saved run with the
    inputs and organoids it used (the mask file stays); Save Results refuses the run's own directory."""
    from organoidtracker import cli
    from organoidtracker.gui_tk import main_window
    from organoidtracker.services.saved_results import load_saved_result

    cx, cy = small_disc.centers[N - 1]
    box = tuple(int(v) for v in small_disc.box(N - 1))
    first_report, second_report = tmp_path / "report-1", tmp_path / "report-2"
    first_report.mkdir()
    second_report.mkdir()
    app.time_lapse_var.set(6.0)  # the values when tracking ends
    app.conversion_factor_var.set(1.0)
    state = {}

    def check_first_save():
        state["run_dir"] = run_dir = saved_run(app, app.tracking_run_id)
        state["masks"] = sorted(p.name for p in run_dir.glob("masks-*.npz"))
        saved = load_saved_result(run_dir)
        assert saved.session.timing.time_lapse_days == 6.0 and saved.session.calibration.um_per_pixel == 1.0

    def report_into(directory, time_lapse):
        def request():
            app.time_lapse_var.set(time_lapse)
            app.conversion_factor_var.set(2.0)
            monkeypatch.setattr(main_window.filedialog, "askdirectory", lambda *a, **k: str(directory))
            app.generate_analysis_report()

        return request

    def add_empty_organoid_and_report():
        app.on_canvas_click(5, 5)  # an organoid without cysts, added after tracking
        report_into(first_report, 14.0)()

    def reports_done(count):
        def done():
            return (
                str(app.analysis_btn["state"]) == "normal"
                and log_of(app).count("now holds the inputs of the analysis report") == count
                and not app.autosaves_in_progress
            )

        return done

    def check_first_update_and_refuse_save_results():
        run_dir = state["run_dir"]
        saved = load_saved_result(run_dir)
        assert saved.session.timing.time_lapse_days == 14.0 and saved.session.calibration.um_per_pixel == 2.0
        assert saved.session.annotations.organoid_data() == {
            1: {"point": (cx - 30, cy - 30), "cysts": [{"cyst_id": 1, "bbox": box}]},
            2: {"point": (5, 5), "cysts": []},
        }
        assert sorted(p.name for p in run_dir.glob("masks-*.npz")) == state["masks"]  # the masks did not change
        assert mask_digests(saved.result) == mask_digests(app.video_segments) and saved.run_id == app.tracking_run_id
        record = json.loads((run_dir / "prompts.json").read_text())
        assert record["analysis_inputs"]["time_lapse_days"] == 14.0 and len(record["organoids"]) == 2
        out = tmp_path / "exported"
        assert (
            cli.main(["export", "--run", str(run_dir), "--out", str(out), "--no-videos", "--log-level", "WARNING"]) == 0
        )
        for name in ("raw_cyst_data.csv", "cyst_summary.csv", "organoid_summary.csv"):
            assert (out / name).read_bytes() == (first_report / name).read_bytes(), name
        state["files"] = {p.name: p.read_bytes() for p in run_dir.iterdir()}
        monkeypatch.setattr(main_window.filedialog, "askdirectory", lambda *a, **k: str(run_dir))
        app.save_results()
        assert "run's own directory" in app.status_label["text"]
        assert {p.name: p.read_bytes() for p in run_dir.iterdir()} == state["files"]

    (
        _track_in_window(app, monkeypatch, small_video, box, point=(cx - 30, cy - 30))
        .step("first save", check_first_save, None, 5)
        .step("first report", add_empty_organoid_and_report, reports_done(1), 120)
        .step("saved run updated, Save Results refused", check_first_update_and_refuse_save_results, None, 30)
        .step("second report", report_into(second_report, 10.0), reports_done(2), 120)
        .run()
    )
    assert load_saved_result(state["run_dir"]).session.timing.time_lapse_days == 10.0  # the latest report's inputs


def test_a_report_requested_while_the_run_is_being_saved_updates_it_afterwards(
    app, monkeypatch, small_disc, small_video, tmp_path
):
    """One write per run directory at a time: the update waits for the first save, then the report's inputs win."""
    import threading

    from organoidtracker.gui_tk import main_window
    from organoidtracker.services import export_service
    from organoidtracker.services.saved_results import load_saved_result

    cx, cy = small_disc.centers[N - 1]
    box = tuple(int(v) for v in small_disc.box(N - 1))
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    entered, release = threading.Event(), threading.Event()
    original_stage = export_service.stage_saved_result
    calls = []

    def first_save_held(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            entered.set()
            release.wait(30)
        return original_stage(*args, **kwargs)

    monkeypatch.setattr(export_service, "stage_saved_result", first_save_held)
    app.time_lapse_var.set(6.0)

    def tracked_and_held():
        return not app.tracking_in_progress and app.video_segments is not None and entered.is_set()

    def report_while_held():
        app.time_lapse_var.set(14.0)
        monkeypatch.setattr(main_window.filedialog, "askdirectory", lambda *a, **k: str(report_dir))
        app.generate_analysis_report()
        assert app.tracking_run_id in app._run_directory_updates and len(calls) == 1  # waiting, not writing

    def report_done():
        return str(app.analysis_btn["state"]) == "normal" and "analysis report generated successfully" in log_of(app)

    def both_written():
        return not app.autosaves_in_progress and "now holds the inputs of the analysis report" in log_of(app)

    try:
        (
            _prepare_tracking(app, monkeypatch, small_video, box, (cx - 30, cy - 30))
            .step("tracked, save held", app.start_tracking, tracked_and_held, 60)
            .step("report while the save is held", report_while_held, report_done, 120)
            .step("save released", release.set, both_written, 30)
            .run()
        )
    finally:
        release.set()
    log = log_of(app)
    assert log.index("saved automatically") < log.index("now holds the inputs of the analysis report")
    assert len(calls) == 2
    assert load_saved_result(saved_run(app, app.tracking_run_id)).session.timing.time_lapse_days == 14.0
