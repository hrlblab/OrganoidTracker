"""Qt coordinate and lifecycle checks use synthetic videos and an isolated settings file."""

import os
from dataclasses import replace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("pytestqt", reason="install the qt extra and qt-test dependency group")
pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, QSettings, Qt  # noqa: E402

from helpers import FakeTracker  # noqa: E402
from organoidtracker.gui_qt.canvas import AnnotationCanvas  # noqa: E402
from organoidtracker.gui_qt.window import MainWindow  # noqa: E402
from organoidtracker.gui_qt.worker import preview_session  # noqa: E402
from organoidtracker.services.annotations import AnnotationSet  # noqa: E402
from organoidtracker.services.export_service import ExportError, ExportService  # noqa: E402
from organoidtracker.services.run_manifest import mask_digests  # noqa: E402
from organoidtracker.services.saved_results import load_saved_result  # noqa: E402
from organoidtracker.services.session import load_session  # noqa: E402
from organoidtracker.services.tracking_service import TrackingService  # noqa: E402
from test_services import make_session, run  # noqa: E402
from test_services import small_disc as small_disc  # noqa: E402
from test_services import small_video as small_video  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def window(qtbot, tmp_path, small_disc):
    settings = QSettings(str(tmp_path / "preferences.ini"), QSettings.Format.IniFormat)
    window = MainWindow(
        settings=settings,
        tracking_factory=lambda spec: TrackingService(
            FakeTracker(small_disc, enable_reverse_tracking=spec.reverse, frame_delay_s=0.03)
        ),
    )
    window.run_root = tmp_path / "实验 runs"
    window.videos.setChecked(False)
    qtbot.addWidget(window)
    window.show()
    yield window
    if window.busy:
        window.cancel_tracking()
        qtbot.waitUntil(lambda: not window.busy, timeout=60000)
    window.close()


def opened(window, small_disc, small_video):
    session = make_session(
        small_disc, small_video, extra_empty=True, timing={"frame_times_days": [1, 1.2, 1.7, 2.8, 3, 4.4, 5.9, 8]}
    )
    window.use_opened(preview_session(session))
    return session


@pytest.mark.parametrize("zoom", [0.25, 1, 3])
def test_canvas_source_coordinates_and_mouse_gestures(qtbot, zoom):
    canvas = AnnotationCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(700, 700)
    canvas.show()
    canvas.set_frame(np.zeros((256, 256, 3), dtype=np.uint8), fit=True)
    canvas.resetTransform()
    canvas.zoom(zoom)
    canvas.editable = True
    point = QPointF(128, 128)
    assert (canvas.source_point(canvas.mapFromScene(point)) - point).manhattanLength() <= 2 / zoom
    with qtbot.waitSignal(canvas.organoid_clicked) as signal:
        qtbot.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=canvas.mapFromScene(point))
    assert signal.args == pytest.approx([128, 128], abs=1 / zoom)
    start, end = canvas.mapFromScene(QPointF(80, 80)), canvas.mapFromScene(QPointF(180, 180))
    with qtbot.waitSignal(canvas.cyst_drawn) as signal:
        qtbot.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, pos=start)
        qtbot.mouseMove(canvas.viewport(), end)
        qtbot.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton, pos=end)
    assert signal.args == pytest.approx([80, 80, 180, 180], abs=1 / zoom)


def test_annotations_undo_and_shorter_reopen(window, small_disc, small_video, qtbot):
    session = opened(window, small_disc, small_video)
    window.use_opened(replace(window.opened, session=replace(session, annotations=AnnotationSet(()))))
    window.add_cyst(10, 10, 20, 20)
    assert not window.annotations.cysts
    window.add_organoid(20, 20)
    window.add_cyst(10, 10, 30, 30)
    window.add_organoid(50, 50)  # empty organoids stay in the denominator
    assert len(window.annotations.organoids) == 2 and len(window.annotations.cysts) == 1
    window.undo_stack.undo()
    assert len(window.annotations.organoids) == 1
    window.undo_stack.undo()
    assert not window.annotations.cysts
    window.undo_stack.redo()
    assert len(window.annotations.cysts) == 1
    one = replace(
        window.opened,
        source=replace(window.opened.source, frame_map=(0,), annotation_frame=0),
        frames=window.opened.frames[:1],
    )
    window.use_opened(one)
    assert window.slider.maximum() == 0 and window.slider.value() == 0


def test_qt_run_equals_cli_reopens_and_exports_new_analysis(
    window, small_disc, small_video, qtbot, tmp_path, monkeypatch
):
    from PySide6.QtWidgets import QFileDialog

    session = opened(window, small_disc, small_video)
    reference = run(session, tmp_path / "reference", small_disc)
    window.start_tracking()
    qtbot.waitUntil(lambda: not window.busy, timeout=60000)
    saved = window.opened.saved
    assert saved is not None and saved.result.status == "completed"
    assert mask_digests(saved.result) == mask_digests(reference.result)
    assert (saved.directory / "raw_cyst_data.csv").read_bytes() == (
        reference.output_dir / "raw_cyst_data.csv"
    ).read_bytes()
    original = saved.path.read_bytes()
    window._begin("open", path=saved.path)
    qtbot.waitUntil(lambda: not window.busy, timeout=15000)
    assert window.opened.saved.run_id == saved.run_id and not window.canvas.editable
    window.calibration.setText("2.000123456789123")
    window.explicit_times.setText("1, 2, 3, 5, 7, 9, 10, 12")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args: str(tmp_path / "copies"))
    window.export_copy()
    qtbot.waitUntil(lambda: not window.busy, timeout=60000)
    copied = load_saved_result(window.opened.saved.path)
    assert copied.path != saved.path and saved.path.read_bytes() == original
    assert copied.session.calibration.um_per_pixel == 2.000123456789123
    assert copied.session.timing.frame_times_days == (1, 2, 3, 5, 7, 9, 10, 12)
    assert load_session(copied.directory / "session.json").calibration == copied.session.calibration
    assert load_session(copied.directory / "prompts.json").timing == copied.session.timing
    assert mask_digests(copied.result) == mask_digests(saved.result)
    # Reopening and exporting the copy again must retain the changed measurements.
    raw = (copied.directory / "raw_cyst_data.csv").read_bytes()
    window._begin("open", path=copied.path)
    qtbot.waitUntil(lambda: not window.busy, timeout=15000)
    window.export_copy()
    qtbot.waitUntil(lambda: not window.busy, timeout=60000)
    assert (window.opened.saved.directory / "raw_cyst_data.csv").read_bytes() == raw


def test_cancel_close_and_reopen_preserve_partial_status(window, small_disc, small_video, qtbot):
    opened(window, small_disc, small_video)
    window.tracking_factory = lambda spec: TrackingService(FakeTracker(small_disc, frame_delay_s=0.1))
    window.start_tracking()
    qtbot.waitUntil(lambda: window.progress.maximum() == 8 and window.progress.value() >= 1, timeout=15000)
    window.close()  # waits cooperatively instead of destroying a live QThread
    assert window.busy and window._closing
    qtbot.waitUntil(lambda: not window.busy, timeout=60000)
    saved = load_saved_result(window.opened.saved.path)
    assert saved.result.status == "cancelled" and 0 < saved.result.frames_done < 8
    assert not window.isVisible()


def test_export_failure_recovers_masks_and_failed_model_keeps_previous_state(
    window, small_disc, small_video, qtbot, monkeypatch
):
    opened(window, small_disc, small_video)

    def fail(*args, **kwargs):
        raise ExportError("injected report failure")

    monkeypatch.setattr(ExportService, "write_report", fail)
    window.start_tracking()
    qtbot.waitUntil(lambda: not window.busy, timeout=15000)
    saved = window.opened.saved
    assert saved is not None and saved.path.is_file()
    assert "injected report failure" in window.banner.text() and "Masks were saved" in window.banner.text()
    window.edit_saved_prompts()
    window.tracking_factory = lambda spec: TrackingService(FakeTracker(small_disc, fail_load=True))
    window.start_tracking()
    qtbot.waitUntil(lambda: not window.busy, timeout=15000)
    assert window.opened.saved is None and window.annotations.cysts
    assert load_saved_result(saved.path).result.status == "completed"


def test_stale_progress_and_invalid_analysis_cannot_start_work(window, small_disc, small_video):
    opened(window, small_disc, small_video)
    text = window.banner.text()
    window._progress("old-token", "tracking", 7, 8, "stale")
    assert window.banner.text() == text
    window.explicit_times.setText("1, 2")
    with pytest.raises(ValueError, match="unique frames"):
        window.start_tracking()
    assert not window.busy and not window.run_root.exists()


def test_missing_video_relocation_checks_content(window, small_disc, small_video, qtbot, tmp_path):
    import shutil

    session = opened(window, small_disc, small_video)
    source = tmp_path / "原始.mp4"
    shutil.copyfile(small_video, source)
    saved = run(session.with_video(source), tmp_path / "run", small_disc).saved_result
    moved = tmp_path / "新位置.mp4"
    source.rename(moved)
    window._begin("open", path=saved.path)
    qtbot.waitUntil(lambda: not window.busy, timeout=15000)
    assert not window.opened.frames and not window.videos.isChecked()
    assert "Source video unavailable" in window.banner.text()
    wrong = tmp_path / "wrong.mp4"
    wrong.write_bytes(b"wrong content")
    window._begin("relocate", saved=saved, path=wrong)
    qtbot.waitUntil(lambda: not window.busy, timeout=15000)
    assert not window.opened.frames and "does not match" in window.banner.text()
    window._begin("relocate", saved=saved, path=moved)
    qtbot.waitUntil(lambda: not window.busy, timeout=15000)
    assert len(window.opened.frames) == 8 and window.current_session().video.path == moved


def test_cancel_before_first_mask_can_retry(window, small_disc, small_video, qtbot):
    opened(window, small_disc, small_video)
    window.tracking_factory = lambda spec: TrackingService(FakeTracker(small_disc, start_delay_s=0.15))
    window.start_tracking()
    window.cancel_tracking()
    qtbot.waitUntil(lambda: not window.busy, timeout=15000)
    assert window.opened.saved is None
    assert "Cancelled before any masks" in window.banner.text()
    window.start_tracking()
    qtbot.waitUntil(lambda: not window.busy, timeout=60000)
    assert window.opened.saved.result.status == "completed"
