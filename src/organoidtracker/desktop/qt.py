"""Run the existing Qt workflow with persistent container paths and device policy."""

import json
import os
import sys
import time
from pathlib import Path

from PySide6.QtCore import QSettings, QTimer
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import QApplication

from ..gui_qt.signals import StopBridge
from ..gui_qt.window import MainWindow
from ..logging_config import configure_logging
from ..services.tracking_service import TrackingService
from .checks import check_device
from .models import validate_checkpoint

RUNTIME = Path("/tmp/organoidtracker")


class ContainerWindow(MainWindow):
    def __init__(self, mode, **kwargs):
        self.mode = mode
        preferences = QSettings("/preferences/desktop.ini", QSettings.Format.IniFormat)
        if preferences.value("run_root") is None:
            preferences.setValue("run_root", "/results")
        if preferences.value("last_video_dir") is None:
            preferences.setValue("last_video_dir", "/videos")
        super().__init__(settings=preferences, tracking_factory=self.create_tracker, output_root="/results", **kwargs)
        self.run_root = self.output_path(self.run_root)
        self.backend.setCurrentIndex(self.backend.findData("sam2"))
        index = self.backend.findData("hf-sam2")
        self.backend.setItemText(index, "SAM 2 Transformers (not included)")
        model = self.backend.model()
        assert isinstance(model, QStandardItemModel)
        model.item(index).setEnabled(False)
        self.device.setCurrentText(mode)
        self.device.setEnabled(False)
        self.checkpoint_timer = QTimer(self)
        self.checkpoint_timer.timeout.connect(self.update_checkpoint_status)
        self.checkpoint_timer.start(2000)

    def create_tracker(self, spec):
        if spec.backend != "sam2" or spec.device != self.mode:
            raise ValueError(f"This image supports the reference SAM 2 backend on {self.mode}.")
        check_device(self.mode)
        service = TrackingService.create(spec)
        checkpoint = getattr(service.tracker, "checkpoint_path", None)
        if checkpoint is None:
            raise ValueError("Select a checkpoint before tracking.")
        validate_checkpoint(Path(checkpoint))
        return service

    def apply_session_fields(self, session):
        super().apply_session_fields(session)
        # Saved provenance is retained in the saved result. A new run uses the
        # deployment device selected visibly by the recipient at launch.
        self.device.setCurrentText(self.mode)

    def open_report_folder(self):
        if self.opened and self.opened.saved:
            relative = self.opened.saved.directory.resolve().relative_to(Path("/results"))
            value = {"directory": str(relative), "sequence": time.time_ns()}
            temporary = RUNTIME / "report.tmp"
            temporary.write_text(json.dumps(value), encoding="utf-8")
            temporary.replace(RUNTIME / "report.json")


def main():
    configure_logging(log_file="/results/organoidtracker.log")
    app = QApplication(sys.argv)
    app.setApplicationName("Organoid Tracker")
    window = ContainerWindow(os.environ["OT_DEVICE"])
    bridge = StopBridge(window)
    bridge.install()
    window.showMaximized()
    RUNTIME.mkdir(exist_ok=True)
    QTimer.singleShot(0, lambda: (RUNTIME / "ready").write_text(str(os.getpid())))
    try:
        return app.exec()
    finally:
        (RUNTIME / "ready").unlink(missing_ok=True)
        bridge.restore()


if __name__ == "__main__":
    raise SystemExit(main())
