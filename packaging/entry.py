"""Frozen entry point: Qt by default, the headless CLI with --cli."""

import os
import sys
import tempfile
from pathlib import Path


def main():
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    if sys.argv[1:2] == ["--cli"]:
        from organoidtracker.cli import main as cli_main

        return cli_main(sys.argv[2:])
    if sys.argv[1:2] == ["--smoke-test"]:
        from PySide6.QtCore import QSettings, QTimer
        from PySide6.QtWidgets import QApplication

        from organoidtracker.gui_qt.window import MainWindow

        app = QApplication([])
        with tempfile.TemporaryDirectory(prefix="organoidtracker-smoke-") as directory:
            settings = QSettings(str(Path(directory) / "preferences.ini"), QSettings.Format.IniFormat)
            window = MainWindow(settings=settings)
            window.show()
            QTimer.singleShot(150, app.quit)
            code = app.exec()
            window.close()
            settings.sync()
        print("Qt window initialized and closed successfully", flush=True)
        return code
    from organoidtracker.gui_qt.launcher import main as qt_main

    return qt_main()


if __name__ == "__main__":
    raise SystemExit(main())
