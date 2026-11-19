"""Launch the optional PySide6 desktop application."""

import logging
import os
import sys


def main() -> int:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("Install the Qt interface with: uv sync --extra qt", file=sys.stderr)
        return 2
    from ..logging_config import configure_logging
    from ..settings import SettingsError

    try:
        from .. import config

        configure_logging(getattr(config, "LOG_LEVEL", "INFO"))
        from .window import MainWindow

        app = QApplication(sys.argv)
        app.setApplicationName("Organoid Tracker")
        window = MainWindow()
        window.show()
        return app.exec()
    except SettingsError as error:
        logging.error("Invalid settings: %s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
