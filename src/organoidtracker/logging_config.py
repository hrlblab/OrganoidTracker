"""Logging setup for the application entry points.

Modules log through ``logging.getLogger(__name__)``. An entry point (the Tk launcher, later
the CLI) calls :func:`configure_logging` once: records go to the console and, when an output
directory is writable, to a rotating log file next to the outputs
(``data/output_videos/organoidtracker.log`` by default). The GUI adds its own handler for
its log panel.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

CONSOLE_FORMAT = "%(levelname)-8s %(message)s"
FILE_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DEFAULT_LOG_FILE = Path("data") / "output_videos" / "organoidtracker.log"


def configure_logging(level: str | int | None = None, log_file: Path | str | None = DEFAULT_LOG_FILE) -> Path | None:
    """Configure the root logger for a process.

    ``level`` defaults to ``config.LOG_LEVEL``. ``log_file`` is created (with its directory)
    when possible; ``None`` disables the file. Returns the log file path in use, or None.
    Calling this twice replaces the handlers installed by the first call.
    """
    if level is None:
        from . import config

        level = getattr(config, "LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)

    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_organoidtracker", False):
            root.removeHandler(handler)
            handler.close()
    root.setLevel(min(level, logging.INFO) if log_file else level)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT))
    console._organoidtracker = True  # type: ignore[attr-defined]
    root.addHandler(console)

    if log_file is None:
        return None
    path = Path(log_file)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    except OSError as error:
        logging.getLogger(__name__).warning("Log file %s not writable (%s); logging to the console only", path, error)
        return None
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
    file_handler._organoidtracker = True  # type: ignore[attr-defined]
    root.addHandler(file_handler)
    return path
