#!/usr/bin/env python3
"""
Compatibility launcher for the Tk desktop application.

The application is installed as the ``organoidtracker`` package; its console script is
``organoidtracker-tk`` (``uv run organoidtracker-tk``). This file keeps the documented
``python video_tracker_gui.py`` working inside that environment.
"""

import sys

try:
    from organoidtracker.gui_tk.launcher import main
except ImportError as error:
    sys.stderr.write(
        f"The organoidtracker package is not installed in this interpreter ({error}).\n"
        "Install it with `uv sync --extra tk` (or `pip install -e .`) and run `organoidtracker-tk`.\n"
    )
    sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())
