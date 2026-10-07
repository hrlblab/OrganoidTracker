"""Locations the application reads from.

Outputs under ``data/`` have always been relative to the working directory and remain so.
The checkpoints directory and the optional ``user_config.py`` used to be located relative to
the source tree; with an installed package they are resolved here, in this order:

1. an explicit environment variable (``ORGANOIDTRACKER_CHECKPOINTS``,
   ``ORGANOIDTRACKER_USER_CONFIG``),
2. the current working directory (``./checkpoints``, ``./user_config.py``), which is the
   documented way of running the application from a clone of the repository,
3. the root of the source checkout, when the package is installed in editable mode.
"""

from __future__ import annotations

import os
from pathlib import Path


def source_checkout_root() -> Path | None:
    """Repository root when the package runs from an editable install of a checkout, else None."""
    candidate = Path(__file__).resolve().parents[2]
    return candidate if (candidate / "pyproject.toml").is_file() else None


def checkpoints_dir() -> Path:
    """Directory holding the SAM2 checkpoints (``*.pt``)."""
    env = os.environ.get("ORGANOIDTRACKER_CHECKPOINTS")
    if env:
        return Path(env).expanduser()
    local = Path.cwd() / "checkpoints"
    if local.is_dir():
        return local
    root = source_checkout_root()
    if root is not None and (root / "checkpoints").is_dir():
        return root / "checkpoints"
    return local


def user_config_path() -> Path | None:
    """The ``user_config.py`` override file to load, or None when there is none."""
    env = os.environ.get("ORGANOIDTRACKER_USER_CONFIG")
    if env:
        return Path(env).expanduser()
    for base in (Path.cwd(), source_checkout_root()):
        if base is not None and (base / "user_config.py").is_file():
            return base / "user_config.py"
    return None
