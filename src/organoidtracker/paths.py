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


def source_revision() -> str | None:
    """Commit hash of the source checkout when running from one, else None (read from .git, no subprocess)."""
    root = source_checkout_root()
    if root is None:
        return None
    try:
        git_dir = root / ".git"
        if git_dir.is_file():  # a linked worktree: "gitdir: <path>"
            git_dir = (root / git_dir.read_text().split(":", 1)[1].strip()).resolve()
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head or None
        ref = head[5:]
        common = git_dir / "commondir"
        base = (git_dir / common.read_text().strip()).resolve() if common.is_file() else git_dir
        for ref_file in (git_dir / ref, base / ref):
            if ref_file.is_file():
                return ref_file.read_text().strip() or None
        packed = base / "packed-refs"
        if packed.is_file():
            for line in packed.read_text().splitlines():
                if line.endswith(" " + ref):
                    return line.split()[0]
    except OSError:
        return None
    return None


def user_config_path() -> Path | None:
    """The ``user_config.py`` override file to load, or None when there is none."""
    env = os.environ.get("ORGANOIDTRACKER_USER_CONFIG")
    if env:
        return Path(env).expanduser()
    for base in (Path.cwd(), source_checkout_root()):
        if base is not None and (base / "user_config.py").is_file():
            return base / "user_config.py"
    return None
