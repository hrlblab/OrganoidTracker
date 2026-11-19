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

import hashlib
import json
import os
import subprocess
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
    """Revision of a checkout or frozen build; dirty source runs add a digest of their local changes."""
    root = source_checkout_root()
    if root is None:
        try:
            info = json.loads(Path(__file__).with_name("_build_info.json").read_text(encoding="utf-8"))
            revision = info.get("source_revision")
            if not isinstance(revision, str) or len(revision) not in (40, 64):
                return None
            if any(character not in "0123456789abcdef" for character in revision):
                return None
            return revision + ("-dirty" if info.get("dirty") else "")
        except (OSError, ValueError, AttributeError):
            return None
    try:

        def git(*args):
            return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, timeout=10)

        revision = git("rev-parse", "HEAD").decode().strip()
        state = git("status", "--porcelain", "--untracked-files=all")
        if not state:
            return revision
        digest = hashlib.sha256(git("diff", "--binary", "HEAD"))
        # Include uncommitted new implementations as well as tracked edits. Only the
        # digest is recorded; no working-tree content or filenames leave this process.
        for name in git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0"):
            if name:
                path = root / os.fsdecode(name)
                digest.update(name + b"\0")
                if path.is_file():
                    digest.update(hashlib.sha256(path.read_bytes()).digest())
        return revision + "-dirty-" + digest.hexdigest()[:16]
    except (OSError, subprocess.SubprocessError):
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
