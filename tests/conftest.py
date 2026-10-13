"""Shared fixtures. Tests never touch lab data; everything here is synthetic.

Window tests (marker ``gui``) never use the desktop of the person running them, where their windows would take the
keyboard focus from other applications: on a Linux desktop session the test session runs on a private Xvfb display
started here, and without Xvfb the window tests are skipped with the reason. ``OT_GUI_ON_DESKTOP=1`` runs them on
the desktop anyway. Elsewhere nothing changes (CI runs them under ``xvfb-run``; Windows runs the smoke test).
"""

from __future__ import annotations

import os
import pathlib
import select
import shutil
import subprocess
import sys
import tempfile
import time

import pytest

from helpers import DiscVideo
from organoidtracker.paths import checkpoints_dir

TINY_CHECKPOINT = checkpoints_dir() / "sam2.1_hiera_tiny.pt"

# The private display while the session runs (process, display, the DISPLAY it replaced), or why window tests skip
_private_display: dict = {}
_window_tests_skipped: list[str] = []


def _on_linux_desktop() -> bool:
    return sys.platform.startswith("linux") and (
        os.environ.get("XDG_SESSION_TYPE") in ("wayland", "x11") or bool(os.environ.get("WAYLAND_DISPLAY"))
    )


def _start_xvfb(xvfb: str, timeout: float = 15.0) -> tuple[subprocess.Popen, str]:
    """Start Xvfb on a display number it chooses itself and reports through ``-displayfd`` once it accepts clients.

    The report is read up to its newline before the pipe is closed: Xvfb writes the digits and the newline
    separately and exits if the second write fails, which a pipe closed after the digits made it do. The server
    must still be running afterwards; otherwise the session stops with Xvfb's own error output.
    """
    read_fd, write_fd = os.pipe()
    errors = tempfile.TemporaryFile()
    try:
        process = subprocess.Popen(
            [xvfb, "-displayfd", str(write_fd), "-screen", "0", "1920x1080x24", "-nolisten", "tcp"],
            pass_fds=(write_fd,),
            stdout=subprocess.DEVNULL,
            stderr=errors,
        )
    finally:
        os.close(write_fd)
    report = b""
    deadline = time.monotonic() + timeout
    try:
        while not report.endswith(b"\n"):
            ready, _, _ = select.select([read_fd], [], [], max(deadline - time.monotonic(), 0))
            chunk = os.read(read_fd, 64) if ready else b""
            if not chunk:  # timed out, or Xvfb exited without finishing its report
                break
            report += chunk
    finally:
        os.close(read_fd)
    number = report.decode(errors="replace").strip()
    if report.endswith(b"\n") and number.isdigit() and process.poll() is None:
        errors.close()
        return process, f":{number}"
    if process.poll() is None:
        process.kill()
    process.wait()
    errors.seek(0)
    lines = [line for line in errors.read().decode(errors="replace").splitlines() if line.strip()]
    errors.close()
    raise pytest.UsageError(
        f"{xvfb} did not start a display (report {report!r}, exit status {process.returncode}): "
        + " | ".join(lines[-4:])
        + "; set OT_GUI_ON_DESKTOP=1 to run the window tests on the desktop instead"
    )


def pytest_configure(config):
    if not _on_linux_desktop() or os.environ.get("OT_GUI_ON_DESKTOP") == "1":
        return
    xvfb = shutil.which("Xvfb")
    if xvfb is None:
        _window_tests_skipped.append(
            "on this desktop window tests would take the focus from other applications; install Xvfb "
            "(sudo apt install xvfb) to run them on a private display, or set OT_GUI_ON_DESKTOP=1"
        )
        return
    process, display = _start_xvfb(xvfb)
    _private_display.update(process=process, display=display, previous=os.environ.get("DISPLAY"))
    os.environ["DISPLAY"] = display  # every Tk interpreter of the session (and its subprocesses) uses it
    os.environ["OT_PRIVATE_DISPLAY"] = display  # a window test that cannot reach it fails instead of skipping


def pytest_unconfigure(config):
    process = _private_display.pop("process", None)
    if process is None:
        return
    previous = _private_display.pop("previous", None)
    os.environ.pop("OT_PRIVATE_DISPLAY", None)
    if previous is None:
        os.environ.pop("DISPLAY", None)
    else:
        os.environ["DISPLAY"] = previous
    process.terminate()
    try:
        process.wait(10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def pytest_report_header(config):
    if "display" in _private_display:
        return f"window tests: private Xvfb display {_private_display['display']}"
    if _window_tests_skipped:
        return f"window tests: skipped ({_window_tests_skipped[0]})"
    return None


def pytest_collection_modifyitems(config, items):
    if not _window_tests_skipped:
        return
    skip = pytest.mark.skip(reason=_window_tests_skipped[0])
    for item in items:
        if item.get_closest_marker("gui") is not None:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def tiny_checkpoint() -> pathlib.Path:
    if not TINY_CHECKPOINT.exists():
        pytest.skip(f"{TINY_CHECKPOINT} is missing; run `bash checkpoints/download_ckpts.sh`")
    return TINY_CHECKPOINT


@pytest.fixture(scope="session")
def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def disc() -> DiscVideo:
    return DiscVideo()


@pytest.fixture(scope="session")
def disc_video_path(disc, tmp_path_factory) -> pathlib.Path:
    return disc.write(tmp_path_factory.mktemp("video") / "disc.mp4")
