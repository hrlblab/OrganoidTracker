"""The private display of the window tests (conftest.py): it is started reliably, or the session stops with the reason.

The start-up is checked with stand-in servers (no X server needed) and, where Xvfb is installed, with the real one
started several times in a row, each time serving a Tk client. Nothing here touches the session's own display.
"""

import os
import shutil
import subprocess
import sys
import textwrap
import time

import pytest

import conftest

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="the private display serves Linux desktops"
)


def stand_in(tmp_path, body: str) -> str:
    """An executable that behaves like Xvfb's ``-displayfd`` report as ``body`` scripts it (``fd`` is the report's)."""
    script = tmp_path / "Xvfb"
    script.write_text(
        f"#!{sys.executable}\nimport os, sys, time\nfd = int(sys.argv[sys.argv.index('-displayfd') + 1])\n"
        + textwrap.dedent(body)
    )
    script.chmod(0o755)
    return str(script)


def test_the_report_is_read_up_to_its_newline(tmp_path):
    """Regression: the digits were read and the pipe closed before Xvfb wrote the newline; its second write failed,
    Xvfb exited ("Cannot write display number to fd") and the window tests skipped."""
    xvfb = stand_in(
        tmp_path,
        """
        os.write(fd, b"42")
        time.sleep(0.3)
        try:
            os.write(fd, b"\\n")
        except OSError:
            sys.exit(1)  # what Xvfb does when it cannot finish its report
        time.sleep(30)
        """,
    )
    process, display = conftest._start_xvfb(xvfb)
    try:
        time.sleep(0.6)  # past the newline: a pipe closed after the digits has made the server exit by now
        assert display == ":42" and process.poll() is None
    finally:
        process.kill()
        process.wait()


def test_a_server_that_exits_before_finishing_its_report_stops_the_session(tmp_path):
    xvfb = stand_in(
        tmp_path,
        """
        os.write(fd, b"42")
        sys.stderr.write("(EE) Fatal server error: injected\\n")
        sys.exit(1)
        """,
    )
    with pytest.raises(
        pytest.UsageError, match=r"did not start a display.*exit status 1.*Fatal server error: injected"
    ):
        conftest._start_xvfb(xvfb)


def test_a_server_that_reports_nothing_in_time_stops_the_session(tmp_path):
    xvfb = stand_in(tmp_path, "time.sleep(30)\n")
    with pytest.raises(pytest.UsageError, match="did not start a display"):
        conftest._start_xvfb(xvfb, timeout=0.5)


@pytest.mark.gui
@pytest.mark.skipif(shutil.which("Xvfb") is None, reason="Xvfb is not installed")
def test_the_real_xvfb_starts_and_serves_several_times_in_a_row():
    """Five back-to-back starts (the failure was intermittent): each reports a display that a Tk client reaches."""
    for _ in range(5):
        process, display = conftest._start_xvfb(shutil.which("Xvfb"))
        try:
            client = subprocess.run(
                [sys.executable, "-c", "import tkinter; root = tkinter.Tk(); root.update(); root.destroy()"],
                env={**os.environ, "DISPLAY": display},
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert client.returncode == 0, client.stderr
            assert process.poll() is None
        finally:
            process.terminate()
            process.wait(10)
