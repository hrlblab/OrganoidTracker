"""The Tk application builds its window and processes events; needs a display (marker: gui).

The application is created directly and is the only Tk interpreter in the process, exactly as
in normal use. On Windows with Python 3.13 (both the python.org and the uv-managed builds) a
second Tk interpreter in the same process fails with "Can't find a usable init.tcl", so no probe
interpreter is created first; Tk failing to start at all skips the test.
"""

import tkinter

import pytest

pytestmark = pytest.mark.gui


@pytest.fixture
def app():
    from organoidtracker.gui_tk.main_window import VideoTrackerApp

    try:
        app = VideoTrackerApp()
    except tkinter.TclError as error:  # no display, or an unusable Tcl installation
        pytest.skip(f"Tk cannot start here: {error}")
    yield app
    app.root.destroy()


def test_main_window_opens_and_closes(app):
    app.root.update()
    assert app.current_model is None
    assert str(app.track_btn["state"]) == "disabled"
    assert app.model_combo["values"], "the model registry lists no backend"
    assert "Select a model" in app.status_label["text"]
