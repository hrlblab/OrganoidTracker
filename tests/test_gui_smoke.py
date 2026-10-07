"""The Tk application builds its window and processes events; needs a display (marker: gui)."""

import tkinter

import pytest

pytestmark = pytest.mark.gui


@pytest.fixture
def display():
    try:
        root = tkinter.Tk()
    except tkinter.TclError as error:
        pytest.skip(f"no display for Tk: {error}")
    root.destroy()


def test_main_window_opens_and_closes(display):
    from organoidtracker.gui_tk.main_window import VideoTrackerApp

    app = VideoTrackerApp()
    try:
        app.root.update()
        assert app.current_model is None
        assert str(app.track_btn["state"]) == "disabled"
        assert app.model_combo["values"], "the model registry lists no backend"
        assert "Select a model" in app.status_label["text"]
    finally:
        app.root.destroy()
