"""The Tk application builds its window and processes events (marker: gui on the window test).

The application is created directly and is the only Tk interpreter in the process, exactly as in
normal use: on Windows with Python 3.13 a second Tk interpreter in the same process fails with
"Can't find a usable init.tcl". The test is skipped only when Tk's error confirms that no display
or no usable Tcl installation is available; any other error while the window is built (an invalid
widget option, a missing attribute) fails the test.
"""

import tkinter

import pytest

# Messages with which Tk reports an unavailable display or an unusable Tcl/Tk installation.
ENVIRONMENT_ERRORS = (
    "no display name and no $display environment variable",
    "couldn't connect to display",
    "can't find a usable init.tcl",
    "can't find a usable tk.tcl",
    "tcl wasn't installed properly",
    "tk wasn't installed properly",
)


def environment_skip_reason(error: BaseException) -> str | None:
    """A skip reason when ``error`` says Tk cannot run in this environment, else None (a real failure)."""
    message = str(error).lower()
    if isinstance(error, tkinter.TclError) and any(pattern in message for pattern in ENVIRONMENT_ERRORS):
        return f"Tk cannot start here: {error}"
    return None


@pytest.fixture
def app():
    from organoidtracker.gui_tk.main_window import VideoTrackerApp

    try:
        app = VideoTrackerApp()
    except tkinter.TclError as error:
        reason = environment_skip_reason(error)
        if reason is None:
            raise  # an error in the application's own widgets must fail, not skip
        pytest.skip(reason)
    yield app
    app.root.destroy()


@pytest.mark.gui
def test_main_window_opens_and_closes(app):
    app.root.update()
    assert app.current_model is None
    assert str(app.track_btn["state"]) == "disabled"
    assert app.model_combo["values"], "the model registry lists no backend"
    assert "Select a model" in app.status_label["text"]


@pytest.mark.parametrize(
    "message",
    [
        "no display name and no $DISPLAY environment variable",
        'couldn\'t connect to display ":99"',
        "Can't find a usable init.tcl in the following directories: {C:/x/tcl8.6}",
    ],
)
def test_environment_errors_are_skips(message):
    assert environment_skip_reason(tkinter.TclError(message)) is not None


@pytest.mark.parametrize(
    "error",
    [
        tkinter.TclError('unknown option "-foo"'),
        tkinter.TclError('bad window path name ".!frame.!button"'),
        tkinter.TclError('invalid command name "ttk::combobox"'),
        AttributeError("'VideoTrackerApp' object has no attribute 'track_btn'"),
    ],
)
def test_application_errors_are_failures(error):
    assert environment_skip_reason(error) is None


def test_an_injected_widget_error_fails_the_window_test(monkeypatch, request):
    """The fixture re-raises a widget-construction error instead of skipping."""
    from organoidtracker.gui_tk import main_window

    def broken_app():
        raise tkinter.TclError('unknown option "-foo"')

    monkeypatch.setattr(main_window, "VideoTrackerApp", broken_app)
    fixture = request.getfixturevalue  # resolve the `app` fixture through pytest with the broken class
    with pytest.raises(tkinter.TclError, match="unknown option"):
        fixture("app")
