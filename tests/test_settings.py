"""Typed settings keep the original config.py defaults and load checked overrides from organoidtracker.toml."""

import json
import os
import subprocess
import sys

import pytest

from organoidtracker import config
from organoidtracker.settings import Settings, SettingsError, load_settings, settings_path

# The constants of config.py as shipped before the settings module existed (WP1, commit ac45423),
# plus GUI_LOG_LEVEL, which the logging change added. Frozen on purpose: a change here is a
# behavior change and belongs in the changelog.
LEGACY_DEFAULTS = {
    "AUTO_OPEN_OUTPUT_DIRECTORY": False,
    "DEFAULT_TOTAL_ORGANOIDS": 13,
    "DEFAULT_TIME_LAPSE_DAYS": 6.0,
    "DEFAULT_CONVERSION_FACTOR": 1.6934,
    "DEFAULT_VIDEO_QUALITY": "mid",
    "DEBUG_MODE_ENABLED": False,
    "ENABLE_COMPREHENSIVE_REPORTS": True,
    "MATPLOTLIB_DPI": 150,
    "VISUALIZATION_FORMAT": "png",
    "FIGURE_SIZE": [10, 6],
    "SUPPRESS_MATPLOTLIB_WARNINGS": True,
    "FALLBACK_FOR_OPTIONAL_DEPS": True,
    "GENERATE_DEBUG_FRAMES": False,
    "AUTO_SAVE_ANALYSIS_DATA": True,
    "INCLUDE_SESSION_METADATA": True,
    "PLOT_COLORS": {
        "primary": [70, 130, 180],
        "secondary": [220, 20, 60],
        "accent": [255, 165, 0],
        "success": [34, 139, 34],
        "warning": [255, 140, 0],
        "error": [178, 34, 34],
    },
    "CSV_ENCODING": "utf-8",
    "JSON_ENSURE_ASCII": False,
    "PDF_PAGE_SIZE": "A4",
    "FONT_SCALE_FACTOR": 1.0,
    "DISABLE_VISUALIZATION_TEXT": False,
    "DISABLE_VISUALIZATION_TITLES": False,
    "COLLAPSE_DUPLICATE_FRAMES": True,
    "DUPLICATE_FRAME_MAD_THRESHOLD": 1.0,
    "SAM2_CHECKPOINT_FAMILY": "2.1",
    "MAX_VIDEO_FRAMES": None,
    "MEMORY_OPTIMIZATION": True,
    "GARBAGE_COLLECTION_FREQUENCY": 10,
    "TORCH_DEVICE_OVERRIDE": None,
    "SAM2_MODEL_CACHE": True,
    "ENABLE_MODEL_VALIDATION": True,
    "SAM2_IMPROVED_TRACKING": True,
    "SAM2_USE_IMPROVED_CONFIG": True,
    "SAM2_MIN_MASK_AREA": 50,
    "SAM2_MIN_CONFIDENCE": 0.5,
    "SAM2_MEMORY_FRAMES": 1000,
    "LOG_LEVEL": "INFO",
    "GUI_LOG_LEVEL": "WARNING",
    "ENABLE_PERFORMANCE_LOGGING": False,
    "LOG_ANALYSIS_DETAILS": True,
}


def plain(value):
    """Tuples become lists so that defaults compare with the JSON-style snapshot."""
    return json.loads(json.dumps(value, default=list))


def test_defaults_equal_the_original_config_constants():
    assert plain(Settings().as_constants()) == LEGACY_DEFAULTS


def test_config_module_exposes_every_setting_as_a_constant():
    for name in Settings().as_constants():
        assert hasattr(config, name), name
    assert config.SETTINGS.sam2_min_mask_area == config.SAM2_MIN_MASK_AREA


def test_toml_overrides_are_applied_with_type_conversion(tmp_path):
    path = tmp_path / "organoidtracker.toml"
    path.write_text(
        "sam2_min_mask_area = 80\n"
        "default_time_lapse_days = 7\n"
        "figure_size = [12, 8]\n"
        "max_video_frames = 20\n"
        "plot_colors = { primary = [1, 2, 3] }\n"
        'log_level = "DEBUG"\n'
    )
    settings = load_settings(path)
    assert settings.sam2_min_mask_area == 80
    assert settings.default_time_lapse_days == 7.0 and isinstance(settings.default_time_lapse_days, float)
    assert settings.figure_size == (12.0, 8.0)
    assert settings.max_video_frames == 20
    assert settings.plot_colors == {"primary": (1, 2, 3)}
    assert settings.log_level == "DEBUG"
    assert settings.sam2_min_confidence == 0.5  # untouched default


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("sam2_min_mask_area = 1.5", "expected an integer"),
        ("collapse_duplicate_frames = 1", "expected true or false"),
        ("sam2_min_mask_areas = 3", "unknown setting 'sam2_min_mask_areas'"),
        ("figure_size = [1, 2, 3]", "expected 2 values"),
        ('max_video_frames = "many"', "expected an integer"),
        ("plot_colors = { primary = [1, 2] }", "plot_colors.primary: expected 3 values"),
        ("not toml at all", "cannot read settings file"),
    ],
)
def test_invalid_settings_name_the_key(tmp_path, text, message):
    path = tmp_path / "organoidtracker.toml"
    path.write_text(text + "\n")
    with pytest.raises(SettingsError, match=message):
        load_settings(path)


def test_no_file_means_defaults():
    assert load_settings(None) == Settings()


def test_settings_file_lookup_order(tmp_path, monkeypatch):
    monkeypatch.setattr("organoidtracker.settings.source_checkout_root", lambda: None)
    monkeypatch.delenv("ORGANOIDTRACKER_SETTINGS", raising=False)
    monkeypatch.chdir(tmp_path)
    assert settings_path() is None
    local = tmp_path / "organoidtracker.toml"
    local.write_text("")
    assert settings_path() == local
    explicit = tmp_path / "elsewhere.toml"
    monkeypatch.setenv("ORGANOIDTRACKER_SETTINGS", str(explicit))
    assert settings_path() == explicit


def test_invalid_settings_file_aborts_the_import(tmp_path):
    """An explicitly supplied invalid file must never be replaced by the defaults."""
    bad = tmp_path / "organoidtracker.toml"
    bad.write_text("sam2_min_mask_areas = 3\n")
    env = {**os.environ, "ORGANOIDTRACKER_SETTINGS": str(bad)}
    proc = subprocess.run(
        [sys.executable, "-c", "import organoidtracker.config"], env=env, capture_output=True, text=True
    )
    assert proc.returncode != 0
    assert "SettingsError" in proc.stderr and "unknown setting 'sam2_min_mask_areas'" in proc.stderr


def test_launcher_refuses_to_start_with_an_invalid_settings_file(tmp_path, monkeypatch):
    import organoidtracker
    from organoidtracker.gui_tk import launcher, main_window  # imported before the config module is evicted

    bad = tmp_path / "organoidtracker.toml"
    bad.write_text("collapse_duplicate_frames = 1\n")
    monkeypatch.setenv("ORGANOIDTRACKER_SETTINGS", str(bad))
    # force `from .. import config` inside main() to import the module afresh
    monkeypatch.delitem(sys.modules, "organoidtracker.config", raising=False)
    monkeypatch.delattr(organoidtracker, "config", raising=False)
    shown = []
    monkeypatch.setattr(launcher, "_show_startup_error", shown.append)

    def refuse_to_open_a_window():
        raise AssertionError("the launcher must not start the GUI with invalid settings")

    monkeypatch.setattr(main_window, "VideoTrackerApp", refuse_to_open_a_window)
    assert launcher.main() == 2
    assert shown and "expected true or false" in shown[0]
