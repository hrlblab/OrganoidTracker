"""Typed settings keep the original config.py defaults and load checked overrides from organoidtracker.toml."""

import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest

from organoidtracker import config
from organoidtracker.settings import (
    REMOVED_SETTINGS,
    Settings,
    SettingsError,
    coerce_setting,
    load_settings,
    settings_path,
)

EXAMPLE_FILE = Path(__file__).resolve().parents[1] / "organoidtracker.example.toml"

# The constants of config.py as shipped before the settings module existed (WP1, commit ac45423),
# plus GUI_LOG_LEVEL, which the logging change added, minus the 22 settings no code ever read
# (REMOVED_SETTINGS). Frozen on purpose: a change here is a behavior change and belongs in the changelog.
LEGACY_DEFAULTS = {
    "AUTO_OPEN_OUTPUT_DIRECTORY": False,
    "DEFAULT_TIME_LAPSE_DAYS": 6.0,
    "DEFAULT_CONVERSION_FACTOR": 1.6934,
    "MATPLOTLIB_DPI": 150,
    "VISUALIZATION_FORMAT": "png",
    "FONT_SCALE_FACTOR": 1.0,
    "DISABLE_VISUALIZATION_TEXT": False,
    "DISABLE_VISUALIZATION_TITLES": False,
    "COLLAPSE_DUPLICATE_FRAMES": True,
    "DUPLICATE_FRAME_MAD_THRESHOLD": 1.0,
    "SAM2_CHECKPOINT_FAMILY": "2.1",
    "SAM2_IMPROVED_TRACKING": True,
    "SAM2_USE_IMPROVED_CONFIG": True,
    "SAM2_MIN_MASK_AREA": 50,
    "SAM2_MIN_CONFIDENCE": 0.5,
    "SAM2_MEMORY_FRAMES": 1000,
    "LOG_LEVEL": "INFO",
    "GUI_LOG_LEVEL": "WARNING",
}


def plain(value):
    """Tuples become lists so that defaults compare with the JSON-style snapshot."""
    return json.loads(json.dumps(value, default=list))


def test_defaults_equal_the_original_config_constants():
    assert plain(Settings().as_constants()) == LEGACY_DEFAULTS


def test_the_removed_settings_are_the_original_constants_nothing_read():
    """18 kept plus 22 removed are the 40 settings before the removal; none of the removed is still a field."""
    kept = {f.name for f in fields(Settings)}
    assert len(kept) == 18 and len(REMOVED_SETTINGS) == 22
    assert kept.isdisjoint(REMOVED_SETTINGS)
    assert {name.upper() for name in kept} == set(LEGACY_DEFAULTS)
    for name in REMOVED_SETTINGS:
        assert not hasattr(config, name.upper()), name


def test_config_module_exposes_every_setting_as_a_constant():
    for name in Settings().as_constants():
        assert hasattr(config, name), name
    assert config.SETTINGS.sam2_min_mask_area == config.SAM2_MIN_MASK_AREA


def test_toml_overrides_are_applied_with_type_conversion(tmp_path):
    path = tmp_path / "organoidtracker.toml"
    path.write_text(
        "sam2_min_mask_area = 80\n"
        "default_time_lapse_days = 7\n"
        "duplicate_frame_mad_threshold = 2\n"
        "matplotlib_dpi = 300\n"
        'visualization_format = "svg"\n'
        "disable_visualization_titles = true\n"
        'log_level = "DEBUG"\n'
    )
    settings = load_settings(path)
    assert settings.sam2_min_mask_area == 80
    assert settings.default_time_lapse_days == 7.0 and isinstance(settings.default_time_lapse_days, float)
    assert settings.duplicate_frame_mad_threshold == 2.0 and isinstance(settings.duplicate_frame_mad_threshold, float)
    assert settings.matplotlib_dpi == 300
    assert settings.visualization_format == "svg"
    assert settings.disable_visualization_titles is True
    assert settings.log_level == "DEBUG"
    assert settings.sam2_min_confidence == 0.5  # untouched default


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("sam2_min_mask_area = 1.5", "expected an integer"),
        ("sam2_min_mask_area = true", "expected an integer"),
        ("collapse_duplicate_frames = 1", "expected true or false"),
        ('sam2_min_confidence = "half"', "expected a number"),
        ("default_conversion_factor = true", "expected a number"),
        ("sam2_checkpoint_family = 2.1", "expected a string"),
        ("sam2_min_mask_areas = 3", "unknown setting 'sam2_min_mask_areas'"),
        ("figure_sizes = [1, 2]", "unknown setting 'figure_sizes'"),  # close to a removed name, but never a setting
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


def test_legacy_user_config_values_are_type_checked(tmp_path):
    legacy = tmp_path / "user_config.py"
    legacy.write_text('SAM2_IMPROVED_TRACKING = "false"\n')
    env = {**os.environ, "ORGANOIDTRACKER_USER_CONFIG": str(legacy)}
    env.pop("ORGANOIDTRACKER_SETTINGS", None)
    proc = subprocess.run(
        [sys.executable, "-c", "import organoidtracker.config"], env=env, capture_output=True, text=True
    )
    assert proc.returncode != 0
    assert "SAM2_IMPROVED_TRACKING: expected true or false" in proc.stderr


def test_legacy_user_config_values_are_applied_when_valid(tmp_path):
    legacy = tmp_path / "user_config.py"
    legacy.write_text("SAM2_MIN_MASK_AREA = 80\nDEFAULT_CONVERSION_FACTOR = 2\nMY_OWN_FLAG = True\n")
    env = {**os.environ, "ORGANOIDTRACKER_USER_CONFIG": str(legacy)}
    env.pop("ORGANOIDTRACKER_SETTINGS", None)
    code = (
        "from organoidtracker import config; "
        "print(config.SAM2_MIN_MASK_AREA, repr(config.DEFAULT_CONVERSION_FACTOR), config.MY_OWN_FLAG, "
        "config.SETTINGS.sam2_min_mask_area, config.LOADED_USER_CONFIG is not None)"
    )
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "80 2.0 True 80 True"
    assert "deprecated" in proc.stderr and "MY_OWN_FLAG is not a known setting" in proc.stderr


def test_every_setting_has_a_checked_type_and_its_default_passes_the_check():
    """Every field is a bool, int, float or str, the types the checker knows; None is refused for all of them."""
    defaults = Settings()
    for f in fields(Settings):
        value = getattr(defaults, f.name)
        assert type(value) in (bool, int, float, str), f.name
        assert coerce_setting(f.name, value, "x") == value
        with pytest.raises(SettingsError, match="expected"):
            coerce_setting(f.name, None, "x")


def test_a_removed_setting_in_the_file_is_ignored_with_a_warning(tmp_path, caplog):
    path = tmp_path / "organoidtracker.toml"
    path.write_text('figure_size = "big"\nsam2_min_mask_area = 80\nmax_video_frames = 20\n')
    with caplog.at_level(logging.WARNING, logger="organoidtracker.settings"):
        settings = load_settings(path)
    assert settings == Settings(sam2_min_mask_area=80)  # the other key applies, the removed ones change nothing
    assert not hasattr(settings, "figure_size") and not hasattr(settings, "max_video_frames")
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert "'figure_size' was removed" in warnings[0] and "'max_video_frames' was removed" in warnings[1]
    assert all("ignored" in w and "a later release will refuse it" in w for w in warnings)


def test_a_settings_file_with_a_removed_key_still_starts_the_application(tmp_path):
    path = tmp_path / "organoidtracker.toml"
    path.write_text("plot_colors = { primary = [1, 2, 3] }\nsam2_min_mask_area = 80\n")
    env = {**os.environ, "ORGANOIDTRACKER_SETTINGS": str(path)}
    env.pop("ORGANOIDTRACKER_USER_CONFIG", None)
    code = "from organoidtracker import config; print(config.SAM2_MIN_MASK_AREA, hasattr(config, 'PLOT_COLORS'))"
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "80 False"
    assert "'plot_colors' was removed" in proc.stderr


def test_legacy_user_config_with_removed_names_applies_the_rest(tmp_path):
    legacy = tmp_path / "user_config.py"
    legacy.write_text("MAX_VIDEO_FRAMES = None\nTORCH_DEVICE_OVERRIDE = None\nSAM2_MEMORY_FRAMES = 4\n")
    env = {**os.environ, "ORGANOIDTRACKER_USER_CONFIG": str(legacy)}
    env.pop("ORGANOIDTRACKER_SETTINGS", None)
    code = "from organoidtracker import config; print(config.SAM2_MEMORY_FRAMES, config.SETTINGS.sam2_memory_frames)"
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "4 4"
    assert "MAX_VIDEO_FRAMES was removed" in proc.stderr and "TORCH_DEVICE_OVERRIDE was removed" in proc.stderr


def test_the_example_file_lists_every_setting_with_its_default(tmp_path):
    """Uncommented, the example file is a valid settings file that sets every setting to its default."""
    pattern = re.compile(r'^#\s*([a-z][a-z0-9_]*)\s*=\s*("[^"]*"|[^\s#]+)')
    matches = [m for m in map(pattern.match, EXAMPLE_FILE.read_text().splitlines()) if m]
    assert sorted(m.group(1) for m in matches) == sorted(f.name for f in fields(Settings))
    uncommented = tmp_path / "organoidtracker.toml"
    uncommented.write_text("".join(f"{m.group(1)} = {m.group(2)}\n" for m in matches))
    assert load_settings(uncommented) == Settings()
