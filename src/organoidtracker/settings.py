"""Typed application settings.

``Settings`` holds every setting the application reads, with the defaults the original
``config.py`` shipped. An optional ``organoidtracker.toml`` with flat, lowercase keys (see
``organoidtracker.example.toml`` at the repository root) overrides them; it is found through
:func:`settings_path`: the ``ORGANOIDTRACKER_SETTINGS`` environment variable, then the working
directory, then the root of the source checkout. Values are type-checked against the field
annotations and unknown keys are rejected, so a typo cannot silently do nothing.

``organoidtracker.config`` exposes the same values as module-level UPPERCASE constants for the
existing modules. The names in ``REMOVED_SETTINGS`` were settings of the original ``config.py``
that no code ever read; a file that still sets one loads with a warning.
"""

from __future__ import annotations

import logging
import os
import tomllib
import typing
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from .paths import source_checkout_root

logger = logging.getLogger(__name__)

SETTINGS_FILENAME = "organoidtracker.toml"

# Settings of the original config.py that no code ever read, removed. A settings file that still sets one
# of them loads with a warning; a later release will refuse them like any unknown key.
REMOVED_SETTINGS = frozenset(
    {
        "default_total_organoids",
        "default_video_quality",
        "debug_mode_enabled",
        "enable_comprehensive_reports",
        "figure_size",
        "suppress_matplotlib_warnings",
        "fallback_for_optional_deps",
        "plot_colors",
        "generate_debug_frames",
        "auto_save_analysis_data",
        "include_session_metadata",
        "csv_encoding",
        "json_ensure_ascii",
        "pdf_page_size",
        "max_video_frames",
        "memory_optimization",
        "garbage_collection_frequency",
        "torch_device_override",
        "sam2_model_cache",
        "enable_model_validation",
        "enable_performance_logging",
        "log_analysis_details",
    }
)


class SettingsError(ValueError):
    """An invalid settings file: unreadable, an unknown key, or a value of the wrong type."""


@dataclass
class Settings:
    """All application settings; the defaults are those of the original ``config.py``."""

    # GUI
    auto_open_output_directory: bool = False  # open the output folder in the file manager after a run

    # Analysis defaults shown in the GUI
    default_time_lapse_days: float = 6.0
    default_conversion_factor: float = 1.6934  # micrometers per pixel

    # Visualizations
    matplotlib_dpi: int = 150  # 150 for screen, 300 for publication
    visualization_format: str = "png"  # "png" or "svg"
    font_scale_factor: float = 1.0
    disable_visualization_text: bool = False  # hide axis labels and legends
    disable_visualization_titles: bool = False  # hide titles (publication mode)

    # Frames and checkpoints
    collapse_duplicate_frames: bool = True  # merge consecutive near-identical frames into one time point
    duplicate_frame_mad_threshold: float = 1.0  # mean absolute grayscale difference (0-255) below which frames repeat
    sam2_checkpoint_family: str = "2.1"  # "2.1" (default) or "2" (the original SAM 2 checkpoints)

    # Tracking quality filter
    sam2_improved_tracking: bool = True  # keep the largest component and apply the quality filter
    sam2_use_improved_config: bool = True  # base-plus model: use the application's "improved tracking" config
    sam2_min_mask_area: int = 50  # pixels
    sam2_min_confidence: float = 0.5
    sam2_memory_frames: int = 1000  # previous frames kept in SAM2's memory (1000 = all)

    # Logging
    log_level: str = "INFO"  # console and log file
    gui_log_level: str = "WARNING"  # records at this level and above also appear in the GUI log panel

    def as_constants(self) -> dict[str, Any]:
        """The settings under their UPPERCASE names, the interface ``organoidtracker.config`` exposes."""
        return {f.name.upper(): getattr(self, f.name) for f in fields(self)}


def settings_path() -> Path | None:
    """The settings file to load, or None when there is none."""
    env = os.environ.get("ORGANOIDTRACKER_SETTINGS")
    if env:
        return Path(env).expanduser()
    for base in (Path.cwd(), source_checkout_root()):
        if base is not None and (base / SETTINGS_FILENAME).is_file():
            return base / SETTINGS_FILENAME
    return None


def load_settings(path: Path | None = None) -> Settings:
    """Settings with the defaults, overridden by the TOML file at ``path`` (None: defaults only)."""
    settings = Settings()
    if path is None:
        return settings
    try:
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise SettingsError(f"cannot read settings file {path}: {error}") from error
    hints = typing.get_type_hints(Settings)
    known = {f.name for f in fields(Settings)}
    for key, value in data.items():
        if key in REMOVED_SETTINGS:
            logger.warning(
                f"{path}: {key!r} was removed (no code ever read it) and is ignored; "
                "delete it from the file, a later release will refuse it"
            )
            continue
        if key not in known:
            raise SettingsError(
                f"{path}: unknown setting {key!r} (see organoidtracker.example.toml for the valid keys)"
            )
        setattr(settings, key, _convert(value, hints[key], f"{path}: {key}"))
    return settings


def coerce_setting(name: str, value: Any, where: str) -> Any:
    """Check ``value`` for the setting ``name`` (a ``Settings`` field) and return it in the field's type."""
    hints = typing.get_type_hints(Settings)
    if name not in hints:
        raise SettingsError(f"{where}: unknown setting {name!r}")
    return _convert(value, hints[name], where)


def _convert(value: Any, annotation: Any, where: str) -> Any:
    """Check ``value`` against ``annotation`` (every setting is a bool, int, float or str) and return it in that type."""
    if annotation is bool:
        if isinstance(value, bool):
            return value
        raise SettingsError(f"{where}: expected true or false, got {value!r}")
    if annotation is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        raise SettingsError(f"{where}: expected an integer, got {value!r}")
    if annotation is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        raise SettingsError(f"{where}: expected a number, got {value!r}")
    if annotation is str:
        if isinstance(value, str):
            return value
        raise SettingsError(f"{where}: expected a string, got {value!r}")
    raise SettingsError(f"{where}: unsupported setting type {annotation!r}")
