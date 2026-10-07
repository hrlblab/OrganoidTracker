"""Typed application settings.

``Settings`` holds every setting the application reads, with the defaults the original
``config.py`` shipped. An optional ``organoidtracker.toml`` with flat, lowercase keys (see
``organoidtracker.example.toml`` at the repository root) overrides them; it is found through
:func:`settings_path`: the ``ORGANOIDTRACKER_SETTINGS`` environment variable, then the working
directory, then the root of the source checkout. Values are type-checked against the field
annotations and unknown keys are rejected, so a typo cannot silently do nothing.

``organoidtracker.config`` exposes the same values as module-level UPPERCASE constants for the
existing modules. Several settings are carried over from ``config.py`` although nothing reads
them yet; they are marked as such in the example file and will be pruned or wired in when the
core services are extracted.
"""

from __future__ import annotations

import os
import tomllib
import types
import typing
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from .paths import source_checkout_root

SETTINGS_FILENAME = "organoidtracker.toml"


class SettingsError(ValueError):
    """An invalid settings file: unreadable, an unknown key, or a value of the wrong type."""


def _legacy_plot_colors() -> dict[str, tuple[int, int, int]]:
    return {
        "primary": (70, 130, 180),  # steel blue
        "secondary": (220, 20, 60),  # crimson
        "accent": (255, 165, 0),  # orange
        "success": (34, 139, 34),  # forest green
        "warning": (255, 140, 0),  # dark orange
        "error": (178, 34, 34),  # fire brick
    }


@dataclass
class Settings:
    """All application settings; the defaults are those of the original ``config.py``."""

    # GUI
    auto_open_output_directory: bool = False  # open the output folder in the file manager after a run

    # Analysis defaults shown in the GUI
    default_total_organoids: int = 13
    default_time_lapse_days: float = 6.0
    default_conversion_factor: float = 1.6934  # micrometers per pixel

    # Video processing
    default_video_quality: str = "mid"  # "original", "mid" or "low"
    debug_mode_enabled: bool = False

    # Visualizations
    enable_comprehensive_reports: bool = True
    matplotlib_dpi: int = 150  # 150 for screen, 300 for publication
    visualization_format: str = "png"  # "png" or "svg"
    figure_size: tuple[float, float] = (10, 6)  # inches
    suppress_matplotlib_warnings: bool = True
    fallback_for_optional_deps: bool = True
    font_scale_factor: float = 1.0
    disable_visualization_text: bool = False  # hide axis labels and legends
    disable_visualization_titles: bool = False  # hide titles (publication mode)
    plot_colors: dict[str, tuple[int, int, int]] = field(default_factory=_legacy_plot_colors)

    # Reports and files
    generate_debug_frames: bool = False
    auto_save_analysis_data: bool = True
    include_session_metadata: bool = True
    csv_encoding: str = "utf-8"
    json_ensure_ascii: bool = False
    pdf_page_size: str = "A4"

    # Frames and checkpoints
    collapse_duplicate_frames: bool = True  # merge consecutive near-identical frames into one time point
    duplicate_frame_mad_threshold: float = 1.0  # mean absolute grayscale difference (0-255) below which frames repeat
    sam2_checkpoint_family: str = "2.1"  # "2.1" (default) or "2" (the original SAM 2 checkpoints)

    # Performance and expert options
    max_video_frames: int | None = None
    memory_optimization: bool = True
    garbage_collection_frequency: int = 10
    torch_device_override: str | None = None
    sam2_model_cache: bool = True
    enable_model_validation: bool = True

    # Tracking quality filter
    sam2_improved_tracking: bool = True  # keep the largest component and apply the quality filter
    sam2_use_improved_config: bool = True  # base-plus model: use the application's "improved tracking" config
    sam2_min_mask_area: int = 50  # pixels
    sam2_min_confidence: float = 0.5
    sam2_memory_frames: int = 1000  # previous frames kept in SAM2's memory (1000 = all)

    # Logging
    log_level: str = "INFO"  # console and log file
    gui_log_level: str = "WARNING"  # records at this level and above also appear in the GUI log panel
    enable_performance_logging: bool = False
    log_analysis_details: bool = True

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
    """Check ``value`` against ``annotation`` and return it in the annotated type."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin in (types.UnionType, typing.Union):
        members = [a for a in args if a is not type(None)]  # TOML cannot express None; omit the key instead
        errors = []
        for member in members:
            try:
                return _convert(value, member, where)
            except SettingsError as error:
                errors.append(str(error))
        raise SettingsError(errors[0])
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
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise SettingsError(f"{where}: expected a list, got {value!r}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(item, args[0], where) for item in value)
        if len(value) != len(args):
            raise SettingsError(f"{where}: expected {len(args)} values, got {len(value)}")
        return tuple(_convert(item, arg, where) for item, arg in zip(value, args, strict=True))
    if origin is dict:
        if not isinstance(value, dict):
            raise SettingsError(f"{where}: expected a table, got {value!r}")
        return {_convert(k, args[0], where): _convert(v, args[1], f"{where}.{k}") for k, v in value.items()}
    raise SettingsError(f"{where}: unsupported setting type {annotation!r}")
