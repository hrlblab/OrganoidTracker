"""
Application settings as module-level constants.

The values are those of :class:`organoidtracker.settings.Settings` (the defaults the original
``config.py`` shipped), overridden by an optional ``organoidtracker.toml`` found through
:func:`organoidtracker.settings.settings_path`. A legacy ``user_config.py`` next to it is still
applied on top with a deprecation warning. An invalid settings file (unreadable, unknown key,
wrong type) raises :class:`SettingsError` when this module is imported, so an explicitly
supplied configuration can never be silently replaced by the defaults; the launcher reports the
error and refuses to start. Modules import constants by name from here, for example
``from ..config import FONT_SCALE_FACTOR``; ``SETTINGS`` is the typed view of the same values.
"""

import importlib.util
import logging

from .paths import user_config_path
from .settings import SETTINGS_FILENAME, Settings, SettingsError, coerce_setting, load_settings, settings_path

logger = logging.getLogger(__name__)

LOADED_SETTINGS_FILE = None  # the organoidtracker.toml that was applied, if any
LOADED_USER_CONFIG = None  # the legacy user_config.py that was applied, if any

_settings_file = settings_path()
SETTINGS: Settings = load_settings(_settings_file)  # raises SettingsError on an invalid file
LOADED_SETTINGS_FILE = _settings_file

globals().update(SETTINGS.as_constants())

# Legacy override file (any UPPERCASE name); organoidtracker.toml replaces it.
_user_config_file = user_config_path()
if _user_config_file is not None:
    try:
        spec = importlib.util.spec_from_file_location("user_config", _user_config_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {_user_config_file}")
        user_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(user_config)
    except Exception as error:
        raise SettingsError(f"{_user_config_file}: cannot apply the legacy user configuration: {error}") from error
    for attr in dir(user_config):
        if not attr.startswith("_") and attr.isupper():
            value = getattr(user_config, attr)
            if hasattr(SETTINGS, attr.lower()):
                # a recognized setting gets the same type check as a TOML value
                value = coerce_setting(attr.lower(), value, f"{_user_config_file}: {attr}")
                setattr(SETTINGS, attr.lower(), value)
            else:
                logger.warning(f"{_user_config_file}: {attr} is not a known setting; nothing reads it")
            globals()[attr] = value
    LOADED_USER_CONFIG = _user_config_file
    logger.warning(
        f"{_user_config_file} is deprecated and will stop being read in a later release; "
        f"move its settings to {SETTINGS_FILENAME} (lowercase keys, see organoidtracker.example.toml)"
    )
