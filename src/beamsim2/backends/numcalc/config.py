"""NumCalc binary path resolution.

The binary path is never hardcoded in source. Resolution order:
  1. Explicit ``binary_path`` argument (highest priority).
  2. ``BEAMSIM2_NUMCALC_BIN`` environment variable.
  3. ``~/.config/beamsim2/settings.toml``  [numcalc] bin key.
  4. ``FileNotFoundError`` with guidance.

This keeps the adapter portable: the absolute path recorded in docs/SETUP_NOTES.md
stays in documentation, not in code.  The GUI writes tier-3 on first launch so
the user never has to set an environment variable manually.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path


def _settings_path() -> Path:
    """Return the path to the user settings TOML file.

    Respects ``XDG_CONFIG_HOME`` if set; defaults to ``~/.config``.

    # HEURISTIC: XDG Base Directory Specification (freedesktop.org).
    # ~/.config is the correct location for a terminal-launched tool on macOS.
    # ~/Library/Application Support is for App Store bundles.
    """
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return Path(base) / "beamsim2" / "settings.toml"


def _read_config_bin() -> str | None:
    """Read the NumCalc binary path from the user settings TOML file.

    Returns the ``[numcalc] bin`` value if the file exists and is valid,
    or ``None`` if the file is absent, unreadable, or missing the key.

    Returns
    -------
    str or None
        The configured binary path string, or None.
    """
    p = _settings_path()
    if not p.is_file():
        return None
    try:
        with open(p, "rb") as fh:  # tomllib requires binary mode
            data = tomllib.load(fh)
        return data.get("numcalc", {}).get("bin") or None
    except (tomllib.TOMLDecodeError, OSError):
        return None


def _write_numcalc_config(path: str) -> None:
    """Persist the NumCalc binary path to the user settings TOML file.

    Creates ``~/.config/beamsim2/settings.toml`` (or the XDG override
    location) with the ``[numcalc] bin`` key.  The entire file is
    overwritten — any other sections present in a prior version of the
    file will be lost.  This is acceptable while the file contains only
    this one key; revisit if additional settings are added.

    Parameters
    ----------
    path : str
        Absolute path to the NumCalc executable to persist.
    """
    p = _settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    escaped = path.replace("\\", "\\\\").replace('"', '\\"')
    p.write_text(f'[numcalc]\nbin = "{escaped}"\n', encoding="utf-8")


def resolve_numcalc_binary(binary_path: str | None = None) -> str:
    """Return the absolute path to the NumCalc executable.

    Resolution order (highest priority first):
      1. Explicit ``binary_path`` argument.
      2. ``BEAMSIM2_NUMCALC_BIN`` environment variable.
      3. ``[numcalc] bin`` key in ``~/.config/beamsim2/settings.toml``
         (or ``$XDG_CONFIG_HOME/beamsim2/settings.toml``).
      4. ``FileNotFoundError`` with guidance.

    Parameters
    ----------
    binary_path : str or None
        Explicit path to the NumCalc binary.  If None, falls through to
        the environment variable and then the config file.

    Returns
    -------
    str
        Absolute path to an existing NumCalc executable.

    Raises
    ------
    FileNotFoundError
        If the binary cannot be located via any source, or if the resolved
        path does not exist on disk.
    """
    candidate = binary_path or os.environ.get("BEAMSIM2_NUMCALC_BIN")
    if not candidate:
        candidate = _read_config_bin()  # tier 3 — user settings file

    if not candidate:
        raise FileNotFoundError(
            "NumCalc binary not found. Options:\n"
            "  1. Launch the GUI — it will prompt you to locate the binary once\n"
            "     and save your choice permanently.\n"
            "  2. Set the BEAMSIM2_NUMCALC_BIN environment variable, e.g.:\n"
            "       export BEAMSIM2_NUMCALC_BIN=/path/to/NumCalc/bin/NumCalc\n"
            "See docs/SETUP_NOTES.md for the binary path recorded for this machine."
        )

    if not os.path.isfile(candidate):
        raise FileNotFoundError(
            f"NumCalc binary not found at: {candidate}\n"
            "Check BEAMSIM2_NUMCALC_BIN, the binary_path argument, or\n"
            f"the config file at {_settings_path()}."
        )

    return candidate
