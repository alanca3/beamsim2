"""NumCalc binary path resolution + section-preserving settings layer.

The binary path is never hardcoded in source.  Resolution order:
  1. Explicit ``binary_path`` argument (highest priority).
  2. ``BEAMSIM2_NUMCALC_BIN`` environment variable.
  3. ``~/.config/beamsim2/settings.toml``  [numcalc] bin key.
  4. ``FileNotFoundError`` with guidance.

This keeps the adapter portable: the absolute path recorded in docs/SETUP_NOTES.md
stays in documentation, not in code.  The GUI writes tier-3 on first launch so
the user never has to set an environment variable manually.

Settings persistence
--------------------
``update_settings(section, mapping)`` reads the TOML file, merges the new
mapping into the named section, and rewrites the whole file — so all sections
coexist.  The file supports these value types: str, bool, int, float,
list[str].  No external dependency (no tomli-w); the writer is ~20 lines.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Union

_RECENT_CAP = 8  # max entries in recent-projects list


def _settings_path() -> Path:
    """Return the path to the user settings TOML file.

    Respects ``XDG_CONFIG_HOME`` if set; defaults to ``~/.config``.

    # HEURISTIC: XDG Base Directory Specification (freedesktop.org).
    # ~/.config is the correct location for a terminal-launched tool on macOS.
    # ~/Library/Application Support is for App Store bundles.
    """
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return Path(base) / "beamsim2" / "settings.toml"


# ---------------------------------------------------------------------------
# Minimal hand-rolled TOML writer (no new dependency)
# ---------------------------------------------------------------------------


def _escape_str(s: str) -> str:
    """Escape backslashes and double-quotes for a TOML basic string."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


_TomlValue = Union[str, bool, int, float, list]


def _write_toml(data: dict, path: Path) -> None:
    """Write a flat-section TOML dict to *path*.

    Supports exactly the value types used by BeamSimII settings:
    str, bool, int, float, list[str].  Keys within a section are emitted
    in insertion order.  Sections are separated by a blank line.

    Parameters
    ----------
    data : dict
        Mapping of section name → {key: value, ...}.
    path : Path
        Destination file (created or overwritten).

    Raises
    ------
    TypeError
        If a value type is not one of str, bool, int, float, list[str].
    """
    lines: list[str] = []
    for section, values in data.items():
        if not isinstance(values, dict):
            continue
        lines.append(f"[{section}]")
        for k, v in values.items():
            if isinstance(v, bool):  # bool before int — bool subclasses int
                lines.append(f"{k} = {'true' if v else 'false'}")
            elif isinstance(v, int):
                lines.append(f"{k} = {v}")
            elif isinstance(v, float):
                lines.append(f"{k} = {v}")
            elif isinstance(v, str):
                lines.append(f'{k} = "{_escape_str(v)}"')
            elif isinstance(v, list):
                # list[str] only — for recent-projects; validate element types
                for s in v:
                    if not isinstance(s, str):
                        raise TypeError(
                            f"settings TOML writer: unsupported type "
                            f"{type(s).__name__!r} in list for key {section}.{k}"
                        )
                items = ", ".join(f'"{_escape_str(s)}"' for s in v)
                lines.append(f"{k} = [{items}]")
            else:
                raise TypeError(
                    f"settings TOML writer: unsupported type {type(v).__name__!r} "
                    f"for key {section}.{k}"
                )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Section-preserving read / write
# ---------------------------------------------------------------------------


def read_settings() -> dict:
    """Read the full settings TOML, returning ``{}`` if missing or unreadable.

    Returns
    -------
    dict
        Section-keyed mapping, as returned by ``tomllib.load``.  An empty
        dict is returned on any read error so callers never need to handle
        missing-file failures.
    """
    p = _settings_path()
    if not p.is_file():
        return {}
    try:
        with open(p, "rb") as fh:  # tomllib requires binary mode
            return tomllib.load(fh)
    except (tomllib.TOMLDecodeError, OSError):
        return {}


def update_settings(section: str, mapping: dict) -> None:
    """Merge *mapping* into *section*, preserving all other sections, and write.

    If the file is absent, it is created.  If *section* is absent, it is
    created.  All other existing sections survive unchanged.

    Parameters
    ----------
    section : str
        TOML section name (e.g. ``"numcalc"``, ``"logging"``).
    mapping : dict
        Key/value pairs to merge into the section.  Values must be str,
        bool, int, float, or list[str].
    """
    data = read_settings()
    if section not in data:
        data[section] = {}
    data[section].update(mapping)
    p = _settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    _write_toml(data, p)


# ---------------------------------------------------------------------------
# Logging preferences
# ---------------------------------------------------------------------------


def read_logging_prefs() -> dict:
    """Read ``[logging]`` section from settings, returning defaults if absent.

    Returns
    -------
    dict
        Keys: ``enabled`` (bool), ``level`` (str), ``logfile`` (str).
    """
    section = read_settings().get("logging", {})
    return {
        "enabled": bool(section.get("enabled", False)),
        "level": str(section.get("level", "INFO")),
        "logfile": str(section.get("logfile", "")),
    }


def write_logging_prefs(*, enabled: bool, level: str, logfile: str) -> None:
    """Persist logging preferences to the ``[logging]`` section.

    Parameters
    ----------
    enabled : bool
        Whether GUI logging is active.
    level : str
        Log level name (e.g. ``"INFO"``, ``"DEBUG"``).
    logfile : str
        Absolute path for the log file, or ``""`` for no file output.
    """
    update_settings("logging", {"enabled": enabled, "level": level, "logfile": logfile})


# ---------------------------------------------------------------------------
# Recent-projects list
# ---------------------------------------------------------------------------


def read_recent_projects() -> list[str]:
    """Read the ``[recent] projects`` list from settings.

    Returns
    -------
    list[str]
        Most-recent-first list of project paths (may be empty).
    """
    section = read_settings().get("recent", {})
    raw = section.get("projects", [])
    # Defensive: filter out non-string entries from a malformed file.
    return [str(p) for p in raw if isinstance(p, str) and p]


def push_recent_project(path: str | Path) -> None:
    """Prepend *path* to the recent-projects list, dedup, and cap at _RECENT_CAP.

    Parameters
    ----------
    path : str or Path
        The project file path to record as the most recent.
    """
    s = str(path)
    existing = read_recent_projects()
    # Remove any prior occurrence so the same project doesn't appear twice.
    deduped = [p for p in existing if p != s]
    new_list = [s] + deduped
    update_settings("recent", {"projects": new_list[:_RECENT_CAP]})


# ---------------------------------------------------------------------------
# NumCalc binary helpers (unchanged public API)
# ---------------------------------------------------------------------------


def _read_config_bin() -> str | None:
    """Read the NumCalc binary path from the user settings TOML file.

    Returns the ``[numcalc] bin`` value if the file exists and is valid,
    or ``None`` if the file is absent, unreadable, or missing the key.

    Returns
    -------
    str or None
        The configured binary path string, or None.
    """
    return read_settings().get("numcalc", {}).get("bin") or None


def _write_numcalc_config(path: str) -> None:
    """Persist the NumCalc binary path to the ``[numcalc]`` section.

    Uses ``update_settings`` so other sections (``[logging]``, ``[recent]``)
    survive the write.

    Parameters
    ----------
    path : str
        Absolute path to the NumCalc executable to persist.
    """
    update_settings("numcalc", {"bin": path})


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
