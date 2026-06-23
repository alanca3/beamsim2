"""Tests for the section-preserving settings layer in backends.numcalc.config.

All tests redirect XDG_CONFIG_HOME to a tmpdir so they never touch the real
~/.config/beamsim2/settings.toml.  CI-safe (no display, no NumCalc binary).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from beamsim2.backends.numcalc.config import (
    _write_toml,
    push_recent_project,
    read_logging_prefs,
    read_recent_projects,
    read_settings,
    update_settings,
    write_logging_prefs,
)


def _isolate(monkeypatch, tmp_path: Path) -> None:
    """Redirect config to tmp_path and clear the env var."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("BEAMSIM2_NUMCALC_BIN", raising=False)


# ---------------------------------------------------------------------------
# _write_toml — TOML writer unit tests
# ---------------------------------------------------------------------------


def test_write_toml_all_types(tmp_path: Path):
    """_write_toml handles str, bool, int, float, list[str]."""
    p = tmp_path / "out.toml"
    data = {
        "sec": {
            "s": "hello world",
            "b_true": True,
            "b_false": False,
            "i": 42,
            "f": 3.14,
            "lst": ["alpha", "beta/gamma"],
        }
    }
    _write_toml(data, p)
    with open(p, "rb") as fh:
        loaded = tomllib.load(fh)
    assert loaded["sec"]["s"] == "hello world"
    assert loaded["sec"]["b_true"] is True
    assert loaded["sec"]["b_false"] is False
    assert loaded["sec"]["i"] == 42
    assert abs(loaded["sec"]["f"] - 3.14) < 1e-9
    assert loaded["sec"]["lst"] == ["alpha", "beta/gamma"]


def test_write_toml_escapes_backslash_and_quote(tmp_path: Path):
    """_write_toml correctly escapes backslashes and double-quotes in strings."""
    p = tmp_path / "out.toml"
    path_with_backslash = r"C:\Users\andy\NumCalc"
    path_with_quote = 'path "with" quotes'
    _write_toml({"numcalc": {"bin": path_with_backslash, "label": path_with_quote}}, p)
    with open(p, "rb") as fh:
        loaded = tomllib.load(fh)
    assert loaded["numcalc"]["bin"] == path_with_backslash
    assert loaded["numcalc"]["label"] == path_with_quote


def test_write_toml_unsupported_type_raises(tmp_path: Path):
    """_write_toml raises TypeError for unsupported value types."""
    with pytest.raises(TypeError, match="unsupported type"):
        _write_toml({"s": {"k": [1, 2, 3]}}, tmp_path / "out.toml")  # list[int] unsupported


# ---------------------------------------------------------------------------
# update_settings — section-preserving merge
# ---------------------------------------------------------------------------


def test_update_settings_creates_file(monkeypatch, tmp_path: Path):
    """update_settings creates the TOML file if it does not exist."""
    _isolate(monkeypatch, tmp_path)
    update_settings("numcalc", {"bin": "/path/to/NumCalc"})
    cfg_path = tmp_path / "beamsim2" / "settings.toml"
    assert cfg_path.is_file()
    with open(cfg_path, "rb") as fh:
        data = tomllib.load(fh)
    assert data["numcalc"]["bin"] == "/path/to/NumCalc"


def test_update_settings_preserves_other_sections(monkeypatch, tmp_path: Path):
    """update_settings does not clobber unrelated sections."""
    _isolate(monkeypatch, tmp_path)
    update_settings("numcalc", {"bin": "/path/NumCalc"})
    update_settings("logging", {"enabled": True, "level": "DEBUG", "logfile": ""})
    update_settings("recent", {"projects": ["/a/b.bsim", "/c/d.bsim"]})

    data = read_settings()
    assert data["numcalc"]["bin"] == "/path/NumCalc"
    assert data["logging"]["enabled"] is True
    assert data["logging"]["level"] == "DEBUG"
    assert data["recent"]["projects"] == ["/a/b.bsim", "/c/d.bsim"]


def test_update_settings_merges_within_section(monkeypatch, tmp_path: Path):
    """update_settings merges new keys into an existing section."""
    _isolate(monkeypatch, tmp_path)
    update_settings("logging", {"enabled": True})
    update_settings("logging", {"level": "INFO"})  # should not drop "enabled"

    data = read_settings()
    assert data["logging"]["enabled"] is True
    assert data["logging"]["level"] == "INFO"


def test_read_settings_returns_empty_dict_when_missing(monkeypatch, tmp_path: Path):
    """read_settings returns {} if the settings file does not exist."""
    _isolate(monkeypatch, tmp_path)
    assert read_settings() == {}


# ---------------------------------------------------------------------------
# Logging preferences helpers
# ---------------------------------------------------------------------------


def test_write_read_logging_prefs_round_trip(monkeypatch, tmp_path: Path):
    """write_logging_prefs / read_logging_prefs round-trip."""
    _isolate(monkeypatch, tmp_path)
    write_logging_prefs(enabled=True, level="WARNING", logfile="/tmp/beamsim.log")
    prefs = read_logging_prefs()
    assert prefs["enabled"] is True
    assert prefs["level"] == "WARNING"
    assert prefs["logfile"] == "/tmp/beamsim.log"


def test_read_logging_prefs_defaults(monkeypatch, tmp_path: Path):
    """read_logging_prefs returns sensible defaults when the section is absent."""
    _isolate(monkeypatch, tmp_path)
    prefs = read_logging_prefs()
    assert prefs["enabled"] is False
    assert prefs["level"] == "INFO"
    assert prefs["logfile"] == ""


def test_logging_prefs_coexist_with_numcalc(monkeypatch, tmp_path: Path):
    """Writing logging prefs does not clobber [numcalc] section."""
    _isolate(monkeypatch, tmp_path)
    update_settings("numcalc", {"bin": "/usr/bin/NumCalc"})
    write_logging_prefs(enabled=False, level="ERROR", logfile="")
    data = read_settings()
    assert data["numcalc"]["bin"] == "/usr/bin/NumCalc"
    assert data["logging"]["enabled"] is False


# ---------------------------------------------------------------------------
# Recent projects helpers
# ---------------------------------------------------------------------------


def test_push_recent_project_prepends(monkeypatch, tmp_path: Path):
    """push_recent_project prepends the new path (most-recent first)."""
    _isolate(monkeypatch, tmp_path)
    push_recent_project("/a/first.bsim")
    push_recent_project("/b/second.bsim")
    recent = read_recent_projects()
    assert recent[0] == "/b/second.bsim"
    assert recent[1] == "/a/first.bsim"


def test_push_recent_project_deduplicates(monkeypatch, tmp_path: Path):
    """push_recent_project removes a prior occurrence of the same path."""
    _isolate(monkeypatch, tmp_path)
    push_recent_project("/a/proj.bsim")
    push_recent_project("/b/other.bsim")
    push_recent_project("/a/proj.bsim")  # again — should move to front
    recent = read_recent_projects()
    assert recent.count("/a/proj.bsim") == 1
    assert recent[0] == "/a/proj.bsim"


def test_push_recent_project_caps_at_eight(monkeypatch, tmp_path: Path):
    """recent-projects list is capped at 8 entries (_RECENT_CAP)."""
    _isolate(monkeypatch, tmp_path)
    for i in range(12):
        push_recent_project(f"/proj/file_{i}.bsim")
    recent = read_recent_projects()
    assert len(recent) == 8
    # Most recent (file_11) should be first
    assert recent[0] == "/proj/file_11.bsim"


def test_recent_projects_coexist_with_numcalc(monkeypatch, tmp_path: Path):
    """Writing recent projects does not clobber [numcalc] section."""
    _isolate(monkeypatch, tmp_path)
    update_settings("numcalc", {"bin": "/bin/NumCalc"})
    push_recent_project("/some/proj.bsim")
    data = read_settings()
    assert data["numcalc"]["bin"] == "/bin/NumCalc"
    assert "/some/proj.bsim" in data["recent"]["projects"]


def test_numcalc_config_still_uses_update_settings(monkeypatch, tmp_path: Path):
    """_write_numcalc_config uses update_settings so other sections survive."""
    _isolate(monkeypatch, tmp_path)
    from beamsim2.backends.numcalc.config import _write_numcalc_config

    write_logging_prefs(enabled=True, level="DEBUG", logfile="")
    _write_numcalc_config("/path/to/NumCalc")
    data = read_settings()
    # Logging section must survive the numcalc write
    assert data["logging"]["enabled"] is True
    assert data["numcalc"]["bin"] == "/path/to/NumCalc"
