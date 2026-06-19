"""Tests for NumCalc binary-path resolution tiers and _write_numcalc_config.

All tests isolate from the real ~/.config/beamsim2/settings.toml via
XDG_CONFIG_HOME and strip BEAMSIM2_NUMCALC_BIN so the env-var tier cannot
mask the config-file tier.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from beamsim2.backends.numcalc.config import (
    _write_numcalc_config,
    resolve_numcalc_binary,
)


def _fake_binary(directory: Path, name: str = "NumCalc") -> Path:
    """Create a zero-byte file that passes os.path.isfile (resolver only checks isfile)."""
    p = directory / name
    p.write_bytes(b"")
    return p


def _isolate(monkeypatch, tmp_path: Path) -> None:
    """Redirect config dir to tmp_path and strip env var."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("BEAMSIM2_NUMCALC_BIN", raising=False)


def test_explicit_arg_wins(monkeypatch, tmp_path: Path) -> None:
    """Explicit binary_path argument takes priority over config file."""
    _isolate(monkeypatch, tmp_path)
    binary = _fake_binary(tmp_path, "NumCalc")
    # Write a config pointing to a ghost path — should be ignored
    _write_numcalc_config(str(tmp_path / "nonexistent"))
    result = resolve_numcalc_binary(str(binary))
    assert result == str(binary)


def test_env_var_wins_over_config(monkeypatch, tmp_path: Path) -> None:
    """BEAMSIM2_NUMCALC_BIN env var takes priority over config file."""
    _isolate(monkeypatch, tmp_path)
    env_binary = _fake_binary(tmp_path, "from_env")
    config_binary = _fake_binary(tmp_path, "from_config")
    _write_numcalc_config(str(config_binary))
    monkeypatch.setenv("BEAMSIM2_NUMCALC_BIN", str(env_binary))
    result = resolve_numcalc_binary()
    assert result == str(env_binary)


def test_config_file_used_when_no_arg_or_env(monkeypatch, tmp_path: Path) -> None:
    """Config file tier is used when no explicit arg and no env var is set."""
    _isolate(monkeypatch, tmp_path)
    binary = _fake_binary(tmp_path, "NumCalc")
    _write_numcalc_config(str(binary))
    result = resolve_numcalc_binary()
    assert result == str(binary)


def test_raises_when_nothing_configured(monkeypatch, tmp_path: Path) -> None:
    """FileNotFoundError raised when all three tiers are absent."""
    _isolate(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError):
        resolve_numcalc_binary()


def test_raises_when_config_path_missing(monkeypatch, tmp_path: Path) -> None:
    """FileNotFoundError raised when config file points to a non-existent binary."""
    _isolate(monkeypatch, tmp_path)
    _write_numcalc_config(str(tmp_path / "ghost" / "NumCalc"))
    with pytest.raises(FileNotFoundError):
        resolve_numcalc_binary()


def test_write_read_roundtrip_with_space_in_path(monkeypatch, tmp_path: Path) -> None:
    """Paths containing spaces survive TOML write/read round-trip."""
    _isolate(monkeypatch, tmp_path)
    spaced_dir = tmp_path / "path with spaces"
    spaced_dir.mkdir()
    binary = _fake_binary(spaced_dir, "NumCalc")
    _write_numcalc_config(str(binary))
    result = resolve_numcalc_binary()
    assert result == str(binary)


def test_write_produces_valid_toml(monkeypatch, tmp_path: Path) -> None:
    """_write_numcalc_config writes valid TOML with the expected [numcalc] bin key."""
    _isolate(monkeypatch, tmp_path)
    binary = _fake_binary(tmp_path, "NumCalc")
    _write_numcalc_config(str(binary))
    settings_path = tmp_path / "beamsim2" / "settings.toml"
    assert settings_path.is_file()
    with open(settings_path, "rb") as fh:
        data = tomllib.load(fh)
    assert data["numcalc"]["bin"] == str(binary)
