"""Tests for GUI logging toggle via core.logging_setup.configure_logging.

These tests verify that enabling/disabling logging through the Preferences
dialog's backing code results in the expected handler state on the
beamsim2 root logger.  CI-safe (no display, no NumCalc binary).
"""

from __future__ import annotations

import logging

import pytest

from beamsim2.core.logging_setup import (
    _OWNED_HANDLERS,
    PACKAGE_LOGGER,
    configure_logging,
    get_logger,
)


@pytest.fixture(autouse=True)
def _reset_logging():
    """Restore clean logging state after every test."""
    yield
    # Remove all owned handlers so tests don't bleed into each other.
    pkg = logging.getLogger(PACKAGE_LOGGER)
    for h in list(_OWNED_HANDLERS):
        pkg.removeHandler(h)
        h.close()
    _OWNED_HANDLERS.clear()
    # Restore propagation and level to defaults.
    pkg.propagate = True
    pkg.setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Basic configure_logging behaviour
# ---------------------------------------------------------------------------


def test_configure_logging_adds_stream_handler():
    """configure_logging(console=True) installs a StreamHandler on the root logger."""
    configure_logging(logging.INFO, console=True)
    pkg = logging.getLogger(PACKAGE_LOGGER)
    handlers = [h for h in pkg.handlers if isinstance(h, logging.StreamHandler)]
    assert handlers, "Expected at least one StreamHandler after configure_logging"


def test_configure_logging_idempotent():
    """Calling configure_logging twice does not accumulate handlers."""
    configure_logging(logging.INFO, console=True)
    n_before = len(_OWNED_HANDLERS)
    configure_logging(logging.INFO, console=True)
    assert len(_OWNED_HANDLERS) == n_before


def test_configure_logging_silence():
    """Calling configure_logging with CRITICAL + no console effectively silences beamsim2."""
    configure_logging(logging.INFO, console=True)
    assert _OWNED_HANDLERS, "Precondition: should have handlers"
    configure_logging(logging.CRITICAL, console=False)
    # OWNED_HANDLERS is empty (or only has a NullHandler which we don't add ourselves)
    # and level is CRITICAL — nothing below CRITICAL will pass through.
    pkg = logging.getLogger(PACKAGE_LOGGER)
    assert pkg.level == logging.CRITICAL
    # No owned stream/file handlers remain
    for h in _OWNED_HANDLERS:
        assert not isinstance(h, (logging.StreamHandler, logging.FileHandler))


def test_configure_logging_sets_level():
    """configure_logging sets the package logger's level correctly."""
    configure_logging(logging.DEBUG, console=False)
    pkg = logging.getLogger(PACKAGE_LOGGER)
    assert pkg.level == logging.DEBUG


def test_configure_logging_file_handler(tmp_path):
    """configure_logging with logfile writes records to the file."""
    log_file = tmp_path / "test.log"
    configure_logging(logging.DEBUG, logfile=str(log_file), console=False)
    logger = get_logger(__name__)
    logger.warning("hello from test")

    assert log_file.exists(), "Log file should have been created"
    content = log_file.read_text()
    assert "hello from test" in content


def test_configure_logging_removes_old_file_handler(tmp_path):
    """Reconfiguring with a new log file replaces the old FileHandler."""
    f1 = tmp_path / "log1.log"
    f2 = tmp_path / "log2.log"
    configure_logging(logging.INFO, logfile=str(f1), console=False)
    configure_logging(logging.INFO, logfile=str(f2), console=False)

    # Only the new file should be open; f1 should not receive new records.
    logger = get_logger(__name__)
    logger.warning("to f2 only")
    assert "to f2 only" in f2.read_text()
    assert not f1.exists() or "to f2 only" not in f1.read_text()


# ---------------------------------------------------------------------------
# Preferences-dialog scenario: enable → disable → re-enable
# ---------------------------------------------------------------------------


def test_disable_then_enable_logging():
    """The enable/disable cycle from PreferencesDialog works end-to-end."""
    # Simulate 'enable logging' from Preferences OK
    configure_logging(logging.INFO, console=True)
    pkg = logging.getLogger(PACKAGE_LOGGER)
    assert any(isinstance(h, logging.StreamHandler) for h in pkg.handlers)

    # Simulate 'disable logging'
    configure_logging(logging.CRITICAL, console=False)
    owned_non_null = [h for h in _OWNED_HANDLERS if not isinstance(h, logging.NullHandler)]
    assert not owned_non_null, "Expected no active owned handlers after disable"

    # Re-enable
    configure_logging(logging.WARNING, console=True)
    assert any(isinstance(h, logging.StreamHandler) for h in pkg.handlers)
