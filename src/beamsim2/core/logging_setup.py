"""Logging foundation for BeamSimII (file + level, opt-in).

This is the infrastructure half of bug #5.  It follows the standard library
discipline so the package never imposes output on its host:

* **Library code** (``pipeline``, ``backends``, ``assembly``, ``io``) only ever
  calls :func:`get_logger` and emits records.  It never adds handlers, sets
  levels, or calls :func:`logging.basicConfig`.
* The ``beamsim2`` root logger carries a :class:`logging.NullHandler` (attached
  in ``beamsim2/__init__.py``) so emitting without configuration is silent.
* The **application / CLI / tests** call :func:`configure_logging` once to route
  records to a file and/or the console at a chosen level.

The GUI Preferences toggle that drives :func:`configure_logging` lands in
Chunk 5; this module is what it will call.

References
----------
docs/Bug_Fix_Proposal.md Chunk 1 (#5, logging foundation).
Python logging HOWTO — "Configuring Logging for a Library".
"""

from __future__ import annotations

import logging
from pathlib import Path

PACKAGE_LOGGER = "beamsim2"

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Handlers installed by configure_logging, tracked so repeat calls are idempotent
# (the GUI may toggle logging on/off many times in one session).
_OWNED_HANDLERS: list[logging.Handler] = []


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``beamsim2`` namespace.

    Library modules call ``logger = get_logger(__name__)`` at import and then
    emit (``logger.info(...)``) without ever configuring handlers or levels.

    Parameters
    ----------
    name : str
        Usually ``__name__``.  Names already under ``beamsim2`` are returned
        as-is; any other name is nested under ``beamsim2.`` so all package
        records share one configurable parent.

    Returns
    -------
    logging.Logger
    """
    if name == PACKAGE_LOGGER or name.startswith(PACKAGE_LOGGER + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{PACKAGE_LOGGER}.{name}")


def install_null_handler() -> None:
    """Attach a :class:`logging.NullHandler` to the package logger (idempotent).

    Called from ``beamsim2/__init__.py`` so that library code can emit records
    before (or without) any :func:`configure_logging` call without triggering
    the "No handlers could be found" warning or printing to stderr.
    """
    pkg = logging.getLogger(PACKAGE_LOGGER)
    if not any(isinstance(h, logging.NullHandler) for h in pkg.handlers):
        pkg.addHandler(logging.NullHandler())


def configure_logging(
    level: int | str = logging.INFO,
    *,
    logfile: str | Path | None = None,
    console: bool = True,
) -> logging.Logger:
    """Route ``beamsim2`` log records to a file and/or the console.

    Intended for the application, the CLI, and tests — **not** for library code.
    Idempotent: handlers installed by a previous call are removed first, so the
    GUI can call this whenever the user toggles logging or changes the log file.

    Parameters
    ----------
    level : int or str, optional
        Threshold for the ``beamsim2`` logger (e.g. ``logging.DEBUG`` or
        ``"INFO"``).  Default ``logging.INFO``.
    logfile : str or Path or None, optional
        If set, append records to this file (parent directories created).
        ``None`` → no file handler.
    console : bool, optional
        If ``True`` (default) also emit to ``stderr``.

    Returns
    -------
    logging.Logger
        The configured ``beamsim2`` package logger.
    """
    pkg = logging.getLogger(PACKAGE_LOGGER)
    pkg.setLevel(level)
    # Records are emitted by this logger's hierarchy only; don't double-log via root.
    pkg.propagate = False

    # Remove handlers we previously owned (leave a NullHandler / foreign handlers).
    for h in list(_OWNED_HANDLERS):
        pkg.removeHandler(h)
        h.close()
    _OWNED_HANDLERS.clear()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    if console:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        pkg.addHandler(sh)
        _OWNED_HANDLERS.append(sh)

    if logfile is not None:
        path = Path(logfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(formatter)
        pkg.addHandler(fh)
        _OWNED_HANDLERS.append(fh)

    return pkg


def _self_test() -> None:
    """Verify get_logger namespacing and idempotent configure — no files left."""
    import tempfile

    assert get_logger("beamsim2").name == "beamsim2"
    assert get_logger("beamsim2.pipeline").name == "beamsim2.pipeline"
    assert get_logger("pipeline.run").name == "beamsim2.pipeline.run"

    with tempfile.TemporaryDirectory() as d:
        log_path = Path(d) / "bs.log"
        configure_logging(logging.DEBUG, logfile=log_path, console=False)
        n_owned = len(_OWNED_HANDLERS)
        get_logger("pipeline.run").info("hello")
        # second call must not accumulate handlers
        configure_logging(logging.INFO, logfile=log_path, console=False)
        assert len(_OWNED_HANDLERS) == n_owned, "configure_logging not idempotent"
        assert log_path.exists() and "hello" in log_path.read_text()
        # reset to a clean library state for the rest of the process
        for h in list(_OWNED_HANDLERS):
            logging.getLogger(PACKAGE_LOGGER).removeHandler(h)
            h.close()
        _OWNED_HANDLERS.clear()
    print("core/logging_setup.py self-test: PASS")


if __name__ == "__main__":
    _self_test()
