"""NumCalc binary path resolution.

The binary path is never hardcoded in source. Resolution order:
  1. Explicit ``binary_path`` argument (highest priority).
  2. ``BEAMSIM2_NUMCALC_BIN`` environment variable.
  3. ``FileNotFoundError`` with guidance.

This keeps the adapter portable: the absolute path recorded in docs/SETUP_NOTES.md
stays in documentation, not in code.
"""

from __future__ import annotations

import os


def resolve_numcalc_binary(binary_path: str | None = None) -> str:
    """Return the absolute path to the NumCalc executable.

    Checks an explicit argument first, then the ``BEAMSIM2_NUMCALC_BIN``
    environment variable. Raises if neither is set or the resolved path
    does not exist.

    Parameters
    ----------
    binary_path : str or None
        Explicit path to the NumCalc binary. If None, falls back to the
        environment variable.

    Returns
    -------
    str
        Absolute path to an existing NumCalc executable.

    Raises
    ------
    FileNotFoundError
        If the binary cannot be located via either source.
    """
    candidate = binary_path or os.environ.get("BEAMSIM2_NUMCALC_BIN")

    if not candidate:
        raise FileNotFoundError(
            "NumCalc binary not found. Set the BEAMSIM2_NUMCALC_BIN environment "
            "variable to the absolute path of the NumCalc executable, e.g.:\n"
            "  export BEAMSIM2_NUMCALC_BIN=/path/to/NumCalc/bin/NumCalc\n"
            "See docs/SETUP_NOTES.md for the binary path recorded for this machine."
        )

    if not os.path.isfile(candidate):
        raise FileNotFoundError(
            f"NumCalc binary not found at: {candidate}\n"
            "Check BEAMSIM2_NUMCALC_BIN or the binary_path argument."
        )

    return candidate
