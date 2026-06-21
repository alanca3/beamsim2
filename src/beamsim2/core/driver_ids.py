"""Driver-identifier generation and uniqueness validation.

Every driver in a :class:`~beamsim2.assembly.tensor.RadiationDataset` is keyed by
a unique ``driver_id`` string.  That uniqueness is load-bearing: the HDF5 contract
stores each driver as a named group (``/drivers/<driver_id>/``) and records a
``driver_order`` list so the Phase-2 beamformer can map steering-matrix rows back
to drivers.  A duplicate ``driver_id`` silently corrupts that mapping —
``h5py`` raises on the second ``create_group`` mid-write (leaving a partial file
whose ``driver_order`` is longer than its group set), and the reader then
duplicates the surviving group and drops the rest.

This module is the single source of truth for two concerns, kept Qt-free and
dependency-light so both the GUI and the headless pipeline use the same logic:

* **generation** — :func:`next_driver_id` / :func:`make_unique_id` hand out ids
  that never collide with the drivers already placed (the GUI's old
  ``f"driver_{len(drivers)}"`` scheme reused an index after a middle driver was
  deleted, which is exactly how ``HDF5/Dr1.h5`` ended up with two ``driver_4``s);
* **validation** — :func:`validate_unique_driver_ids` is the contract guard that
  the pipeline, assembly, and writer all call so a duplicate fails loud and early
  (before a multi-hour solve, before truncating a good file) instead of silently.

References
----------
DATA_CONTRACT.md §3.5/§3.6 (driver groups, ``driver_order``).
BEAMSIMII_Gameplan.md §3.4 (cardinal phase-origin rule — driver identity is how
per-driver responses are kept distinct).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

_DEFAULT_PREFIX = "driver"


def next_driver_id(existing: Iterable[str], prefix: str = _DEFAULT_PREFIX) -> str:
    """Return the smallest ``"{prefix}_{k}"`` (k ≥ 0) not already in ``existing``.

    Unlike a count-based scheme (``f"{prefix}_{len(existing)}"``), this never
    collides with a currently-present id: it fills the lowest free slot.  After a
    middle driver is deleted its number is reused, which is harmless because that
    id is, by construction, free — the collision bug was assigning an id equal to
    one still in use.

    Parameters
    ----------
    existing : iterable of str
        Driver ids already placed (any iterable; consumed once).
    prefix : str, optional
        Id stem.  Default ``"driver"`` → ``"driver_0"``, ``"driver_1"``, …

    Returns
    -------
    str
        A ``"{prefix}_{k}"`` id guaranteed absent from ``existing``.

    Examples
    --------
    >>> next_driver_id(["driver_0", "driver_2"])
    'driver_1'
    >>> next_driver_id([])
    'driver_0'
    """
    taken = set(existing)
    k = 0
    while f"{prefix}_{k}" in taken:
        k += 1
    return f"{prefix}_{k}"


def make_unique_id(desired: str, existing: Iterable[str]) -> str:
    """Return ``desired`` if free, else ``desired`` with a numeric suffix appended.

    Used when a user types a custom id that may collide with another driver: the
    intent (their chosen name) is preserved as closely as possible while
    guaranteeing uniqueness against the current set.  An empty/whitespace
    ``desired`` falls back to the default-prefix auto-numbering.

    Parameters
    ----------
    desired : str
        The requested id (may be empty, whitespace, or a duplicate).
    existing : iterable of str
        Driver ids already placed.

    Returns
    -------
    str
        ``desired`` if absent from ``existing`` and non-empty; otherwise
        ``"{desired}_2"``, ``"{desired}_3"``, … until unique, or a fresh
        auto-numbered id when ``desired`` is blank.

    Examples
    --------
    >>> make_unique_id("woofer", ["tweeter"])
    'woofer'
    >>> make_unique_id("woofer", ["woofer"])
    'woofer_2'
    """
    taken = set(existing)
    clean = desired.strip()
    if not clean:
        return next_driver_id(taken)
    if clean not in taken:
        return clean
    suffix = 2
    while f"{clean}_{suffix}" in taken:
        suffix += 1
    return f"{clean}_{suffix}"


def validate_unique_driver_ids(ids: Sequence[str]) -> None:
    """Raise :class:`ValueError` unless every id is non-empty and unique.

    The contract guard called by :func:`~beamsim2.pipeline.run.run_simulation`
    (before the solve), :func:`~beamsim2.assembly.tensor.build_dataset`, and
    :func:`~beamsim2.io.hdf5_store.write_dataset` (before opening the file).  It
    fails loud rather than letting a duplicate corrupt the on-disk dataset.

    Parameters
    ----------
    ids : sequence of str
        The driver ids in dataset/solve order.

    Raises
    ------
    ValueError
        If any id is empty/whitespace, or if any id appears more than once.
        The message names the offending ids so the GUI can surface it verbatim.
    """
    blanks = [i for i, s in enumerate(ids) if not str(s).strip()]
    if blanks:
        raise ValueError(
            f"driver_id must be a non-empty string; blank at position(s) {blanks}. "
            "Give every driver a name before solving."
        )
    seen: dict[str, int] = {}
    dups: list[str] = []
    for s in ids:
        seen[s] = seen.get(s, 0) + 1
    dups = [s for s, n in seen.items() if n > 1]
    if dups:
        raise ValueError(
            f"driver_id values must be unique; duplicates: {sorted(dups)}. "
            "Each driver is stored under its own HDF5 group, so a repeated id "
            "would silently overwrite/drop a driver. Rename the duplicate(s)."
        )


def _self_test() -> None:
    """Quick invariants — no Qt, no I/O."""
    # generation never collides with the current set
    assert next_driver_id([]) == "driver_0"
    assert next_driver_id(["driver_0", "driver_1"]) == "driver_2"
    assert next_driver_id(["driver_0", "driver_2"]) == "driver_1"  # fills the gap

    # the exact GUI collision scenario: place 3, delete middle, add one
    ids = []
    for _ in range(3):
        ids.append(next_driver_id(ids))  # driver_0, driver_1, driver_2
    assert ids == ["driver_0", "driver_1", "driver_2"]
    del ids[1]  # remove driver_1 → [driver_0, driver_2]
    new = next_driver_id(ids)
    assert new not in ids, f"regenerated id {new!r} collides with {ids}"
    assert new == "driver_1"

    # custom-id de-duplication
    assert make_unique_id("woofer", []) == "woofer"
    assert make_unique_id("woofer", ["woofer"]) == "woofer_2"
    assert make_unique_id("woofer", ["woofer", "woofer_2"]) == "woofer_3"
    assert make_unique_id("   ", ["driver_0"]) == "driver_1"

    # validation
    validate_unique_driver_ids(["a", "b", "c"])  # ok
    for bad in (["a", "a"], ["a", ""], ["a", "  "]):
        try:
            validate_unique_driver_ids(bad)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"expected ValueError for {bad}")
    print("core/driver_ids.py self-test: PASS")


if __name__ == "__main__":
    _self_test()
