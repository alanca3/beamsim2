"""Native HDF5 read/write for the H-tensor data contract.

Writes (and reads back) the Phase-1 output in the exact layout defined in
DATA_CONTRACT.md §3.6:

    /frequencies            [F] float64
    /directions/
        unit_vectors        [N×3] float64
        weights             [N]   float64
        theta_phi           [N×2] float64
        attrs: scheme, order, weight_convention
    /drivers/<id>/
        H_bem               [F×N] complex128
        terminal_response   [F]   complex128
        H_full              [F×N] complex128
        convergence_flags   [F]   bool
        attrs: name, position, orientation, radius, profile, ts_params, …
    root attrs: schema_version, beamsim_version, created_utc, …

Complex128 is stored as two-field compound (real/imag float64 each) so it
roundtrips exactly — ``np.array_equal`` on every element.  Dict-valued attrs
(ts_params, profile params) are JSON-encoded as strings so h5py never has to
serialise arbitrary Python objects.

schema_version = "1.0" — bumped only when the on-disk layout changes (§11.2,
CLAUDE.md).  Separate from the application version.

References
----------
DATA_CONTRACT.md §3.5, §3.6.  BEAMSIMII_Gameplan.md §3 (phase-origin rule §3.4).
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from beamsim2.assembly.tensor import DriverData, RadiationDataset
from beamsim2.core.driver_ids import validate_unique_driver_ids
from beamsim2.core.types import ObservationPoints

SCHEMA_VERSION = "1.0"


def _check_schema_version(file_version: str, path: Path) -> None:
    """Guard against reading an incompatible on-disk schema (contract R-09).

    The reader compares the file's ``schema_version`` to the package's
    :data:`SCHEMA_VERSION`. A different MAJOR version means the on-disk layout is
    incompatible and is refused (``ValueError``); a matching major with a different
    minor, or a missing version, is accepted with a :class:`UserWarning` so that
    downstream tools (e.g. the Phase-2 filter designer) never silently consume a file
    whose contract they do not understand.

    Parameters
    ----------
    file_version : str
        The ``schema_version`` attribute read from the file ("" if absent).
    path : Path
        File path, for the message.
    """
    if not file_version:
        warnings.warn(
            f"{path.name}: no schema_version attribute; assuming compatible with "
            f"{SCHEMA_VERSION}. File may predate the schema-version contract.",
            UserWarning,
            stacklevel=3,
        )
        return
    file_major = file_version.split(".")[0]
    cur_major = SCHEMA_VERSION.split(".")[0]
    if file_major != cur_major:
        raise ValueError(
            f"{path.name}: incompatible schema_version {file_version!r} "
            f"(reader supports {SCHEMA_VERSION!r}; major versions differ). "
            f"Re-export the dataset from a matching BeamSimII version."
        )
    if file_version != SCHEMA_VERSION:
        warnings.warn(
            f"{path.name}: schema_version {file_version!r} differs from reader "
            f"{SCHEMA_VERSION!r} (same major). Reading anyway; verify fields.",
            UserWarning,
            stacklevel=3,
        )


def _write_attrs(h5obj: h5py.HLObject, attrs: dict, context: str = "") -> None:
    """Write a metadata dict onto an HDF5 object's attribute space.

    Encoding rules (DATA_CONTRACT.md §3.5):

    * ``None`` → **skipped** (absent attr = unset; the canonical case is
      ``box_volume_m3=None`` for a free-air / infinite-baffle driver).
    * ``dict`` / ``list`` / ``tuple`` → JSON-encoded string (tuple coerced to
      list first); round-trips via :func:`_read_attrs`.
    * ``np.ndarray`` and plain scalars / strings → passed directly to h5py.
    * Any other value that h5py cannot serialise raises :class:`TypeError` with
      a message naming the attr key, caller context, value type, and the
      original h5py error.

    Parameters
    ----------
    h5obj : h5py group or dataset
    attrs : dict
    context : str
        Human-readable label for the owning object, included in error messages
        (e.g. ``" for driver 'drv_1'"``, ``" (root attrs)"``).
    """
    for key, val in attrs.items():
        # None means "not set" — skip rather than crash (h5py cannot store None).
        if val is None:
            continue
        if isinstance(val, (dict, list, tuple)):
            # Coerce tuple → list so JSON round-trip is lossless.
            h5obj.attrs[key] = json.dumps(list(val) if isinstance(val, tuple) else val)
        elif isinstance(val, np.ndarray):
            try:
                h5obj.attrs[key] = val
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"Cannot write attr {key!r}{context}: "
                    f"ndarray with dtype {val.dtype!r} is not HDF5-serialisable "
                    f"(common cause: object-dtype array). "
                    f"Original h5py error: {exc}"
                ) from exc
        else:
            try:
                h5obj.attrs[key] = val
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"Cannot write attr {key!r}{context}: "
                    f"value of type {type(val).__name__!r} is not HDF5-serialisable. "
                    f"Original h5py error: {exc}"
                ) from exc


def _read_attrs(h5obj: h5py.HLObject) -> dict:
    """Read HDF5 attributes back to a Python dict, un-JSON-encoding where needed.

    Parameters
    ----------
    h5obj : h5py group or dataset

    Returns
    -------
    dict  — same structure as what was passed to ``_write_attrs``
    """
    result: dict = {}
    for key, val in h5obj.attrs.items():
        if isinstance(val, str):
            try:
                decoded = json.loads(val)
                if isinstance(decoded, (dict, list)):
                    val = decoded
            except (json.JSONDecodeError, ValueError):
                pass
        result[key] = val
    return result


def write_dataset(path: str | Path, ds: RadiationDataset) -> None:
    """Write a RadiationDataset to the on-disk HDF5 contract format.

    The write is **atomic**: data goes to a temp file in the same directory and
    is renamed over the target only on complete success.  A failed write never
    truncates or corrupts the pre-existing file.

    Parameters
    ----------
    path : str or Path
        Output file path.  Extension should be ``.h5`` or ``.bsim``; both are
        HDF5 internally.
    ds : RadiationDataset
        Assembled dataset from ``build_dataset()``.

    Notes
    -----
    - complex128 stored as compound (real, imag) float64 for exact roundtrip.
    - All §3.5 root attrs are written; dict/list/tuple attrs are JSON-encoded.
    - ``None``-valued driver attrs are skipped (not written); the canonical case
      is ``box_volume_m3=None`` for a free-air driver.
    - ``beamsim_version`` read from installed package metadata; falls back to
      ``"unknown"`` in editable installs before first build.
    - Temp file is created via ``tempfile.mkstemp`` (mode 0600); on success the
      final file inherits owner-only permissions.  Acceptable for a single-user
      desktop application.
    """
    try:
        bsim_version = importlib.metadata.version("beamsim2")
    except importlib.metadata.PackageNotFoundError:
        bsim_version = "unknown"

    # Validate BEFORE touching disk: a duplicate driver_id would otherwise crash
    # mid-write, leaving a corrupt partial file.
    validate_unique_driver_ids([d.driver_id for d in ds.drivers])

    path = Path(path)

    # Write to a temp file in the same directory so os.replace is a same-
    # filesystem atomic rename (never touches the target until fully written).
    fd, tmp_path_str = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)  # h5py opens its own fd; close the mkstemp one immediately
    tmp_path = Path(tmp_path_str)

    try:
        with h5py.File(tmp_path, "w") as f:
            # ── root datasets ──────────────────────────────────────────────
            f.create_dataset("frequencies", data=ds.frequencies)  # [F] float64

            # ── root attrs (§3.5) ─────────────────────────────────────────
            root_meta = {
                "schema_version": SCHEMA_VERSION,
                "beamsim_version": bsim_version,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "spacing": ds.spacing,
            }
            if ds.fractional_octave is not None:
                root_meta["fractional_octave"] = ds.fractional_octave
            root_meta.update(ds.attrs)
            _write_attrs(f, root_meta, context=" (root attrs)")

            # ── /directions group ─────────────────────────────────────────
            dg = f.create_group("directions")
            obs = ds.directions
            dg.create_dataset("unit_vectors", data=obs.unit_vectors)  # [N×3] float64
            dg.create_dataset("weights", data=obs.weights)  # [N] float64
            if obs.theta_phi is not None:
                dg.create_dataset("theta_phi", data=obs.theta_phi)  # [N×2] float64
            _write_attrs(
                dg,
                {
                    "scheme": obs.scheme,
                    "order": obs.order,
                    "weight_convention": obs.weight_convention,
                    "radius": obs.radius,
                },
                context=" (directions attrs)",
            )

            # ── /drivers group ────────────────────────────────────────────
            # Persist driver order explicitly: HDF5 group key iteration is
            # alphabetical, not insertion-order, so stacked_h_full row indices
            # (= driver indices) would be silently permuted on read without this.
            driver_order = [d.driver_id for d in ds.drivers]
            f.attrs["driver_order"] = json.dumps(driver_order)

            drv_grp = f.create_group("drivers")
            for d in ds.drivers:
                dg2 = drv_grp.create_group(d.driver_id)
                # h5py stores/reads complex128 natively — exact lossless roundtrip
                dg2.create_dataset("H_bem", data=d.H_bem.astype(np.complex128))  # [F×N]
                dg2.create_dataset(
                    "terminal_response", data=d.terminal_response.astype(np.complex128)
                )  # [F]
                dg2.create_dataset("H_full", data=d.H_full.astype(np.complex128))  # [F×N]
                dg2.create_dataset("convergence_flags", data=d.convergence_flags)  # [F] bool
                _write_attrs(dg2, d.attrs, context=f" for driver {d.driver_id!r}")

    except Exception:
        # Write failed — clean up the temp file and leave the original untouched.
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    # Full write succeeded — atomically replace the target.
    os.replace(tmp_path, path)


def read_dataset(path: str | Path) -> RadiationDataset:
    """Read a RadiationDataset from the on-disk HDF5 contract format.

    Parameters
    ----------
    path : str or Path
        Path to the ``.h5`` / ``.bsim`` file written by ``write_dataset()``.

    Returns
    -------
    RadiationDataset
        Fully reconstructed dataset with complex128 arrays and decoded attrs.

    Raises
    ------
    KeyError
        If a mandatory dataset or group is missing (file does not conform to
        schema_version "1.0").
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        frequencies = f["frequencies"][:]  # [F] float64
        root_attrs = _read_attrs(f)
        _check_schema_version(str(root_attrs.get("schema_version", "")), path)

        # ── /directions ───────────────────────────────────────────────────
        dg = f["directions"]
        unit_vectors = dg["unit_vectors"][:]  # [N×3] float64
        weights = dg["weights"][:]  # [N] float64
        theta_phi = dg["theta_phi"][:] if "theta_phi" in dg else None  # [N×2] float64
        dir_attrs = _read_attrs(dg)
        directions = ObservationPoints(
            unit_vectors=unit_vectors,
            radius=float(dir_attrs.get("radius", 1.0)),
            weights=weights,
            scheme=str(dir_attrs.get("scheme", "lebedev")),
            order=int(dir_attrs.get("order", 0)),
            weight_convention=str(dir_attrs.get("weight_convention", "sum_4pi")),
            theta_phi=theta_phi,
        )

        # ── /drivers ──────────────────────────────────────────────────────
        # Restore insertion order from the persisted driver_order attr.
        # HDF5 group key iteration is alphabetical, not insertion-order;
        # without this, stacked_h_full row indices would be silently permuted.
        drv_grp = f["drivers"]
        group_ids = list(drv_grp.keys())
        raw_order = root_attrs.pop("driver_order", None)
        if raw_order is None:
            # Genuinely old file written before the driver_order attr existed:
            # group keys are unique, so alphabetical order is a safe fallback.
            ordered_ids = sorted(group_ids)
        elif not isinstance(raw_order, list):
            # The attr is PRESENT but unreadable (not a JSON list) — this is not a
            # legacy file, so silently re-sorting would risk mis-mapping drivers.
            raise ValueError(
                f"{path.name}: corrupt dataset — driver_order attribute is present "
                f"but is not a list ({type(raw_order).__name__}). Re-run the solve "
                "to regenerate it."
            )
        else:
            # Guard against corrupt files (e.g. a duplicate driver_id crashed an
            # earlier write mid-stream, leaving driver_order longer than the
            # surviving group set). Refuse to silently duplicate/drop drivers.
            if len(set(raw_order)) != len(raw_order):
                dups = sorted({x for x in raw_order if raw_order.count(x) > 1})
                raise ValueError(
                    f"{path.name}: corrupt dataset — driver_order has duplicate "
                    f"id(s) {dups}. This file was written from a driver set with "
                    "colliding ids; re-run the solve to regenerate it."
                )
            if set(raw_order) != set(group_ids):
                raise ValueError(
                    f"{path.name}: corrupt dataset — driver_order lists "
                    f"{len(raw_order)} driver(s) {sorted(raw_order)} but the file "
                    f"contains {len(group_ids)} group(s) {sorted(group_ids)}. "
                    "Re-run the solve to regenerate it."
                )
            ordered_ids = raw_order

        drivers: list[DriverData] = []
        for driver_id in ordered_ids:
            dg2 = drv_grp[driver_id]
            H_bem = dg2["H_bem"][:].astype(np.complex128)  # [F×N] complex128
            terminal_response = dg2["terminal_response"][:].astype(np.complex128)  # [F] complex128
            H_full = dg2["H_full"][:].astype(np.complex128)  # [F×N] complex128
            convergence_flags = dg2["convergence_flags"][:].astype(bool)  # [F] bool
            driver_attrs = _read_attrs(dg2)
            drivers.append(
                DriverData(
                    driver_id=str(driver_id),
                    H_bem=H_bem,
                    terminal_response=terminal_response,
                    H_full=H_full,
                    convergence_flags=convergence_flags,
                    attrs=driver_attrs,
                )
            )

    # spacing attrs may be stored in root_attrs
    spacing = str(root_attrs.get("spacing", "log"))
    fo = root_attrs.get("fractional_octave", None)
    if fo is not None:
        fo = float(fo)

    # interpolated_mask not stored on disk yet (all False); reconstruct
    interpolated_mask = np.zeros(len(frequencies), dtype=bool)

    return RadiationDataset(
        frequencies=frequencies,
        spacing=spacing,
        fractional_octave=fo,
        interpolated_mask=interpolated_mask,
        directions=directions,
        drivers=drivers,
        attrs=root_attrs,
    )
