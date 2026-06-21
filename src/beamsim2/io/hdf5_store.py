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
import warnings
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from beamsim2.assembly.tensor import DriverData, RadiationDataset
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


def _write_attrs(h5obj: h5py.HLObject, attrs: dict) -> None:
    """Write a metadata dict onto an HDF5 object's attribute space.

    Dict-valued or list-valued entries are JSON-encoded so h5py does not have
    to serialise arbitrary Python objects.  Scalars and strings pass through.

    Parameters
    ----------
    h5obj : h5py group or dataset
    attrs : dict
    """
    for key, val in attrs.items():
        if isinstance(val, (dict, list)):
            h5obj.attrs[key] = json.dumps(val)
        elif isinstance(val, np.ndarray):
            h5obj.attrs[key] = val
        else:
            h5obj.attrs[key] = val


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
    - All §3.5 root attrs are written; dict/list attrs are JSON-encoded strings.
    - ``beamsim_version`` read from the installed package metadata; falls back
      to ``"unknown"`` in editable installs before first build.
    """
    try:
        bsim_version = importlib.metadata.version("beamsim2")
    except importlib.metadata.PackageNotFoundError:
        bsim_version = "unknown"

    path = Path(path)
    with h5py.File(path, "w") as f:
        # ── root datasets ──────────────────────────────────────────────────
        f.create_dataset("frequencies", data=ds.frequencies)  # [F] float64

        # ── root attrs (§3.5) ─────────────────────────────────────────────
        root_meta = {
            "schema_version": SCHEMA_VERSION,
            "beamsim_version": bsim_version,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "spacing": ds.spacing,
        }
        if ds.fractional_octave is not None:
            root_meta["fractional_octave"] = ds.fractional_octave
        root_meta.update(ds.attrs)
        _write_attrs(f, root_meta)

        # ── /directions group ─────────────────────────────────────────────
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
        )

        # ── /drivers group ────────────────────────────────────────────────
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
            _write_attrs(dg2, d.attrs)


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
        raw_order = root_attrs.pop("driver_order", None)
        if raw_order is not None and isinstance(raw_order, list):
            ordered_ids = raw_order
        else:
            # fallback for files written before driver_order attr existed
            ordered_ids = sorted(drv_grp.keys())

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
